"""Typed checks; independent reference expressions are data, never executable code."""
import ast
from dataclasses import replace
from difflib import SequenceMatcher
import hashlib
import json
import math
import operator
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree
import zipfile

from .verification_types import VerificationResult, VerificationStatus as S
from .failure_state import safe_detail, text_summary


def summary(value):
    text = str(value)
    return {**text_summary(text), "snippet": safe_detail(text, 160)}


def result(kind, artifact_type, checks, *, artifact_id="", evidence=None, inconclusive="", code="", message=""):
    failed = next((k for k, v in checks.items() if v is False), "")
    status = S.FAIL if failed else S.INCONCLUSIVE if inconclusive else S.PASS
    if not message and failed:
        message = f"No se cumplió el chequeo {safe_detail(failed, 120)} del verificador {kind}."
    return VerificationResult(kind, artifact_type, status, checks, evidence or {}, artifact_id,
        failed or inconclusive, code or ("CHECK_FAILED" if failed else "EVIDENCE_UNAVAILABLE" if inconclusive else ""),
        message, safe_detail(failed or inconclusive))


class ReferenceOracle:
    """Bounded deterministic primitives. Trust is supplied by the host, not the payload."""
    @staticmethod
    def evaluate(reference, stdin):
        kind = reference.get("kind")
        if kind == "COUNT_CHARACTERS":
            value = stdin.rstrip("\r\n")
            if reference.get("casefold", True):
                value = value.casefold()
            if reference.get("strip_accents", False):
                value = "".join(c for c in unicodedata.normalize("NFD", value) if not unicodedata.combining(c))
            alphabet = str(reference["alphabet"])
            if len(value) > 10000 or len(alphabet) > 256:
                raise ValueError("oracle_limits")
            return sum(c in alphabet for c in value)
        if kind != "EXPRESSION":
            raise ValueError("oracle_reference_unsupported")
        expression = str(reference["expression"])
        if len(expression) > 256:
            raise ValueError("oracle_expression_limit")
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 64:
            raise ValueError("oracle_expression_limit")
        tokens = stdin.split()
        variables = {name: float(tokens[int(index)]) for name, index in reference.get("variables", {}).items()}
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Mod: operator.mod}
        def calculate(node):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                value = node.value
            elif isinstance(node, ast.Name) and node.id in variables:
                value = variables[node.id]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                value = ops[type(node.op)](calculate(node.left), calculate(node.right))
            else:
                raise ValueError("oracle_expression_forbidden")
            if not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError("oracle_number_limit")
            return value
        return calculate(tree.body)


class CodeArtifactVerifier:
    @staticmethod
    def verify(*, test=None, stdout="", exit_code=None, artifact_id="", trusted_oracle=False,
               oracle_selector=None, execution_only=False, execution_succeeded=False, **unused):
        if execution_only:
            return result("CODE","CODE",{"execution_succeeded":execution_succeeded,"exit_code":exit_code==0},
                artifact_id=artifact_id,evidence={"coverage":"EXECUTION_ONLY","exit_code":exit_code,
                    "oracle_source":"NOT_APPLICABLE"},code="" if execution_succeeded and exit_code==0 else "CODE_EXECUTION_FAILED")
        if test is None:
            return result("CODE","CODE",{},artifact_id=artifact_id,inconclusive="test_spec",
                code="TEST_SPEC_MISSING")
        oracle_metadata = {}
        if oracle_selector is not None:
            selection, resolved = oracle_selector.resolve(test)
            oracle_metadata = {"oracle_selection": selection.reason}
            if selection.definition:
                oracle_metadata.update(oracle_id=selection.definition["oracle_id"],
                    oracle_version=selection.definition["version"])
            if resolved is not None:
                hint = test.expected_hint if test.expected_hint is not None else test.expected
                oracle_metadata.update(generated_expected_hint=summary(hint), tolerance=resolved.tolerance,
                    oracle_source="DETERMINISTIC_REFERENCE", oracle_confidence="DETERMINISTIC")
                test, trusted_oracle = resolved, True
            elif test.operation_semantics or test.oracle_requirement:
                return result("CODE", "CODE", {}, artifact_id=artifact_id,
                    evidence={**oracle_metadata,"actual":summary(stdout),"input":text_summary(test.stdin),
                        "expected_hint":summary(test.expected_hint if test.expected_hint is not None else test.expected),
                        "oracle_confidence":"UNVERIFIED","comparison_mode":test.comparison_mode,"exit_code":exit_code},
                    inconclusive=test.test_id, code=selection.reason,
                    message="No hay una referencia inequívoca para la semántica o el comportamiento de esa entrada.")
        expected = test.expected
        source = test.oracle_source if trusted_oracle else "MODEL_EXPECTATION"
        evidence = {"input": text_summary(test.stdin), "actual": summary(stdout),
            "expected": summary(expected), "oracle_source": source, "comparison_mode": test.comparison_mode,
            "exit_code": exit_code, "oracle_confidence": "UNVERIFIED", "tolerance":test.tolerance}
        evidence.update(oracle_metadata)
        if not trusted_oracle:
            return result("CODE", "CODE", {}, artifact_id=artifact_id, evidence=evidence,
                inconclusive=test.test_id, code="ORACLE_UNVERIFIED",
                message="El programa se ejecutó, pero no hay un oráculo independiente validado para esa prueba.")
        evidence["oracle_confidence"] = oracle_metadata.get("oracle_confidence", "LOCAL_DECLARED")
        if test.reference:
            try:
                expected = ReferenceOracle.evaluate(test.reference, test.stdin)
            except (ValueError, KeyError, IndexError, SyntaxError, ZeroDivisionError, TypeError):
                return result("CODE", "CODE", {}, artifact_id=artifact_id, evidence=evidence,
                    inconclusive=test.test_id, code="ORACLE_REFERENCE_INVALID")
            try:
                disagrees = not math.isfinite(float(test.expected)) or float(test.expected) != float(expected)
            except (ValueError, TypeError):
                disagrees = test.expected != expected
            evidence.update(oracle_source="INDEPENDENT_LOCAL_REFERENCE", oracle_confidence="DETERMINISTIC",
                reference_expected=summary(expected), generated_expectation_disagrees=disagrees)
        if expected is None:
            return result("CODE", "CODE", {}, artifact_id=artifact_id, evidence=evidence,
                inconclusive=test.test_id, code="ORACLE_MISSING")
        mode, actual = test.comparison_mode, stdout.strip()
        try:
            if mode == "EXACT":
                matched = actual == str(expected).strip()
            elif mode in {"NUMERIC", "FLOAT_TOLERANCE"}:
                a, b = float(actual), float(expected)
                matched = math.isfinite(a) and math.isfinite(b) and abs(a-b) <= (test.tolerance if mode == "FLOAT_TOLERANCE" else 0)
            elif mode == "STRUCTURED":
                parsed = json.loads(actual)
                matched = json.dumps(parsed, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
            elif mode == "CONTAINS":
                matched = isinstance(expected, str) and expected in stdout
            elif mode == "PROPERTY":
                # A property proves only itself; it is not equivalence to the full goal.
                properties = {"nonempty": lambda: bool(actual), "finite_number": lambda: math.isfinite(float(actual))}
                if not isinstance(expected, str) or expected not in properties:
                    return result("CODE", "CODE", {}, artifact_id=artifact_id, evidence=evidence,
                        inconclusive=test.test_id, code="PROPERTY_UNSUPPORTED")
                matched = properties[expected]()
                evidence["coverage"] = "DECLARED_PROPERTY_ONLY"
            else:
                raise ValueError("unsupported_comparison")
        except (ValueError, TypeError, OverflowError):
            matched = False
        verified = result("CODE", "CODE", {"exit_code": exit_code == test.expected_exit_code, "output": matched},
            artifact_id=artifact_id, evidence=evidence, code="" if matched and exit_code == test.expected_exit_code else "TEST_ASSERTION_FAILED",
            message=f"El programa no superó la prueba {safe_detail(test.test_id, 120)}." if not matched else "")
        return replace(verified, failed_check=test.test_id if verified.status != S.PASS else "")


def collapse(value):
    return re.sub(r"\s+", " ", value).strip()


class DocumentArtifactVerifier:
    @staticmethod
    def verify(*, path, format_id, expected_text=None, artifact_id="", body_text=None,
               title="", target_words=None, tolerance=0.2, required_paragraphs=0, **unused):
        path = Path(path)
        checks = {"exists": path.is_file()}
        if not checks["exists"]:
            return result("DOCUMENT", "DOCUMENT", checks, artifact_id=artifact_id)
        try:
            if format_id == "docx":
                with zipfile.ZipFile(path) as archive:
                    if sum(info.file_size for info in archive.infolist()) > 64000000 or len(archive.infolist()) > 1024:
                        return result("DOCUMENT", "DOCUMENT", checks, artifact_id=artifact_id,
                            inconclusive="bounded_reader", code="DOCUMENT_VERIFICATION_LIMIT")
                    checks["container_crc"] = archive.testzip() is None
                    info = archive.getinfo("word/document.xml")
                    if info.file_size > 4000000:
                        raise ValueError("document_limit")
                    root = ElementTree.fromstring(archive.read(info.filename))
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                paragraphs = ["".join(t.text or "" for t in p.iter(ns+"t")) for p in root.iter(ns+"p")]
                actual = "\n".join(paragraphs)
            elif format_id in {"txt", "md", "markdown"}:
                if path.stat().st_size > 4000000:
                    raise ValueError("document_limit")
                actual = path.read_text(encoding="utf-8")
                paragraphs = actual.splitlines()
            else:
                return result("DOCUMENT", "DOCUMENT", checks, artifact_id=artifact_id,
                    inconclusive="reader", code="FORMAT_UNSUPPORTED")
            checks.update(readable=True, nonempty=bool(actual.strip()), structure=len(paragraphs) >= required_paragraphs)
            if expected_text is not None:
                checks["content_preserved"] = collapse(actual) == collapse(expected_text)
            if target_words is not None:
                # Count only a body demonstrated by the extracted document, never just expected input.
                observed_body = actual
                if title and paragraphs and paragraphs[0] == title:
                    observed_body = "\n".join(paragraphs[1:])
                if body_text is not None:
                    checks["body_preserved"] = collapse(observed_body) == collapse(body_text)
                count = len(re.findall(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", observed_body, re.UNICODE))
                checks["body_word_count"] = int(target_words*(1-tolerance)) <= count <= int(target_words*(1+tolerance))
            else:
                count = None
            return result("TEXT" if format_id in {"txt","md","markdown"} else "DOCUMENT", "DOCUMENT", checks,
                artifact_id=artifact_id, evidence={"body_word_count": count, "text": text_summary(actual)})
        except (OSError, ValueError, KeyError, ElementTree.ParseError, zipfile.BadZipFile):
            return result("DOCUMENT", "DOCUMENT", {**checks, "readable": False}, artifact_id=artifact_id,
                code="DOCUMENT_INVALID")


class PdfArtifactVerifier:
    @staticmethod
    def verify(*, path, artifact_id="", expected_text=None, **unused):
        from ..official.document_tool import _read_own_pdf
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 4000000:
            return result("PDF", "DOCUMENT", {"bounded_file": False}, artifact_id=artifact_id)
        data = path.read_bytes()
        checks = {"header": data.startswith(b"%PDF-"), "eof": data.rstrip().endswith(b"%%EOF")}
        # Supported reader is the existing local renderer's uncompressed PDF dialect.
        if b"/Filter" in data or b"/Encrypt" in data:
            return result("PDF", "DOCUMENT", checks, artifact_id=artifact_id,
                inconclusive="extractor", code="PDF_READER_UNAVAILABLE")
        pages = len(re.findall(rb"/Type\s*/Page\b", data))
        start = re.search(rb"startxref\s+(\d+)\s+%%EOF", data)
        checks.update(pages=pages > 0, xref=bool(start and data[int(start[1]):].startswith(b"xref")))
        valid_offsets = False
        if checks["xref"]:
            lines = data[int(start[1]):].splitlines()
            try:
                first, count = map(int, lines[1].split())
                valid_offsets = first == 0 and 1 <= count <= 10000
                for index in range(1, count):
                    offset = int(lines[2+index].split()[0])
                    valid_offsets = valid_offsets and data[offset:].startswith(f"{index} 0 obj".encode())
            except (ValueError, IndexError):
                valid_offsets = False
        checks["object_offsets"] = valid_offsets
        actual = _read_own_pdf(path)
        checks["extractable_nonempty"] = bool(actual.strip())
        if expected_text is not None:
            checks["content_preserved"] = SequenceMatcher(None, collapse(expected_text), collapse(actual), autojunk=False).ratio() >= .90
        return result("PDF", "DOCUMENT", checks, artifact_id=artifact_id,
            evidence={"page_count": pages, "text": text_summary(actual), "reader": "LOCAL_UNCOMPRESSED_PDF"})


class ConversionVerifier:
    @staticmethod
    def verify(*, source_ref, target_ref, expected_source_id, **kwargs):
        from ..official.document_tool import _read_docx, _read_own_pdf
        readers = {"docx": _read_docx, "pdf": _read_own_pdf,
            "txt": lambda p: p.read_text(encoding="utf-8"), "md": lambda p: p.read_text(encoding="utf-8")}
        evidence = {"source_artifact_id": source_ref.artifact_id, "target_artifact_id": target_ref.artifact_id,
            "source_type": source_ref.artifact_type, "target_type": target_ref.artifact_type,
            "source_version": source_ref.version, "source_hash": source_ref.hash}
        checks = {"source_relation": source_ref.artifact_id == expected_source_id}
        if source_ref.format not in readers:
            return result("CONVERSION", target_ref.artifact_type, checks, evidence=evidence,
                inconclusive="source_reader", code="CONVERSION_READER_UNAVAILABLE")
        try:
            source = readers[source_ref.format](Path(source_ref.path))
            checks["source_nonempty"] = bool(source.strip())
            checks["source_hash"] = hashlib.sha256(Path(source_ref.path).read_bytes()).hexdigest() == source_ref.hash
            checks["target_hash"] = hashlib.sha256(Path(target_ref.path).read_bytes()).hexdigest() == target_ref.hash
            target = VerificationDispatcher.verify("PDF" if target_ref.format == "pdf" else "DOCUMENT",
                path=target_ref.path, format_id=target_ref.format, artifact_id=target_ref.artifact_id, expected_text=source)
            combined = result("CONVERSION", target_ref.artifact_type, {**checks, **target.checks},
                artifact_id=target_ref.artifact_id, evidence={**evidence, **target.evidence},
                inconclusive=target.failed_check if target.status == S.INCONCLUSIVE else "")
            return combined
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return result("CONVERSION", target_ref.artifact_type, {**checks, "source_readable": False}, evidence=evidence)


class ArchiveVerifier:
    @staticmethod
    def verify(*, path, expected_members, artifact_id="", **unused):
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                checks = {"opens": True, "members_exact": len(names) == len(set(names)) and set(names) == set(expected_members)}
                if len(names) > 256 or sum(i.file_size for i in archive.infolist()) > 64000000:
                    return result("ARCHIVE", "ARCHIVE", checks, artifact_id=artifact_id,
                        inconclusive="bounded_reader", code="ARCHIVE_VERIFICATION_LIMIT")
                checks["crc"] = archive.testzip() is None
                def member_hash(name):
                    h = hashlib.sha256()
                    with archive.open(name) as stream:
                        for block in iter(lambda: stream.read(1048576), b""):
                            h.update(block)
                    return h.hexdigest()
                checks["member_hashes"] = checks["members_exact"] and all(
                    member_hash(name) == digest for name,digest in expected_members.items())
                return result("ARCHIVE", "ARCHIVE", checks, artifact_id=artifact_id, evidence={"members": names})
        except (OSError, ValueError, zipfile.BadZipFile):
            return result("ARCHIVE", "ARCHIVE", {"opens": False}, artifact_id=artifact_id)


class FilesystemVerifier:
    @staticmethod
    def verify(*, operation, target, source=None, before_hash=None, artifact_id="", require_change=True, **unused):
        target = Path(target)
        checks = {"target_absent": not target.exists()} if operation == "DELETE" else {"target_exists": target.exists()}
        if operation in {"MOVE", "RENAME"}:
            checks["source_absent"] = source is not None and not Path(source).exists()
        elif operation == "EDIT":
            if before_hash is None:
                return result("FILESYSTEM_CHANGE", "FILESYSTEM", checks, artifact_id=artifact_id,
                    inconclusive="before_snapshot", code="BEFORE_SNAPSHOT_MISSING")
            checks["expected_hash_effect"] = target.is_file() and (
                not require_change or hashlib.sha256(target.read_bytes()).hexdigest() != before_hash)
        elif operation not in {"CREATE", "DELETE", "MOVE", "RENAME"}:
            return result("FILESYSTEM_CHANGE", "FILESYSTEM", {}, inconclusive="effect", code="EFFECT_UNSUPPORTED")
        return result("FILESYSTEM_CHANGE", "FILESYSTEM", checks, artifact_id=artifact_id)


class VerificationDispatcher:
    TYPES = {"CODE": CodeArtifactVerifier, "DOCUMENT": DocumentArtifactVerifier, "TEXT": DocumentArtifactVerifier,
        "PDF": PdfArtifactVerifier, "ARCHIVE": ArchiveVerifier, "FILESYSTEM_CHANGE": FilesystemVerifier,
        "CONVERSION": ConversionVerifier}

    @classmethod
    def verify(cls, verifier_type, **kwargs):
        if verifier_type not in cls.TYPES:
            return result(verifier_type, "UNKNOWN", {}, inconclusive="verifier", code="VERIFIER_UNAVAILABLE")
        return cls.TYPES[verifier_type].verify(**kwargs)
