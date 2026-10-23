from __future__ import annotations

from html import escape
import os
from pathlib import Path
import re
import textwrap
from typing import Any
import zipfile
from xml.etree import ElementTree

from ..context import ToolRuntimeContext
from ..contracts import PermissionEffect, ToolManifest, ToolStatus


class DocumentTool:
    def __init__(self, manifest: ToolManifest) -> None:
        self.manifest = manifest

    def health(self) -> tuple[ToolStatus, str]:
        return ToolStatus.AVAILABLE, "stdlib_docx_pdf_backend_ready"

    def effect_for(self, capability: str, arguments: dict[str, Any]) -> PermissionEffect:
        if capability == "document.read_document":
            return PermissionEffect.READ
        if capability == "document.edit_document":
            return PermissionEffect.EDIT
        if bool(arguments.get("overwrite")):
            return PermissionEffect.OVERWRITE
        return PermissionEffect.CREATE

    def execute(self, capability: str, arguments: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        if capability == "document.read_document":
            return self._read(arguments, context)
        if capability == "document.edit_document":
            return self._edit(arguments, context)
        formats = {
            "document.create_txt": ".txt",
            "document.create_markdown": ".md",
            "document.create_docx": ".docx",
            "document.create_pdf": ".pdf",
        }
        expected = formats[capability]
        path = context.authorize_path(_required(arguments, "path"), operation="WRITE")
        if path.suffix.casefold() != expected:
            raise ValueError(f"document_extension_must_be:{expected}")
        existed = path.exists()
        if existed and not bool(arguments.get("overwrite", False)):
            raise FileExistsError(path)
        content = str(arguments.get("content", ""))
        title = str(arguments.get("title", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        context.revalidate(path, "WRITE")
        with path.open("wb" if bool(arguments.get("overwrite")) else "xb") as handle:
            if expected in {".txt", ".md"}:
                handle.write(content.encode("utf-8"))
            elif expected == ".docx":
                _write_docx(handle, content, title)
            else:
                _write_pdf(handle, content, title)
        effect = PermissionEffect.OVERWRITE if existed else PermissionEffect.CREATE
        return _effect({"path": str(path), "format": expected[1:], "bytes": path.stat().st_size}, effect, path)

    def _read(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="READ")
        context.revalidate(path, "READ")
        suffix = path.suffix.casefold()
        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            content = _read_docx(path)
        elif suffix == ".pdf":
            content = _read_own_pdf(path)
        else:
            raise ValueError("unsupported_document_format")
        if len(content) > 2_000_000:
            raise ValueError("document_content_too_large")
        return {"path": str(path), "format": suffix[1:], "content": content}

    def _edit(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = context.authorize_path(_required(args, "path"), operation="WRITE")
        suffix = path.suffix.casefold()
        edit_spec = args.get("document_edit_spec")
        if suffix == ".docx" and isinstance(edit_spec, dict):
            context.revalidate(path, "WRITE")
            _edit_docx_structure(path, edit_spec, context)
            return _effect(
                {"path": str(path), "edited": True, "structural": True},
                PermissionEffect.EDIT,
                path,
            )
        old = str(_required(args, "old_text"))
        new = str(args.get("new_text", ""))
        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            content = _read_docx(path)
        else:
            raise ValueError("editing_supported_for_txt_md_docx")
        if old not in content:
            raise ValueError("edit_anchor_not_found")
        updated = content.replace(old, new, 1)
        context.revalidate(path, "WRITE")
        if suffix in {".txt", ".md"}:
            path.write_text(updated, encoding="utf-8")
        else:
            _write_docx(path, updated, str(args.get("title", "")))
        return _effect({"path": str(path), "edited": True}, PermissionEffect.EDIT, path)


def _write_docx(path: Path, content: str, title: str) -> None:
    paragraphs = ([title] if title else []) + content.splitlines()
    body = "".join(
        "<w:p><w:r><w:t xml:space=\"preserve\">"
        + escape(line)
        + "</w:t></w:r></w:p>"
        for line in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > 4_000_000:
            raise ValueError("docx_document_xml_too_large")
        data = archive.read("word/document.xml")
    root = ElementTree.fromstring(data)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        paragraphs.append("".join(node.text or "" for node in paragraph.iter(namespace + "t")))
    return "\n".join(paragraphs)


def _edit_docx_structure(path: Path, specification: dict[str, Any], context=None) -> None:
    operation = str(specification.get("operation", ""))
    payload = str(specification.get("payload", ""))
    anchor = str(specification.get("anchor", ""))
    if not payload:
        raise ValueError("document_edit_payload_missing")
    with zipfile.ZipFile(path) as source:
        infos = source.infolist()
        entries = {info.filename: source.read(info.filename) for info in infos}
    document_name = "word/document.xml"
    if document_name not in entries:
        raise ValueError("docx_document_xml_missing")
    root = ElementTree.fromstring(entries[document_name])
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ElementTree.register_namespace("w", namespace)
    body = root.find(f"{{{namespace}}}body")
    if body is None:
        raise ValueError("docx_body_missing")
    paragraph = _docx_paragraph(payload, namespace)
    children = list(body)
    if operation == "APPEND_PARAGRAPH":
        section_index = next(
            (index for index, child in enumerate(children) if child.tag == f"{{{namespace}}}sectPr"),
            len(children),
        )
        body.insert(section_index, paragraph)
    elif operation == "PREPEND_PARAGRAPH":
        body.insert(0, paragraph)
    elif operation in {"INSERT_AFTER", "INSERT_BEFORE"}:
        matches = [
            index for index, child in enumerate(children)
            if child.tag == f"{{{namespace}}}p" and anchor in _paragraph_text(child, namespace)
        ]
        if len(matches) != 1:
            raise ValueError("document_edit_anchor_not_unique")
        body.insert(matches[0] + (1 if operation == "INSERT_AFTER" else 0), paragraph)
    elif operation == "REPLACE_TEXT":
        text_nodes = [
            node for node in root.iter(f"{{{namespace}}}t")
            if anchor and anchor in (node.text or "")
        ]
        if len(text_nodes) != 1:
            raise ValueError("document_edit_anchor_not_unique_run")
        text_nodes[0].text = (text_nodes[0].text or "").replace(anchor, payload, 1)
    else:
        raise ValueError("document_edit_operation_unsupported")
    entries[document_name] = ElementTree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    )
    from uuid import uuid4
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.structural-edit.tmp")
    try:
        if context is not None:
            context.revalidate(temporary, "WRITE")
        with zipfile.ZipFile(temporary, "x") as target:
            for info in infos:
                target.writestr(info, entries[info.filename])
        if context is not None:
            context.revalidate(temporary, "WRITE")
            context.revalidate(path, "WRITE")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _docx_paragraph(text: str, namespace: str):
    paragraph = ElementTree.Element(f"{{{namespace}}}p")
    run = ElementTree.SubElement(paragraph, f"{{{namespace}}}r")
    node = ElementTree.SubElement(run, f"{{{namespace}}}t")
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    return paragraph


def _paragraph_text(paragraph, namespace: str) -> str:
    return "".join(
        node.text or "" for node in paragraph.iter(f"{{{namespace}}}t")
    )


def _write_pdf(path: Path, content: str, title: str) -> None:
    lines: list[str] = []
    if title:
        lines.extend(textwrap.wrap(title, width=75) or [""])
        lines.append("")
    for raw in content.splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, width=92, replace_whitespace=False) or [""])
    pages = [lines[index:index + 48] for index in range(0, max(len(lines), 1), 48)]
    objects: list[bytes] = []
    page_ids = [4 + index * 2 for index in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{item} 0 R" for item in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for page_id, page_lines in zip(page_ids, pages):
        content_id = page_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
        )
        commands = ["BT", "/F1 11 Tf", "14 TL", "54 738 Td"]
        for index, line in enumerate(page_lines):
            if index:
                commands.append("T*")
            commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("cp1252", errors="replace")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream")
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, item in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(item)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    if isinstance(path, Path):
        path.write_bytes(payload)
    else:
        path.write(payload)


def _read_own_pdf(path: Path) -> str:
    if path.stat().st_size > 4_000_000:
        raise ValueError("pdf_read_limit_exceeded")
    data = path.read_bytes().decode("cp1252", errors="replace")
    values = re.findall(r"\((.*?)(?<!\\)\)\s*Tj", data)
    return "\n".join(_pdf_unescape(item) for item in values)


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_unescape(value: str) -> str:
    return value.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")


def _required(args: dict[str, Any], key: str) -> Any:
    if key not in args or args[key] in {None, ""}:
        raise ValueError(f"missing_argument:{key}")
    return args[key]


def _effect(result: dict[str, Any], effect: PermissionEffect, path: Path) -> dict[str, Any]:
    result["_side_effects"] = [{"effect": effect.value, "path": str(path)}]
    return result
