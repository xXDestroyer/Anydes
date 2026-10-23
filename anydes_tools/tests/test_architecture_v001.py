from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from anydes_tools.audit import ToolAuditLog
from anydes_tools.bridge import AnydesActionBridge
from anydes_tools.composition import ToolComposer
from anydes_tools.contracts import (
    CapabilityIntent,
    CapabilityResolutionKind,
    CompositionPlan,
    CompositionStep,
    ExecutionStatus,
    PermissionEffect,
    ToolRequest,
)
from anydes_tools.executor import (
    ToolExecutor,
    _cleanup_worker_runtime,
    _prepare_worker_runtime,
    _worker_environment,
)
from anydes_tools.path_policy import FilesystemBoundary
from anydes_tools.permissions import PermissionBridge
from anydes_tools.registry import ToolRegistry, build_official_registry
from anydes_tools.resolver import CapabilityResolver


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"
WORK = TOOLS / "workspaces" / "automated_v001"
MANAGED_DEV = TOOLS / "workspaces" / "managed_dev_v001"


class ArchitectureV001Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(WORK, ignore_errors=True)
        shutil.rmtree(MANAGED_DEV, ignore_errors=True)
        WORK.mkdir(parents=True)
        cls.registry = build_official_registry(TOOLS)
        cls.boundary = FilesystemBoundary(ROOT, TOOLS)
        cls.audit_path = TOOLS / "logs" / "test-audit.jsonl"
        cls.audit_path.unlink(missing_ok=True)
        cls.executor = ToolExecutor(
            ROOT,
            TOOLS,
            cls.registry,
            PermissionBridge(),
            ToolAuditLog(cls.audit_path),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(WORK, ignore_errors=True)
        shutil.rmtree(MANAGED_DEV, ignore_errors=True)

    def request(
        self,
        capability: str,
        arguments: dict[str, object],
        *,
        confirmed: tuple[PermissionEffect, ...] = (),
        reflection: bool = True,
        scopes: tuple[str, ...] = (),
    ):
        tool = self.registry.find_by_capability(capability)[0]
        return self.executor.execute(
            ToolRequest(
                request_id=f"test:{capability}",
                turn_id="test-turn",
                tool_id=tool.manifest.tool_id,
                capability=capability,
                arguments=arguments,
                authorized_scopes=scopes,
                confirmed_effects=confirmed,
                reflection_approved=reflection,
                explicit_user_request=True,
            )
        )

    def test_01_registry_manifests_and_snapshot(self) -> None:
        self.assertEqual(len(self.registry.list_tools(enabled_only=True)), 4)
        self.assertGreaterEqual(len(self.registry.list_capabilities(enabled_only=True)), 27)
        snapshot = self.registry.capability_snapshot()
        self.assertEqual(snapshot.source, "ToolRegistry")
        self.assertIn("filesystem.read", snapshot.available_capabilities)

    def test_02_capability_resolution(self) -> None:
        resolver = CapabilityResolver(self.registry, self.boundary)
        exact = resolver.resolve(CapabilityIntent(("document.create_docx",)))
        self.assertEqual(exact.kind, CapabilityResolutionKind.EXACT_TOOL)
        composite = resolver.resolve(CapabilityIntent(operation="text_files_to_pdf"))
        self.assertEqual(composite.kind, CapabilityResolutionKind.COMPOSABLE)
        missing = resolver.resolve(CapabilityIntent(("media.video.trim",)))
        self.assertEqual(missing.kind, CapabilityResolutionKind.MISSING_CAPABILITY)
        forbidden = resolver.resolve(CapabilityIntent(("payment.send",)))
        self.assertEqual(forbidden.kind, CapabilityResolutionKind.FORBIDDEN)

    def test_03_filesystem_create_read_edit_copy_move(self) -> None:
        original = WORK / "hello.txt"
        created = self.request("filesystem.create_text", {"path": str(original), "content": "hola"})
        self.assertEqual(created.status, ExecutionStatus.SUCCESS)
        read = self.request("filesystem.read", {"path": str(original)})
        self.assertEqual(read.output["content"], "hola")
        edited = self.request("filesystem.edit_text", {"path": str(original), "old_text": "hola", "new_text": "hola Anydes"})
        self.assertEqual(edited.status, ExecutionStatus.SUCCESS)
        copied_path = WORK / "copy.txt"
        copied = self.request("filesystem.copy", {"source": str(original), "destination": str(copied_path)})
        self.assertEqual(copied.status, ExecutionStatus.SUCCESS)
        moved_path = WORK / "moved.txt"
        moved = self.request(
            "filesystem.move",
            {"source": str(copied_path), "destination": str(moved_path)},
            confirmed=(PermissionEffect.MOVE,),
        )
        self.assertEqual(moved.status, ExecutionStatus.SUCCESS)
        self.assertTrue(moved_path.is_file())
        folder = WORK / "nested"
        made = self.request("filesystem.mkdir", {"path": str(folder)})
        self.assertEqual(made.status, ExecutionStatus.SUCCESS)
        renamed = self.request(
            "filesystem.rename",
            {"path": str(moved_path), "new_name": "renamed.txt"},
            confirmed=(PermissionEffect.MOVE,),
        )
        self.assertEqual(renamed.status, ExecutionStatus.SUCCESS)
        search = self.request("filesystem.search", {"root": str(WORK), "pattern": "*.txt"})
        self.assertEqual(search.status, ExecutionStatus.SUCCESS)
        self.assertTrue(any(item.endswith("renamed.txt") for item in search.output["matches"]))

    def test_04_document_docx_and_pdf(self) -> None:
        docx = WORK / "sample.docx"
        pdf = WORK / "sample.pdf"
        for capability, path in (("document.create_docx", docx), ("document.create_pdf", pdf)):
            result = self.request(capability, {"path": str(path), "title": "Anydes", "content": "Linea uno\nLinea dos"})
            self.assertEqual(result.status, ExecutionStatus.SUCCESS)
            self.assertTrue(path.is_file())
            read = self.request("document.read_document", {"path": str(path)})
            self.assertEqual(read.status, ExecutionStatus.SUCCESS)
            self.assertIn("Linea uno", read.output["content"])
        edited = self.request(
            "document.edit_document",
            {"path": str(docx), "old_text": "Linea uno", "new_text": "Linea editada"},
        )
        self.assertEqual(edited.status, ExecutionStatus.SUCCESS)
        reread = self.request("document.read_document", {"path": str(docx)})
        self.assertIn("Linea editada", reread.output["content"])

    def test_05_archive_roundtrip_and_zip_slip(self) -> None:
        source = WORK / "archive-source.txt"
        source.write_text("contenido", encoding="utf-8")
        archive = WORK / "sample.zip"
        compressed = self.request("archive.compress_zip", {"sources": [str(source)], "destination": str(archive)})
        self.assertEqual(compressed.status, ExecutionStatus.SUCCESS)
        inspected = self.request("archive.inspect", {"path": str(archive)})
        self.assertTrue(inspected.output["safe"])
        extracted = WORK / "extracted"
        result = self.request("archive.extract_zip", {"path": str(archive), "destination": str(extracted)})
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        malicious = WORK / "unsafe.zip"
        with zipfile.ZipFile(malicious, "w") as handle:
            handle.writestr("../escape.txt", "no")
        denied = self.request("archive.extract_zip", {"path": str(malicious), "destination": str(WORK / "unsafe-out")})
        self.assertEqual(denied.status, ExecutionStatus.FAILED)
        self.assertFalse((WORK.parent / "escape.txt").exists())

    def test_06_python_success_error_and_timeout_isolation(self) -> None:
        good = WORK / "good.py"
        good.write_text("print(2 + 3)\n", encoding="utf-8")
        result = self.request("developer.run_python", {"path": str(good)}, confirmed=(PermissionEffect.EXECUTE_CODE,))
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.output["returncode"], 0)
        self.assertIn("5", result.output["stdout"])
        bad = WORK / "bad.py"
        bad.write_text("raise RuntimeError('controlled')\n", encoding="utf-8")
        failed_program = self.request("developer.run_python", {"path": str(bad)}, confirmed=(PermissionEffect.EXECUTE_CODE,))
        self.assertEqual(failed_program.status, ExecutionStatus.SUCCESS)
        self.assertNotEqual(failed_program.output["returncode"], 0)
        self.assertIn("RuntimeError", failed_program.output["stderr"])
        loop = WORK / "loop.py"
        loop.write_text("while True:\n    pass\n", encoding="utf-8")
        timed = self.request("developer.run_python", {"path": str(loop), "timeout": 0.2}, confirmed=(PermissionEffect.EXECUTE_CODE,))
        self.assertEqual(timed.status, ExecutionStatus.TIMEOUT)
        after = self.request("filesystem.inspect", {"path": str(good)})
        self.assertEqual(after.status, ExecutionStatus.SUCCESS)

    def test_07_cpp_if_available(self) -> None:
        source = WORK / "hello.cpp"
        source.write_text('#include <iostream>\nint main(){std::cout << "ok"; return 0;}\n', encoding="utf-8")
        result = self.request("developer.run_cpp", {"path": str(source)}, confirmed=(PermissionEffect.EXECUTE_CODE,))
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        if result.output.get("available") is False:
            self.skipTest("g++ is not available")
        self.assertEqual(result.output["returncode"], 0)
        self.assertEqual(result.output["stdout"], "ok")

    def test_08_core_and_traversal_denied(self) -> None:
        core_target = ROOT / "programa_principal" / "denied-v001.txt"
        denied = self.request("filesystem.create_text", {"path": str(core_target), "content": "forbidden"})
        self.assertEqual(denied.status, ExecutionStatus.DENIED)
        self.assertIn("DENIED_CORE_PROTECTED", denied.error)
        escaped = self.request("filesystem.create_text", {"path": "../../programa_principal/escape.txt", "content": "forbidden"})
        self.assertEqual(escaped.status, ExecutionStatus.DENIED)
        self.assertFalse(core_target.exists())

    def test_09_delete_requires_strong_confirmation(self) -> None:
        path = WORK / "delete-me.txt"
        path.write_text("x", encoding="utf-8")
        denied = self.request("filesystem.delete", {"path": str(path)})
        self.assertEqual(denied.status, ExecutionStatus.DENIED)
        self.assertTrue(path.exists())
        allowed = self.request("filesystem.delete", {"path": str(path)}, confirmed=(PermissionEffect.DELETE,))
        self.assertEqual(allowed.status, ExecutionStatus.SUCCESS)
        self.assertFalse(path.exists())

    def test_10_reflection_is_mandatory(self) -> None:
        result = self.request("filesystem.inspect", {"path": str(WORK)}, reflection=False)
        self.assertEqual(result.status, ExecutionStatus.DENIED)
        self.assertEqual(result.error, "DENIED_REFLECTION_REQUIRED")

    def test_11_composition_txt_to_pdf(self) -> None:
        source = WORK / "compose.txt"
        source.write_text("contenido compuesto", encoding="utf-8")
        destination = WORK / "compose.pdf"
        plan = CompositionPlan(
            "txt-to-pdf",
            (
                CompositionStep("read", "filesystem.read", {"path": str(source)}),
                CompositionStep("pdf", "document.create_pdf", {"path": str(destination), "content": "$steps.read.content"}),
            ),
        )
        template = ToolRequest(
            request_id="composition",
            turn_id="test-turn",
            tool_id="",
            capability="",
            reflection_approved=True,
            explicit_user_request=True,
        )
        result = ToolComposer(self.registry, self.executor).execute(plan, template)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertTrue(destination.is_file())

    def test_12_audit_is_structured_and_redacted(self) -> None:
        secretish = WORK / "audit.txt"
        self.request("filesystem.create_text", {"path": str(secretish), "content": "private body"})
        entries = [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(entries)
        serialized = json.dumps(entries)
        self.assertNotIn("private body", serialized)
        self.assertIn("<redacted:length=12>", serialized)

    def test_13_malformed_worker_output_is_isolated(self) -> None:
        class MalformedProcess:
            returncode = 0
            pid = 999999

            def communicate(self, *args, **kwargs):
                return "not-json", ""

            def poll(self):
                return 0

        with patch("anydes_tools.executor.subprocess.Popen", return_value=MalformedProcess()):
            result = self.request("filesystem.inspect", {"path": str(WORK)})
        self.assertEqual(result.status, ExecutionStatus.INVALID_OUTPUT)
        self.assertEqual(result.error, "malformed_worker_output")

    def test_14_existing_reflection_action_bridge(self) -> None:
        path = WORK / "bridge.txt"
        path.write_text("bridge works", encoding="utf-8")
        result = AnydesActionBridge(self.registry, self.executor).execute(
            SimpleNamespace(id="bridge-request", action_type="filesystem.read", parameters={"path": str(path)}),
            SimpleNamespace(approved=True),
            turn_id="bridge-turn",
            authorized_scopes=(),
            explicit_user_request=True,
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["output"]["content"], "bridge works")

    def test_15_composition_stops_after_failure(self) -> None:
        destination = WORK / "must-not-exist.pdf"
        plan = CompositionPlan(
            "stop-on-failure",
            (
                CompositionStep("read", "filesystem.read", {"path": str(WORK / "missing.txt")}),
                CompositionStep("pdf", "document.create_pdf", {"path": str(destination), "content": "$steps.read.content"}),
            ),
        )
        template = ToolRequest(
            request_id="failure-composition",
            turn_id="test-turn",
            tool_id="",
            capability="",
            reflection_approved=True,
            explicit_user_request=True,
        )
        result = ToolComposer(self.registry, self.executor).execute(plan, template)
        self.assertEqual(len(result.steps), 1)
        self.assertFalse(destination.exists())

    def test_16_protected_surface_matrix(self) -> None:
        protected = (
            ROOT / ".env",
            ROOT / "Abrir Anydes.bat",
            ROOT / "programa_principal",
            ROOT / "data",
            TOOLS / "manifests",
            TOOLS / "official",
            TOOLS / "generated",
        )
        for path in protected:
            with self.subTest(path=path):
                result = self.request("filesystem.inspect", {"path": str(path)})
                self.assertEqual(result.status, ExecutionStatus.DENIED)
                self.assertEqual(result.error, "DENIED_CORE_PROTECTED")

    def test_17_registry_lifecycle(self) -> None:
        registry = ToolRegistry()
        tool = self.registry.get("official.filesystem")
        registry.register(tool)
        self.assertTrue(registry.find_by_capability("filesystem.read"))
        registry.disable("official.filesystem")
        self.assertFalse(registry.find_by_capability("filesystem.read"))
        registry.enable("official.filesystem")
        self.assertTrue(registry.find_by_capability("filesystem.read"))
        registry.unregister("official.filesystem")
        self.assertFalse(registry.list_tools())

    def test_18_managed_developer_flow(self) -> None:
        created = self.request("developer.create_workspace", {"name": "managed_dev_v001"})
        self.assertEqual(created.status, ExecutionStatus.SUCCESS)
        source = MANAGED_DEV / "managed.py"
        written = self.request(
            "developer.write_code",
            {"path": str(source), "code": "value = 4\nprint(value)\n"},
        )
        self.assertEqual(written.status, ExecutionStatus.SUCCESS)
        patched = self.request(
            "developer.patch_code",
            {"path": str(source), "old_text": "value = 4", "new_text": "value = 7"},
        )
        self.assertEqual(patched.status, ExecutionStatus.SUCCESS)
        tested = self.request(
            "developer.test",
            {"path": str(source), "language": "python"},
            confirmed=(PermissionEffect.EXECUTE_CODE,),
        )
        self.assertEqual(tested.status, ExecutionStatus.SUCCESS)
        self.assertIn("7", tested.output["stdout"])
        inspected = self.request(
            "developer.inspect_error",
            {"stderr": "Traceback\nRuntimeError: sample"},
        )
        self.assertEqual(inspected.status, ExecutionStatus.SUCCESS)
        self.assertEqual(inspected.output["category"], "runtime_error")

    def test_19_document_formats_in_authorized_external_scope(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anydes-tools-v001-") as temporary:
            scope = (temporary,)
            cases = (
                ("document.create_txt", "human.txt"),
                ("document.create_markdown", "human.md"),
                ("document.create_docx", "human.docx"),
                ("document.create_pdf", "human.pdf"),
            )
            for capability, name in cases:
                with self.subTest(capability=capability):
                    path = Path(temporary) / name
                    result = self.request(
                        capability,
                        {"path": str(path), "content": "Prueba humana de Anydes."},
                        scopes=scope,
                    )
                    self.assertEqual(result.status, ExecutionStatus.SUCCESS, result.error)
                    self.assertTrue(path.is_file())

    def test_20_worker_minimal_environment_has_private_home_and_no_secrets(self) -> None:
        runtime_home, runtime_temp = _prepare_worker_runtime(TOOLS, "environment-regression")
        try:
            env = _worker_environment(ROOT, runtime_home, runtime_temp)
            for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"):
                self.assertNotIn(key, env)
            probe = (
                "import json, os; from pathlib import Path; "
                "print(json.dumps({'home':str(Path.home()),'expanded':os.path.expanduser('~'),"
                "'temp':os.environ.get('TEMP'),'tmp':os.environ.get('TMP')}))"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", probe],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            values = json.loads(completed.stdout)
            self.assertEqual(Path(values["home"]).resolve(), runtime_home)
            self.assertEqual(Path(values["expanded"]).resolve(), runtime_home)
            self.assertEqual(Path(values["temp"]).resolve(), runtime_temp)
            self.assertEqual(Path(values["tmp"]).resolve(), runtime_temp)
            self.assertEqual(
            self.boundary.runtime_directory(runtime_home, "runtime_home"),
                runtime_home,
            )
        finally:
            _cleanup_worker_runtime(runtime_home, runtime_temp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
