"""B1 adversarial fixtures: never modify production targets."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anydes_tools.audit import ToolAuditLog
from anydes_tools.contracts import ToolRequest, ExecutionStatus, PermissionEffect, ToolStatus
from anydes_tools.errors import PathDeniedError
from anydes_tools.executor import ToolExecutor, _path_snapshot, _verify_path_snapshot
from anydes_tools.path_policy import FilesystemBoundary
from anydes_tools.permissions import PermissionBridge
from anydes_tools.registry import build_official_registry
from anydes_tools.schema_validation import validate, check_schema

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


class BoundaryB1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TOOLS / "workspaces", prefix="b1-")
        self.base = Path(self.temp.name)
        self.project = self.base / "project"
        self.tools = self.project / "anydes_tools"
        self.work = self.tools / "workspaces" / "allowed"
        self.work.mkdir(parents=True)
        self.boundary = FilesystemBoundary(self.project, self.tools)

    def tearDown(self):
        self.temp.cleanup()

    def deny(self, path, scopes=()):
        with self.assertRaises(PathDeniedError):
            self.boundary.authorize(path, operation="WRITE", authorized_scopes=scopes)

    def test_every_project_area_protected_even_with_grant(self):
        for relative in ("programa_principal/anydes/core/x.py", "anydes_tools/conversation/x.py",
            "anydes_tools/settings/x.py", "anydes_tools/registry/x.py", "anydes_tools/official/x.py",
            "anydes_tools/sandbox/implementation.py", "anydes_tools/executor.py", "anydes_tools/permissions.py",
            "anydes_tools/path_policy.py", "anydes_tools/manifests/x.json", "anydes_tools/tests/x.py",
            "anydes_tools/logs/audit.jsonl", "anydes_tools/dependencies/x.py", "memory/x", "SELF/x",
            "datos/x", "archivo_versiones/x", ".env", "Abrir Anydes.bat", "runtime_patch.py"):
            with self.subTest(relative=relative):
                self.deny(self.project / relative, (self.project / relative,))

    def test_traversal_absolute_escape_drive_and_namespaces(self):
        for value in ("../escape", "x/../../escape", str(self.base / "outside.txt"),
            "Z:/escape.txt", "C:relative.txt", r"\\server\share\file", r"\\?\C:\file", "secret.txt:stream", "NUL.txt", "tail. "):
            with self.subTest(value=value): self.deny(value)

    def test_broad_scope_is_not_silently_ignored(self):
        self.deny(self.work / "file", (self.project,))
        self.deny(self.work / "file", (self.base,))

    def test_authorized_external_scope_and_spaces(self):
        outside = self.base / "Allowed with spaces"
        outside.mkdir()
        result = self.boundary.authorize(outside / "file.txt", operation="WRITE", authorized_scopes=(outside,))
        self.assertEqual(result, outside / "file.txt")

    def test_unknown_operation_and_scope_root(self):
        with self.assertRaises(PathDeniedError): self.boundary.authorize(self.work, operation="EXECUTE")
        self.deny(self.work, (self.work,))

    def test_env_in_workspace(self):
        self.deny(self.work / ".env")
        self.deny(self.work / ".env.backup")

    def test_junction_escape(self):
        if os.name != "nt": self.skipTest("Windows junction fixture")
        target = self.base / "outside"
        target.mkdir()
        link = self.work / "junction"
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        self.assertEqual(completed.returncode, 0)
        try:
            self.deny(link / "escape.txt")
            self.deny(link / "escape.txt", (link,))
        finally:
            # rmdir removes only this verified fixture junction, never its target.
            self.assertEqual(link.parent, self.work)
            link.rmdir()

    def test_broken_symlink_escape(self):
        link = self.work / "broken"
        try: link.symlink_to(self.base / "missing")
        except OSError: self.skipTest("symlink privilege unavailable; junction tested separately")
        try: self.deny(link)
        finally: link.unlink()

    def test_unverifiable_reparse_is_denied(self):
        with patch.object(Path, "lstat", side_effect=PermissionError()):
            self.assertTrue(self.boundary._contains_reparse_point(self.work))

    def test_stale_file_and_directory_identity(self):
        file = self.work / "file.txt"
        file.write_text("before")
        snapshot = _path_snapshot([file])
        file.write_text("after changed size")
        with self.assertRaisesRegex(PathDeniedError, "DENIED_STALE_PATH"):
            _verify_path_snapshot(snapshot)

    def test_revalidate_rejects_swapped_parent(self):
        from anydes_tools.context import ToolRuntimeContext
        context = ToolRuntimeContext(self.project, self.tools, self.boundary, (), self.work, self.work)
        safe = context.authorize_path(self.work / "file.txt", operation="WRITE")
        with patch.object(self.boundary, "_contains_reparse_point", return_value=True):
            with self.assertRaises(PathDeniedError): context.revalidate(safe, "WRITE")


class ExecutorB1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TOOLS / "workspaces", prefix="b1exec-")
        self.work = Path(self.temp.name)
        self.registry = build_official_registry(TOOLS)
        self.log = self.work / "audit.jsonl"
        self.executor = ToolExecutor(ROOT, TOOLS, self.registry, PermissionBridge(), ToolAuditLog(self.log))

    def tearDown(self): self.temp.cleanup()

    def req(self, cap="filesystem.create_text", args=None, tool="official.filesystem", confirmed=()):
        return ToolRequest("b1-step", "b1-turn", tool, cap,
            args if args is not None else {"path": str(self.work / "created.txt"), "content": "hello"},
            (str(self.work),), confirmed, True, True)

    def reject_without_worker(self, request, reason):
        with patch("anydes_tools.executor.subprocess.Popen") as spawn:
            result = self.executor.execute(request)
        spawn.assert_not_called()
        self.assertIn(reason, result.error)
        self.assertNotEqual(result.status, ExecutionStatus.SUCCESS)

    def test_unknown_tool(self): self.reject_without_worker(self.req(tool="unknown"), "DENIED_TOOL_UNAVAILABLE")

    def test_disabled_tool(self):
        self.registry.disable("official.filesystem")
        self.reject_without_worker(self.req(), "DENIED_TOOL_UNAVAILABLE")

    def test_disabled_capability(self):
        self.registry.disable_capability("filesystem.create_text")
        self.assertNotIn("filesystem.create_text", self.registry.capability_snapshot().available_capabilities)
        self.reject_without_worker(self.req(), "DENIED_CAPABILITY_DISABLED")

    def test_wrong_tool_capability(self): self.reject_without_worker(self.req(cap="document.create_txt"), "DENIED_CAPABILITY_UNKNOWN")

    def test_unhealthy_tool(self):
        with patch.object(self.registry.get("official.filesystem"), "health", return_value=(ToolStatus.BROKEN, "fixture")):
            self.reject_without_worker(self.req(), "DENIED_TOOL_UNAVAILABLE")

    def test_invalid_types_required_extra_and_size(self):
        for args in ({"path": 4, "content": "a"}, {"path": str(self.work / "x")},
            {"path": str(self.work / "x"), "content": "a", "evil": True},
            {"path": str(self.work / "x"), "content": "a", "overwrite": "false"},
            {"path": str(self.work / "x"), "content": "a" * 1000001}):
            self.reject_without_worker(self.req(args=args), "DENIED_SCHEMA_INVALID")

    def test_nested_schema_and_enum(self):
        for operation, payload in (("BAD", "text"), ("APPEND_PARAGRAPH", ["bad"])):
            self.reject_without_worker(self.req("document.edit_document", {"path": str(self.work / "x.docx"), "document_edit_spec": {"operation": operation, "payload": payload}}, "official.document"), "DENIED_SCHEMA_INVALID")

    def test_oversized_array(self):
        self.reject_without_worker(self.req("archive.compress_zip", {"sources": [str(self.work / "x")] * 2001, "destination": str(self.work / "x.zip")}, "official.archive"), "DENIED_SCHEMA_INVALID")

    def test_permissions_remain_strict(self):
        for effect in (PermissionEffect.INSTALL_DEPENDENCY, PermissionEffect.EXTERNAL_ACTION):
            tool = self.registry.get("official.filesystem")
            manifest = replace(tool.manifest, permission_policy={effect.value: "confirm"})
            self.assertFalse(PermissionBridge().authorize(manifest, self.req(confirmed=(effect,)), effect).allowed)
        self.reject_without_worker(replace(self.req(), reflection_approved="true"), "DENIED_REFLECTION_REQUIRED")

    def test_real_worker_and_security_audit(self):
        self.assertEqual(self.executor.execute(self.req()).status, ExecutionStatus.SUCCESS)
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        decisions = [e["details"] for e in events if e["event"] == "security_decision"]
        self.assertTrue(decisions)
        self.assertNotIn(str(self.work), json.dumps(decisions))
        self.assertNotIn("hello", json.dumps(decisions))
        self.assertEqual(decisions[-1]["reason_code"], "AUTHORIZED_IN_SCOPE")

    def test_stale_destination_between_executor_and_worker(self):
        source, destination = self.work / "staged.txt", self.work / "final.txt"
        source.write_text("new")
        destination.write_text("old")
        real_spawn = subprocess.Popen
        def changed_before_spawn(*args, **kwargs):
            destination.write_text("fixture changed after preflight")
            return real_spawn(*args, **kwargs)
        request = self.req("filesystem.atomic_replace", {"source": str(source), "destination": str(destination), "commit_effect": "OVERWRITE"}, confirmed=(PermissionEffect.OVERWRITE,))
        with patch("anydes_tools.executor.subprocess.Popen", side_effect=changed_before_spawn):
            result = self.executor.execute(request)
        self.assertEqual(result.status, ExecutionStatus.DENIED)
        self.assertIn("DENIED_STALE_PATH", result.error)
        self.assertEqual(destination.read_text(), "fixture changed after preflight")
        self.assertTrue(source.exists())

    def test_create_collision_and_missing_confirmations(self):
        destination = self.work / "original.txt"
        destination.write_text("keep")
        result = self.executor.execute(self.req(args={"path": str(destination), "content": "new"}))
        self.assertNotEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(destination.read_text(), "keep")
        self.reject_without_worker(self.req(args={"path": str(destination), "content": "new", "overwrite": True}), "DENIED_CONFIRMATION_REQUIRED")
        self.reject_without_worker(self.req("filesystem.delete", {"path": str(destination)}), "DENIED_CONFIRMATION_REQUIRED")
        self.reject_without_worker(self.req("developer.run_python", {"path": str(self.work / "code.py")}, "official.developer"), "DENIED_CONFIRMATION_REQUIRED")

    def test_private_runtime_failure_is_typed(self):
        with patch("anydes_tools.executor._prepare_worker_runtime", side_effect=OSError("fixture")):
            result = self.executor.execute(self.req())
        self.assertEqual(result.status, ExecutionStatus.DENIED)
        self.assertEqual(result.error, "DENIED_RUNTIME_DIRECTORY_UNAVAILABLE")


class SchemaB1Tests(unittest.TestCase):
    def test_standard_nested_nullable_array_validation(self):
        schema = {"type": "object", "properties": {"items": {"type": "array", "maxItems": 2, "items": {"type": "object", "required": ["n"], "properties": {"n": {"type": ["integer", "null"], "minimum": 2}}, "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
        self.assertTrue(validate({"items": [{"n": None}, {"n": 2}]}, schema))
        self.assertFalse(validate({"items": [{"n": True}]}, schema))
        self.assertFalse(validate({"items": [{"n": 1}]}, schema))
        self.assertFalse(validate({"items": [{"n": 2, "extra": 1}]}, schema))

    def test_remote_refs_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): check_schema({"$ref": "https://example.com/schema"})
        self.assertFalse(validate(float("nan"), {"type": "number"}))


if __name__ == "__main__": unittest.main()
