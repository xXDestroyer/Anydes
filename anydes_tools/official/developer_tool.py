from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from ..context import ToolRuntimeContext
from ..contracts import PermissionEffect, ToolManifest, ToolStatus


_PYTHON_IMPORT_ALLOWLIST = frozenset(
    {"math", "json", "re", "statistics", "collections", "itertools", "functools", "decimal", "fractions", "random", "datetime"}
)
_PYTHON_BLOCKED_NAMES = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "breakpoint", "help", "globals", "locals", "vars"}
)
_CPP_BLOCKED = (
    "<filesystem>", "<fstream>", "<cstdio>", "<windows.h>", "<winsock", "system(", "popen(", "fopen(", "createprocess", "shellexecute", "winexec", "remove(", "rename("
)


class DeveloperTool:
    def __init__(self, manifest: ToolManifest) -> None:
        self.manifest = manifest

    def health(self) -> tuple[ToolStatus, str]:
        return ToolStatus.AVAILABLE, "python_ready;cpp_optional"

    def effect_for(self, capability: str, arguments: dict[str, Any]) -> PermissionEffect:
        if capability in {"developer.read_code", "developer.inspect_error"}:
            return PermissionEffect.READ
        if capability in {"developer.run_python", "developer.run_cpp", "developer.test"}:
            return PermissionEffect.EXECUTE_CODE
        if capability == "developer.patch_code":
            return PermissionEffect.EDIT
        if bool(arguments.get("overwrite")):
            return PermissionEffect.OVERWRITE
        return PermissionEffect.CREATE

    def execute(self, capability: str, arguments: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        handlers = {
            "developer.create_workspace": self._create_workspace,
            "developer.read_code": self._read_code,
            "developer.write_code": self._write_code,
            "developer.patch_code": self._patch_code,
            "developer.run_python": self._run_python,
            "developer.run_cpp": self._run_cpp,
            "developer.test": self._test,
            "developer.inspect_error": self._inspect_error,
        }
        return handlers[capability](arguments, context)

    def _path(self, context: ToolRuntimeContext, value: Any, operation: str) -> Path:
        return context.authorize_path(str(value), operation=operation, developer_only=True)

    def _create_workspace(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        name = str(args.get("name") or "workspace").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", name) or name in {".", ".."}:
            raise ValueError("invalid_workspace_name")
        path = self._path(context, context.tools_root / "workspaces" / name, "WRITE")
        if path.exists():
            raise FileExistsError(path)
        path.mkdir(parents=True)
        return _effect({"path": str(path)}, PermissionEffect.CREATE, path)

    def _read_code(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = self._path(context, args.get("path", ""), "READ")
        data = context.revalidate(path, "READ").read_text(encoding="utf-8")
        if len(data) > 1_000_000:
            raise ValueError("code_file_too_large")
        return {"path": str(path), "content": data}

    def _write_code(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = self._path(context, args.get("path", ""), "WRITE")
        existed = path.exists()
        if existed and not bool(args.get("overwrite", False)):
            raise FileExistsError(path)
        source = str(args.get("code", ""))
        _validate_source(path.suffix.removeprefix("."), source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with context.revalidate(path, "WRITE").open("w" if bool(args.get("overwrite")) else "x", encoding="utf-8") as handle:
            handle.write(source)
        effect = PermissionEffect.OVERWRITE if existed else PermissionEffect.CREATE
        return _effect({"path": str(path), "bytes": len(source.encode("utf-8"))}, effect, path)

    def _patch_code(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = self._path(context, args.get("path", ""), "WRITE")
        source = path.read_text(encoding="utf-8")
        old = str(args.get("old_text", ""))
        new = str(args.get("new_text", ""))
        if not old or old not in source:
            raise ValueError("patch_anchor_not_found")
        updated = source.replace(old, new, 1)
        _validate_source(path.suffix.removeprefix("."), updated)
        context.revalidate(path, "WRITE").write_text(updated, encoding="utf-8")
        return _effect({"path": str(path), "patched": True}, PermissionEffect.EDIT, path)

    def _run_python(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = self._path(context, args.get("path", ""), "READ")
        source = _execution_source(context, path)
        _validate_python(source)
        context.revalidate(path, "READ")
        return self._isolated(path, source, "python", args, context)

    def _run_cpp(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        path = self._path(context, args.get("path", ""), "READ")
        source = _execution_source(context, path)
        _validate_cpp(source)
        context.revalidate(path, "READ")
        return self._isolated(path, source, "cpp", args, context)

    def _isolated(self, path, source, language, args, context):
        from ..windows_sandbox import execute_private, Limits
        output = execute_private(context.tools_root, source, language=language,
            stdin=str(args.get("stdin", "")), limits=Limits(timeout=min(float(args.get("timeout", 8.0)), 20.0)),
            runtime_temp=context.runtime_temp)
        if output.get("timeout"):
            raise subprocess.TimeoutExpired("strong_windows_sandbox", args.get("timeout", 8.0))
        if output["resource_limit_events"]:
            raise RuntimeError("SANDBOX_RESOURCE_LIMIT:" + ",".join(output["resource_limit_events"]))
        # Execution contracts do not request artifact publication. Outputs stay
        # private; the trusted host API is used only by explicitly verified flows.
        return {"path": str(path), "returncode": output["returncode"],
                "stdout": output["stdout"], "stderr": output["stderr"],
                **({"phase": output["phase"]} if language == "cpp" else {})}

    def _test(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        language = str(args.get("language", "python")).casefold()
        return self._run_python(args, context) if language == "python" else self._run_cpp(args, context)

    def _inspect_error(self, args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        _ = context
        stderr = str(args.get("stderr", ""))[-12000:]
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        category = "runtime_error"
        if any("SyntaxError" in line for line in lines):
            category = "syntax_error"
        elif any(
            marker in line
            for line in lines
            for marker in ("RuntimeError", "ValueError", "TypeError", "AssertionError")
        ):
            category = "runtime_error"
        elif any("error:" in line.casefold() for line in lines):
            category = "compile_error"
        return {"category": category, "summary": lines[-1] if lines else "no_error_text", "lines": lines[-12:]}


def _execution_source(context, path):
    with context.revalidate(path, "READ").open("rb") as handle:
        content = handle.read(500_001)
    if len(content) > 500_000:
        raise ValueError("source_too_large")
    return content.decode("utf-8")


def _validate_source(language: str, source: str) -> None:
    if len(source.encode("utf-8")) > 500_000:
        raise ValueError("source_too_large")
    if language == "py":
        _validate_python(source)
    elif language in {"cpp", "cc", "cxx"}:
        _validate_cpp(source)
    else:
        raise ValueError("unsupported_language")


def _validate_python(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in _PYTHON_IMPORT_ALLOWLIST for alias in node.names):
                raise ValueError("python_import_not_allowed")
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in _PYTHON_IMPORT_ALLOWLIST:
                raise ValueError("python_import_not_allowed")
        if isinstance(node, ast.Name) and node.id in _PYTHON_BLOCKED_NAMES:
            raise ValueError(f"python_name_not_allowed:{node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("python_private_attribute_not_allowed")


def _validate_cpp(source: str) -> None:
    normalized = source.casefold().replace(" ", "")
    for token in _CPP_BLOCKED:
        if token.casefold().replace(" ", "") in normalized:
            raise ValueError(f"cpp_operation_not_allowed:{token}")


def _process_result(path: Path, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def _effect(result: dict[str, Any], effect: PermissionEffect, path: Path) -> dict[str, Any]:
    result["_side_effects"] = [{"effect": effect.value, "path": str(path)}]
    return result
