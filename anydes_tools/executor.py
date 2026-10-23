from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .audit import ToolAuditLog
from .contracts import (
    ExecutionStatus,
    FileEffect,
    PermissionEffect,
    ToolExecutionResult,
    ToolRequest,
)
from .permissions import PermissionBridge
from .errors import PathDeniedError
from .path_policy import FilesystemBoundary
from .registry import ToolRegistry
from .schema_validation import validate as _validate_schema


class ToolExecutor:
    """The only execution entry point exposed to Reflection/Action Manager."""

    def __init__(
        self,
        project_root: Path,
        tools_root: Path,
        registry: ToolRegistry,
        permissions: PermissionBridge,
        audit: ToolAuditLog,
    ) -> None:
        self.project_root = project_root.resolve()
        self.tools_root = tools_root.resolve()
        self.registry = registry
        self.permissions = permissions
        self.audit = audit

    def execute(self, request: ToolRequest) -> ToolExecutionResult:
        started = time.perf_counter()
        denial = self.registry.execution_denial(request.tool_id, request.capability)
        if denial:
            return self._security_failure(request, started, denial)
        try:
            tool = self.registry.get(request.tool_id)
        except KeyError:
            result = self._result(request, ExecutionStatus.MISSING, started, error="unknown_tool")
            self._audit_result(result, "not_evaluated", request)
            return result
        if request.capability not in tool.manifest.capabilities:
            result = self._result(
                request,
                ExecutionStatus.INVALID_INPUT,
                started,
                error="capability_not_declared",
            )
            self._audit_result(result, "not_evaluated", request)
            return result
        contract = tool.manifest.capability_contracts[request.capability]
        if not _validate_schema(request.arguments, contract["input_schema"]):
            result = self._result(
                request,
                ExecutionStatus.INVALID_INPUT,
                started,
                error="DENIED_SCHEMA_INVALID",
            )
            self._audit_result(result, "not_evaluated", request)
            return result
        effect = tool.effect_for(request.capability, request.arguments)
        decision = self.permissions.authorize(tool.manifest, request, effect)
        self.audit.record(
            "tool_requested",
            turn_id=request.turn_id,
            request_id=request.request_id,
            tool_id=request.tool_id,
            version=tool.manifest.version,
            capability=request.capability,
            inputs=request.arguments,
            permission=decision.reason,
            effect=effect.value,
        )
        if not decision.allowed:
            return self._security_failure(request, started, decision.reason)
        try:
            paths = _preflight_paths(
                request,
                FilesystemBoundary(self.project_root, self.tools_root),
            )
            snapshots = _path_snapshot(paths)
        except PathDeniedError as exc:
            return self._security_failure(request, started, str(exc))

        payload = {
            "project_root": str(self.project_root),
            "tools_root": str(self.tools_root),
            "tool_id": request.tool_id,
            "capability": request.capability,
            "arguments": request.arguments,
            "authorized_scopes": list(request.authorized_scopes),
            "confirmed_effects": [str(item) for item in request.confirmed_effects],
            "path_snapshot": snapshots,
            "manifest_hash": _manifest_hash(tool.manifest),
        }
        try:
            runtime_home, runtime_temp = _prepare_worker_runtime(
                self.tools_root, request.request_id,
            )
        except PathDeniedError as exc:
            return self._security_failure(request, started, str(exc))
        except OSError:
            return self._security_failure(request, started, "DENIED_RUNTIME_DIRECTORY_UNAVAILABLE")
        payload["runtime_home"] = str(runtime_home)
        payload["runtime_temp"] = str(runtime_temp)
        command = [sys.executable, "-m", "anydes_tools.worker"]
        denial = self.registry.execution_denial(request.tool_id, request.capability)
        if denial:
            _cleanup_worker_runtime(runtime_home, runtime_temp)
            return self._security_failure(request, started, denial)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(
                command, cwd=self.project_root, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", creationflags=creationflags,
                env=_worker_environment(self.project_root, runtime_home, runtime_temp),
            )
        except OSError:
            _cleanup_worker_runtime(runtime_home, runtime_temp)
            return self._security_failure(request, started, "DENIED_WORKER_START_FAILED")
        try:
            try:
                stdout, stderr = process.communicate(
                    json.dumps(payload, ensure_ascii=True),
                    timeout=min(tool.manifest.timeout, contract["timeout"]),
                )
            except subprocess.TimeoutExpired:
                _terminate_tree(process)
                stdout, stderr = process.communicate(timeout=5)
                result = self._result(
                    request,
                    ExecutionStatus.TIMEOUT,
                    started,
                    error="tool_timeout",
                    stdout=stdout,
                    stderr=stderr,
                )
                self._audit_result(result, decision.reason, request)
                return result
        finally:
            _cleanup_worker_runtime(runtime_home, runtime_temp)
        if len(stdout) > 2_000_000 or len(stderr) > 1_000_000:
            result = self._result(
                request,
                ExecutionStatus.INVALID_OUTPUT,
                started,
                error="worker_output_limit_exceeded",
            )
            self._audit_result(result, decision.reason, request)
            return result
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            result = self._result(
                request,
                ExecutionStatus.INVALID_OUTPUT,
                started,
                error="malformed_worker_output",
                stdout=stdout,
                stderr=stderr,
            )
            self._audit_result(result, decision.reason, request)
            return result
        if not isinstance(envelope, dict):
            return self._security_failure(request, started, "DENIED_INVALID_WORKER_ENVELOPE")
        if process.returncode != 0 or not envelope.get("ok"):
            worker_error = str(envelope.get("error") or f"worker_exit_{process.returncode}")
            failed_status = (
                ExecutionStatus.DENIED if "DENIED_" in worker_error else
                ExecutionStatus.TIMEOUT
                if "TimeoutExpired" in worker_error
                else ExecutionStatus.FAILED
            )
            result = self._result(
                request,
                failed_status,
                started,
                error=worker_error,
                stderr=stderr,
            )
            self._audit_result(result, decision.reason, request)
            return result
        output = envelope.get("output")
        if not isinstance(output, dict) or not _validate_schema(output, contract["output_schema"]):
            result = self._result(
                request,
                ExecutionStatus.INVALID_OUTPUT,
                started,
                error="output_schema_validation_failed",
            )
            self._audit_result(result, decision.reason, request)
            return result
        raw_effects = envelope.get("side_effects", [])
        if not _validate_schema(raw_effects, {
            "type": "array", "maxItems": 5000, "items": {
                "type": "object", "required": ["effect", "path"],
                "additionalProperties": False, "properties": {
                    "effect": {"enum": [effect.value for effect in PermissionEffect]},
                    "path": {"type": "string", "minLength": 1, "maxLength": 32767},
                },
            },
        }):
            return self._security_failure(request, started, "DENIED_INVALID_WORKER_EFFECTS")
        effects = tuple(FileEffect(PermissionEffect(item["effect"]), item["path"]) for item in raw_effects)
        result = self._result(
            request,
            ExecutionStatus.SUCCESS,
            started,
            output=output,
            side_effects=effects,
        )
        self._audit_result(result, decision.reason, request)
        return result

    def _security_failure(self, request, started, reason):
        result = self._result(request, ExecutionStatus.DENIED, started, error=reason)
        self._audit_result(result, reason, request)
        return result

    def _result(
        self,
        request: ToolRequest,
        status: ExecutionStatus,
        started: float,
        *,
        output: dict[str, Any] | None = None,
        error: str = "",
        side_effects: tuple[FileEffect, ...] = (),
        stdout: str = "",
        stderr: str = "",
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            request_id=request.request_id,
            turn_id=request.turn_id,
            tool_id=request.tool_id,
            capability=request.capability,
            status=status,
            output=output or {},
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            side_effects=side_effects,
            stdout=stdout[-8000:],
            stderr=stderr[-8000:],
        )

    def _audit_result(self, result: ToolExecutionResult, permission: str, request=None) -> None:
        typed_reason = re.search(r"DENIED_[A-Z_]+", result.error)
        self.audit.record("security_decision", tool_id=result.tool_id,
            capability_id=result.capability, scope_id=hashlib.sha256(json.dumps([str(self.tools_root / "workspaces"), *getattr(request, "authorized_scopes", ())], sort_keys=True).encode()).hexdigest()[:16],
            decision="DENY" if result.status in {ExecutionStatus.DENIED, ExecutionStatus.INVALID_INPUT} else "AUTHORIZED_DISPATCH",
            reason_code=typed_reason.group(0) if typed_reason else permission,
            task_id=getattr(request, "task_id", None), step_id=result.request_id, turn_id=result.turn_id)
        try:
            version = self.registry.get(result.tool_id).manifest.version
        except KeyError:
            version = "unknown"
        self.audit.record(
            "tool_completed",
            turn_id=result.turn_id,
            request_id=result.request_id,
            tool_id=result.tool_id,
            version=version,
            capability=result.capability,
            permission=permission,
            result=result.status.value,
            duration_ms=round(result.duration_ms, 3),
            side_effects=[{"effect": item.effect.value, "path": item.path} for item in result.side_effects],
            files_created=[item.path for item in result.side_effects if item.effect == PermissionEffect.CREATE],
            files_modified=[item.path for item in result.side_effects if item.effect in {PermissionEffect.EDIT, PermissionEffect.OVERWRITE}],
            files_deleted=[item.path for item in result.side_effects if item.effect == PermissionEffect.DELETE],
            error=result.error,
        )


def _manifest_hash(manifest):
    from dataclasses import asdict
    return hashlib.sha256(json.dumps(asdict(manifest), sort_keys=True).encode()).hexdigest()


def _path_identity(info):
    import stat
    return [info.st_dev, info.st_ino, info.st_size if stat.S_ISREG(info.st_mode) else None, info.st_mtime_ns if stat.S_ISREG(info.st_mode) else None]


def _path_snapshot(paths):
    snapshot = {}
    for path in paths:
        for item in (path, *path.parents):
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise PathDeniedError("DENIED_UNVERIFIABLE_PATH") from exc
            snapshot[str(item)] = _path_identity(info)
    return snapshot


def _verify_path_snapshot(snapshot):
    for path, identity in snapshot.items():
        try:
            info = Path(path).lstat()
        except OSError as exc:
            raise PathDeniedError("DENIED_STALE_PATH") from exc
        if _path_identity(info) != identity or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PathDeniedError("DENIED_STALE_PATH")


def _worker_environment(
    project_root: Path,
    runtime_home: Path,
    runtime_temp: Path,
) -> dict[str, str]:
    allowed = {"SystemRoot", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "ANYDES_SANDBOX_BACKEND"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    drive, home_path = os.path.splitdrive(str(runtime_home))
    roaming = runtime_home / "AppData" / "Roaming"
    local = runtime_home / "AppData" / "Local"
    roaming.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(runtime_home),
            "USERPROFILE": str(runtime_home),
            "HOMEDRIVE": drive,
            "HOMEPATH": home_path,
            "APPDATA": str(roaming),
            "LOCALAPPDATA": str(local),
            "TEMP": str(runtime_temp),
            "TMP": str(runtime_temp),
        }
    )
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _prepare_worker_runtime(tools_root: Path, request_id: str) -> tuple[Path, Path]:
    from uuid import uuid4
    identifier = uuid4().hex
    home = tools_root / "sandbox" / "runtime_home" / identifier
    temp = tools_root / "sandbox" / "runtime_temp" / identifier
    boundary = FilesystemBoundary(tools_root.parent, tools_root)
    if any(boundary._contains_reparse_point(p) for p in (home, temp)):
        raise PathDeniedError("DENIED_REPARSE_POINT")
    home.mkdir(parents=True, exist_ok=False)
    try:
        temp.mkdir(parents=True, exist_ok=False)
    except OSError:
        home.rmdir()
        raise
    return home.resolve(), temp.resolve()


def _cleanup_worker_runtime(runtime_home: Path, runtime_temp: Path) -> None:
    from .windows_sandbox import cleanup_worker_profiles
    cleanup_worker_profiles(runtime_temp)
    for path in (runtime_home, runtime_temp):
        shutil.rmtree(path, ignore_errors=True)


def _preflight_paths(request: ToolRequest, boundary: FilesystemBoundary) -> None:
    capability = request.capability
    args = request.arguments
    developer_only = capability.startswith("developer.")
    checked = []

    def check(key: str, operation: str) -> None:
        value = args.get(key)
        if value in {None, ""}:
            return
        checked.append(boundary.authorize(
            str(value),
            operation=operation,
            authorized_scopes=request.authorized_scopes,
            developer_only=developer_only,
        ))

    if capability == "filesystem.search":
        check("root", "READ")
    elif capability in {"filesystem.copy", "filesystem.move", "filesystem.atomic_replace"}:
        check("source", "WRITE" if capability in {"filesystem.move", "filesystem.atomic_replace"} else "READ")
        check("destination", "WRITE")
    elif capability == "filesystem.rename":
        check("path", "WRITE")
    elif capability == "archive.compress_zip":
        for value in args.get("sources", []):
            checked.append(boundary.authorize(
                str(value),
                operation="READ",
                authorized_scopes=request.authorized_scopes,
            ))
        check("destination", "WRITE")
    elif capability == "archive.extract_zip":
        check("path", "READ")
        check("destination", "WRITE")
    elif capability in {
        "filesystem.inspect",
        "filesystem.read",
        "developer.read_code",
        "developer.run_python",
        "developer.run_cpp",
        "developer.test",
        "document.read_document",
        "archive.inspect",
    }:
        check("path", "READ")
    else:
        check("path", "WRITE")
    if capability == "filesystem.rename":
        checked.append(boundary.authorize(Path(args["path"]).with_name(args["new_name"]), operation="WRITE", authorized_scopes=request.authorized_scopes))
    nested = args.get("document_edit_spec", {}).get("target_path")
    if nested:
        checked.append(boundary.authorize(nested, operation="WRITE", authorized_scopes=request.authorized_scopes))
    if capability == "developer.create_workspace":
        checked.append(boundary.authorize(boundary.workspace_root / args["name"], operation="WRITE", authorized_scopes=request.authorized_scopes))
    return checked


def _terminate_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    else:
        process.kill()
