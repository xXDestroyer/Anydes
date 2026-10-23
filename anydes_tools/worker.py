from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

from .context import ToolRuntimeContext
from .path_policy import FilesystemBoundary
from .registry import build_official_registry
from .executor import _preflight_paths, _verify_path_snapshot, _manifest_hash
from .schema_validation import validate
from .contracts import ToolRequest, PermissionEffect
from .permissions import PermissionBridge
from .errors import PathDeniedError


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        project_root = Path(payload["project_root"]).resolve()
        tools_root = Path(payload["tools_root"]).resolve()
        boundary = FilesystemBoundary(project_root, tools_root)
        registry = build_official_registry(tools_root)
        tool = registry.get(str(payload["tool_id"]))
        capability = str(payload["capability"])
        denial = registry.execution_denial(str(payload["tool_id"]), capability)
        if denial:
            raise PathDeniedError(denial)
        if payload.get("manifest_hash") != _manifest_hash(tool.manifest):
            raise PathDeniedError("DENIED_MANIFEST_CHANGED")
        contract = tool.manifest.capability_contracts[capability]
        if not validate(payload.get("arguments"), contract["input_schema"]):
            raise PathDeniedError("DENIED_SCHEMA_INVALID")
        request = ToolRequest("worker", "worker", tool.manifest.tool_id, capability,
            payload["arguments"], tuple(payload.get("authorized_scopes", [])),
            tuple(PermissionEffect(v) for v in payload.get("confirmed_effects", [])), True, True)
        decision = PermissionBridge().authorize(tool.manifest, request, tool.effect_for(capability, request.arguments))
        if not decision.allowed:
            raise PathDeniedError(decision.reason)
        _preflight_paths(request, boundary)
        _verify_path_snapshot(payload.get("path_snapshot", {}))
        context = ToolRuntimeContext(
            project_root=project_root,
            tools_root=tools_root,
            boundary=boundary,
            authorized_scopes=tuple(str(item) for item in payload.get("authorized_scopes", [])),
            runtime_home=boundary.runtime_directory(payload["runtime_home"], "runtime_home"),
            runtime_temp=boundary.runtime_directory(payload["runtime_temp"], "runtime_temp"),
        )
        result = tool.execute(
            str(payload["capability"]),
            dict(payload.get("arguments", {})),
            context,
        )
        side_effects = result.pop("_side_effects", [])
        sys.stdout.write(
            json.dumps(
                {"ok": True, "output": result, "side_effects": side_effects},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc(limit=8),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
