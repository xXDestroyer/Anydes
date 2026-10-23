from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..contracts import OfficialTool, ToolCapabilitySnapshot, ToolManifest, ToolStatus
from ..schema_validation import check_schema


ToolFactory = Callable[[ToolManifest], OfficialTool]


class ToolRegistry:
    """Verified registry. It never imports code named by a manifest."""

    def __init__(self) -> None:
        self._tools: dict[str, OfficialTool] = {}
        self._enabled: dict[str, bool] = {}
        self._capability_enabled: dict[str, bool] = {}

    def register(self, tool: OfficialTool) -> None:
        manifest = tool.manifest
        if set(manifest.capability_contracts) != set(manifest.capabilities):
            raise ValueError("capability_contracts_required")
        for cap, contract in manifest.capability_contracts.items():
            required = {"id", "enabled", "input_schema", "output_schema", "side_effects", "risk_level", "permission_requirements", "allowed_scope_types", "timeout", "verifier_type"}
            if not required.issubset(contract) or not isinstance(contract["timeout"], (int, float)) or not 0 < contract["timeout"] <= manifest.timeout:
                raise ValueError("invalid_capability_contract")
            if contract.get("id") != cap or type(contract.get("enabled")) is not bool:
                raise ValueError("invalid_capability_contract")
            check_schema(contract["input_schema"])
            check_schema(contract["output_schema"])
        if manifest.tool_id in self._tools:
            raise ValueError(f"duplicate_tool_id:{manifest.tool_id}")
        collisions = set(manifest.capabilities).intersection(self.list_capabilities())
        if collisions:
            raise ValueError(f"duplicate_capabilities:{','.join(sorted(collisions))}")
        health, reason = tool.health()
        if manifest.status == ToolStatus.QUARANTINED:
            raise ValueError(f"quarantined_tool:{manifest.tool_id}")
        self._tools[manifest.tool_id] = tool
        self._capability_enabled.update({cap: data["enabled"] for cap, data in manifest.capability_contracts.items()})
        self._enabled[manifest.tool_id] = (
            manifest.status == ToolStatus.AVAILABLE and health == ToolStatus.AVAILABLE
        )
        if health == ToolStatus.BROKEN:
            self._enabled[manifest.tool_id] = False
        _ = reason

    def unregister(self, tool_id: str) -> None:
        if tool_id in self._tools:
            for cap in self.get(tool_id).manifest.capabilities:
                self._capability_enabled.pop(cap, None)
        self._tools.pop(tool_id, None)
        self._enabled.pop(tool_id, None)

    def enable(self, tool_id: str) -> None:
        tool = self.get(tool_id)
        if tool.manifest.status != ToolStatus.AVAILABLE:
            raise RuntimeError("DENIED_TOOL_UNAVAILABLE")
        status, reason = tool.health()
        if status != ToolStatus.AVAILABLE:
            raise RuntimeError(f"tool_not_healthy:{tool_id}:{reason}")
        self._enabled[tool_id] = True

    def disable(self, tool_id: str) -> None:
        self.get(tool_id)
        self._enabled[tool_id] = False

    def disable_capability(self, capability: str) -> None:
        if capability not in self._capability_enabled:
            raise KeyError(capability)
        self._capability_enabled[capability] = False

    def enable_capability(self, capability: str) -> None:
        tool = next(t for t in self._tools.values() if capability in t.manifest.capabilities)
        if tool.manifest.capability_contracts[capability]["enabled"] is not True:
            raise RuntimeError("DENIED_CAPABILITY_DISABLED")
        self._capability_enabled[capability] = True

    def execution_denial(self, tool_id: str, capability: str) -> str:
        tool = self._tools.get(tool_id)
        if tool is None or not self._enabled.get(tool_id) or tool.manifest.status != ToolStatus.AVAILABLE:
            return "DENIED_TOOL_UNAVAILABLE"
        if capability not in tool.manifest.capabilities:
            return "DENIED_CAPABILITY_UNKNOWN"
        if not self._capability_enabled.get(capability) or tool.manifest.capability_contracts[capability]["enabled"] is not True:
            return "DENIED_CAPABILITY_DISABLED"
        try:
            if tool.health()[0] != ToolStatus.AVAILABLE:
                return "DENIED_TOOL_UNAVAILABLE"
        except Exception:
            return "DENIED_TOOL_UNAVAILABLE"
        return ""

    def get(self, tool_id: str) -> OfficialTool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"unknown_tool:{tool_id}") from exc

    def list_tools(self, *, enabled_only: bool = False) -> tuple[OfficialTool, ...]:
        items = (
            tool
            for tool_id, tool in self._tools.items()
            if not enabled_only or self._enabled.get(tool_id, False)
        )
        return tuple(sorted(items, key=lambda item: item.manifest.tool_id))

    def list_capabilities(self, *, enabled_only: bool = False) -> tuple[str, ...]:
        return tuple(
            sorted(
                capability
                for tool in self.list_tools(enabled_only=enabled_only)
                for capability in tool.manifest.capabilities
                if not enabled_only or not self.execution_denial(tool.manifest.tool_id, capability)
            )
        )

    def find_by_capability(self, capability: str) -> tuple[OfficialTool, ...]:
        return tuple(
            tool
            for tool in self.list_tools(enabled_only=True)
            if capability in tool.manifest.capabilities
            and not self.execution_denial(tool.manifest.tool_id, capability)
        )

    def health_status(self) -> dict[str, dict[str, str | bool]]:
        result: dict[str, dict[str, str | bool]] = {}
        for tool_id, tool in sorted(self._tools.items()):
            status, reason = tool.health()
            result[tool_id] = {
                "status": status.value,
                "reason": reason,
                "enabled": self._enabled.get(tool_id, False),
            }
        return result

    def capability_snapshot(self) -> ToolCapabilitySnapshot:
        return ToolCapabilitySnapshot(
            source="ToolRegistry",
            available_capabilities=self.list_capabilities(enabled_only=True),
            tools=tuple(
                {
                    "tool_id": tool.manifest.tool_id,
                    "version": tool.manifest.version,
                    "status": self.health_status()[tool.manifest.tool_id]["status"],
                }
                for tool in self.list_tools(enabled_only=True)
            ),
        )


def build_official_registry(tools_root: Path) -> ToolRegistry:
    """Load only the four factories explicitly approved for v001."""

    from ..official.archive_tool import ArchiveTool
    from ..official.developer_tool import DeveloperTool
    from ..official.document_tool import DocumentTool
    from ..official.filesystem_tool import FileSystemTool

    factories: dict[str, ToolFactory] = {
        "official.filesystem": FileSystemTool,
        "official.developer": DeveloperTool,
        "official.document": DocumentTool,
        "official.archive": ArchiveTool,
    }
    manifest_dir = tools_root / "manifests"
    registry = ToolRegistry()
    for tool_id, factory in factories.items():
        path = manifest_dir / f"{tool_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = ToolManifest.from_dict(payload)
        if manifest.tool_id != tool_id:
            raise ValueError(f"manifest_tool_id_mismatch:{path.name}")
        registry.register(factory(manifest))
    return registry
