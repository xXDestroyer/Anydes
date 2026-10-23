from __future__ import annotations

from pathlib import Path

from .contracts import (
    CapabilityIntent,
    CapabilityResolution,
    CapabilityResolutionKind,
)
from .path_policy import FilesystemBoundary
from .registry import ToolRegistry


_FORBIDDEN_PREFIXES = (
    "browser.",
    "email.",
    "calendar.",
    "payment.",
    "system.global_control",
    "dependency.install",
    "tool.generate",
    "core.",
)

_RECIPES = {
    "text_files_to_pdf": (
        "filesystem.search",
        "filesystem.read",
        "document.create_pdf",
    ),
    "debug_python": (
        "developer.create_workspace",
        "developer.write_code",
        "developer.run_python",
        "developer.inspect_error",
    ),
}


class CapabilityResolver:
    def __init__(self, registry: ToolRegistry, boundary: FilesystemBoundary) -> None:
        self.registry = registry
        self.boundary = boundary

    def resolve(self, intention: CapabilityIntent) -> CapabilityResolution:
        required = intention.required_capabilities
        if intention.target_path:
            decision = self.boundary.classify(
                Path(intention.target_path),
                operation="READ",
                authorized_scopes=(),
            )
            if decision.category == "PROTECTED_CORE":
                return CapabilityResolution(
                    kind=CapabilityResolutionKind.FORBIDDEN,
                    capabilities=required,
                    reason="DENIED_CORE_PROTECTED",
                )
        if any(item.startswith(_FORBIDDEN_PREFIXES) for item in required):
            return CapabilityResolution(
                kind=CapabilityResolutionKind.FORBIDDEN,
                capabilities=required,
                reason="capability_forbidden_in_v001",
            )
        if intention.operation in _RECIPES:
            recipe = _RECIPES[intention.operation]
            if all(self.registry.find_by_capability(item) for item in recipe):
                return CapabilityResolution(
                    kind=CapabilityResolutionKind.COMPOSABLE,
                    tool_ids=tuple(
                        self.registry.find_by_capability(item)[0].manifest.tool_id
                        for item in recipe
                    ),
                    capabilities=recipe,
                    reason="known_composition_recipe",
                    recipe_id=intention.operation,
                )
        if not required:
            return CapabilityResolution(
                kind=CapabilityResolutionKind.REASONING_ONLY,
                reason="no_operational_capability_required",
            )
        matches = [self.registry.find_by_capability(item) for item in required]
        missing = [item for item, tools in zip(required, matches) if not tools]
        if missing:
            if intention.allow_temporary_script and all(
                item.startswith("developer.") for item in missing
            ):
                return CapabilityResolution(
                    kind=CapabilityResolutionKind.TEMPORARY_SCRIPT,
                    capabilities=tuple(missing),
                    reason="temporary_script_requires_explicit_developer_flow",
                )
            return CapabilityResolution(
                kind=CapabilityResolutionKind.MISSING_CAPABILITY,
                capabilities=tuple(missing),
                reason="no_verified_tool_registered; generation_disabled_in_v001",
            )
        tool_ids = tuple(tools[0].manifest.tool_id for tools in matches)
        kind = (
            CapabilityResolutionKind.EXACT_TOOL
            if len(required) == 1
            else CapabilityResolutionKind.COMPOSABLE
        )
        return CapabilityResolution(
            kind=kind,
            tool_ids=tool_ids,
            capabilities=required,
            reason="verified_registry_match",
        )
