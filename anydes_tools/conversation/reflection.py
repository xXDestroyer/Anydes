from __future__ import annotations

from dataclasses import dataclass

from ..contracts import CapabilityResolution, CapabilityResolutionKind
from .contracts import PlannedToolAction


@dataclass(frozen=True)
class ToolReflectionDecision:
    approved: bool
    reason: str


class ConversationToolReflection:
    """Local authority gate between natural-language planning and execution."""

    def reflect(
        self,
        action: PlannedToolAction,
        resolution: CapabilityResolution,
    ) -> ToolReflectionDecision:
        if not action.explicit_request or not action.source_text.strip():
            return ToolReflectionDecision(False, "explicit_user_request_required")
        if resolution.kind == CapabilityResolutionKind.FORBIDDEN:
            return ToolReflectionDecision(False, resolution.reason or "forbidden")
        if resolution.kind in {
            CapabilityResolutionKind.MISSING_CAPABILITY,
            CapabilityResolutionKind.UNSUPPORTED,
            CapabilityResolutionKind.TEMPORARY_SCRIPT,
        }:
            return ToolReflectionDecision(False, resolution.reason or "capability_unavailable")
        if not resolution.tool_ids or not resolution.capabilities:
            return ToolReflectionDecision(False, "verified_tool_resolution_required")
        if any(item.startswith(("tool.generate", "dependency.install", "core.")) for item in resolution.capabilities):
            return ToolReflectionDecision(False, "self_extension_disabled")
        return ToolReflectionDecision(True, "approved_verified_v001_capabilities")
