from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ExecutionStatus, PermissionEffect, ToolRequest
from .executor import ToolExecutor
from .registry import ToolRegistry


@dataclass
class AnydesActionBridge:
    """Adapter for existing ActionRequest/ReflectionResult without Core edits."""

    registry: ToolRegistry
    executor: ToolExecutor

    def execute(
        self,
        action_request: Any,
        reflection_result: Any,
        *,
        turn_id: str,
        authorized_scopes: tuple[str, ...],
        confirmed_effects: tuple[PermissionEffect, ...] = (),
        explicit_user_request: bool,
    ) -> Any:
        capability = str(action_request.action_type)
        tools = self.registry.find_by_capability(capability)
        if not tools:
            return {
                "request_id": str(action_request.id),
                "status": "failed",
                "error": "MISSING_CAPABILITY",
            }
        result = self.executor.execute(
            ToolRequest(
                request_id=str(action_request.id),
                turn_id=turn_id,
                tool_id=tools[0].manifest.tool_id,
                capability=capability,
                arguments=dict(action_request.parameters),
                authorized_scopes=authorized_scopes,
                confirmed_effects=confirmed_effects,
                reflection_approved=bool(reflection_result.approved),
                explicit_user_request=explicit_user_request,
            )
        )
        return {
            "request_id": result.request_id,
            "status": "executed" if result.status == ExecutionStatus.SUCCESS else "failed",
            "output": result.output,
            "error": result.error or None,
        }
