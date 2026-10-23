from __future__ import annotations

import json
from dataclasses import dataclass

from anydes.cognition.contracts import ActionRequest, ActionResult, Intention, Observation, ReflectionResult
from anydes.funciones_extra.registry import ToolRegistry
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.permissions import PermissionManager


ACTION_TOOL_ALIASES = {
    "search_google": "chrome_search",
    "google_search": "chrome_search",
    "search_youtube": "youtube_search",
    "open_spotify": "open_spotify",
    "open_youtube": "open_youtube",
    "open_url": "open_url",
    "open_app": "open_app",
}


@dataclass
class ActionManager:
    """Executes approved ActionRequests through the existing ToolRegistry.

    This manager does not reason, reflect, write memory, speak through TTS, or
    modify Assistant.respond(). It only executes tool actions after Reflection
    approval.
    """

    tools: ToolRegistry
    permissions: PermissionManager
    audit: AuditLog

    def action_requests_from_intention(self, intention: Intention) -> tuple[ActionRequest, ...]:
        return tuple(intention.actions)

    def execute(
        self,
        intention: Intention,
        reflection: ReflectionResult,
        *,
        interactive: bool,
    ) -> tuple[ActionResult, ...]:
        requests = self.action_requests_from_intention(intention)
        if not requests:
            return ()

        if not reflection.approved:
            status = "cancelled" if reflection.needs_confirmation else "blocked"
            return tuple(
                ActionResult(
                    request_id=request.id,
                    status=status,
                    message=f"Action not executed because reflection decision was {reflection.decision}.",
                    error=reflection.reason or None,
                )
                for request in requests
            )

        results: list[ActionResult] = []
        for request in requests:
            tool_name = self._tool_name_for_request(request)
            if tool_name is None:
                results.append(
                    ActionResult(
                        request_id=request.id,
                        status="failed",
                        message=f"No registered tool matches action_type '{request.action_type}'.",
                        error="unknown_action_type",
                    )
                )
                continue

            argument = self._argument_for_request(request)
            tool_result = self.tools.run_structured(
                tool_name,
                argument,
                self.permissions,
                self.audit,
                interactive=interactive,
            )
            status = _action_status_from_tool_status(tool_result.status)
            results.append(
                ActionResult(
                    request_id=request.id,
                    status=status,
                    message="tool_result_available" if status == "executed" else "tool_not_executed",
                    output={
                        "tool_name": tool_name,
                        "argument": argument,
                        "tool_status": tool_result.status,
                        "data": tool_result.data,
                    },
                    error=tool_result.error,
                )
            )
        return tuple(results)

    def execute_explicit_request(
        self,
        tool_name: str,
        argument: str = "",
    ) -> ActionResult:
        """Execute one already matched owner request through normal permissions."""
        request_id = f"realtime:{tool_name}"
        known = {tool.name for tool in self.tools.list_tools()}
        if tool_name not in known:
            return ActionResult(
                request_id=request_id,
                status="failed",
                message="tool_not_executed",
                output={
                    "tool_name": tool_name,
                    "argument": argument,
                    "tool_status": "missing",
                    "data": {"requested_tool": tool_name},
                },
                error="unknown_tool",
            )
        tool_result = self.tools.run_structured(
            tool_name,
            argument,
            self.permissions,
            self.audit,
            interactive=False,
            explicit_request=True,
        )
        status = _action_status_from_tool_status(tool_result.status)
        return ActionResult(
            request_id=request_id,
            status=status,
            message="tool_result_available" if status == "executed" else "tool_not_executed",
            output={
                "tool_name": tool_name,
                "argument": argument,
                "tool_status": tool_result.status,
                "data": tool_result.data,
            },
            error=tool_result.error,
        )

    def observations_from_results(self, results: tuple[ActionResult, ...]) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        for result in results:
            tool_name = str(result.output.get("tool_name", "unknown_tool"))
            tool_status = str(result.output.get("tool_status", result.status))
            observations.append(
                Observation(
                    id=f"observation:tool:{result.request_id}",
                    source="tool",
                    summary=f"Tool {tool_name} finished with status {tool_status}.",
                    confidence=1.0 if result.status == "executed" else 0.7,
                    salience=0.75 if result.status == "executed" else 0.65,
                    data={
                        "request_id": result.request_id,
                        "status": result.status,
                        "tool_name": tool_name,
                        "tool_status": tool_status,
                        "argument": result.output.get("argument", ""),
                        "result": result.output.get("data", {}),
                        "error": result.error,
                    },
                )
            )
        return tuple(observations)

    def _tool_name_for_request(self, request: ActionRequest) -> str | None:
        candidate = ACTION_TOOL_ALIASES.get(request.action_type, request.action_type)
        known = {tool.name for tool in self.tools.list_tools()}
        return candidate if candidate in known else None

    def _argument_for_request(self, request: ActionRequest) -> str:
        params = request.parameters
        for key in ("argument", "query", "url", "text", "name", "value"):
            value = params.get(key)
            if value is not None:
                return str(value)
        if request.target:
            return request.target
        if params:
            return json.dumps(params, ensure_ascii=True, sort_keys=True)
        return ""


def _action_status_from_tool_status(status: str) -> str:
    if status == "success":
        return "executed"
    if status == "denied":
        return "cancelled"
    if status == "missing":
        return "failed"
    return "failed"
