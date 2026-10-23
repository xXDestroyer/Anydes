from __future__ import annotations

from dataclasses import replace
from typing import Any

from .contracts import (
    CompositionPlan,
    CompositionResult,
    ExecutionStatus,
    PermissionEffect,
    ToolRequest,
)
from .executor import ToolExecutor
from .registry import ToolRegistry


class ToolComposer:
    def __init__(self, registry: ToolRegistry, executor: ToolExecutor) -> None:
        self.registry = registry
        self.executor = executor

    def execute(
        self,
        plan: CompositionPlan,
        request_template: ToolRequest,
    ) -> CompositionResult:
        results = []
        outputs: dict[str, dict[str, Any]] = {}
        for step in plan.steps:
            tools = self.registry.find_by_capability(step.capability)
            if not tools:
                return CompositionResult(
                    plan.plan_id,
                    ExecutionStatus.MISSING,
                    tuple(results),
                    f"missing_capability:{step.capability}",
                )
            try:
                arguments = _resolve_references(step.arguments, outputs)
            except (KeyError, ValueError) as exc:
                return CompositionResult(
                    plan.plan_id,
                    ExecutionStatus.INVALID_INPUT,
                    tuple(results),
                    f"invalid_step_reference:{exc}",
                )
            request = replace(
                request_template,
                request_id=f"{request_template.request_id}:{step.step_id}",
                tool_id=tools[0].manifest.tool_id,
                capability=step.capability,
                arguments=arguments,
            )
            result = self.executor.execute(request)
            results.append(result)
            if result.status != ExecutionStatus.SUCCESS:
                return CompositionResult(
                    plan.plan_id,
                    result.status,
                    tuple(results),
                    f"step_failed:{step.step_id}:{result.error}",
                )
            outputs[step.step_id] = result.output
        return CompositionResult(plan.plan_id, ExecutionStatus.SUCCESS, tuple(results))


def _resolve_references(value: Any, outputs: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_references(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, outputs) for item in value]
    if not isinstance(value, str) or not value.startswith("$steps."):
        return value
    parts = value.split(".")[1:]
    if len(parts) < 2:
        raise ValueError(value)
    current: Any = outputs[parts[0]]
    for part in parts[1:]:
        current = current[part]
    return current
