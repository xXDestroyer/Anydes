from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from ..contracts import CapabilityIntent, CapabilityResolutionKind
from ..path_policy import FilesystemBoundary
from ..registry import ToolRegistry
from ..resolver import CapabilityResolver
from .code_generation import CodeGenerationSpec


_ALLOWED_TASK_TYPES = frozenset({
    "REASONING_ONLY",
    "GENERATE_CONVERSATIONAL_CONTENT",
    "GENERATE_CODE",
    "MODIFY_CODE",
    "GENERATE_DOCUMENT",
    "TOOL_COMPOSITION",
    "MISSING_CAPABILITY",
    "UNSUPPORTED",
    "FORBIDDEN",
})
_FORBIDDEN_CAPABILITIES = (
    "core.",
    "tool.generate",
    "dependency.install",
    "system.global_control",
    "browser.",
    "email.",
    "payment.",
)


@dataclass(frozen=True)
class GeneralTaskStep:
    step_id: str
    capability: str
    purpose: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    produces_artifacts: tuple[str, ...] = ()
    consumes_artifacts: tuple[str, ...] = ()
    produces_values: tuple[str, ...] = ()
    consumes_values: tuple[str, ...] = ()
    step_type: str = "CAPABILITY"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GeneralTaskStep":
        return cls(
            step_id=str(value.get("step_id", "")).strip(),
            capability=str(value.get("capability", "")).strip(),
            purpose=str(value.get("purpose", "")).strip(),
            arguments=dict(value.get("arguments") or {}),
            depends_on=tuple(str(item) for item in (value.get("depends_on") or ())),
            produces_artifacts=tuple(
                str(item) for item in (value.get("produces_artifacts") or ())
            ),
            consumes_artifacts=tuple(
                str(item) for item in (value.get("consumes_artifacts") or ())
            ),
            produces_values=tuple(str(item) for item in value.get("produces_values", ())),
            consumes_values=tuple(str(item) for item in value.get("consumes_values", ())),
            step_type=str(value.get("step_type", "CAPABILITY")),
        )


# Deprecated import location; one operative result contract is used in Phase C.
from .executable_plan import StepResult


@dataclass(frozen=True)
class GeneralTaskPlan:
    task_id: str
    goal: str
    task_type: str
    inputs: dict[str, Any]
    desired_outputs: tuple[dict[str, Any], ...]
    constraints: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    steps: tuple[GeneralTaskStep, ...]
    artifact_requirements: dict[str, Any]
    needs_content_generation: bool
    needs_reasoning: bool
    needs_tools: bool
    risk: str
    ambiguities: tuple[str, ...]
    success_criteria: tuple[str, ...]
    requested_final_outputs: tuple[dict, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GeneralTaskPlan":
        outputs = value.get("desired_outputs") or ()
        if isinstance(outputs, dict):
            outputs = (outputs,)
        output_rows = tuple(dict(item) for item in outputs if isinstance(item, dict))
        raw_steps = value.get("steps") or ()
        step_rows = tuple(dict(item) for item in raw_steps if isinstance(item, dict))
        return cls(
            task_id=str(value.get("task_id") or f"general:{uuid4().hex}"),
            goal=str(value.get("goal", "")).strip(),
            task_type=str(value.get("task_type", "")).strip().upper(),
            inputs=dict(value.get("inputs") or {}),
            desired_outputs=output_rows,
            constraints=tuple(
                str(item) for item in (value.get("constraints") or ()) if item is not None
            ),
            required_capabilities=tuple(
                str(item).strip()
                for item in (value.get("required_capabilities") or ())
                if item is not None
            ),
            steps=tuple(GeneralTaskStep.from_dict(item) for item in step_rows),
            artifact_requirements=dict(value.get("artifact_requirements") or {}),
            needs_content_generation=bool(value.get("needs_content_generation", False)),
            needs_reasoning=bool(value.get("needs_reasoning", False)),
            needs_tools=bool(value.get("needs_tools", False)),
            risk=str(value.get("risk", "low")).strip().casefold(),
            ambiguities=tuple(
                str(item) for item in (value.get("ambiguities") or ()) if item is not None
            ),
            success_criteria=tuple(
                str(item) for item in (value.get("success_criteria") or ()) if item is not None
            ),
            requested_final_outputs=tuple(dict(item) for item in value.get("requested_final_outputs", ()) if isinstance(item, dict)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneralTaskPlanValidation:
    valid: bool
    errors: tuple[str, ...]
    resolution_kind: str
    capabilities: tuple[str, ...]


class GeneralTaskReasoner(Protocol):
    def __call__(self, request: dict[str, Any]) -> GeneralTaskPlan:
        ...

    def repair(
        self,
        request: dict[str, Any],
        plan: GeneralTaskPlan,
        errors: tuple[str, ...],
    ) -> GeneralTaskPlan:
        ...


class GeneralTaskPlanValidator:
    """Validate a brain proposal against local capabilities and path policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        resolver: CapabilityResolver,
        boundary: FilesystemBoundary,
        *,
        max_steps: int = 8,
    ) -> None:
        self.registry = registry
        self.resolver = resolver
        self.boundary = boundary
        self.max_steps = max_steps

    def validate(
        self,
        plan: GeneralTaskPlan,
        *,
        authorized_scopes: tuple[str, ...],
        external_artifacts: tuple[str, ...] = (),
    ) -> GeneralTaskPlanValidation:
        errors: list[str] = []
        if not plan.goal:
            errors.append("goal_required")
        if plan.task_type not in _ALLOWED_TASK_TYPES:
            errors.append("task_type_not_allowed")
        if len(plan.steps) > self.max_steps:
            errors.append("step_limit_exceeded")
        if plan.risk not in {"none", "low", "medium", "high"}:
            errors.append("risk_invalid")
        if plan.task_type in {"MISSING_CAPABILITY", "UNSUPPORTED"}:
            if plan.steps or plan.needs_tools:
                errors.append("missing_capability_plan_must_not_execute")
            return GeneralTaskPlanValidation(
                valid=not errors,
                errors=tuple(dict.fromkeys(errors)),
                resolution_kind=CapabilityResolutionKind.MISSING_CAPABILITY.value,
                capabilities=tuple(dict.fromkeys(plan.required_capabilities)),
            )
        if plan.task_type == "FORBIDDEN":
            if plan.steps or plan.needs_tools:
                errors.append("forbidden_plan_must_not_execute")
            return GeneralTaskPlanValidation(
                valid=not errors,
                errors=tuple(dict.fromkeys(errors)),
                resolution_kind=CapabilityResolutionKind.FORBIDDEN.value,
                capabilities=tuple(dict.fromkeys(plan.required_capabilities)),
            )
        step_ids = [step.step_id for step in plan.steps]
        if any(not value for value in step_ids) or len(set(step_ids)) != len(step_ids):
            errors.append("step_ids_invalid")
        seen: set[str] = set()
        available_artifacts: set[str] = set(external_artifacts)
        if plan.task_type == 'MODIFY_CODE':
            available_artifacts.add('selected_artifact')
        for step in plan.steps:
            if any(item not in seen for item in step.depends_on):
                errors.append(f"dependency_not_preceding:{step.step_id}")
            seen.add(step.step_id)
            if not step.purpose:
                errors.append(f"step_purpose_required:{step.step_id}")
            if any(step.capability.startswith(prefix) for prefix in _FORBIDDEN_CAPABILITIES):
                errors.append(f"capability_forbidden:{step.capability}")
            if not self.registry.find_by_capability(step.capability):
                errors.append(f"capability_missing:{step.capability}")
            for artifact_id in step.consumes_artifacts:
                if artifact_id not in available_artifacts:
                    errors.append(
                        f"artifact_dependency_not_preceding:{step.step_id}:{artifact_id}"
                    )
            for artifact_id in step.produces_artifacts:
                if not artifact_id or artifact_id in available_artifacts:
                    errors.append(f"artifact_output_invalid:{step.step_id}:{artifact_id}")
                else:
                    available_artifacts.add(artifact_id)
            self._validate_paths(step, authorized_scopes, errors)
        declared = tuple(dict.fromkeys(plan.required_capabilities))
        used = tuple(dict.fromkeys(step.capability for step in plan.steps))
        if set(declared) != set(used):
            errors.append("required_capabilities_do_not_match_steps")
        if plan.needs_tools != bool(plan.steps):
            errors.append("needs_tools_steps_mismatch")
        resolution = self.resolver.resolve(CapabilityIntent(required_capabilities=declared))
        if plan.needs_tools and resolution.kind not in {
            CapabilityResolutionKind.EXACT_TOOL,
            CapabilityResolutionKind.COMPOSABLE,
        }:
            errors.append(f"capability_resolution:{resolution.kind.value}")
        if not plan.needs_tools and declared:
            errors.append("capabilities_declared_without_tools")
        if plan.task_type == "GENERATE_CODE":
            if not plan.needs_content_generation:
                errors.append("code_generation_flag_required")
            raw_code_spec = plan.inputs.get("code_spec")
            if not isinstance(raw_code_spec, dict):
                errors.append("code_generation_spec_required")
            else:
                specification = CodeGenerationSpec.from_dict(raw_code_spec)
                suffix = Path(specification.target_filename).suffix.casefold()
                expected_suffix = (
                    ".py" if specification.language == "python"
                    else ".cpp" if specification.language in {"cpp", "c++"}
                    else ""
                )
                if not specification.goal:
                    errors.append("code_generation_goal_required")
                if not specification.functional_requirements:
                    errors.append("code_functional_requirements_required")
                if not expected_suffix:
                    errors.append("code_generation_language_unsupported")
                elif suffix != expected_suffix:
                    errors.append("code_generation_filename_extension_mismatch")
                from .verification_types import VerificationRequirement, code_verification_requirement
                requirement = code_verification_requirement(specification)
                if requirement in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE} and not specification.execution_required:
                    errors.append("code_verification_requires_execution")
                if specification.execution_required and requirement in {VerificationRequirement.NONE,VerificationRequirement.BASIC}:
                    errors.append("code_execution_requirement_contradiction")
                if requirement == VerificationRequirement.TEST_SUITE and not specification.tests:
                    errors.append("code_generation_tests_required")
                if specification.dependencies:
                    errors.append("code_generation_dependencies_not_allowed")
            if "archive.compress_zip" in declared:
                archive_steps = [
                    step for step in plan.steps
                    if step.capability == "archive.compress_zip"
                ]
                if not archive_steps or not archive_steps[0].consumes_artifacts:
                    errors.append("archive_step_requires_code_artifact_binding")
        if plan.task_type == "MODIFY_CODE":
            raw_modification = plan.inputs.get("code_modification_spec")
            if not isinstance(raw_modification, dict):
                errors.append("code_modification_spec_required")
            else:
                if not str(raw_modification.get("requested_change", "")).strip():
                    errors.append("code_modification_change_required")
                if not str(raw_modification.get("language", "")).strip():
                    errors.append("code_modification_language_required")
            if "developer.read_code" not in declared:
                errors.append("code_modification_read_required")
            if "developer.write_code" not in declared:
                errors.append("code_modification_write_required")
        if plan.task_type == "GENERATE_CONVERSATIONAL_CONTENT" and plan.needs_tools:
            errors.append("conversation_content_must_not_require_artifact")
        return GeneralTaskPlanValidation(
            valid=not errors,
            errors=tuple(dict.fromkeys(errors)),
            resolution_kind=resolution.kind.value,
            capabilities=declared,
        )

    def _validate_paths(
        self,
        step: GeneralTaskStep,
        authorized_scopes: tuple[str, ...],
        errors: list[str],
    ) -> None:
        developer_only = step.capability.startswith("developer.")
        for key in ("path", "source", "destination", "root"):
            raw = step.arguments.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            if raw.startswith(("$steps.", "$artifacts.", "$values.")):
                continue
            operation = "READ" if key in {"source", "root"} else "WRITE"
            if step.capability.endswith(("read", "read_code", "run_python", "run_cpp")):
                operation = "READ"
            decision = self.boundary.classify(
                Path(raw),
                operation=operation,
                authorized_scopes=authorized_scopes,
                developer_only=developer_only,
            )
            if not decision.allowed:
                errors.append(f"path_denied:{step.step_id}:{decision.reason}")


def compact_capability_vocabulary(registry: ToolRegistry) -> dict[str, Any]:
    """Descriptions only. No executable objects or implementation paths leave runtime."""

    return {
        "source": "ToolRegistry",
        "tools": [
            {
                "tool_id": tool.manifest.tool_id,
                "name": tool.manifest.name,
                "description": tool.manifest.description,
                "capabilities": [cap for cap in tool.manifest.capabilities
                    if not registry.execution_denial(tool.manifest.tool_id, cap)],
                "contracts": [{"id": cap, "input_fields": list(contract["input_schema"].get("properties", {})),
                    "output_fields": list(contract["output_schema"].get("properties", {})),
                    "side_effects": contract["side_effects"], "verifier_type": contract["verifier_type"]}
                    for cap, contract in tool.manifest.capability_contracts.items()
                    if not registry.execution_denial(tool.manifest.tool_id, cap)],
                "risk_level": tool.manifest.risk_level,
                "status": tool.health()[0].value,
            }
            for tool in registry.list_tools(enabled_only=True)
        ],
        "generated_tools_enabled": False,
        "max_steps": 8,
    }
