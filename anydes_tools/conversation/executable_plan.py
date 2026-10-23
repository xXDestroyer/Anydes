"""Phase C local execution contract; legacy operations are explicit adapters.

Artifact declarations are validated here and resolved by the local dataflow host.
"""
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from copy import deepcopy

from .contracts import PlannedToolAction


@dataclass(frozen=True)
class BrainPlanProposal:
    proposal: Any


@dataclass(frozen=True)
class ExecutableStep:
    step_id: str
    step_type: str
    capability_id: str
    args: dict
    consumes_artifacts: tuple[str, ...] = ()
    produces_artifacts: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    verifier_type: str = ""
    risk_class: str = ""
    depends_on: tuple[str, ...] = ()
    requires_verified_inputs: bool = False
    consumes_values: tuple[str, ...] = ()
    produces_values: tuple[str, ...] = ()
    operation_descriptor: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutableTaskPlan:
    schema_version: str
    task_id: str
    revision: int
    goal: str
    typed_inputs: dict
    steps: tuple[ExecutableStep, ...]
    declared_outputs: tuple[dict, ...]
    success_criteria: tuple[str, ...]
    execution_budget: dict
    source_request: str
    plan_source: str
    legacy_action: PlannedToolAction = field(repr=False, compare=False)
    requested_final_outputs: tuple[dict, ...] = ()
    artifact_bindings: tuple = ()


@dataclass(frozen=True)
class StepResult:
    step_id: str
    status: str
    artifact_ids: tuple[str, ...] = ()
    values: dict = field(default_factory=dict)
    error: str = ""
    verification: dict = field(default_factory=dict)
    produced_artifacts: tuple[str, ...] = ()
    consumed_artifacts: tuple[str, ...] = ()
    typed_values: dict = field(default_factory=dict)
    observed_effects: tuple[dict, ...] = ()
    blocked_by_step_id: str = ""
    blocked_artifact_id: str = ""


class ExecutablePlanAdapter:
    """Local compiler. The brain cannot supply risk, enabled state or a verifier."""
    def __init__(self, registry, path_authorizer=None, artifact_lookup=None, requested_effects_provider=None):
        self.registry = registry
        self.path_authorizer = path_authorizer
        self.artifact_lookup = artifact_lookup
        self.requested_effects_provider = requested_effects_provider

    def compile(self, action, *, source, task_id="", revision=1):
        action = deepcopy(action)
        task_id = task_id or (action.task_spec.task_id if action.task_spec else f"task:{uuid4().hex}")
        requested = tuple(self.requested_effects_provider() or ()) if self.requested_effects_provider else ()
        required = {cap for cap in requested if self.registry.find_by_capability(cap)}
        supplied = set(action.capabilities)
        from .semantic_proposal import catalog
        effect_contracts = catalog().get('effect_contracts', {})
        # Effect coverage is not execution authority: actual requests still pass B1.
        for candidate in required - supplied:
            contract = effect_contracts.get(candidate)
            if contract and any(effect_contracts.get(cap) == contract for cap in action.capabilities):
                supplied.add(candidate)
        if supplied.intersection({"developer.run_python", "developer.run_cpp"}):
            supplied.add("developer.test")
        if required - supplied:
            raise ValueError("PLAN_INCOMPLETE:DATAFLOW_LEGACY_INCOMPLETE_EFFECTS:" + ",".join(sorted(required - supplied)))
        operations = {
            "document_conversion", "structural_document_edit", "rename_existing", "copy_existing",
            "text_files_to_pdf", "append_document", "verified_create_document", "generated_create_document",
            "general_generated_document", "generated_document_edit", "structured_edit_text",
            "document_transform", "delete_collection", "delete_collection_request", "run_inline_python",
            "general_generate_code", "general_modify_code", "general_tool_composition", "retain_unverified", "",
        }
        if action.operation not in operations:
            raise ValueError("executable_operation_not_registered")
        from .planner import validate_structured_action_plan
        errors = validate_structured_action_plan(action, self.registry)
        if errors:
            raise ValueError("executable_requirements_invalid:" + ",".join(errors))
        raw = action.arguments.get("general_task_plan", {})
        declarations = raw.get("steps", ())
        rows = declarations or [{"step_id": "operation", "capability": cap}
                                for cap in action.capabilities]
        if not rows or len(rows) > 8:
            raise ValueError("executable_step_budget_invalid")
        steps = []
        for index, row in enumerate(rows):
            cap = row["capability"]
            tools = self.registry.find_by_capability(cap)
            if not tools:
                raise ValueError("DENIED_CAPABILITY_UNAVAILABLE:" + cap)
            contract = tools[0].manifest.capability_contracts[cap]
            args = dict(row.get("arguments", action.arguments))
            if not action.operation:
                from ..schema_validation import validate
                if not validate(args, contract["input_schema"]):
                    raise ValueError("executable_arguments_invalid:" + cap)
            steps.append(ExecutableStep(
                step_id=row.get("step_id", "") or f"step_{index + 1}",
                step_type="LEGACY_OPERATION_ADAPTER" if action.operation else "CAPABILITY",
                capability_id=cap,
                args=args,
                consumes_artifacts=tuple(row.get("consumes_artifacts", ())),
                produces_artifacts=tuple(row.get("produces_artifacts", ())),
                expected_effects=tuple(contract["side_effects"]),
                verifier_type=contract["verifier_type"],
                risk_class=contract["risk_level"],
                depends_on=tuple(row.get("depends_on", ())),
                requires_verified_inputs=cap == "archive.compress_zip" or (
                    cap.startswith("document.create_") and bool(row.get("consumes_artifacts"))),
                consumes_values=tuple(row.get("consumes_values", ())),
                produces_values=tuple(row.get("produces_values", ())),
            ))
        # Legacy multi-capability macros without named steps still need unique IDs.
        if not declarations:
            from dataclasses import replace
            steps = [replace(s, step_id=f"step_{i + 1}") for i, s in enumerate(steps)]
        # Only adapters with one unambiguous file producer may fill old declarations.
        if not declarations and action.operation in {"general_generate_code", "general_modify_code"}:
            from dataclasses import replace
            writers = [s for s in steps if s.capability_id == "developer.write_code"]
            if len(writers) != 1:
                raise ValueError("DATAFLOW_CODE_PRODUCER_AMBIGUOUS")
            writer = writers[0]
            binding = writer.produces_artifacts[0] if len(writer.produces_artifacts) == 1 else "code.main"
            if len(writer.produces_artifacts) > 1:
                raise ValueError("DATAFLOW_CODE_PRODUCER_AMBIGUOUS")
            steps = [replace(s, produces_artifacts=(binding,)) if s is writer else
                replace(s, consumes_artifacts=s.consumes_artifacts or (binding,))
                if s.capability_id.startswith(("developer.run", "developer.test", "archive.compress"))
                else s for s in steps]
        for index, step in enumerate(steps):
            if (step.capability_id.startswith("document.create") or step.capability_id == "document.edit_document"
                    or step.capability_id == "archive.compress_zip") and not step.produces_artifacts and not declarations:
                from dataclasses import replace
                steps[index] = replace(step, produces_artifacts=(f"output.{step.step_id}",))
        if action.operation in {"generated_create_document", "general_generated_document"}:
            from dataclasses import replace
            position = next((i for i, s in enumerate(steps) if s.capability_id.startswith("document.create")), None)
            if position is None:
                raise ValueError("DATAFLOW_DOCUMENT_PRODUCER_MISSING")
            writer = steps[position]
            values = writer.consumes_values or ("content.main",)
            steps[position] = replace(writer, consumes_values=values)
            steps.insert(position, ExecutableStep("content_generation", "CONTENT_GENERATION", "", {},
                produces_values=values))
        if len({s.step_id for s in steps}) != len(steps):
            raise ValueError("executable_step_ids_invalid")
        if len(steps) > 8:
            raise ValueError("executable_step_budget_invalid")
        typed_inputs = {"requirements": action.task_spec, "brain_proposal": bool(declarations)}
        if action.resolved_artifact_bindings:
            typed_inputs['artifact_bindings'] = dict(action.resolved_artifact_bindings)
        from .operation_semantics import canonical_descriptor
        descriptor = canonical_descriptor((action.arguments.get("code_generation_spec") or
            action.arguments.get('code_modification_spec') or {}).get("operation_descriptor", {}))
        if descriptor:
            from dataclasses import replace
            typed_inputs["operation_descriptor"] = descriptor
            steps = [replace(s, operation_descriptor=descriptor) if s.verifier_type == "CODE" else s for s in steps]
        source_paths = []
        if "archive.compress_zip" in action.capabilities and not declarations:
            from pathlib import Path
            from .planner import _extract_path
            sources = action.arguments.get("sources", ())
            directory_sources = [str(Path(p).resolve(strict=False)) for p in sources if Path(p).is_dir()]
            explicit = _extract_path(action.source_text)
            if directory_sources:
                if len(directory_sources) != 1 or not explicit or str(Path(explicit).resolve(strict=False)) != directory_sources[0]:
                    raise ValueError("DATAFLOW_EXPLICIT_DIRECTORY_REQUIRED")
                typed_inputs["explicit_directory_sources"] = tuple(directory_sources)
            source_paths.extend(p for p in sources if Path(p).is_file())
        if action.operation in {"document_conversion", "general_modify_code", "structural_document_edit",
                "structured_edit_text", "document_transform", "generated_document_edit", "append_document"}:
            source_paths.append(action.arguments.get("path") or action.arguments.get("conversion_spec", {}).get("source_path"))
        if action.arguments.get("source_path"):
            source_paths.append(action.arguments["source_path"])
        if source_paths and self.path_authorizer:
            from pathlib import Path
            from .dataflow import ArtifactBindings, ArtifactState
            inputs = ArtifactBindings(task_id or (action.task_spec.task_id if action.task_spec else ""))
            for index, path in enumerate(source_paths):
                if path and self.path_authorizer(path) and Path(path).is_file():
                    selected = tuple(ref for ref in action.resolved_artifacts if ref.path == str(Path(path).resolve(strict=False)))
                    existing = selected[0] if len(selected) == 1 else self.artifact_lookup(path) if self.artifact_lookup else None
                    if action.resolved_artifacts and len(selected) != 1:
                        raise ValueError("DATAFLOW_REFERENCE_BINDING_MISMATCH")
                    key = next((name for name, ref in action.resolved_artifact_bindings.items() if ref.path == str(Path(path).resolve(strict=False))), f'input.{index}')
                    if existing is not None:
                        from dataclasses import replace
                        inputs.bindings[key] = replace(existing, producer_step_id="INPUT")
                    else:
                        inputs.produce(key, path, "INPUT", state=ArtifactState.COMMITTED)
            typed_inputs["artifact_bindings"] = inputs.bindings
            if "archive.compress_zip" in action.capabilities and not declarations:
                from dataclasses import replace
                steps = [replace(s, consumes_artifacts=tuple(inputs.bindings))
                    if s.capability_id == "archive.compress_zip" else s for s in steps]
            if len(inputs.bindings) == 1:
                from dataclasses import replace
                binding = next(iter(inputs.bindings))
                steps = [replace(s, consumes_artifacts=(binding,))
                    if not s.consumes_artifacts and (
                        s.capability_id in {"document.read_document", "developer.read_code"}
                        or action.operation == "document_conversion" and s.capability_id.startswith("document.create"))
                    else s for s in steps]
        from .plan_completeness import output_type
        declared_outputs = tuple(raw.get("desired_outputs", ()))
        if not declarations:
            declared_outputs = tuple({"binding": b, "artifact_type": output_type(s.capability_id)[0],
                "format": output_type(s.capability_id)[1]} for s in steps for b in s.produces_artifacts)
        requested_outputs = tuple(raw.get("requested_final_outputs", ())) or declared_outputs
        inferred = tuple({"artifact_type": output_type(cap)[0], "format": output_type(cap)[1]}
            for cap in required if output_type(cap)[0])
        inferred += tuple({k: effect_contracts[cap][k] for k in ('artifact_type', 'format')}
            for cap in required if not output_type(cap)[0] and cap in effect_contracts)
        requested_outputs = tuple((*requested_outputs, *inferred))
        plan = ExecutableTaskPlan(
            schema_version="C.1", task_id=task_id or (
                action.task_spec.task_id if action.task_spec else f"task:{uuid4().hex}"
            ), revision=revision, goal=action.intent, typed_inputs=typed_inputs,
            steps=tuple(steps), declared_outputs=declared_outputs,
            success_criteria=tuple(raw.get("success_criteria", ())),
            execution_budget={"max_steps": 8, "max_retries": 1},
            source_request=action.source_text, plan_source=source, legacy_action=action,
            requested_final_outputs=requested_outputs,
        )
        from .dataflow import validate_dataflow
        validate_dataflow(plan)
        from dataclasses import replace
        from .plan_completeness import PlanCompletenessValidator
        return replace(plan, artifact_bindings=PlanCompletenessValidator().validate(plan))

    def revalidate(self, plan):
        if not isinstance(plan, ExecutableTaskPlan) or plan.schema_version != "C.1":
            raise ValueError("executable_contract_required")
        for step in plan.steps:
            if step.capability_id and not self.registry.find_by_capability(step.capability_id):
                raise ValueError("DENIED_CAPABILITY_UNAVAILABLE:" + step.capability_id)
        rebuilt = self.compile(plan.legacy_action, source=plan.plan_source,
            task_id=plan.task_id, revision=plan.revision)
        if rebuilt.steps != plan.steps:
            raise ValueError("executable_plan_changed")
        from .plan_completeness import PlanCompletenessValidator
        if PlanCompletenessValidator().validate(plan) != plan.artifact_bindings:
            raise ValueError("PLAN_INCOMPLETE:bindings_changed")
        from .dataflow import ArtifactBindings
        inputs = ArtifactBindings(plan.task_id, plan.typed_inputs.get("artifact_bindings", {}))
        for binding in inputs.bindings:
            inputs.require(binding, verified=True, allow_unverified=all(
                step.capability_id in {'archive.compress_zip','developer.read_code','developer.write_code','developer.run_python',
                    'developer.run_cpp','developer.test','filesystem.read','filesystem.inspect'}
                for step in plan.steps if binding in step.consumes_artifacts))
        # Actual expanded ToolRequests still pass B1 schemas/scopes/permissions.
        # Macro arguments are not falsely validated as primitive capability args.
        return plan
