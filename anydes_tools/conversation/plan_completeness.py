"""Local pre-execution proof of declared deliverables and dependency bindings."""
from dataclasses import dataclass

from .dataflow import DataflowError, validate_dataflow


@dataclass(frozen=True)
class ArtifactBinding:
    logical_name: str
    producer_step_id: str
    artifact_type: str
    format: str
    required_state: str
    consumer_step_ids: tuple[str, ...]


class PlanIncomplete(ValueError):
    def __init__(self, errors):
        self.errors = tuple(dict.fromkeys(errors))
        super().__init__("PLAN_INCOMPLETE:" + ";".join(self.errors))


def output_type(capability):
    if capability == "developer.write_code":
        return "CODE", ""
    if capability == "archive.compress_zip":
        return "ARCHIVE", "zip"
    if capability.startswith("document.create_"):
        return "DOCUMENT", {"document.create_markdown": "md"}.get(
            capability, capability.removeprefix("document.create_"))
    if capability == "document.edit_document":
        return "DOCUMENT", ""
    return "", ""


class PlanCompletenessValidator:
    def validate(self, plan):
        errors = []
        from .code_generation import CodeGenerationSpec
        from .verification_types import VerificationRequirement, code_verification_requirement
        raw_spec = plan.legacy_action.arguments.get("code_generation_spec")
        requirement = VerificationRequirement(plan.typed_inputs.get("verification_requirement", "NONE"))
        if raw_spec:
            spec = CodeGenerationSpec.from_dict(raw_spec)
            spec_requirement = code_verification_requirement(spec)
            if requirement == VerificationRequirement.NONE:
                requirement = spec_requirement
            if requirement in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE} and not spec.execution_required:
                errors.append("verification_execution_required")
        if requirement in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE}:
            if not any(s.verifier_type == "CODE" and s.capability_id in {
                    "developer.run_python", "developer.run_cpp", "developer.test"} for s in plan.steps):
                errors.append("required_code_verifier_missing")
            if requirement == VerificationRequirement.TEST_SUITE and not (raw_spec or {}).get("tests"):
                errors.append("required_test_suite_missing")
        try:
            validate_dataflow(plan)
        except DataflowError as exc:
            errors.append(str(exc))
        producers = {}
        inputs = plan.typed_inputs.get("artifact_bindings", {})
        for name, ref in inputs.items():
            producers[name] = ("INPUT", ref.artifact_type, ref.format)
        generated = bool(plan.typed_inputs.get("brain_proposal"))
        for step in plan.steps:
            kind, fmt = output_type(step.capability_id)
            if kind == "CODE":
                from pathlib import Path
                spec = plan.legacy_action.arguments.get("code_generation_spec", {})
                filename = step.args.get("path") or spec.get("target_filename", "")
                fmt = Path(filename).suffix.lower().lstrip(".")
            if kind and not step.produces_artifacts:
                errors.append("producer_output_required:" + step.step_id)
            for name in step.produces_artifacts:
                if name in producers:
                    errors.append("ambiguous_binding:" + name)
                producers[name] = (step.step_id, kind, fmt)
            if step.capability_id == "archive.compress_zip":
                # Explicit directory-only commands are legacy input selections, not
                # permission to substitute a folder for a concrete artifact task.
                if not step.consumes_artifacts and not plan.typed_inputs.get("explicit_directory_sources"):
                    errors.append("archive_members_binding_required:" + step.step_id)
                if step.args.get("sources") == [] and not step.consumes_artifacts:
                    errors.append("archive_members_empty:" + step.step_id)
            if generated and (step.capability_id.startswith("developer.run_")
                    or step.capability_id == "developer.test") and not step.consumes_artifacts:
                errors.append("code_execution_binding_required:" + step.step_id)
            if step.capability_id.startswith("document.create_") and (
                    step.args.get("source_artifact_id") or step.args.get("source_path")
                    or step.args.get("conversion_spec") or step.step_type == "CONVERSION"):
                if len(step.consumes_artifacts) != 1:
                    errors.append("conversion_source_binding_required:" + step.step_id)
            if len(step.consumes_artifacts) > 1 and step.capability_id.startswith("document.create_"):
                errors.append("conversion_source_ambiguous:" + step.step_id)
            if generated and step.capability_id.startswith("document.create_") and not (
                    step.consumes_artifacts or step.consumes_values or step.args.get("content")):
                errors.append("document_content_or_source_binding_required:" + step.step_id)
            for name in step.consumes_artifacts:
                if name not in producers:
                    errors.append("consumer_binding_required:" + step.step_id + ":" + name)
                elif step.capability_id.startswith("document.create_") and producers[name][1] != "DOCUMENT":
                    errors.append("conversion_type_mismatch:" + step.step_id + ":" + name)
            if generated:
                for key in ("path", "destination"):
                    if kind and not step.args.get(key) and key == (
                            "destination" if kind == "ARCHIVE" else "path"):
                        # Generated code/document macros derive one destination from
                        # their validated specification; generic composition cannot.
                        if plan.legacy_action.operation == "general_tool_composition":
                            errors.append("destination_required:" + step.step_id)
        declared = {}
        for row in plan.declared_outputs:
            name = row.get("binding") or row.get("artifact_id") or row.get("logical_name")
            if not name or name not in producers or producers[name][0] == "INPUT":
                errors.append("declared_output_producer_required:" + str(name or "missing"))
                continue
            if name in declared:
                errors.append("duplicate_declared_output:" + name)
            declared[name] = row
            kind, fmt = producers[name][1:]
            if row.get("artifact_type") and row["artifact_type"].upper() != kind:
                errors.append("declared_output_type_mismatch:" + name)
            if row.get("format") and fmt and row["format"].lower().lstrip(".") != fmt:
                errors.append("declared_output_format_mismatch:" + name)
        required = plan.requested_final_outputs
        if any(output_type(s.capability_id)[0] for s in plan.steps) and not declared:
            errors.append("final_output_declaration_required")
        for requirement in required:
            matching = [name for name, row in declared.items()
                if (not requirement.get("binding") or requirement["binding"] == name)
                and (not requirement.get("format") or producers[name][2] == requirement["format"])
                and (not requirement.get("artifact_type") or producers[name][1] == requirement["artifact_type"])]
            if not matching:
                errors.append("requested_final_output_missing:" + str(requirement.get("binding") or requirement))
        if errors:
            raise PlanIncomplete(errors)
        bindings = []
        for name, (producer, kind, fmt) in producers.items():
            consumers = tuple(s.step_id for s in plan.steps if name in s.consumes_artifacts)
            verified = any(s.requires_verified_inputs for s in plan.steps if s.step_id in consumers)
            bindings.append(ArtifactBinding(name, producer, kind, fmt,
                "VERIFIED" if verified else "TEMPORARY", consumers))
        return tuple(bindings)
