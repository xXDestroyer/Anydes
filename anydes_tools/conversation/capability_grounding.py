"""Exact Registry authority and explicit, context-restricted aliases only."""
from dataclasses import dataclass, replace
from .plan_completeness import output_type


@dataclass(frozen=True)
class CapabilityAlias:
    target: str
    step_type: str = "CAPABILITY"


# No production aliases are assumed: a renderer is not a conversion primitive.
CAPABILITY_ALIAS_MAP = {}


class CapabilityGroundingValidator:
    def __init__(self, registry, aliases=None):
        self.registry = registry
        self.aliases = CAPABILITY_ALIAS_MAP if aliases is None else dict(aliases)

    def ground(self, plan):
        errors = []
        steps = []
        types = {}
        normalized = {}
        for step in plan.steps:
            cap = step.capability
            alias = self.aliases.get(cap)
            if alias:
                if step.step_type != alias.step_type:
                    errors.append("ALIAS_CONTEXT_MISMATCH:" + step.step_id)
                    continue
                normalized[cap] = cap = alias.target
            all_tools = [t for t in self.registry.list_tools() if cap in t.manifest.capabilities]
            if not all_tools:
                errors.append("UNKNOWN_CAPABILITY:" + step.step_id + ":" + cap)
            elif self.registry.execution_denial(all_tools[0].manifest.tool_id, cap):
                errors.append(self.registry.execution_denial(all_tools[0].manifest.tool_id, cap) + ":" + step.step_id)
            if step.step_type == "CONVERSION" and (
                    not cap.startswith("document.create_") or len(step.consumes_artifacts) != 1):
                errors.append("CAPABILITY_STEP_TYPE_MISMATCH:" + step.step_id)
            if step.step_type not in {"CAPABILITY", "CONVERSION"}:
                errors.append("CAPABILITY_STEP_TYPE_MISMATCH:" + step.step_id)
            for binding in step.consumes_artifacts:
                kind = types.get(binding)
                if kind and cap.startswith("developer.") and kind != "CODE":
                    errors.append("CAPABILITY_INPUT_TYPE_MISMATCH:" + step.step_id)
                if kind and cap.startswith("document.create_") and kind != "DOCUMENT":
                    errors.append("CAPABILITY_INPUT_TYPE_MISMATCH:" + step.step_id)
            kind, _ = output_type(cap)
            for binding in step.produces_artifacts:
                types[binding] = kind
            steps.append(replace(step, capability=cap))
        declared = tuple(normalized.get(cap, cap) for cap in plan.required_capabilities)
        for cap in declared:
            if not self.registry.find_by_capability(cap):
                errors.append("DECLARED_CAPABILITY_UNAVAILABLE:" + cap)
        if errors:
            raise ValueError("CAPABILITY_GROUNDING_FAILED:" + ";".join(dict.fromkeys(errors)))
        return replace(plan, steps=tuple(steps), required_capabilities=declared)
