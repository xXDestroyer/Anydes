from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Protocol

from anydes.cognition.contracts import (
    InternalContext,
    Intention,
    ReflectionResult,
)
from anydes.cognition.reasoning import (
    extract_reasoning_model_object,
    grounded_intention_from_model_dict,
)
from anydes.cognition.reflection import ReflectionLayer


class PlanRepairModel(Protocol):
    """Model boundary used only for one conditional grounded-plan repair."""

    def generate(self, prompt: str) -> str:
        """Return one compact grounded-plan JSON object."""


@dataclass(frozen=True)
class PlanRepairDifference:
    """One auditable change between the rejected and repaired plans."""

    field: str
    before: object
    after: object

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True)
class ConditionalPlanRepairResult:
    """Complete result of at most one conditional repair attempt."""

    attempted: bool
    accepted: bool
    original_intention: Intention
    original_reflection: ReflectionResult
    repaired_intention: Intention | None = None
    repaired_reflection: ReflectionResult | None = None
    boundary_issues: tuple[str, ...] = ()
    error_type: str = ""
    error_reason: str = ""
    model_output_keys: tuple[str, ...] = ()
    differences: tuple[PlanRepairDifference, ...] = ()

    @property
    def final_intention(self) -> Intention:
        return self.repaired_intention or self.original_intention

    @property
    def final_reflection(self) -> ReflectionResult:
        return self.repaired_reflection or self.original_reflection

    def to_dict(self) -> dict[str, object]:
        original_plan = self.original_intention.grounded_plan
        repaired_plan = (
            self.repaired_intention.grounded_plan
            if self.repaired_intention is not None
            else None
        )
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "error_type": self.error_type,
            "error_reason": self.error_reason,
            "model_output_keys": list(self.model_output_keys),
            "original_reflection": self.original_reflection.to_dict(),
            "repaired_reflection": (
                self.repaired_reflection.to_dict()
                if self.repaired_reflection is not None
                else None
            ),
            "boundary_issues": list(self.boundary_issues),
            "differences": [item.to_dict() for item in self.differences],
            "speech_text_preserved": (
                self.repaired_intention is None
                or self.repaired_intention.speech == self.original_intention.speech
            ),
            "original_plan": (
                original_plan.to_dict() if original_plan is not None else None
            ),
            "repaired_plan": (
                repaired_plan.to_dict() if repaired_plan is not None else None
            ),
        }


@dataclass
class ConditionalGroundedPlanRepairer:
    """Repairs one rejected plan without applying behavior or side effects."""

    model: PlanRepairModel
    reflection: ReflectionLayer

    def repair(
        self,
        context: InternalContext,
        intention: Intention,
        reflection: ReflectionResult,
    ) -> ConditionalPlanRepairResult:
        plan = intention.grounded_plan
        if (
            reflection.approved
            or reflection.decision != "revise"
            or not reflection.blocking_issues
            or plan is None
            or context.evidence_bundle is None
        ):
            return ConditionalPlanRepairResult(
                attempted=False,
                accepted=False,
                original_intention=intention,
                original_reflection=reflection,
            )

        model_output_keys: tuple[str, ...] = ()
        try:
            raw = self.model.generate(
                build_plan_repair_prompt(context, intention, reflection)
            )
            patch = extract_reasoning_model_object(raw)
            model_output_keys = tuple(sorted(str(key) for key in patch))
            data = {
                **_intention_to_repair_wire(intention),
                **_unwrap_repair_patch(patch),
            }
            data["speech"] = intention.speech
            repaired = grounded_intention_from_model_dict(data, context)
            repaired = _preserve_speech_manifestation(intention, repaired)
            repaired_reflection = self.reflection.reflect(repaired, context)
            boundary_issues = repair_boundary_issues(
                intention,
                repaired,
                context,
            )
            if boundary_issues:
                repaired_reflection = ReflectionResult(
                    decision="revise",
                    approved=False,
                    risk_level=repaired_reflection.risk_level,
                    needs_confirmation=False,
                    reason="Conditional repair crossed its allowed boundary.",
                    blocking_issues=tuple(
                        dict.fromkeys(
                            (
                                *repaired_reflection.blocking_issues,
                                *boundary_issues,
                            )
                        )
                    ),
                )
            accepted = repaired_reflection.approved and not boundary_issues
            return ConditionalPlanRepairResult(
                attempted=True,
                accepted=accepted,
                original_intention=intention,
                original_reflection=reflection,
                repaired_intention=repaired,
                repaired_reflection=repaired_reflection,
                boundary_issues=boundary_issues,
                differences=plan_differences(intention, repaired),
                model_output_keys=model_output_keys,
            )
        except Exception as exc:
            return ConditionalPlanRepairResult(
                attempted=True,
                accepted=False,
                original_intention=intention,
                original_reflection=reflection,
                error_type=type(exc).__name__,
                error_reason=str(exc)[:320],
                model_output_keys=model_output_keys,
            )


def build_plan_repair_prompt(
    context: InternalContext,
    intention: Intention,
    reflection: ReflectionResult,
) -> str:
    """Build the constrained repair prompt from only authorized turn state."""

    bundle = context.evidence_bundle
    plan = intention.grounded_plan
    if bundle is None or plan is None:
        raise ValueError("Plan repair requires an EvidenceBundle and rejected plan.")
    payload = {
        "evidence_bundle": bundle.to_model_dict(),
        "rejected_plan": plan.to_dict(),
        "reflection": {
            "decision": reflection.decision,
            "reason": reflection.reason,
            "blocking_issues": list(reflection.blocking_issues),
        },
        "invariants": [
            "Use only references and capabilities present in evidence_bundle.",
            "Do not add memory, relationship, self, curiosity or action side effects.",
            "Do not convert USER preferences or reports into SELF facts.",
            "A request proves what was requested, not that its desired result is true.",
            "Preserve the valid conversational intent and repair only the rejected commitments.",
            "Do not make safety checks less strict and do not mention internal architecture.",
            "Speech text is outside this experiment and will be preserved unchanged.",
        ],
        "output_contract": {
            "required": [
                "obligation",
                "mode",
                "goal",
                "topic",
                "attention_ids",
                "subject",
                "source",
                "status",
                "evidence_ids",
                "gap_ids",
                "claim_kind",
                "claim",
                "speak",
                "initiative",
                "engagement_evidence_ids",
                "confidence",
                "expression",
            ],
            "expression": [
                "stance",
                "warmth",
                "energy",
                "formality",
                "closeness",
                "intensity",
            ],
            "optional": [
                "action",
                "action_parameters",
                "deferred_intents",
                "state_proposals",
            ],
            "allowed_values": {
                "obligation": [
                    "answer",
                    "acknowledge",
                    "clarify",
                    "act",
                    "observe",
                    "listen",
                    "initiate",
                    "recover",
                ],
                "mode": [
                    "respond",
                    "ask",
                    "deepen",
                    "recall",
                    "shift_topic",
                    "clarify",
                    "share_observation",
                    "listen",
                ],
                "source": [
                    "user",
                    "self",
                    "memory",
                    "relationship",
                    "tool",
                    "observation",
                    "environment",
                    "general_knowledge",
                    "conversation",
                ],
                "status": [
                    "known",
                    "unknown",
                    "uncertain",
                    "unavailable",
                    "not_applicable",
                ],
                "claim_kind": [
                    "assertion",
                    "uncertainty",
                    "acknowledgement",
                    "observation",
                ],
            },
        },
    }
    return "\n".join(
        [
            "Repara una sola vez el GroundedResponsePlan rechazado.",
            "No vuelvas a responder el turno desde cero.",
            "Corrige unicamente los invariantes indicados por Reflection.",
            "No agregues evidencia, capacidades, efectos persistentes ni hechos sin respaldo.",
            "DATOS AUTORIZADOS Y CONTRATOS:",
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "FIN DE DATOS AUTORIZADOS.",
            "Devuelve exclusivamente el parche del plan como un objeto JSON compacto.",
            "Usa solo claves declaradas en output_contract.",
            "Nunca devuelvas evidence_bundle, rejected_plan, reflection, invariants ni output_contract.",
            "No incluyas explicaciones, markdown ni una copia de los datos de entrada.",
        ]
    )


def repair_boundary_issues(
    original: Intention,
    repaired: Intention,
    context: InternalContext,
) -> tuple[str, ...]:
    """Reject repaired output that expands evidence or behavioral authority."""

    issues: list[str] = []
    bundle = context.evidence_bundle
    original_plan = original.grounded_plan
    repaired_plan = repaired.grounded_plan
    if bundle is None or original_plan is None or repaired_plan is None:
        return ("repair_boundary_missing_grounded_state",)

    unknown = repaired_plan.referenced_evidence_ids - bundle.reference_ids
    for reference_id in sorted(unknown):
        issues.append(f"repair_added_unknown_reference:{reference_id}")
    unauthorized = {
        item.id for item in bundle.items if not item.authorized
    }
    for reference_id in sorted(
        repaired_plan.referenced_evidence_ids.intersection(unauthorized)
    ):
        issues.append(f"repair_added_unauthorized_reference:{reference_id}")

    original_action_types = {item.action_type for item in original.actions}
    for action in repaired.actions:
        if action.action_type not in original_action_types:
            issues.append(f"repair_added_action:{action.action_type}")

    if (
        repaired_plan.action.capability_id
        and repaired_plan.action.capability_id
        != original_plan.action.capability_id
    ):
        issues.append(
            f"repair_changed_capability:{repaired_plan.action.capability_id}"
        )

    _append_added_ids(
        issues,
        "memory",
        {item.id for item in original.memory_updates},
        {item.id for item in repaired.memory_updates},
    )
    _append_added_ids(
        issues,
        "relationship",
        {item.id for item in original.relationship_updates},
        {item.id for item in repaired.relationship_updates},
    )
    _append_added_ids(
        issues,
        "self",
        {item.id for item in original.self_updates},
        {item.id for item in repaired.self_updates},
    )
    _append_added_ids(
        issues,
        "curiosity",
        {item.id for item in original.curiosity_updates},
        {item.id for item in repaired.curiosity_updates},
    )
    if len(repaired_plan.claims) > len(original_plan.claims):
        issues.append("repair_added_claim")
    return tuple(dict.fromkeys(issues))


def plan_differences(
    original: Intention,
    repaired: Intention,
) -> tuple[PlanRepairDifference, ...]:
    original_plan = original.grounded_plan
    repaired_plan = repaired.grounded_plan
    if original_plan is None or repaired_plan is None:
        return ()
    before = original_plan.to_dict()
    after = repaired_plan.to_dict()
    fields = (
        "primary_obligation",
        "response_goal",
        "attention",
        "knowledge",
        "claims",
        "engagement",
        "action",
        "deferred_intents",
        "expression",
        "output_manifest",
    )
    return tuple(
        PlanRepairDifference(field=field, before=before.get(field), after=after.get(field))
        for field in fields
        if before.get(field) != after.get(field)
    )


def _preserve_speech_manifestation(
    original: Intention,
    repaired: Intention,
) -> Intention:
    plan = repaired.grounded_plan
    if plan is None:
        return repaired
    required_claim_ids = tuple(item.id for item in plan.claims if item.required)
    speech = replace(
        original.speech_intention,
        text=original.speech,
        plan_id=plan.id,
        realized_claim_ids=required_claim_ids,
    )
    body = original.body_intention
    if body is not None:
        body = replace(body, plan_id=plan.id)
    return replace(
        repaired,
        speech=original.speech,
        speech_intention=speech,
        body_intention=body,
    )


def _intention_to_repair_wire(intention: Intention) -> dict[str, object]:
    plan = intention.grounded_plan
    if plan is None:
        raise ValueError("Repair wire projection requires a grounded plan.")
    claim = plan.claims[0] if plan.claims else None
    action = intention.actions[0] if intention.actions else None
    return {
        "obligation": plan.primary_obligation,
        "mode": intention.conversation_intent.mode,
        "goal": plan.response_goal,
        "topic": plan.attention.topic,
        "attention_ids": list(plan.attention.target_ids),
        "subject": plan.knowledge.subject,
        "source": plan.knowledge.source,
        "status": plan.knowledge.epistemic_status,
        "evidence_ids": list(plan.knowledge.evidence_ids),
        "gap_ids": list(plan.knowledge.knowledge_gap_ids),
        "boundaries": list(plan.knowledge.boundaries),
        "claim_kind": claim.predicate if claim is not None else "assertion",
        "claim": claim.content if claim is not None else "",
        "speak": plan.engagement.should_speak,
        "initiative": plan.engagement.initiative,
        "engagement_evidence_ids": list(plan.engagement.evidence_ids),
        "confidence": plan.confidence,
        "expression": {
            "stance": plan.expression.stance,
            "warmth": plan.expression.warmth,
            "energy": plan.expression.energy,
            "formality": plan.expression.formality,
            "closeness": plan.expression.closeness,
            "intensity": plan.expression.intensity,
            "gaze": plan.expression.attention_target,
            "evidence_ids": list(plan.expression.evidence_ids),
        },
        "action": action.action_type if action is not None else "none",
        "action_parameters": dict(action.parameters) if action is not None else {},
        "deferred_intents": [
            item.to_dict() for item in plan.deferred_intents
        ],
        "state_proposals": {
            "memory_updates": [
                item.to_dict() for item in intention.memory_updates
            ],
            "relationship_updates": [
                item.to_dict() for item in intention.relationship_updates
            ],
            "self_updates": [
                item.to_dict() for item in intention.self_updates
            ],
            "curiosity_updates": [
                item.to_dict() for item in intention.curiosity_updates
            ],
        },
    }


def _unwrap_repair_patch(
    payload: dict[str, object],
) -> dict[str, object]:
    """Normalize structural wrappers without changing any repair value."""

    for key in ("repaired_plan", "grounded_plan", "repair"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return _unwrap_repair_patch(dict(nested))
    if (
        isinstance(payload.get("attention"), dict)
        and isinstance(payload.get("knowledge"), dict)
        and isinstance(payload.get("claims"), list)
    ):
        return _grounded_plan_to_repair_wire(payload)
    decision = payload.get("decision")
    if isinstance(decision, dict):
        outer = {
            key: value for key, value in payload.items() if key != "decision"
        }
        return _unwrap_repair_patch({**outer, **decision})
    compact = dict(payload)
    mode = str(compact.get("mode") or "").strip().casefold()
    if mode in {"speak", "silence"}:
        compact.pop("mode", None)
        compact.setdefault("speak", mode == "speak")
    return compact


def _grounded_plan_to_repair_wire(
    plan: dict[str, object],
) -> dict[str, object]:
    attention = (
        plan["attention"]
        if isinstance(plan.get("attention"), dict)
        else {}
    )
    knowledge = (
        plan["knowledge"]
        if isinstance(plan.get("knowledge"), dict)
        else {}
    )
    engagement = (
        plan["engagement"]
        if isinstance(plan.get("engagement"), dict)
        else {}
    )
    expression = (
        plan["expression"]
        if isinstance(plan.get("expression"), dict)
        else {}
    )
    action = (
        plan["action"]
        if isinstance(plan.get("action"), dict)
        else {}
    )
    claims = plan.get("claims")
    claim = (
        claims[0]
        if isinstance(claims, list)
        and claims
        and isinstance(claims[0], dict)
        else {}
    )
    action_type = str(action.get("action_type") or "")
    action_phase = str(action.get("phase") or "none")
    return {
        "obligation": plan.get("primary_obligation"),
        "goal": plan.get("response_goal"),
        "topic": attention.get("topic"),
        "attention_ids": attention.get("target_ids"),
        "subject": knowledge.get("subject"),
        "source": knowledge.get("source"),
        "status": knowledge.get("epistemic_status"),
        "evidence_ids": knowledge.get("evidence_ids"),
        "gap_ids": knowledge.get("knowledge_gap_ids"),
        "boundaries": knowledge.get("boundaries"),
        "claim_kind": claim.get("predicate"),
        "claim": claim.get("content"),
        "speak": engagement.get("should_speak"),
        "initiative": engagement.get("initiative"),
        "engagement_evidence_ids": engagement.get("evidence_ids"),
        "confidence": plan.get("confidence"),
        "expression": {
            "stance": expression.get("stance"),
            "warmth": expression.get("warmth"),
            "energy": expression.get("energy"),
            "formality": expression.get("formality"),
            "closeness": expression.get("closeness"),
            "intensity": expression.get("intensity"),
            "gaze": expression.get("attention_target"),
            "evidence_ids": expression.get("evidence_ids"),
        },
        "action": (
            action_type
            if action_type and action_phase != "none"
            else "none"
        ),
        "deferred_intents": plan.get("deferred_intents"),
    }


def _append_added_ids(
    issues: list[str],
    label: str,
    original_ids: set[str],
    repaired_ids: set[str],
) -> None:
    for item_id in sorted(repaired_ids - original_ids):
        issues.append(f"repair_added_{label}_proposal:{item_id}")
