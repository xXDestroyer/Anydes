from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from anydes.cognition.contracts import (
    ActionRequest,
    ConversationIntent,
    InternalContext,
    Intention,
    SpeechIntention,
)
from anydes.cognition.grounding_contracts import (
    ActionCommitment,
    AttentionCommitment,
    BodyIntention,
    EngagementCommitment,
    EvidenceBundle,
    ExpressionCommitment,
    GroundedClaim,
    GroundedResponsePlan,
    KnowledgeCommitment,
    OutputManifest,
)
from anydes.cognition.provenance_resolver import (
    ProvenanceResolution,
    resolve_grounded_wire_provenance,
)
from anydes.cognition.reasoning import extract_reasoning_model_object
from anydes.primitives import JsonValue, clamp01


EpistemicBasis = Literal[
    "evidence",
    "general_knowledge",
    "unknown",
    "not_applicable",
]


class MinimalGroundedModel(Protocol):
    def generate_for_context(self, context: InternalContext, prompt: str) -> str:
        """Return one MinimalGroundedPlan wire object."""


class MinimalPlanCompileError(ValueError):
    """A minimal plan cannot be compiled without guessing or weakening safety."""


@dataclass(frozen=True)
class MinimalActionChoice:
    action_type: str = "none"
    parameters: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            {str(key): _json_value(value) for key, value in self.parameters.items()},
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class MinimalGroundedPlan:
    """Only semantic choices that cannot be reconstructed from contracts."""

    id: str
    evidence_bundle_id: str
    primary_obligation: str
    attention_ids: tuple[str, ...]
    epistemic_status: str
    epistemic_basis: EpistemicBasis
    evidence_ids: tuple[str, ...]
    knowledge_gap_ids: tuple[str, ...]
    response_commitment: str
    should_speak: bool
    action: MinimalActionChoice = field(default_factory=MinimalActionChoice)
    provenance: tuple[ProvenanceResolution, ...] = ()

    def __post_init__(self) -> None:
        if self.primary_obligation not in {
            "answer",
            "acknowledge",
            "clarify",
            "act",
            "observe",
            "listen",
            "initiate",
            "recover",
        }:
            raise MinimalPlanCompileError(
                f"unsupported_primary_obligation:{self.primary_obligation}"
            )
        if self.epistemic_status not in {
            "known",
            "unknown",
            "uncertain",
            "unavailable",
            "not_applicable",
        }:
            raise MinimalPlanCompileError(
                f"unsupported_epistemic_status:{self.epistemic_status}"
            )
        if self.epistemic_basis not in {
            "evidence",
            "general_knowledge",
            "unknown",
            "not_applicable",
        }:
            raise MinimalPlanCompileError(
                f"unsupported_epistemic_basis:{self.epistemic_basis}"
            )
        object.__setattr__(self, "attention_ids", tuple(self.attention_ids))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(
            self,
            "knowledge_gap_ids",
            tuple(self.knowledge_gap_ids),
        )
        object.__setattr__(self, "provenance", tuple(self.provenance))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "primary_obligation": self.primary_obligation,
            "attention_ids": list(self.attention_ids),
            "epistemic_status": self.epistemic_status,
            "epistemic_basis": self.epistemic_basis,
            "evidence_ids": list(self.evidence_ids),
            "knowledge_gap_ids": list(self.knowledge_gap_ids),
            "response_commitment": self.response_commitment,
            "should_speak": self.should_speak,
            "action": self.action.to_dict(),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True)
class CompilationResult:
    minimal_plan: MinimalGroundedPlan
    intention: Intention
    trace: tuple[dict[str, JsonValue], ...]


@dataclass
class MinimalGroundedReasoningEngine:
    model: MinimalGroundedModel

    def deliberate(self, context: InternalContext) -> MinimalGroundedPlan:
        bundle = _require_bundle(context)
        raw = self.model.generate_for_context(
            context,
            build_minimal_grounded_prompt(context),
        )
        payload = extract_reasoning_model_object(raw)
        return minimal_plan_from_model_dict(payload, bundle)

    def compile(
        self,
        minimal: MinimalGroundedPlan,
        context: InternalContext,
    ) -> CompilationResult:
        return DeterministicPlanCompiler().compile(minimal, context)

    def reason(self, context: InternalContext) -> CompilationResult:
        minimal = self.deliberate(context)
        return self.compile(minimal, context)


class DeterministicPlanCompiler:
    """Compile one semantic plan without adding evidence or changing its meaning."""

    def compile(
        self,
        minimal: MinimalGroundedPlan,
        context: InternalContext,
    ) -> CompilationResult:
        bundle = _require_bundle(context)
        if minimal.evidence_bundle_id != bundle.id:
            raise MinimalPlanCompileError("evidence_bundle_mismatch")
        failures = [item for item in minimal.provenance if not item.succeeded]
        if failures:
            reasons = ",".join(
                f"{item.field_path}:{item.failure_reason}" for item in failures
            )
            raise MinimalPlanCompileError(f"provenance_resolution_failed:{reasons}")

        primary_id = _primary_evidence_id(bundle)
        known_ids = bundle.reference_ids
        selected_ids = tuple(
            dict.fromkeys(
                (
                    *minimal.attention_ids,
                    *minimal.evidence_ids,
                    *minimal.knowledge_gap_ids,
                )
            )
        )
        unknown_ids = [item for item in selected_ids if item not in known_ids]
        if unknown_ids:
            raise MinimalPlanCompileError(
                f"unknown_evidence:{','.join(sorted(unknown_ids))}"
            )
        unauthorized = {
            item.id for item in bundle.items if not item.authorized
        }
        unauthorized_ids = [
            item for item in selected_ids if item in unauthorized
        ]
        if unauthorized_ids:
            raise MinimalPlanCompileError(
                f"unauthorized_evidence:{','.join(sorted(unauthorized_ids))}"
            )
        if not minimal.should_speak and not bundle.interaction.silence_allowed:
            raise MinimalPlanCompileError("silence_not_allowed")
        if minimal.should_speak and not minimal.response_commitment.strip():
            raise MinimalPlanCompileError("spoken_commitment_missing")

        evidence_by_id = {item.id: item for item in bundle.items}
        gap_by_id = {item.id: item for item in bundle.knowledge_gaps}
        request_ids = {
            item.id
            for item in bundle.items
            if item.attributes.get("epistemic_use") == "interaction_request"
        }
        factual_ids = tuple(
            item
            for item in minimal.evidence_ids
            if item not in request_ids
        )
        trace: list[dict[str, JsonValue]] = [
            {
                "field": "attention.primary",
                "rule": "interaction_primary_is_mandatory",
                "inputs": [primary_id] if primary_id else [],
                "outputs": [primary_id] if primary_id else [],
            },
            {
                "field": "knowledge.factual_evidence",
                "rule": "interaction_requests_are_not_factual_support",
                "inputs": list(minimal.evidence_ids),
                "outputs": list(factual_ids),
            },
        ]
        _validate_epistemic_basis(
            minimal,
            factual_ids=factual_ids,
            request_ids=request_ids,
        )
        knowledge_source, knowledge_subject = _derive_knowledge_identity(
            minimal,
            factual_ids,
            evidence_by_id,
            gap_by_id,
        )

        attention_ids = tuple(
            dict.fromkeys(
                item
                for item in (
                    primary_id,
                    *minimal.attention_ids,
                    *factual_ids,
                    *minimal.knowledge_gap_ids,
                )
                if item
            )
        )
        plan_id = f"plan:minimal:{bundle.context_revision[:16]}"
        claim_id = f"claim:minimal:{bundle.context_revision[:16]}:primary"
        confidence = _derived_confidence(
            minimal,
            factual_ids,
            evidence_by_id,
        )
        claim_predicate = _claim_predicate(minimal)
        claim_role = (
            "caveat"
            if minimal.epistemic_status in {"unknown", "uncertain", "unavailable"}
            else "answer"
        )
        should_have_claim = bool(minimal.response_commitment.strip())
        claims = (
            (
                GroundedClaim(
                    id=claim_id,
                    subject=knowledge_subject,
                    predicate=claim_predicate,
                    content=minimal.response_commitment.strip(),
                    role=claim_role,
                    evidence_ids=factual_ids,
                    knowledge_gap_ids=minimal.knowledge_gap_ids,
                    certainty=confidence,
                    required=minimal.should_speak,
                ),
            )
            if should_have_claim
            else ()
        )

        action, action_commitment = _compile_action(
            minimal,
            bundle,
            plan_id=plan_id,
            primary_id=primary_id,
        )
        expression = _compile_expression(context)
        output_manifest = OutputManifest(
            conversation_required=minimal.should_speak,
            speech_required=minimal.should_speak,
            body_required=True,
            action_ids=tuple(item.id for item in action),
        )
        plan = GroundedResponsePlan(
            id=plan_id,
            evidence_bundle_id=bundle.id,
            primary_obligation=minimal.primary_obligation,
            response_goal=f"{minimal.primary_obligation}:current_interaction",
            attention=AttentionCommitment(
                target_ids=attention_ids,
                topic=knowledge_subject,
                focus=minimal.primary_obligation,
                priority=1.0,
            ),
            knowledge=KnowledgeCommitment(
                subject=knowledge_subject,
                source=knowledge_source,
                epistemic_status=minimal.epistemic_status,
                evidence_ids=factual_ids,
                knowledge_gap_ids=minimal.knowledge_gap_ids,
                confidence=confidence,
            ),
            claims=claims,
            engagement=EngagementCommitment(
                mode="speak" if minimal.should_speak else "silence",
                should_speak=minimal.should_speak,
                initiative=(
                    bundle.interaction.kind == "proactive_opportunity"
                    and minimal.primary_obligation == "initiate"
                ),
                reason_code=minimal.primary_obligation,
                evidence_ids=attention_ids,
            ),
            action=action_commitment,
            expression=expression,
            output_manifest=output_manifest,
            confidence=confidence,
            extensions={
                "model_wire_format": "minimal_grounded_v1",
                "minimal_plan": minimal.to_dict(),
                "compiler_trace": trace,
            },
        )
        conversation = ConversationIntent(
            mode=_conversation_mode(minimal.primary_obligation),
            target_topic=knowledge_subject,
            reason=minimal.primary_obligation,
            plan_id=plan_id,
            claim_ids=tuple(item.id for item in claims),
        )
        speech_text = (
            minimal.response_commitment.strip()
            if minimal.should_speak
            else ""
        )
        speech = SpeechIntention(
            text=speech_text,
            emotion=expression.stance,
            energy=expression.energy,
            warmth=expression.warmth,
            confidence=confidence,
            thinking=clamp01(1.0 - confidence),
            curiosity=0.0,
            pause_before=0.14,
            pause_after=0.14,
            speech_rate=0.94,
            pitch_variation=0.12,
            volume=0.88,
            smile=clamp01(expression.warmth * 0.22),
            plan_id=plan_id,
            realized_claim_ids=(
                tuple(item.id for item in claims)
                if minimal.should_speak
                else ()
            ),
            metadata={"origin": "minimal_grounded_compiler"},
        )
        body = BodyIntention(
            plan_id=plan_id,
            expression=expression.stance,
            gesture="idle",
            gaze="user",
            posture="relaxed",
            attention="user",
            intensity=expression.intensity,
            movement_energy=expression.energy,
            metadata={"origin": "minimal_grounded_compiler"},
        )
        intention = Intention(
            id=f"intention:minimal:{bundle.context_revision[:16]}",
            intent_type="conversation",
            conversation_intent=conversation,
            speech=speech_text,
            speech_intention=speech,
            body_intention=body,
            actions=action,
            confidence=confidence,
            reasoning_summary="minimal_grounded_plan_compiled",
            grounded_plan=plan,
            avatar={
                "expression": expression.stance,
                "gesture": "idle",
                "gaze": "user",
                "plan_id": plan_id,
            },
        )
        return CompilationResult(
            minimal_plan=minimal,
            intention=intention,
            trace=tuple(trace),
        )


def build_minimal_grounded_prompt(context: InternalContext) -> str:
    bundle = _require_bundle(context)
    evidence_json = json.dumps(
        bundle.to_model_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "Eres la unica mente deliberativa de Anydes durante este turno.",
            "Devuelve un unico MinimalGroundedPlan JSON. No describas razonamiento.",
            "Decide solo el significado; el compilador creara enlaces, fuente, claims, Speech, Body y manifiestos.",
            "La observacion primaria ya se incluye automaticamente en atencion.",
            "attention_ids contiene solo evidencia adicional realmente relevante.",
            "epistemic_basis=evidence requiere hechos autorizados en evidence_ids.",
            "epistemic_basis=general_knowledge no necesita evidence_ids.",
            "epistemic_basis=unknown requiere knowledge_gap_ids existentes.",
            "epistemic_basis=not_applicable sirve para reconocer, actuar, escuchar o guardar silencio sin afirmar hechos.",
            "Una request prueba lo pedido, no la verdad de una respuesta.",
            "response_commitment es exactamente el contenido natural que Anydes se compromete a comunicar; no copies la peticion.",
            "Elige action=none salvo que el usuario solicite ejecutar ahora una capacidad disponible.",
            "should_speak=false solo cuando el contexto permite silencio y callar cumple la interaccion.",
            "No propongas memoria, relacion, curiosidad, expresion ni estado.",
            "Valores obligation: answer|acknowledge|clarify|act|observe|listen|initiate|recover.",
            "Valores status: known|unknown|uncertain|unavailable|not_applicable.",
            "EVIDENCE BUNDLE:",
            evidence_json,
            "Devuelve exactamente estas claves y completa todos los valores vacios segun este turno.",
            "Los valores vacios son marcadores de esquema, nunca contenido de respuesta.",
            '{"obligation":"","attention_ids":[],"status":"",'
            '"epistemic_basis":"","evidence_ids":[],"knowledge_gap_ids":[],'
            '"response_commitment":"","should_speak":true,"action":"",'
            '"action_parameters":{}}',
        ]
    )


def minimal_plan_from_model_dict(
    data: dict[str, object],
    bundle: EvidenceBundle,
) -> MinimalGroundedPlan:
    normalized = _strip_keys(data)
    nested = normalized.get("minimal_grounded_plan")
    if isinstance(nested, dict):
        normalized = _strip_keys(nested)
    provenance = resolve_grounded_wire_provenance(normalized, bundle)
    payload = provenance.payload
    action_value = payload.get("action")
    if isinstance(action_value, dict):
        action_type = str(
            action_value.get("action_type")
            or action_value.get("type")
            or "none"
        )
        parameters = action_value.get("parameters")
    else:
        action_type = str(action_value or "none")
        parameters = payload.get("action_parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    basis = str(payload.get("epistemic_basis") or "").strip()
    return MinimalGroundedPlan(
        id=f"minimal-plan:{bundle.context_revision[:16]}",
        evidence_bundle_id=bundle.id,
        primary_obligation=str(
            payload.get("primary_obligation")
            or payload.get("obligation")
            or ""
        ).strip(),
        attention_ids=_strings(payload.get("attention_ids")),
        epistemic_status=str(
            payload.get("epistemic_status")
            or payload.get("status")
            or ""
        ).strip(),
        epistemic_basis=basis,
        evidence_ids=_strings(payload.get("evidence_ids")),
        knowledge_gap_ids=_strings(
            payload.get("knowledge_gap_ids", payload.get("gap_ids"))
        ),
        response_commitment=str(
            payload.get("response_commitment") or ""
        ).strip(),
        should_speak=_boolean(payload.get("should_speak"), True),
        action=MinimalActionChoice(
            action_type=action_type.strip(),
            parameters=parameters,
        ),
        provenance=provenance.records,
    )


def _validate_epistemic_basis(
    minimal: MinimalGroundedPlan,
    *,
    factual_ids: tuple[str, ...],
    request_ids: set[str],
) -> None:
    selected_requests = request_ids.intersection(minimal.evidence_ids)
    if minimal.epistemic_basis == "evidence" and not factual_ids:
        raise MinimalPlanCompileError("evidence_basis_without_factual_evidence")
    if minimal.epistemic_basis == "general_knowledge" and factual_ids:
        raise MinimalPlanCompileError(
            "general_knowledge_basis_with_selected_factual_evidence"
        )
    if (
        minimal.epistemic_basis == "unknown"
        and not minimal.knowledge_gap_ids
    ):
        raise MinimalPlanCompileError("unknown_basis_without_knowledge_gap")
    if (
        minimal.epistemic_basis == "not_applicable"
        and minimal.epistemic_status != "not_applicable"
    ):
        raise MinimalPlanCompileError("not_applicable_basis_status_mismatch")
    if (
        minimal.epistemic_basis == "general_knowledge"
        and selected_requests
    ):
        # Requests remain in attention but never become factual claim support.
        return


def _derive_knowledge_identity(
    minimal: MinimalGroundedPlan,
    factual_ids: tuple[str, ...],
    evidence_by_id: dict[str, Any],
    gap_by_id: dict[str, Any],
) -> tuple[str, str]:
    if minimal.epistemic_basis == "general_knowledge":
        return "general_knowledge", "general_knowledge"
    if minimal.epistemic_basis == "not_applicable":
        return "conversation", "current_interaction"
    if minimal.epistemic_basis == "unknown":
        gaps = [
            gap_by_id[item]
            for item in minimal.knowledge_gap_ids
            if item in gap_by_id
        ]
        if not gaps:
            raise MinimalPlanCompileError("unknown_gap_not_found")
        subjects = {item.subject for item in gaps}
        subject = next(iter(subjects)) if len(subjects) == 1 else "unknown"
        source = "self" if subject.casefold() == "anydes" else "none"
        return source, subject
    facts = [evidence_by_id[item] for item in factual_ids]
    sources = {item.source for item in facts}
    subjects = {item.subject for item in facts}
    source = next(iter(sources)) if len(sources) == 1 else "conversation"
    subject = next(iter(subjects)) if len(subjects) == 1 else "conversation"
    return source, subject


def _compile_action(
    minimal: MinimalGroundedPlan,
    bundle: EvidenceBundle,
    *,
    plan_id: str,
    primary_id: str,
) -> tuple[tuple[ActionRequest, ...], ActionCommitment]:
    action_type = minimal.action.action_type.strip()
    if action_type.casefold() in {"", "none"}:
        if minimal.primary_obligation == "act":
            raise MinimalPlanCompileError("act_obligation_without_action")
        return (), ActionCommitment(phase="none")
    if minimal.primary_obligation != "act":
        raise MinimalPlanCompileError("action_conflicts_with_primary_obligation")
    matches = [
        item
        for item in bundle.capabilities
        if action_type in {item.id, item.action_type, f"capability:{item.action_type}"}
    ]
    if len(matches) != 1:
        raise MinimalPlanCompileError(
            "ambiguous_action" if matches else f"unknown_action:{action_type}"
        )
    capability = matches[0]
    unknown_parameters = set(minimal.action.parameters).difference(
        capability.parameter_names
    )
    if unknown_parameters:
        raise MinimalPlanCompileError(
            "unknown_action_parameters:"
            + ",".join(sorted(unknown_parameters))
        )
    action_id = f"action:minimal:{bundle.context_revision[:16]}:primary"
    evidence_ids = (primary_id,) if primary_id else ()
    request = ActionRequest(
        id=action_id,
        action_type=capability.action_type,
        parameters=minimal.action.parameters,
        requires_confirmation=capability.requires_confirmation,
        risk_level=capability.risk_level,
        reason=minimal.response_commitment,
        plan_id=plan_id,
        evidence_ids=evidence_ids,
    )
    commitment = ActionCommitment(
        phase="request",
        capability_id=capability.id,
        action_type=capability.action_type,
        objective=minimal.response_commitment,
        evidence_ids=evidence_ids,
        expected_observation=f"tool_result:{capability.action_type}",
        requires_followup=True,
    )
    return (request,), commitment


def _compile_expression(context: InternalContext) -> ExpressionCommitment:
    warmth = clamp01(
        context.emotion.affection * 0.56
        + context.emotion.interest * 0.28
        + 0.16
    )
    energy = clamp01(context.emotion.energy)
    return ExpressionCommitment(
        stance=context.emotion.dominant_label or "calm",
        warmth=warmth,
        energy=energy,
        formality=0.24,
        closeness=clamp01(warmth * 0.86),
        intensity=clamp01(0.22 + energy * 0.28),
        attention_target="user",
    )


def _derived_confidence(
    minimal: MinimalGroundedPlan,
    factual_ids: tuple[str, ...],
    evidence_by_id: dict[str, Any],
) -> float:
    if factual_ids:
        return clamp01(
            sum(evidence_by_id[item].confidence for item in factual_ids)
            / len(factual_ids)
        )
    return {
        "known": 0.82,
        "unknown": 0.72,
        "uncertain": 0.58,
        "unavailable": 0.72,
        "not_applicable": 0.86,
    }[minimal.epistemic_status]


def _claim_predicate(minimal: MinimalGroundedPlan) -> str:
    if minimal.epistemic_status in {"unknown", "uncertain", "unavailable"}:
        return "uncertainty"
    if minimal.primary_obligation in {"acknowledge", "act", "listen"}:
        return "acknowledgement"
    if minimal.primary_obligation == "observe":
        return "observation"
    return "assertion"


def _conversation_mode(obligation: str) -> str:
    return {
        "answer": "respond",
        "acknowledge": "respond",
        "clarify": "clarify",
        "act": "respond",
        "observe": "share_observation",
        "listen": "listen",
        "initiate": "respond",
        "recover": "respond",
    }[obligation]


def _primary_evidence_id(bundle: EvidenceBundle) -> str:
    observation_id = bundle.interaction.primary_observation_id
    if not observation_id:
        return ""
    candidate = f"evidence:observation:{observation_id}"
    return candidate if bundle.contains_reference(candidate) else ""


def _require_bundle(context: InternalContext) -> EvidenceBundle:
    if context.evidence_bundle is None:
        raise MinimalPlanCompileError("evidence_bundle_missing")
    return context.evidence_bundle


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _boolean(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _strip_keys(value: dict[str, object]) -> dict[str, object]:
    return {str(key).strip(): item for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
