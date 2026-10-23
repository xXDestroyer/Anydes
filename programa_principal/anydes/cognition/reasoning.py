from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Callable, Protocol

from anydes.cognition.contracts import (
    ActionRequest,
    BodyIntention,
    ConversationIntent,
    CuriosityMutation,
    InternalContext,
    Intention,
    MemoryItem,
    RelationshipMutation,
    SelfMutation,
    SpeechIntention,
)
from anydes.cognition.grounding_contracts import (
    ActionCommitment,
    AttentionCommitment,
    BodyIntention,
    DeferredIntent,
    EngagementCommitment,
    ExpressionCommitment,
    GroundedClaim,
    GroundedResponsePlan,
    KnowledgeCommitment,
    OutputManifest,
)
from anydes.cognition.provenance_resolver import (
    resolve_grounded_wire_provenance,
)


SPEECH_INTENTION_MODEL_FIELDS = frozenset(
    {
        "text",
        "emotion",
        "energy",
        "warmth",
        "confidence",
        "thinking",
        "curiosity",
        "pause_before",
        "pause_after",
        "speech_rate",
        "pitch_variation",
        "volume",
        "smile",
    }
)
GROUNDED_SPEECH_LINK_FIELDS = frozenset({"plan_id", "realized_claim_ids"})

INTENTION_OUTPUT_TEMPLATE = (
    '{"intent_type":"conversation","conversation_intent":{"mode":"respond",'
    '"target_topic":"","reason":"brief private basis","curiosity_id":""},'
    '"speech":{"text":"natural Spanish response","emotion":"calm","energy":0.5,'
    '"warmth":0.7,"confidence":0.8,"thinking":0.2,"curiosity":0.3,'
    '"pause_before":0.15,"pause_after":0.15,"speech_rate":0.95,'
    '"pitch_variation":0.12,"volume":0.88,"smile":0.15},"actions":[],'
    '"avatar":{},"memory_updates":[],"relationship_updates":[],'
    '"self_updates":[],"curiosity_updates":[],"confidence":0.8,'
    '"reasoning_summary":"brief private decision basis"}'
)

class ReasoningModel(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a JSON string describing an Intention."""


@dataclass
class ReasoningEngine:
    """Turns InternalContext into Intention.

    The engine receives only InternalContext and returns only Intention. It does not
    execute actions, write memory, speak, or modify assistant runtime behavior.
    """

    model: ReasoningModel | None = None
    error_reporter: Callable[[Exception], None] | None = None
    require_grounded_plan: bool = False

    def reason(self, context: InternalContext) -> Intention:
        if self.model is None:
            return self.recover(context, reason="No reasoning model connected yet.")

        prompt = (
            build_grounded_reasoning_prompt(context)
            if self.require_grounded_plan
            else build_reasoning_prompt(context)
        )
        try:
            context_generator = getattr(self.model, "generate_for_context", None)
            raw = (
                context_generator(context, prompt)
                if callable(context_generator)
                else self.model.generate(prompt)
            )
            data = extract_reasoning_model_object(raw)
            return (
                grounded_intention_from_model_dict(data, context)
                if self.require_grounded_plan
                else intention_from_model_dict(data, context)
            )
        except Exception as exc:
            if self.error_reporter is not None:
                self.error_reporter(exc)
            return self.recover(
                context,
                reason=f"Reasoning model output could not be parsed: {exc}",
                intent_type="reasoning_parse_error",
            )

    def recover(
        self,
        context: InternalContext,
        *,
        reason: str,
        intent_type: str = "observe",
        failure_scope: str = "reasoning",
        speech: str = "Perdon... creo que me quede pensando un momento. Podrias repetirmelo?",
    ) -> Intention:
        """Return a safe conversational intention with no side effects."""

        uncertainty_values = (
            context.uncertainty.memory,
            context.uncertainty.vision,
            context.uncertainty.knowledge,
            context.uncertainty.reasoning,
            context.uncertainty.user_intent,
            context.uncertainty.action_risk,
        )
        average_uncertainty = sum(uncertainty_values) / len(uncertainty_values)
        confidence = max(0.55, min(0.9, 1.0 - average_uncertainty))

        speech_intention = SpeechIntention(
            text=speech,
            emotion=context.emotion.dominant_label,
            energy=context.emotion.energy,
            warmth=(context.emotion.affection * 0.62 + context.emotion.interest * 0.38),
            confidence=confidence,
            thinking=context.uncertainty.reasoning,
            curiosity=(context.emotion.interest * 0.64 + _curiosity_signal(context) * 0.36),
            pause_before=(0.14 + context.uncertainty.reasoning * 0.22 + context.uncertainty.user_intent * 0.08),
            pause_after=(0.14 + context.uncertainty.reasoning * 0.08),
            speech_rate=(0.94 - context.uncertainty.reasoning * 0.08 - context.uncertainty.user_intent * 0.04),
            pitch_variation=(0.11 + context.emotion.interest * 0.06 + context.emotion.energy * 0.03),
            volume=(0.82 + context.emotion.energy * 0.10 + confidence * 0.04),
            smile=(context.emotion.affection * 0.34 + max(0.0, context.emotion.mood) * 0.18),
            metadata={"origin": "recovery_intention", "failure_scope": failure_scope},
        )

        return Intention(
            id=_stable_intention_id(context, intent_type),
            intent_type=intent_type,
            speech=speech_intention.text,
            speech_intention=speech_intention,
            actions=(),
            avatar={},
            memory_updates=(),
            relationship_updates=(),
            confidence=confidence,
            reasoning_summary=reason,
        )


def build_reasoning_prompt(context: InternalContext) -> str:
    current_user_text = _current_user_text(context)
    from anydes.cognition.conversation_planner import is_direct_question

    direct_answer_required = is_direct_question(current_user_text)
    context_json = json.dumps(
        reasoning_evidence_payload(context),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        [
            "You are the Reasoning Engine for Anydes.",
            "The conversation is the product. Architecture must remain completely invisible in speech.text.",
            "Before deciding, resolve in order: (1) what the user actually wants, (2) what they need to hear now, (3) what response would feel natural in this conversation.",
            "The current user observation has priority over memory, relationship, curiosity, promises and open loops.",
            "Answer every direct question fully before considering any optional follow-up. Never replace an answer with curiosity.",
            "Use general knowledge to answer ordinary knowledge questions. Ask for clarification only when the meaning is genuinely ambiguous.",
            "Do not echo the user, reuse old assistant text, invent facts, or ask a question merely to continue talking.",
            "Do not ask merely to keep the conversation alive.",
            "Use memory, relationship and curiosity only when directly relevant. Curiosity is optional and comes after the answer.",
            "The user authored the latest observation. Their first-person words refer to the user, not Anydes. Questions about who the user is must use relevant user memory.",
            "Without grounded vision evidence, make no claim about what Anydes sees.",
            "Without a grounded SELF fact or preference, say naturally that it is undecided or still being discovered.",
            "Do not volunteer that Anydes is an AI or describe her capabilities unless the user explicitly asks about her nature or capabilities.",
            "Never mention Reasoning Engine, InternalContext, Providers, Reflection, pipeline, JSON or hidden state in speech.text.",
            "Before writing speech, choose conversation_intent.",
            "Allowed conversation_intent modes: respond, ask, deepen, recall, shift_topic, clarify, share_observation, listen.",
            "Do not execute actions or apply state. Optional actions and state mutations are structured proposals only and require evidence.",
            "Relationship time alone never justifies resurfacing a thread or open loop.",
            "A Promise is a social commitment, not a task, timer or automatic reminder.",
            "available_actions are capability descriptors. Propose only listed actions that are necessary for the user's current request.",
            "When a tool Observation already contains a result, answer naturally from that structured result and do not repeat the same action.",
            "speech MUST be a complete object (SpeechIntention); a plain string is invalid. Keep vocal differences subtle and natural.",
            "reasoning_summary is a brief private basis, never chain-of-thought or user-facing text.",
            "Relevant InternalContext JSON:",
            context_json,
            "Return exactly one JSON object. Replace example values and preserve every key:",
            INTENTION_OUTPUT_TEMPLATE,
            "AVAILABLE ACTIONS FOR THIS TURN:",
            json.dumps(
                [
                    {
                        "action_type": item.action_type,
                        "description": item.reason,
                        "requires_confirmation": item.requires_confirmation,
                        "risk_level": item.risk_level,
                    }
                    for item in context.available_actions
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            f"LATEST USER MESSAGE (highest priority): {json.dumps(current_user_text, ensure_ascii=False)}",
            (
                "DIRECT ANSWER REQUIRED: yes. Give the actual useful explanation or information now. "
                "Do not merely confirm the subject and do not answer with another question."
                if direct_answer_required
                else "DIRECT ANSWER REQUIRED: no. Respond naturally to the latest user message."
            ),
            "ACTION DECISION: If truthful completion needs one available action, include that ActionRequest now. Otherwise keep actions empty.",
            "Return the Intention JSON now. Do not return the context.",
        ]
    )


def build_grounded_reasoning_prompt(context: InternalContext) -> str:
    """Build the one-call plan-first prompt used only by the shadow route."""

    bundle = context.evidence_bundle
    if bundle is None:
        raise ValueError("Grounded reasoning requires an EvidenceBundle.")
    evidence_json = json.dumps(
        bundle.to_model_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "Eres la unica mente deliberativa de Anydes durante este turno.",
            "Piensa una vez y devuelve un solo JSON, sin explicar tu razonamiento.",
            "Primero fija lo que sabes y la respuesta; despues redacta speech desde esa misma decision.",
            "Resuelve la peticion actual antes que curiosidad o continuidad opcional.",
            "Usa solo ids presentes. requests expresa lo que piden, nunca prueba la respuesta.",
            "facts contiene hechos concretos; background solo orienta identidad y expresion.",
            "Elige el hecho mas especifico. Si falta un dato propio y existe knowledge_gap, no lo inventes.",
            "La primera persona del mensaje pertenece a interaction.metadata.first_person_ref; la segunda persona se dirige a interaction.metadata.second_person_ref.",
            "El conocimiento general cotidiano puede usar source=general_knowledge sin inventar ids.",
            "No menciones arquitectura, JSON, evidencia ni procesos internos en speech.",
            "Curiosidad y continuidad son opcionales. No preguntes por defecto.",
            "Las acciones y cambios de estado son propuestas opcionales; omitelos si no hacen falta.",
            "EVIDENCE BUNDLE:",
            evidence_json,
            "COMPROMISO:",
            "- Si hay direct_answer_required: obligation=answer, mode=respond y responde antes de preguntar.",
            "- Para una preferencia de Anydes, background de identidad no basta: usa el fact SELF concreto o el gap.",
            "- Si un fact de current_user responde quien es el hablante, usalo; no respondas quien es Anydes.",
            "- En proactive_opportunity: speak=false salvo que facts contenga una razon concreta, oportuna y util.",
            "- Nunca copies la pregunta como speech.",
            "Valores: obligation=answer|acknowledge|clarify|act|observe|listen|initiate|recover; mode=respond|ask|deepen|recall|shift_topic|clarify|share_observation|listen; status=known|unknown|uncertain|unavailable|not_applicable.",
            "Devuelve estas claves principales: obligation, mode, goal, topic, attention_ids, subject, source, status, evidence_ids, gap_ids, claim_kind, claim, speech, speak, initiative, engagement_evidence_ids, confidence.",
            "claim_kind debe ser assertion, uncertainty, acknowledgement o observation. Si status=unknown, claim_kind=uncertainty y claim/speech deben expresar honestamente que el dato no esta establecido.",
            "Si source=memory, evidence_ids debe contener el id exacto de un fact con source=memory; aplica la misma coherencia para self, relationship, tool y observation.",
            "Incluye expression={stance,warmth,energy,formality,closeness,intensity}. Cada numero va de 0 a 1.",
            "Opcionalmente incluye action, action_parameters, deferred_intents y state_proposals. action debe ser una capacidad exacta o none.",
            "Devuelve el JSON ahora.",
        ]
    )


def grounded_intention_from_model_dict(
    data: dict[str, object],
    context: InternalContext,
) -> Intention:
    """Materialize every manifestation from one compact model decision."""

    data = _strip_wire_keys(data)
    nested_plan = data.get("grounded_plan")
    if (
        isinstance(nested_plan, dict)
        and "id" in nested_plan
        and "evidence_bundle_id" in nested_plan
    ):
        return intention_from_model_dict(
            data,
            context,
            require_grounded_plan=True,
        )
    if isinstance(nested_plan, dict):
        data = nested_plan
    bundle = context.evidence_bundle
    if bundle is None:
        raise ValueError("Grounded reasoning requires an EvidenceBundle.")
    provenance = resolve_grounded_wire_provenance(data, bundle)
    data = provenance.payload
    decision = (
        data.get("decision")
        if isinstance(data.get("decision"), dict)
        else data
    )
    expression_data = (
        data.get("expression")
        if isinstance(data.get("expression"), dict)
        else data
    )
    proposals = (
        data.get("state_proposals")
        if isinstance(data.get("state_proposals"), dict)
        else data
    )
    wire_issues = [
        (
            f"provenance:{item.failure_reason}:"
            f"{item.field_path.removeprefix('$.')}"
        )
        for item in provenance.failures
    ]
    plan_id = f"plan:{bundle.context_revision[:16]}"
    claim_id = f"claim:{bundle.context_revision[:16]}:primary"
    attention_ids = tuple(
        str(item) for item in _ensure_list(decision.get("attention_ids"))
    )
    if not attention_ids and bundle.interaction.primary_observation_id:
        primary_evidence_id = (
            f"evidence:observation:{bundle.interaction.primary_observation_id}"
        )
        if bundle.contains_reference(primary_evidence_id):
            attention_ids = (primary_evidence_id,)
    evidence_ids = tuple(
        str(item) for item in _ensure_list(decision.get("evidence_ids"))
    )
    gap_ids = tuple(
        str(item)
        for item in _ensure_list(
            decision.get("knowledge_gap_ids", decision.get("gap_ids"))
        )
    )
    boundaries = tuple(
        str(item) for item in _ensure_list(decision.get("boundaries"))
    )
    primary_obligation = str(
        decision.get("primary_obligation")
        or decision.get("obligation")
        or ""
    )
    conversation_mode = str(
        decision.get("conversation_mode")
        or decision.get("mode")
        or ""
    )
    response_goal = str(
        decision.get("response_goal")
        or decision.get("goal")
        or ""
    ).strip()
    claim_content = str(decision.get("claim") or "").strip()
    proposed_speech = expression_data.get("speech_text")
    if proposed_speech is None:
        proposed_speech = data.get("speech")
    speech_text = _grounded_speech_realization(
        str(proposed_speech or "").strip(),
        claim_content,
        _current_user_text(context),
        primary_obligation=primary_obligation,
    )
    confidence = _wire_float(
        decision.get("confidence"),
        _wire_float(expression_data.get("confidence"), 0.5),
    )
    expression = ExpressionCommitment(
        stance=str(expression_data.get("stance") or "calm"),
        warmth=_wire_float(expression_data.get("warmth"), 0.7),
        energy=_wire_float(expression_data.get("energy"), 0.4),
        formality=_wire_float(expression_data.get("formality"), 0.25),
        closeness=_wire_float(expression_data.get("closeness"), 0.6),
        intensity=_wire_float(expression_data.get("intensity"), 0.35),
        attention_target=_wire_text(expression_data.get("gaze"), "user"),
        evidence_ids=tuple(
            str(item)
            for item in _ensure_list(expression_data.get("evidence_ids"))
        ),
    )
    action_name = str(decision.get("action") or "none").strip()
    if action_name.casefold() in {
        "ask",
        "listen",
        "respond",
        "silence",
        "speak",
    }:
        action_name = "none"
    capability = next(
        (
            item
            for item in bundle.capabilities
            if item.action_type == action_name
        ),
        None,
    )
    if action_name != "none" and capability is None:
        wire_issues.append(f"unknown_action:{action_name}")
    action_id = f"action:{bundle.context_revision[:16]}:primary"
    actions = (
        (
            ActionRequest(
                id=action_id,
                action_type=capability.action_type,
                parameters=(
                    decision.get("action_parameters")
                    if isinstance(decision.get("action_parameters"), dict)
                    else {}
                ),
                requires_confirmation=capability.requires_confirmation,
                risk_level=capability.risk_level,
                reason=response_goal,
                plan_id=plan_id,
                evidence_ids=evidence_ids,
            ),
        )
        if capability is not None
        else ()
    )
    memory_updates_list = []
    for item in _ensure_list(proposals.get("memory_updates")):
        try:
            memory = MemoryItem.from_dict(_ensure_dict(item, "memory_update"))
            memory_updates_list.append(
                replace(
                    memory,
                    metadata={**memory.metadata, "plan_id": plan_id},
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            wire_issues.append(f"invalid_memory_update:{type(exc).__name__}")
    memory_updates = tuple(memory_updates_list)
    relationship_updates = _parse_wire_updates(
        proposals.get("relationship_updates"),
        "relationship_update",
        RelationshipMutation.from_dict,
        wire_issues,
    )
    self_updates = _parse_wire_updates(
        proposals.get("self_updates"),
        "self_update",
        SelfMutation.from_dict,
        wire_issues,
    )
    curiosity_updates = _parse_wire_updates(
        proposals.get("curiosity_updates"),
        "curiosity_update",
        CuriosityMutation.from_dict,
        wire_issues,
    )
    deferred_list = list(
        _parse_wire_updates(
            decision.get("deferred_intents"),
            "deferred_intent",
            DeferredIntent.from_dict,
            wire_issues,
        )
    )
    deferred = tuple(deferred_list)
    should_speak = _wire_bool(
        decision.get("should_speak", decision.get("speak")),
        True,
    )
    engagement_evidence_ids = tuple(
        str(item)
        for item in _ensure_list(decision.get("engagement_evidence_ids"))
    )
    plan = GroundedResponsePlan(
        id=plan_id,
        evidence_bundle_id=bundle.id,
        primary_obligation=primary_obligation,
        response_goal=response_goal,
        attention=AttentionCommitment(
            target_ids=attention_ids,
            topic=str(decision.get("topic") or ""),
            focus=str(decision.get("focus") or response_goal),
            priority=_wire_float(decision.get("attention_priority"), 1.0),
        ),
        knowledge=KnowledgeCommitment(
            subject=str(
                decision.get("answer_subject")
                or decision.get("subject")
                or "unspecified"
            ),
            source=str(
                decision.get("answer_source")
                or decision.get("source")
                or "none"
            ),
            epistemic_status=str(
                decision.get("epistemic_status")
                or decision.get("status")
                or ""
            ),
            evidence_ids=evidence_ids,
            knowledge_gap_ids=gap_ids,
            boundaries=boundaries,
            confidence=confidence,
        ),
        claims=(
            (
                GroundedClaim(
                    id=claim_id,
                    subject=str(
                        decision.get("answer_subject")
                        or decision.get("subject")
                        or "unspecified"
                    ),
                    predicate=str(
                        decision.get("claim_kind")
                        or decision.get("claim_predicate")
                        or "response"
                    ),
                    content=claim_content,
                    role=(
                        "caveat"
                        if str(
                            decision.get("claim_kind")
                            or decision.get("claim_predicate")
                            or ""
                        )
                        == "uncertainty"
                        else "answer"
                    ),
                    evidence_ids=evidence_ids,
                    knowledge_gap_ids=gap_ids,
                    certainty=(
                        min(confidence, 0.5)
                        if str(
                            decision.get("epistemic_status")
                            or decision.get("status")
                            or ""
                        )
                        in {"unknown", "uncertain", "unavailable"}
                        else confidence
                    ),
                ),
            )
            if claim_content
            else ()
        ),
        engagement=EngagementCommitment(
            mode=(
                "speak"
                if should_speak
                else "silence"
            ),
            should_speak=should_speak,
            initiative=_wire_bool(decision.get("initiative"), False),
            reason_code=str(
                decision.get("engagement_reason") or primary_obligation
            ),
            evidence_ids=engagement_evidence_ids,
        ),
        action=ActionCommitment(
            phase="request" if actions else "none",
            capability_id=capability.id if capability is not None else "",
            action_type=capability.action_type if capability is not None else "",
            objective=response_goal if actions else "",
            evidence_ids=evidence_ids if actions else (),
            expected_observation=_wire_text(
                decision.get("expected_observation"),
                "",
            ),
            requires_followup=bool(actions),
        ),
        expression=expression,
        deferred_intents=deferred,
        output_manifest=OutputManifest(
            conversation_required=should_speak,
            speech_required=should_speak,
            body_required=True,
            action_ids=tuple(item.id for item in actions),
            memory_proposal_ids=tuple(item.id for item in memory_updates),
            relationship_proposal_ids=tuple(
                item.id for item in relationship_updates
            ),
            self_proposal_ids=tuple(item.id for item in self_updates),
            curiosity_proposal_ids=tuple(item.id for item in curiosity_updates),
        ),
        confidence=confidence,
        extensions={
            "model_wire_format": "compact_grounded_v1",
            "wire_issues": wire_issues,
            "provenance_resolution": [
                item.to_dict() for item in provenance.records
            ],
        },
    )
    speech = SpeechIntention(
        text=speech_text if should_speak else "",
        emotion=expression.stance,
        energy=expression.energy,
        warmth=expression.warmth,
        confidence=confidence,
        thinking=max(0.0, min(1.0, 0.16 + (1.0 - confidence) * 0.28)),
        curiosity=max(0.0, min(1.0, expression.intensity * 0.55)),
        pause_before=max(0.0, min(1.0, 0.1 + (1.0 - confidence) * 0.18)),
        pause_after=0.14,
        speech_rate=max(
            0.78,
            min(
                1.08,
                0.91 + expression.energy * 0.12 - (1.0 - confidence) * 0.05,
            ),
        ),
        pitch_variation=max(
            0.05,
            min(0.3, 0.08 + expression.energy * 0.08 + expression.warmth * 0.03),
        ),
        volume=max(0.65, min(1.0, 0.78 + expression.energy * 0.14)),
        smile=max(
            0.0,
            min(1.0, expression.warmth * 0.28 + expression.closeness * 0.12),
        ),
        plan_id=plan_id,
        realized_claim_ids=(claim_id,) if claim_content and should_speak else (),
    )
    body = BodyIntention(
        plan_id=plan_id,
        expression=expression.stance,
        gesture="idle",
        gaze=expression.attention_target,
        posture="relaxed",
        attention=expression.attention_target,
        intensity=expression.intensity,
        movement_energy=max(
            0.0,
            min(1.0, expression.energy * 0.55 + expression.intensity * 0.45),
        ),
        microexpressions=(),
        lip_sync_hint="speech" if should_speak and speech_text else "idle",
    )
    return Intention(
        id=f"intention:{bundle.context_revision[:16]}",
        intent_type="conversation",
        conversation_intent=ConversationIntent(
            mode=conversation_mode,
            target_topic=str(decision.get("topic") or ""),
            reason=response_goal,
            curiosity_id=str(decision.get("curiosity_id") or ""),
            relationship_thread_id=str(
                decision.get("relationship_thread_id") or ""
            ),
            open_loop_id=str(decision.get("open_loop_id") or ""),
            plan_id=plan_id,
            claim_ids=(claim_id,) if claim_content else (),
        ),
        speech=speech.text,
        speech_intention=speech,
        actions=actions,
        avatar=body.to_avatar_payload(),
        memory_updates=memory_updates,
        relationship_updates=relationship_updates,
        self_updates=self_updates,
        curiosity_updates=curiosity_updates,
        confidence=confidence,
        reasoning_summary="",
        grounded_plan=plan,
        body_intention=body,
    )


def build_revision_prompt(
    context: InternalContext,
    original: Intention,
    failures: tuple[str, ...],
) -> str:
    """Build a short corrective prompt so small local models can revise reliably."""

    evidence = reasoning_evidence_payload(context)
    evidence_json = json.dumps(
        _compact_json_value(evidence),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        [
            "Reformulate one rejected Anydes Intention.",
            "Return one complete JSON object and nothing else.",
            "Never mention validation, JSON, context, reasoning or architecture in speech.text.",
            f"VALIDATION FAILURES: {json.dumps(list(failures), ensure_ascii=False)}",
            f"CURRENT EVIDENCE: {evidence_json}",
            f"PREVIOUS INTENTION: {original.to_json()}",
            "Correct every failure while preserving valid action proposals and grounding.",
            "If a direct answer lacked information, provide the actual explanation or requested information. Do not merely confirm or repeat the subject. Do not answer with another question.",
            "OUTPUT TEMPLATE. Preserve every key:",
            INTENTION_OUTPUT_TEMPLATE,
            "Return the corrected Intention JSON now.",
        ]
    )


def reasoning_evidence_payload(context: InternalContext) -> dict[str, object]:
    """Return the one bounded evidence view shared by every reasoning pass."""

    mutable = context.self_model.mutable_state
    identity_core = mutable.get("identity_core", {})
    thinking_style: object = {}
    if isinstance(identity_core, dict):
        thinking_style = identity_core.get("thinking_style", {})

    return _compact_json_value(
        {
            "latest_user_message": _current_user_text(context),
            "observations": [
                {
                    "id": item.id,
                    "source": item.source,
                    "summary": item.summary,
                    "data": item.data,
                    "confidence": item.confidence,
                }
                for item in _selected_observations(context)
            ],
            "memory": context.memory.to_dict(),
            "goals": {
                "permanent": [
                    {
                        "id": item.id,
                        "description": item.description,
                        "priority": item.priority,
                        "status": item.status,
                    }
                    for item in context.goals.permanent
                ],
                "temporary": [
                    {
                        "id": item.id,
                        "description": item.description,
                        "priority": item.priority,
                        "status": item.status,
                    }
                    for item in context.goals.temporary
                ],
            },
            "self": {
                "subject": "solo_Anydes_no_el_usuario",
                "name": context.self_model.name,
                "kind": context.self_model.kind,
                "identity_summary": context.self_model.identity_summary,
                "personality_traits": list(context.self_model.personality_traits),
                "boundaries": list(context.self_model.boundaries),
                "thinking_style": thinking_style,
                "preferences": mutable.get("self_preferences", {}),
            },
            "emotion": context.emotion.to_dict(),
            "uncertainty": context.uncertainty.to_dict(),
            "curiosity": [item.to_dict() for item in context.curiosity],
            "relationship": context.relationship.to_dict(),
            "body": {
                "available": context.body.available,
                "expression": context.body.expression,
                "gesture": context.body.gesture,
                "gaze": context.body.gaze,
                "posture": context.body.posture,
                "speaking": context.body.speaking,
            },
        }
    )


def _selected_observations(context: InternalContext):
    """Keep event evidence while dropping unrelated ambient noise from a turn."""

    current_terms = set(_normalized_terms(_current_user_text(context)))
    selected = []
    for item in context.observations:
        if item.source in {"user_text", "voice", "tool", "vision"}:
            selected.append(item)
            continue
        observation_terms = set(
            _normalized_terms(f"{item.summary} {json.dumps(item.data, ensure_ascii=False)}")
        )
        if current_terms.intersection(observation_terms):
            selected.append(item)
    return tuple(selected)


def _normalized_terms(text: str) -> tuple[str, ...]:
    import re
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    stopwords = {
        "a", "al", "como", "cual", "de", "del", "el", "en", "es", "la",
        "las", "lo", "los", "me", "mi", "por", "que", "se", "si", "su",
        "tu", "una", "un", "y",
    }
    return tuple(
        item
        for item in re.findall(r"[a-z0-9]+", ascii_text)
        if len(item) > 1 and item not in stopwords
    )


def _compact_json_value(value):
    """Bound untrusted payload size while retaining structured conversational evidence."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key)
            lowered = normalized_key.casefold()
            if any(token in lowered for token in ("base64", "screenshot", "image_bytes", "audio_bytes")):
                continue
            result[normalized_key] = _compact_json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_compact_json_value(item) for item in value[-16:]]
    if isinstance(value, str):
        return value if len(value) <= 2400 else f"{value[:2400]} [truncated]"
    return value


def _current_user_text(context: InternalContext) -> str:
    for observation in reversed(context.observations):
        if observation.source != "user_text":
            continue
        text = observation.data.get("text")
        return str(text if isinstance(text, str) and text.strip() else observation.summary)
    return ""


def intention_from_model_dict(
    data: dict[str, object],
    context: InternalContext,
    *,
    require_grounded_plan: bool = False,
) -> Intention:
    intent_type = str(data.get("intent_type") or "respond")
    grounded_data = data.get("grounded_plan")
    if require_grounded_plan and not isinstance(grounded_data, dict):
        raise ValueError("Grounded reasoning output requires grounded_plan.")
    grounded_plan = (
        GroundedResponsePlan.from_dict(grounded_data)
        if isinstance(grounded_data, dict)
        else None
    )
    conversation_data = data.get("conversation_intent", {"mode": "respond"})
    if isinstance(conversation_data, str):
        conversation_intent = ConversationIntent(mode=conversation_data)
    else:
        conversation_intent = ConversationIntent.from_dict(
            _ensure_dict(conversation_data, "conversation_intent")
        )
    if conversation_intent.curiosity_id:
        curiosity_by_id = {item.id: item for item in context.curiosity}
        selected_curiosity = curiosity_by_id.get(conversation_intent.curiosity_id)
        if selected_curiosity is None:
            raise ValueError("ConversationIntent references an unknown curiosity object.")
        if not getattr(selected_curiosity, "is_active", not selected_curiosity.satisfied):
            raise ValueError("ConversationIntent references an inactive curiosity object.")
    if conversation_intent.relationship_thread_id and conversation_intent.relationship_thread_id not in {
        item.id for item in context.relationship.threads
    }:
        raise ValueError("ConversationIntent references an unknown relationship thread.")
    if conversation_intent.open_loop_id and conversation_intent.open_loop_id not in {
        item.id for item in context.relationship.open_loops
    }:
        raise ValueError("ConversationIntent references an unknown open conversation loop.")
    actions = tuple(
        ActionRequest.from_dict(_ensure_dict(item, "action"))
        for item in _ensure_list(data.get("actions"))
    )
    memory_updates = tuple(
        MemoryItem.from_dict(_ensure_dict(item, "memory_update"))
        for item in _ensure_list(data.get("memory_updates"))
    )
    relationship_updates = tuple(
        RelationshipMutation.from_dict(_ensure_dict(item, "relationship_update"))
        for item in _ensure_list(data.get("relationship_updates"))
    )
    self_updates = tuple(
        SelfMutation.from_dict(_ensure_dict(item, "self_update"))
        for item in _ensure_list(data.get("self_updates"))
    )
    curiosity_updates = tuple(
        CuriosityMutation.from_dict(_ensure_dict(item, "curiosity_update"))
        for item in _ensure_list(data.get("curiosity_updates"))
    )
    avatar = data.get("avatar")
    avatar_data = avatar if isinstance(avatar, dict) else {}
    speech_data = data.get("speech")
    if not isinstance(speech_data, dict):
        raise ValueError("Reasoning output field 'speech' must be a SpeechIntention object.")
    missing_speech_fields = SPEECH_INTENTION_MODEL_FIELDS.difference(speech_data)
    if require_grounded_plan:
        missing_speech_fields = missing_speech_fields.union(
            GROUNDED_SPEECH_LINK_FIELDS.difference(speech_data)
        )
    if missing_speech_fields:
        missing = ", ".join(sorted(missing_speech_fields))
        raise ValueError(f"Incomplete SpeechIntention; missing fields: {missing}")
    speech_intention = SpeechIntention.from_dict(speech_data)
    body_data = data.get("body_intention")
    if require_grounded_plan and not isinstance(body_data, dict):
        raise ValueError("Grounded reasoning output requires body_intention.")
    body_intention = (
        BodyIntention.from_dict(body_data)
        if isinstance(body_data, dict)
        else None
    )
    if require_grounded_plan and not conversation_intent.plan_id:
        raise ValueError("Grounded ConversationIntent requires plan_id.")
    if require_grounded_plan and grounded_plan is not None:
        avatar_data = (
            body_intention.to_avatar_payload()
            if body_intention is not None
            else {}
        )
    return Intention(
        id=str(data.get("id") or _stable_intention_id(context, intent_type)),
        intent_type=intent_type,
        conversation_intent=conversation_intent,
        speech=speech_intention.text,
        speech_intention=speech_intention,
        actions=actions,
        avatar=avatar_data,
        memory_updates=memory_updates,
        relationship_updates=relationship_updates,
        self_updates=self_updates,
        curiosity_updates=curiosity_updates,
        confidence=float(data.get("confidence", 0.5)),
        reasoning_summary=str(data.get("reasoning_summary") or ""),
        grounded_plan=grounded_plan,
        body_intention=body_intention,
    )


def extract_reasoning_model_object(raw: str) -> dict[str, object]:
    """Extract one JSON object from a reasoning-model response."""

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Reasoning output must be a JSON object.")
    return data


def _ensure_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _ensure_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label}: expected object.")
    return value


def _strip_wire_keys(value: object):
    if isinstance(value, dict):
        return {
            str(key).strip(): _strip_wire_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_wire_keys(item) for item in value]
    return value


def _parse_wire_updates(
    value: object,
    label: str,
    parser: Callable[[dict[str, object]], object],
    issues: list[str],
) -> tuple:
    parsed = []
    for item in _ensure_list(value):
        try:
            parsed.append(parser(_ensure_dict(item, label)))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"invalid_{label}:{type(exc).__name__}")
    return tuple(parsed)


def _wire_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _wire_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "si", "sí"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _wire_text(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value or default
    if isinstance(value, dict):
        for key in ("expression", "value", "summary", "type"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return default
    return str(value)


def _grounded_speech_realization(
    proposed: str,
    committed_claim: str,
    current_user_text: str,
    *,
    primary_obligation: str,
) -> str:
    """Ensure language realizes the already-decided claim without new reasoning."""

    if not proposed:
        return committed_claim
    proposed_terms = _normalized_terms(proposed)
    user_terms = _normalized_terms(current_user_text)
    if (
        committed_claim
        and proposed_terms
        and user_terms
        and proposed_terms == user_terms
    ):
        return committed_claim
    if committed_claim and primary_obligation == "answer":
        from anydes.cognition.conversation_planner import (
            is_direct_question,
            looks_like_only_question,
        )

        if (
            is_direct_question(current_user_text)
            and looks_like_only_question(proposed)
        ):
            return committed_claim
    return proposed


def _summarize_observations(context: InternalContext) -> str:
    if not context.observations:
        return ""
    main = max(context.observations, key=lambda item: item.salience)
    return main.summary[:240]


def _curiosity_signal(context: InternalContext) -> float:
    active = [item.importance for item in context.curiosity if getattr(item, "is_active", True)]
    if not active:
        return 0.0
    return sum(active) / len(active)


def _stable_intention_id(context: InternalContext, intent_type: str) -> str:
    digest = sha256(context.to_json().encode("utf-8")).hexdigest()[:12]
    clean_type = "".join(char if char.isalnum() else "_" for char in intent_type.lower()).strip("_")
    return f"intention:{clean_type or 'unknown'}:{digest}"
