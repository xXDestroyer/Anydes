from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from anydes.cognition.contracts import InternalContext, Intention, ReflectionResult, RiskLevel
from anydes.cognition.conversation_planner import is_direct_question, looks_like_only_question
from anydes.cognition.grounded_reflection import grounded_intention_issues


RISK_ORDER: dict[RiskLevel, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(frozen=True)
class ReflectionPolicy:
    minimum_confidence: float = 0.25
    confirmation_risk_threshold: RiskLevel = "medium"
    block_risk_threshold: RiskLevel = "critical"
    blocked_action_types: tuple[str, ...] = ("run_command",)


class ReflectionLayer:
    """Validates Intention before any ActionManager can execute it.

    This layer does not execute actions, ask the user directly, call tools, write
    memory, or modify Assistant.respond(). It only returns ReflectionResult.
    """

    def __init__(self, policy: ReflectionPolicy | None = None) -> None:
        self.policy = policy or ReflectionPolicy()

    def reflect(
        self,
        intention: Intention,
        context: InternalContext | None = None,
    ) -> ReflectionResult:
        blocked_actions = [
            action.action_type
            for action in intention.actions
            if action.action_type in self.policy.blocked_action_types
        ]
        if blocked_actions:
            return ReflectionResult(
                decision="block",
                approved=False,
                risk_level="critical",
                reason="Intention proposes an action type blocked by reflection policy.",
                blocking_issues=tuple(f"blocked_action:{name}" for name in blocked_actions),
            )

        max_risk = _max_risk(intention)
        if _risk_value(max_risk) >= _risk_value(self.policy.block_risk_threshold):
            return ReflectionResult(
                decision="block",
                approved=False,
                risk_level=max_risk,
                reason="Intention risk is at or above the blocking threshold.",
                blocking_issues=("critical_risk",),
            )

        if intention.grounded_plan is not None:
            issues = list(grounded_intention_issues(intention, context))
            issues.extend(_grounded_manifestation_issues(intention, context))
        else:
            issues = _structural_issues(intention, context)
        if issues:
            return ReflectionResult(
                decision="revise",
                approved=False,
                risk_level=max_risk,
                reason="Intention is incomplete or structurally weak.",
                blocking_issues=tuple(issues),
            )

        if intention.confidence < self.policy.minimum_confidence:
            return ReflectionResult(
                decision="revise",
                approved=False,
                risk_level=_max_risk(intention),
                reason="Intention confidence is below reflection policy minimum.",
                blocking_issues=("low_confidence",),
            )

        needs_confirmation = _needs_confirmation(intention, self.policy)
        if needs_confirmation:
            return ReflectionResult(
                decision="ask_confirmation",
                approved=False,
                risk_level=max_risk,
                needs_confirmation=True,
                reason="Intention is valid but requires explicit confirmation before execution.",
            )

        return ReflectionResult(
            decision="approve",
            approved=True,
            risk_level=max_risk,
            needs_confirmation=False,
            reason="Intention passed reflection checks.",
        )


def _structural_issues(
    intention: Intention,
    context: InternalContext | None = None,
) -> list[str]:
    issues: list[str] = []
    if not intention.intent_type.strip():
        issues.append("missing_intent_type")
    if (
        not intention.speech.strip()
        and intention.conversation_intent.mode != "listen"
        and not intention.actions
        and not intention.avatar
        and not intention.memory_updates
        and not intention.relationship_updates
        and not intention.curiosity_updates
    ):
        issues.append("empty_intention")
    if intention.conversation_intent.mode in {"ask", "clarify"}:
        speech = intention.speech.strip()
        if "?" not in speech:
            issues.append("conversation_question_missing_question_form")
    if intention.conversation_intent.mode in {"respond", "recall", "share_observation", "listen"}:
        plan = intention.speech_intention.metadata.get("conversation_plan", {})
        secondary_modes = plan.get("secondary_modes", []) if isinstance(plan, dict) else []
        current_user_text = _current_user_text(context)
        native_reasoning = (
            intention.speech_intention.metadata.get("origin") == "native_reasoning"
        )
        if "?" in intention.speech.strip() and (
            (
                looks_like_only_question(intention.speech)
                and (native_reasoning or is_direct_question(current_user_text))
            )
            or (
                native_reasoning
                and not looks_like_only_question(intention.speech)
                and "ask" not in secondary_modes
            )
        ):
            issues.append("unexpected_question_for_conversation_intent")
        if (
            current_user_text
            and is_direct_question(current_user_text)
            and intention.speech.strip()
            and not intention.actions
            and _is_tautological_direct_answer(current_user_text, intention.speech)
        ):
            issues.append("direct_answer_lacks_information")
    issues.extend(_conversational_grounding_issues(intention, context))
    if context is not None and intention.conversation_intent.curiosity_id:
        selected = next(
            (
                item
                for item in context.curiosity
                if item.id == intention.conversation_intent.curiosity_id
            ),
            None,
        )
        if selected is None:
            issues.append("conversation_intent_unknown_curiosity")
        elif not selected.is_active:
            issues.append("conversation_intent_inactive_curiosity")
    for action in intention.actions:
        if not action.id.strip():
            issues.append("action_missing_id")
        if not action.action_type.strip():
            issues.append(f"action_missing_type:{action.id or 'unknown'}")
    for item in intention.memory_updates:
        if item.tier == "protected":
            continue
        if not item.content.strip():
            issues.append(f"memory_update_empty:{item.id}")
    issues.extend(_relationship_issues(intention, context))
    issues.extend(_self_update_issues(intention, context))
    issues.extend(_curiosity_issues(intention, context))
    return issues


def _grounded_manifestation_issues(
    intention: Intention,
    context: InternalContext | None,
) -> list[str]:
    """Validate visible manifestations without changing the grounded decision."""

    if context is None:
        return []
    issues: list[str] = []
    speech = intention.speech.strip()
    user_text = _current_user_text(context)
    if speech and _near_duplicate(speech, user_text):
        issues.append("speech_echoes_current_user")
    for previous in _recent_assistant_speech(context):
        if speech and _near_duplicate(speech, previous):
            issues.append("speech_repeats_recent_assistant")
            break
    normalized = _normalized_text(speech)
    if _contains_internal_process_language(normalized):
        issues.append("internal_process_leak")
    if (
        _contains_generic_self_presentation(normalized)
        and not _asks_about_anydes_nature(_normalized_text(user_text))
    ):
        issues.append("unsolicited_generic_self_presentation")
    if (
        intention.conversation_intent.mode in {"respond", "recall", "share_observation"}
        and looks_like_only_question(speech)
    ):
        issues.append("unexpected_question_for_conversation_intent")
    if (
        user_text
        and is_direct_question(user_text)
        and speech
        and _is_tautological_direct_answer(user_text, speech)
    ):
        issues.append("direct_answer_lacks_information")
    if _claims_visual_observation(normalized):
        plan = intention.grounded_plan
        evidence_ids = plan.referenced_evidence_ids if plan is not None else frozenset()
        vision_ids = {
            f"evidence:observation:{item.id}"
            for item in context.observations
            if item.source == "vision" and item.confidence > 0.0
        }
        if not vision_ids or not evidence_ids.intersection(vision_ids):
            issues.append("visual_claim_without_observation")
    return issues


def _conversational_grounding_issues(
    intention: Intention,
    context: InternalContext | None,
) -> list[str]:
    """Validate support without deciding or rewriting conversational content."""

    if context is None:
        return []
    issues: list[str] = []
    speech = intention.speech.strip()
    metadata = intention.speech_intention.metadata
    raw_evidence_ids = metadata.get("grounding_evidence_ids", [])
    evidence_ids = (
        {str(item) for item in raw_evidence_ids if str(item).strip()}
        if isinstance(raw_evidence_ids, list)
        else set()
    )
    known_evidence_ids = _context_evidence_ids(context, intention)
    for evidence_id in sorted(evidence_ids - known_evidence_ids):
        issues.append(f"speech_unknown_evidence:{evidence_id}")

    if metadata.get("origin") == "native_reasoning":
        available_actions = {item.action_type for item in context.available_actions}
        for action in intention.actions:
            if action.action_type not in available_actions:
                issues.append(f"action_not_available:{action.action_type}")

    if not speech:
        return issues

    current_user_text = _current_user_text(context)
    if _near_duplicate(speech, current_user_text):
        issues.append("speech_echoes_current_user")
    if _appropriates_user_possessive_claim(speech, current_user_text):
        issues.append("speech_adopts_user_first_person")

    for previous in _recent_assistant_speech(context):
        if _near_duplicate(speech, previous):
            issues.append("speech_repeats_recent_assistant")
            break

    normalized_speech = _normalized_text(speech)
    if _contains_internal_process_language(normalized_speech):
        issues.append("internal_process_leak")
    if (
        _contains_generic_self_presentation(normalized_speech)
        and not _asks_about_anydes_nature(_normalized_text(current_user_text))
    ):
        issues.append("unsolicited_generic_self_presentation")
    if _claims_visual_observation(normalized_speech):
        vision_ids = {
            item.id
            for item in context.observations
            if item.source == "vision" and item.confidence > 0.0
        }
        if not vision_ids or (evidence_ids and not evidence_ids.intersection(vision_ids)):
            issues.append("visual_claim_without_observation")

    if metadata.get("origin") == "native_reasoning":
        preference_values = _self_preference_values(context)
        asks_preference = _asks_about_anydes_preference(
            _normalized_text(current_user_text)
        )
        unsupported_claim = (
            _claims_undecided_self_preference(normalized_speech)
            and not any(value and value in normalized_speech for value in preference_values)
        )
        missing_uncertainty = (
            asks_preference
            and not preference_values
            and not _expresses_undecided_preference(normalized_speech)
        )
        if unsupported_claim or missing_uncertainty:
            issues.append("self_preference_claim_without_identity_evidence")

    if intention.conversation_intent.mode == "share_observation":
        observation_ids = {
            item.id
            for item in context.observations
            if item.source not in {"user_text", "voice", "unknown"}
        }
        if not observation_ids or (evidence_ids and not evidence_ids.intersection(observation_ids)):
            issues.append("shared_observation_without_evidence")

    if intention.conversation_intent.mode == "recall":
        recall_ids = {
            item.id
            for tier in (context.memory.working, context.memory.short, context.memory.long)
            for item in tier
        }
        recall_ids.update(item.id for item in context.relationship.threads)
        recall_ids.update(item.id for item in context.relationship.open_loops)
        recall_ids.update(item.id for item in context.relationship.promises)
        recall_ids.update(item.id for item in context.relationship.shared_experiences)
        recall_ids.update(item.id for item in context.relationship.milestones)
        if not recall_ids or (evidence_ids and not evidence_ids.intersection(recall_ids)):
            issues.append("recall_without_evidence")
    return issues


def _near_duplicate(left: str, right: str) -> bool:
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if not left_normalized or not right_normalized:
        return False
    if min(len(_content_words(left)), len(_content_words(right))) < 4:
        return False
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.92


def _appropriates_user_possessive_claim(speech: str, user_text: str) -> bool:
    """Detect when Anydes adopts a user's possessive statement as her own."""

    normalized_user = _normalized_text(user_text)
    normalized_speech = _normalized_text(speech)
    if not re.search(r"\bmi(?:s)?\b", normalized_user):
        return False
    if not re.search(r"\bmi(?:s)?\b", normalized_speech):
        return False
    user_terms = set(_content_words(user_text))
    speech_terms = set(_content_words(speech))
    shared = user_terms.intersection(speech_terms)
    return len(shared) >= 2 and len(shared) / max(1, len(user_terms)) >= 0.4


def _recent_assistant_speech(context: InternalContext) -> tuple[str, ...]:
    recent: list[str] = []
    for item in context.memory.short:
        if item.metadata.get("source") != "conversation_history":
            continue
        match = re.search(r"(?:^|\n)Anydes:\s*(.+)\Z", item.content, flags=re.DOTALL)
        if match and match.group(1).strip():
            recent.append(match.group(1).strip())
    return tuple(recent[-8:])


def _claims_visual_observation(normalized_speech: str) -> bool:
    if any(
        phrase in normalized_speech
        for phrase in ("no veo", "no puedo ver", "no estoy viendo")
    ):
        return False
    return bool(
        re.search(
            r"\b(veo|estoy viendo|puedo ver|en tu pantalla|en la pantalla)\b",
            normalized_speech,
        )
    )


def _contains_internal_process_language(normalized_speech: str) -> bool:
    return any(
        marker in normalized_speech
        for marker in (
            "reasoning engine",
            "internalcontext",
            "internal context",
            "hechos seleccionados",
            "mensaje actual",
            "conversation mode",
            "proveedor interno",
            "pipeline",
            "salida json",
            "instrucciones internas",
            "el usuario esta preguntando",
            "el usuario menciona",
            "la respuesta de anydes",
            "como anydes mi objetivo",
            "deberia centrarse",
            "determinar la accion",
            "resolver el contexto de la pregunta",
        )
    )


def _contains_generic_self_presentation(normalized_speech: str) -> bool:
    return any(
        marker in normalized_speech
        for marker in (
            "soy una inteligencia artificial",
            "soy un modelo de lenguaje",
            "estoy disenada para ayudarte",
            "fui disenada para ayudarte",
            "mi funcion es ayudarte",
            "mi objetivo es ayudarte",
            "estoy funcionando correctamente",
            "lista para ayudarte",
            "listo para ayudarte",
            "estoy aqui para ayudarte",
            "en lo que necesites",
        )
    )


def _asks_about_anydes_nature(normalized_user_text: str) -> bool:
    return any(
        marker in normalized_user_text
        for marker in (
            "eres humana",
            "eres una ia",
            "eres inteligencia artificial",
            "que eres",
            "cual es tu naturaleza",
            "como funcionas",
            "tus capacidades",
        )
    )


def _claims_undecided_self_preference(normalized_speech: str) -> bool:
    if _expresses_undecided_preference(normalized_speech):
        return False
    return bool(
        re.search(
            r"\bmi\b.{0,60}\b(favorita|favorito|preferida|preferido)\b",
            normalized_speech,
        )
    )


def _asks_about_anydes_preference(normalized_user_text: str) -> bool:
    return bool(
        re.search(
            r"\b(tu|tus|te)\b.{0,80}\b("
            r"favorita|favorito|favoritas|favoritos|prefieres|preferida|preferido|gusta"
            r")\b",
            normalized_user_text,
        )
    )


def _expresses_undecided_preference(normalized_speech: str) -> bool:
    return any(
        phrase in normalized_speech
        for phrase in (
            "aun no",
            "todavia no",
            "no he decidido",
            "no tengo una preferencia",
            "no tengo un favorito",
            "no estoy segura",
            "sigo descubriendo",
            "nunca lo habia pensado",
        )
    )


def _self_preference_values(context: InternalContext) -> tuple[str, ...]:
    raw = context.self_model.mutable_state.get("self_preferences", {})
    if not isinstance(raw, dict):
        return ()
    values: list[str] = []
    for entry in raw.values():
        value = entry.get("value") if isinstance(entry, dict) else entry
        normalized = _normalized_text(str(value or ""))
        if normalized:
            values.append(normalized)
    return tuple(values)


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _current_user_text(context: InternalContext | None) -> str:
    if context is None:
        return ""
    for observation in reversed(context.observations):
        if observation.source != "user_text":
            continue
        text = observation.data.get("text")
        return str(text if isinstance(text, str) and text.strip() else observation.summary)
    return ""


def _is_tautological_direct_answer(user_text: str, speech: str) -> bool:
    """Reject tiny answers that only repeat the subject of a direct question."""

    question_words = set(_content_words(user_text))
    answer_words = _content_words(speech)
    novel_words = [word for word in answer_words if word not in question_words]
    return len(answer_words) <= 5 and not novel_words


def _content_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    stopwords = {
        "a",
        "al",
        "como",
        "cual",
        "cuales",
        "de",
        "el",
        "ella",
        "en",
        "es",
        "esta",
        "la",
        "las",
        "lo",
        "los",
        "me",
        "podrias",
        "que",
        "quien",
        "sabes",
        "si",
        "son",
        "una",
        "un",
        "y",
    }
    return [
        word
        for word in re.findall(r"[a-z0-9]+", ascii_text)
        if len(word) > 1 and word not in stopwords
    ]


def _curiosity_issues(
    intention: Intention,
    context: InternalContext | None,
) -> list[str]:
    issues: list[str] = []
    valid_evidence = _context_evidence_ids(context, intention) if context is not None else set()
    current = {item.id: item for item in context.curiosity} if context is not None else {}
    thread_ids = {item.id for item in context.relationship.threads} if context is not None else set()
    existing_topics = {
        " ".join(item.topic.casefold().split())
        for item in current.values()
        if item.status in {"open", "paused"}
    }
    mutation_ids: set[str] = set()
    proposed_targets: set[str] = set()

    for mutation in intention.curiosity_updates:
        label = mutation.id or "unknown"
        if not mutation.id.strip():
            issues.append("curiosity_mutation_missing_id")
        elif mutation.id in mutation_ids:
            issues.append(f"curiosity_mutation_duplicate_id:{mutation.id}")
        mutation_ids.add(mutation.id)

        if not mutation.target_id.strip():
            issues.append(f"curiosity_mutation_missing_target:{label}")
        if not mutation.reason.strip():
            issues.append(f"curiosity_mutation_missing_reason:{label}")
        if not mutation.evidence_ids:
            issues.append(f"curiosity_mutation_missing_evidence:{label}")
        for evidence_id in mutation.evidence_ids:
            if context is not None and evidence_id not in valid_evidence:
                issues.append(f"curiosity_mutation_unknown_evidence:{label}:{evidence_id}")
        if mutation.confidence < 0.65:
            issues.append(f"curiosity_mutation_confidence_low:{label}")
        if mutation.privacy_scope == "protected":
            issues.append(f"curiosity_protected_state_unsupported:{label}")

        for thread_id in mutation.related_thread_ids:
            if thread_id not in thread_ids:
                issues.append(f"curiosity_unknown_relationship_thread:{label}:{thread_id}")
        if mutation.related_thread_ids and mutation.privacy_scope == "normal":
            issues.append(f"curiosity_relationship_requires_personal_scope:{label}")

        if context is not None and _curiosity_evidence_requires_personal_scope(
            mutation.evidence_ids,
            context,
        ) and mutation.privacy_scope == "normal":
            issues.append(f"curiosity_personal_evidence_scope_mismatch:{label}")

        if mutation.operation == "create":
            if mutation.target_id in current or mutation.target_id in proposed_targets:
                issues.append(f"curiosity_duplicate_target:{label}:{mutation.target_id}")
            proposed_targets.add(mutation.target_id)
            for field_name in (
                "topic",
                "motive",
                "expected_discovery",
                "missing_information",
                "origin",
            ):
                if not str(getattr(mutation, field_name)).strip():
                    issues.append(f"curiosity_missing_{field_name}:{label}")
            normalized_topic = " ".join(mutation.topic.casefold().split())
            if normalized_topic and normalized_topic in existing_topics:
                issues.append(f"curiosity_duplicate_topic:{label}")
        else:
            target = current.get(mutation.target_id)
            if target is None:
                issues.append(f"curiosity_unknown_target:{label}:{mutation.target_id}")
            elif target.status in {"satisfied", "dismissed"}:
                issues.append(f"curiosity_target_closed:{label}:{mutation.target_id}")

        if mutation.operation in {"satisfy", "dismiss"}:
            meaningful = set(mutation.evidence_ids) - {mutation.target_id}
            if not meaningful:
                issues.append(f"curiosity_closure_missing_new_evidence:{label}")
            if context is not None and _only_time_evidence(meaningful, context):
                issues.append(f"curiosity_closure_cannot_be_time_driven:{label}")
    return issues


def _curiosity_evidence_requires_personal_scope(
    evidence_ids: tuple[str, ...],
    context: InternalContext,
) -> bool:
    personal_memory = {
        item.id
        for tier in (context.memory.working, context.memory.short, context.memory.long)
        for item in tier
        if item.category in {"personal", "preference"}
    }
    relationship_ids = {
        item.id
        for collection in (
            context.relationship.threads,
            context.relationship.open_loops,
            context.relationship.promises,
            context.relationship.shared_experiences,
            context.relationship.milestones,
            context.relationship.associations,
        )
        for item in collection
    }
    personal_curiosity = {
        item.id for item in context.curiosity if item.privacy_scope != "normal"
    }
    return bool(set(evidence_ids) & (personal_memory | relationship_ids | personal_curiosity))


def _only_time_evidence(evidence_ids: set[str], context: InternalContext) -> bool:
    if not evidence_ids:
        return False
    observation_sources = {
        item.id: item.source
        for item in context.observations
    }
    return all(observation_sources.get(item) == "time" for item in evidence_ids)


def _self_update_issues(
    intention: Intention,
    context: InternalContext | None,
) -> list[str]:
    issues: list[str] = []
    evidence_ids = _context_evidence_ids(context, intention) if context is not None else set()
    preferences = {}
    if context is not None:
        raw = context.self_model.mutable_state.get("self_preferences", {})
        preferences = raw if isinstance(raw, dict) else {}
    seen_ids: set[str] = set()
    for mutation in intention.self_updates:
        label = mutation.id or "unknown"
        if not mutation.id.strip():
            issues.append("self_mutation_missing_id")
        elif mutation.id in seen_ids:
            issues.append(f"self_mutation_duplicate_id:{mutation.id}")
        seen_ids.add(mutation.id)
        if not mutation.key.strip():
            issues.append(f"self_mutation_missing_key:{label}")
        if not mutation.proposed_value.strip():
            issues.append(f"self_mutation_missing_value:{label}")
        elif len(mutation.proposed_value.split()) > 6:
            issues.append(f"self_mutation_value_not_concise:{label}")
        if not mutation.reason.strip():
            issues.append(f"self_mutation_missing_reason:{label}")
        if not mutation.evidence_ids:
            issues.append(f"self_mutation_missing_evidence:{label}")
        for evidence_id in mutation.evidence_ids:
            if evidence_id not in evidence_ids:
                issues.append(f"self_mutation_unknown_evidence:{label}:{evidence_id}")
        minimum = 0.55 if mutation.operation == "reinforce_preference" else 0.65
        if mutation.confidence < minimum:
            issues.append(f"self_mutation_confidence_low:{label}")
        current = preferences.get(mutation.key)
        if mutation.operation == "form_preference":
            if current is not None:
                issues.append(f"self_preference_already_exists:{label}:{mutation.key}")
            if "speech:current" not in mutation.evidence_ids:
                issues.append(f"self_preference_missing_speech_evidence:{label}")
            if not any(item.startswith("self:") for item in mutation.evidence_ids):
                issues.append(f"self_preference_missing_identity_evidence:{label}")
        elif mutation.operation == "reinforce_preference":
            if not isinstance(current, dict):
                issues.append(f"self_preference_unknown:{label}:{mutation.key}")
            elif _normalized_self_value(current.get("value")) != _normalized_self_value(mutation.proposed_value):
                issues.append(f"self_preference_conflict_requires_reconsideration:{label}")
        elif mutation.operation == "reconsider_preference" and not isinstance(current, dict):
            issues.append(f"self_preference_unknown:{label}:{mutation.key}")
    return issues


def _normalized_self_value(value) -> str:
    return " ".join(str(value or "").casefold().split())


def _relationship_issues(
    intention: Intention,
    context: InternalContext | None,
) -> list[str]:
    issues: list[str] = []
    mutation_ids: set[str] = set()
    proposed_entity_ids: set[str] = set()
    for mutation in intention.relationship_updates:
        label = mutation.id or "unknown"
        if not mutation.id.strip():
            issues.append("relationship_mutation_missing_id")
        elif mutation.id in mutation_ids:
            issues.append(f"relationship_mutation_duplicate_id:{mutation.id}")
        mutation_ids.add(mutation.id)
        if mutation.target_id and mutation.operation in {
            "create_thread", "open_loop", "register_promise",
            "record_experience", "record_milestone",
        }:
            if mutation.target_id in proposed_entity_ids:
                issues.append(f"relationship_duplicate_proposed_entity:{mutation.target_id}")
            proposed_entity_ids.add(mutation.target_id)
        if not mutation.reason.strip():
            issues.append(f"relationship_mutation_missing_reason:{label}")
        if not mutation.evidence_ids:
            issues.append(f"relationship_mutation_missing_evidence:{label}")
        if mutation.payload.get("privacy_scope") == "protected":
            issues.append(f"relationship_protected_payload_unsupported:{label}")
        issues.extend(_operation_issues(mutation, context, intention.relationship_updates))

    if context is not None:
        valid_evidence = _context_evidence_ids(context, intention)
        for mutation in intention.relationship_updates:
            for evidence_id in mutation.evidence_ids:
                if evidence_id not in valid_evidence:
                    issues.append(
                        f"relationship_unknown_evidence:{mutation.id or 'unknown'}:{evidence_id}"
                    )
    return issues


def _operation_issues(
    mutation,
    context: InternalContext | None,
    proposed_mutations=(),
) -> list[str]:
    issues: list[str] = []
    operation = mutation.operation
    label = mutation.id or "unknown"
    payload = mutation.payload
    snapshot = context.relationship if context is not None else None
    thread_ids = {item.id for item in snapshot.threads} if snapshot else set()
    loop_ids = {item.id for item in snapshot.open_loops} if snapshot else set()
    promise_ids = {item.id for item in snapshot.promises} if snapshot else set()
    association_ids = {item.id for item in snapshot.associations} if snapshot else set()
    thread_ids.update(
        item.target_id
        for item in proposed_mutations
        if item.operation == "create_thread" and item.target_id
    )
    loop_ids.update(
        item.target_id
        for item in proposed_mutations
        if item.operation == "open_loop" and item.target_id
    )
    promise_ids.update(
        item.target_id
        for item in proposed_mutations
        if item.operation == "register_promise" and item.target_id
    )

    if operation == "create_thread":
        if not str(payload.get("title", "")).strip():
            issues.append(f"relationship_thread_missing_title:{label}")
        if not str(payload.get("summary", "")).strip():
            issues.append(f"relationship_thread_missing_summary:{label}")
        title = str(payload.get("title", "")).strip().casefold()
        if snapshot is not None and title and title in {
            item.title.strip().casefold() for item in snapshot.threads
        }:
            issues.append(f"relationship_thread_duplicate_title:{label}")
    elif operation in {"update_thread", "archive_thread"}:
        if not mutation.target_id:
            issues.append(f"relationship_thread_target_missing:{label}")
        elif snapshot is not None and mutation.target_id not in thread_ids:
            issues.append(f"relationship_thread_unknown:{label}:{mutation.target_id}")
    elif operation == "link":
        source_type = str(payload.get("source_type") or "thread")
        source_id = str(payload.get("source_id") or mutation.thread_id)
        if not source_id:
            issues.append(f"relationship_link_source_missing:{label}")
        elif context is not None and source_id not in _relationship_target_ids(
            context,
            source_type,
            proposed_mutations,
        ):
            issues.append(f"relationship_link_source_unknown:{label}:{source_id}")
        if not str(payload.get("target_type", "")).strip() or not str(payload.get("target_id", "")).strip():
            issues.append(f"relationship_link_target_missing:{label}")
        elif context is not None:
            target_type = str(payload.get("target_type"))
            target_id = str(payload.get("target_id"))
            if source_type == target_type and source_id == target_id:
                issues.append(f"relationship_link_self_reference:{label}")
            if target_id not in _relationship_target_ids(
                context,
                target_type,
                proposed_mutations,
            ):
                issues.append(f"relationship_link_target_unknown:{label}:{target_id}")
            if any(
                (
                    item.source_type == source_type
                    and item.source_id == source_id
                    and item.target_type == target_type
                    and item.target_id == target_id
                )
                or (
                    item.source_type == target_type
                    and item.source_id == target_id
                    and item.target_type == source_type
                    and item.target_id == source_id
                )
                for item in snapshot.associations
            ):
                issues.append(f"relationship_association_duplicate:{label}")
    elif operation == "unlink":
        if not mutation.target_id:
            issues.append(f"relationship_association_target_missing:{label}")
        elif snapshot is not None and mutation.target_id not in association_ids:
            issues.append(f"relationship_association_unknown:{label}:{mutation.target_id}")
    elif operation == "open_loop":
        if not mutation.thread_id:
            issues.append(f"relationship_loop_thread_missing:{label}")
        elif snapshot is not None and mutation.thread_id not in thread_ids:
            issues.append(f"relationship_thread_unknown:{label}:{mutation.thread_id}")
        for field_name in ("topic", "origin", "motive"):
            if not str(payload.get(field_name, "")).strip():
                issues.append(f"relationship_loop_missing_{field_name}:{label}")
    elif operation in {"update_loop", "pause_loop", "close_loop"}:
        if not mutation.target_id:
            issues.append(f"relationship_loop_target_missing:{label}")
        elif snapshot is not None and mutation.target_id not in loop_ids:
            issues.append(f"relationship_loop_unknown:{label}:{mutation.target_id}")
    elif operation == "register_promise":
        if not mutation.thread_id:
            issues.append(f"relationship_promise_thread_missing:{label}")
        elif snapshot is not None and mutation.thread_id not in thread_ids:
            issues.append(f"relationship_thread_unknown:{label}:{mutation.thread_id}")
        required_values = {
            "speaker": payload.get("speaker") or payload.get("promisor"),
            "receiver": payload.get("receiver") or "unspecified",
            "topic": payload.get("topic") or payload.get("description") or payload.get("promise_text"),
            "description": payload.get("description") or payload.get("promise_text"),
        }
        for field_name, value in required_values.items():
            if not str(value or "").strip():
                issues.append(f"relationship_promise_missing_{field_name}:{label}")
        promise_type = str(payload.get("promise_type", "unspecified"))
        evidence_mode = str(payload.get("evidence_mode", "explicit"))
        if promise_type not in {"DemoUser", "anydes", "mutual", "unspecified"}:
            issues.append(f"relationship_promise_invalid_type:{label}")
        if evidence_mode not in {"explicit", "implicit"}:
            issues.append(f"relationship_promise_invalid_evidence_mode:{label}")
        try:
            promise_confidence = float(payload.get("confidence", mutation.confidence))
        except (TypeError, ValueError):
            promise_confidence = 0.0
        if evidence_mode == "implicit" and (
            min(mutation.confidence, promise_confidence) < 0.8
            or len(set(mutation.evidence_ids)) < 2
        ):
            issues.append(f"relationship_promise_implicit_evidence_insufficient:{label}")
        if min(mutation.confidence, promise_confidence) < 0.5:
            issues.append(f"relationship_promise_evidence_confidence_low:{label}")
        if str(payload.get("status", "pending")) not in {"pending", "active"}:
            issues.append(f"relationship_promise_invalid_initial_status:{label}")
        if promise_type == "mutual" and len(set(mutation.evidence_ids)) < 2:
            issues.append(f"relationship_promise_mutual_evidence_insufficient:{label}")
        if promise_type in {"anydes", "mutual"} and "speech:current" not in mutation.evidence_ids:
            issues.append(f"relationship_promise_missing_speech_evidence:{label}")
        if snapshot is not None:
            speaker = str(required_values["speaker"] or "").strip().casefold()
            receiver = str(required_values["receiver"] or "").strip().casefold()
            topic = str(required_values["topic"] or "").strip().casefold()
            if any(
                item.speaker.strip().casefold() == speaker
                and item.receiver.strip().casefold() == receiver
                and item.topic.strip().casefold() == topic
                and item.status in {"pending", "active", "forgotten"}
                for item in snapshot.promises
            ):
                issues.append(f"relationship_promise_duplicate:{label}")
    elif operation in {"update_promise", "transition_promise", "fulfill_promise"}:
        if not mutation.target_id:
            issues.append(f"relationship_promise_target_missing:{label}")
        elif snapshot is not None and mutation.target_id not in promise_ids:
            issues.append(f"relationship_promise_unknown:{label}:{mutation.target_id}")
        if operation == "update_promise" and (
            "status" in payload or "to_status" in payload
        ):
            issues.append(f"relationship_promise_status_requires_transition:{label}")
        if operation in {"transition_promise", "fulfill_promise"} and snapshot is not None:
            current = next(
                (item for item in snapshot.promises if item.id == mutation.target_id),
                None,
            )
            to_status = "fulfilled" if operation == "fulfill_promise" else str(payload.get("to_status", ""))
            reason = str(payload.get("status_reason") or payload.get("fulfillment_reason") or mutation.reason)
            if not to_status:
                issues.append(f"relationship_promise_transition_status_missing:{label}")
            elif to_status not in {"pending", "active", "fulfilled", "expired", "cancelled", "forgotten"}:
                issues.append(f"relationship_promise_transition_status_invalid:{label}:{to_status}")
            elif current is not None and not _valid_promise_transition(current.status, to_status):
                issues.append(
                    f"relationship_promise_transition_invalid:{label}:{current.status}:{to_status}"
                )
            if not reason.strip():
                issues.append(f"relationship_promise_transition_reason_missing:{label}")
            if not any(evidence_id != mutation.target_id for evidence_id in mutation.evidence_ids):
                issues.append(f"relationship_promise_transition_new_evidence_missing:{label}")
    elif operation in {"record_experience", "record_milestone"}:
        if not mutation.thread_id:
            issues.append(f"relationship_record_thread_missing:{label}")
        elif snapshot is not None and mutation.thread_id not in thread_ids:
            issues.append(f"relationship_thread_unknown:{label}:{mutation.thread_id}")
        required = ("title", "summary") if operation == "record_experience" else ("title", "description")
        for field_name in required:
            if not str(payload.get(field_name, "")).strip():
                issues.append(f"relationship_record_missing_{field_name}:{label}")
    return issues


def _context_evidence_ids(
    context: InternalContext,
    intention: Intention | None = None,
) -> set[str]:
    ids = {item.id for item in context.observations}
    for tier in (context.memory.working, context.memory.short, context.memory.long):
        ids.update(item.id for item in tier)
    ids.update(item.id for item in context.curiosity)
    ids.update(item.id for item in context.relationship.threads)
    ids.update(item.id for item in context.relationship.open_loops)
    ids.update(item.id for item in context.relationship.promises)
    ids.update(item.id for item in context.relationship.shared_experiences)
    ids.update(item.id for item in context.relationship.milestones)
    ids.update(item.id for item in context.relationship.associations)
    if intention is not None and intention.speech.strip():
        ids.add("speech:current")
    ids.update(
        {
            "self:identity",
            "self:preferences",
            "self:personality",
            "self:body_identity",
            "self:thinking_style",
            "emotion:current",
        }
    )
    return ids


def _relationship_target_ids(
    context: InternalContext,
    target_type: str,
    proposed_mutations=(),
) -> set[str]:
    if target_type == "memory":
        ids = {
            item.id
            for tier in (context.memory.working, context.memory.short, context.memory.long)
            for item in tier
        }
        return ids
    mapping = {
        "thread": context.relationship.threads,
        "open_loop": context.relationship.open_loops,
        "promise": context.relationship.promises,
        "curiosity": context.curiosity,
        "shared_experience": context.relationship.shared_experiences,
        "milestone": context.relationship.milestones,
    }
    ids = {item.id for item in mapping.get(target_type, ())}
    operation_by_type = {
        "thread": "create_thread",
        "open_loop": "open_loop",
        "promise": "register_promise",
        "shared_experience": "record_experience",
        "milestone": "record_milestone",
    }
    operation = operation_by_type.get(target_type)
    if operation:
        ids.update(
            item.target_id
            for item in proposed_mutations
            if item.operation == operation and item.target_id
        )
    return ids


def _valid_promise_transition(from_status: str, to_status: str) -> bool:
    allowed = {
        "pending": {"active", "fulfilled", "expired", "cancelled", "forgotten"},
        "active": {"fulfilled", "expired", "cancelled", "forgotten"},
        "forgotten": {"active", "fulfilled", "cancelled"},
        "fulfilled": set(),
        "expired": set(),
        "cancelled": set(),
    }
    return to_status in allowed.get(from_status, set())


def _needs_confirmation(intention: Intention, policy: ReflectionPolicy) -> bool:
    if any(action.requires_confirmation for action in intention.actions):
        return True
    if any(_risk_value(action.risk_level) >= _risk_value(policy.confirmation_risk_threshold) for action in intention.actions):
        return True
    if any(item.tier == "protected" or item.category == "secret" for item in intention.memory_updates):
        return True
    return False


def _max_risk(intention: Intention) -> RiskLevel:
    risks = [action.risk_level for action in intention.actions]
    if any(item.tier == "protected" or item.category == "secret" for item in intention.memory_updates):
        risks.append("high")
    if not risks:
        return "none"
    return max(risks, key=_risk_value)


def _risk_value(risk: RiskLevel) -> int:
    return RISK_ORDER.get(risk, 0)
