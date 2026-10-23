from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from anydes.cognition.contracts import InternalContext, MemoryItem, Observation
from anydes.cognition.conversation_planner import is_direct_question
from anydes.cognition.grounding_contracts import (
    CapabilityEvidence,
    EvidenceBundle,
    EvidenceCoverage,
    EvidenceItem,
    EvidenceSelectionBudget,
    InteractionFrame,
    KnowledgeGap,
)


@dataclass(frozen=True)
class EvidenceSelectionPolicy:
    """Bounds evidence selection without making conversational decisions."""

    max_items: int = 18
    max_characters: int = 9000
    max_observations: int = 5
    max_memories: int = 5
    max_relationship_items: int = 5
    max_curiosities: int = 3
    max_goals: int = 3
    max_capabilities: int = 6
    minimum_relevance: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "max_items",
            "max_characters",
            "max_observations",
            "max_memories",
            "max_relationship_items",
            "max_curiosities",
            "max_goals",
            "max_capabilities",
        ):
            minimum = 256 if name == "max_characters" else 1
            object.__setattr__(self, name, max(minimum, int(getattr(self, name))))
        object.__setattr__(
            self,
            "minimum_relevance",
            max(0.0, min(1.0, float(self.minimum_relevance))),
        )


@dataclass
class EvidenceBundleBuilder:
    """Selects a compact, authorized evidence view from InternalContext.

    Selection is deterministic and provenance-preserving. It ranks existing
    facts; it never decides what Anydes should say or do.
    """

    policy: EvidenceSelectionPolicy = EvidenceSelectionPolicy()

    def build(self, context: InternalContext) -> EvidenceBundle:
        primary = _primary_observation(context.observations)
        frame = _interaction_frame(primary, context)
        query = (
            ""
            if frame.kind == "proactive_opportunity"
            else _observation_text(primary)
        )
        query_terms = set(_terms(query))

        candidates: list[tuple[float, int, EvidenceItem]] = []
        coverage: list[EvidenceCoverage] = []
        gaps: list[KnowledgeGap] = []
        order = 0

        def offer(item: EvidenceItem, *, score: float, required: bool = False) -> None:
            nonlocal order
            if not item.authorized:
                return
            rank = 2.0 if required else max(0.0, min(1.0, score))
            if required or rank >= self.policy.minimum_relevance:
                candidates.append((rank, order, item))
                order += 1

        observation_candidates = tuple(
            _observation_evidence(item, query_terms, primary)
            for item in context.observations
            if _observation_authorized(item)
            and not (
                item is primary
                and item.data.get("event") == "proactive_opportunity"
            )
        )
        selected_observations = _top(
            observation_candidates,
            self.policy.max_observations,
        )
        for score, item in selected_observations:
            offer(item, score=score, required=primary is not None and item.source_ref == primary.id)
        coverage.append(
            _coverage(
                "observation",
                len(context.observations),
                len(selected_observations),
                context.timestamp,
            )
        )

        self_items = _self_evidence(context, query_terms)
        preferences_are_empty = not _self_preferences(context)
        selected_self_ids: set[str] = set()
        for score, item in self_items:
            required = (
                preferences_are_empty
                and item.predicate == "established_preferences"
                and score >= self.policy.minimum_relevance
            )
            offer(
                item,
                score=score,
                required=required,
            )
            if required or score >= self.policy.minimum_relevance:
                selected_self_ids.add(item.id)
        coverage.append(
            _coverage(
                "self",
                len(self_items),
                len(selected_self_ids),
                context.self_model.updated_at,
            )
        )
        if (
            preferences_are_empty
            and query_terms
            and "evidence:self:preferences" in selected_self_ids
        ):
            gaps.append(
                KnowledgeGap(
                    id="gap:self:unestablished_preferences",
                    subject=context.self_model.name or "Anydes",
                    predicate="unlisted_preferences",
                    status="unknown",
                    reason="SELF no contiene un valor establecido para esa preferencia.",
                    consulted_sources=("self",),
                    related_evidence_ids=("evidence:self:preferences",),
                )
            )

        memory_candidates = _memory_evidence(context, query_terms)
        selected_memories = _top(memory_candidates, self.policy.max_memories)
        forced_working = 0
        forced_recent_conversation = 0
        forced_pending_input = 0
        for score, item in selected_memories:
            source_kind = str(item.attributes.get("memory_source", ""))
            required = False
            if item.attributes.get("tier") == "working" and forced_working < 2:
                required = True
                forced_working += 1
            if (
                source_kind == "conversation_history"
                and forced_recent_conversation < 2
            ):
                required = True
                forced_recent_conversation += 1
            if (
                source_kind == "pending_user_observation"
                and forced_pending_input < 2
            ):
                required = True
                forced_pending_input += 1
            offer(item, score=score, required=required)
        memory_total = sum(
            len(items)
            for items in (
                context.memory.working,
                context.memory.short,
                context.memory.long,
            )
        )
        coverage.append(
            _coverage(
                "memory",
                memory_total,
                len(selected_memories),
                _latest_memory_revision(context),
            )
        )

        relationship_candidates = _relationship_evidence(context, query_terms)
        selected_relationship = _top(
            relationship_candidates,
            self.policy.max_relationship_items,
        )
        for score, item in selected_relationship:
            offer(item, score=score)
        coverage.append(
            _coverage(
                "relationship",
                len(relationship_candidates),
                len(selected_relationship),
                str(context.relationship.revision),
                available=context.relationship.available,
            )
        )

        curiosity_candidates = _curiosity_evidence(context, query_terms)
        selected_curiosities = _top(
            curiosity_candidates,
            self.policy.max_curiosities,
        )
        for score, item in selected_curiosities:
            offer(item, score=score)
        coverage.append(
            _coverage(
                "curiosity",
                len(curiosity_candidates),
                len(selected_curiosities),
                context.timestamp,
            )
        )

        goal_candidates = _goal_evidence(context, query_terms)
        selected_goals = _top(goal_candidates, self.policy.max_goals)
        for score, item in selected_goals:
            offer(item, score=score)
        coverage.append(
            _coverage(
                "goal",
                len(goal_candidates),
                len(selected_goals),
                context.timestamp,
            )
        )

        for score, item in _state_evidence(context, query_terms):
            offer(item, score=score)

        capabilities = _capability_evidence(
            context,
            query_terms,
            self.policy.max_capabilities,
        )
        supporting_size = sum(
            len(json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":")))
            for item in (*gaps, *capabilities, *coverage)
        )
        selected, item_characters, truncated = _apply_budget(
            candidates,
            max_items=self.policy.max_items,
            max_characters=max(256, self.policy.max_characters - supporting_size),
        )
        estimated_characters = item_characters + supporting_size
        truncated = truncated or len(capabilities) < len(context.available_actions)
        revision = _context_revision(context, selected)
        return EvidenceBundle(
            id=f"evidence-bundle:{revision[:16]}",
            context_revision=revision,
            interaction=frame,
            items=selected,
            knowledge_gaps=tuple(gaps),
            capabilities=capabilities,
            coverage=tuple(coverage),
            selection_budget=EvidenceSelectionBudget(
                max_items=self.policy.max_items,
                max_characters=self.policy.max_characters,
                selected_items=len(selected),
                estimated_characters=estimated_characters,
                truncated=truncated,
            ),
            extensions={
                "selection_strategy": "bounded_provenance_relevance_v1",
            },
        )


def _primary_observation(observations: tuple[Observation, ...]) -> Observation | None:
    for source in ("user_text", "voice", "tool"):
        for item in reversed(observations):
            if item.source == source:
                return item
    for item in reversed(observations):
        if item.data.get("event") == "proactive_opportunity":
            return item
    return max(observations, key=lambda item: item.salience, default=None)


def _interaction_frame(
    primary: Observation | None,
    context: InternalContext,
) -> InteractionFrame:
    if primary is None:
        kind = "system_event"
        channel = "system"
        response_expected = False
        silence_allowed = True
        observation_id = ""
    elif primary.source in {"user_text", "voice"}:
        kind = "user_turn"
        channel = "voice" if primary.source == "voice" else "typed"
        response_expected = True
        silence_allowed = bool(primary.data.get("silence_allowed", False))
        observation_id = primary.id
    elif primary.source == "tool":
        kind = "tool_result"
        channel = "tool"
        response_expected = True
        silence_allowed = False
        observation_id = primary.id
    elif primary.data.get("event") == "proactive_opportunity":
        kind = "proactive_opportunity"
        channel = "system"
        response_expected = False
        silence_allowed = True
        observation_id = primary.id
    else:
        kind = "system_event"
        channel = str(primary.source)
        response_expected = False
        silence_allowed = True
        observation_id = primary.id
    speaker_ref = (
        str(primary.data.get("speaker_ref", "")).strip()
        if primary
        else ""
    ) or _interlocutor_ref(context)
    return InteractionFrame(
        id=f"interaction:{observation_id or _digest(context.timestamp)[:12]}",
        kind=kind,
        primary_observation_id=observation_id,
        speaker_ref=speaker_ref,
        channel=channel,
        response_expected=response_expected,
        silence_allowed=silence_allowed,
        constraints=tuple(
            constraint
            for constraint, enabled in (
                (
                    "direct_answer_required",
                    primary is not None
                    and primary.source in {"user_text", "voice"}
                    and is_direct_question(_observation_text(primary)),
                ),
                (
                    "proactive_requires_grounded_trigger",
                    kind == "proactive_opportunity",
                ),
            )
            if enabled
        ),
        occurred_at=primary.timestamp if primary else context.timestamp,
        metadata={
            "first_person_ref": speaker_ref,
            "second_person_ref": context.self_model.name or "Anydes",
            "utterance_form": (
                "question"
                if primary is not None
                and primary.source in {"user_text", "voice"}
                and is_direct_question(_observation_text(primary))
                else "statement"
            ),
        },
    )


def _interlocutor_ref(context: InternalContext) -> str:
    for observation in reversed(context.observations):
        if observation.source != "interlocutor":
            continue
        state = str(observation.data.get("state", "")).strip()
        if state:
            return state
    return "unknown"


def _observation_evidence(
    observation: Observation,
    query_terms: set[str],
    primary: Observation | None,
) -> tuple[float, EvidenceItem]:
    text = _observation_text(observation)
    score = 1.0 if primary is observation else _relevance(query_terms, text, observation.salience)
    if observation.source == "interlocutor":
        score = max(score, 0.55)
    source = {
        "user_text": "user",
        "voice": "user",
        "tool": "tool",
        "environment": "environment",
        "system": "system",
    }.get(str(observation.source), "observation")
    if source == "user":
        evidence_role = (
            "request"
            if is_direct_question(text)
            else "user_report"
        )
    elif source == "tool":
        evidence_role = "tool_result"
    elif source in {"environment", "observation"}:
        evidence_role = "observation"
    else:
        evidence_role = "event"
    epistemic_use = {
        "request": "interaction_request",
        "user_report": "speaker_report",
        "tool_result": "factual_result",
        "observation": "observed_state",
        "event": "event_context",
    }[evidence_role]
    return score, EvidenceItem(
        id=f"evidence:observation:{observation.id}",
        source=source,
        source_ref=observation.id,
        subject="user" if source == "user" else str(observation.source),
        predicate="current_observation",
        value=_bounded_value(observation.data),
        summary=_bounded_text(observation.summary, 800),
        confidence=observation.confidence,
        relevance=score,
        observed_at=observation.timestamp,
        privacy_scope=_privacy(observation.data.get("privacy_scope")),
        authorized=True,
        modality="text" if source == "user" else str(observation.source),
        attributes={
            "salience": observation.salience,
            "evidence_role": evidence_role,
            "epistemic_use": epistemic_use,
        },
    )


def _self_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    identity = {
        "name": context.self_model.name,
        "kind": context.self_model.kind,
        "identity_summary": _bounded_text(context.self_model.identity_summary, 420),
        "personality_traits": list(context.self_model.personality_traits[:8]),
        "boundaries": list(context.self_model.boundaries[:6]),
        "thinking_style": _thinking_style(context),
    }
    preferences = _self_preferences(context)
    preference_concepts = (
        "preferencia preferencias favorito favorita favoritos favoritas "
        "prefiere prefieres gusta gustos"
    )
    return (
        (
            _relevance(query_terms, json.dumps(identity, ensure_ascii=False), 0.72),
            EvidenceItem(
                id="evidence:self:identity",
                source="self",
                source_ref="self_model",
                subject=context.self_model.name or "Anydes",
                predicate="identity_core",
                value=_bounded_value(identity),
                summary="Established identity, personality, boundaries and thinking style.",
                confidence=1.0,
                relevance=0.72,
                freshness="timeless",
                observed_at=context.self_model.updated_at,
                privacy_scope="personal",
                attributes={"evidence_role": "background"},
            ),
        ),
        (
            _relevance(
                query_terms,
                f"{preference_concepts} {json.dumps(preferences, ensure_ascii=False)}",
                0.2,
            ),
            EvidenceItem(
                id="evidence:self:preferences",
                source="self",
                source_ref="self_model.mutable_state.self_preferences",
                subject=context.self_model.name or "Anydes",
                predicate="established_preferences",
                value=_bounded_value(preferences),
                summary=(
                    "Preferencias establecidas de Anydes."
                    if preferences
                    else "Anydes no tiene una preferencia establecida para temas no listados."
                ),
                confidence=1.0,
                relevance=0.42,
                freshness="timeless",
                observed_at=context.self_model.updated_at,
                privacy_scope="personal",
                attributes={
                    "evidence_role": (
                        "gap_support" if not preferences else "fact"
                    )
                },
            ),
        ),
    )


def _memory_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    candidates: list[tuple[float, EvidenceItem]] = []
    for tier_name, items in (
        ("working", context.memory.working),
        ("short", context.memory.short),
        ("long", context.memory.long),
    ):
        for memory in items:
            if memory.category == "secret" and not context.memory.protected_available:
                continue
            base = memory.importance * 0.46 + (0.34 if tier_name == "working" else 0.18)
            score = _relevance(query_terms, memory.content, base)
            memory_key = str(memory.metadata.get("key", "")).strip()
            candidates.append(
                (
                    score,
                    EvidenceItem(
                        id=f"evidence:memory:{memory.id}",
                        source="memory",
                        source_ref=memory.id,
                        subject=str(
                            memory.metadata.get(
                                "subject",
                                "current_user" if memory_key else "shared_history",
                            )
                        ),
                        predicate=str(
                            memory.metadata.get(
                                "predicate",
                                memory_key or memory.category,
                            )
                        ),
                        value=_bounded_text(memory.content, 900),
                        summary=_bounded_text(memory.content, 420),
                        confidence=memory.confidence,
                        relevance=score,
                        observed_at=memory.last_seen_at,
                        privacy_scope=_memory_privacy(memory),
                        modality="memory",
                        attributes={
                            "tier": memory.tier,
                            "category": memory.category,
                            "source_observation_id": memory.source_observation_id,
                            "memory_source": memory.metadata.get("source", ""),
                            "evidence_role": "fact",
                        },
                    ),
                )
            )
    return tuple(candidates)


def _relationship_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    groups = (
        ("thread", context.relationship.threads, 0.38),
        ("open_loop", context.relationship.open_loops, 0.28),
        ("promise", context.relationship.promises, 0.34),
        ("shared_experience", context.relationship.shared_experiences, 0.3),
        ("milestone", context.relationship.milestones, 0.3),
    )
    candidates: list[tuple[float, EvidenceItem]] = []
    for predicate, entries, base in groups:
        for entry in entries:
            payload = entry.to_dict()
            privacy = _privacy(payload.get("privacy_scope"))
            if privacy == "protected" and not context.memory.protected_available:
                continue
            text = _relationship_text(payload)
            score = _relevance(query_terms, text, base)
            candidates.append(
                (
                    score,
                    EvidenceItem(
                        id=f"evidence:relationship:{entry.id}",
                        source="relationship",
                        source_ref=entry.id,
                        subject="relationship",
                        predicate=predicate,
                        value=_bounded_value(payload),
                        summary=_bounded_text(text, 500),
                        confidence=float(payload.get("confidence", 1.0)),
                        relevance=score,
                        observed_at=str(
                            payload.get("last_active_at")
                            or payload.get("last_reference")
                            or payload.get("last_mentioned_at")
                            or payload.get("occurred_at")
                            or context.relationship.updated_at
                        ),
                        privacy_scope=privacy,
                        modality="relationship",
                        attributes={"evidence_role": "continuity"},
                    ),
                )
            )
    return tuple(candidates)


def _curiosity_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    candidates: list[tuple[float, EvidenceItem]] = []
    for curiosity in context.curiosity:
        if not getattr(curiosity, "is_active", True):
            continue
        payload = curiosity.to_dict()
        text = " ".join(
            str(payload.get(key, ""))
            for key in ("topic", "motive", "expected_discovery", "missing_information")
        )
        score = _relevance(query_terms, text, curiosity.importance * 0.45)
        candidates.append(
            (
                score,
                EvidenceItem(
                    id=f"evidence:curiosity:{curiosity.id}",
                    source="relationship",
                    source_ref=curiosity.id,
                    subject="current_curiosity",
                    predicate="curiosity",
                    value=_bounded_value(payload),
                    summary=_bounded_text(text, 420),
                    confidence=1.0,
                    relevance=score,
                    observed_at=str(payload.get("updated_at") or payload.get("appeared_at") or ""),
                    privacy_scope=_privacy(payload.get("privacy_scope")),
                    modality="state",
                    attributes={"evidence_role": "curiosity"},
                ),
            )
        )
    return tuple(candidates)


def _goal_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    candidates: list[tuple[float, EvidenceItem]] = []
    for goal in (*context.goals.permanent, *context.goals.temporary):
        if goal.status != "active":
            continue
        score = _relevance(query_terms, goal.description, goal.priority * 0.38)
        candidates.append(
            (
                score,
                EvidenceItem(
                    id=f"evidence:goal:{goal.id}",
                    source="goal",
                    source_ref=goal.id,
                    subject="anydes",
                    predicate=f"{goal.tier}_goal",
                    value={
                        "description": _bounded_text(goal.description, 500),
                        "priority": goal.priority,
                        "status": goal.status,
                    },
                    summary=_bounded_text(goal.description, 420),
                    confidence=1.0,
                    relevance=score,
                    privacy_scope="personal",
                    modality="state",
                    attributes={"evidence_role": "background"},
                ),
            )
        )
    return tuple(candidates)


def _state_evidence(
    context: InternalContext,
    query_terms: set[str],
) -> tuple[tuple[float, EvidenceItem], ...]:
    emotion_value = context.emotion.to_dict()
    uncertainty_value = context.uncertainty.to_dict()
    items: list[tuple[float, EvidenceItem]] = [
        (
            _relevance(query_terms, context.emotion.summary, 0.3),
            EvidenceItem(
                id="evidence:emotion:current",
                source="emotion",
                source_ref="emotion_snapshot",
                subject="anydes",
                predicate="current_internal_state",
                value=emotion_value,
                summary=_bounded_text(
                    context.emotion.summary or context.emotion.dominant_label,
                    300,
                ),
                confidence=1.0,
                relevance=0.3,
                observed_at=context.timestamp,
                privacy_scope="personal",
                modality="state",
                attributes={"evidence_role": "state"},
            ),
        ),
        (
            0.24,
            EvidenceItem(
                id="evidence:system:uncertainty",
                source="system",
                source_ref="uncertainty_snapshot",
                subject="current_context",
                predicate="uncertainty",
                value=uncertainty_value,
                summary="Current uncertainty by source.",
                confidence=1.0,
                relevance=0.24,
                observed_at=context.timestamp,
                privacy_scope="personal",
                modality="state",
                attributes={"evidence_role": "state"},
            ),
        ),
    ]
    if context.body.available:
        body = context.body.to_dict()
        items.append(
            (
                0.16,
                EvidenceItem(
                    id="evidence:body:current",
                    source="body",
                    source_ref="body_snapshot",
                    subject="anydes",
                    predicate="current_body_state",
                    value=_bounded_value(body),
                    summary="Current published body state.",
                    confidence=1.0,
                    relevance=0.16,
                    observed_at=context.body.updated_at,
                    privacy_scope="normal",
                    modality="body",
                    attributes={"evidence_role": "state"},
                ),
            )
        )
    return tuple(items)


def _capability_evidence(
    context: InternalContext,
    query_terms: set[str],
    limit: int,
) -> tuple[CapabilityEvidence, ...]:
    if not query_terms:
        return ()
    ranked = sorted(
        (
            (
                _relevance(
                    query_terms,
                    f"{item.action_type} {item.reason}",
                    0.12,
                ),
                item,
            )
            for item in context.available_actions
        ),
        key=lambda pair: (-pair[0], pair[1].id),
    )
    selected = [
        item
        for score, item in ranked
        if score >= 0.1
    ][:limit]
    return tuple(
        CapabilityEvidence(
            id=item.id,
            action_type=item.action_type,
            description=_bounded_text(item.reason, 360),
            requires_confirmation=item.requires_confirmation,
            risk_level=item.risk_level,
            parameter_names=tuple(sorted(item.parameters)),
            source_ref=item.id,
        )
        for item in selected
    )


def _top(
    candidates: Iterable[tuple[float, EvidenceItem]],
    limit: int,
) -> tuple[tuple[float, EvidenceItem], ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda pair: (-pair[0], pair[1].id),
        )[:limit]
    )


def _apply_budget(
    candidates: list[tuple[float, int, EvidenceItem]],
    *,
    max_items: int,
    max_characters: int,
) -> tuple[tuple[EvidenceItem, ...], int, bool]:
    selected: list[EvidenceItem] = []
    used = 0
    truncated = False
    for _, _, item in sorted(candidates, key=lambda entry: (-entry[0], entry[1])):
        size = len(json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":")))
        if len(selected) >= max_items or used + size > max_characters:
            truncated = True
            continue
        selected.append(item)
        used += size
    return tuple(selected), used, truncated


def _coverage(
    source: str,
    available_count: int,
    selected_count: int,
    revision: str,
    *,
    available: bool = True,
) -> EvidenceCoverage:
    if not available:
        status = "unavailable"
        reason = "The source did not publish a valid snapshot."
    elif selected_count == available_count:
        status = "complete"
        reason = ""
    else:
        status = "partial"
        reason = "Only contextually relevant items were selected."
    return EvidenceCoverage(
        source=source,
        snapshot_revision=str(revision),
        available_count=available_count,
        selected_count=selected_count,
        status=status,
        omitted_reason=reason,
    )


def _observation_authorized(observation: Observation) -> bool:
    return bool(observation.data.get("authorized", True))


def _observation_text(observation: Observation | None) -> str:
    if observation is None:
        return ""
    text = observation.data.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return f"{observation.summary} {json.dumps(observation.data, ensure_ascii=False)}"


def _self_preferences(context: InternalContext) -> dict[str, object]:
    raw = context.self_model.mutable_state.get("self_preferences", {})
    return raw if isinstance(raw, dict) else {}


def _thinking_style(context: InternalContext) -> object:
    identity_core = context.self_model.mutable_state.get("identity_core", {})
    if not isinstance(identity_core, dict):
        return {}
    value = identity_core.get("thinking_style", {})
    return _bounded_value(value) if isinstance(value, (dict, list, str)) else {}


def _latest_memory_revision(context: InternalContext) -> str:
    seen = [
        item.last_seen_at
        for tier in (context.memory.working, context.memory.short, context.memory.long)
        for item in tier
        if item.last_seen_at
    ]
    return max(seen, default=context.timestamp)


def _memory_privacy(memory: MemoryItem) -> str:
    if memory.category == "secret" or memory.tier == "protected":
        return "protected"
    if memory.category in {"personal", "preference", "project"}:
        return "personal"
    return "normal"


def _privacy(value: object) -> str:
    normalized = str(value or "normal").casefold()
    if normalized in {"secret", "protected"}:
        return "protected"
    if normalized == "personal":
        return "personal"
    return "normal"


def _relationship_text(payload: dict[str, object]) -> str:
    return " ".join(
        str(payload.get(key, ""))
        for key in (
            "title",
            "summary",
            "topic",
            "description",
            "motive",
            "context_note",
            "evolution_summary",
        )
    ).strip()


def _relevance(query_terms: set[str], text: str, base: float) -> float:
    if not query_terms:
        return max(0.0, min(1.0, base))
    candidate_terms = set(_terms(text))
    if not candidate_terms:
        return max(0.0, min(1.0, base * 0.15))
    overlap = len(query_terms.intersection(candidate_terms)) / max(1, len(query_terms))
    return max(0.0, min(1.0, base * 0.15 + overlap * 0.85))


def _terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    ascii_text = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    stopwords = {
        "a",
        "al",
        "como",
        "cual",
        "de",
        "del",
        "el",
        "en",
        "es",
        "la",
        "las",
        "lo",
        "los",
        "me",
        "mi",
        "por",
        "que",
        "se",
        "si",
        "su",
        "tu",
        "una",
        "un",
        "y",
    }
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", ascii_text)
        if len(token) > 1 and token not in stopwords
    )


def _bounded_text(value: object, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]} [truncated]"


def _bounded_value(value: object, *, depth: int = 0):
    if depth >= 5:
        return "[depth-limited]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:24]:
            lowered = str(key).casefold()
            if any(
                marker in lowered
                for marker in ("base64", "image_bytes", "audio_bytes", "screenshot")
            ):
                continue
            result[str(key)] = _bounded_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth=depth + 1) for item in value[-16:]]
    if isinstance(value, str):
        return _bounded_text(value, 900)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _bounded_text(value, 300)


def _context_revision(
    context: InternalContext,
    selected: tuple[EvidenceItem, ...],
) -> str:
    source = "|".join(
        [
            context.timestamp,
            context.self_model.updated_at,
            str(context.relationship.revision),
            *(item.id for item in selected),
        ]
    )
    return _digest(source)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
