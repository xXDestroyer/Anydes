from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal

from anydes.primitives import JsonValue, clamp01
from anydes.identity.relationship_contracts import (
    RelationshipMutation,
    RelationshipSnapshot,
)
from anydes.cognition.grounding_contracts import (
    BodyIntention,
    EvidenceBundle,
    GroundedResponsePlan,
)

ObservationSource = Literal[
    "user_text",
    "voice",
    "vision",
    "system",
    "time",
    "environment",
    "tool",
    "body",
    "memory",
    "self",
    "emotion",
    "goal",
    "unknown",
]
MemoryTier = Literal["working", "short", "long", "protected"]
MemoryCategory = Literal["normal", "personal", "secret", "procedural", "preference", "project"]
MemoryWriteStatus = Literal["saved", "skipped", "blocked", "failed"]
SelfWriteStatus = Literal["saved", "skipped", "blocked", "failed"]
SelfPreferenceStatus = Literal["emerging", "established", "reconsidering"]
SelfMutationOperation = Literal["form_preference", "reinforce_preference", "reconsider_preference"]
BodyUpdateStatus = Literal["updated", "skipped", "failed"]
VoiceOutputStatus = Literal["spoken", "skipped", "failed"]
GoalTier = Literal["permanent", "temporary"]
GoalStatus = Literal["active", "paused", "completed", "blocked"]
ActionStatus = Literal["pending", "approved", "executed", "failed", "cancelled", "blocked"]
RiskLevel = Literal["none", "low", "medium", "high", "critical"]
ReflectionDecision = Literal["approve", "revise", "ask_confirmation", "block", "defer"]
ConversationIntentMode = Literal[
    "respond",
    "ask",
    "deepen",
    "recall",
    "shift_topic",
    "clarify",
    "share_observation",
    "listen",
]
ConversationActSource = Literal[
    "user_request",
    "reasoning",
    "curiosity",
    "relationship",
    "observation",
    "recovery",
]
CuriosityStatus = Literal["open", "paused", "satisfied", "dismissed"]
CuriosityMutationOperation = Literal[
    "create",
    "update",
    "satisfy",
    "pause",
    "resume",
    "dismiss",
]
CuriosityPrivacyScope = Literal["normal", "personal", "protected"]
CuriosityWriteStatus = Literal["saved", "skipped", "blocked", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> JsonValue:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata(value: dict[str, Any] | None) -> dict[str, JsonValue]:
    if not value:
        return {}
    ready = _json_ready(value)
    return ready if isinstance(ready, dict) else {}


class ContractMixin:
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, JsonValue]:
        ready = _json_ready(self)
        if not isinstance(ready, dict):
            raise TypeError("Contract serialization produced a non-object value.")
        return ready

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


@dataclass(frozen=True)
class Observation(ContractMixin):
    id: str
    source: ObservationSource
    summary: str
    timestamp: str = field(default_factory=utc_now)
    confidence: float = 1.0
    salience: float = 0.5
    data: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "salience", clamp01(self.salience))
        object.__setattr__(self, "data", _metadata(self.data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(
            id=str(data["id"]),
            source=data.get("source", "unknown"),
            summary=str(data.get("summary", "")),
            timestamp=str(data.get("timestamp") or utc_now()),
            confidence=float(data.get("confidence", 1.0)),
            salience=float(data.get("salience", 0.5)),
            data=_metadata(data.get("data")),
        )


@dataclass(frozen=True)
class MemoryItem(ContractMixin):
    id: str
    tier: MemoryTier
    category: MemoryCategory
    content: str
    importance: float = 0.5
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    last_seen_at: str = field(default_factory=utc_now)
    source_observation_id: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in {"working", "short", "long", "protected"}:
            raise ValueError(f"Unsupported memory tier: {self.tier}")
        if self.category not in {
            "normal",
            "personal",
            "secret",
            "procedural",
            "preference",
            "project",
        }:
            raise ValueError(f"Unsupported memory category: {self.category}")
        object.__setattr__(self, "importance", clamp01(self.importance))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return cls(
            id=str(data["id"]),
            tier=data.get("tier", "working"),
            category=data.get("category", "normal"),
            content=str(data.get("content", "")),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 1.0)),
            created_at=str(data.get("created_at") or utc_now()),
            last_seen_at=str(data.get("last_seen_at") or utc_now()),
            source_observation_id=data.get("source_observation_id"),
            metadata=_metadata(data.get("metadata")),
        )


@dataclass(frozen=True)
class MemorySnapshot(ContractMixin):
    working: tuple[MemoryItem, ...] = ()
    short: tuple[MemoryItem, ...] = ()
    long: tuple[MemoryItem, ...] = ()
    protected_available: bool = False
    protected_summary: str = "Protected memory hidden unless explicitly authorized."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorySnapshot":
        return cls(
            working=tuple(MemoryItem.from_dict(item) for item in data.get("working", [])),
            short=tuple(MemoryItem.from_dict(item) for item in data.get("short", [])),
            long=tuple(MemoryItem.from_dict(item) for item in data.get("long", [])),
            protected_available=bool(data.get("protected_available", False)),
            protected_summary=str(data.get("protected_summary", "")),
        )


@dataclass(frozen=True)
class MemoryWriteResult(ContractMixin):
    memory_id: str
    tier: MemoryTier
    status: MemoryWriteStatus
    message: str = ""
    key: str = ""
    error: str | None = None
    saved_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryWriteResult":
        return cls(
            memory_id=str(data["memory_id"]),
            tier=data.get("tier", "working"),
            status=data.get("status", "skipped"),
            message=str(data.get("message", "")),
            key=str(data.get("key", "")),
            error=data.get("error"),
            saved_at=str(data.get("saved_at") or utc_now()),
        )


@dataclass(frozen=True)
class Goal(ContractMixin):
    id: str
    tier: GoalTier
    description: str
    priority: float = 0.5
    status: GoalStatus = "active"
    success_criteria: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", clamp01(self.priority))
        object.__setattr__(self, "success_criteria", tuple(self.success_criteria))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        return cls(
            id=str(data["id"]),
            tier=data.get("tier", "temporary"),
            description=str(data.get("description", "")),
            priority=float(data.get("priority", 0.5)),
            status=data.get("status", "active"),
            success_criteria=tuple(str(item) for item in data.get("success_criteria", [])),
            metadata=_metadata(data.get("metadata")),
        )


@dataclass(frozen=True)
class GoalSnapshot(ContractMixin):
    permanent: tuple[Goal, ...] = ()
    temporary: tuple[Goal, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalSnapshot":
        return cls(
            permanent=tuple(Goal.from_dict(item) for item in data.get("permanent", [])),
            temporary=tuple(Goal.from_dict(item) for item in data.get("temporary", [])),
        )


@dataclass(frozen=True)
class SelfModel(ContractMixin):
    name: str
    kind: str = "AI assistant"
    identity_summary: str = ""
    personality_traits: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()
    mutable_state: dict[str, JsonValue] = field(default_factory=dict)
    body_state: dict[str, JsonValue] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "personality_traits", tuple(self.personality_traits))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "mutable_state", _metadata(self.mutable_state))
        object.__setattr__(self, "body_state", _metadata(self.body_state))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModel":
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind", "AI assistant")),
            identity_summary=str(data.get("identity_summary", "")),
            personality_traits=tuple(str(item) for item in data.get("personality_traits", [])),
            boundaries=tuple(str(item) for item in data.get("boundaries", [])),
            mutable_state=_metadata(data.get("mutable_state")),
            body_state=_metadata(data.get("body_state")),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass(frozen=True)
class SelfPreference(ContractMixin):
    """A mutable preference formed by Anydes from traceable evidence."""

    key: str
    value: str
    status: SelfPreferenceStatus = "emerging"
    confidence: float = 0.6
    rationale: str = ""
    evidence_ids: tuple[str, ...] = ()
    reinforcement_count: int = 0
    domain: str = ""
    origin: str = ""
    associations: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    change_history: tuple[dict[str, JsonValue], ...] = ()
    last_confirmed_at: str = ""
    formed_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "reinforcement_count", max(0, int(self.reinforcement_count)))
        object.__setattr__(self, "associations", tuple(self.associations))
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        object.__setattr__(
            self,
            "change_history",
            tuple(dict(item) for item in self.change_history),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfPreference":
        return cls(
            key=str(data["key"]),
            value=str(data.get("value") or ""),
            status=data.get("status", "emerging"),
            confidence=float(data.get("confidence", 0.6)),
            rationale=str(data.get("rationale") or ""),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            reinforcement_count=int(data.get("reinforcement_count", 0)),
            domain=str(data.get("domain") or ""),
            origin=str(data.get("origin") or ""),
            associations=tuple(str(item) for item in data.get("associations", [])),
            alternatives=tuple(str(item) for item in data.get("alternatives", [])),
            change_history=tuple(
                dict(item)
                for item in data.get("change_history", [])
                if isinstance(item, dict)
            ),
            last_confirmed_at=str(data.get("last_confirmed_at") or ""),
            formed_at=str(data.get("formed_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass(frozen=True)
class SelfMutation(ContractMixin):
    id: str
    operation: SelfMutationOperation
    key: str
    proposed_value: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        key_aliases = {
            "aesthetic_affinities": "favorite_color",
            "color_preference": "favorite_color",
            "color_favorito": "favorite_color",
            "favourite_color": "favorite_color",
            "favourite_colour": "favorite_color",
        }
        object.__setattr__(self, "key", key_aliases.get(self.key.casefold(), self.key))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfMutation":
        return cls(
            id=str(data["id"]),
            operation=data.get("operation", "form_preference"),
            key=str(data.get("key") or ""),
            proposed_value=str(data.get("proposed_value") or ""),
            reason=str(data.get("reason") or ""),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            confidence=float(data.get("confidence", 0.5)),
        )


@dataclass(frozen=True)
class SelfMutationResult(ContractMixin):
    mutation_id: str
    operation: SelfMutationOperation
    key: str
    status: SelfWriteStatus
    preference: SelfPreference | None = None
    message: str = ""
    error: str | None = None
    written_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfMutationResult":
        preference = data.get("preference")
        return cls(
            mutation_id=str(data["mutation_id"]),
            operation=data.get("operation", "form_preference"),
            key=str(data.get("key") or ""),
            status=data.get("status", "skipped"),
            preference=SelfPreference.from_dict(preference) if isinstance(preference, dict) else None,
            message=str(data.get("message") or ""),
            error=data.get("error"),
            written_at=str(data.get("written_at") or utc_now()),
        )


@dataclass(frozen=True)
class EmotionSnapshot(ContractMixin):
    energy: float = 0.7
    mood: float = 0.0
    interest: float = 0.5
    tension: float = 0.0
    affection: float = 0.5
    dominant_label: str = "neutral"
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", clamp01(self.energy))
        object.__setattr__(self, "interest", clamp01(self.interest))
        object.__setattr__(self, "tension", clamp01(self.tension))
        object.__setattr__(self, "affection", clamp01(self.affection))
        object.__setattr__(self, "mood", max(-1.0, min(1.0, float(self.mood))))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionSnapshot":
        return cls(
            energy=float(data.get("energy", 0.7)),
            mood=float(data.get("mood", 0.0)),
            interest=float(data.get("interest", 0.5)),
            tension=float(data.get("tension", 0.0)),
            affection=float(data.get("affection", 0.5)),
            dominant_label=str(data.get("dominant_label", "neutral")),
            summary=str(data.get("summary", "")),
        )


@dataclass(frozen=True)
class UncertaintySnapshot(ContractMixin):
    memory: float = 0.0
    vision: float = 0.0
    knowledge: float = 0.0
    reasoning: float = 0.0
    user_intent: float = 0.0
    action_risk: float = 0.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("memory", "vision", "knowledge", "reasoning", "user_intent", "action_risk"):
            object.__setattr__(self, name, clamp01(getattr(self, name)))
        object.__setattr__(self, "notes", tuple(self.notes))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UncertaintySnapshot":
        return cls(
            memory=float(data.get("memory", 0.0)),
            vision=float(data.get("vision", 0.0)),
            knowledge=float(data.get("knowledge", 0.0)),
            reasoning=float(data.get("reasoning", 0.0)),
            user_intent=float(data.get("user_intent", 0.0)),
            action_risk=float(data.get("action_risk", 0.0)),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )


@dataclass(frozen=True)
class CuriosityModel(ContractMixin):
    """A persistent object of curiosity; it never decides or writes a question."""

    id: str
    intensity: float
    topic: str
    motive: str
    priority: float
    origin: str
    appeared_at: str
    age_seconds: float = 0.0
    satisfied: bool = False
    expected_discovery: str = ""
    missing_information: str = ""
    evidence: tuple[str, ...] = ()
    status: CuriosityStatus = "open"
    updated_at: str = ""
    privacy_scope: CuriosityPrivacyScope = "personal"
    related_thread_ids: tuple[str, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"open", "paused", "satisfied", "dismissed"}:
            raise ValueError(f"Unsupported curiosity status: {self.status}")
        status = "satisfied" if self.satisfied and self.status == "open" else self.status
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "satisfied", status in {"satisfied", "dismissed"})
        object.__setattr__(self, "intensity", clamp01(self.intensity))
        object.__setattr__(self, "priority", clamp01(self.priority))
        object.__setattr__(self, "age_seconds", max(0.0, float(self.age_seconds)))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "related_thread_ids", tuple(self.related_thread_ids))
        object.__setattr__(self, "revision", max(0, int(self.revision)))

    @property
    def importance(self) -> float:
        """Compatibility signal for consumers of the former CuriosityPrompt."""
        if self.status != "open" or self.satisfied:
            return 0.0
        return clamp01(self.intensity * 0.58 + self.priority * 0.42)

    @property
    def is_active(self) -> bool:
        return self.status == "open" and not self.satisfied

    @property
    def reason(self) -> str:
        return self.motive

    @property
    def question(self) -> str:
        return ""

    @property
    def should_ask_now(self) -> bool:
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CuriosityModel":
        return cls(
            id=str(data["id"]),
            intensity=float(data.get("intensity", data.get("importance", 0.5))),
            topic=str(data.get("topic", data.get("missing_information", ""))),
            motive=str(data.get("motive", data.get("reason", ""))),
            priority=float(data.get("priority", data.get("importance", 0.5))),
            origin=str(data.get("origin", "legacy_curiosity")),
            appeared_at=str(data.get("appeared_at") or utc_now()),
            age_seconds=float(data.get("age_seconds", 0.0)),
            satisfied=bool(data.get("satisfied", False)),
            expected_discovery=str(data.get("expected_discovery", "")),
            missing_information=str(data.get("missing_information", "")),
            evidence=tuple(str(item) for item in data.get("evidence", [])),
            status=str(data.get("status") or ("satisfied" if data.get("satisfied") else "open")),
            updated_at=str(data.get("updated_at", "")),
            privacy_scope=str(data.get("privacy_scope", "personal")),
            related_thread_ids=tuple(str(item) for item in data.get("related_thread_ids", [])),
            revision=int(data.get("revision", 0)),
        )


@dataclass(frozen=True)
class CuriosityMutation(ContractMixin):
    """A reasoned proposal to change curiosity state; it has no write authority."""

    id: str
    operation: CuriosityMutationOperation
    target_id: str = ""
    topic: str = ""
    motive: str = ""
    expected_discovery: str = ""
    missing_information: str = ""
    intensity: float = 0.5
    priority: float = 0.5
    origin: str = "reasoning"
    evidence_ids: tuple[str, ...] = ()
    privacy_scope: CuriosityPrivacyScope = "personal"
    related_thread_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    reason: str = ""

    def __post_init__(self) -> None:
        allowed = {"create", "update", "satisfy", "pause", "resume", "dismiss"}
        if self.operation not in allowed:
            raise ValueError(f"Unsupported curiosity mutation operation: {self.operation}")
        if self.privacy_scope not in {"normal", "personal", "protected"}:
            raise ValueError(f"Unsupported curiosity privacy scope: {self.privacy_scope}")
        object.__setattr__(self, "intensity", clamp01(self.intensity))
        object.__setattr__(self, "priority", clamp01(self.priority))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "related_thread_ids", tuple(self.related_thread_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CuriosityMutation":
        return cls(
            id=str(data["id"]),
            operation=data.get("operation", "create"),
            target_id=str(data.get("target_id", "")),
            topic=str(data.get("topic", "")),
            motive=str(data.get("motive", "")),
            expected_discovery=str(data.get("expected_discovery", "")),
            missing_information=str(data.get("missing_information", "")),
            intensity=float(data.get("intensity", 0.5)),
            priority=float(data.get("priority", 0.5)),
            origin=str(data.get("origin", "reasoning")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            privacy_scope=data.get("privacy_scope", "personal"),
            related_thread_ids=tuple(str(item) for item in data.get("related_thread_ids", [])),
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class CuriosityWriteResult(ContractMixin):
    mutation_id: str
    curiosity_id: str
    operation: CuriosityMutationOperation
    status: CuriosityWriteStatus
    message: str = ""
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CuriosityWriteResult":
        return cls(
            mutation_id=str(data["mutation_id"]),
            curiosity_id=str(data.get("curiosity_id", "")),
            operation=data.get("operation", "create"),
            status=data.get("status", "failed"),
            message=str(data.get("message", "")),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class CuriosityPrompt(ContractMixin):
    """Deprecated compatibility contract. New providers emit CuriosityModel."""

    id: str
    question: str
    reason: str
    importance: float = 0.5
    should_ask_now: bool = False
    missing_information: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "importance", clamp01(self.importance))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CuriosityPrompt":
        return cls(
            id=str(data["id"]),
            question=str(data.get("question", "")),
            reason=str(data.get("reason", "")),
            importance=float(data.get("importance", 0.5)),
            should_ask_now=bool(data.get("should_ask_now", False)),
            missing_information=str(data.get("missing_information", "")),
        )


@dataclass(frozen=True)
class SpeechIntention(ContractMixin):
    text: str
    emotion: str = "calm"
    energy: float = 0.38
    warmth: float = 0.82
    confidence: float = 0.72
    thinking: float = 0.12
    curiosity: float = 0.45
    pause_before: float = 0.18
    pause_after: float = 0.18
    speech_rate: float = 0.9
    pitch_variation: float = 0.14
    volume: float = 0.9
    smile: float = 0.24
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    plan_id: str = ""
    realized_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "realized_claim_ids", tuple(self.realized_claim_ids))
        object.__setattr__(self, "energy", clamp01(self.energy))
        object.__setattr__(self, "warmth", clamp01(self.warmth))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "thinking", clamp01(self.thinking))
        object.__setattr__(self, "curiosity", clamp01(self.curiosity))
        object.__setattr__(self, "pause_before", max(0.0, min(3.0, float(self.pause_before))))
        object.__setattr__(self, "pause_after", max(0.0, min(3.0, float(self.pause_after))))
        object.__setattr__(self, "speech_rate", max(0.65, min(1.35, float(self.speech_rate))))
        object.__setattr__(self, "pitch_variation", clamp01(self.pitch_variation))
        object.__setattr__(self, "volume", max(0.0, min(1.2, float(self.volume))))
        object.__setattr__(self, "smile", clamp01(self.smile))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_text(cls, text: str) -> "SpeechIntention":
        """Compatibility constructor for legacy text-producing boundaries."""
        return cls(text=str(text or ""))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechIntention":
        return cls(
            text=str(data.get("text", "")),
            plan_id=str(data.get("plan_id", "")),
            realized_claim_ids=tuple(
                str(item) for item in data.get("realized_claim_ids", [])
            ),
            emotion=str(data.get("emotion", "calm")),
            energy=float(data.get("energy", 0.38)),
            warmth=float(data.get("warmth", 0.82)),
            confidence=float(data.get("confidence", 0.72)),
            thinking=float(data.get("thinking", 0.12)),
            curiosity=float(data.get("curiosity", 0.45)),
            pause_before=float(data.get("pause_before", 0.18)),
            pause_after=float(data.get("pause_after", 0.18)),
            speech_rate=float(data.get("speech_rate", 0.9)),
            pitch_variation=float(data.get("pitch_variation", 0.14)),
            volume=float(data.get("volume", 0.9)),
            smile=float(data.get("smile", 0.24)),
            metadata=_metadata(data.get("metadata")),
        )


@dataclass(frozen=True)
class ActionRequest(ContractMixin):
    id: str
    action_type: str
    target: str = ""
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    requires_confirmation: bool = False
    risk_level: RiskLevel = "low"
    reason: str = ""
    plan_id: str = ""
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "parameters", _metadata(self.parameters))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionRequest":
        return cls(
            id=str(data["id"]),
            action_type=str(data.get("action_type", "")),
            plan_id=str(data.get("plan_id", "")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            target=str(data.get("target", "")),
            parameters=_metadata(data.get("parameters")),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
            risk_level=data.get("risk_level", "low"),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class ConversationIntent(ContractMixin):
    mode: ConversationIntentMode = "respond"
    target_topic: str = ""
    reason: str = ""
    curiosity_id: str = ""
    relationship_thread_id: str = ""
    open_loop_id: str = ""
    plan_id: str = ""
    claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {
            "respond",
            "ask",
            "deepen",
            "recall",
            "shift_topic",
            "clarify",
            "share_observation",
            "listen",
        }
        if self.mode not in allowed:
            raise ValueError(f"Unsupported conversation intent mode: {self.mode}")
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationIntent":
        return cls(
            mode=data.get("mode", "respond"),
            plan_id=str(data.get("plan_id", "")),
            claim_ids=tuple(str(item) for item in data.get("claim_ids", [])),
            target_topic=str(data.get("target_topic") or ""),
            reason=str(data.get("reason") or ""),
            curiosity_id=str(data.get("curiosity_id") or ""),
            relationship_thread_id=str(data.get("relationship_thread_id") or ""),
            open_loop_id=str(data.get("open_loop_id") or ""),
        )


@dataclass(frozen=True)
class ConversationAct(ContractMixin):
    """One conversational move proposed for the current turn.

    Acts describe conversational obligations and opportunities. They never
    generate speech or execute behavior.
    """

    id: str
    mode: ConversationIntentMode
    target_topic: str = ""
    priority: float = 0.5
    required: bool = False
    source: ConversationActSource = "reasoning"
    curiosity_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", clamp01(self.priority))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationAct":
        return cls(
            id=str(data["id"]),
            mode=data.get("mode", "respond"),
            target_topic=str(data.get("target_topic") or ""),
            priority=float(data.get("priority", 0.5)),
            required=bool(data.get("required", False)),
            source=data.get("source", "reasoning"),
            curiosity_id=str(data.get("curiosity_id") or ""),
            reason=str(data.get("reason") or ""),
        )


@dataclass(frozen=True)
class ConversationPlan(ContractMixin):
    """Ordered conversational obligations produced before final expression."""

    observation_id: str
    primary: ConversationAct
    secondary: tuple[ConversationAct, ...] = ()
    deferred_curiosity_ids: tuple[str, ...] = ()
    direct_question: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "secondary", tuple(self.secondary))
        object.__setattr__(self, "deferred_curiosity_ids", tuple(self.deferred_curiosity_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationPlan":
        return cls(
            observation_id=str(data.get("observation_id") or ""),
            primary=ConversationAct.from_dict(data.get("primary", {"id": "act:respond", "mode": "respond"})),
            secondary=tuple(ConversationAct.from_dict(item) for item in data.get("secondary", [])),
            deferred_curiosity_ids=tuple(str(item) for item in data.get("deferred_curiosity_ids", [])),
            direct_question=bool(data.get("direct_question", False)),
            rationale=str(data.get("rationale") or ""),
        )


@dataclass(frozen=True)
class Intention(ContractMixin):
    id: str
    intent_type: str
    conversation_intent: ConversationIntent | dict[str, JsonValue] | str = field(
        default_factory=ConversationIntent
    )
    speech: str | dict[str, JsonValue] = ""
    speech_intention: SpeechIntention | dict[str, JsonValue] | None = None
    actions: tuple[ActionRequest, ...] = ()
    avatar: dict[str, JsonValue] = field(default_factory=dict)
    memory_updates: tuple[MemoryItem, ...] = ()
    relationship_updates: tuple[RelationshipMutation, ...] = ()
    self_updates: tuple[SelfMutation, ...] = ()
    curiosity_updates: tuple[CuriosityMutation, ...] = ()
    confidence: float = 0.5
    reasoning_summary: str = ""
    grounded_plan: GroundedResponsePlan | dict[str, JsonValue] | None = None
    body_intention: BodyIntention | dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        grounded_plan = self.grounded_plan
        if isinstance(grounded_plan, dict):
            grounded_plan = GroundedResponsePlan.from_dict(grounded_plan)
        body_intention = self.body_intention
        if isinstance(body_intention, dict):
            body_intention = BodyIntention.from_dict(body_intention)
        object.__setattr__(self, "grounded_plan", grounded_plan)
        object.__setattr__(self, "body_intention", body_intention)

        conversation_intent = self.conversation_intent
        if isinstance(conversation_intent, dict):
            conversation_intent = ConversationIntent.from_dict(conversation_intent)
        elif isinstance(conversation_intent, str):
            conversation_intent = ConversationIntent(mode=conversation_intent or "respond")
        object.__setattr__(self, "conversation_intent", conversation_intent)

        speech_value = self.speech
        speech_intention = self.speech_intention
        if speech_intention is None and isinstance(speech_value, dict):
            speech_intention = SpeechIntention.from_dict(speech_value)
            speech_text = speech_intention.text
        elif isinstance(speech_intention, SpeechIntention):
            speech_text = str(speech_value or speech_intention.text)
        elif isinstance(speech_intention, dict):
            speech_intention = SpeechIntention.from_dict(speech_intention)
            speech_text = str(speech_value or speech_intention.text)
        else:
            speech_text = str(speech_value or "")
            speech_intention = SpeechIntention.from_text(speech_text)

        speech_metadata = dict(speech_intention.metadata)
        speech_metadata.setdefault("intention_id", self.id)
        speech_intention = replace(speech_intention, metadata=speech_metadata)

        object.__setattr__(self, "speech", speech_text)
        object.__setattr__(self, "speech_intention", speech_intention)
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "memory_updates", tuple(self.memory_updates))
        object.__setattr__(self, "relationship_updates", tuple(self.relationship_updates))
        object.__setattr__(self, "self_updates", tuple(self.self_updates))
        object.__setattr__(self, "curiosity_updates", tuple(self.curiosity_updates))
        object.__setattr__(self, "avatar", _metadata(self.avatar))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        payload = {
            "id": self.id,
            "intent_type": self.intent_type,
            "conversation_intent": self.conversation_intent.to_dict(),
            "speech": self.speech_intention.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
            "avatar": self.avatar,
            "memory_updates": [item.to_dict() for item in self.memory_updates],
            "relationship_updates": [item.to_dict() for item in self.relationship_updates],
            "self_updates": [item.to_dict() for item in self.self_updates],
            "curiosity_updates": [item.to_dict() for item in self.curiosity_updates],
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
        }
        if self.grounded_plan is not None:
            payload["grounded_plan"] = self.grounded_plan.to_dict()
        if self.body_intention is not None:
            payload["body_intention"] = self.body_intention.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intention":
        return cls(
            id=str(data["id"]),
            intent_type=str(data.get("intent_type", "")),
            grounded_plan=data.get("grounded_plan"),
            conversation_intent=data.get("conversation_intent", {"mode": "respond"}),
            speech=data.get("speech", ""),
            speech_intention=data.get("speech_intention"),
            body_intention=data.get("body_intention"),
            actions=tuple(ActionRequest.from_dict(item) for item in data.get("actions", [])),
            avatar=_metadata(data.get("avatar")),
            memory_updates=tuple(MemoryItem.from_dict(item) for item in data.get("memory_updates", [])),
            relationship_updates=tuple(
                RelationshipMutation.from_dict(item) for item in data.get("relationship_updates", [])
            ),
            self_updates=tuple(SelfMutation.from_dict(item) for item in data.get("self_updates", [])),
            curiosity_updates=tuple(
                CuriosityMutation.from_dict(item) for item in data.get("curiosity_updates", [])
            ),
            confidence=float(data.get("confidence", 0.5)),
            reasoning_summary=str(data.get("reasoning_summary", "")),
        )


@dataclass(frozen=True)
class ReflectionResult(ContractMixin):
    decision: ReflectionDecision
    approved: bool
    risk_level: RiskLevel = "low"
    needs_confirmation: bool = False
    reason: str = ""
    revised_intention: Intention | None = None
    blocking_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocking_issues", tuple(self.blocking_issues))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionResult":
        revised = data.get("revised_intention")
        return cls(
            decision=data.get("decision", "approve"),
            approved=bool(data.get("approved", False)),
            risk_level=data.get("risk_level", "low"),
            needs_confirmation=bool(data.get("needs_confirmation", False)),
            reason=str(data.get("reason", "")),
            revised_intention=Intention.from_dict(revised) if isinstance(revised, dict) else None,
            blocking_issues=tuple(str(item) for item in data.get("blocking_issues", [])),
        )


@dataclass(frozen=True)
class ActionResult(ContractMixin):
    request_id: str
    status: ActionStatus
    message: str = ""
    output: dict[str, JsonValue] = field(default_factory=dict)
    error: str | None = None
    completed_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _metadata(self.output))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResult":
        return cls(
            request_id=str(data["request_id"]),
            status=data.get("status", "pending"),
            message=str(data.get("message", "")),
            output=_metadata(data.get("output")),
            error=data.get("error"),
            completed_at=str(data.get("completed_at") or utc_now()),
        )


@dataclass(frozen=True)
class BodyStateSnapshot(ContractMixin):
    """Read-only description of the currently published virtual body state."""

    available: bool = False
    state: str = "neutral"
    expression: str = "neutral"
    gesture: str = "idle"
    gaze: str = "user"
    posture: str = "relaxed"
    attention: str = "user"
    expression_intensity: float = 0.5
    speaking: bool = False
    idle: bool = True
    lip_sync: dict[str, JsonValue] = field(default_factory=dict)
    model_3d: dict[str, JsonValue] = field(default_factory=dict)
    updated_at: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lip_sync", _metadata(self.lip_sync))
        object.__setattr__(self, "model_3d", _metadata(self.model_3d))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "expression_intensity", clamp01(self.expression_intensity))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyStateSnapshot":
        return cls(
            available=bool(data.get("available", False)),
            state=str(data.get("state", "neutral")),
            expression=str(data.get("expression", "neutral")),
            gesture=str(data.get("gesture", "idle")),
            gaze=str(data.get("gaze", "user")),
            posture=str(data.get("posture", "relaxed")),
            attention=str(data.get("attention", "user")),
            expression_intensity=float(data.get("expression_intensity", 0.5)),
            speaking=bool(data.get("speaking", False)),
            idle=bool(data.get("idle", True)),
            lip_sync=_metadata(data.get("lip_sync")),
            model_3d=_metadata(data.get("model_3d")),
            updated_at=str(data.get("updated_at", "")),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
        )


@dataclass(frozen=True)
class BodyUpdateResult(ContractMixin):
    intention_id: str
    status: BodyUpdateStatus
    state: str = "neutral"
    expression: str = "neutral"
    gesture: str = "idle"
    gaze: str = "user"
    message: str = ""
    output_path: str = ""
    error: str | None = None
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyUpdateResult":
        return cls(
            intention_id=str(data["intention_id"]),
            status=data.get("status", "skipped"),
            state=str(data.get("state", "neutral")),
            expression=str(data.get("expression", "neutral")),
            gesture=str(data.get("gesture", "idle")),
            gaze=str(data.get("gaze", "user")),
            message=str(data.get("message", "")),
            output_path=str(data.get("output_path", "")),
            error=data.get("error"),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass(frozen=True)
class VoiceOutputResult(ContractMixin):
    intention_id: str
    status: VoiceOutputStatus
    text: str = ""
    provider: str = ""
    error: str | None = None
    speech: SpeechIntention | None = None
    render_plan: dict[str, JsonValue] = field(default_factory=dict)
    spoken_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        speech = self.speech
        if isinstance(speech, dict):
            speech = SpeechIntention.from_dict(speech)
        object.__setattr__(self, "speech", speech)
        object.__setattr__(self, "render_plan", _metadata(self.render_plan))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceOutputResult":
        return cls(
            intention_id=str(data["intention_id"]),
            status=data.get("status", "skipped"),
            text=str(data.get("text", "")),
            provider=str(data.get("provider", "")),
            error=data.get("error"),
            speech=SpeechIntention.from_dict(data["speech"]) if isinstance(data.get("speech"), dict) else None,
            render_plan=_metadata(data.get("render_plan")),
            spoken_at=str(data.get("spoken_at") or utc_now()),
        )


@dataclass(frozen=True)
class InternalContext(ContractMixin):
    observations: tuple[Observation, ...]
    memory: MemorySnapshot
    goals: GoalSnapshot
    self_model: SelfModel
    emotion: EmotionSnapshot
    uncertainty: UncertaintySnapshot
    curiosity: tuple[CuriosityModel | CuriosityPrompt, ...] = ()
    relationship: RelationshipSnapshot = field(default_factory=RelationshipSnapshot)
    body: BodyStateSnapshot = field(default_factory=BodyStateSnapshot)
    available_actions: tuple[ActionRequest, ...] = ()
    timestamp: str = field(default_factory=utc_now)
    evidence_bundle: EvidenceBundle | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "curiosity", tuple(self.curiosity))
        object.__setattr__(self, "available_actions", tuple(self.available_actions))
        evidence_bundle = self.evidence_bundle
        if isinstance(evidence_bundle, dict):
            evidence_bundle = EvidenceBundle.from_dict(evidence_bundle)
        object.__setattr__(self, "evidence_bundle", evidence_bundle)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalContext":
        return cls(
            observations=tuple(Observation.from_dict(item) for item in data.get("observations", [])),
            memory=MemorySnapshot.from_dict(data.get("memory", {})),
            goals=GoalSnapshot.from_dict(data.get("goals", {})),
            self_model=SelfModel.from_dict(data["self_model"]),
            emotion=EmotionSnapshot.from_dict(data.get("emotion", {})),
            uncertainty=UncertaintySnapshot.from_dict(data.get("uncertainty", {})),
            curiosity=tuple(
                CuriosityModel.from_dict(item)
                if "topic" in item or "intensity" in item
                else CuriosityPrompt.from_dict(item)
                for item in data.get("curiosity", [])
            ),
            relationship=RelationshipSnapshot.from_dict(data.get("relationship", {})),
            body=BodyStateSnapshot.from_dict(data.get("body", {})),
            available_actions=tuple(ActionRequest.from_dict(item) for item in data.get("available_actions", [])),
            evidence_bundle=(
                EvidenceBundle.from_dict(data["evidence_bundle"])
                if isinstance(data.get("evidence_bundle"), dict)
                else None
            ),
            timestamp=str(data.get("timestamp") or utc_now()),
        )


@dataclass(frozen=True)
class MetaCognitionReport(ContractMixin):
    interaction_id: str
    quality: Literal["unknown", "poor", "mixed", "good", "excellent"] = "unknown"
    lessons: tuple[str, ...] = ()
    self_adjustments: dict[str, JsonValue] = field(default_factory=dict)
    memory_candidates: tuple[MemoryItem, ...] = ()
    goal_updates: tuple[Goal, ...] = ()
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lessons", tuple(self.lessons))
        object.__setattr__(self, "self_adjustments", _metadata(self.self_adjustments))
        object.__setattr__(self, "memory_candidates", tuple(self.memory_candidates))
        object.__setattr__(self, "goal_updates", tuple(self.goal_updates))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaCognitionReport":
        return cls(
            interaction_id=str(data["interaction_id"]),
            quality=data.get("quality", "unknown"),
            lessons=tuple(str(item) for item in data.get("lessons", [])),
            self_adjustments=_metadata(data.get("self_adjustments")),
            memory_candidates=tuple(MemoryItem.from_dict(item) for item in data.get("memory_candidates", [])),
            goal_updates=tuple(Goal.from_dict(item) for item in data.get("goal_updates", [])),
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at") or utc_now()),
        )
