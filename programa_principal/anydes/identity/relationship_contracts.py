from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from anydes.primitives import JsonValue, clamp01


RelationshipThreadStatus = Literal["active", "dormant", "archived"]
RelationshipTargetType = Literal[
    "memory",
    "open_loop",
    "promise",
    "curiosity",
    "shared_experience",
    "milestone",
    "thread",
]
OpenLoopStatus = Literal["open", "paused", "closed"]
PromiseStatus = Literal["pending", "active", "fulfilled", "expired", "cancelled", "forgotten"]
PromiseType = Literal["DemoUser", "anydes", "mutual", "unspecified"]
PromiseEvidenceMode = Literal["explicit", "implicit"]
RelationshipMutationOperation = Literal[
    "create_thread",
    "update_thread",
    "archive_thread",
    "link",
    "unlink",
    "open_loop",
    "update_loop",
    "pause_loop",
    "close_loop",
    "register_promise",
    "update_promise",
    "transition_promise",
    "fulfill_promise",
    "record_experience",
    "record_milestone",
]
RelationshipMutationStatus = Literal["saved", "skipped", "blocked", "failed"]


def relationship_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata(value: Any) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    return copy.deepcopy(value)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class RelationshipAssociation:
    id: str
    thread_id: str
    target_type: RelationshipTargetType
    target_id: str
    relationship_role: str
    relevance: float = 0.5
    contextual_note: str = ""
    evidence_ids: tuple[str, ...] = ()
    linked_at: str = field(default_factory=relationship_utc_now)
    source_type: RelationshipTargetType = "thread"
    source_id: str = ""

    def __post_init__(self) -> None:
        allowed_types = {
            "memory", "open_loop", "promise", "curiosity",
            "shared_experience", "milestone", "thread",
        }
        if self.target_type not in allowed_types:
            raise ValueError(f"Unsupported relationship target type: {self.target_type}")
        if self.source_type not in allowed_types:
            raise ValueError(f"Unsupported relationship source type: {self.source_type}")
        source_id = self.source_id or self.thread_id
        thread_id = self.thread_id
        if self.source_type == "thread":
            thread_id = thread_id or source_id
            if source_id and thread_id != source_id:
                raise ValueError("Thread association source_id and thread_id disagree.")
        if not source_id:
            raise ValueError("Relationship association requires source_id.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "thread_id", thread_id)
        object.__setattr__(self, "relevance", clamp01(self.relevance))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "relationship_role": self.relationship_role,
            "relevance": self.relevance,
            "contextual_note": self.contextual_note,
            "evidence_ids": list(self.evidence_ids),
            "linked_at": self.linked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipAssociation":
        return cls(
            id=str(data["id"]),
            thread_id=str(data.get("thread_id", "")),
            source_type=data.get("source_type", "thread"),
            source_id=str(data.get("source_id") or data.get("thread_id", "")),
            target_type=data.get("target_type", "memory"),
            target_id=str(data.get("target_id", "")),
            relationship_role=str(data.get("relationship_role", "related")),
            relevance=float(data.get("relevance", 0.5)),
            contextual_note=str(data.get("contextual_note", "")),
            evidence_ids=_strings(data.get("evidence_ids")),
            linked_at=str(data.get("linked_at") or relationship_utc_now()),
        )


@dataclass(frozen=True)
class RelationshipThread:
    id: str
    title: str
    summary: str
    status: RelationshipThreadStatus = "active"
    significance: float = 0.5
    aliases: tuple[str, ...] = ()
    emotional_associations: tuple[str, ...] = ()
    evolution_summary: str = ""
    created_at: str = field(default_factory=relationship_utc_now)
    last_active_at: str = field(default_factory=relationship_utc_now)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"active", "dormant", "archived"}:
            raise ValueError(f"Unsupported relationship thread status: {self.status}")
        object.__setattr__(self, "significance", clamp01(self.significance))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "emotional_associations", tuple(self.emotional_associations))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "significance": self.significance,
            "aliases": list(self.aliases),
            "emotional_associations": list(self.emotional_associations),
            "evolution_summary": self.evolution_summary,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipThread":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            status=data.get("status", "active"),
            significance=float(data.get("significance", 0.5)),
            aliases=_strings(data.get("aliases")),
            emotional_associations=_strings(data.get("emotional_associations")),
            evolution_summary=str(data.get("evolution_summary", "")),
            created_at=str(data.get("created_at") or relationship_utc_now()),
            last_active_at=str(data.get("last_active_at") or relationship_utc_now()),
            metadata=_metadata(data.get("metadata")),
        )


@dataclass(frozen=True)
class OpenConversationLoop:
    id: str
    topic: str
    origin: str
    motive: str
    priority: float
    status: OpenLoopStatus = "open"
    interest_level: float = 0.5
    related_context: tuple[str, ...] = ()
    recommended_resume_conditions: tuple[str, ...] = ()
    linked_curiosity_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=relationship_utc_now)
    last_mentioned_at: str = field(default_factory=relationship_utc_now)
    closed_at: str = ""
    closure_reason: str = ""
    privacy_scope: Literal["normal", "personal", "protected"] = "normal"

    def __post_init__(self) -> None:
        if self.status not in {"open", "paused", "closed"}:
            raise ValueError(f"Unsupported open-loop status: {self.status}")
        if self.privacy_scope not in {"normal", "personal", "protected"}:
            raise ValueError(f"Unsupported open-loop privacy scope: {self.privacy_scope}")
        object.__setattr__(self, "priority", clamp01(self.priority))
        object.__setattr__(self, "interest_level", clamp01(self.interest_level))
        for name in (
            "related_context",
            "recommended_resume_conditions",
            "linked_curiosity_ids",
            "evidence_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "topic": self.topic,
            "origin": self.origin,
            "motive": self.motive,
            "priority": self.priority,
            "status": self.status,
            "interest_level": self.interest_level,
            "related_context": list(self.related_context),
            "recommended_resume_conditions": list(self.recommended_resume_conditions),
            "linked_curiosity_ids": list(self.linked_curiosity_ids),
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at,
            "last_mentioned_at": self.last_mentioned_at,
            "closed_at": self.closed_at,
            "closure_reason": self.closure_reason,
            "privacy_scope": self.privacy_scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenConversationLoop":
        return cls(
            id=str(data["id"]),
            topic=str(data.get("topic", "")),
            origin=str(data.get("origin", "")),
            motive=str(data.get("motive", "")),
            priority=float(data.get("priority", 0.5)),
            status=data.get("status", "open"),
            interest_level=float(data.get("interest_level", 0.5)),
            related_context=_strings(data.get("related_context")),
            recommended_resume_conditions=_strings(data.get("recommended_resume_conditions")),
            linked_curiosity_ids=_strings(data.get("linked_curiosity_ids")),
            evidence_ids=_strings(data.get("evidence_ids")),
            created_at=str(data.get("created_at") or relationship_utc_now()),
            last_mentioned_at=str(data.get("last_mentioned_at") or relationship_utc_now()),
            closed_at=str(data.get("closed_at", "")),
            closure_reason=str(data.get("closure_reason", "")),
            privacy_scope=data.get("privacy_scope", "normal"),
        )


@dataclass(frozen=True)
class PromiseTransition:
    from_status: PromiseStatus
    to_status: PromiseStatus
    reason: str
    evidence_ids: tuple[str, ...] = ()
    changed_at: str = field(default_factory=relationship_utc_now)

    def __post_init__(self) -> None:
        allowed = {"pending", "active", "fulfilled", "expired", "cancelled", "forgotten"}
        if self.from_status not in allowed or self.to_status not in allowed:
            raise ValueError("Promise transition contains an unsupported status.")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "changed_at": self.changed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromiseTransition":
        return cls(
            from_status=data.get("from_status", "pending"),
            to_status=data.get("to_status", "pending"),
            reason=str(data.get("reason", "")),
            evidence_ids=_strings(data.get("evidence_ids")),
            changed_at=str(data.get("changed_at") or relationship_utc_now()),
        )


@dataclass(frozen=True, init=False)
class RelationshipPromise:
    """A social commitment in shared relational history, never a task or timer."""

    id: str
    speaker: str
    receiver: str
    topic: str
    description: str
    importance: float
    confidence: float
    creation_time: str
    last_reference: str
    status: PromiseStatus
    fulfillment_reason: str
    related_thread_ids: tuple[str, ...]
    related_loop_ids: tuple[str, ...]
    related_memory_ids: tuple[str, ...]
    related_curiosity_ids: tuple[str, ...]
    related_experience_ids: tuple[str, ...]
    related_milestone_ids: tuple[str, ...]
    privacy_scope: Literal["normal", "personal", "protected"]
    promise_type: PromiseType
    evidence_mode: PromiseEvidenceMode
    evidence_ids: tuple[str, ...]
    status_reason: str
    status_changed_at: str
    context_note: str
    transition_history: tuple[PromiseTransition, ...]

    def __init__(
        self,
        id: str,
        speaker: str = "",
        receiver: str = "",
        topic: str = "",
        description: str = "",
        importance: float = 0.5,
        confidence: float = 0.5,
        creation_time: str = "",
        last_reference: str = "",
        status: PromiseStatus = "pending",
        fulfillment_reason: str = "",
        related_thread_ids: tuple[str, ...] = (),
        related_loop_ids: tuple[str, ...] = (),
        related_memory_ids: tuple[str, ...] = (),
        related_curiosity_ids: tuple[str, ...] = (),
        related_experience_ids: tuple[str, ...] = (),
        related_milestone_ids: tuple[str, ...] = (),
        privacy_scope: Literal["normal", "personal", "protected"] = "normal",
        promise_type: PromiseType = "unspecified",
        evidence_mode: PromiseEvidenceMode = "explicit",
        evidence_ids: tuple[str, ...] = (),
        status_reason: str = "",
        status_changed_at: str = "",
        context_note: str = "",
        transition_history: tuple[PromiseTransition, ...] = (),
        *,
        promise_text: str | None = None,
        promisor: str | None = None,
        context: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        now = relationship_utc_now()
        description = description or str(promise_text or "")
        speaker = speaker or str(promisor or "")
        topic = topic or description[:160]
        creation_time = creation_time or str(created_at or now)
        last_reference = last_reference or str(updated_at or creation_time)
        context_note = context_note or str(context or "")
        if status not in {"pending", "active", "fulfilled", "expired", "cancelled", "forgotten"}:
            raise ValueError(f"Unsupported promise status: {status}")
        if privacy_scope not in {"normal", "personal", "protected"}:
            raise ValueError(f"Unsupported promise privacy scope: {privacy_scope}")
        if promise_type not in {"DemoUser", "anydes", "mutual", "unspecified"}:
            raise ValueError(f"Unsupported promise type: {promise_type}")
        if evidence_mode not in {"explicit", "implicit"}:
            raise ValueError(f"Unsupported promise evidence mode: {evidence_mode}")
        values = {
            "id": str(id), "speaker": speaker, "receiver": receiver or "unspecified",
            "topic": topic, "description": description,
            "importance": clamp01(importance), "confidence": clamp01(confidence),
            "creation_time": creation_time, "last_reference": last_reference,
            "status": status, "fulfillment_reason": fulfillment_reason,
            "related_thread_ids": tuple(dict.fromkeys(related_thread_ids)),
            "related_loop_ids": tuple(dict.fromkeys(related_loop_ids)),
            "related_memory_ids": tuple(dict.fromkeys(related_memory_ids)),
            "related_curiosity_ids": tuple(dict.fromkeys(related_curiosity_ids)),
            "related_experience_ids": tuple(dict.fromkeys(related_experience_ids)),
            "related_milestone_ids": tuple(dict.fromkeys(related_milestone_ids)),
            "privacy_scope": privacy_scope, "promise_type": promise_type,
            "evidence_mode": evidence_mode, "evidence_ids": tuple(evidence_ids),
            "status_reason": status_reason,
            "status_changed_at": status_changed_at or creation_time,
            "context_note": context_note,
            "transition_history": tuple(
                item if isinstance(item, PromiseTransition) else PromiseTransition.from_dict(item)
                for item in transition_history
                if isinstance(item, (PromiseTransition, dict))
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def promise_text(self) -> str:
        return self.description

    @property
    def promisor(self) -> str:
        return self.speaker

    @property
    def context(self) -> str:
        return self.context_note

    @property
    def created_at(self) -> str:
        return self.creation_time

    @property
    def updated_at(self) -> str:
        return self.last_reference

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            **self.to_record_dict(),
            "related_thread_ids": list(self.related_thread_ids),
            "related_loop_ids": list(self.related_loop_ids),
            "related_memory_ids": list(self.related_memory_ids),
            "related_curiosity_ids": list(self.related_curiosity_ids),
            "related_experience_ids": list(self.related_experience_ids),
            "related_milestone_ids": list(self.related_milestone_ids),
        }

    def to_record_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "receiver": self.receiver,
            "topic": self.topic,
            "description": self.description,
            "importance": self.importance,
            "confidence": self.confidence,
            "creation_time": self.creation_time,
            "last_reference": self.last_reference,
            "status": self.status,
            "fulfillment_reason": self.fulfillment_reason,
            "privacy_scope": self.privacy_scope,
            "promise_type": self.promise_type,
            "evidence_mode": self.evidence_mode,
            "evidence_ids": list(self.evidence_ids),
            "status_reason": self.status_reason,
            "status_changed_at": self.status_changed_at,
            "context_note": self.context_note,
            "transition_history": [item.to_dict() for item in self.transition_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipPromise":
        description = str(data.get("description") or data.get("promise_text", ""))
        return cls(
            id=str(data["id"]),
            speaker=str(data.get("speaker") or data.get("promisor", "")),
            receiver=str(data.get("receiver") or "unspecified"),
            topic=str(data.get("topic") or description[:160]),
            description=description,
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            creation_time=str(data.get("creation_time") or data.get("created_at") or relationship_utc_now()),
            last_reference=str(data.get("last_reference") or data.get("updated_at") or data.get("created_at") or relationship_utc_now()),
            status=data.get("status", "pending"),
            fulfillment_reason=str(data.get("fulfillment_reason", "")),
            related_thread_ids=_strings(data.get("related_thread_ids")),
            related_loop_ids=_strings(data.get("related_loop_ids")),
            related_memory_ids=_strings(data.get("related_memory_ids")),
            related_curiosity_ids=_strings(data.get("related_curiosity_ids")),
            related_experience_ids=_strings(data.get("related_experience_ids")),
            related_milestone_ids=_strings(data.get("related_milestone_ids")),
            privacy_scope=data.get("privacy_scope", "normal"),
            promise_type=data.get("promise_type", "unspecified"),
            evidence_mode=data.get("evidence_mode", "explicit"),
            evidence_ids=_strings(data.get("evidence_ids")),
            status_reason=str(data.get("status_reason", "")),
            status_changed_at=str(data.get("status_changed_at") or data.get("creation_time") or data.get("created_at") or relationship_utc_now()),
            context_note=str(data.get("context_note") or data.get("context", "")),
            transition_history=tuple(
                item if isinstance(item, PromiseTransition) else PromiseTransition.from_dict(item)
                for item in data.get("transition_history", [])
                if isinstance(item, (PromiseTransition, dict))
            ),
        )


PromiseModel = RelationshipPromise


@dataclass(frozen=True)
class SharedExperience:
    id: str
    title: str
    summary: str
    occurred_at: str
    emotional_associations: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "emotional_associations", tuple(self.emotional_associations))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "occurred_at": self.occurred_at,
            "emotional_associations": list(self.emotional_associations),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedExperience":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            occurred_at=str(data.get("occurred_at") or relationship_utc_now()),
            emotional_associations=_strings(data.get("emotional_associations")),
            evidence_ids=_strings(data.get("evidence_ids")),
        )


@dataclass(frozen=True)
class RelationshipMilestone:
    id: str
    title: str
    description: str
    significance: float = 0.5
    occurred_at: str = field(default_factory=relationship_utc_now)
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "significance", clamp01(self.significance))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "significance": self.significance,
            "occurred_at": self.occurred_at,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipMilestone":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            significance=float(data.get("significance", 0.5)),
            occurred_at=str(data.get("occurred_at") or relationship_utc_now()),
            evidence_ids=_strings(data.get("evidence_ids")),
        )


@dataclass(frozen=True)
class RelationshipSnapshot:
    revision: int = 0
    threads: tuple[RelationshipThread, ...] = ()
    associations: tuple[RelationshipAssociation, ...] = ()
    open_loops: tuple[OpenConversationLoop, ...] = ()
    promises: tuple[RelationshipPromise, ...] = ()
    shared_experiences: tuple[SharedExperience, ...] = ()
    milestones: tuple[RelationshipMilestone, ...] = ()
    updated_at: str = ""
    available: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "threads",
            "associations",
            "open_loops",
            "promises",
            "shared_experiences",
            "milestones",
            "warnings",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "revision": self.revision,
            "threads": [item.to_dict() for item in self.threads],
            "associations": [item.to_dict() for item in self.associations],
            "open_loops": [item.to_dict() for item in self.open_loops],
            "promises": [item.to_dict() for item in self.promises],
            "shared_experiences": [item.to_dict() for item in self.shared_experiences],
            "milestones": [item.to_dict() for item in self.milestones],
            "updated_at": self.updated_at,
            "available": self.available,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipSnapshot":
        return cls(
            revision=int(data.get("revision", 0)),
            threads=tuple(RelationshipThread.from_dict(item) for item in data.get("threads", [])),
            associations=tuple(
                RelationshipAssociation.from_dict(item) for item in data.get("associations", [])
            ),
            open_loops=tuple(OpenConversationLoop.from_dict(item) for item in data.get("open_loops", [])),
            promises=tuple(RelationshipPromise.from_dict(item) for item in data.get("promises", [])),
            shared_experiences=tuple(
                SharedExperience.from_dict(item) for item in data.get("shared_experiences", [])
            ),
            milestones=tuple(RelationshipMilestone.from_dict(item) for item in data.get("milestones", [])),
            updated_at=str(data.get("updated_at", "")),
            available=bool(data.get("available", True)),
            warnings=_strings(data.get("warnings")),
        )

    @classmethod
    def from_collections(
        cls,
        *,
        revision: int,
        updated_at: str,
        collections: dict[str, dict[str, dict[str, Any]]],
    ) -> "RelationshipSnapshot":
        def values(name: str) -> list[dict[str, Any]]:
            return [collections[name][key] for key in sorted(collections.get(name, {}))]

        associations = values("associations")
        promises = _promise_payloads_with_associations(values("promises"), associations)
        return cls.from_dict(
            {
                "revision": revision,
                "updated_at": updated_at,
                "threads": values("threads"),
                "associations": associations,
                "open_loops": values("open_loops"),
                "promises": promises,
                "shared_experiences": values("shared_experiences"),
                "milestones": values("milestones"),
            }
        )


@dataclass(frozen=True)
class RelationshipMutation:
    id: str
    operation: RelationshipMutationOperation
    reason: str
    thread_id: str = ""
    target_id: str = ""
    payload: dict[str, JsonValue] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.operation not in {
            "create_thread", "update_thread", "archive_thread", "link", "unlink",
            "open_loop", "update_loop", "pause_loop", "close_loop",
            "register_promise", "update_promise", "transition_promise", "fulfill_promise",
            "record_experience", "record_milestone",
        }:
            raise ValueError(f"Unsupported relationship mutation operation: {self.operation}")
        object.__setattr__(self, "payload", _metadata(self.payload))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "operation": self.operation,
            "reason": self.reason,
            "thread_id": self.thread_id,
            "target_id": self.target_id,
            "payload": self.payload,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipMutation":
        return cls(
            id=str(data["id"]),
            operation=data.get("operation", "create_thread"),
            reason=str(data.get("reason", "")),
            thread_id=str(data.get("thread_id", "")),
            target_id=str(data.get("target_id", "")),
            payload=_metadata(data.get("payload")),
            evidence_ids=_strings(data.get("evidence_ids")),
            confidence=float(data.get("confidence", 0.5)),
        )


def _promise_payloads_with_associations(
    promise_records: list[dict[str, Any]],
    association_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    related_fields = {
        "thread": "related_thread_ids",
        "open_loop": "related_loop_ids",
        "memory": "related_memory_ids",
        "curiosity": "related_curiosity_ids",
        "shared_experience": "related_experience_ids",
        "milestone": "related_milestone_ids",
    }
    projected: dict[str, dict[str, set[str]]] = {
        str(item.get("id", "")): {field: set() for field in related_fields.values()}
        for item in promise_records
    }
    for raw in association_records:
        try:
            association = RelationshipAssociation.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            continue
        if association.source_type == "promise" and association.source_id in projected:
            field_name = related_fields.get(association.target_type)
            if field_name:
                projected[association.source_id][field_name].add(association.target_id)
        if association.target_type == "promise" and association.target_id in projected:
            field_name = related_fields.get(association.source_type)
            if field_name:
                projected[association.target_id][field_name].add(association.source_id)

    output: list[dict[str, Any]] = []
    for record in promise_records:
        promise_id = str(record.get("id", ""))
        relations = projected.get(promise_id, {})
        output.append(
            {
                **record,
                **{name: sorted(values) for name, values in relations.items()},
            }
        )
    return output


@dataclass(frozen=True)
class RelationshipMutationResult:
    mutation_id: str
    operation: RelationshipMutationOperation
    status: RelationshipMutationStatus
    entity_id: str = ""
    message: str = ""
    error: str | None = None
    revision: int = 0
    applied_at: str = field(default_factory=relationship_utc_now)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "mutation_id": self.mutation_id,
            "operation": self.operation,
            "status": self.status,
            "entity_id": self.entity_id,
            "message": self.message,
            "error": self.error,
            "revision": self.revision,
            "applied_at": self.applied_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipMutationResult":
        return cls(
            mutation_id=str(data["mutation_id"]),
            operation=data.get("operation", "create_thread"),
            status=data.get("status", "failed"),
            entity_id=str(data.get("entity_id", "")),
            message=str(data.get("message", "")),
            error=data.get("error"),
            revision=int(data.get("revision", 0)),
            applied_at=str(data.get("applied_at") or relationship_utc_now()),
        )
