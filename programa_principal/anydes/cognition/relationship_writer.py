from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from anydes.cognition.contracts import InternalContext, Intention, ReflectionResult
from anydes.cognition.live_runtime.events import (
    RELATIONSHIP_CHANGED,
    DomainEvent,
    EventPublisher,
    NullEventPublisher,
)
from anydes.identity.relationship_contracts import (
    OpenConversationLoop,
    RelationshipAssociation,
    RelationshipMilestone,
    RelationshipMutation,
    RelationshipMutationResult,
    RelationshipPromise,
    PromiseTransition,
    RelationshipThread,
    SharedExperience,
)
from anydes.identity.relationship_store import RelationshipStateStore
from anydes.nucleo.config import Config


class RelationshipWriter:
    """The only semantic writer for reflected relationship mutations."""

    def __init__(
        self,
        store: RelationshipStateStore,
        events: EventPublisher | None = None,
    ) -> None:
        self.store = store
        self.events = events or NullEventPublisher()

    @classmethod
    def from_config(
        cls,
        config: Config,
        events: EventPublisher | None = None,
    ) -> "RelationshipWriter":
        return cls(RelationshipStateStore.from_config(config), events=events)

    def write(
        self,
        intention: Intention,
        reflection: ReflectionResult,
        context: InternalContext,
    ) -> tuple[RelationshipMutationResult, ...]:
        active = reflection.revised_intention or intention
        mutations = tuple(active.relationship_updates)
        if not mutations:
            return ()
        if not reflection.approved:
            return tuple(
                RelationshipMutationResult(
                    mutation_id=item.id,
                    operation=item.operation,
                    status="blocked",
                    message="Relationship mutation was not approved by Reflection.",
                    error=reflection.reason or "reflection_not_approved",
                    revision=context.relationship.revision,
                )
                for item in mutations
            )

        applied: list[tuple[RelationshipMutation, str]] = []

        def mutate(collections: dict[str, dict[str, dict[str, Any]]]) -> None:
            for mutation in _ordered_mutations(mutations):
                entity_id = _apply_mutation(collections, mutation)
                applied.append((mutation, entity_id))

        try:
            document = self.store.transact(
                mutate,
                expected_revision=context.relationship.revision,
            )
        except Exception as exc:
            return tuple(
                RelationshipMutationResult(
                    mutation_id=item.id,
                    operation=item.operation,
                    status="failed",
                    message="Relationship transaction was rolled back.",
                    error=f"{type(exc).__name__}: {exc}",
                    revision=context.relationship.revision,
                )
                for item in mutations
            )

        entities = {mutation.id: entity_id for mutation, entity_id in applied}
        results = tuple(
            RelationshipMutationResult(
                mutation_id=item.id,
                operation=item.operation,
                status="saved",
                entity_id=entities.get(item.id, item.target_id),
                message="Relationship mutation committed.",
                revision=document.revision,
            )
            for item in mutations
        )
        self._publish_change(results, document.revision)
        return results

    def _publish_change(
        self,
        results: tuple[RelationshipMutationResult, ...],
        revision: int,
    ) -> None:
        try:
            self.events.publish(
                DomainEvent(
                    event_type=RELATIONSHIP_CHANGED,
                    source="RelationshipWriter",
                    aggregate_id="relationship",
                    payload={
                        "mutation_ids": tuple(item.mutation_id for item in results),
                        "operations": tuple(item.operation for item in results),
                        "revision": revision,
                    },
                )
            )
        except Exception:
            return


def _apply_mutation(
    collections: dict[str, dict[str, dict[str, Any]]],
    mutation: RelationshipMutation,
) -> str:
    operation = mutation.operation
    now = datetime.now(timezone.utc).isoformat()

    if operation == "create_thread":
        entity_id = mutation.target_id or _new_id("thread")
        records = _collection(collections, "threads")
        _ensure_absent(records, entity_id)
        thread = RelationshipThread.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "created_at": mutation.payload.get("created_at") or now,
                "last_active_at": mutation.payload.get("last_active_at") or now,
            }
        )
        records[entity_id] = thread.to_dict()
        return entity_id

    if operation in {"update_thread", "archive_thread"}:
        records = _collection(collections, "threads")
        current = _require(records, mutation.target_id, "thread")
        updates = dict(mutation.payload)
        if operation == "archive_thread":
            updates["status"] = "archived"
        updates["last_active_at"] = now
        updated = RelationshipThread.from_dict({**current, **updates, "id": mutation.target_id})
        records[mutation.target_id] = updated.to_dict()
        return mutation.target_id

    if operation == "link":
        entity_id = mutation.target_id or str(mutation.payload.get("association_id") or _new_id("association"))
        records = _collection(collections, "associations")
        _ensure_absent(records, entity_id)
        association = RelationshipAssociation.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "thread_id": mutation.payload.get("thread_id") or mutation.thread_id,
                "source_type": mutation.payload.get("source_type") or "thread",
                "source_id": mutation.payload.get("source_id") or mutation.thread_id,
                "evidence_ids": mutation.evidence_ids,
                "linked_at": mutation.payload.get("linked_at") or now,
            }
        )
        _require_relationship_entity(
            collections,
            association.source_type,
            association.source_id,
        )
        _require_relationship_entity(
            collections,
            association.target_type,
            association.target_id,
        )
        records[entity_id] = association.to_dict()
        if association.source_type == "thread":
            _touch_thread(collections, association.source_id, now)
        if association.target_type == "thread":
            _touch_thread(collections, association.target_id, now)
        return entity_id

    if operation == "unlink":
        records = _collection(collections, "associations")
        _require(records, mutation.target_id, "association")
        del records[mutation.target_id]
        return mutation.target_id

    if operation == "open_loop":
        entity_id = mutation.target_id or _new_id("loop")
        records = _collection(collections, "open_loops")
        _ensure_absent(records, entity_id)
        loop = OpenConversationLoop.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "evidence_ids": mutation.evidence_ids,
                "created_at": mutation.payload.get("created_at") or now,
                "last_mentioned_at": mutation.payload.get("last_mentioned_at") or now,
            }
        )
        records[entity_id] = loop.to_dict()
        _associate_created_entity(
            collections,
            thread_id=mutation.thread_id,
            target_type="open_loop",
            target_id=entity_id,
            role="open_conversation",
            evidence_ids=mutation.evidence_ids,
            now=now,
        )
        return entity_id

    if operation in {"update_loop", "pause_loop", "close_loop"}:
        records = _collection(collections, "open_loops")
        current = _require(records, mutation.target_id, "open loop")
        updates = dict(mutation.payload)
        updates["last_mentioned_at"] = now
        if operation == "pause_loop":
            updates["status"] = "paused"
        elif operation == "close_loop":
            updates.update(
                status="closed",
                closed_at=now,
                closure_reason=str(mutation.payload.get("closure_reason") or mutation.reason),
            )
        updated = OpenConversationLoop.from_dict({**current, **updates, "id": mutation.target_id})
        records[mutation.target_id] = updated.to_dict()
        return mutation.target_id

    if operation == "register_promise":
        entity_id = mutation.target_id or _new_id("promise")
        records = _collection(collections, "promises")
        _ensure_absent(records, entity_id)
        promise = RelationshipPromise.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "evidence_ids": mutation.evidence_ids,
                "confidence": mutation.payload.get("confidence", mutation.confidence),
                "creation_time": mutation.payload.get("creation_time") or mutation.payload.get("created_at") or now,
                "last_reference": now,
                "status_changed_at": mutation.payload.get("status_changed_at") or now,
            }
        )
        records[entity_id] = promise.to_record_dict()
        if mutation.thread_id:
            _associate_created_entity(
                collections,
                thread_id=mutation.thread_id,
                target_type="promise",
                target_id=entity_id,
                role="shared_commitment",
                evidence_ids=mutation.evidence_ids,
                now=now,
            )
        return entity_id

    if operation in {"update_promise", "transition_promise", "fulfill_promise"}:
        records = _collection(collections, "promises")
        current = _require(records, mutation.target_id, "promise")
        updates = dict(mutation.payload)
        current_promise = RelationshipPromise.from_dict(current)
        if operation == "update_promise":
            if "status" in updates or "to_status" in updates:
                raise ValueError("Promise status changes require transition_promise.")
        else:
            to_status = "fulfilled" if operation == "fulfill_promise" else str(updates.pop("to_status", ""))
            reason = str(
                updates.pop("status_reason", "")
                or updates.pop("fulfillment_reason", "")
                or mutation.reason
            )
            _ensure_promise_transition(current_promise.status, to_status)
            history = list(current_promise.transition_history)
            history.append(
                PromiseTransition(
                    from_status=current_promise.status,
                    to_status=to_status,
                    reason=reason,
                    evidence_ids=mutation.evidence_ids,
                    changed_at=now,
                )
            )
            updates.update(
                {
                    "status": to_status,
                    "status_reason": reason,
                    "status_changed_at": now,
                    "transition_history": history,
                }
            )
            if to_status == "fulfilled":
                updates["fulfillment_reason"] = reason
        updates["last_reference"] = now
        updated = RelationshipPromise.from_dict({**current, **updates, "id": mutation.target_id})
        records[mutation.target_id] = updated.to_record_dict()
        return mutation.target_id

    if operation == "record_experience":
        entity_id = mutation.target_id or _new_id("experience")
        records = _collection(collections, "shared_experiences")
        _ensure_absent(records, entity_id)
        experience = SharedExperience.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "evidence_ids": mutation.evidence_ids,
                "occurred_at": mutation.payload.get("occurred_at") or now,
            }
        )
        records[entity_id] = experience.to_dict()
        _associate_created_entity(
            collections,
            thread_id=mutation.thread_id,
            target_type="shared_experience",
            target_id=entity_id,
            role="shared_experience",
            evidence_ids=mutation.evidence_ids,
            now=now,
        )
        return entity_id

    if operation == "record_milestone":
        entity_id = mutation.target_id or _new_id("milestone")
        records = _collection(collections, "milestones")
        _ensure_absent(records, entity_id)
        milestone = RelationshipMilestone.from_dict(
            {
                **mutation.payload,
                "id": entity_id,
                "evidence_ids": mutation.evidence_ids,
                "occurred_at": mutation.payload.get("occurred_at") or now,
            }
        )
        records[entity_id] = milestone.to_dict()
        _associate_created_entity(
            collections,
            thread_id=mutation.thread_id,
            target_type="milestone",
            target_id=entity_id,
            role="relationship_milestone",
            evidence_ids=mutation.evidence_ids,
            now=now,
        )
        return entity_id

    raise ValueError(f"Unsupported relationship mutation: {operation}")


def _associate_created_entity(
    collections: dict[str, dict[str, dict[str, Any]]],
    *,
    thread_id: str,
    target_type: str,
    target_id: str,
    role: str,
    evidence_ids: tuple[str, ...],
    now: str,
) -> None:
    _require(_collection(collections, "threads"), thread_id, "thread")
    association = RelationshipAssociation(
        id=_new_id("association"),
        thread_id=thread_id,
        target_type=target_type,
        target_id=target_id,
        relationship_role=role,
        relevance=0.7,
        evidence_ids=evidence_ids,
        linked_at=now,
        source_type="thread",
        source_id=thread_id,
    )
    _collection(collections, "associations")[association.id] = association.to_dict()
    _touch_thread(collections, thread_id, now)


def _touch_thread(
    collections: dict[str, dict[str, dict[str, Any]]],
    thread_id: str,
    now: str,
) -> None:
    records = _collection(collections, "threads")
    current = _require(records, thread_id, "thread")
    records[thread_id] = RelationshipThread.from_dict(
        {**current, "id": thread_id, "last_active_at": now}
    ).to_dict()


def _collection(
    collections: dict[str, dict[str, dict[str, Any]]],
    name: str,
) -> dict[str, dict[str, Any]]:
    return collections.setdefault(name, {})


def _require(records: dict[str, dict[str, Any]], entity_id: str, label: str) -> dict[str, Any]:
    if entity_id not in records:
        raise ValueError(f"Unknown {label}: {entity_id}")
    return records[entity_id]


def _ordered_mutations(
    mutations: tuple[RelationshipMutation, ...],
) -> tuple[RelationshipMutation, ...]:
    creation_operations = {
        "create_thread": 0,
        "open_loop": 1,
        "register_promise": 1,
        "record_experience": 1,
        "record_milestone": 1,
        "link": 3,
        "unlink": 4,
    }
    return tuple(
        item
        for _, item in sorted(
            enumerate(mutations),
            key=lambda pair: (creation_operations.get(pair[1].operation, 2), pair[0]),
        )
    )


def _require_relationship_entity(
    collections: dict[str, dict[str, dict[str, Any]]],
    entity_type: str,
    entity_id: str,
) -> None:
    collection_name = {
        "thread": "threads",
        "open_loop": "open_loops",
        "promise": "promises",
        "shared_experience": "shared_experiences",
        "milestone": "milestones",
    }.get(entity_type)
    if collection_name is not None:
        _require(_collection(collections, collection_name), entity_id, entity_type)


def _ensure_promise_transition(from_status: str, to_status: str) -> None:
    allowed = {
        "pending": {"active", "fulfilled", "expired", "cancelled", "forgotten"},
        "active": {"fulfilled", "expired", "cancelled", "forgotten"},
        "forgotten": {"active", "fulfilled", "cancelled"},
        "fulfilled": set(),
        "expired": set(),
        "cancelled": set(),
    }
    if to_status not in allowed.get(from_status, set()):
        raise ValueError(f"Invalid promise transition: {from_status} -> {to_status}")


def _ensure_absent(records: dict[str, dict[str, Any]], entity_id: str) -> None:
    if entity_id in records:
        raise ValueError(f"Relationship entity already exists: {entity_id}")


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid4().hex}"
