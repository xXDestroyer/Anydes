from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from anydes.cognition.contracts import (
    CuriosityModel,
    CuriosityMutation,
    CuriosityWriteResult,
    Intention,
    ReflectionResult,
)
from anydes.cognition.curiosity_store import CuriosityStateError, CuriosityStateStore
from anydes.cognition.live_runtime.events import (
    CURIOSITY_CHANGED,
    DomainEvent,
    EventPublisher,
    NullEventPublisher,
)
from anydes.nucleo.config import Config


@dataclass
class CuriosityWriter:
    """The sole semantic write boundary for persistent curiosity state."""

    store: CuriosityStateStore
    events: EventPublisher = field(default_factory=NullEventPublisher)

    @classmethod
    def from_config(
        cls,
        config: Config,
        events: EventPublisher | None = None,
    ) -> "CuriosityWriter":
        return cls(
            store=CuriosityStateStore.from_config(config),
            events=events or NullEventPublisher(),
        )

    def write(
        self,
        intention: Intention,
        reflection: ReflectionResult,
    ) -> tuple[CuriosityWriteResult, ...]:
        active = reflection.revised_intention or intention
        updates = tuple(active.curiosity_updates)
        if not updates:
            return ()
        if not reflection.approved:
            return tuple(
                CuriosityWriteResult(
                    mutation_id=item.id,
                    curiosity_id=item.target_id,
                    operation=item.operation,
                    status="blocked",
                    message=f"Curiosity was not changed because reflection decided {reflection.decision}.",
                    error=reflection.reason or "reflection_not_approved",
                )
                for item in updates
            )

        results: list[CuriosityWriteResult] = []

        def apply(records: dict[str, CuriosityModel]) -> None:
            for mutation in updates:
                results.append(self._apply_mutation(records, mutation))

        try:
            self.store.transact(apply)
        except CuriosityStateError as exc:
            return tuple(
                CuriosityWriteResult(
                    mutation_id=item.id,
                    curiosity_id=item.target_id,
                    operation=item.operation,
                    status="failed",
                    message="Persistent curiosity state could not be updated.",
                    error=type(exc).__name__,
                )
                for item in updates
            )

        saved = tuple(item for item in results if item.status == "saved")
        if saved:
            self._publish_change(saved)
        return tuple(results)

    def _apply_mutation(
        self,
        records: dict[str, CuriosityModel],
        mutation: CuriosityMutation,
    ) -> CuriosityWriteResult:
        curiosity_id = mutation.target_id.strip()
        if mutation.privacy_scope == "protected":
            return _result(mutation, "blocked", "Protected curiosity requires an encrypted store.")
        if not curiosity_id:
            return _result(mutation, "failed", "Curiosity target id is required.")

        now = datetime.now(timezone.utc).isoformat()
        current = records.get(curiosity_id)
        if mutation.operation == "create":
            if current is not None:
                return _result(mutation, "skipped", "Curiosity already exists.")
            records[curiosity_id] = CuriosityModel(
                id=curiosity_id,
                intensity=mutation.intensity,
                topic=mutation.topic.strip(),
                motive=mutation.motive.strip(),
                priority=mutation.priority,
                origin=mutation.origin.strip() or "reasoning",
                appeared_at=now,
                expected_discovery=mutation.expected_discovery.strip(),
                missing_information=mutation.missing_information.strip(),
                evidence=mutation.evidence_ids,
                status="open",
                updated_at=now,
                privacy_scope=mutation.privacy_scope,
                related_thread_ids=mutation.related_thread_ids,
                revision=1,
            )
            return _result(mutation, "saved", "Curiosity created.")

        if current is None:
            return _result(mutation, "failed", "Curiosity target does not exist.")

        if mutation.operation == "update":
            if current.status not in {"open", "paused"}:
                return _result(mutation, "skipped", "Closed curiosity cannot be updated.")
            records[curiosity_id] = replace(
                current,
                topic=mutation.topic.strip() or current.topic,
                motive=mutation.motive.strip() or current.motive,
                expected_discovery=(
                    mutation.expected_discovery.strip() or current.expected_discovery
                ),
                missing_information=(
                    mutation.missing_information.strip() or current.missing_information
                ),
                intensity=mutation.intensity,
                priority=mutation.priority,
                evidence=tuple(dict.fromkeys((*current.evidence, *mutation.evidence_ids))),
                privacy_scope=mutation.privacy_scope,
                related_thread_ids=tuple(
                    dict.fromkeys((*current.related_thread_ids, *mutation.related_thread_ids))
                ),
                updated_at=now,
                revision=current.revision + 1,
            )
            return _result(mutation, "saved", "Curiosity updated.")

        transitions = {
            "satisfy": ({"open", "paused"}, "satisfied"),
            "pause": ({"open"}, "paused"),
            "resume": ({"paused"}, "open"),
            "dismiss": ({"open", "paused"}, "dismissed"),
        }
        allowed_from, target_status = transitions[mutation.operation]
        if current.status not in allowed_from:
            return _result(
                mutation,
                "skipped",
                f"Curiosity cannot transition from {current.status} to {target_status}.",
            )
        records[curiosity_id] = replace(
            current,
            status=target_status,
            satisfied=target_status in {"satisfied", "dismissed"},
            evidence=tuple(dict.fromkeys((*current.evidence, *mutation.evidence_ids))),
            updated_at=now,
            revision=current.revision + 1,
        )
        return _result(mutation, "saved", f"Curiosity transitioned to {target_status}.")

    def _publish_change(self, results: tuple[CuriosityWriteResult, ...]) -> None:
        try:
            self.events.publish(
                DomainEvent(
                    event_type=CURIOSITY_CHANGED,
                    source="CuriosityWriter",
                    aggregate_id="curiosity",
                    payload={
                        "curiosity_ids": tuple(item.curiosity_id for item in results),
                        "operations": tuple(item.operation for item in results),
                        "saved_count": len(results),
                    },
                )
            )
        except Exception:
            return


def _result(
    mutation: CuriosityMutation,
    status: str,
    message: str,
) -> CuriosityWriteResult:
    return CuriosityWriteResult(
        mutation_id=mutation.id,
        curiosity_id=mutation.target_id,
        operation=mutation.operation,
        status=status,
        message=message,
        error=None if status in {"saved", "skipped"} else message,
    )
