from __future__ import annotations

from dataclasses import dataclass, field, replace

from anydes.cognition.contracts import (
    Intention,
    ReflectionResult,
    SelfMutation,
    SelfMutationResult,
    SelfPreference,
    utc_now,
)
from anydes.cognition.live_runtime.events import (
    SELF_CHANGED,
    DomainEvent,
    EventPublisher,
    NullEventPublisher,
)
from anydes.identity.self_preference_store import SelfPreferenceStore
from anydes.nucleo.config import Config


@dataclass
class SelfWriter:
    """The only component authorized to mutate persistent SELF preferences."""

    store: SelfPreferenceStore
    events: EventPublisher = field(default_factory=NullEventPublisher)

    @classmethod
    def from_config(
        cls,
        config: Config,
        events: EventPublisher | None = None,
    ) -> "SelfWriter":
        return cls(
            SelfPreferenceStore.from_config(config),
            events=events or NullEventPublisher(),
        )

    def ensure_preference(self, preference: SelfPreference) -> SelfPreference:
        current = self.store.get(preference.key)
        if current is not None:
            return current
        self.store.put_preference(preference)
        return preference

    def record_realtime_inclination(
        self,
        *,
        key: str,
        domain: str,
        value: str,
        origin: str,
        rationale: str,
        evidence_ids: tuple[str, ...],
        confidence: float = 0.58,
    ) -> SelfPreference | None:
        clean_key = key.strip()
        clean_value = value.strip()
        if not clean_key or not clean_value or not evidence_ids:
            return None
        now = utc_now()
        current = self.store.get(clean_key)
        history_entry = {
            "at": now,
            "origin": origin,
            "candidate": clean_value,
            "reason": rationale,
        }
        if current is None:
            preference = SelfPreference(
                key=clean_key,
                value=clean_value,
                status="emerging",
                confidence=confidence,
                rationale=rationale,
                evidence_ids=evidence_ids,
                reinforcement_count=0,
                domain=domain,
                origin=origin,
                associations=(rationale,) if rationale else (),
                change_history=(history_entry,),
                formed_at=now,
                updated_at=now,
            )
        elif _normalized(current.value) == _normalized(clean_value):
            count = current.reinforcement_count + 1
            next_confidence = min(
                1.0,
                current.confidence * 0.76 + confidence * 0.24 + 0.03,
            )
            preference = replace(
                current,
                status=(
                    "established"
                    if count >= 2 and next_confidence >= 0.62
                    else "emerging"
                ),
                confidence=next_confidence,
                rationale=rationale or current.rationale,
                evidence_ids=tuple(
                    dict.fromkeys((*current.evidence_ids, *evidence_ids))
                ),
                reinforcement_count=count,
                domain=current.domain or domain,
                origin=current.origin or origin,
                associations=tuple(
                    dict.fromkeys(
                        (
                            *current.associations,
                            *((rationale,) if rationale else ()),
                        )
                    )
                ),
                change_history=(*current.change_history, history_entry),
                last_confirmed_at=now,
                updated_at=now,
            )
        else:
            previous_candidates = sum(
                1
                for item in current.change_history
                if _normalized(str(item.get("candidate") or ""))
                == _normalized(clean_value)
            )
            can_evolve = current.status != "established" and previous_candidates >= 1
            if can_evolve:
                alternatives = tuple(
                    dict.fromkeys(
                        (
                            *current.alternatives,
                            current.value,
                        )
                    )
                )
                preference = replace(
                    current,
                    value=clean_value,
                    status="emerging",
                    confidence=confidence,
                    rationale=rationale,
                    evidence_ids=tuple(
                        dict.fromkeys((*current.evidence_ids, *evidence_ids))
                    ),
                    reinforcement_count=1,
                    domain=current.domain or domain,
                    alternatives=alternatives,
                    change_history=(*current.change_history, history_entry),
                    last_confirmed_at=now,
                    updated_at=now,
                )
            else:
                preference = replace(
                    current,
                    status=(
                        current.status
                        if current.status == "established"
                        else "reconsidering"
                    ),
                    alternatives=tuple(
                        dict.fromkeys((*current.alternatives, clean_value))
                    ),
                    change_history=(*current.change_history, history_entry),
                    updated_at=now,
                )
        self.store.put_preference(preference)
        return preference

    def write(
        self,
        intention: Intention,
        reflection: ReflectionResult,
    ) -> tuple[SelfMutationResult, ...]:
        active = reflection.revised_intention or intention
        if not active.self_updates:
            return ()
        if not reflection.approved:
            return tuple(
                SelfMutationResult(
                    mutation_id=item.id,
                    operation=item.operation,
                    key=item.key,
                    status="blocked",
                    message="SELF mutation was not approved by Reflection.",
                    error=reflection.reason or "reflection_rejected",
                )
                for item in active.self_updates
            )
        results = tuple(self._apply(item) for item in active.self_updates)
        saved = tuple(item for item in results if item.status == "saved")
        if saved:
            self._publish_change(saved)
        return results

    def _apply(self, mutation: SelfMutation) -> SelfMutationResult:
        current = self.store.preferences().get(mutation.key)
        if mutation.operation == "form_preference":
            if current is not None:
                return self._result(mutation, "skipped", current, "Preference already exists.")
            preference = SelfPreference(
                key=mutation.key,
                value=mutation.proposed_value,
                status="emerging",
                confidence=mutation.confidence,
                rationale=mutation.reason,
                evidence_ids=mutation.evidence_ids,
                domain=mutation.key.replace("favorite_", "").replace("_", " "),
                origin="reasoning_engine",
                associations=(mutation.reason,) if mutation.reason else (),
                change_history=(
                    {
                        "at": utc_now(),
                        "origin": "reasoning_engine",
                        "candidate": mutation.proposed_value,
                        "reason": mutation.reason,
                    },
                ),
            )
        elif mutation.operation == "reinforce_preference":
            if current is None:
                return self._result(mutation, "failed", None, "Preference does not exist.", "unknown_preference")
            if _normalized(current.value) != _normalized(mutation.proposed_value):
                return self._result(
                    mutation,
                    "blocked",
                    current,
                    "Conflicting value requires reconsider_preference.",
                    "reconsideration_required",
                )
            count = current.reinforcement_count + 1
            confidence = min(1.0, current.confidence * 0.72 + mutation.confidence * 0.28 + 0.04)
            preference = replace(
                current,
                status="established" if count >= 2 and confidence >= 0.78 else "emerging",
                confidence=confidence,
                rationale=mutation.reason,
                evidence_ids=tuple(dict.fromkeys((*current.evidence_ids, *mutation.evidence_ids))),
                reinforcement_count=count,
                last_confirmed_at=utc_now(),
                change_history=(
                    *current.change_history,
                    {
                        "at": utc_now(),
                        "origin": "reasoning_engine",
                        "candidate": mutation.proposed_value,
                        "reason": mutation.reason,
                    },
                ),
                updated_at=utc_now(),
            )
        else:
            if current is None:
                return self._result(mutation, "failed", None, "Preference does not exist.", "unknown_preference")
            preference = replace(
                current,
                value=mutation.proposed_value,
                status="reconsidering",
                confidence=mutation.confidence,
                rationale=mutation.reason,
                evidence_ids=tuple(dict.fromkeys((*current.evidence_ids, *mutation.evidence_ids))),
                reinforcement_count=0,
                alternatives=tuple(
                    dict.fromkeys((*current.alternatives, current.value))
                ),
                change_history=(
                    *current.change_history,
                    {
                        "at": utc_now(),
                        "origin": "reasoning_engine",
                        "candidate": mutation.proposed_value,
                        "reason": mutation.reason,
                    },
                ),
                updated_at=utc_now(),
            )
        self.store.put_preference(preference)
        return self._result(mutation, "saved", preference, "SELF preference updated.")

    def _publish_change(self, results: tuple[SelfMutationResult, ...]) -> None:
        try:
            self.events.publish(
                DomainEvent(
                    event_type=SELF_CHANGED,
                    source="SelfWriter",
                    aggregate_id="self",
                    payload={
                        "mutation_ids": tuple(item.mutation_id for item in results),
                        "operations": tuple(item.operation for item in results),
                        "keys": tuple(item.key for item in results),
                    },
                )
            )
        except Exception:
            return

    @staticmethod
    def _result(
        mutation: SelfMutation,
        status,
        preference: SelfPreference | None,
        message: str,
        error: str | None = None,
    ) -> SelfMutationResult:
        return SelfMutationResult(
            mutation_id=mutation.id,
            operation=mutation.operation,
            key=mutation.key,
            status=status,
            preference=preference,
            message=message,
            error=error,
        )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())
