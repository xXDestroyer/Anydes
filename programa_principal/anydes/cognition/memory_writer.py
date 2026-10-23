from __future__ import annotations

import re
from dataclasses import dataclass, field

from anydes.cognition.contracts import Intention, MemoryItem, MemoryWriteResult, ReflectionResult
from anydes.cognition.live_runtime.events import (
    MEMORY_CHANGED,
    DomainEvent,
    EventPublisher,
    NullEventPublisher,
)
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.config import Config


@dataclass
class MemoryWriter:
    """Writes explicit memory candidates after Reflection approval.

    It does not record whole conversations automatically and only persists
    MemoryItem candidates that the Reasoning Engine proposed and Reflection
    approved.
    """

    memory: MemoryManager
    working_limit: int = 32
    events: EventPublisher = field(default_factory=NullEventPublisher)
    _working: dict[str, MemoryItem] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: Config,
        events: EventPublisher | None = None,
    ) -> "MemoryWriter":
        return cls(
            memory=MemoryManager.from_config(config),
            events=events or NullEventPublisher(),
        )

    def write(self, intention: Intention, reflection: ReflectionResult) -> tuple[MemoryWriteResult, ...]:
        active_intention = reflection.revised_intention or intention
        updates = tuple(active_intention.memory_updates)
        if not updates:
            return ()

        if not reflection.approved:
            return tuple(
                MemoryWriteResult(
                    memory_id=item.id,
                    tier=item.tier,
                    status="blocked",
                    message=f"Memory not written because reflection decision was {reflection.decision}.",
                    key=_key_for_item(item),
                    error=reflection.reason or None,
                )
                for item in updates
            )

        results = tuple(self._write_item(item) for item in updates)
        saved = tuple(item for item in results if item.status == "saved")
        if saved:
            self._publish_change(saved)
        return results

    def working_snapshot(self) -> tuple[MemoryItem, ...]:
        return tuple(self._working.values())

    def _write_item(self, item: MemoryItem) -> MemoryWriteResult:
        key = _key_for_item(item)
        content = item.content.strip()
        if not content:
            return MemoryWriteResult(
                memory_id=item.id,
                tier=item.tier,
                status="skipped",
                message="Empty memory candidate was skipped.",
                key=key,
                error="empty_memory_content",
            )

        if item.category == "secret" and item.tier != "protected":
            return MemoryWriteResult(
                memory_id=item.id,
                tier=item.tier,
                status="blocked",
                message="Secret memory candidates must use protected memory.",
                key=key,
                error="secret_requires_protected_memory",
            )

        if item.tier == "working":
            self._remember_working(item)
            return MemoryWriteResult(
                memory_id=item.id,
                tier=item.tier,
                status="saved",
                message=f"Saved in volatile working memory: {key}",
                key=key,
            )

        if item.tier == "short":
            result = self.memory.set_temporary(key, content)
            return _result_from_manager(item, key, result.ok, result.message)

        if item.tier == "long":
            result = self.memory.set_permanent(key, content)
            return _result_from_manager(item, key, result.ok, result.message)

        if item.tier == "protected":
            result = self.memory.set_protected(key, content)
            return _result_from_manager(item, key, result.ok, result.message)

        return MemoryWriteResult(
            memory_id=item.id,
            tier=item.tier,
            status="failed",
            message=f"Unknown memory tier: {item.tier}",
            key=key,
            error="unknown_memory_tier",
        )

    def _remember_working(self, item: MemoryItem) -> None:
        self._working[item.id] = item
        while len(self._working) > self.working_limit:
            oldest_key = next(iter(self._working))
            self._working.pop(oldest_key, None)

    def _publish_change(self, results: tuple[MemoryWriteResult, ...]) -> None:
        try:
            self.events.publish(
                DomainEvent(
                    event_type=MEMORY_CHANGED,
                    source="MemoryWriter",
                    aggregate_id="memory",
                    payload={
                        "memory_ids": tuple(item.memory_id for item in results),
                        "tiers": tuple(item.tier for item in results),
                        "saved_count": len(results),
                    },
                )
            )
        except Exception:
            # A telemetry/invalidation failure cannot undo a committed memory.
            return


def _result_from_manager(item: MemoryItem, key: str, ok: bool, message: str) -> MemoryWriteResult:
    return MemoryWriteResult(
        memory_id=item.id,
        tier=item.tier,
        status="saved" if ok else "failed",
        message=message,
        key=key,
        error=None if ok else "memory_manager_rejected",
    )


def _key_for_item(item: MemoryItem) -> str:
    metadata_key = item.metadata.get("key")
    if isinstance(metadata_key, str) and metadata_key.strip():
        return _slug(metadata_key)
    return _slug(item.id)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned or "memory_item"
