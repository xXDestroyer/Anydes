from __future__ import annotations

import copy
import threading
from collections import deque
from typing import Any

from anydes.cognition.live_runtime.contracts import (
    LiveCognitiveSnapshot,
    LiveSlice,
    SlicePublication,
    utc_now,
)


class AtomicLiveStateStore:
    """Copy-on-write store with constant-time, non-blocking state reads.

    Provider work happens before publication. The lock only protects a root
    pointer swap, so readers never observe partially committed slices.
    """

    def __init__(self, *, history_limit: int = 32) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be at least one.")
        self._lock = threading.RLock()
        self._current = LiveCognitiveSnapshot()
        self._history: deque[LiveCognitiveSnapshot] = deque(
            (self._current,),
            maxlen=history_limit,
        )

    def snapshot(self) -> LiveCognitiveSnapshot:
        with self._lock:
            return self._current

    def snapshot_at(self, revision: int) -> LiveCognitiveSnapshot | None:
        with self._lock:
            for snapshot in reversed(self._history):
                if snapshot.revision == revision:
                    return snapshot
        return None

    def publish(self, publication: SlicePublication[Any]) -> LiveCognitiveSnapshot:
        return self.publish_batch((publication,))

    def publish_batch(
        self,
        publications: tuple[SlicePublication[Any], ...],
    ) -> LiveCognitiveSnapshot:
        if not publications:
            return self.snapshot()
        provider_ids = [item.provider_id for item in publications]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("A batch cannot publish the same provider more than once.")

        # Isolate provider-owned values before taking the short commit lock.
        isolated = tuple(
            (publication, copy.deepcopy(publication.sample.value))
            for publication in publications
        )
        published_at = utc_now()
        with self._lock:
            global_revision = self._current.revision + 1
            slices = dict(self._current.slices)
            for publication, value in isolated:
                previous = slices.get(publication.provider_id)
                slices[publication.provider_id] = LiveSlice(
                    provider_id=publication.provider_id,
                    value=value,
                    revision=(previous.revision + 1) if previous is not None else 1,
                    global_revision=global_revision,
                    observed_at=publication.sample.observed_at,
                    published_at=published_at,
                    source_revision=publication.sample.source_revision,
                    confidence=publication.sample.confidence,
                    mode=publication.mode,
                    fresh_for_seconds=publication.policy.fresh_for_seconds,
                    usable_for_seconds=publication.policy.usable_for_seconds,
                    metadata=publication.sample.metadata,
                )
            self._current = LiveCognitiveSnapshot(
                revision=global_revision,
                slices=slices,
                published_at=published_at,
            )
            self._history.append(self._current)
            return self._current
