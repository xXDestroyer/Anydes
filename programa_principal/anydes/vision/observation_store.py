from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class VisualObservationRecord:
    """Sensor-owned visual evidence with no conversational or action decision."""

    id: str
    summary: str
    observed_at: datetime
    confidence: float = 0.8
    salience: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("Visual observation timestamps must be timezone-aware.")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "salience", max(0.0, min(1.0, float(self.salience))))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class VisionObservationStore:
    """Thread-safe bounded last-known visual evidence store."""

    def __init__(self, capacity: int = 12) -> None:
        self._records: deque[VisualObservationRecord] = deque(maxlen=max(1, capacity))
        self._lock = threading.RLock()

    def publish(
        self,
        summary: str,
        *,
        confidence: float = 0.8,
        salience: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualObservationRecord:
        record = VisualObservationRecord(
            id=f"vision:observed:{uuid4()}",
            summary=summary.strip(),
            observed_at=datetime.now(timezone.utc),
            confidence=confidence,
            salience=salience,
            metadata=metadata or {},
        )
        with self._lock:
            self._records.append(record)
        return record

    def snapshot(
        self,
        *,
        limit: int = 3,
        max_age_seconds: float | None = 120.0,
    ) -> tuple[VisualObservationRecord, ...]:
        with self._lock:
            records = tuple(self._records)
        if max_age_seconds is not None:
            now = datetime.now(timezone.utc)
            records = tuple(
                item
                for item in records
                if (now - item.observed_at).total_seconds() <= max_age_seconds
            )
        return records[-max(1, limit) :]

    def latest_for_frame(
        self,
        *,
        frame_hash: str,
        source_id: str,
        session_id: str,
        max_age_seconds: float,
    ) -> VisualObservationRecord | None:
        records = self.snapshot(limit=self._records.maxlen, max_age_seconds=max_age_seconds)
        for item in reversed(records):
            metadata = item.metadata
            if (
                metadata.get("frame_hash") == frame_hash
                and metadata.get("frame_source_id") == source_id
                and metadata.get("frame_session_id") == session_id
            ):
                return item
        return None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


_visual_observations = VisionObservationStore()


def get_vision_observation_store() -> VisionObservationStore:
    return _visual_observations
