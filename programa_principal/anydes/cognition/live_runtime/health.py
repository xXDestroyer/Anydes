from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from anydes.cognition.live_runtime.contracts import (
    LiveCognitiveSnapshot,
    ProviderHealth,
    ProviderMode,
    ProviderUpdatePolicy,
    RuntimeHealthSnapshot,
    utc_now,
)


@dataclass
class _MutableHealth:
    provider_id: str
    mode: ProviderMode
    running: bool = False
    update_in_progress: bool = False
    last_error: str = ""
    last_updated_at: datetime | None = None
    last_started_at: datetime | None = None
    update_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    restart_count: int = 0
    last_duration_ms: float = 0.0
    successful_updates: deque[datetime] = field(default_factory=lambda: deque(maxlen=32))


class ProviderHealthMonitor:
    """Thread-safe operational health read model for the developer dashboard."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, _MutableHealth] = {}

    def register(self, provider_id: str, mode: ProviderMode) -> None:
        if not provider_id.strip():
            raise ValueError("provider_id cannot be empty.")
        with self._lock:
            if provider_id in self._providers:
                raise ValueError(f"Provider '{provider_id}' is already registered.")
            self._providers[provider_id] = _MutableHealth(provider_id, mode)

    def started(self, provider_id: str) -> None:
        now = utc_now()
        with self._lock:
            item = self._require(provider_id)
            if item.last_started_at is not None:
                item.restart_count += 1
            item.running = True
            item.last_started_at = now

    def update_started(self, provider_id: str) -> None:
        with self._lock:
            self._require(provider_id).update_in_progress = True

    def update_succeeded(self, provider_id: str, duration_seconds: float) -> None:
        now = utc_now()
        with self._lock:
            item = self._require(provider_id)
            item.update_in_progress = False
            item.last_error = ""
            item.last_updated_at = now
            item.update_count += 1
            item.consecutive_failures = 0
            item.last_duration_ms = max(0.0, duration_seconds * 1000.0)
            item.successful_updates.append(now)

    def update_failed(
        self,
        provider_id: str,
        error: BaseException,
        duration_seconds: float,
    ) -> None:
        with self._lock:
            item = self._require(provider_id)
            item.update_in_progress = False
            item.last_error = f"{type(error).__name__}: {error}"[:1000]
            item.failure_count += 1
            item.consecutive_failures += 1
            item.last_duration_ms = max(0.0, duration_seconds * 1000.0)

    def stopped(self, provider_id: str) -> None:
        with self._lock:
            item = self._require(provider_id)
            item.running = False
            item.update_in_progress = False

    def snapshot(
        self,
        state: LiveCognitiveSnapshot,
        policies: dict[str, ProviderUpdatePolicy],
        *,
        running: bool,
    ) -> RuntimeHealthSnapshot:
        now = utc_now()
        with self._lock:
            reports = tuple(
                self._report(item, state, policies.get(item.provider_id), now)
                for item in sorted(self._providers.values(), key=lambda value: value.provider_id)
            )
        return RuntimeHealthSnapshot(running=running, providers=reports, captured_at=now)

    def _report(
        self,
        item: _MutableHealth,
        state: LiveCognitiveSnapshot,
        policy: ProviderUpdatePolicy | None,
        now: datetime,
    ) -> ProviderHealth:
        live_slice = state.get(item.provider_id)
        freshness = live_slice.freshness_at(now) if live_slice is not None else "unavailable"
        if not item.running:
            status = "stopped"
        elif item.mode == "simulated":
            status = "simulated"
        elif freshness == "stale":
            status = "stale"
        elif item.consecutive_failures:
            status = "degraded"
        else:
            status = "active"
        return ProviderHealth(
            provider_id=item.provider_id,
            status=status,
            mode=item.mode,
            freshness=freshness,
            last_error=item.last_error,
            last_updated_at=item.last_updated_at,
            last_started_at=item.last_started_at,
            update_in_progress=item.update_in_progress,
            update_count=item.update_count,
            failure_count=item.failure_count,
            consecutive_failures=item.consecutive_failures,
            restart_count=item.restart_count,
            actual_frequency_hz=_frequency(item.successful_updates),
            last_duration_ms=item.last_duration_ms,
        )

    def _require(self, provider_id: str) -> _MutableHealth:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"Unknown provider '{provider_id}'.") from exc


def _frequency(samples: deque[datetime]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = (samples[-1] - samples[0]).total_seconds()
    return (len(samples) - 1) / elapsed if elapsed > 0 else 0.0
