from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Generic, Literal, Mapping, TypeVar


ProviderMode = Literal["live", "simulated"]
ProviderTrigger = Literal["startup", "periodic", "event", "manual", "retry"]
ProviderUpdateMode = Literal["event", "periodic", "hybrid", "manual"]
ProviderHealthStatus = Literal["active", "degraded", "stale", "simulated", "stopped"]
FreshnessStatus = Literal["fresh", "acceptable", "stale", "unavailable"]

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ProviderUpdatePolicy:
    """Scheduling and freshness policy owned by a provider registration."""

    update_mode: ProviderUpdateMode = "event"
    interval_seconds: float | None = None
    fresh_for_seconds: float | None = None
    usable_for_seconds: float | None = None
    timeout_seconds: float = 10.0
    priority: int = 50

    def __post_init__(self) -> None:
        if self.update_mode in {"periodic", "hybrid"} and self.interval_seconds is None:
            raise ValueError("Periodic and hybrid providers require interval_seconds.")
        for name in (
            "interval_seconds",
            "fresh_for_seconds",
            "usable_for_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero when configured.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if (
            self.fresh_for_seconds is not None
            and self.usable_for_seconds is not None
            and self.usable_for_seconds < self.fresh_for_seconds
        ):
            raise ValueError("usable_for_seconds cannot be shorter than fresh_for_seconds.")


@dataclass(frozen=True)
class ProviderSample(Generic[T]):
    """A provider result before it is versioned and published by the Store."""

    value: T
    observed_at: datetime = field(default_factory=utc_now)
    source_revision: str | int | None = None
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("ProviderSample.observed_at must be timezone-aware.")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ProviderUpdateRequest(Generic[T]):
    provider_id: str
    mode: ProviderMode
    trigger: ProviderTrigger
    previous: LiveSlice[T] | None = None
    requested_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class SlicePublication(Generic[T]):
    provider_id: str
    sample: ProviderSample[T]
    policy: ProviderUpdatePolicy
    mode: ProviderMode = "live"


@dataclass(frozen=True)
class LiveSlice(Generic[T]):
    """Immutable, versioned last-known-good state for one provider."""

    provider_id: str
    value: T
    revision: int
    global_revision: int
    observed_at: datetime
    published_at: datetime
    source_revision: str | int | None = None
    confidence: float = 1.0
    mode: ProviderMode = "live"
    fresh_for_seconds: float | None = None
    usable_for_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    def age_seconds(self, now: datetime | None = None) -> float:
        moment = now or utc_now()
        return max(0.0, (moment - self.observed_at).total_seconds())

    def freshness_at(self, now: datetime | None = None) -> FreshnessStatus:
        age = self.age_seconds(now)
        if self.fresh_for_seconds is None:
            return "fresh"
        if age <= self.fresh_for_seconds:
            return "fresh"
        if self.usable_for_seconds is None or age <= self.usable_for_seconds:
            return "acceptable"
        return "stale"


@dataclass(frozen=True)
class LiveCognitiveSnapshot:
    """One atomically published root revision of the live cognitive state."""

    revision: int = 0
    slices: Mapping[str, LiveSlice[Any]] = field(default_factory=dict)
    published_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slices", MappingProxyType(dict(self.slices)))

    def get(self, provider_id: str) -> LiveSlice[Any] | None:
        return self.slices.get(provider_id)

    def require(self, provider_id: str) -> LiveSlice[Any]:
        try:
            return self.slices[provider_id]
        except KeyError as exc:
            raise KeyError(f"No live snapshot is published for provider '{provider_id}'.") from exc


@dataclass(frozen=True)
class ProviderHealth:
    provider_id: str
    status: ProviderHealthStatus
    mode: ProviderMode
    freshness: FreshnessStatus = "unavailable"
    last_error: str = ""
    last_updated_at: datetime | None = None
    last_started_at: datetime | None = None
    update_in_progress: bool = False
    update_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    restart_count: int = 0
    actual_frequency_hz: float = 0.0
    last_duration_ms: float = 0.0


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    running: bool
    providers: tuple[ProviderHealth, ...] = ()
    captured_at: datetime = field(default_factory=utc_now)

    @property
    def active_provider_count(self) -> int:
        return sum(item.status in {"active", "simulated"} for item in self.providers)


@dataclass(frozen=True)
class RuntimeSettings:
    mode: ProviderMode = "live"
    max_workers: int = 4
    scheduler_resolution_seconds: float = 0.05
    developer_metrics: bool = False
    snapshot_history_limit: int = 32

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least one.")
        if self.scheduler_resolution_seconds <= 0:
            raise ValueError("scheduler_resolution_seconds must be greater than zero.")
        if self.snapshot_history_limit < 1:
            raise ValueError("snapshot_history_limit must be at least one.")


def immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return immutable_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value
