from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

from anydes.cognition.live_runtime.contracts import (
    LiveCognitiveSnapshot,
    ProviderMode,
    ProviderSample,
    ProviderUpdatePolicy,
    ProviderUpdateRequest,
    RuntimeHealthSnapshot,
    SlicePublication,
)


T = TypeVar("T")


@runtime_checkable
class LiveStateProvider(Protocol, Generic[T]):
    """Replaceable source of one live cognitive-state slice.

    ``initial_state`` must be fast and local. In simulated mode both methods
    must avoid mandatory external resources and still return a valid value.
    """

    @property
    def provider_id(self) -> str: ...

    @property
    def policy(self) -> ProviderUpdatePolicy: ...

    def initial_state(self, mode: ProviderMode) -> ProviderSample[T]: ...

    def update(self, request: ProviderUpdateRequest[T]) -> ProviderSample[T]: ...


@runtime_checkable
class LiveStateStore(Protocol):
    def snapshot(self) -> LiveCognitiveSnapshot: ...

    def publish(self, publication: SlicePublication[Any]) -> LiveCognitiveSnapshot: ...

    def publish_batch(
        self,
        publications: tuple[SlicePublication[Any], ...],
    ) -> LiveCognitiveSnapshot: ...


@runtime_checkable
class HealthMonitor(Protocol):
    def register(self, provider_id: str, mode: ProviderMode) -> None: ...

    def started(self, provider_id: str) -> None: ...

    def update_started(self, provider_id: str) -> None: ...

    def update_succeeded(self, provider_id: str, duration_seconds: float) -> None: ...

    def update_failed(self, provider_id: str, error: BaseException, duration_seconds: float) -> None: ...

    def stopped(self, provider_id: str) -> None: ...

    def snapshot(
        self,
        state: LiveCognitiveSnapshot,
        policies: dict[str, ProviderUpdatePolicy],
        *,
        running: bool,
    ) -> RuntimeHealthSnapshot: ...


@runtime_checkable
class MetricsCollector(Protocol):
    @property
    def enabled(self) -> bool: ...

    def observe_duration(self, name: str, seconds: float, **labels: str) -> None: ...

    def increment(self, name: str, amount: int = 1, **labels: str) -> None: ...

    def gauge(self, name: str, value: float, **labels: str) -> None: ...


@runtime_checkable
class TaskExecutor(Protocol):
    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]: ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None: ...


ExecutorFactory = Callable[[int], TaskExecutor]


@runtime_checkable
class RuntimeControl(Protocol):
    def snapshot(self) -> LiveCognitiveSnapshot: ...

    def health_snapshot(self) -> RuntimeHealthSnapshot: ...

    def stop(self, *, wait: bool = True) -> None: ...
