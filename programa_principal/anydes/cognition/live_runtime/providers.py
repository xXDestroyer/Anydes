from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from anydes.cognition.live_runtime.contracts import (
    ProviderMode,
    ProviderSample,
    ProviderUpdatePolicy,
    ProviderUpdateRequest,
)


T = TypeVar("T")
InitialFactory = Callable[[], ProviderSample[T]]
UpdateFactory = Callable[[ProviderUpdateRequest[T]], ProviderSample[T]]


@dataclass(frozen=True)
class FunctionalLiveProvider(Generic[T]):
    """Small dependency-injection adapter for real and simulated sources.

    Domain providers can implement the protocol directly. This adapter is
    useful for composition roots and tests where callables already exist.
    """

    provider_id: str
    policy: ProviderUpdatePolicy
    live_initial: InitialFactory[T]
    live_update: UpdateFactory[T]
    simulated_initial: InitialFactory[T]
    simulated_update: UpdateFactory[T]

    def initial_state(self, mode: ProviderMode) -> ProviderSample[T]:
        factory = self.simulated_initial if mode == "simulated" else self.live_initial
        return factory()

    def update(self, request: ProviderUpdateRequest[T]) -> ProviderSample[T]:
        factory = self.simulated_update if request.mode == "simulated" else self.live_update
        return factory(request)
