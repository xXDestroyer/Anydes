from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from uuid import uuid4

from anydes.cognition.live_runtime.contracts import (
    ProviderTrigger,
    immutable_mapping,
    utc_now,
)
from anydes.cognition.live_runtime.interfaces import MetricsCollector


MEMORY_CHANGED = "memory.changed"
CONVERSATION_RECORDED = "memory.conversation_recorded"
RELATIONSHIP_CHANGED = "relationship.changed"
SELF_CHANGED = "self.changed"
BODY_CHANGED = "body.changed"
CURIOSITY_CHANGED = "curiosity.changed"


@dataclass(frozen=True)
class DomainEvent:
    event_type: str
    source: str
    aggregate_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"event:{uuid4()}")
    occurred_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("DomainEvent.event_type cannot be empty.")
        if self.occurred_at.tzinfo is None:
            raise ValueError("DomainEvent.occurred_at must be timezone-aware.")
        object.__setattr__(self, "payload", immutable_mapping(self.payload))


EventHandler = Callable[[DomainEvent], None]


@runtime_checkable
class EventPublisher(Protocol):
    def publish(self, event: DomainEvent) -> None: ...


@runtime_checkable
class ProviderUpdateRequester(Protocol):
    @property
    def is_running(self) -> bool: ...

    def request_update(
        self,
        provider_id: str,
        *,
        trigger: ProviderTrigger = "event",
    ) -> None: ...


class NullEventPublisher:
    def publish(self, event: DomainEvent) -> None:
        return None


class CognitiveEventBus:
    """Thread-safe synchronous dispatcher for short, non-blocking handlers."""

    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        self._lock = threading.RLock()
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, event: DomainEvent) -> None:
        with self._lock:
            handlers = tuple(
                (*self._handlers.get(event.event_type, ()), *self._handlers.get("*", ()))
            )
        self._metrics.increment("domain_event_published", event_type=event.event_type)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # An optional consumer cannot roll back an already committed write.
                self._metrics.increment(
                    "domain_event_handler_failure",
                    event_type=event.event_type,
                )


class ProviderInvalidationBridge:
    """Maps domain events to lightweight Runtime refresh requests."""

    def __init__(
        self,
        requester: ProviderUpdateRequester,
        routes: Mapping[str, tuple[str, ...]],
    ) -> None:
        self._requester = requester
        self._routes = immutable_mapping(routes)

    def handle(self, event: DomainEvent) -> None:
        if not self._requester.is_running:
            return
        for provider_id in self._routes.get(event.event_type, ()):
            self._requester.request_update(provider_id, trigger="event")


def connect_default_invalidation_routes(
    bus: CognitiveEventBus,
    requester: ProviderUpdateRequester,
) -> tuple[Callable[[], None], ...]:
    bridge = ProviderInvalidationBridge(
        requester,
        {
            MEMORY_CHANGED: ("memory", "curiosity"),
            CONVERSATION_RECORDED: ("memory", "curiosity"),
            RELATIONSHIP_CHANGED: ("relationship", "curiosity"),
            SELF_CHANGED: ("self",),
            BODY_CHANGED: ("body",),
            CURIOSITY_CHANGED: ("curiosity",),
        },
    )
    return tuple(
        bus.subscribe(event_type, bridge.handle)
        for event_type in (
            MEMORY_CHANGED,
            CONVERSATION_RECORDED,
            RELATIONSHIP_CHANGED,
            SELF_CHANGED,
            BODY_CHANGED,
            CURIOSITY_CHANGED,
        )
    )
