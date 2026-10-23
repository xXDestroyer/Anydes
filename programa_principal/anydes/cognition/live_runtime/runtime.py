from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from anydes.cognition.live_runtime.contracts import (
    LiveCognitiveSnapshot,
    ProviderMode,
    ProviderSample,
    ProviderTrigger,
    ProviderUpdatePolicy,
    ProviderUpdateRequest,
    RuntimeHealthSnapshot,
    RuntimeSettings,
    SlicePublication,
)
from anydes.cognition.live_runtime.interfaces import (
    ExecutorFactory,
    HealthMonitor,
    LiveStateProvider,
    LiveStateStore,
    MetricsCollector,
    TaskExecutor,
)


@dataclass
class _InFlight:
    future: Future[ProviderSample[Any]]
    started_at: float
    deadline: float
    trigger: ProviderTrigger
    timed_out: bool = False


class LiveCognitiveRuntime:
    """Coordinates replaceable providers without depending on their internals."""

    def __init__(
        self,
        *,
        store: LiveStateStore,
        health: HealthMonitor,
        metrics: MetricsCollector,
        executor_factory: ExecutorFactory,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.store = store
        self.health_monitor = health
        self.metrics = metrics
        self.settings = settings or RuntimeSettings()
        self._executor_factory = executor_factory
        self._providers: dict[str, LiveStateProvider[Any]] = {}
        self._policies: dict[str, ProviderUpdatePolicy] = {}
        self._modes: dict[str, ProviderMode] = {}
        self._condition = threading.Condition(threading.RLock())
        self._due: dict[str, float] = {}
        self._triggers: dict[str, ProviderTrigger] = {}
        self._inflight: dict[str, _InFlight] = {}
        self._running = False
        self._scheduler: threading.Thread | None = None
        self._executor: TaskExecutor | None = None

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running

    def register(
        self,
        provider: LiveStateProvider[Any],
        *,
        mode: ProviderMode | None = None,
    ) -> None:
        provider_id = provider.provider_id.strip()
        if not provider_id:
            raise ValueError("Provider ids cannot be empty.")
        with self._condition:
            if self._running:
                raise RuntimeError("Providers must be registered before the runtime starts.")
            if provider_id in self._providers:
                raise ValueError(f"Provider '{provider_id}' is already registered.")
            selected_mode = mode or self.settings.mode
            self._providers[provider_id] = provider
            self._policies[provider_id] = provider.policy
            self._modes[provider_id] = selected_mode
            self.health_monitor.register(provider_id, selected_mode)

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._executor = self._executor_factory(self.settings.max_workers)
            self._running = True

        for provider_id, provider in self._providers.items():
            self.health_monitor.started(provider_id)
            self._publish_initial(provider_id, provider)

        with self._condition:
            now = time.monotonic()
            for provider_id, policy in self._policies.items():
                if policy.update_mode in {"periodic", "hybrid"}:
                    self._due[provider_id] = now + float(policy.interval_seconds or 0.0)
                    self._triggers[provider_id] = "periodic"
            self._scheduler = threading.Thread(
                target=self._scheduler_loop,
                name="anydes-live-runtime",
                daemon=True,
            )
            self._scheduler.start()
        self.metrics.gauge("active_providers", float(len(self._providers)))

    def stop(self, *, wait: bool = True) -> None:
        with self._condition:
            if not self._running:
                return
            self._running = False
            self._condition.notify_all()
            scheduler = self._scheduler
            executor = self._executor
        if scheduler is not None:
            scheduler.join(timeout=3.0)
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
        for provider_id in self._providers:
            self.health_monitor.stopped(provider_id)
        self.metrics.gauge("active_providers", 0.0)
        self.metrics.gauge("provider_updates_in_progress", 0.0)
        with self._condition:
            self._scheduler = None
            self._executor = None
            self._due.clear()
            self._triggers.clear()
            self._inflight.clear()

    def request_update(
        self,
        provider_id: str,
        *,
        trigger: ProviderTrigger = "event",
    ) -> None:
        with self._condition:
            if provider_id not in self._providers:
                raise KeyError(f"Unknown provider '{provider_id}'.")
            if not self._running:
                raise RuntimeError("The live cognitive runtime is not running.")
            self._due[provider_id] = time.monotonic()
            self._triggers[provider_id] = trigger
            self._condition.notify_all()

    def snapshot(self) -> LiveCognitiveSnapshot:
        return self.store.snapshot()

    def health_snapshot(self) -> RuntimeHealthSnapshot:
        return self.health_monitor.snapshot(
            self.store.snapshot(),
            dict(self._policies),
            running=self.is_running,
        )

    def _publish_initial(
        self,
        provider_id: str,
        provider: LiveStateProvider[Any],
    ) -> None:
        started = time.perf_counter()
        self.health_monitor.update_started(provider_id)
        try:
            sample = provider.initial_state(self._modes[provider_id])
            self.store.publish(self._publication(provider_id, sample))
        except Exception as exc:
            duration = time.perf_counter() - started
            self.health_monitor.update_failed(provider_id, exc, duration)
            self.metrics.increment("provider_initialization_failure", provider=provider_id)
            return
        duration = time.perf_counter() - started
        self.health_monitor.update_succeeded(provider_id, duration)
        self.metrics.observe_duration("provider_initialization", duration, provider=provider_id)

    def _scheduler_loop(self) -> None:
        while True:
            with self._condition:
                if not self._running:
                    return
                now = time.monotonic()
                self._mark_timeouts(now)
                ready = [
                    provider_id
                    for provider_id, due_at in self._due.items()
                    if due_at <= now and provider_id not in self._inflight
                ]
                for provider_id in sorted(
                    ready,
                    key=lambda item: self._policies[item].priority,
                    reverse=True,
                ):
                    self._due.pop(provider_id, None)
                    trigger = self._triggers.pop(provider_id, "periodic")
                    self._dispatch(provider_id, trigger)
                timeout = self._next_wait(now)
                self._condition.wait(timeout=timeout)

    def _dispatch(self, provider_id: str, trigger: ProviderTrigger) -> None:
        executor = self._executor
        if executor is None:
            return
        provider = self._providers[provider_id]
        previous = self.store.snapshot().get(provider_id)
        request = ProviderUpdateRequest(
            provider_id=provider_id,
            mode=self._modes[provider_id],
            trigger=trigger,
            previous=previous,
        )
        started_at = time.monotonic()
        self.health_monitor.update_started(provider_id)
        future = executor.submit(provider.update, request)
        self._inflight[provider_id] = _InFlight(
            future=future,
            started_at=started_at,
            deadline=started_at + self._policies[provider_id].timeout_seconds,
            trigger=trigger,
        )
        self.metrics.gauge("provider_updates_in_progress", float(len(self._inflight)))
        future.add_done_callback(
            lambda completed, current_id=provider_id: self._update_completed(current_id, completed)
        )

    def _update_completed(
        self,
        provider_id: str,
        future: Future[ProviderSample[Any]],
    ) -> None:
        duration = 0.0
        timed_out = False
        with self._condition:
            flight = self._inflight.pop(provider_id, None)
            if flight is not None:
                duration = time.monotonic() - flight.started_at
                timed_out = flight.timed_out
            running = self._running
            self.metrics.gauge("provider_updates_in_progress", float(len(self._inflight)))

        if not timed_out and running:
            try:
                sample = future.result()
                self.store.publish(self._publication(provider_id, sample))
            except Exception as exc:
                self.health_monitor.update_failed(provider_id, exc, duration)
                self.metrics.increment("provider_update_failure", provider=provider_id)
            else:
                self.health_monitor.update_succeeded(provider_id, duration)
                self.metrics.observe_duration("provider_update", duration, provider=provider_id)

        with self._condition:
            if self._running and provider_id not in self._due:
                policy = self._policies[provider_id]
                if policy.update_mode in {"periodic", "hybrid"}:
                    self._due[provider_id] = time.monotonic() + float(policy.interval_seconds or 0.0)
                    self._triggers[provider_id] = "periodic"
            self._condition.notify_all()

    def _mark_timeouts(self, now: float) -> None:
        for provider_id, flight in self._inflight.items():
            if flight.timed_out or now < flight.deadline:
                continue
            flight.timed_out = True
            duration = now - flight.started_at
            error = TimeoutError(
                f"Provider '{provider_id}' exceeded {self._policies[provider_id].timeout_seconds:.3f}s."
            )
            self.health_monitor.update_failed(provider_id, error, duration)
            self.metrics.increment("provider_update_timeout", provider=provider_id)

    def _next_wait(self, now: float) -> float:
        deadlines = list(self._due.values())
        deadlines.extend(
            flight.deadline
            for flight in self._inflight.values()
            if not flight.timed_out
        )
        if not deadlines:
            return self.settings.scheduler_resolution_seconds
        return max(
            0.001,
            min(self.settings.scheduler_resolution_seconds, min(deadlines) - now),
        )

    def _publication(
        self,
        provider_id: str,
        sample: ProviderSample[Any],
    ) -> SlicePublication[Any]:
        return SlicePublication(
            provider_id=provider_id,
            sample=sample,
            policy=self._policies[provider_id],
            mode=self._modes[provider_id],
        )
