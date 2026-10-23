from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterator


class VisionResidencyState(str, Enum):
    OFF = "vision_off"
    LOADING = "vision_loading"
    READY = "vision_ready"
    ACTIVE = "vision_active"
    COOLING_DOWN = "vision_cooling_down"


@dataclass(frozen=True)
class VisionResidencyEvent:
    event: str
    monotonic: float
    state: str
    reason: str = ""
    duration_ms: float = 0.0
    hot: bool = False


class ElasticVisionResidencyManager:
    """Own the local vision model lifetime without changing vision semantics."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        competing_model: str,
        competing_num_ctx: int,
        grace_period_seconds: float = 120.0,
        enabled: bool = True,
        request_timeout_seconds: float = 60.0,
        audit=None,
        opener=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.competing_model = competing_model
        self.competing_num_ctx = int(competing_num_ctx)
        self.grace_period_seconds = max(0.0, float(grace_period_seconds))
        self.enabled = bool(enabled and model)
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self._audit = audit
        self._opener = opener or urllib.request.urlopen
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._state = VisionResidencyState.OFF
        self._active_count = 0
        self._loading_thread: threading.Thread | None = None
        self._cooldown_timer: threading.Timer | None = None
        self._last_request_monotonic = 0.0
        self._last_ready_monotonic = 0.0
        self._closed = False
        self._load_count = 0
        self._hot_reuse_count = 0
        self._unload_count = 0
        self._load_times_ms: list[float] = []
        self._unload_times_ms: list[float] = []
        self._events: list[VisionResidencyEvent] = []
        self._last_error = ""

    @property
    def state(self) -> VisionResidencyState:
        with self._condition:
            return self._state

    @property
    def keep_alive(self) -> str:
        seconds = max(1, int(round(self.grace_period_seconds)))
        return f"{seconds}s"

    def prefetch(self, *, reason: str = "vision_intent") -> bool:
        if not self.enabled:
            return False
        with self._condition:
            if self._closed:
                return False
            self._last_request_monotonic = self._clock()
            if self._state in {
                VisionResidencyState.READY,
                VisionResidencyState.ACTIVE,
                VisionResidencyState.COOLING_DOWN,
            }:
                self._hot_reuse_count += 1
                self._cancel_cooldown_locked()
                if self._state == VisionResidencyState.COOLING_DOWN:
                    self._transition_locked(VisionResidencyState.READY, "hot_reuse")
                return True
            if self._state == VisionResidencyState.LOADING:
                return True
            self._transition_locked(VisionResidencyState.LOADING, reason)
            worker = threading.Thread(
                target=self._load_worker,
                args=(reason,),
                name="anydes-vision-elastic-load",
                daemon=True,
            )
            self._loading_thread = worker
            worker.start()
            return True

    def wait_until_ready(self, *, deadline: float | None = None) -> bool:
        if not self.enabled:
            return False
        current = self.state
        if current == VisionResidencyState.OFF:
            self.prefetch(reason="vision_wait")
        elif current == VisionResidencyState.COOLING_DOWN:
            self.prefetch(reason="vision_wait")
        with self._condition:
            while self._state == VisionResidencyState.LOADING and not self._closed:
                timeout = self.request_timeout_seconds
                if deadline is not None:
                    timeout = max(0.0, min(timeout, deadline - self._clock()))
                if timeout <= 0:
                    return False
                self._condition.wait(timeout=timeout)
            return self._state in {
                VisionResidencyState.READY,
                VisionResidencyState.ACTIVE,
                VisionResidencyState.COOLING_DOWN,
            }

    @contextmanager
    def active_vision(self, *, deadline: float | None = None) -> Iterator[bool]:
        ready = self.wait_until_ready(deadline=deadline)
        if not ready:
            yield False
            return
        with self._condition:
            self._cancel_cooldown_locked()
            self._active_count += 1
            self._transition_locked(VisionResidencyState.ACTIVE, "vision_inference")
        try:
            yield True
        finally:
            with self._condition:
                self._active_count = max(0, self._active_count - 1)
                self._last_request_monotonic = self._clock()
                if self._active_count == 0 and not self._closed:
                    self._transition_locked(
                        VisionResidencyState.COOLING_DOWN,
                        "vision_complete",
                    )
                    self._schedule_cooldown_locked()

    def release_if_idle(self, *, reason: str = "grace_expired") -> dict[str, object]:
        with self._condition:
            if self._active_count > 0:
                return {"status": "busy", "state": self._state.value, "reason": reason}
            if self._state == VisionResidencyState.LOADING:
                return {"status": "busy", "state": self._state.value, "reason": reason}
            self._cancel_cooldown_locked()
        started = self._clock()
        resident = self.is_resident()
        if resident:
            try:
                self._set_residency(self.model, "0", num_ctx=None)
            except (OSError, ValueError, TimeoutError, urllib.error.URLError) as exc:
                with self._condition:
                    self._last_error = type(exc).__name__
                return {
                    "status": "failed",
                    "state": self.state.value,
                    "reason": reason,
                    "error_type": type(exc).__name__,
                }
        duration_ms = (self._clock() - started) * 1000.0
        with self._condition:
            if resident:
                self._unload_count += 1
                self._unload_times_ms.append(duration_ms)
            self._transition_locked(VisionResidencyState.OFF, reason, duration_ms=duration_ms)
            self._condition.notify_all()
        return {
            "status": "completed" if resident else "not_resident",
            "state": VisionResidencyState.OFF.value,
            "reason": reason,
            "release_ms": round(duration_ms, 3),
            "model": self.model,
        }

    def note_competing_model_loaded(self) -> None:
        if not self.enabled:
            return
        if self.is_resident():
            return
        with self._condition:
            if self._state != VisionResidencyState.OFF:
                self._cancel_cooldown_locked()
                self._transition_locked(
                    VisionResidencyState.OFF,
                    "evicted_by_conversation_model",
                )
                self._condition.notify_all()

    def is_resident(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/ps", method="GET")
            with self._opener(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            target = self.model.partition(":")[0].casefold()
            return any(
                str(item.get("name") or item.get("model") or "")
                .partition(":")[0]
                .casefold()
                == target
                for item in payload.get("models", [])
                if isinstance(item, dict)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, urllib.error.URLError):
            return self.state != VisionResidencyState.OFF

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "enabled": self.enabled,
                "model": self.model,
                "state": self._state.value,
                "grace_period_seconds": self.grace_period_seconds,
                "active_count": self._active_count,
                "load_count": self._load_count,
                "hot_reuse_count": self._hot_reuse_count,
                "unload_count": self._unload_count,
                "load_times_ms": list(self._load_times_ms),
                "unload_times_ms": list(self._unload_times_ms),
                "last_error": self._last_error,
                "events": [asdict(item) for item in self._events[-256:]],
            }

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._cancel_cooldown_locked()
        self.release_if_idle(reason="application_close")

    def _load_worker(self, reason: str) -> None:
        started = self._clock()
        error = ""
        try:
            if self.competing_model:
                self._set_residency(self.competing_model, "0", num_ctx=self.competing_num_ctx)
            self._set_residency(self.model, self.keep_alive, num_ctx=None)
        except (OSError, ValueError, TimeoutError, urllib.error.URLError) as exc:
            error = type(exc).__name__
        duration_ms = (self._clock() - started) * 1000.0
        with self._condition:
            self._loading_thread = None
            self._load_times_ms.append(duration_ms)
            if error:
                self._last_error = error
                self._transition_locked(VisionResidencyState.OFF, "load_failed", duration_ms)
            else:
                self._load_count += 1
                self._last_ready_monotonic = self._clock()
                self._transition_locked(VisionResidencyState.READY, reason, duration_ms)
            self._condition.notify_all()

    def _set_residency(self, model: str, keep_alive: str, *, num_ctx: int | None) -> None:
        payload: dict[str, object] = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        if num_ctx:
            payload["options"] = {"num_ctx": num_ctx}
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self.request_timeout_seconds) as response:
            response.read()

    def _schedule_cooldown_locked(self) -> None:
        self._cancel_cooldown_locked()
        if self.grace_period_seconds <= 0:
            timer = threading.Timer(0.0, self.release_if_idle)
        else:
            timer = threading.Timer(
                self.grace_period_seconds,
                lambda: self.release_if_idle(reason="grace_expired"),
            )
        timer.daemon = True
        self._cooldown_timer = timer
        timer.start()

    def _cancel_cooldown_locked(self) -> None:
        timer = self._cooldown_timer
        self._cooldown_timer = None
        if timer is not None:
            timer.cancel()

    def _transition_locked(
        self,
        state: VisionResidencyState,
        reason: str,
        duration_ms: float = 0.0,
    ) -> None:
        self._state = state
        event = VisionResidencyEvent(
            event="vision_residency_state",
            monotonic=self._clock(),
            state=state.value,
            reason=reason,
            duration_ms=round(duration_ms, 3),
            hot=reason == "hot_reuse",
        )
        self._events.append(event)
        if self._audit is not None:
            try:
                details = asdict(event)
                details.pop("event", None)
                self._audit.record(event.event, **details)
            except Exception:
                pass


_active_manager: ElasticVisionResidencyManager | None = None
_active_lock = threading.Lock()


def set_active_vision_residency_manager(
    manager: ElasticVisionResidencyManager | None,
) -> None:
    global _active_manager
    with _active_lock:
        _active_manager = manager


def get_active_vision_residency_manager() -> ElasticVisionResidencyManager | None:
    with _active_lock:
        return _active_manager


def build_vision_residency_manager(config, *, audit=None) -> ElasticVisionResidencyManager:
    return ElasticVisionResidencyManager(
        base_url=config.ollama_url,
        model=config.resolved_vision_model(),
        competing_model=config.realtime_model,
        competing_num_ctx=config.realtime_num_ctx,
        grace_period_seconds=config.vision_grace_period_seconds,
        enabled=config.vision_elastic_residency_enabled,
        audit=audit,
    )
