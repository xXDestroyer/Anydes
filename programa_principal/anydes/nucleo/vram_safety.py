from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

from anydes.voz.process import hidden_process_kwargs


class VramPressure(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class VramSample:
    monotonic: float
    total_mb: float
    used_mb: float
    free_mb: float
    temperature_c: float = 0.0
    utilization_percent: float = 0.0
    tts_allocated_mb: float = 0.0
    tts_reserved_mb: float = 0.0
    tts_metrics_age_seconds: float = -1.0


@dataclass(frozen=True)
class VramActionResult:
    name: str
    pressure: str
    started_monotonic: float
    duration_ms: float
    status: str
    free_before_mb: float
    free_after_mb: float
    recovered_mb: float
    details: dict[str, object]


@dataclass(frozen=True)
class _RegisteredAction:
    name: str
    callback: Callable[[], dict[str, object] | None]
    warning: bool
    critical: bool
    blocked_by_tasks: tuple[str, ...]


class VramSafetyManager:
    """Low-frequency VRAM guard with deterministic, non-semantic actions."""

    def __init__(
        self,
        *,
        warning_free_mb: float = 500.0,
        critical_free_mb: float = 250.0,
        emergency_floor_mb: float = 96.0,
        poll_interval_seconds: float = 2.0,
        action_cooldown_seconds: float = 8.0,
        trend_window: int = 8,
        tts_metrics_path: Path | None = None,
        event_log_path: Path | None = None,
        sampler: Callable[[], VramSample] | None = None,
        audit=None,
    ) -> None:
        if critical_free_mb >= warning_free_mb:
            raise ValueError("critical_free_mb must be lower than warning_free_mb")
        self.warning_free_mb = float(warning_free_mb)
        self.critical_free_mb = float(critical_free_mb)
        self.emergency_floor_mb = float(emergency_floor_mb)
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.action_cooldown_seconds = max(1.0, float(action_cooldown_seconds))
        self.trend_window = max(3, int(trend_window))
        self.tts_metrics_path = Path(tts_metrics_path) if tts_metrics_path else None
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self._sampler = sampler or self._sample_nvidia_smi
        self._audit = audit
        self._lock = threading.RLock()
        self._action_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: deque[VramSample] = deque(maxlen=max(4096, self.trend_window * 8))
        self._actions: list[_RegisteredAction] = []
        self._action_results: deque[VramActionResult] = deque(maxlen=256)
        self._active_tasks: dict[str, int] = {}
        self._state = VramPressure.NORMAL
        self._state_counts = {item.value: 0 for item in VramPressure}
        self._last_action_monotonic = 0.0
        self._last_action_free_mb = -1.0
        self._last_action_pressure: VramPressure | None = None
        self._last_periodic_log = 0.0

    @property
    def state(self) -> VramPressure:
        with self._lock:
            return self._state

    @property
    def latest_sample(self) -> VramSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def register_action(
        self,
        name: str,
        callback: Callable[[], dict[str, object] | None],
        *,
        warning: bool = True,
        critical: bool = True,
        blocked_by_tasks: tuple[str, ...] = (),
    ) -> None:
        action = _RegisteredAction(
            name=name,
            callback=callback,
            warning=bool(warning),
            critical=bool(critical),
            blocked_by_tasks=tuple(blocked_by_tasks),
        )
        with self._lock:
            self._actions = [item for item in self._actions if item.name != name]
            self._actions.append(action)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="anydes-vram-safety",
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.poll_interval_seconds * 2.0))
        with self._lock:
            self._thread = None

    def sample_now(self, *, apply_actions: bool = True) -> VramSample:
        sample = self._sampler()
        previous_state: VramPressure
        with self._lock:
            previous_state = self._state
            self._samples.append(sample)
            self._state = self._classify(sample.free_mb)
            current_state = self._state
            if current_state != previous_state:
                self._state_counts[current_state.value] += 1
        if current_state != previous_state:
            self._record_event(
                "vram_state_changed",
                previous=previous_state.value,
                current=current_state.value,
                free_mb=round(sample.free_mb, 1),
                trend_mb_per_minute=round(self.trend_mb_per_minute(), 2),
            )
        now = time.monotonic()
        if now - self._last_periodic_log >= 30.0:
            self._last_periodic_log = now
            self._record_event(
                "vram_sample",
                state=current_state.value,
                free_mb=round(sample.free_mb, 1),
                used_mb=round(sample.used_mb, 1),
                tts_reserved_mb=round(sample.tts_reserved_mb, 1),
                trend_mb_per_minute=round(self.trend_mb_per_minute(), 2),
            )
        if apply_actions and current_state != VramPressure.NORMAL:
            self._maybe_apply_actions(current_state, sample)
        return sample

    def prepare_for_task(
        self,
        name: str,
        *,
        primary: bool,
        explicitly_requested: bool = True,
    ) -> bool:
        sample = self.latest_sample
        if sample is None or time.monotonic() - sample.monotonic > self.poll_interval_seconds * 1.75:
            sample = self.sample_now(apply_actions=True)
        state = self.state
        if not primary and not explicitly_requested and state != VramPressure.NORMAL:
            self._record_event(
                "vram_task_deferred",
                task=name,
                state=state.value,
                reason="secondary_gpu_work_under_pressure",
            )
            return False
        if primary and sample.free_mb < self.emergency_floor_mb:
            self._maybe_apply_actions(VramPressure.CRITICAL, sample, force=True)
            refreshed = self.sample_now(apply_actions=False)
            if refreshed.free_mb < self.emergency_floor_mb:
                self._record_event(
                    "vram_primary_task_suspended",
                    task=name,
                    free_mb=round(refreshed.free_mb, 1),
                    emergency_floor_mb=self.emergency_floor_mb,
                )
                return False
        return True

    @contextmanager
    def active_task(self, name: str) -> Iterator[None]:
        with self._lock:
            self._active_tasks[name] = self._active_tasks.get(name, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                remaining = self._active_tasks.get(name, 0) - 1
                if remaining > 0:
                    self._active_tasks[name] = remaining
                else:
                    self._active_tasks.pop(name, None)

    def task_active(self, name: str) -> bool:
        with self._lock:
            return self._active_tasks.get(name, 0) > 0

    def trend_mb_per_minute(self) -> float:
        with self._lock:
            samples = list(self._samples)[-self.trend_window :]
        if len(samples) < 2:
            return 0.0
        elapsed = samples[-1].monotonic - samples[0].monotonic
        if elapsed <= 0:
            return 0.0
        return (samples[-1].free_mb - samples[0].free_mb) / elapsed * 60.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            samples = list(self._samples)
            actions = list(self._action_results)
            state = self._state
            counts = dict(self._state_counts)
            active = dict(self._active_tasks)
        return {
            "state": state.value,
            "thresholds_mb": {
                "warning": self.warning_free_mb,
                "critical": self.critical_free_mb,
                "emergency_floor": self.emergency_floor_mb,
            },
            "state_entries": counts,
            "sample_count": len(samples),
            "minimum_free_mb": min((item.free_mb for item in samples), default=0.0),
            "maximum_used_mb": max((item.used_mb for item in samples), default=0.0),
            "trend_mb_per_minute": self.trend_mb_per_minute(),
            "active_tasks": active,
            "latest_sample": asdict(samples[-1]) if samples else None,
            "actions": [asdict(item) for item in actions],
        }

    def sample_history(self) -> tuple[VramSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_now(apply_actions=True)
            except Exception as exc:
                self._record_event("vram_monitor_failure", error_type=type(exc).__name__)
            self._stop.wait(self.poll_interval_seconds)

    def _classify(self, free_mb: float) -> VramPressure:
        if free_mb < self.critical_free_mb:
            return VramPressure.CRITICAL
        if free_mb <= self.warning_free_mb:
            return VramPressure.WARNING
        return VramPressure.NORMAL

    def _maybe_apply_actions(
        self,
        pressure: VramPressure,
        sample: VramSample,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_action_monotonic < self.action_cooldown_seconds:
            return
        if (
            not force
            and self._last_action_pressure == pressure
            and self._last_action_free_mb >= 0
            and sample.free_mb >= self._last_action_free_mb - 32.0
            and self.trend_mb_per_minute() >= -64.0
        ):
            return
        if not self._action_lock.acquire(blocking=False):
            return
        self._last_action_monotonic = now
        self._last_action_free_mb = sample.free_mb
        self._last_action_pressure = pressure
        try:
            with self._lock:
                actions = list(self._actions)
                active_tasks = dict(self._active_tasks)
            current_free = sample.free_mb
            for action in actions:
                if pressure == VramPressure.WARNING and not action.warning:
                    continue
                if pressure == VramPressure.CRITICAL and not action.critical:
                    continue
                if any(active_tasks.get(name, 0) > 0 for name in action.blocked_by_tasks):
                    self._record_event(
                        "vram_action_skipped",
                        action=action.name,
                        state=pressure.value,
                        reason="protected_task_active",
                    )
                    continue
                started = time.monotonic()
                status = "completed"
                details: dict[str, object] = {}
                try:
                    details = dict(action.callback() or {})
                    status = str(details.pop("status", "completed"))
                except Exception as exc:
                    status = "failed"
                    details = {"error_type": type(exc).__name__}
                try:
                    after = self._sampler()
                    with self._lock:
                        self._samples.append(after)
                    free_after = after.free_mb
                except Exception:
                    free_after = current_free
                result = VramActionResult(
                    name=action.name,
                    pressure=pressure.value,
                    started_monotonic=started,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    status=status,
                    free_before_mb=current_free,
                    free_after_mb=free_after,
                    recovered_mb=free_after - current_free,
                    details=details,
                )
                with self._lock:
                    self._action_results.append(result)
                self._record_event("vram_action", **asdict(result))
                current_free = free_after
                if current_free > self.warning_free_mb:
                    break
        finally:
            self._action_lock.release()

    def _sample_nvidia_smi(self) -> VramSample:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            **hidden_process_kwargs(),
        )
        values = [float(item.strip()) for item in completed.stdout.strip().split(",")]
        tts = self._read_tts_metrics()
        return VramSample(
            monotonic=time.monotonic(),
            total_mb=values[0],
            used_mb=values[1],
            free_mb=values[2],
            temperature_c=values[3],
            utilization_percent=values[4],
            tts_allocated_mb=float(tts.get("allocated_mb") or 0.0),
            tts_reserved_mb=float(tts.get("reserved_mb") or 0.0),
            tts_metrics_age_seconds=float(tts.get("age_seconds") or -1.0),
        )

    def _read_tts_metrics(self) -> dict[str, object]:
        path = self.tts_metrics_path
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            timestamp = float(payload.get("monotonic") or 0.0)
            payload["age_seconds"] = max(0.0, time.monotonic() - timestamp)
            return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _record_event(self, event: str, **details: object) -> None:
        if self._audit is not None:
            try:
                self._audit.record(event, **details)
            except Exception:
                pass
        path = self.event_log_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"monotonic": time.monotonic(), "event": event, "details": details}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except OSError:
            pass


_active_manager: VramSafetyManager | None = None
_active_manager_lock = threading.Lock()


def set_active_vram_safety_manager(manager: VramSafetyManager | None) -> None:
    global _active_manager
    with _active_manager_lock:
        _active_manager = manager


def get_active_vram_safety_manager() -> VramSafetyManager | None:
    with _active_manager_lock:
        return _active_manager


def build_vram_safety_manager(config, *, audit=None) -> VramSafetyManager:
    return VramSafetyManager(
        warning_free_mb=config.vram_warning_free_mb,
        critical_free_mb=config.vram_critical_free_mb,
        emergency_floor_mb=config.vram_emergency_floor_mb,
        poll_interval_seconds=config.vram_poll_interval_seconds,
        action_cooldown_seconds=config.vram_action_cooldown_seconds,
        tts_metrics_path=config.vram_tts_metrics_path,
        event_log_path=config.vram_event_log_path,
        audit=audit,
    )


def release_ollama_model(base_url: str, model: str) -> dict[str, object]:
    payload = {"model": model, "stream": False, "keep_alive": "0"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    return {
        "status": "completed",
        "model": model,
        "release_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
