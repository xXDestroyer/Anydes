from __future__ import annotations

import os
import json
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Generic, Iterator, Mapping, TypeVar

from anydes.cognition.live_runtime.interfaces import MetricsCollector


T = TypeVar("T")

_ACTIVE_INTERACTION: ContextVar[
    tuple["DeveloperInteractionTracer", str] | None
] = ContextVar("anydes_active_interaction", default=None)
_ACTIVE_TRACE_LABELS: ContextVar[Mapping[str, str]] = ContextVar(
    "anydes_active_trace_labels",
    default=MappingProxyType({}),
)

_STAGE_ALIASES = {
    "pipeline.observation_reception": "observation_reception",
    "pipeline.context_planning": "context_planning",
    "pipeline.internal_context_assembler": "internal_context",
    "pipeline.context_builder": "internal_context",
    "pipeline.memory_manager": "memory_search",
    "pipeline.memory_search": "memory_search",
    "pipeline.reasoning": "reasoning_engine",
    "pipeline.reflection": "reflection",
    "pipeline.action_manager": "action_manager",
    "pipeline.tts": "tts",
    "pipeline.total": "response_ready",
}

_REQUIRED_TRACE_STAGES = (
    "observation_reception",
    "internal_context",
    "memory_search",
    "context_planning",
    "reasoning_engine",
    "reflection",
    "action_manager",
    "text_generation",
    "tts",
    "response_ready",
)

_ALLOWED_TRACE_STATS = frozenset(
    {
        "input_characters",
        "internal_context_characters",
        "internal_context_approx_tokens",
        "memories_retrieved",
        "open_loops_considered",
        "promises_considered",
        "curiosities_considered",
        "observations_considered",
        "selected_providers",
        "model_calls",
        "active_model_calls",
        "shadow_model_calls",
        "model_prompt_characters",
        "model_prompt_approx_tokens",
        "active_model_prompt_characters",
        "active_model_prompt_approx_tokens",
        "shadow_model_prompt_characters",
        "shadow_model_prompt_approx_tokens",
        "final_prompt_characters",
        "final_prompt_approx_tokens",
        "model_response_characters",
    }
)


@dataclass(frozen=True)
class ModelCallTrace:
    stage: str
    cycle: str
    duration_ms: float
    prompt_characters: int
    approximate_tokens: int
    response_characters: int
    status: str
    error_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "cycle": self.cycle,
            "duration_ms": round(self.duration_ms, 3),
            "prompt_characters": self.prompt_characters,
            "approximate_tokens": self.approximate_tokens,
            "response_characters": self.response_characters,
            "status": self.status,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class InteractionTraceReport:
    interaction_id: str
    interaction_kind: str
    started_at: str
    completed_at: str
    response_ready_ms: float
    experience_complete_ms: float
    slow: bool
    threshold_ms: float
    status: str
    error_type: str
    stages_ms: Mapping[str, float] = field(default_factory=dict)
    measurements: Mapping[str, int] = field(default_factory=dict)
    model_calls: tuple[ModelCallTrace, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages_ms", MappingProxyType(dict(self.stages_ms)))
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "interaction_id": self.interaction_id,
            "interaction_kind": self.interaction_kind,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "response_ready_ms": round(self.response_ready_ms, 3),
            "experience_complete_ms": round(self.experience_complete_ms, 3),
            "slow": self.slow,
            "threshold_ms": round(self.threshold_ms, 3),
            "status": self.status,
            "error_type": self.error_type,
            "stages_ms": {
                key: round(value, 3)
                for key, value in sorted(self.stages_ms.items())
            },
            "measurements": dict(sorted(self.measurements.items())),
            "model_calls": [item.to_dict() for item in self.model_calls],
        }


@dataclass
class _MutableInteractionTrace:
    interaction_id: str
    interaction_kind: str
    started_at: str
    started_perf: float
    threshold_ms: float
    stages_ms: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    measurements: dict[str, int] = field(default_factory=dict)
    model_calls: list[ModelCallTrace] = field(default_factory=list)
    completed_at: str = ""
    response_ready_ms: float = 0.0
    experience_complete_ms: float = 0.0
    status: str = "running"
    error_type: str = ""

    def __post_init__(self) -> None:
        for stage in _REQUIRED_TRACE_STAGES:
            self.stages_ms.setdefault(stage, 0.0)

    def report(self) -> InteractionTraceReport:
        return InteractionTraceReport(
            interaction_id=self.interaction_id,
            interaction_kind=self.interaction_kind,
            started_at=self.started_at,
            completed_at=self.completed_at,
            response_ready_ms=self.response_ready_ms,
            experience_complete_ms=self.experience_complete_ms,
            slow=(
                max(self.response_ready_ms, self.experience_complete_ms)
                > self.threshold_ms
            ),
            threshold_ms=self.threshold_ms,
            status=self.status,
            error_type=self.error_type,
            stages_ms=dict(self.stages_ms),
            measurements=dict(self.measurements),
            model_calls=tuple(self.model_calls),
        )


class DeveloperInteractionTracer:
    """Developer-only per-turn traces containing no conversational payloads."""

    def __init__(
        self,
        path: Path,
        slow_path: Path,
        *,
        slow_threshold_ms: float = 5000.0,
        history_limit: int = 128,
    ) -> None:
        if slow_threshold_ms <= 0:
            raise ValueError("slow_threshold_ms must be positive.")
        self.path = path
        self.slow_path = slow_path
        self.slow_threshold_ms = float(slow_threshold_ms)
        self._lock = threading.RLock()
        self._history_limit = max(1, history_limit)
        self._history: deque[InteractionTraceReport] = deque(
            maxlen=self._history_limit
        )
        self._traces: dict[str, _MutableInteractionTrace] = {}

    @contextmanager
    def interaction(self, interaction_kind: str) -> Iterator[str]:
        interaction_id = f"turn:{uuid.uuid4().hex}"
        trace = _MutableInteractionTrace(
            interaction_id=interaction_id,
            interaction_kind=interaction_kind,
            started_at=datetime.now(timezone.utc).isoformat(),
            started_perf=time.perf_counter(),
            threshold_ms=self.slow_threshold_ms,
        )
        with self._lock:
            self._traces[interaction_id] = trace
        token = _ACTIVE_INTERACTION.set((self, interaction_id))
        try:
            yield interaction_id
        except Exception as exc:
            trace.status = "failed"
            trace.error_type = type(exc).__name__
            raise
        finally:
            _ACTIVE_INTERACTION.reset(token)
            self._finalize(interaction_id)

    def record_duration(
        self,
        interaction_id: str,
        name: str,
        seconds: float,
    ) -> None:
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is None:
                return
            milliseconds = max(0.0, float(seconds)) * 1000.0
            trace.stages_ms[name] += milliseconds
            alias = _STAGE_ALIASES.get(name)
            if alias and alias != name:
                trace.stages_ms[alias] += milliseconds

    def set_measurement(self, interaction_id: str, name: str, value: int) -> None:
        if name not in _ALLOWED_TRACE_STATS:
            raise ValueError(f"Unsupported interaction measurement: {name}")
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is not None:
                trace.measurements[name] = max(0, int(value))

    def increment_measurement(
        self,
        interaction_id: str,
        name: str,
        amount: int = 1,
    ) -> None:
        if name not in _ALLOWED_TRACE_STATS:
            raise ValueError(f"Unsupported interaction measurement: {name}")
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is not None:
                trace.measurements[name] = max(
                    0,
                    trace.measurements.get(name, 0) + int(amount),
                )

    def record_model_call(
        self,
        interaction_id: str,
        *,
        stage: str,
        cycle: str,
        seconds: float,
        prompt_characters: int,
        approximate_tokens: int,
        response_characters: int,
        status: str,
        error_type: str = "",
    ) -> None:
        call = ModelCallTrace(
            stage=stage,
            cycle=cycle,
            duration_ms=max(0.0, seconds) * 1000.0,
            prompt_characters=max(0, prompt_characters),
            approximate_tokens=max(0, approximate_tokens),
            response_characters=max(0, response_characters),
            status=status,
            error_type=error_type,
        )
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is None:
                return
            trace.model_calls.append(call)
            trace.stages_ms[stage] += call.duration_ms
            is_shadow = stage.startswith("unified_shadow")
            route = "shadow" if is_shadow else "active"
            trace.measurements[f"{route}_model_calls"] = (
                trace.measurements.get(f"{route}_model_calls", 0) + 1
            )
            trace.measurements[f"{route}_model_prompt_characters"] = (
                trace.measurements.get(f"{route}_model_prompt_characters", 0)
                + call.prompt_characters
            )
            trace.measurements[f"{route}_model_prompt_approx_tokens"] = (
                trace.measurements.get(f"{route}_model_prompt_approx_tokens", 0)
                + call.approximate_tokens
            )
            if stage.startswith("text_generation") and stage != "text_generation":
                trace.stages_ms["text_generation"] += call.duration_ms
            trace.measurements["model_calls"] = len(trace.model_calls)
            trace.measurements["model_prompt_characters"] = (
                trace.measurements.get("model_prompt_characters", 0)
                + call.prompt_characters
            )
            trace.measurements["model_prompt_approx_tokens"] = (
                trace.measurements.get("model_prompt_approx_tokens", 0)
                + call.approximate_tokens
            )
            trace.measurements["model_response_characters"] = (
                trace.measurements.get("model_response_characters", 0)
                + call.response_characters
            )
            if stage.startswith("text_generation"):
                trace.measurements["final_prompt_characters"] = (
                    call.prompt_characters
                )
                trace.measurements["final_prompt_approx_tokens"] = (
                    call.approximate_tokens
                )

    def record_external_duration(
        self,
        interaction_id: str,
        name: str,
        seconds: float,
    ) -> None:
        self.record_duration(interaction_id, name, seconds)
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is None or not trace.completed_at:
                return
            trace.experience_complete_ms = max(
                trace.response_ready_ms,
                (time.perf_counter() - trace.started_perf) * 1000.0,
            )
            report = trace.report()
            self._store_history(report)
            self._append(self.path, report, event="external_stage_completed")
            if report.slow:
                self._append(
                    self.slow_path,
                    report,
                    event="external_stage_completed",
                )

    def latest(self) -> InteractionTraceReport | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def reports(self) -> tuple[InteractionTraceReport, ...]:
        with self._lock:
            return tuple(self._history)

    def _finalize(self, interaction_id: str) -> None:
        with self._lock:
            trace = self._traces.get(interaction_id)
            if trace is None:
                return
            elapsed_ms = max(
                0.0,
                (time.perf_counter() - trace.started_perf) * 1000.0,
            )
            trace.completed_at = datetime.now(timezone.utc).isoformat()
            trace.response_ready_ms = elapsed_ms
            trace.experience_complete_ms = elapsed_ms
            if trace.status == "running":
                trace.status = "completed"
            report = trace.report()
            self._store_history(report)
            self._append(self.path, report, event="response_ready")
            if report.slow:
                self._append(self.slow_path, report, event="response_ready")
            while len(self._traces) > self._history_limit:
                oldest_id = next(iter(self._traces))
                if oldest_id == interaction_id and len(self._traces) == 1:
                    break
                self._traces.pop(oldest_id, None)

    def _store_history(self, report: InteractionTraceReport) -> None:
        retained = [
            item
            for item in self._history
            if item.interaction_id != report.interaction_id
        ]
        self._history.clear()
        self._history.extend(retained)
        self._history.append(report)

    @staticmethod
    def _append(
        path: Path,
        report: InteractionTraceReport,
        *,
        event: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, **report.to_dict()}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
            )


@dataclass(frozen=True)
class DurationMetric:
    name: str
    count: int
    latest_ms: float
    average_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class PerformanceSnapshot:
    durations: tuple[DurationMetric, ...] = ()
    counters: Mapping[str, int] = field(default_factory=dict)
    gauges: Mapping[str, float] = field(default_factory=dict)
    cpu_percent: float | None = None
    memory_mb: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "counters", MappingProxyType(dict(self.counters)))
        object.__setattr__(self, "gauges", MappingProxyType(dict(self.gauges)))


class NullMetricsCollector:
    """Production default with no allocations and no retained telemetry."""

    @property
    def enabled(self) -> bool:
        return False

    def observe_duration(self, name: str, seconds: float, **labels: str) -> None:
        return None

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        return None

    def gauge(self, name: str, value: float, **labels: str) -> None:
        return None

    @property
    def interaction_tracer(self) -> None:
        return None


class InMemoryMetricsCollector:
    """Bounded developer-only metrics collector without conversation payloads."""

    def __init__(self, *, samples_per_metric: int = 128) -> None:
        if samples_per_metric < 1:
            raise ValueError("samples_per_metric must be at least one.")
        self._lock = threading.RLock()
        self._sample_limit = samples_per_metric
        self._durations: dict[str, deque[float]] = {}
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._resources = ProcessResourceSampler()
        self._interaction_tracer: DeveloperInteractionTracer | None = None

    @property
    def enabled(self) -> bool:
        return True

    def observe_duration(self, name: str, seconds: float, **labels: str) -> None:
        key = _metric_key(name, labels)
        with self._lock:
            samples = self._durations.setdefault(key, deque(maxlen=self._sample_limit))
            samples.append(max(0.0, float(seconds)) * 1000.0)

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        key = _metric_key(name, labels)
        with self._lock:
            self._counters[key] += int(amount)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        key = _metric_key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    @property
    def interaction_tracer(self) -> DeveloperInteractionTracer | None:
        return self._interaction_tracer

    def enable_interaction_tracing(
        self,
        path: Path,
        slow_path: Path,
        *,
        slow_threshold_ms: float = 5000.0,
    ) -> DeveloperInteractionTracer:
        with self._lock:
            if self._interaction_tracer is None:
                self._interaction_tracer = DeveloperInteractionTracer(
                    path,
                    slow_path,
                    slow_threshold_ms=slow_threshold_ms,
                )
            return self._interaction_tracer

    def snapshot(self, *, include_resources: bool = True) -> PerformanceSnapshot:
        with self._lock:
            durations = tuple(
                DurationMetric(
                    name=name,
                    count=len(samples),
                    latest_ms=samples[-1],
                    average_ms=sum(samples) / len(samples),
                    maximum_ms=max(samples),
                )
                for name, samples in sorted(self._durations.items())
                if samples
            )
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        cpu, memory = self._resources.sample() if include_resources else (None, None)
        return PerformanceSnapshot(
            durations=durations,
            counters=counters,
            gauges=gauges,
            cpu_percent=cpu,
            memory_mb=memory,
        )


class ProcessResourceSampler:
    """Low-cost process sampler; psutil is optional and never required to start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_wall = time.monotonic()
        self._last_cpu = time.process_time()
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except (ImportError, OSError):
            self._process = None

    def sample(self) -> tuple[float | None, float | None]:
        with self._lock:
            now_wall = time.monotonic()
            now_cpu = time.process_time()
            elapsed_wall = max(1e-9, now_wall - self._last_wall)
            elapsed_cpu = max(0.0, now_cpu - self._last_cpu)
            self._last_wall = now_wall
            self._last_cpu = now_cpu
            cpu = min(100.0, elapsed_cpu / elapsed_wall * 100.0 / max(1, os.cpu_count() or 1))
            memory = None
            if self._process is not None:
                try:
                    memory = self._process.memory_info().rss / (1024.0 * 1024.0)
                except (AttributeError, OSError):
                    memory = None
            if memory is None:
                memory = _stdlib_memory_mb()
            return cpu, memory


@dataclass(frozen=True)
class PipelineProfiler(Generic[T]):
    collector: MetricsCollector

    def call(
        self,
        name: str,
        callback: Callable[[], T],
        **labels: str,
    ) -> T:
        if not getattr(self.collector, "enabled", True):
            return callback()
        started = time.perf_counter()
        previous_labels = dict(_ACTIVE_TRACE_LABELS.get())
        trace_labels = {**previous_labels, **labels, "pipeline_stage": name}
        label_token = _ACTIVE_TRACE_LABELS.set(
            MappingProxyType(trace_labels)
        )
        try:
            return callback()
        finally:
            _ACTIVE_TRACE_LABELS.reset(label_token)
            elapsed = time.perf_counter() - started
            self.collector.observe_duration(name, elapsed, **labels)
            record_current_stage(name, elapsed, **labels)

    @contextmanager
    def interaction(self, interaction_kind: str) -> Iterator[str]:
        tracer = getattr(self.collector, "interaction_tracer", None)
        if tracer is None:
            yield ""
            return
        with tracer.interaction(interaction_kind) as interaction_id:
            yield interaction_id

    def call_external(
        self,
        name: str,
        callback: Callable[[], T],
        *,
        interaction_id: str = "",
        **labels: str,
    ) -> T:
        if not getattr(self.collector, "enabled", True):
            return callback()
        started = time.perf_counter()
        try:
            return callback()
        finally:
            elapsed = time.perf_counter() - started
            self.collector.observe_duration(name, elapsed, **labels)
            tracer = getattr(self.collector, "interaction_tracer", None)
            if tracer is not None and interaction_id:
                tracer.record_external_duration(interaction_id, name, elapsed)
            else:
                record_current_stage(name, elapsed)


def current_interaction_id() -> str:
    active = _ACTIVE_INTERACTION.get()
    return active[1] if active is not None else ""


def record_current_stage(name: str, seconds: float, **labels: str) -> None:
    active = _ACTIVE_INTERACTION.get()
    if active is not None:
        active[0].record_duration(active[1], name, seconds)
        cycle = str(labels.get("cycle") or "").strip()
        if cycle:
            active[0].record_duration(
                active[1],
                f"{name}.{cycle}",
                seconds,
            )


def set_current_interaction_measurement(name: str, value: int) -> None:
    active = _ACTIVE_INTERACTION.get()
    if active is not None:
        active[0].set_measurement(active[1], name, value)


def record_current_model_call(
    *,
    stage: str,
    seconds: float,
    prompt_characters: int,
    approximate_tokens: int,
    response_characters: int,
    status: str,
    error_type: str = "",
) -> None:
    active = _ACTIVE_INTERACTION.get()
    if active is None:
        return
    cycle = str(_ACTIVE_TRACE_LABELS.get().get("cycle") or "")
    active[0].record_model_call(
        active[1],
        stage=stage,
        cycle=cycle,
        seconds=seconds,
        prompt_characters=prompt_characters,
        approximate_tokens=approximate_tokens,
        response_characters=response_characters,
        status=status,
        error_type=error_type,
    )


def approximate_token_count(character_count: int) -> int:
    """Language-agnostic estimate for diagnostics, never for billing."""
    count = max(0, int(character_count))
    return (count + 3) // 4


def _metric_key(name: str, labels: Mapping[str, str]) -> str:
    if not labels:
        return name
    safe_labels = ",".join(f"{key}={labels[key]}" for key in sorted(labels))
    return f"{name}{{{safe_labels}}}"


def _stdlib_memory_mb() -> float | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_process_memory.restype = wintypes.BOOL
            process = get_current_process()
            success = get_process_memory(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return None

    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    @property
    def enabled(self) -> bool:
        return False
