from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from anydes.cognition.live_runtime.contracts import RuntimeSettings
from anydes.cognition.live_runtime.health import ProviderHealthMonitor
from anydes.cognition.live_runtime.metrics import (
    InMemoryMetricsCollector,
    NullMetricsCollector,
)
from anydes.cognition.live_runtime.runtime import LiveCognitiveRuntime
from anydes.cognition.live_runtime.state_store import AtomicLiveStateStore


@dataclass(frozen=True)
class LiveRuntimeBundle:
    runtime: LiveCognitiveRuntime
    metrics: InMemoryMetricsCollector | NullMetricsCollector


def build_live_runtime(settings: RuntimeSettings | None = None) -> LiveRuntimeBundle:
    """Application composition root for the default local implementation."""

    selected = settings or RuntimeSettings()
    metrics = (
        InMemoryMetricsCollector()
        if selected.developer_metrics
        else NullMetricsCollector()
    )
    runtime = LiveCognitiveRuntime(
        store=AtomicLiveStateStore(history_limit=selected.snapshot_history_limit),
        health=ProviderHealthMonitor(),
        metrics=metrics,
        executor_factory=lambda workers: ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="anydes-provider",
        ),
        settings=selected,
    )
    return LiveRuntimeBundle(runtime=runtime, metrics=metrics)
