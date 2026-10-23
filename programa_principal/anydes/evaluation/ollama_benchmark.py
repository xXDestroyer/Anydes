from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from anydes.cerebro.llm import OllamaConversationBrain
from anydes.cognition.contracts import InternalContext
from anydes.cognition.live_runtime.metrics import (
    approximate_token_count,
    record_current_model_call,
)
from anydes.cognition.reasoning import build_grounded_reasoning_prompt


_GROUNDED_BENCHMARK_STAGES = frozenset(
    {
        "unified_shadow_inference",
        "minimal_grounded_inference",
        "understanding_inference",
        "grounded_plan_repair",
    }
)


@dataclass(frozen=True)
class OllamaBenchmarkSettings:
    keep_alive: str | None = None
    think: bool | None = None
    num_predict: int | None = None
    compact_evidence: bool = False

    def __post_init__(self) -> None:
        if self.num_predict is not None and self.num_predict < 1:
            raise ValueError("num_predict must be positive.")


@dataclass(frozen=True)
class OllamaCallMetric:
    stage: str
    wall_ms: float
    time_to_first_token_ms: float
    time_to_first_content_ms: float
    load_ms: float
    prompt_eval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    done_reason: str
    thinking_characters: int
    content_characters: int
    failed: bool = False
    error_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "wall_ms": round(self.wall_ms, 3),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 3),
            "time_to_first_content_ms": round(self.time_to_first_content_ms, 3),
            "load_ms": round(self.load_ms, 3),
            "prompt_eval_ms": round(self.prompt_eval_ms, 3),
            "generation_ms": round(self.generation_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": round(self.tokens_per_second, 3),
            "done_reason": self.done_reason,
            "thinking_characters": self.thinking_characters,
            "content_characters": self.content_characters,
            "failed": self.failed,
            "error_type": self.error_type,
        }


class OllamaCallMetrics:
    """Thread-safe payload-free metrics from Ollama streaming responses."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._calls: list[OllamaCallMetric] = []

    def record(self, metric: OllamaCallMetric) -> None:
        with self._lock:
            self._calls.append(metric)

    def calls(self, *, stage: str | None = None) -> tuple[OllamaCallMetric, ...]:
        with self._lock:
            rows = tuple(self._calls)
        if stage is None:
            return rows
        return tuple(item for item in rows if item.stage == stage)

    def summary(self, *, stage: str | None = None) -> dict[str, object]:
        rows = self.calls(stage=stage)
        if not rows:
            return {
                "call_count": 0,
                "failed_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        input_tokens = sum(item.input_tokens for item in rows)
        output_tokens = sum(item.output_tokens for item in rows)
        generation_seconds = sum(item.generation_ms for item in rows) / 1000.0
        return {
            "call_count": len(rows),
            "failed_calls": sum(1 for item in rows if item.failed),
            "wall_ms": _distribution(item.wall_ms for item in rows),
            "time_to_first_token_ms": _distribution(
                item.time_to_first_token_ms for item in rows
            ),
            "time_to_first_content_ms": _distribution(
                item.time_to_first_content_ms for item in rows
            ),
            "load_ms": _distribution(item.load_ms for item in rows),
            "prompt_eval_ms": _distribution(item.prompt_eval_ms for item in rows),
            "generation_ms": _distribution(item.generation_ms for item in rows),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_per_call": round(input_tokens / len(rows), 3),
            "output_tokens_per_call": round(output_tokens / len(rows), 3),
            "input_tokens_distribution": _distribution(
                item.input_tokens for item in rows
            ),
            "output_tokens_distribution": _distribution(
                item.output_tokens for item in rows
            ),
            "tokens_per_second": round(
                output_tokens / generation_seconds if generation_seconds else 0.0,
                3,
            ),
            "done_reasons": _counts(item.done_reason for item in rows),
            "thinking_calls": sum(1 for item in rows if item.thinking_characters),
        }


class BenchmarkOllamaBrain(OllamaConversationBrain):
    """Ollama brain with transport-only benchmark controls and streaming metrics."""

    def __init__(
        self,
        model: str,
        base_url: str,
        config,
        *,
        settings: OllamaBenchmarkSettings,
        collector: OllamaCallMetrics,
    ) -> None:
        super().__init__(model=model, base_url=base_url, config=config)
        self.benchmark_settings = settings
        self.benchmark_collector = collector

    def unified_intention_model(self):
        return _BenchmarkUnifiedIntentionModel(self)

    def plan_repair_model(self):
        return _BenchmarkPlanRepairModel(self)

    def minimal_grounded_model(self):
        return _BenchmarkMinimalGroundedModel(self)

    def understanding_model(self):
        return _BenchmarkUnderstandingModel(self)

    def _chat(
        self,
        model: str,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        json_mode: bool = False,
        think: bool | None = None,
        deadline: float | None = None,
        max_wait: float | None = None,
        trace_stage: str = "model_call",
    ) -> str:
        settings = self.benchmark_settings
        options: dict[str, object] = {"temperature": temperature}
        if (
            trace_stage in _GROUNDED_BENCHMARK_STAGES
            and settings.num_predict is not None
        ):
            options["num_predict"] = settings.num_predict
        payload: dict[str, object] = {
            "model": model,
            "stream": True,
            "messages": messages,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if settings.keep_alive is not None:
            payload["keep_alive"] = settings.keep_alive
        if (
            trace_stage in _GROUNDED_BENCHMARK_STAGES
            and (settings.think is not None or think is not None)
        ):
            payload["think"] = settings.think if settings.think is not None else think

        prompt_characters = sum(
            len(str(message.get("content") or ""))
            for message in messages
            if isinstance(message, dict)
        )
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        remaining = (
            max(0.0, deadline - time.monotonic())
            if deadline is not None
            else max(5.0, self.config.ollama_turn_timeout_seconds)
        )
        timeout = remaining if max_wait is None else min(remaining, max(0.5, max_wait))
        if timeout < 0.5:
            raise TimeoutError("Ollama turn budget exhausted before the optional stage.")

        started = time.perf_counter()
        first_token_at: float | None = None
        first_content_at: float | None = None
        content_parts: list[str] = []
        thinking_characters = 0
        final: dict[str, Any] = {}
        error_type = ""
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event = json.loads(line.decode("utf-8"))
                    message = event.get("message", {})
                    content = str(message.get("content") or "")
                    thinking = str(message.get("thinking") or "")
                    now = time.perf_counter()
                    if first_token_at is None and (content or thinking):
                        first_token_at = now
                    if first_content_at is None and content:
                        first_content_at = now
                    if content:
                        content_parts.append(content)
                    thinking_characters += len(thinking)
                    if event.get("done"):
                        final = event
            return "".join(content_parts).strip()
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            completed = time.perf_counter()
            wall_ms = (completed - started) * 1000.0
            metric = _call_metric(
                trace_stage=trace_stage,
                wall_ms=wall_ms,
                first_token_ms=(
                    (first_token_at - started) * 1000.0
                    if first_token_at is not None
                    else wall_ms
                ),
                first_content_ms=(
                    (first_content_at - started) * 1000.0
                    if first_content_at is not None
                    else wall_ms
                ),
                final=final,
                thinking_characters=thinking_characters,
                content_characters=sum(len(item) for item in content_parts),
                error_type=error_type,
            )
            self.benchmark_collector.record(metric)
            record_current_model_call(
                stage=trace_stage,
                seconds=wall_ms / 1000.0,
                prompt_characters=prompt_characters,
                approximate_tokens=approximate_token_count(prompt_characters),
                response_characters=metric.content_characters,
                status="failed" if error_type else "completed",
                error_type=error_type,
            )


class _BenchmarkUnifiedIntentionModel:
    def __init__(self, brain: BenchmarkOllamaBrain) -> None:
        self.brain = brain

    def generate_for_context(self, context: InternalContext, prompt: str) -> str:
        selected_prompt = (
            compact_grounded_reasoning_prompt(context)
            if self.brain.benchmark_settings.compact_evidence
            else prompt
        )
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        return self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Eres la unica mente deliberativa de Anydes durante este turno. "
                        "Produce exactamente un plan grounded compacto en JSON y nada mas. "
                        "Compromete primero lo que sabe y lo que respondera; speech solo expresa "
                        "ese compromiso. No describas pasos de razonamiento ni delegues decisiones "
                        "semanticas. Usa el hecho mas especifico, respeta gaps y no inventes ids. "
                        "Las acciones y cambios persistentes son opcionales y deben omitirse si "
                        "la evidencia no los justifica. Responde en espanol natural."
                    ),
                },
                {"role": "user", "content": selected_prompt},
            ],
            temperature=0.0,
            json_mode=True,
            deadline=deadline,
            trace_stage="unified_shadow_inference",
        )


class _BenchmarkPlanRepairModel:
    """One strict, non-thinking repair call for an already rejected plan."""

    def __init__(self, brain: BenchmarkOllamaBrain) -> None:
        self.brain = brain

    def generate(self, prompt: str) -> str:
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        return self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Reparas un GroundedResponsePlan rechazado por Reflection. "
                        "No respondes el turno desde cero, no agregas evidencia ni "
                        "efectos secundarios y no relajas ningun invariante. "
                        "Devuelve exactamente un JSON compacto y nada mas."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
            think=False,
            deadline=deadline,
            trace_stage="grounded_plan_repair",
        )


class _BenchmarkMinimalGroundedModel:
    """Research-only transport for the frozen minimal-plan comparison."""

    def __init__(self, brain: BenchmarkOllamaBrain) -> None:
        self.brain = brain

    def generate_for_context(self, context: InternalContext, prompt: str) -> str:
        _ = context
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        return self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Eres la unica mente deliberativa de Anydes en este "
                        "turno. Decide un compromiso semantico minimo y devuelve "
                        "solamente el JSON solicitado. No produzcas campos de "
                        "Speech, Body, manifiestos ni estado."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
            think=False,
            deadline=deadline,
            trace_stage="minimal_grounded_inference",
        )


class _BenchmarkUnderstandingModel:
    """Research-only transport that interprets a turn without deciding a reply."""

    def __init__(self, brain: BenchmarkOllamaBrain) -> None:
        self.brain = brain

    def generate_for_bundle(self, bundle, prompt: str) -> str:
        _ = bundle
        deadline = time.monotonic() + max(
            5.0,
            self.brain.config.ollama_turn_timeout_seconds,
        )
        return self.brain._chat(
            self.brain.model,
            [
                {
                    "role": "system",
                    "content": (
                        "Interpreta exclusivamente lo comunicado por el "
                        "usuario. No decidas respuestas, acciones, estilo ni "
                        "estado de Anydes. Devuelve solo el UnderstandingFrame "
                        "JSON solicitado y conserva toda incertidumbre."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
            think=False,
            deadline=deadline,
            trace_stage="understanding_inference",
        )


def compact_grounded_reasoning_prompt(context: InternalContext) -> str:
    """Re-encode every evidence field with aliases; no evidence item is removed."""

    standard = build_grounded_reasoning_prompt(context)
    bundle = context.evidence_bundle
    if bundle is None:
        raise ValueError("Grounded reasoning requires an EvidenceBundle.")
    full_json = json.dumps(
        bundle.to_model_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = _alias_keys(bundle.to_model_dict())
    compact_json = json.dumps(
        {
            "key_aliases": _KEY_ALIASES,
            "bundle": encoded,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return standard.replace(full_json, compact_json, 1)


class NvidiaGpuSampler:
    """Samples GPU utilization and VRAM without retaining process payloads."""

    def __init__(self, interval_ms: int = 200) -> None:
        self.interval_ms = max(100, int(interval_ms))
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        creation_flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        self._process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=creation_flags,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            return {"sample_count": 0}
        utilization = [item[0] for item in samples]
        memory = [item[1] for item in samples]
        return {
            "sample_count": len(samples),
            "gpu_utilization_percent": _distribution(utilization),
            "vram_used_mb": _distribution(memory),
            "vram_peak_mb": round(max(memory), 3),
        }

    def _read(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                sample = (float(parts[0]), float(parts[1]))
            except ValueError:
                continue
            with self._lock:
                self._samples.append(sample)


def unload_model(base_url: str, model: str) -> None:
    if not loaded_model_snapshot(base_url, model).get("loaded"):
        return
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(
            {"model": model, "keep_alive": 0, "stream": False}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30.0):
        return


def loaded_model_snapshot(base_url: str, model: str) -> dict[str, object]:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/api/ps",
        timeout=10.0,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    item = select_loaded_model(payload.get("models", ()), model)
    if item is not None:
        size = int(item.get("size") or 0)
        size_vram = int(item.get("size_vram") or 0)
        return {
            "loaded": True,
            "name": str(item.get("name") or ""),
            "size_bytes": size,
            "size_vram_bytes": size_vram,
            "gpu_residency_percent": round(
                size_vram * 100.0 / size if size else 0.0,
                3,
            ),
            "context_length": int(item.get("context_length") or 0),
        }
    return {"loaded": False}


def select_loaded_model(
    models: object,
    requested_model: str,
) -> dict[str, object] | None:
    """Select an exact Ollama tag without confusing model sizes in one family."""

    if not isinstance(models, (list, tuple)):
        return None
    requested = requested_model.strip().lower()
    requested_with_tag = requested if ":" in requested else f"{requested}:latest"
    candidates = [item for item in models if isinstance(item, dict)]
    for item in candidates:
        name = str(item.get("name") or item.get("model") or "").strip().lower()
        normalized = name if ":" in name else f"{name}:latest"
        if normalized == requested_with_tag:
            return item
    return None


def _call_metric(
    *,
    trace_stage: str,
    wall_ms: float,
    first_token_ms: float,
    first_content_ms: float,
    final: dict[str, Any],
    thinking_characters: int,
    content_characters: int,
    error_type: str,
) -> OllamaCallMetric:
    load_ms = _nanoseconds_to_ms(final.get("load_duration"))
    prompt_ms = _nanoseconds_to_ms(final.get("prompt_eval_duration"))
    generation_ms = _nanoseconds_to_ms(final.get("eval_duration"))
    output_tokens = int(final.get("eval_count") or 0)
    return OllamaCallMetric(
        stage=trace_stage,
        wall_ms=wall_ms,
        time_to_first_token_ms=first_token_ms,
        time_to_first_content_ms=first_content_ms,
        load_ms=load_ms,
        prompt_eval_ms=prompt_ms,
        generation_ms=generation_ms,
        input_tokens=int(final.get("prompt_eval_count") or 0),
        output_tokens=output_tokens,
        tokens_per_second=(
            output_tokens / (generation_ms / 1000.0) if generation_ms else 0.0
        ),
        done_reason=str(final.get("done_reason") or ""),
        thinking_characters=thinking_characters,
        content_characters=content_characters,
        failed=bool(error_type),
        error_type=error_type,
    )


def _nanoseconds_to_ms(value: object) -> float:
    return float(value or 0.0) / 1_000_000.0


def _distribution(values) -> dict[str, float]:
    rows = tuple(float(value) for value in values)
    if not rows:
        return {"mean": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "mean": round(statistics.mean(rows), 3),
        "p95": round(_percentile(rows, 0.95), 3),
        "maximum": round(max(rows), 3),
    }


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile) - 1e-9)))
    return ordered[index]


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


_KEY_ALIASES = {
    "id": "i",
    "source": "s",
    "source_ref": "r",
    "subject": "u",
    "predicate": "p",
    "value": "v",
    "summary": "m",
    "confidence": "c",
    "relevance": "l",
    "observed_at": "t",
    "privacy_scope": "y",
    "modality": "o",
    "attributes": "a",
    "authorized": "z",
}
_ALIASES_BY_KEY = {key: alias for key, alias in _KEY_ALIASES.items()}


def _alias_keys(value: object) -> object:
    if isinstance(value, dict):
        return {
            _ALIASES_BY_KEY.get(str(key), str(key)): _alias_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_alias_keys(item) for item in value]
    return value
