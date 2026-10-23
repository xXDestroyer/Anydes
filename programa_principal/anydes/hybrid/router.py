from __future__ import annotations

import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol


PROGRAM_DIR = Path(__file__).resolve().parents[2]
LOCAL_FALLBACK_MODEL = "qwen3:8b"
SUPPORTED_GROQ_MODELS = (
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
)


class BrainState(str, Enum):
    GROQ_HEALTHY = "GROQ_HEALTHY"
    GROQ_DEGRADED = "GROQ_DEGRADED"
    PRELOADING_QWEN8B = "PRELOADING_QWEN8B"
    LOCAL_FALLBACK_8B = "LOCAL_FALLBACK_8B"
    PROBING_GROQ = "PROBING_GROQ"
    GROQ_RECOVERED = "GROQ_RECOVERED"


class ProviderFailure(RuntimeError):
    def __init__(
        self,
        error_type: str,
        *,
        status: int = 0,
        retry_after: float | None = None,
        partial_text: str = "",
        rate_limits: dict[str, str] | None = None,
    ) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.status = int(status)
        self.retry_after = retry_after
        self.partial_text = partial_text
        self.rate_limits = dict(rate_limits or {})

    @property
    def immediate_persistent(self) -> bool:
        return self.status in {401, 403, 404} or (
            self.status == 429
            and self.retry_after is not None
            and self.retry_after >= 5.0
        )


@dataclass(frozen=True)
class HybridRouterConfig:
    primary_model: str = "openai/gpt-oss-120b"
    local_model: str = LOCAL_FALLBACK_MODEL
    failure_threshold: int = 2
    inline_retries: int = 1
    connect_timeout_seconds: float = 3.0
    first_token_timeout_seconds: float = 3.0
    total_request_timeout_seconds: float = 30.0
    latency_degraded_ms: float = 2500.0
    latency_slow_request_threshold: int = 3
    probe_backoff_seconds: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 60.0)
    recovery_successes_required: int = 2
    local_hot_retention_seconds: float = 90.0
    preload_wait_seconds: float = 45.0
    housekeeping_interval_seconds: float = 0.25
    local_keep_alive: str = "10m"
    local_num_ctx: int = 768
    max_output_tokens: int = 220
    log_path: Path | None = None
    auto_housekeeping: bool = True

    def __post_init__(self) -> None:
        if self.primary_model not in SUPPORTED_GROQ_MODELS:
            raise ValueError("unsupported_primary_groq_model")
        if self.local_model != LOCAL_FALLBACK_MODEL:
            raise ValueError("automatic_fallback_must_be_qwen3_8b")
        if self.failure_threshold < 2:
            raise ValueError("failure_threshold_must_avoid_single_failure_preload")
        if not 60.0 <= self.local_hot_retention_seconds <= 120.0:
            raise ValueError("local_hot_retention_must_be_60_to_120_seconds")

    @classmethod
    def from_environment(cls, *, log_path: Path | None = None) -> "HybridRouterConfig":
        model = os.getenv("PRIMARY_GROQ_MODEL", cls.primary_model).strip()
        return cls(
            primary_model=model,
            local_hot_retention_seconds=float(
                os.getenv("ANYDES_HYBRID_LOCAL_RETENTION_SECONDS", "90")
            ),
            log_path=log_path,
        )


@dataclass(frozen=True)
class TransitionEvent:
    timestamp: str
    monotonic: float
    old_state: str
    new_state: str
    reason: str
    error_type: str
    consecutive_failures: int


@dataclass
class RouterMetrics:
    online_requests: int = 0
    online_successes: int = 0
    online_failures: int = 0
    online_retries: int = 0
    rate_limit_events: int = 0
    local_requests: int = 0
    preload_started: int = 0
    preload_completed: int = 0
    preload_cancelled_or_discarded: int = 0
    preload_ms: list[float] = field(default_factory=list)
    probes: int = 0
    probe_successes: int = 0
    false_recoveries: int = 0
    flaps_avoided: int = 0
    local_residency_seconds: float = 0.0
    privacy_redactions: int = 0
    retry_after_seconds: list[float] = field(default_factory=list)
    last_rate_limits: dict[str, str] = field(default_factory=dict)
    online_ttft_ms: list[float] = field(default_factory=list)
    online_wall_ms: list[float] = field(default_factory=list)
    online_tokens_per_second: list[float] = field(default_factory=list)
    online_input_tokens: int = 0
    online_output_tokens: int = 0
    local_ttft_ms: list[float] = field(default_factory=list)
    local_wall_ms: list[float] = field(default_factory=list)


class BrainClient(Protocol):
    model: str

    def stream(self, messages, *, on_content=None, max_output_tokens=None): ...


class LocalClient(BrainClient, Protocol):
    def preload(self) -> float: ...
    def unload(self) -> None: ...
    def is_ready(self) -> bool: ...


@dataclass(frozen=True)
class SanitizationReport:
    redactions: int
    dropped_messages: int
    characters_before: int
    characters_after: int


class OnlineContextSanitizer:
    """Select and redact outbound chat text without touching the local context."""

    _secret_patterns = (
        re.compile(r"(?i)(?:api[_ -]?key|password|passwd|secret|token)\s*[:=]\s*\S+"),
        re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
        re.compile(r"\b(?:gsk|sk)-[A-Za-z0-9_-]{12,}\b"),
    )
    _blocked_sections = re.compile(
        r"(?is)(?:MEMORIA PROTEGIDA|PROTECTED MEMORY|CREDENCIALES|SECRETS?)\s*:\s*.*?(?=\n[A-ZÁÉÍÓÚÜÑ _]{4,}:|\Z)"
    )

    def sanitize(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], SanitizationReport]:
        clean: list[dict[str, str]] = []
        redactions = 0
        dropped = 0
        before = 0
        after = 0
        for message in messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            before += len(content)
            if role not in {"system", "user", "assistant"}:
                dropped += 1
                continue
            content, count = self._blocked_sections.subn("[CONTEXTO PROTEGIDO OMITIDO]", content)
            redactions += count
            for pattern in self._secret_patterns:
                content, count = pattern.subn("[SECRETO OMITIDO]", content)
                redactions += count
            if len(content) > 16000:
                content = content[:16000].rstrip() + "\n[CONTEXTO RECORTADO]"
                redactions += 1
            after += len(content)
            clean.append({"role": role, "content": content})
        return clean, SanitizationReport(redactions, dropped, before, after)


def _stream_outcome(**values):
    if str(PROGRAM_DIR) not in os.sys.path:
        os.sys.path.insert(0, str(PROGRAM_DIR))
    from anydes.realtime.transport import StreamOutcome

    defaults = {
        "text": "",
        "wall_ms": 0.0,
        "time_to_first_token_ms": 0.0,
        "time_to_first_content_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_per_second": 0.0,
        "load_ms": 0.0,
        "prompt_eval_ms": 0.0,
        "generation_ms": 0.0,
        "done_reason": "",
        "timed_out": False,
        "connection_state": "completed",
        "interruption_reason": "",
        "error_type": "",
    }
    defaults.update(values)
    return StreamOutcome(**defaults)


class GroqChatClient:
    host = "api.groq.com"

    def __init__(self, config: HybridRouterConfig) -> None:
        self.config = config
        self.model = config.primary_model
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY_missing")

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        limit = max(8, int(max_output_tokens or self.config.max_output_tokens))
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": limit,
            "stream": True,
            "reasoning_format": "hidden",
        }
        payload["reasoning_effort"] = "none" if self.model.startswith("qwen/") else "low"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPSConnection(
            self.host,
            timeout=self.config.connect_timeout_seconds,
        )
        started = time.perf_counter()
        first_token = None
        first_content = None
        parts: list[str] = []
        usage: dict[str, object] = {}
        rate_limits: dict[str, str] = {}
        try:
            connection.request(
                "POST",
                "/openai/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Anydes-Hybrid-Router-Lab/1.0",
                },
            )
            response = connection.getresponse()
            rate_limits = {
                key.lower(): value
                for key, value in response.getheaders()
                if key.lower().startswith("x-ratelimit-")
                or key.lower() in {"retry-after"}
            }
            if response.status >= 400:
                retry_after = _parse_retry_after(response.getheader("Retry-After"))
                response.read(2048)
                raise ProviderFailure(
                    _http_error_type(response.status),
                    status=response.status,
                    retry_after=retry_after,
                    rate_limits=rate_limits,
                )
            if connection.sock is not None:
                connection.sock.settimeout(self.config.first_token_timeout_seconds)
            while True:
                if time.perf_counter() - started > self.config.total_request_timeout_seconds:
                    raise ProviderFailure("total_request_timeout", partial_text="".join(parts), rate_limits=rate_limits)
                raw = response.fp.readline() if response.fp is not None else b""
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                now = time.perf_counter()
                if first_token is None:
                    first_token = now
                    if connection.sock is not None:
                        remaining = max(1.0, self.config.total_request_timeout_seconds - (now - started))
                        connection.sock.settimeout(remaining)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = str((choices[0].get("delta") or {}).get("content") or "")
                if content:
                    if first_content is None:
                        first_content = now
                    parts.append(content)
                    if on_content is not None:
                        on_content("".join(parts))
        except ProviderFailure:
            raise
        except (TimeoutError, socket.timeout) as exc:
            kind = "first_token_timeout" if first_token is None else "stream_timeout"
            raise ProviderFailure(kind, partial_text="".join(parts), rate_limits=rate_limits) from exc
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise ProviderFailure(type(exc).__name__, partial_text="".join(parts), rate_limits=rate_limits) from exc
        finally:
            connection.close()
        completed = time.perf_counter()
        text = "".join(parts).strip()
        output_tokens = int(usage.get("completion_tokens") or max(1, (len(text) + 3) // 4)) if text else 0
        input_tokens = int(usage.get("prompt_tokens") or 0)
        generation_s = max(0.001, completed - (first_content or first_token or started))
        wall_ms = (completed - started) * 1000.0
        return _stream_outcome(
            text=text,
            wall_ms=wall_ms,
            time_to_first_token_ms=((first_token or completed) - started) * 1000.0,
            time_to_first_content_ms=((first_content or completed) - started) * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / generation_s if output_tokens else 0.0,
            prompt_eval_ms=0.0,
            generation_ms=generation_s * 1000.0,
            done_reason="stop",
        )

    def probe(self):
        return self.stream(
            [
                {"role": "system", "content": "Responde únicamente OK."},
                {"role": "user", "content": "Estado."},
            ],
            max_output_tokens=4,
        )


class FaultInjectingOnlineClient:
    """Deterministic diagnostic failures; disabled without an explicit env plan."""

    def __init__(self, delegate: GroqChatClient, plan: list[str]) -> None:
        self.delegate = delegate
        self.plan = list(plan)
        self.model = delegate.model

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        if self.plan:
            kind = self.plan.pop(0)
            if kind == "rate_limit":
                raise ProviderFailure(kind, status=429, retry_after=60.0)
            if kind == "invalid_credential":
                raise ProviderFailure(kind, status=401)
            if kind == "model_unavailable":
                raise ProviderFailure(kind, status=404)
            if kind == "provider_5xx":
                raise ProviderFailure(kind, status=503)
            raise ProviderFailure(kind)
        return self.delegate.stream(
            messages,
            on_content=on_content,
            max_output_tokens=max_output_tokens,
        )

    def probe(self):
        return self.delegate.probe()


class Ollama8BClient:
    def __init__(
        self,
        config: HybridRouterConfig,
        base_url: str = "http://127.0.0.1:11434",
        *,
        transport_class=None,
    ) -> None:
        self.config = config
        self.model = LOCAL_FALLBACK_MODEL
        self.base_url = base_url.rstrip("/")
        if transport_class is None:
            if str(PROGRAM_DIR) not in os.sys.path:
                os.sys.path.insert(0, str(PROGRAM_DIR))
            from anydes.realtime.transport import OllamaRealtimeTransport

            transport_class = OllamaRealtimeTransport
        self.transport = transport_class(
            base_url=self.base_url,
            model=self.model,
            keep_alive=config.local_keep_alive,
            timeout_seconds=max(45.0, config.total_request_timeout_seconds),
            max_output_tokens=config.max_output_tokens,
            num_ctx=config.local_num_ctx,
        )

    def preload(self) -> float:
        return self.transport.warmup(timeout_seconds=120.0)

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        local_messages = list(messages)
        fallback_note = {
            "role": "system",
            "content": (
                "MODO INTERNO: LOCAL_FALLBACK_8B. Mantén la identidad, SELF, memoria y continuidad "
                "recibidas. Sé conservadora ante ambigüedad, acciones de riesgo o tareas complejas; "
                "pide aclaración antes de inventar o actuar. Reflection y Action Manager validan fuera de ti."
            ),
        }
        local_messages.insert(1 if local_messages and local_messages[0].get("role") == "system" else 0, fallback_note)
        return self.transport.stream(
            local_messages,
            on_content=on_content,
            max_output_tokens=max_output_tokens,
        )

    def unload(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20.0) as response:
                response.read()
        except (OSError, urllib.error.URLError):
            pass

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=4.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return any(str(item.get("name") or "").startswith(self.model) for item in payload.get("models") or [])
        except (OSError, ValueError, urllib.error.URLError):
            return False


class HybridBrainRouter:
    def __init__(
        self,
        config: HybridRouterConfig,
        *,
        online: BrainClient | None = None,
        local: LocalClient | None = None,
        probe_guard: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.online = online or GroqChatClient(config)
        self.local = local or Ollama8BClient(config)
        if self.local.model != LOCAL_FALLBACK_MODEL:
            raise ValueError("forbidden_local_fallback_model")
        self.sanitizer = OnlineContextSanitizer()
        self.probe_guard = probe_guard or (lambda: True)
        self.clock = clock
        self.sleeper = sleeper
        self.state = BrainState.GROQ_HEALTHY
        self.metrics = RouterMetrics()
        self.transitions: list[TransitionEvent] = []
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="anydes-hybrid-router")
        self._preload_future: Future | None = None
        self._local_ready = threading.Event()
        self._discard_preload_on_ready = False
        self._consecutive_failures = 0
        self._slow_requests = 0
        self._probe_successes = 0
        self._probe_index = 0
        self._next_probe_at = 0.0
        self._turn_busy = False
        self._probe_busy = False
        self._closed = False
        self._local_ready_since: float | None = None
        self._retention_deadline: float | None = None
        self._housekeeping_stop = threading.Event()
        self._housekeeping_thread: threading.Thread | None = None
        if config.auto_housekeeping:
            self._housekeeping_thread = threading.Thread(
                target=self._housekeeping,
                name="anydes-hybrid-housekeeping",
                daemon=True,
            )
            self._housekeeping_thread.start()

    def warmup(self, *, timeout_seconds: float = 60.0) -> float:
        _ = timeout_seconds
        started = self.clock()
        try:
            outcome = self.online.stream(
                [{"role": "system", "content": "Responde brevemente."}, {"role": "user", "content": "Di lista."}],
                max_output_tokens=4,
            )
            self.metrics.online_requests += 1
            self.metrics.online_successes += 1
            self._append_outcome_metrics(outcome, online=True)
        except ProviderFailure as failure:
            self.metrics.online_requests += 1
            self._record_failure(failure, reason="warmup_failure")
        return (self.clock() - started) * 1000.0

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("hybrid_router_closed")
            self._turn_busy = True
            route = self.state
        try:
            if route in {BrainState.LOCAL_FALLBACK_8B, BrainState.PROBING_GROQ}:
                return self._stream_local(messages, on_content=on_content, max_output_tokens=max_output_tokens)
            if route == BrainState.GROQ_RECOVERED:
                return self._confirm_recovery(messages, on_content=on_content, max_output_tokens=max_output_tokens)
            return self._stream_online_or_degrade(messages, on_content=on_content, max_output_tokens=max_output_tokens)
        finally:
            with self._lock:
                self._turn_busy = False

    def _stream_online_or_degrade(self, messages, *, on_content, max_output_tokens):
        try:
            outcome = self._online_with_retry(messages, on_content=on_content, max_output_tokens=max_output_tokens)
        except ProviderFailure as failure:
            self._record_failure(failure, reason="online_turn_failed")
            with self._lock:
                persistent = failure.immediate_persistent or self._consecutive_failures >= self.config.failure_threshold
            if persistent:
                self._enter_degraded(failure)
                if self._local_ready.wait(timeout=self.config.preload_wait_seconds):
                    self._transition(BrainState.LOCAL_FALLBACK_8B, "fallback_ready", failure.error_type)
                    return self._stream_local(messages, on_content=on_content, max_output_tokens=max_output_tokens)
            return self._failure_outcome(failure)
        self._record_online_success(outcome)
        with self._lock:
            recovered_during_preload = self.state in {
                BrainState.GROQ_DEGRADED,
                BrainState.PRELOADING_QWEN8B,
            }
            if recovered_during_preload:
                self._discard_preload_on_ready = True
        if recovered_during_preload:
            self._transition(BrainState.GROQ_HEALTHY, "online_recovered_during_preload", "")
            if self._local_ready.is_set():
                self._executor.submit(self._unload_local, "online_recovered_during_preload")
        return outcome

    def _confirm_recovery(self, messages, *, on_content, max_output_tokens):
        try:
            outcome = self._online_with_retry(messages, on_content=on_content, max_output_tokens=max_output_tokens)
        except ProviderFailure as failure:
            with self._lock:
                self.metrics.false_recoveries += 1
                self._probe_successes = 0
            self._transition(BrainState.LOCAL_FALLBACK_8B, "recovery_turn_failed", failure.error_type)
            self._schedule_next_probe(reset=True)
            return self._stream_local(messages, on_content=on_content, max_output_tokens=max_output_tokens)
        self._record_online_success(outcome)
        with self._lock:
            self._retention_deadline = self.clock() + self.config.local_hot_retention_seconds
        self._transition(BrainState.GROQ_HEALTHY, "real_turn_confirmed_recovery", "")
        return outcome

    def _online_with_retry(self, messages, *, on_content, max_output_tokens):
        outbound, report = self.sanitizer.sanitize(messages)
        with self._lock:
            self.metrics.privacy_redactions += report.redactions
        self._log_event(
            "outbound_context_selected",
            roles=[str(message.get("role") or "") for message in outbound],
            message_count=len(outbound),
            characters_before=report.characters_before,
            characters_after=report.characters_after,
            redactions=report.redactions,
            dropped_messages=report.dropped_messages,
        )
        attempt = 0
        while True:
            with self._lock:
                self.metrics.online_requests += 1
            try:
                return self.online.stream(outbound, on_content=on_content, max_output_tokens=max_output_tokens)
            except ProviderFailure as failure:
                with self._lock:
                    self.metrics.last_rate_limits = dict(failure.rate_limits)
                    if failure.status == 429:
                        self.metrics.rate_limit_events += 1
                if failure.partial_text or attempt >= self.config.inline_retries or failure.immediate_persistent:
                    raise
                attempt += 1
                with self._lock:
                    self.metrics.online_retries += 1
                wait = failure.retry_after if failure.retry_after is not None else min(2.0, float(attempt))
                if failure.retry_after is not None:
                    with self._lock:
                        self.metrics.retry_after_seconds.append(wait)
                self.sleeper(max(0.0, wait))

    def _stream_local(self, messages, *, on_content, max_output_tokens):
        if not self._local_ready.is_set() and not self.local.is_ready():
            self._start_preload("local_requested_not_ready")
            if not self._local_ready.wait(timeout=self.config.preload_wait_seconds):
                return self._failure_outcome(ProviderFailure("local_preload_timeout"))
        with self._lock:
            self.metrics.local_requests += 1
        outcome = self.local.stream(messages, on_content=on_content, max_output_tokens=max_output_tokens)
        self._append_outcome_metrics(outcome, online=False)
        return outcome

    def _record_online_success(self, outcome) -> None:
        with self._lock:
            self.metrics.online_successes += 1
            self._consecutive_failures = 0
            if float(outcome.time_to_first_content_ms) > self.config.latency_degraded_ms:
                self._slow_requests += 1
            else:
                self._slow_requests = 0
            slow_persistent = self._slow_requests >= self.config.latency_slow_request_threshold
        self._append_outcome_metrics(outcome, online=True)
        if slow_persistent:
            self._enter_degraded(ProviderFailure("persistent_high_latency"))

    def _append_outcome_metrics(self, outcome, *, online: bool) -> None:
        with self._lock:
            if online:
                self.metrics.online_ttft_ms.append(float(outcome.time_to_first_content_ms))
                self.metrics.online_wall_ms.append(float(outcome.wall_ms))
                self.metrics.online_tokens_per_second.append(float(outcome.tokens_per_second))
                self.metrics.online_input_tokens += int(outcome.input_tokens)
                self.metrics.online_output_tokens += int(outcome.output_tokens)
            else:
                self.metrics.local_ttft_ms.append(float(outcome.time_to_first_content_ms))
                self.metrics.local_wall_ms.append(float(outcome.wall_ms))

    def _record_failure(self, failure: ProviderFailure, *, reason: str) -> None:
        with self._lock:
            self.metrics.online_failures += 1
            self._consecutive_failures += 1
            failures = self._consecutive_failures
        self._log_event(
            "provider_failure",
            reason=reason,
            error_type=failure.error_type,
            status=failure.status,
            retry_after=failure.retry_after,
            consecutive_failures=failures,
            rate_limits=failure.rate_limits,
        )

    def _enter_degraded(self, failure: ProviderFailure) -> None:
        with self._lock:
            if self.state in {BrainState.LOCAL_FALLBACK_8B, BrainState.PRELOADING_QWEN8B}:
                return
        self._transition(BrainState.GROQ_DEGRADED, "persistent_online_failure", failure.error_type)
        self._start_preload(failure.error_type)

    def _start_preload(self, reason: str) -> None:
        with self._lock:
            if self._local_ready.is_set() or self.local.is_ready():
                self._local_ready.set()
                if self._local_ready_since is None:
                    self._local_ready_since = self.clock()
                self._transition(BrainState.LOCAL_FALLBACK_8B, "local_already_resident", "")
                return
            if self._preload_future is not None and not self._preload_future.done():
                return
            self.metrics.preload_started += 1
            self._transition(BrainState.PRELOADING_QWEN8B, "preload_started", reason)
            started = self.clock()
            self._preload_future = self._executor.submit(self.local.preload)

        def completed(future: Future) -> None:
            try:
                reported_ms = float(future.result())
                elapsed_ms = max(reported_ms, (self.clock() - started) * 1000.0)
            except Exception as exc:
                self._log_event("preload_failed", error_type=type(exc).__name__)
                return
            with self._lock:
                self.metrics.preload_completed += 1
                self.metrics.preload_ms.append(elapsed_ms)
                self._local_ready.set()
                self._local_ready_since = self.clock()
                discard = self._discard_preload_on_ready or self.state == BrainState.GROQ_HEALTHY
                self._discard_preload_on_ready = False
            if discard:
                with self._lock:
                    self.metrics.preload_cancelled_or_discarded += 1
                # This callback already runs off the UI thread. Unload directly so
                # shutdown cannot race a new submission and leave the model hot.
                self._unload_local("online_recovered_during_preload")
            else:
                self._transition(BrainState.LOCAL_FALLBACK_8B, "preload_completed", "")

        self._preload_future.add_done_callback(completed)

    def tick(self) -> None:
        now = self.clock()
        with self._lock:
            if self._closed or self._turn_busy or self._probe_busy:
                return
            if self._retention_deadline is not None and now >= self._retention_deadline:
                self._retention_deadline = None
                self._executor.submit(self._unload_local, "stable_retention_elapsed")
            if self.state != BrainState.LOCAL_FALLBACK_8B or now < self._next_probe_at:
                return
            if not self.probe_guard():
                return
            self._probe_busy = True
            self._transition(BrainState.PROBING_GROQ, "scheduled_probe", "")
            self._executor.submit(self._run_probe)

    def _run_probe(self) -> None:
        success = False
        failure_type = ""
        try:
            with self._lock:
                self.metrics.probes += 1
            self.online.probe()
            success = True
        except ProviderFailure as failure:
            failure_type = failure.error_type
        except Exception as exc:
            failure_type = type(exc).__name__
        with self._lock:
            self._probe_busy = False
            if success:
                self.metrics.probe_successes += 1
                self._probe_successes += 1
                recovered = self._probe_successes >= self.config.recovery_successes_required
            else:
                if self._probe_successes:
                    self.metrics.flaps_avoided += 1
                self._probe_successes = 0
                recovered = False
        if recovered:
            self._transition(BrainState.GROQ_RECOVERED, "two_consecutive_probe_successes", "")
        else:
            self._transition(
                BrainState.LOCAL_FALLBACK_8B,
                "probe_succeeded_once" if success else "probe_failed",
                failure_type,
            )
            self._schedule_next_probe(reset=not success)

    def _schedule_next_probe(self, *, reset: bool) -> None:
        with self._lock:
            if reset:
                self._probe_index = min(self._probe_index + 1, len(self.config.probe_backoff_seconds) - 1)
            delay = self.config.probe_backoff_seconds[self._probe_index]
            if not reset and self._probe_index < len(self.config.probe_backoff_seconds) - 1:
                self._probe_index += 1
            self._next_probe_at = self.clock() + delay

    def _transition(self, new_state: BrainState, reason: str, error_type: str) -> None:
        with self._lock:
            old = self.state
            if old == new_state:
                return
            self.state = new_state
            if new_state == BrainState.LOCAL_FALLBACK_8B and self._next_probe_at <= self.clock():
                self._probe_index = 0
                self._next_probe_at = self.clock() + self.config.probe_backoff_seconds[0]
            event = TransitionEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                monotonic=self.clock(),
                old_state=old.value,
                new_state=new_state.value,
                reason=reason,
                error_type=error_type,
                consecutive_failures=self._consecutive_failures,
            )
            self.transitions.append(event)
        self._write_log({"event": "state_transition", **asdict(event)})

    def _log_event(self, event: str, **details) -> None:
        self._write_log(
            {
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monotonic": self.clock(),
                **details,
            }
        )

    def _write_log(self, payload: dict[str, object]) -> None:
        path = self.config.log_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _failure_outcome(self, failure: ProviderFailure):
        return _stream_outcome(
            text=failure.partial_text,
            done_reason="error",
            timed_out="timeout" in failure.error_type,
            connection_state="interrupted",
            interruption_reason=failure.error_type,
            error_type=failure.error_type,
        )

    def _unload_local(self, reason: str) -> None:
        with self._lock:
            if not self._local_ready.is_set() and not self.local.is_ready():
                return
            if self._local_ready_since is not None:
                self.metrics.local_residency_seconds += max(0.0, self.clock() - self._local_ready_since)
        self.local.unload()
        with self._lock:
            self._local_ready.clear()
            self._local_ready_since = None
        self._log_event("local_unloaded", reason=reason, model=LOCAL_FALLBACK_MODEL)

    def _housekeeping(self) -> None:
        while not self._housekeeping_stop.wait(self.config.housekeeping_interval_seconds):
            self.tick()

    def wait_for_preload(self, timeout: float = 5.0) -> bool:
        return self._local_ready.wait(timeout=timeout)

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self.state.value,
                "primary_model": self.config.primary_model,
                "local_fallback_model": LOCAL_FALLBACK_MODEL,
                "local_ready": self._local_ready.is_set(),
                "consecutive_failures": self._consecutive_failures,
                "probe_successes": self._probe_successes,
                "next_probe_in_seconds": max(0.0, self._next_probe_at - self.clock()),
                "metrics": asdict(self.metrics),
                "transitions": [asdict(item) for item in self.transitions],
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._housekeeping_stop.set()
        if self._housekeeping_thread is not None:
            self._housekeeping_thread.join(timeout=2.0)
        future = self._preload_future
        if future is not None and not future.done():
            self._discard_preload_on_ready = True
            future.cancel()
        self._unload_local("router_close")
        self._executor.shutdown(wait=True, cancel_futures=True)


_ROUTERS: list[HybridBrainRouter] = []
_ROUTERS_LOCK = threading.Lock()


class HybridRealtimeTransport:
    """Official hybrid transport. The legacy local model argument is intentionally ignored."""

    native_local_transport_class = None

    def __init__(self, *, base_url, model, keep_alive, timeout_seconds, max_output_tokens, num_ctx=768):
        _ = (model, keep_alive, timeout_seconds)
        log_root = Path(
            os.getenv(
                "ANYDES_HYBRID_LOG_DIR",
                str(PROGRAM_DIR / "datos" / "telemetria" / "hybrid_router"),
            )
        )
        config = HybridRouterConfig.from_environment(
            log_path=log_root / f"transitions-{int(time.time() * 1000)}.jsonl"
        )
        config = HybridRouterConfig(
            **{
                **asdict(config),
                "local_num_ctx": int(num_ctx),
                "max_output_tokens": int(max_output_tokens),
            }
        )
        online = GroqChatClient(config)
        fault_plan = [
            item.strip()
            for item in os.getenv("ANYDES_HYBRID_FAULT_PLAN", "").split(",")
            if item.strip()
        ]
        if fault_plan:
            online = FaultInjectingOnlineClient(online, fault_plan)
        self.router = HybridBrainRouter(
            config,
            online=online,
            local=Ollama8BClient(
                config,
                base_url=base_url,
                transport_class=self.native_local_transport_class,
            ),
            probe_guard=_default_runtime_probe_guard,
        )
        self.model = f"groq:{config.primary_model}|fallback:{LOCAL_FALLBACK_MODEL}"
        with _ROUTERS_LOCK:
            _ROUTERS.append(self.router)

    def warmup(self, *, timeout_seconds=60.0):
        return self.router.warmup(timeout_seconds=timeout_seconds)

    def stream(self, messages, *, on_content=None, max_output_tokens=None):
        return self.router.stream(
            messages,
            on_content=on_content,
            max_output_tokens=max_output_tokens,
        )


def router_snapshots() -> list[dict[str, object]]:
    with _ROUTERS_LOCK:
        return [router.status() for router in _ROUTERS]


def close_all_routers() -> None:
    with _ROUTERS_LOCK:
        routers = list(_ROUTERS)
        _ROUTERS.clear()
    for router in routers:
        router.close()


def _default_runtime_probe_guard() -> bool:
    """Avoid probes while known B4 generation/playback workers are active."""
    telemetry = os.sys.modules.get("telemetry")
    monitor = getattr(telemetry, "_monitor", None) if telemetry is not None else None
    activity = dict(getattr(monitor, "activity", {}) or {})
    if activity.get("tts") or activity.get("playback") or activity.get("stt"):
        return False
    blocked = {
        "anydes-native-voice-stream",
        "anydes-voice-producer",
        "anydes-voice-warmup",
        "recorded-parakeet-smoke",
    }
    return not any(thread.name in blocked for thread in threading.enumerate())


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _http_error_type(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status in {401, 403}:
        return "invalid_credential"
    if status == 404:
        return "model_unavailable"
    if status >= 500:
        return "provider_5xx"
    return f"http_{status}"
