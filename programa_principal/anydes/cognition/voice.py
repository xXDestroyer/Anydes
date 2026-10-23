from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Protocol

from anydes.cognition.contracts import SpeechIntention, VoiceOutputResult
from anydes.cognition.live_runtime.interfaces import MetricsCollector
from anydes.cognition.live_runtime.metrics import NullMetricsCollector, PipelineProfiler
from anydes.voz.identity import VoiceProfile
from anydes.voz.interpreter import VoiceInterpreter, VoiceRenderPlan


class SpeechBackend(Protocol):
    def speak(self, text: str) -> None:
        """Render text through a concrete TTS engine."""

    def cancel(self) -> None:
        """Stop current playback when the backend supports interruption."""


@dataclass
class VoiceManager:
    """Coordinates expression from SpeechIntention to a hidden TTS backend.

    This is the only public speaking boundary. Callers provide a complete
    SpeechIntention; they never provide text and never call the backend.
    """

    backend: SpeechBackend
    profile: VoiceProfile
    provider_name: str = ""
    fallback_backends: tuple[tuple[str, SpeechBackend], ...] = ()
    metrics: MetricsCollector = field(default_factory=NullMetricsCollector)
    interpreter: VoiceInterpreter = field(init=False)
    profiler: PipelineProfiler = field(init=False)
    _speak_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.interpreter = VoiceInterpreter(self.profile)
        self.profiler = PipelineProfiler(self.metrics)
        self._speak_lock = threading.Lock()

    @classmethod
    def from_backend(
        cls,
        backend: SpeechBackend,
        *,
        profile: VoiceProfile | None = None,
        provider_name: str = "",
        fallback_backends: tuple[tuple[str, SpeechBackend], ...] = (),
        metrics: MetricsCollector | None = None,
    ) -> "VoiceManager":
        return cls(
            backend=backend,
            profile=profile or VoiceProfile(),
            provider_name=provider_name,
            fallback_backends=fallback_backends,
            metrics=metrics or NullMetricsCollector(),
        )

    def speak(self, speech: SpeechIntention) -> VoiceOutputResult:
        if not isinstance(speech, SpeechIntention):
            raise TypeError("VoiceManager.speak accepts only SpeechIntention.")
        interaction_id = str(speech.metadata.get("interaction_trace_id") or "")
        with self._speak_lock:
            return self.profiler.call_external(
                "pipeline.tts",
                lambda: self._speak(speech),
                interaction_id=interaction_id,
                provider=self._provider(),
            )

    def cancel(self) -> None:
        """Interrupt playback without changing VoiceIdentity or selecting another voice."""
        for _, backend in ((self._provider(), self.backend), *self.fallback_backends):
            cancel = getattr(backend, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    continue

    def _speak(self, speech: SpeechIntention) -> VoiceOutputResult:

        text = speech.text.strip()
        intention_id = str(speech.metadata.get("intention_id") or "speech:intention")
        if not text:
            return VoiceOutputResult(
                intention_id=intention_id,
                status="skipped",
                provider=self._provider(),
                error="empty_intention_speech",
                speech=speech,
            )

        plan = self.render_plan(speech)
        failures: list[str] = []
        providers = ((self._provider(), self.backend), *self.fallback_backends)
        if plan.pause_before > 0:
            time.sleep(min(0.9, plan.pause_before))
        for provider_name, backend in providers:
            try:
                self._render_with_backend(plan, backend)
                backend_trace = getattr(backend, "last_trace", None)
                if bool(getattr(backend_trace, "cancelled", False)):
                    return VoiceOutputResult(
                        intention_id=intention_id,
                        status="skipped",
                        text=text,
                        provider=provider_name,
                        error="playback_cancelled",
                        speech=speech,
                        render_plan=plan.to_dict(),
                    )
                if plan.pause_after > 0:
                    time.sleep(min(0.65, plan.pause_after))
                return VoiceOutputResult(
                    intention_id=intention_id,
                    status="spoken",
                    text=text,
                    provider=provider_name,
                    speech=speech,
                    render_plan=plan.to_dict(),
                )
            except Exception as exc:
                failures.append(
                    f"{provider_name}:{type(exc).__name__}: {exc}"
                )

        return VoiceOutputResult(
            intention_id=intention_id,
            status="failed",
            text=text,
            provider=self._provider(),
            error="; ".join(failures),
            speech=speech,
            render_plan=plan.to_dict(),
        )

    def render_plan(self, speech: SpeechIntention) -> VoiceRenderPlan:
        return self.interpreter.interpret(speech)

    def _render_with_backend(self, plan: VoiceRenderPlan, backend: SpeechBackend) -> None:
        expressive_method = getattr(backend, "speak_plan", None)
        if callable(expressive_method):
            expressive_method(plan)
            return
        backend.speak(plan.backend_text)

    def _provider(self) -> str:
        return self.provider_name or self.backend.__class__.__name__


class VoiceOutputManager(VoiceManager):
    """Deprecated name retained while callers migrate to VoiceManager."""

    def __init__(self, tts: SpeechBackend, provider_name: str = "") -> None:
        super().__init__(backend=tts, profile=VoiceProfile(), provider_name=provider_name)
