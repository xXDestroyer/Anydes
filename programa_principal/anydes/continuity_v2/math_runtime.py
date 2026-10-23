from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time

from .math_presentation import present_math


_LOCK = threading.Lock()


def _record(event: str, **details) -> None:
    root = Path(
        os.environ.get(
            "ANYDES_COGNITIVE_V2_LOG_DIR",
            str(Path(__file__).resolve().parents[2] / "datos" / "telemetria" / "continuity_v2"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "time": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    with _LOCK:
        with (root / "math_tts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def install() -> None:
    _install_voice_boundary()
    _install_stream_watchdog()
    _install_ui_hook()


def _install_voice_boundary() -> None:
    import anydes.voz.nueva.manager as manager_module

    cls = manager_module.NewVoiceManager
    if getattr(cls, "_math_tts_v1_installed", False):
        return
    original = cls.speak

    def speak_math_safe(self, speech, *, generation=None):
        started = time.perf_counter()
        source = str(getattr(speech, "text", "") or "")
        presentation = present_math(source)
        metadata = dict(getattr(speech, "metadata", {}) or {})
        metadata.update({
            "math_presentation": "v1",
            "math_regions": presentation.math_regions,
            "math_fallback": presentation.fallback_used,
        })
        safe_speech = replace(speech, text=presentation.speech_text, metadata=metadata)
        request_generation = self.generation if generation is None else generation
        status = "exception"
        error = ""
        try:
            result = original(self, safe_speech, generation=generation)
            status = str(getattr(result, "status", "unknown"))
            error = str(getattr(result, "error", "") or "")
            if status == "failed":
                _recover_failed_request(self, error)
            return result
        except Exception as exc:
            error = type(exc).__name__
            _recover_failed_request(self, error)
            raise
        finally:
            with getattr(self, "_state_lock"):
                if getattr(self, "_active_generation", None) == request_generation:
                    self._active_generation = 0
            _record(
                "tts_request_finalized",
                generation_id=request_generation,
                status=status,
                error_type=error.split(":", 1)[0],
                source_characters=len(source),
                speech_characters=len(presentation.speech_text),
                math_regions=presentation.math_regions,
                fallback_used=presentation.fallback_used,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )

    cls.speak = speak_math_safe
    cls._math_tts_v1_installed = True


def _recover_failed_request(manager, error: str) -> None:
    try:
        manager.audio.cancel()
    except Exception:
        pass
    engine = getattr(manager, "engine", None)
    marker = str(error or "").casefold()
    if engine is not None and any(part in marker for part in ("worker", "stream", "timeout", "synthesis")):
        terminate = getattr(engine, "_terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass
    _record("tts_request_recovered", error_type=type(error).__name__ if not isinstance(error, str) else error[:96])


def _install_stream_watchdog() -> None:
    import anydes.voz.nueva.engines as engines_module
    from anydes.voz.nueva.contracts import EngineError, SynthesisResult

    cls = engines_module.FasterQwen3TTSProvider
    if getattr(cls, "_math_tts_watchdog_v1_installed", False):
        return
    original = cls.synthesize_stream

    def synthesize_stream_with_progress_timeout(self, request):
        first_timeout = max(10.0, float(os.environ.get("ANYDES_TTS_FIRST_CHUNK_TIMEOUT_SECONDS", "45")))
        progress_timeout = max(5.0, float(os.environ.get("ANYDES_TTS_PROGRESS_TIMEOUT_SECONDS", "20")))
        events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=8)
        stop = threading.Event()
        source = original(self, request)

        def publish(kind: str, payload: object) -> bool:
            while not stop.is_set():
                try:
                    events.put((kind, payload), timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def pump() -> None:
            try:
                for item in source:
                    if not publish("item", item):
                        break
            except BaseException as exc:
                publish("error", exc)
            finally:
                publish("done", None)

        thread = threading.Thread(target=pump, name="anydes-b4-request-pump", daemon=True)
        thread.start()
        received = False
        timed_out = False
        source_done = False
        try:
            while True:
                try:
                    kind, payload = events.get(timeout=progress_timeout if received else first_timeout)
                except queue.Empty:
                    timed_out = True
                    _record(
                        "tts_stream_progress_timeout",
                        first_audio_received=received,
                        timeout_seconds=progress_timeout if received else first_timeout,
                    )
                    try:
                        self.cancel()
                    finally:
                        terminate = getattr(self, "_terminate", None)
                        if callable(terminate):
                            terminate()
                    yield SynthesisResult(
                        status="failed",
                        engine_id=self.engine_id,
                        voice_id=self.voice_id,
                        identity_id=request.identity_id,
                        error=EngineError(
                            "worker_stream_progress_timeout",
                            "synthesis",
                            self.engine_id,
                            recoverable=True,
                        ),
                    )
                    return
                if kind == "item":
                    received = True
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    source_done = True
                    return
        finally:
            stop.set()
            if not source_done:
                try:
                    self.cancel()
                except Exception:
                    pass
            if timed_out:
                terminate = getattr(self, "_terminate", None)
                if callable(terminate):
                    terminate()
            thread.join(timeout=6.0)
            if thread.is_alive():
                terminate = getattr(self, "_terminate", None)
                if callable(terminate):
                    terminate()
                thread.join(timeout=2.0)

    cls.synthesize_stream = synthesize_stream_with_progress_timeout
    cls._math_tts_watchdog_v1_installed = True


def _install_ui_hook() -> None:
    try:
        from anydes.voice_input import integration
    except ImportError:
        _record("math_ui_hook_deferred", reason="voice_input_integration_not_loaded")
        return
    if getattr(integration, "_math_presentation_v1_installed", False):
        return
    original_install = integration.install

    def install_with_math(module, *args, **kwargs):
        performance = bool(kwargs.get("performance", args[0] if args else False))
        _patch_ui_module(module, performance=performance)
        return original_install(module, *args, **kwargs)

    integration.install = install_with_math
    integration._math_presentation_v1_installed = True


def _patch_ui_module(module, *, performance: bool) -> None:
    cls = module.PerformanceCompanionWindow if performance else module.AnydesPySideWindow
    if getattr(cls, "_math_presentation_v1_installed", False):
        return
    original_partial = cls._handle_partial
    original_answer = cls._handle_answer

    def handle_partial(self, text: str):
        return original_partial(self, present_math(text).chat_text)

    def handle_answer(self, text: str):
        chat_presentation = present_math(text)
        intention = getattr(self.assistant, "last_intention", None)
        original_intention = intention
        speech_presentation = None
        if intention is not None and getattr(intention, "speech_intention", None) is not None:
            speech = intention.speech_intention
            speech_presentation = present_math(speech.text)
            metadata = dict(speech.metadata)
            metadata.update({
                "math_presentation": "v1",
                "math_regions": speech_presentation.math_regions,
                "math_fallback": speech_presentation.fallback_used,
            })
            patched_speech = replace(speech, text=speech_presentation.speech_text, metadata=metadata)
            self.assistant.last_intention = replace(intention, speech_intention=patched_speech)
        try:
            result = original_answer(self, chat_presentation.chat_text)
        finally:
            if original_intention is not None:
                self.assistant.last_intention = original_intention
        audit = getattr(self.assistant, "audit", None)
        if audit is not None:
            audit.record(
                "math_presentation_completed",
                chat_characters=len(chat_presentation.chat_text),
                speech_characters=len(speech_presentation.speech_text) if speech_presentation else 0,
                math_regions=max(chat_presentation.math_regions, speech_presentation.math_regions if speech_presentation else 0),
                fallback_used=chat_presentation.fallback_used or bool(speech_presentation and speech_presentation.fallback_used),
            )
        return result

    cls._handle_partial = handle_partial
    cls._handle_answer = handle_answer
    cls._math_presentation_v1_installed = True
