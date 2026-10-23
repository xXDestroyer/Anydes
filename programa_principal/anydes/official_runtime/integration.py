"""Official integrated runtime for Parakeet, the hybrid brain, and the UI."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib import request
import uuid
import wave
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
PROGRAM = HERE.parents[1]
ROOT = PROGRAM.parent
OUT = PROGRAM / "datos" / "telemetria" / "official_runtime"
OFFICIAL_LAUNCHER = ROOT / "Abrir Anydes.bat"

RESOURCE_STRESS_PROMPTS = (
    "Hola, Anydes. Responde con una frase breve y tranquila.",
    "Resume en tres frases cómo organizar una pausa saludable durante el trabajo.",
    "Explícame con un ejemplo cotidiano por qué conviene dividir una tarea grande.",
    "Acompáñame un momento sin dar consejos largos.",
    "Dame una respuesta algo más extensa sobre cómo recuperar la concentración.",
    "¿Qué recuerdas del tono que prefiero para nuestras conversaciones?",
    "Haz una lista oral breve de tres cosas para cerrar bien el día.",
    "Cuéntame una curiosidad sencilla sobre la memoria humana.",
    "Responde con calma: ¿cómo distinguirías una duda de una certeza?",
    "Imagina que terminé una tarea difícil y dime algo amable.",
    "Explica en cuatro frases qué significa trabajar de forma incremental.",
    "¿Cómo mantendrías una conversación natural sin interrumpir demasiado?",
)

for path in (str(ROOT), str(PROGRAM)):
    if path not in sys.path:
        sys.path.insert(0, path)

from anydes.voice_input.environment import setup
setup()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cleanup_state() -> dict:
    try:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        with request.urlopen(host + "/api/ps", timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ollama = {"host": host, "models": payload.get("models", [])}
    except Exception as exc:
        ollama = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import psutil
        current = psutil.Process()
        children = [{"pid": child.pid, "name": child.name(), "status": child.status()}
            for child in current.children(recursive=True) if child.is_running()]
    except Exception as exc:
        children = [{"error": f"{type(exc).__name__}: {exc}"}]
    return {"ollama": ollama, "runtime_children": children}


def wait_for_cleanup(timeout_seconds: float = 45.0) -> dict:
    started = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    state = cleanup_state()
    while time.monotonic() < deadline:
        models = state.get("ollama", {}).get("models")
        children = state.get("runtime_children") or []
        blocking_children = [child for child in children
            if str(child.get("name", "")).casefold() != "qtwebengineprocess.exe"]
        if models == [] and not blocking_children:
            break
        time.sleep(0.5)
        state = cleanup_state()
    state["waited_seconds"] = time.monotonic() - started
    state["process_bound_children"] = [child for child in state.get("runtime_children", [])
        if str(child.get("name", "")).casefold() == "qtwebengineprocess.exe"]
    state["blocking_children"] = [child for child in state.get("runtime_children", [])
        if str(child.get("name", "")).casefold() != "qtwebengineprocess.exe"]
    state["timed_out"] = bool(state.get("ollama", {}).get("models")
        or state["blocking_children"])
    return state


def run_mode(
    mode: str,
    target_minutes: int,
    *,
    smoke_audio: Path | None = None,
    smoke_text: str = "",
    resource_stress: bool = False,
    auto_close_ms: int = 0,
    smoke_delay_ms: int = 1200,
    vision_quick_test: bool = False,
    official_runtime: bool = False,
) -> int:
    from anydes.voice_input.telemetry import mark
    mark("runtime_imports_started", mode=mode)
    from runtime_session import run_desktop

    def run() -> None:
        expected_brain = os.environ.get(
            "ANYDES_EXPECTED_BRAIN",
            "groq:openai/gpt-oss-120b|fallback:qwen3:8b",
        )
        mark("exclusive_brain_prepared", mode=mode, expected_brain=expected_brain)
        from lanzar_anydes import load_profile, OFFICIAL_VISUAL_PROFILE
        load_profile()
        if mode == "normal":
            load_profile(OFFICIAL_VISUAL_PROFILE)
            if vision_quick_test:
                # The controlled fixture is already the fresh, authorized frame.
                # A second desktop capture would replace it and make the smoke
                # dependent on the automation host rather than the vision path.
                os.environ["ANYDES_SCREEN_FRESH_ON_ANALYZE"] = "false"
            import anydes.interfaz.pyside_companion.app as module
        else:
            load_profile(PROGRAM / "configuracion" / "anydes_performance.env")
            import anydes.interfaz.performance_companion.app as module
        from anydes.voice_input.integration import install
        from anydes.stt.parakeet.live_lab import ParakeetLivePanel

        class IntegratedParakeetPanel(ParakeetLivePanel):
            def __init__(self, parent=None, on_input=None, on_barge=None):
                super().__init__(parent, on_input, on_barge)
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QLabel, QPushButton
                label = os.environ.get(
                    "ANYDES_EXPECTED_BRAIN",
                    "Groq | fallback Qwen3:8B",
                )
                if official_runtime:
                    self.setWindowTitle("Anydes | Entrada de voz")
                    for notice in self.findChildren(QLabel):
                        if notice.text().startswith("Prueba libre."):
                            notice.hide()
                else:
                    self.setWindowTitle(f"Anydes | STT integrado RC | {label}")
                    self.decoder_info.setText(self.decoder_info.text() +
                        f" | SESION: {target_minutes} min")
                    self._session_started = time.monotonic()
                    self._target_announced = False
                    self.session_clock = QLabel()
                    self.session_clock.setStyleSheet("color:#bcd7ca;font-weight:600;")
                    self.layout().insertWidget(2, self.session_clock)
                    self.finish_session = QPushButton("Finalizar sesion y cerrar limpiamente")
                    self.finish_session.clicked.connect(lambda: parent.close() if parent is not None else self.close())
                    self.layout().addWidget(self.finish_session)
                    self._session_clock_timer = QTimer(self)
                    def update_clock():
                        elapsed = (time.monotonic() - self._session_started) / 60
                        reached = elapsed >= target_minutes
                        self.session_clock.setText(
                            f"Sesion integrada: {elapsed:.1f} / {target_minutes} min" +
                            (" | objetivo cumplido" if reached else ""))
                        if reached and not self._target_announced:
                            self._target_announced = True
                            mark("session_target_reached", mode=mode, target_minutes=target_minutes)
                    self._session_clock_timer.timeout.connect(update_clock)
                    self._session_clock_timer.start(1000); update_clock()
                self._smoke_recognizer = None
                if smoke_audio is not None and not resource_stress:
                    QTimer.singleShot(max(1200, smoke_delay_ms), self._start_recorded_voice_smoke)
                if auto_close_ms > 0:
                    QTimer.singleShot(auto_close_ms,
                        lambda: parent.close() if parent is not None else self.close())

            def _start_recorded_voice_smoke(self):
                def decode():
                    try:
                        import numpy as np
                        from anydes.voice_input.core import UserInputEvent
                        from anydes.stt.parakeet.live_lab import ParakeetLiveRecognizer

                        with wave.open(str(smoke_audio), "rb") as source:
                            sample_rate = source.getframerate()
                            channels = source.getnchannels()
                            width = source.getsampwidth()
                            pcm = source.readframes(source.getnframes())
                        if width != 2 or channels != 1 or sample_rate != 16000:
                            raise ValueError(
                                f"smoke_audio_format:{sample_rate}Hz/{channels}ch/{width * 8}bit"
                            )
                        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                        if self._smoke_recognizer is None:
                            mark("parakeet_loading", source="recorded_voice_smoke")
                            self._smoke_recognizer = ParakeetLiveRecognizer()
                            mark("parakeet_ready", source="recorded_voice_smoke",
                                load_ms=getattr(self._smoke_recognizer, "load_ms", None))
                        result = self._smoke_recognizer.transcribe(audio)
                        audio_id = "smoke-" + uuid.uuid4().hex
                        event = UserInputEvent(
                            "voice",
                            str(result["text"]),
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            result.get("confidence"),
                            audio_id,
                        )
                        mark("recorded_voice_smoke_transcribed",
                            audio_id=audio_id,text_characters=len(event.text),
                            stt_ms=result.get("stt_ms"),rtf=result.get("rtf"),
                            hallucination=bool(result.get("hallucination",{}).get("suspected")),
                            language_drift=bool(result.get("language_drift")))
                        self.bridge.received.emit({
                            "kind": "transcript",
                            "revision": self.engine.revision,
                            "at": time.monotonic(),
                            "input": {
                                "source": event.source,
                                "text": event.text,
                                "timestamp": event.timestamp,
                                "confidence": event.confidence,
                                "audio_id": event.audio_id,
                            },
                            "endpoint_to_text_ms": result.get("stt_ms", 0.0),
                            "voice_end_to_text_ms": result.get("stt_ms", 0.0),
                            **result,
                        })
                    except Exception as exc:
                        mark("recorded_voice_smoke_failed",
                            error=f"{type(exc).__name__}: {exc}")
                threading.Thread(target=decode,
                    name="recorded-parakeet-smoke",daemon=True).start()

            def shutdown(self):
                clean = super().shutdown()
                if self._smoke_recognizer is not None:
                    self._smoke_recognizer.close()
                    self._smoke_recognizer = None
                return clean

        install(module, performance=mode == "performance",
            panel_factory=IntegratedParakeetPanel)
        if smoke_text.strip():
            from PySide6.QtCore import QTimer
            base_window = module.AnydesPySideWindow if mode == "normal" else module.PerformanceCompanionWindow

            class TextSmokeWindow(base_window):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._official_smoke_wait_started=time.monotonic()
                    self._official_smoke_deadline=self._official_smoke_wait_started+240.0
                    self._official_ready_marks=set()
                    mark("b4_warmup_wait_started",mode=mode)
                    mark("parakeet_not_requested",reason="text_smoke_uses_keyboard_pipeline")
                    QTimer.singleShot(100, self._wait_for_official_readiness)

                def _wait_for_official_readiness(self):
                    names=("_realtime_warmup_worker","_voice_warmup_worker") if mode=="normal" else (
                        "_assistant_warmup","_voice_warmup")
                    workers={name:getattr(self,name,None) for name in names}
                    for name,worker in workers.items():
                        if worker is not None and not worker.isRunning() and name not in self._official_ready_marks:
                            self._official_ready_marks.add(name)
                            mark("startup_component_ready",component=name,mode=mode,
                                elapsed_ms=(time.monotonic()-self._official_smoke_wait_started)*1000)
                    input_widget=self.input_dock.input if mode=="normal" else self.chat_window.input
                    manager_warmup=getattr(self.voice_manager,"_warmup_thread",None)
                    manager_ready=manager_warmup is None or not manager_warmup.is_alive()
                    workers_ready=all(worker is None or not worker.isRunning()
                        for worker in workers.values())
                    if workers_ready and manager_ready and input_widget.isEnabled():
                        mark("b4_ready",mode=mode,
                            warmup_wait_ms=(time.monotonic()-self._official_smoke_wait_started)*1000)
                        self._submit_official_smoke_text()
                        return
                    if time.monotonic()>=self._official_smoke_deadline:
                        mark("official_readiness_timeout",mode=mode,
                            workers={name:None if worker is None else worker.isRunning()
                                for name,worker in workers.items()},
                            manager_warmup_running=not manager_ready,input_enabled=input_widget.isEnabled())
                        return
                    QTimer.singleShot(100,self._wait_for_official_readiness)

                def _submit_official_smoke_text(self):
                    mark("text_smoke_submitted", characters=len(smoke_text), mode=mode)
                    self.send_text(smoke_text)

            if mode == "normal":
                module.AnydesPySideWindow = TextSmokeWindow
            else:
                module.PerformanceCompanionWindow = TextSmokeWindow
        if resource_stress:
            from PySide6.QtCore import QTimer
            base_window = module.AnydesPySideWindow if mode == "normal" else module.PerformanceCompanionWindow

            class ResourceStressWindow(base_window):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._stress_steps = []
                    prompt_turns = 72 if mode == "normal" else 36
                    voice_indices = {8, 25, 43, 61} if mode == "normal" else {7, 20, 31}
                    vision_indices = {18, 39, 60} if mode == "normal" else set()
                    self._stress_prompt_turns = prompt_turns
                    self._stress_vision_calls = len(vision_indices)
                    for index in range(prompt_turns):
                        if index in vision_indices:
                            self._stress_steps.append({"kind": "vision", "index": len([
                                value for value in vision_indices if value <= index
                            ])})
                        self._stress_steps.append({
                            "kind": "voice" if smoke_audio is not None and index in voice_indices else "text",
                            "text": RESOURCE_STRESS_PROMPTS[index % len(RESOURCE_STRESS_PROMPTS)],
                            "interrupt": (index + 1) % 5 == 0,
                        })
                    self._stress_index = 0
                    self._stress_idle_since = time.monotonic()
                    self._stress_next_due = time.monotonic() + 5.0
                    self._stress_waiting_interrupt = False
                    self._stress_interrupt_due = 0.0
                    self._stress_wait_started = 0.0
                    self._stress_vision_running = False
                    self._stress_finished = False
                    self._stress_runtime_ready = False
                    self._stress_ready_wait_started = time.monotonic()
                    self._stress_timer = QTimer(self)
                    self._stress_timer.timeout.connect(self._stress_tick)
                    self._stress_timer.start(250)
                    mark("resource_stress_started", steps=len(self._stress_steps),
                        prompt_turns=prompt_turns, voice_turns=len(voice_indices),
                        vision_calls=len(vision_indices),
                        planned_interruptions=prompt_turns // 5)

                def _stress_idle(self):
                    if mode == "normal":
                        return (self._worker is None and not self._voice_workers
                            and self._latest_voice_worker is None and not self._streaming_answer)
                    return (self._worker is None and self._voice_worker is None
                        and not self._streaming and self._turns.pending is None)

                def _stress_ready(self):
                    names=("_realtime_warmup_worker","_voice_warmup_worker") if mode=="normal" else (
                        "_assistant_warmup","_voice_warmup")
                    workers=[getattr(self,name,None) for name in names]
                    manager_warmup=getattr(self.voice_manager,"_warmup_thread",None)
                    input_widget=self.input_dock.input if mode=="normal" else self.chat_window.input
                    return (all(worker is not None and not worker.isRunning() for worker in workers)
                        and (manager_warmup is None or not manager_warmup.is_alive())
                        and input_widget.isEnabled())

                def _stress_tick(self):
                    if self._closing or self._stress_finished or self._stress_vision_running:
                        return
                    now = time.monotonic()
                    if not self._stress_runtime_ready:
                        if not self._stress_ready():
                            return
                        self._stress_runtime_ready = True
                        self._stress_next_due = now + 2.0
                        mark("b4_ready",mode=mode,
                            warmup_wait_ms=(now-self._stress_ready_wait_started)*1000)
                        mark("resource_stress_runtime_ready",mode=mode)
                        return
                    if self._stress_waiting_interrupt:
                        voice_worker=(self._latest_voice_worker if mode=="normal" else self._voice_worker)
                        if voice_worker is not None and not self._stress_interrupt_due:
                            self._stress_interrupt_due = now + 1.25
                        if self._stress_interrupt_due and now >= self._stress_interrupt_due:
                            self._stress_waiting_interrupt = False
                            self._stress_interrupt_due = 0.0
                            self._stress_dispatch(force=True)
                            return
                        if now - self._stress_wait_started > 75:
                            self._stress_waiting_interrupt = False
                            self._stress_interrupt_due = 0.0
                            mark("resource_stress_interrupt_skipped", reason="voice_did_not_start")
                        return
                    if self._stress_index >= len(self._stress_steps):
                        if self._stress_idle():
                            self._stress_finished = True
                            mark("resource_stress_workload_completed",
                                prompt_turns=self._stress_prompt_turns,
                                vision_calls=self._stress_vision_calls,mode=mode)
                        return
                    if not self._stress_idle():
                        self._stress_idle_since = None
                        return
                    if self._stress_idle_since is None:
                        self._stress_idle_since = now
                    if now < self._stress_next_due or now - self._stress_idle_since < 4.0:
                        return
                    self._stress_dispatch(force=False)

                def _stress_dispatch(self, *, force):
                    if self._stress_index >= len(self._stress_steps):
                        return
                    step = self._stress_steps[self._stress_index]
                    self._stress_index += 1
                    self._stress_idle_since = None
                    self._stress_next_due = time.monotonic() + 4.0
                    if step["kind"] == "vision":
                        if force:
                            mark("resource_stress_vision_deferred", index=step["index"])
                            self._stress_steps.insert(self._stress_index, step)
                            return
                        self._stress_start_vision(int(step["index"]))
                        return
                    mark("resource_stress_input", index=self._stress_index,
                        source=step["kind"], forced_interrupt=bool(force))
                    if step["kind"] == "voice":
                        self.input_panel._start_recorded_voice_smoke()
                    else:
                        self.send_text(step["text"])
                    if step.get("interrupt"):
                        self._stress_waiting_interrupt = True
                        self._stress_wait_started = time.monotonic()

                def _stress_start_vision(self, index):
                    self._stress_vision_running = True
                    mark("vision_probe_started", index=index)

                    def run_vision():
                        result = {}
                        try:
                            scripts = str(ROOT / "scripts")
                            if scripts not in sys.path:
                                sys.path.insert(0, scripts)
                            from validate_fast_canonical_voice_30m import _vision_probe
                            result = _vision_probe(self.assistant.config, index)
                        except Exception as exc:
                            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        finally:
                            mark("vision_probe_finished", index=index, result=result)
                            self._stress_vision_running = False
                            self._stress_next_due = time.monotonic() + 4.0

                    threading.Thread(target=run_vision,
                        name=f"resource-vision-{index}", daemon=True).start()

            if mode == "normal":
                module.AnydesPySideWindow = ResourceStressWindow
            else:
                module.PerformanceCompanionWindow = ResourceStressWindow
        if vision_quick_test:
            if mode != "normal":
                raise ValueError("vision_quick_test_requires_normal_mode")
            from PySide6.QtCore import QTimer
            from PIL import Image, ImageDraw
            import io

            base_window = module.AnydesPySideWindow

            class QuickVisionWindow(base_window):
                _VISION_CASES = (
                    ("blue_circle", "Mira mi pantalla y dime brevemente qué figura y color ves."),
                    ("green_squares", "Qué ves en mi pantalla ahora? Responde en una frase."),
                    ("red_triangle", "Describe mi pantalla de forma breve y clara."),
                )

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    from anydes.nucleo.consent import session_consents

                    session_consents.grant("screen_observation")
                    self._vision_case_index = 0
                    self._vision_waiting = False
                    self._vision_saw_voice = False
                    self._vision_dispatched_at = 0.0
                    self._vision_ready_started = time.monotonic()
                    self._vision_results = []
                    self._wrap_vision_brain()
                    self._quick_vision_timer = QTimer(self)
                    self._quick_vision_timer.timeout.connect(self._quick_vision_tick)
                    self._quick_vision_timer.start(100)
                    mark("quick_vision_test_started", planned_queries=3)

                def _wrap_vision_brain(self):
                    original = self.assistant.vision_brain.generate

                    def measured_generate(*args, **kwargs):
                        started = time.monotonic()
                        index = self._vision_case_index
                        mark("quick_vision_inference_started", index=index)
                        try:
                            response = original(*args, **kwargs)
                            elapsed_ms = (time.monotonic() - started) * 1000.0
                            row = {
                                "index": index,
                                "elapsed_ms": elapsed_ms,
                                "provider": getattr(response, "provider", ""),
                                "response": getattr(response, "text", ""),
                            }
                            self._vision_results.append(row)
                            mark("quick_vision_inference_finished", **row)
                            return response
                        except Exception as exc:
                            mark(
                                "quick_vision_inference_failed",
                                index=index,
                                elapsed_ms=(time.monotonic() - started) * 1000.0,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            raise

                    self.assistant.vision_brain.generate = measured_generate

                @staticmethod
                def _fixture(kind):
                    image = Image.new("RGB", (640, 420), (248, 248, 248))
                    draw = ImageDraw.Draw(image)
                    if kind == "blue_circle":
                        draw.ellipse((190, 80, 450, 340), fill=(35, 105, 220))
                    elif kind == "green_squares":
                        for left in (85, 245, 405):
                            draw.rounded_rectangle(
                                (left, 135, left + 120, 255),
                                radius=8,
                                fill=(35, 165, 80),
                            )
                    else:
                        draw.polygon(((320, 65), (150, 340), (490, 340)), fill=(220, 45, 55))
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=96, subsampling=0)
                    return output.getvalue()

                def _runtime_ready(self):
                    workers = [
                        getattr(self, "_realtime_warmup_worker", None),
                        getattr(self, "_voice_warmup_worker", None),
                    ]
                    manager_warmup = getattr(self.voice_manager, "_warmup_thread", None)
                    return (
                        all(worker is not None and not worker.isRunning() for worker in workers)
                        and (manager_warmup is None or not manager_warmup.is_alive())
                        and self.input_dock.input.isEnabled()
                    )

                def _runtime_idle(self):
                    return (
                        self._worker is None
                        and not self._voice_workers
                        and self._latest_voice_worker is None
                        and not self._streaming_answer
                    )

                def _quick_vision_tick(self):
                    if self._closing:
                        return
                    now = time.monotonic()
                    if not self._runtime_ready():
                        if now - self._vision_ready_started > 180.0:
                            mark("quick_vision_test_failed", reason="runtime_readiness_timeout")
                            self.close()
                        return
                    if self._vision_waiting:
                        if self._latest_voice_worker is not None or self._voice_workers:
                            self._vision_saw_voice = True
                        if self._vision_saw_voice and self._runtime_idle():
                            completed = self._vision_case_index
                            mark(
                                "quick_vision_query_finished",
                                index=completed,
                                wall_ms=(now - self._vision_dispatched_at) * 1000.0,
                                spoken_response=True,
                            )
                            self._vision_case_index += 1
                            self._vision_waiting = False
                            self._vision_saw_voice = False
                            if self._vision_case_index >= len(self._VISION_CASES):
                                mark(
                                    "quick_vision_test_completed",
                                    queries=self._vision_case_index,
                                    vision_results=self._vision_results,
                                )
                                QTimer.singleShot(1500, self.close)
                            return
                        if now - self._vision_dispatched_at > 100.0:
                            mark(
                                "quick_vision_test_failed",
                                reason="spoken_response_timeout",
                                index=self._vision_case_index,
                            )
                            self.close()
                        return
                    if self._vision_case_index >= len(self._VISION_CASES):
                        return
                    kind, prompt = self._VISION_CASES[self._vision_case_index]
                    from anydes.vision.state import get_screen_state

                    fixture_path = Path(os.environ["ANYDES_DATA_DIR"]) / f"vision-{self._vision_case_index + 1}-{kind}.jpg"
                    jpeg = self._fixture(kind)
                    fixture_path.write_bytes(jpeg)
                    state = get_screen_state()
                    state.set_watching(True)
                    state.store_capture(
                        jpeg,
                        captured_at=datetime.now(timezone.utc),
                        source_id=f"controlled:{kind}",
                        source_kind="controlled_fixture",
                        source_label=f"Fixture local {kind}",
                        session_id="quick-vision-v032",
                    )
                    self._vision_dispatched_at = now
                    self._vision_waiting = True
                    mark(
                        "quick_vision_query_started",
                        index=self._vision_case_index,
                        fixture=kind,
                        fixture_path=str(fixture_path),
                    )
                    self.send_text(prompt)

            module.AnydesPySideWindow = QuickVisionWindow
        mark("mode_ui_starting", mode=mode, target_minutes=target_minutes,
            stt_model="Parakeet TDT 0.6B v3 Q8", stt_device="CPU",
            resource_gate_allocator=os.environ.get("ANYDES_RESOURCE_GATE_CUDA_ALLOCATOR"),
            resource_gate_renderer_override=os.environ.get("ANYDES_RESOURCE_GATE_RENDERER_OVERRIDE"),
            official_runtime=official_runtime,
            official_launcher_sha256=sha256(OFFICIAL_LAUNCHER))
        # Performance's own parser must not receive this launcher's --mode and
        # --target-minutes flags.  Normal ignores them, but both paths now get
        # the same clean application argv.
        original_argv = sys.argv[:]
        try:
            sys.argv = [original_argv[0]]
            module.main()
        finally:
            sys.argv = original_argv

    return int(run_desktop(mode, run) or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("normal", "performance"), required=True)
    parser.add_argument("--target-minutes", type=int)
    parser.add_argument("--smoke-audio", type=Path)
    parser.add_argument("--smoke-text", default="")
    parser.add_argument("--resource-stress", action="store_true")
    parser.add_argument("--auto-close-ms", type=int, default=0)
    parser.add_argument("--smoke-delay-ms", type=int, default=1200)
    parser.add_argument("--vision-quick-test", action="store_true")
    args = parser.parse_args()
    target = args.target_minutes or (30 if args.mode == "normal" else 15)
    OUT.mkdir(parents=True, exist_ok=True)
    from anydes.voice_input.telemetry import start_monitor, mark
    monitor = start_monitor(args.mode, OUT)
    mark("launcher_started",mode=args.mode,target_minutes=target)
    before_hash = sha256(OFFICIAL_LAUNCHER)
    exit_code = 1
    try:
        try:
            exit_code = run_mode(
                args.mode,
                target,
                smoke_audio=args.smoke_audio,
                smoke_text=args.smoke_text,
                resource_stress=args.resource_stress,
                auto_close_ms=max(0, args.auto_close_ms),
                smoke_delay_ms=max(1200, args.smoke_delay_ms),
                vision_quick_test=args.vision_quick_test,
            )
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        return exit_code
    finally:
        after_hash = sha256(OFFICIAL_LAUNCHER)
        state = wait_for_cleanup()
        mark("post_runtime_cleanup", exit_code=exit_code,
            official_launcher_unchanged=before_hash == after_hash,
            official_launcher_sha256=after_hash, cleanup=state)
        monitor.close()
        try:
            (monitor.directory / "session-metadata.json").write_text(json.dumps({
                "mode": args.mode,
                "target_minutes": target,
                "actual_elapsed_seconds": time.monotonic() - monitor.started,
                "exit_code": exit_code,
                "candidate": "Parakeet TDT 0.6B v3 Q8 CPU",
                "vision_quick_test": args.vision_quick_test,
                "production_promoted": True,
                "official_launcher_unchanged": before_hash == after_hash,
                "official_launcher_sha256": after_hash,
                "cleanup": state,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
