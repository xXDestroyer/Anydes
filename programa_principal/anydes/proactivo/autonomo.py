from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from anydes.cognition import Intention
from anydes.nucleo.config import Config
from anydes.nucleo.orquestador import BehaviorOrchestrator
from anydes.proactivo.initiative import ProactiveEngine
from anydes.voz.manager_factory import build_voice_manager

logger = logging.getLogger("anydes.proactivo.autonomo")


def _minute_marks(raw: str) -> set[int]:
    marks: set[int] = set()
    for part in raw.split(","):
        try:
            minute = int(part.strip())
        except ValueError:
            continue
        if 0 <= minute <= 59:
            marks.add(minute)
    return marks


class ProactiveDaemon:
    """Hilo autónomo que hace que Anydes hable sin esperar input."""

    def __init__(
        self,
        config: Config,
        callback_apend_log: Callable[[str, str], None] | None = None,
        shared_speaking_event: threading.Event | None = None,
        orchestrator: BehaviorOrchestrator | None = None,
    ) -> None:
        self.config = config
        self.callback_apend_log = callback_apend_log or (lambda speaker, text: None)
        self.engine = ProactiveEngine(config)
        self.voice_manager = build_voice_manager(config)
        self.shared_speaking_event = shared_speaking_event or threading.Event()
        self.orchestrator = orchestrator
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_proactive = 0.0
        self._last_user_activity = time.monotonic()

    def start(self) -> None:
        if not self.config.proactive_enabled:
            logger.info("Proactividad autónoma deshabilitada en config")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ProactiveDaemon")
        self._thread.start()
        logger.info("ProactiveDaemon iniciado.")
        self.callback_apend_log("Sistema", "🤖 Proactividad autónoma activada.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        logger.info("ProactiveDaemon detenido.")
        self.callback_apend_log("Sistema", "🤖 Proactividad autónoma desactivada.")

    def notify_user_activity(self) -> None:
        self._last_user_activity = time.monotonic()

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self.config.proactive_enabled:
                time.sleep(1.0)
                continue

            if self.shared_speaking_event.is_set():
                time.sleep(0.2)
                continue

            now = time.monotonic()
            if now - self._last_user_activity < min(60.0, float(self.config.proactive_interval_seconds)):
                time.sleep(1.0)
                continue

            if now - self._last_proactive < self.config.proactive_interval_seconds:
                time.sleep(1.0)
                continue

            current = datetime.now()
            allowed_minutes = _minute_marks(self.config.proactive_minute_marks)
            if allowed_minutes and current.minute not in allowed_minutes:
                time.sleep(5.0)
                continue

            message = self.engine.build_message(current)

            if not message.text.strip():
                time.sleep(1.0)
                continue

            self._last_proactive = time.monotonic()
            self._last_user_activity = self._last_proactive
            self.callback_apend_log("Anydes (proactiva)", message.text)

            self.shared_speaking_event.set()
            try:
                result = self.voice_manager.speak(
                    (
                        message.intention.speech_intention
                        if message.intention is not None
                        else _voice_intention_from_text(message.text).speech_intention
                    )
                )
                if result.status == "failed":
                    raise RuntimeError(result.error or "TTS failed")
            except Exception as exc:
                logger.error("Error en proactividad autónoma: %s", exc, exc_info=True)
                self.callback_apend_log("Sistema", f"Error proactivo: {exc}")
            finally:
                self.shared_speaking_event.clear()

            time.sleep(1.0)


def _voice_intention_from_text(text: str) -> Intention:
    return Intention(
        id="intention:proactive_voice_output",
        intent_type="compatibility:proactive_voice_output",
        speech=text,
        avatar={
            "expression": "speaking" if text.strip() else "neutral",
            "gesture": "talk" if text.strip() else "idle",
        },
        confidence=0.9,
        reasoning_summary="Proactive compatibility adapter created an Intention for TTS output.",
    )
