from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from anydes.cognition import BodyController, Intention, SpeechIntention
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.config import Config
from anydes.nucleo.sensors import sensor_hub
from anydes.proactivo.initiative import ProactiveEngine
from anydes.voz.recorder import AudioRecorder
from anydes.voz.manager_factory import build_voice_manager
from anydes.voz.speaker_id import SpeakerRegistry
from anydes.voz.stt import build_stt, warmup_vosk_if_available

@dataclass
class VoiceCallbacks:
    on_user: Callable[[str], None] | None = None
    on_assistant: Callable[[str], None] | None = None
    on_system: Callable[[str], None] | None = None
    on_status: Callable[[str], None] | None = None

    def user(self, text: str) -> None:
        if self.on_user:
            self.on_user(text)
        else:
            print(f"Tu: {text}")

    def assistant(self, text: str) -> None:
        if self.on_assistant:
            self.on_assistant(text)
        else:
            print(f"Anydes: {text}")

    def system(self, text: str) -> None:
        if self.on_system:
            self.on_system(text)
        else:
            print(text)

    def status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)
        else:
            print(text)


class VoiceController:
    def __init__(
        self,
        assistant: AnydesAssistant,
        config: Config,
        activation: str,
        callbacks: VoiceCallbacks | None = None,
        *,
        embedded: bool = False,
        shared_speaking_event: threading.Event | None = None,
        disable_internal_proactive: bool = False,
    ) -> None:
        self.assistant = assistant
        self.config = config
        self.activation = activation.strip().lower()
        self.callbacks = callbacks or VoiceCallbacks()
        self.embedded = embedded
        self.recorder = AudioRecorder(seconds=config.voice_seconds)
        self.stt = build_stt(config)
        self.voice_manager = build_voice_manager(config, metrics=assistant.metrics)
        self.body_controller = BodyController.from_config(config, events=assistant.events)
        self.speakers = SpeakerRegistry(
            config.speaker_profiles_path,
            threshold=config.speaker_match_threshold,
        )
        self.proactive = ProactiveEngine(config, assistant)
        self._stop_event = threading.Event()
        self._disable_internal_proactive = disable_internal_proactive
        self._speaking = shared_speaking_event or threading.Event()

        self._listen_event = threading.Event()
        self._typed_commands: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.last_activity = time.monotonic()
        self.last_proactive = 0.0
        self._last_short_text = ""
        self._last_short_text_at = 0.0
        self.last_voice_output_result = None
        self._response_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anydes-voice-brain")

    @property
    def tts(self):
        return self.voice_manager.backend

    @tts.setter
    def tts(self, backend) -> None:
        self.voice_manager.backend = backend

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.recorder.reset()
        threading.Thread(target=warmup_vosk_if_available, args=(self.config,), daemon=True).start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="anydes-voice")
        self._thread.start()
        sensor_hub.register("microphone", str(id(self)), self.stop)

    def stop(self) -> None:
        self._stop_event.set()
        self._listen_event.set()
        self.recorder.cancel()
        self.stt.cancel()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        sensor_hub.unregister("microphone", str(id(self)))
        self._response_pool.shutdown(wait=False, cancel_futures=True)

    def request_listen(self) -> None:
        self._listen_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._is_continuous_mode():
            self._run_continuous()
        else:
            self._run_button()

    def _run_button(self) -> None:
        self.callbacks.system(
            "Modo voz por boton. Usa Escuchar ahora o el chat; escribe `salir` en el chat para detener."
        )
        while not self._stop_event.is_set():
            if self._listen_event.wait(timeout=0.4):
                self._listen_event.clear()
                if self._stop_event.is_set():
                    return
                self._listen_and_respond()

    def _run_continuous(self) -> None:
        self.callbacks.system("Modo continuo activo en la interfaz.")
        if self._requires_wake_word():
            self.callbacks.system(f"Solo respondo cuando escuche `{self.config.wake_word}`.")
        if not self.embedded:
            threading.Thread(target=self._read_typed_commands, daemon=True).start()

        while not self._stop_event.is_set():
            while not self._typed_commands.empty():
                command = self._typed_commands.get().strip()
                if self._handle_control_command(command):
                    return
                if command:
                    self._handle_user_text(command, input_mode="typed")

            if not self._disable_internal_proactive and self._maybe_speak_proactively():
                continue

            user_text = self._listen_for_text()
            if user_text:
                self._handle_user_text(user_text)
            else:
                time.sleep(self.config.continuous_idle_seconds)

    def _read_typed_commands(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._typed_commands.put(input())
            except (EOFError, KeyboardInterrupt):
                self._typed_commands.put("salir")
                return

    def _listen_and_respond(self) -> None:
        user_text = self._listen_for_text()
        if user_text:
            self._handle_user_text(user_text)

    def _handle_control_command(self, command: str) -> bool:
        normalized = command.lower()
        if normalized in {"salir", "exit", "quit", "detener voz", "parar voz"}:
            self.callbacks.system("Sesion de voz cerrada.")
            self._stop_event.set()
            return True
        if normalized == "voces":
            self.callbacks.system(self.speakers.list_profiles())
            return False
        if normalized.startswith("registrar voz "):
            name = command[len("registrar voz ") :].strip()
            self._enroll_voice(name)
            return False
        return False

    def _handle_user_text(self, user_text: str, *, input_mode: str = "voice") -> None:
        if not user_text:
            self.callbacks.system("No escuche texto claro.")
            return

        if self._should_ignore_transcript(user_text):
            return

        if self._requires_wake_word():
            wake = self.config.wake_word.lower()
            if wake not in user_text.lower():
                self.callbacks.system(f"Ignorado: no escuche la palabra clave `{self.config.wake_word}`.")
                return

        self.last_activity = time.monotonic()
        self.callbacks.user(user_text)

        self.callbacks.status("Pensando...")
        try:
            future = self._response_pool.submit(
                self.assistant.respond,
                user_text,
                interactive=True,
                input_mode=input_mode,
            )
            response = future.result(timeout=120.0)
        except Exception:
            response = "Perdon... no pude procesar eso ahora. ¿Podrias repetirlo?"
            self._speak(response)
            return
        intention = self.assistant.last_intention or _voice_intention_from_text(response)
        self._speak_intention(intention, display_text=response)

    def _should_ignore_transcript(self, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        if len(normalized) < 2:
            return True

        now = time.monotonic()
        short_greetings = {"hola", "hey", "buenas", "ola", "hi"}
        if normalized in short_greetings:
            if normalized == self._last_short_text and now - self._last_short_text_at < 45.0:
                return True
            self._last_short_text = normalized
            self._last_short_text_at = now
        return False

    def _speak(self, text: str) -> None:
        self._speak_intention(_voice_intention_from_text(text), display_text=text)

    def _speak_intention(self, intention: Intention, *, display_text: str | None = None) -> None:
        text = display_text if display_text is not None else intention.speech
        self.callbacks.assistant(text)
        self.callbacks.status("Hablando...")
        self._speaking.set()
        try:
            self.last_voice_output_result = self.voice_manager.speak(intention.speech_intention)
            if self.last_voice_output_result.status == "failed":
                self.callbacks.system(f"No pude reproducir voz: {self.last_voice_output_result.error}")
        except Exception as exc:
            self.callbacks.system(f"No pude reproducir voz: {exc}")
        finally:
            self._speaking.clear()
            self.callbacks.status("Lista.")

    def _speak_visual(self, text: str) -> None:
        _ = text
        raise RuntimeError("Visual observations must pass through ReasoningEngine before speech.")

    def _maybe_speak_proactively(self) -> bool:
        if not self.config.proactive_enabled:
            return False
        now = time.monotonic()
        if now - self.last_activity < self.config.proactive_interval_seconds:
            return False
        if now - self.last_proactive < self.config.proactive_interval_seconds:
            return False

        current = datetime.now()
        allowed_minutes = _minute_marks(self.config.proactive_minute_marks)
        if allowed_minutes and current.minute not in allowed_minutes:
            return False

        self.last_proactive = now
        self.last_activity = now
        message = self.proactive.build_message(current)
        if not message.text.strip():
            return False
        if message.intention is not None:
            self._speak_intention(message.intention)
        else:
            self._speak(message.text)
        return True

    def _listen_for_text(self) -> str:
        if self._speaking.is_set():
            time.sleep(0.15)
            return ""

        audio_path = self.config.voice_dir / "entrada.wav"
        if self.stt.captures_audio_file and audio_path.exists():
            audio_path.unlink()
        try:
            self._speaking.set()
            self.body_controller.apply(_body_process_intention("listening"))
            self.callbacks.status("Escuchando...")
            user_text = self.stt.listen(self.recorder, audio_path)
            self.body_controller.apply(_body_process_intention("thinking"))
        except Exception as exc:
            self.body_controller.apply(_body_process_intention("neutral"))
            self.callbacks.system(f"Voz no disponible: {exc}")
            return ""
        finally:
            if self._speaking.is_set():
                self._speaking.clear()

        if self.stt.captures_audio_file and audio_path.exists():
            try:
                match = self.speakers.identify(audio_path)
                self.assistant.set_current_speaker(match)
                self.callbacks.system(f"Voz detectada: {match.label()}")
            except Exception as exc:
                self.assistant.set_current_speaker(None)
                self.callbacks.system(f"No pude reconocer perfil de voz: {exc}")
        else:
            self.assistant.set_current_speaker(None)
        return user_text

    def _enroll_voice(self, name: str) -> None:
        audio_path = self.config.voice_dir / f"registro_voz_{name.strip().replace(' ', '_')}.wav"
        if not name:
            self.callbacks.system("Indica un nombre. Ejemplo: registrar voz DemoUser")
            return
        try:
            self.recorder.record_wav(audio_path)
            match = self.speakers.enroll(name, audio_path)
        except Exception as exc:
            self.callbacks.system(f"No pude registrar esa voz: {exc}")
            return
        self.callbacks.system(
            f"Voz registrada para {match.name}. Muestra: pitch {match.features.pitch_hz:.1f} Hz."
        )

    def _is_continuous_mode(self) -> bool:
        return self.activation in {
            "continuous",
            "always",
            "always_on",
            "continuous_wake_word",
            "always_wake_word",
        }

    def _requires_wake_word(self) -> bool:
        return self.activation in {
            "wake_word",
            "continuous_wake_word",
            "always_wake_word",
        }


def _minute_marks(raw: str) -> set[int]:
    marks = set()
    for part in raw.split(","):
        try:
            minute = int(part.strip())
        except ValueError:
            continue
        if 0 <= minute <= 59:
            marks.add(minute)
    return marks


def _body_process_intention(state: str) -> Intention:
    gestures = {
        "listening": "listening",
        "thinking": "thinking",
        "neutral": "idle",
    }
    expressions = {
        "listening": "neutral",
        "thinking": "thinking",
        "neutral": "neutral",
    }
    return Intention(
        id=f"intention:body_process:{state}",
        intent_type="system_body_process",
        speech="",
        avatar={
            "expression": expressions.get(state, "neutral"),
            "gesture": gestures.get(state, "idle"),
            "gaze": "user",
        },
        confidence=1.0,
        reasoning_summary="Deterministic microphone process state; no semantic behavior decision.",
    )


def _voice_intention_from_text(text: str, *, intent_type: str = "compatibility:voice_output") -> Intention:
    speech = SpeechIntention(
        text=text,
        emotion="warm" if text.strip() else "calm",
        energy=0.36,
        warmth=0.84,
        confidence=0.74,
        thinking=0.12,
        curiosity=0.42,
        pause_before=0.16,
        pause_after=0.18,
        speech_rate=0.9,
        pitch_variation=0.14,
        volume=0.9,
        smile=0.28,
    )
    return Intention(
        id=f"intention:{intent_type.replace(':', '_')}",
        intent_type=intent_type,
        speech=text,
        speech_intention=speech,
        avatar={
            "expression": "speaking" if text.strip() else "neutral",
            "gesture": "talk" if text.strip() else "idle",
        },
        confidence=0.9,
        reasoning_summary="Voice compatibility adapter created an Intention for TTS output.",
    )
