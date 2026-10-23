from __future__ import annotations

import json
import subprocess
import threading
import wave
from pathlib import Path

from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision
from anydes.voz.recorder import AudioRecorder
from anydes.voz.process import hidden_process_kwargs
from anydes.voz.transcript import normalize_transcript

_vosk_models: dict[str, object] = {}
_vosk_lock = threading.Lock()


class SpeechToText:
    captures_audio_file = True

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError

    def listen(self, recorder: AudioRecorder, audio_path: Path) -> str:
        recorder.record_wav(audio_path)
        raw = self.transcribe(audio_path)
        return normalize_transcript(raw)

    def cancel(self) -> None:
        return None


class DisabledSpeechToText(SpeechToText):
    def transcribe(self, audio_path: Path) -> str:
        raise RuntimeError("STT desactivado. Activa ANYDES_STT_PROVIDER=windows u openai.")


class WindowsSpeechToText(SpeechToText):
    captures_audio_file = False

    def __init__(self, seconds: int, script_path: Path, culture: str) -> None:
        self.seconds = seconds
        self.script_path = script_path
        self.culture = culture
        self._process = None
        self._process_lock = threading.Lock()

    def transcribe(self, audio_path: Path) -> str:
        raise RuntimeError("Windows STT escucha directamente desde el microfono.")

    def listen(self, recorder: AudioRecorder, audio_path: Path) -> str:
        process = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.script_path),
                "-Seconds",
                str(self.seconds),
                "-Culture",
                self.culture,
            ],
            capture_output=True,
            text=True,
            **hidden_process_kwargs(),
        )
        with self._process_lock:
            self._process = process
        try:
            stdout, stderr = process.communicate(timeout=self.seconds + 10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("Windows STT excedio el tiempo de escucha.")
        finally:
            with self._process_lock:
                self._process = None
        if process.returncode == 2:
            raise RuntimeError(
                f"Windows no encontro un reconocedor de voz para {self.culture}. "
                "Puede tener voz para hablar en espanol, pero no reconocimiento para escuchar."
            )
        if process.returncode != 0:
            raise RuntimeError(stderr.strip() or "Windows STT fallo.")
        return normalize_transcript(stdout.strip())

    def cancel(self) -> None:
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


class OpenAISpeechToText(SpeechToText):
    def __init__(self, model: str, config: Config) -> None:
        self.model = model
        self.config = config

    def transcribe(self, audio_path: Path) -> str:
        if not external_access_decision(self.config, "external_stt").allowed:
            raise RuntimeError("STT externo bloqueado por la configuracion de privacidad.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Falta instalar `openai` para transcripcion.") from exc

        client = OpenAI()
        with audio_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        return normalize_transcript(str(result.text).strip())


def _get_vosk_model(model_dir: Path):
    key = str(model_dir.resolve())
    with _vosk_lock:
        cached = _vosk_models.get(key)
        if cached is not None:
            return cached
        try:
            from vosk import Model
        except ImportError as exc:
            raise RuntimeError("Falta instalar `vosk` para reconocimiento local.") from exc
        if not model_dir.exists():
            raise RuntimeError(
                f"No encontre el modelo Vosk en {model_dir}. "
                "Descarga el modelo espanol y descomprimelo ahi."
            )
        model = Model(key)
        _vosk_models[key] = model
        return model


class VoskSpeechToText(SpeechToText):
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    def transcribe(self, audio_path: Path) -> str:
        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:
            raise RuntimeError("Falta instalar `vosk` para reconocimiento local.") from exc

        model = _get_vosk_model(self.model_dir)
        chunks: list[str] = []
        with wave.open(str(audio_path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise RuntimeError("El audio debe ser mono para Vosk.")
            recognizer = KaldiRecognizer(model, audio.getframerate())
            while True:
                data = audio.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    chunks.append(json.loads(recognizer.Result()).get("text", ""))
            chunks.append(json.loads(recognizer.FinalResult()).get("text", ""))

        return " ".join(chunk for chunk in chunks if chunk).strip()


def build_stt(config: Config) -> SpeechToText:
    if config.stt_provider.lower() == "vosk":
        return VoskSpeechToText(config.vosk_model_dir)
    if config.stt_provider.lower() == "windows":
        script_path = Path(__file__).resolve().parent / "scripts" / "windows_listen.ps1"
        return WindowsSpeechToText(config.voice_seconds, script_path, config.stt_culture)
    if config.stt_provider.lower() == "openai" and external_access_decision(
        config,
        "external_stt",
    ).allowed:
        return OpenAISpeechToText(config.stt_model, config)
    return DisabledSpeechToText()


def warmup_vosk_if_available(config: Config) -> None:
    if config.stt_provider.lower() != "vosk":
        return
    try:
        _get_vosk_model(config.vosk_model_dir)
    except Exception:
        pass
