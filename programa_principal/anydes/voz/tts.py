from __future__ import annotations

import subprocess
import sys
import shutil
import site
import struct
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision
from anydes.voz.process import hidden_process_kwargs
from anydes.voz.spoken_text import prepare_for_continuous_speech, prepare_for_speech

if TYPE_CHECKING:
    from anydes.voz.interpreter import VoiceRenderPlan


@dataclass(frozen=True)
class VoiceBackendTrace:
    status: str = "idle"
    phase: str = ""
    error_code: str = ""
    input_characters: int = 0
    spoken_characters: int = 0
    synthesis_ms: float = 0.0
    processing_ms: float = 0.0
    playback_ms: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    frame_count: int = 0
    cancelled: bool = False
    output_mode: str = "single_wav"


class TextToSpeech:
    def speak(self, text: str) -> None:
        raise NotImplementedError

    def speak_plan(self, plan: VoiceRenderPlan) -> None:
        self.speak(plan.backend_text)

    def cancel(self) -> None:
        """Best-effort interruption hook for interactive playback."""


class ConsoleSpeech(TextToSpeech):
    def speak(self, text: str) -> None:
        print(f"Anydes: {text}")


class WindowsTextToSpeech(TextToSpeech):
    def __init__(self, script_path: Path, output_dir: Path, rate: int) -> None:
        self.script_path = script_path
        self.output_dir = output_dir
        self.rate = rate
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()

    def speak(self, text: str) -> None:
        self._speak_with_rate(text, self.rate)

    def speak_plan(self, plan: VoiceRenderPlan) -> None:
        expressive_rate = self.rate + round((plan.speech_rate - 0.9) * 10.0)
        self._speak_with_rate(plan.backend_text, max(-10, min(10, expressive_rate)))

    def _speak_with_rate(self, text: str, rate: int) -> None:
        spoken = prepare_for_speech(text)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.output_dir / f"respuesta_{uuid.uuid4().hex}.txt"
        text_path.write_text(spoken, encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self.script_path),
                    "-TextPath",
                    str(text_path),
                    "-Rate",
                    str(rate),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **hidden_process_kwargs(),
            )
            self._set_process(process)
            try:
                _, stderr = process.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError("Windows TTS excedio el tiempo de reproduccion.")
            if process.returncode != 0:
                raise RuntimeError(f"Windows TTS fallo: {stderr.strip()}")
        finally:
            self._set_process(None)
            text_path.unlink(missing_ok=True)

    def cancel(self) -> None:
        self._terminate_current_process()

    def _set_process(self, process: subprocess.Popen[str] | None) -> None:
        with self._process_lock:
            self._process = process

    def _terminate_current_process(self) -> None:
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


class OpenAITextToSpeech(TextToSpeech):
    def __init__(self, model: str, voice: str, output_dir: Path, config: Config) -> None:
        self.model = model
        self.voice = voice
        self.output_dir = output_dir
        self.config = config
        self.counter = 0

    def speak(self, text: str) -> None:
        if not external_access_decision(self.config, "external_tts").allowed:
            raise RuntimeError("TTS externo bloqueado por la configuracion de privacidad.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Falta instalar `openai` para voz TTS.") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        output_path = self.output_dir / f"respuesta_{self.counter:03d}.mp3"

        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
        ) as response:
            response.stream_to_file(output_path)



class EdgeTextToSpeech(TextToSpeech):
    def __init__(
        self,
        voice: str,
        output_dir: Path,
        rate: str,
        pitch: str,
        avatar_state_path: Path,
        config: Config,
    ) -> None:
        self.voice = voice
        self.output_dir = output_dir
        self.rate = rate
        self.pitch = pitch
        self.avatar_state_path = avatar_state_path
        self.config = config
        self._cancelled = threading.Event()

    def speak(self, text: str) -> None:
        self._speak_with_settings(text, rate=self.rate, pitch=self.pitch)

    def speak_plan(self, plan: VoiceRenderPlan) -> None:
        base_rate = _percentage_value(self.rate)
        expressive_rate = base_rate + round((plan.speech_rate - 0.9) * 100.0)
        self._speak_with_settings(
            plan.backend_text,
            rate=f"{max(-45, min(30, expressive_rate)):+d}%",
            pitch=self.pitch,
        )

    def _speak_with_settings(self, text: str, *, rate: str, pitch: str) -> None:
        if not external_access_decision(self.config, "external_tts").allowed:
            raise RuntimeError("TTS externo bloqueado por la configuracion de privacidad.")
        spoken = prepare_for_speech(text)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled.clear()
        output_path = self.output_dir / f"respuesta_edge_{uuid.uuid4().hex}.mp3"
        try:
            self._run_async(self._generate(spoken, output_path, rate=rate, pitch=pitch))
        except ImportError as exc:
            raise RuntimeError("Falta instalar `edge-tts` para usar la voz Edge del nuevo core.") from exc
        except Exception as exc:
            raise RuntimeError(f"No pude generar voz Edge TTS: {exc}") from exc
        try:
            if not self._cancelled.is_set():
                self._play_mp3(output_path)
        finally:
            output_path.unlink(missing_ok=True)

    async def _generate(self, text: str, output_path: Path, *, rate: str, pitch: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=rate,
            pitch=pitch,
        )
        await communicate.save(str(output_path))

    def _run_async(self, awaitable) -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(awaitable)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _play_mp3(self, output_path: Path) -> None:
        try:
            import pygame
        except ImportError:
            return

        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(output_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not self._cancelled.is_set():
            time.sleep(0.1)
        if self._cancelled.is_set():
            pygame.mixer.music.stop()
        pygame.mixer.music.unload()

    def cancel(self) -> None:
        self._cancelled.set()
        try:
            import pygame
        except ImportError:
            return
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()


class PiperTextToSpeech(TextToSpeech):
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        output_path: Path,
        length_scale: float,
        sentence_silence: float,
        pitch_factor: float,
        avatar_state_path: Path,
        continuous_audio: bool = False,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.output_path = output_path
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence
        self.pitch_factor = pitch_factor
        self.avatar_state_path = avatar_state_path
        self.continuous_audio = bool(continuous_audio)
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()
        self.last_trace = VoiceBackendTrace()

    def speak(self, text: str) -> None:
        self._speak_with_performance(
            text,
            length_scale=self.length_scale,
            sentence_silence=self.sentence_silence,
            pitch_factor=self.pitch_factor,
        )

    def speak_plan(self, plan: VoiceRenderPlan) -> None:
        rate_ratio = 0.9 / max(0.72, plan.speech_rate)
        length_scale = max(0.78, min(1.5, self.length_scale * rate_ratio))
        silence_scale = 0.82 + plan.micro_pause_scale * 0.36
        sentence_silence = max(0.05, min(1.2, self.sentence_silence * silence_scale))
        self._speak_with_performance(
            plan.backend_text,
            length_scale=length_scale,
            sentence_silence=sentence_silence,
            pitch_factor=self.pitch_factor,
        )

    def _speak_with_performance(
        self,
        text: str,
        *,
        length_scale: float,
        sentence_silence: float,
        pitch_factor: float,
    ) -> None:
        spoken = (
            prepare_for_continuous_speech(text)
            if self.continuous_audio
            else prepare_for_speech(text)
        )
        self._cancelled.clear()

        if not self.model_path.exists() or not self.config_path.exists():
            self.last_trace = VoiceBackendTrace(
                status="failed",
                phase="preflight",
                error_code="piper_model_unavailable",
                input_characters=len(text),
                spoken_characters=len(spoken),
            )
            raise RuntimeError(
                "No encontre el modelo de Piper. Revisa `programa_principal\\modelos\\piper`."
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path.with_name(
            f"{self.output_path.stem}_{uuid.uuid4().hex}{self.output_path.suffix}"
        )
        command = [
            self._piper_executable(),
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--output_file",
            str(output_path),
            "--length-scale",
            str(length_scale),
            "--sentence-silence",
            str(sentence_silence),
        ]
        process: subprocess.Popen[str] | None = None
        stderr = ""
        synthesis_ms = 0.0
        processing_ms = 0.0
        playback_ms = 0.0
        phase = "synthesis"
        try:
            synthesis_started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **hidden_process_kwargs(),
            )
            self._set_process(process)
            try:
                _, stderr = process.communicate(input=spoken, timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError("Piper excedio el tiempo de generacion.")
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0
        except FileNotFoundError as exc:
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0
            self.last_trace = VoiceBackendTrace(
                status="failed",
                phase="synthesis",
                error_code="synthesis_FileNotFoundError",
                input_characters=len(text),
                spoken_characters=len(spoken),
                synthesis_ms=synthesis_ms,
                cancelled=self._cancelled.is_set(),
            )
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                "No encontre el comando `piper`. Instala `piper-tts` con pip."
            ) from exc
        except Exception as exc:
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0
            self.last_trace = VoiceBackendTrace(
                status="failed",
                phase="synthesis",
                error_code=f"synthesis_{type(exc).__name__}",
                input_characters=len(text),
                spoken_characters=len(spoken),
                synthesis_ms=synthesis_ms,
                cancelled=self._cancelled.is_set(),
            )
            output_path.unlink(missing_ok=True)
            raise
        finally:
            self._set_process(None)

        try:
            if process is None:
                raise RuntimeError("Piper no pudo iniciar el proceso de voz.")
            if process.returncode != 0:
                raise RuntimeError(f"Piper fallo: {stderr.strip()}")
            phase = "processing"
            processing_started = time.perf_counter()
            if self.continuous_audio:
                self._post_process_wav(output_path, pitch_factor)
            elif pitch_factor > 1.01:
                self._raise_pitch(output_path, pitch_factor)
            processing_ms = (time.perf_counter() - processing_started) * 1000.0
            sample_rate, channels, frame_count = self._wav_metadata(output_path)
            if self._cancelled.is_set():
                self.last_trace = VoiceBackendTrace(
                    status="cancelled",
                    phase="before_playback",
                    input_characters=len(text),
                    spoken_characters=len(spoken),
                    synthesis_ms=synthesis_ms,
                    processing_ms=processing_ms,
                    sample_rate=sample_rate,
                    channels=channels,
                    frame_count=frame_count,
                    cancelled=True,
                )
                return
            phase = "playback"
            playback_started = time.perf_counter()
            self._play_wav(output_path)
            playback_ms = (time.perf_counter() - playback_started) * 1000.0
            self.last_trace = VoiceBackendTrace(
                status="cancelled" if self._cancelled.is_set() else "spoken",
                phase="completed",
                input_characters=len(text),
                spoken_characters=len(spoken),
                synthesis_ms=synthesis_ms,
                processing_ms=processing_ms,
                playback_ms=playback_ms,
                sample_rate=sample_rate,
                channels=channels,
                frame_count=frame_count,
                cancelled=self._cancelled.is_set(),
            )
        except Exception as exc:
            self.last_trace = VoiceBackendTrace(
                status="failed",
                phase=phase,
                error_code=f"{phase}_{type(exc).__name__}",
                input_characters=len(text),
                spoken_characters=len(spoken),
                synthesis_ms=synthesis_ms,
                processing_ms=processing_ms,
                playback_ms=playback_ms,
                cancelled=self._cancelled.is_set(),
            )
            raise
        finally:
            output_path.unlink(missing_ok=True)

    def _play_wav(self, output_path: Path) -> None:
        if self._cancelled.is_set():
            return
        process = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(New-Object Media.SoundPlayer '{output_path}').PlaySync();",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_process_kwargs(),
        )
        self._set_process(process)
        try:
            process.wait(timeout=120)
        finally:
            self._set_process(None)

    def cancel(self) -> None:
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _set_process(self, process: subprocess.Popen[str] | None) -> None:
        with self._process_lock:
            self._process = process

    def _piper_executable(self) -> str:
        found = shutil.which("piper")
        if found:
            return found

        candidates = [
            Path(site.getusersitepackages()).parent / "Scripts" / "piper.exe",
            Path(sys.executable).parent / "Scripts" / "piper.exe",
            Path(sys.executable).parent / "piper.exe",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate)
            except PermissionError:
                return str(candidate)
        return "piper"

    def _raise_pitch(self, wav_path: Path, factor: float) -> None:
        with wave.open(str(wav_path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())

        if params.sampwidth != 2:
            return

        samples = list(struct.unpack("<" + "h" * (len(frames) // 2), frames))
        channels = params.nchannels
        frame_count = len(samples) // channels
        if frame_count < 4:
            return

        new_frame_count = max(1, int(frame_count / factor))
        shifted: list[int] = []
        for frame_index in range(new_frame_count):
            source_pos = frame_index * factor
            left = int(source_pos)
            frac = source_pos - left
            right = min(left + 1, frame_count - 1)
            for channel in range(channels):
                a = samples[left * channels + channel]
                b = samples[right * channels + channel]
                shifted.append(int(a + (b - a) * frac))

        packed = struct.pack("<" + "h" * len(shifted), *shifted)
        with wave.open(str(wav_path), "wb") as target:
            target.setparams(params)
            target.writeframes(packed)

    def _post_process_wav(self, wav_path: Path, factor: float) -> None:
        """Apply one anti-aliased pitch pass and remove playback edge clicks."""

        try:
            import numpy as np
        except ImportError:
            if factor > 1.01:
                self._raise_pitch(wav_path, factor)
            return

        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        if sample_width != 2 or not frames:
            return

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
        samples = samples.reshape(-1, channels)
        factor = max(0.75, min(1.35, float(factor)))
        if factor > 1.001 and len(samples) >= 64:
            taps = 63
            cutoff = min(0.49, 0.47 / factor)
            positions = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
            kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * positions)
            kernel *= np.hamming(taps)
            kernel /= np.sum(kernel)
            filtered = np.empty_like(samples)
            for channel in range(channels):
                filtered[:, channel] = np.convolve(
                    samples[:, channel],
                    kernel,
                    mode="same",
                )
            output_count = max(1, int(len(filtered) / factor))
            source_positions = np.arange(output_count, dtype=np.float64) * factor
            source_axis = np.arange(len(filtered), dtype=np.float64)
            shifted = np.empty((output_count, channels), dtype=np.float64)
            for channel in range(channels):
                shifted[:, channel] = np.interp(
                    source_positions,
                    source_axis,
                    filtered[:, channel],
                )
            samples = shifted

        samples -= np.mean(samples, axis=0, keepdims=True)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        target_peak = 0.94 * 32767.0
        if peak > target_peak:
            samples *= target_peak / peak
        fade_frames = min(len(samples) // 2, max(1, int(sample_rate * 0.006)))
        if fade_frames > 1:
            fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float64)[:, None]
            samples[:fade_frames] *= fade
            samples[-fade_frames:] *= fade[::-1]
        rendered = np.clip(np.rint(samples), -32768, 32767).astype("<i2")

        with wave.open(str(wav_path), "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(sample_width)
            target.setframerate(sample_rate)
            target.writeframes(rendered.tobytes())

    @staticmethod
    def _wav_metadata(wav_path: Path) -> tuple[int, int, int]:
        with wave.open(str(wav_path), "rb") as source:
            return (
                source.getframerate(),
                source.getnchannels(),
                source.getnframes(),
            )


def build_tts(config: Config) -> TextToSpeech:
    if config.tts_provider.lower() == "piper":
        return PiperTextToSpeech(
            model_path=config.piper_model_path,
            config_path=config.piper_config_path,
            output_path=config.piper_output_path,
            length_scale=config.piper_length_scale,
            sentence_silence=config.piper_sentence_silence,
            pitch_factor=config.piper_pitch_factor,
            avatar_state_path=config.avatar_state_path,
            continuous_audio=config.realtime_audio_stabilization_enabled,
        )
    if config.tts_provider.lower() == "windows":
        script_path = Path(__file__).resolve().parent / "scripts" / "windows_speak.ps1"
        return WindowsTextToSpeech(script_path, config.voice_dir, config.tts_rate)
    if config.tts_provider.lower() == "edge" and external_access_decision(
        config,
        "external_tts",
    ).allowed:
        return EdgeTextToSpeech(
            voice=config.edge_tts_voice,
            output_dir=config.voice_dir,
            rate=config.edge_tts_rate,
            pitch=config.edge_tts_pitch,
            avatar_state_path=config.avatar_state_path,
            config=config,
        )
    if config.tts_provider.lower() == "openai" and external_access_decision(
        config,
        "external_tts",
    ).allowed:
        return OpenAITextToSpeech(
            config.tts_model,
            config.tts_voice,
            config.voice_dir,
            config,
        )
    return ConsoleSpeech()


def _percentage_value(value: str) -> int:
    cleaned = value.strip().replace("%", "")
    try:
        return int(cleaned)
    except ValueError:
        return 0
