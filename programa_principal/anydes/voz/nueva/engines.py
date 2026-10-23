from __future__ import annotations

import base64
import json
import hashlib
import os
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Protocol

from anydes.voz.nueva.contracts import (
    AudioBuffer,
    EngineError,
    SynthesisResult,
    VoiceSynthesisRequest,
)
from anydes.voz.process import hidden_process_kwargs


class TTSEngine(Protocol):
    engine_id: str
    voice_id: str

    def synthesize(self, request: VoiceSynthesisRequest) -> SynthesisResult:
        """Synthesize audio without playing it."""

    def cancel(self) -> None:
        """Cancel current synthesis without selecting another voice."""

    def close(self) -> None:
        """Release local model resources."""


class PiperProvider:
    engine_id = "piper"

    def __init__(self, model_path: Path, config_path: Path, *, voice_id: str) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.voice_id = voice_id
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def synthesize(self, request: VoiceSynthesisRequest) -> SynthesisResult:
        started = time.perf_counter()
        self._cancelled.clear()
        if not self.model_path.is_file() or not self.config_path.is_file():
            return self._failure(request, "piper_model_unavailable", "preflight", started)
        executable = self._executable()
        if not executable:
            return self._failure(request, "piper_executable_unavailable", "preflight", started)
        with tempfile.TemporaryDirectory(prefix="anydes-voice-piper-") as tmp:
            output = Path(tmp) / "speech.wav"
            length_scale = max(0.72, min(1.48, 1.0 / max(0.68, request.direction.speed)))
            command = [
                executable,
                "--model",
                str(self.model_path),
                "--config",
                str(self.config_path),
                "--output_file",
                str(output),
                "--length-scale",
                str(length_scale),
                "--sentence-silence",
                "0.16",
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    **hidden_process_kwargs(),
                )
                with self._lock:
                    self._process = process
                _, stderr = process.communicate(input=request.spoken_text, timeout=120)
                if self._cancelled.is_set():
                    return SynthesisResult(
                        status="cancelled",
                        engine_id=self.engine_id,
                        voice_id=self.voice_id,
                        identity_id=request.identity_id,
                        synthesis_ms=(time.perf_counter() - started) * 1000.0,
                    )
                if process.returncode != 0 or not output.is_file():
                    return self._failure(
                        request,
                        "piper_synthesis_failed",
                        "synthesis",
                        started,
                        message=stderr[-240:],
                    )
                audio = _read_pcm_wav(output)
                elapsed = (time.perf_counter() - started) * 1000.0
                return SynthesisResult(
                    status="ready",
                    engine_id=self.engine_id,
                    voice_id=self.voice_id,
                    identity_id=request.identity_id,
                    audio=audio,
                    synthesis_ms=elapsed,
                    first_audio_ms=elapsed,
                    metadata={"mode": "whole_message", "device": "cpu"},
                )
            except subprocess.TimeoutExpired:
                self.cancel()
                return self._failure(request, "piper_timeout", "synthesis", started)
            except Exception as exc:
                return self._failure(
                    request,
                    f"piper_{type(exc).__name__}",
                    "synthesis",
                    started,
                )
            finally:
                with self._lock:
                    self._process = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self) -> None:
        self.cancel()

    def warmup(self) -> bool:
        return bool(
            self.model_path.is_file()
            and self.config_path.is_file()
            and self._executable()
        )

    def _failure(
        self,
        request: VoiceSynthesisRequest,
        code: str,
        stage: str,
        started: float,
        *,
        message: str = "",
    ) -> SynthesisResult:
        return SynthesisResult(
            status="failed",
            engine_id=self.engine_id,
            voice_id=self.voice_id,
            identity_id=request.identity_id,
            synthesis_ms=(time.perf_counter() - started) * 1000.0,
            error=EngineError(code, stage, self.engine_id, message, recoverable=True),
        )

    @staticmethod
    def _executable() -> str:
        found = shutil.which("piper")
        if found:
            return found
        candidates = [
            Path(sys.executable).parent / "piper.exe",
            Path(sys.executable).parent / "Scripts" / "piper.exe",
            Path(site.getusersitepackages()).parent / "Scripts" / "piper.exe",
        ]
        return next((str(item) for item in candidates if item.is_file()), "")


class WorkerTTSEngine:
    """Persistent JSON-lines client for heavy XTTS/Kokoro runtimes."""

    def __init__(
        self,
        *,
        engine_id: str,
        voice_id: str,
        worker_python: Path,
        worker_script: Path,
        device: str,
        model: str = "",
        runtime_options: dict[str, object] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.voice_id = voice_id
        self.worker_python = Path(worker_python)
        self.worker_script = Path(worker_script)
        self.device = device
        self.model = model
        self.runtime_options = dict(runtime_options or {})
        self._process: subprocess.Popen[str] | None = None
        self._io_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._prepare_metadata: dict[str, str | float | int | bool] = {}

    def synthesize(self, request: VoiceSynthesisRequest) -> SynthesisResult:
        started = time.perf_counter()
        with self._io_lock:
            process, error = self._ensure_process()
            if error is not None or process is None:
                return SynthesisResult(
                    status="failed",
                    engine_id=self.engine_id,
                    voice_id=self.voice_id,
                    identity_id=request.identity_id,
                    synthesis_ms=(time.perf_counter() - started) * 1000.0,
                    error=error,
                )
            with tempfile.TemporaryDirectory(prefix=f"anydes-voice-{self.engine_id}-") as tmp:
                output = Path(tmp) / "speech.wav"
                payload = {
                    "id": uuid.uuid4().hex,
                    "text": request.spoken_text,
                    "output": str(output),
                    "reference": request.reference_audio_path,
                    "language": request.language,
                    "voice": self.voice_id,
                    "speed": request.direction.speed,
                }
                payload.update(self.runtime_options)
                try:
                    assert process.stdin is not None and process.stdout is not None
                    process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    process.stdin.flush()
                    response_line = process.stdout.readline()
                    if not response_line:
                        raise RuntimeError("voice_worker_closed")
                    response = json.loads(response_line)
                    if not response.get("ok") or not output.is_file():
                        return SynthesisResult(
                            status="failed",
                            engine_id=self.engine_id,
                            voice_id=self.voice_id,
                            identity_id=request.identity_id,
                            synthesis_ms=(time.perf_counter() - started) * 1000.0,
                            error=EngineError(
                                str(response.get("error_code") or "worker_synthesis_failed"),
                                "synthesis",
                                self.engine_id,
                                recoverable=True,
                            ),
                            metadata=_public_worker_metadata(response),
                        )
                    audio = _read_pcm_wav(output)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    return SynthesisResult(
                        status="ready",
                        engine_id=self.engine_id,
                        voice_id=self.voice_id,
                        identity_id=request.identity_id,
                        audio=audio,
                        synthesis_ms=elapsed,
                        first_audio_ms=elapsed,
                        metadata={**self._prepare_metadata, **_public_worker_metadata(response)},
                    )
                except Exception as exc:
                    self._terminate()
                    return SynthesisResult(
                        status="failed",
                        engine_id=self.engine_id,
                        voice_id=self.voice_id,
                        identity_id=request.identity_id,
                        synthesis_ms=(time.perf_counter() - started) * 1000.0,
                        error=EngineError(
                            f"worker_{type(exc).__name__}",
                            "synthesis",
                            self.engine_id,
                            recoverable=True,
                        ),
                    )

    def cancel(self) -> None:
        self._terminate()

    def close(self) -> None:
        self.cancel()

    def warmup(self) -> bool:
        with self._io_lock:
            process, error = self._ensure_process()
            return process is not None and error is None

    def release_temporary_gpu_cache(self) -> dict[str, object]:
        if not self._io_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            with self._process_lock:
                process = self._process
            if process is None or process.poll() is not None:
                return {"status": "not_running"}
            payload = {"id": uuid.uuid4().hex, "command": "release_temporary_cache"}
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
            if not response_line:
                return {"status": "worker_closed"}
            response = json.loads(response_line)
            return {
                key: value
                for key, value in response.items()
                if key not in {"id", "ok"}
            }
        except Exception as exc:
            return {"status": "failed", "error_type": type(exc).__name__}
        finally:
            self._io_lock.release()

    def profile_gpu_memory(self) -> dict[str, object]:
        with self._io_lock:
            process, error = self._ensure_process()
            if error is not None or process is None:
                return {
                    "status": "failed",
                    "error_code": error.code if error is not None else "worker_unavailable",
                }
            payload = {"id": uuid.uuid4().hex, "command": "profile_gpu_memory"}
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                process.stdin.flush()
                response_line = process.stdout.readline()
                if not response_line:
                    return {"status": "worker_closed"}
                response = json.loads(response_line)
                return {key: value for key, value in response.items() if key not in {"id", "ok"}}
            except Exception as exc:
                return {"status": "failed", "error_type": type(exc).__name__}

    def gpu_memory_snapshot(self) -> dict[str, object]:
        with self._io_lock:
            process, error = self._ensure_process()
            if error is not None or process is None:
                return {
                    "status": "failed",
                    "error_code": error.code if error is not None else "worker_unavailable",
                }
            payload = {"id": uuid.uuid4().hex, "command": "gpu_memory_snapshot"}
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                process.stdin.flush()
                response_line = process.stdout.readline()
                if not response_line:
                    return {"status": "worker_closed"}
                response = json.loads(response_line)
                return {key: value for key, value in response.items() if key not in {"id", "ok"}}
            except Exception as exc:
                return {"status": "failed", "error_type": type(exc).__name__}

    def _ensure_process(self) -> tuple[subprocess.Popen[str] | None, EngineError | None]:
        if not self.worker_python.is_file():
            return None, EngineError(
                "voice_worker_python_unavailable",
                "preflight",
                self.engine_id,
                recoverable=True,
            )
        if not self.worker_script.is_file():
            return None, EngineError("voice_worker_script_unavailable", "preflight", self.engine_id)
        command = [
            str(self.worker_python),
            str(self.worker_script),
            "--engine",
            self.engine_id,
            "--device",
            self.device,
            "--voice",
            self.voice_id,
        ]
        if self.model:
            command.extend(("--model", self.model))
        with self._process_lock:
            current = self._process
            if current is not None and current.poll() is None:
                return current, None
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env={**os.environ, "COQUI_TOS_AGREED": "1"},
                    **hidden_process_kwargs(),
                )
            except Exception as exc:
                return None, EngineError(
                    f"voice_worker_{type(exc).__name__}",
                    "load",
                    self.engine_id,
                    recoverable=True,
                )
            # Publish the worker before its blocking handshake so cancel/close
            # can terminate an in-progress warmup without waiting for CUDA load.
            self._process = process
        try:
            assert process.stdout is not None
            ready_line = process.stdout.readline()
            ready = json.loads(ready_line) if ready_line else {}
            if not ready.get("ready"):
                self._discard_process(process)
                return None, EngineError(
                    str(ready.get("error_code") or "voice_worker_not_ready"),
                    "load",
                    self.engine_id,
                    recoverable=True,
                )
            if self.engine_id in {"qwen3-tts-base", "qwen3-tts-faster"}:
                prepare_error = self._prepare_voice(process)
                if prepare_error is not None:
                    self._discard_process(process)
                    return None, prepare_error
            return process, None
        except Exception as exc:
            self._discard_process(process)
            return None, EngineError(
                f"voice_worker_{type(exc).__name__}",
                "load",
                self.engine_id,
                recoverable=True,
            )

    def _discard_process(self, process: subprocess.Popen[str]) -> None:
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._prepare_metadata = {}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()

    def _prepare_voice(self, process: subprocess.Popen[str]) -> EngineError | None:
        payload = {"id": uuid.uuid4().hex, "command": "prepare_voice"}
        payload.update(self.runtime_options)
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
            if not response_line:
                raise RuntimeError("voice_worker_closed_during_prepare")
            response = json.loads(response_line)
            if not response.get("ok"):
                return EngineError(
                    str(response.get("error_code") or "voice_conditioning_prepare_failed"),
                    "conditioning",
                    self.engine_id,
                    message=str(response.get("error_message") or ""),
                    recoverable=True,
                )
            prepared = _public_worker_metadata(response)
            self._prepare_metadata = {
                **prepared,
                "conditioning_built_at_startup": not bool(
                    prepared.get("conditioning_cached", False)
                ),
                "conditioning_cached": True,
            }
            return None
        except Exception as exc:
            return EngineError(
                f"voice_conditioning_{type(exc).__name__}",
                "conditioning",
                self.engine_id,
                recoverable=True,
            )

    def _terminate(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
            self._prepare_metadata = {}
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


class XTTSProvider(WorkerTTSEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(engine_id="xtts", voice_id="anydes-reference", **kwargs)


class KokoroProvider(WorkerTTSEngine):
    def __init__(self, *, voice_id: str, **kwargs) -> None:
        super().__init__(engine_id="kokoro", voice_id=voice_id, **kwargs)


class Qwen3TTSBaseProvider(WorkerTTSEngine):
    def __init__(
        self,
        *,
        model: Path,
        manifest_path: Path,
        reference_audio: Path,
        **kwargs,
    ) -> None:
        runtime_options = _canonical_qwen_runtime_options(manifest_path, reference_audio)
        super().__init__(
            engine_id="qwen3-tts-base",
            voice_id="anydes-canonical-voice-v1",
            model=str(Path(model).resolve()),
            runtime_options=runtime_options,
            **kwargs,
        )


class FasterQwen3TTSProvider(WorkerTTSEngine):
    """Canonical B4 provider accelerated with CUDA Graphs."""

    def __init__(
        self,
        *,
        model: Path,
        manifest_path: Path,
        reference_audio: Path,
        **kwargs,
    ) -> None:
        runtime_options = _canonical_qwen_runtime_options(manifest_path, reference_audio)
        self._cancel_path_lock = threading.Lock()
        self._active_cancel_path: Path | None = None
        self.protocol_discarded = 0
        self.cancel_drains = 0
        super().__init__(
            engine_id="qwen3-tts-faster",
            voice_id="anydes-canonical-voice-v1",
            model=str(Path(model).resolve()),
            runtime_options=runtime_options,
            **kwargs,
        )

    def synthesize_stream(self, request: VoiceSynthesisRequest):
        started = time.perf_counter()
        with self._io_lock:
            process, error = self._ensure_process()
            if error is not None or process is None:
                yield SynthesisResult(
                    status="failed",
                    engine_id=self.engine_id,
                    voice_id=self.voice_id,
                    identity_id=request.identity_id,
                    synthesis_ms=(time.perf_counter() - started) * 1000.0,
                    error=error,
                )
                return
            payload = {
                "id": uuid.uuid4().hex,
                "stream": True,
                "text": request.spoken_text,
                "reference": request.reference_audio_path,
                "language": request.language,
                "voice": self.voice_id,
                "speed": request.direction.speed,
            }
            cancel_path = Path(tempfile.gettempdir()) / f"anydes-fast-tts-{uuid.uuid4().hex}.cancel"
            cancel_path.unlink(missing_ok=True)
            with self._cancel_path_lock:
                self._active_cancel_path = cancel_path
            payload["cancel_path"] = str(cancel_path)
            payload.update(self.runtime_options)
            terminal_received = False
            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                process.stdin.flush()
                while True:
                    response_line = process.stdout.readline()
                    if not response_line:
                        raise RuntimeError("voice_worker_closed")
                    response = json.loads(response_line)
                    if response.get("id") != payload["id"]:
                        self.protocol_discarded += 1
                        continue
                    event = str(response.get("event") or "")
                    if event == "chunk":
                        pcm = base64.b64decode(str(response.get("pcm_s16le") or ""))
                        frame_count = int(response.get("frame_count") or 0)
                        if not pcm or frame_count <= 0:
                            continue
                        elapsed = (time.perf_counter() - started) * 1000.0
                        yield SynthesisResult(
                            status="ready",
                            engine_id=self.engine_id,
                            voice_id=self.voice_id,
                            identity_id=request.identity_id,
                            audio=AudioBuffer(
                                pcm_s16le=pcm,
                                sample_rate=int(response.get("sample_rate") or 24000),
                                channels=1,
                                frame_count=frame_count,
                            ),
                            synthesis_ms=elapsed,
                            first_audio_ms=elapsed,
                            metadata={**self._prepare_metadata, **_public_worker_metadata(response), "request_id": payload["id"]},
                        )
                        continue
                    if event == "done" and response.get("ok"):
                        terminal_received = True
                        return
                    if not response.get("ok"):
                        terminal_received = True
                        yield SynthesisResult(
                            status="failed",
                            engine_id=self.engine_id,
                            voice_id=self.voice_id,
                            identity_id=request.identity_id,
                            synthesis_ms=(time.perf_counter() - started) * 1000.0,
                            error=EngineError(
                                str(response.get("error_code") or "worker_stream_failed"),
                                "synthesis",
                                self.engine_id,
                                message=str(response.get("error_message") or ""),
                                recoverable=True,
                            ),
                        )
                        return
            except Exception as exc:
                self._terminate()
                yield SynthesisResult(
                    status="failed",
                    engine_id=self.engine_id,
                    voice_id=self.voice_id,
                    identity_id=request.identity_id,
                    synthesis_ms=(time.perf_counter() - started) * 1000.0,
                    error=EngineError(
                        f"worker_stream_{type(exc).__name__}",
                        "synthesis",
                        self.engine_id,
                        recoverable=True,
                    ),
                )
            finally:
                # A generator may be closed at any yield by its cancelled consumer.
                # Keep the protocol lease AND cancellation file until this request
                # has acknowledged completion. Never let the next request read it.
                if not terminal_received and process.poll() is None:
                    self.cancel_drains += 1
                    drain_finished = threading.Event()
                    drain_guard = threading.Lock()
                    def expire():
                        with drain_guard:
                            if not drain_finished.is_set():
                                self._terminate()
                    watchdog = threading.Timer(5.0, expire)
                    watchdog.daemon = True
                    watchdog.start()
                    try:
                        cancel_path.touch(exist_ok=True)
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            pending = json.loads(line)
                            if pending.get("id") == payload["id"] and (pending.get("event") == "done" or not pending.get("ok")):
                                break
                            self.protocol_discarded += 1
                    except (OSError, ValueError):
                        self._terminate()
                    finally:
                        with drain_guard:
                            drain_finished.set()
                        watchdog.cancel()
                with self._cancel_path_lock:
                    if self._active_cancel_path == cancel_path:
                        self._active_cancel_path = None
                cancel_path.unlink(missing_ok=True)

    def cancel(self) -> None:
        with self._cancel_path_lock:
            cancel_path = self._active_cancel_path
        if cancel_path is not None:
            try:
                cancel_path.touch(exist_ok=True)
            except OSError:
                self._terminate()

    def close(self) -> None:
        self.cancel()
        self._terminate()


def _canonical_qwen_runtime_options(
    manifest_path: Path,
    reference_audio: Path,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    reference_audio = Path(reference_audio)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("voice_identity_status") != "canonical":
        raise ValueError("qwen3_voice_manifest_is_not_canonical")
    if payload.get("identity_id") != "anydes-canonical-voice-v1":
        raise ValueError("qwen3_voice_identity_mismatch")
    reference = payload.get("reference")
    generation = payload.get("generation")
    if not isinstance(reference, dict) or not isinstance(generation, dict):
        raise ValueError("qwen3_voice_manifest_invalid")
    expected_audio = (manifest_path.parent / str(reference.get("audio") or "")).resolve()
    if expected_audio != reference_audio.resolve() or not reference_audio.is_file():
        raise ValueError("qwen3_voice_reference_path_mismatch")
    expected_hash = str(reference.get("sha256") or "").upper()
    if not expected_hash or _sha256(reference_audio) != expected_hash:
        raise ValueError("qwen3_voice_reference_hash_mismatch")
    parameters = generation.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("qwen3_voice_generation_parameters_missing")
    return {
        "reference": str(reference_audio.resolve()),
        "reference_text": str(reference.get("text") or ""),
        "seed": int(generation.get("seed") or 0),
        "generation_parameters": parameters,
    }


class EmergencyFallbackProvider(PiperProvider):
    engine_id = "emergency"


def _read_pcm_wav(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        payload = source.readframes(frame_count)
    if sample_width != 2:
        raise ValueError("voice_engine_must_return_pcm16_wav")
    return AudioBuffer(payload, sample_rate, channels, frame_count)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _public_worker_metadata(response: dict[str, object]) -> dict[str, str | float | int | bool]:
    allowed = {
        "load_ms",
        "generation_ms",
        "device",
        "vram_peak_mb",
        "ram_peak_mb",
        "rtf",
        "prepare_ms",
        "conditioning_cached",
        "conditioning_fingerprint",
        "conditioning_built_at_startup",
        "conditioning_encoders_released",
        "memory_allocated_mb",
        "memory_reserved_mb",
        "memory_unused_reserved_mb",
        "cuda_reserved_delta_mb",
    }
    return {
        key: value
        for key, value in response.items()
        if key in allowed and isinstance(value, (str, float, int, bool))
    }
