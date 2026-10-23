from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from anydes.voz.nueva.contracts import (
    AudioBuffer,
    EngineError,
    PlaybackResult,
    VisemeCue,
    VoiceEvent,
)
from anydes.voz.nueva.dac_output import DacOutputStream
from anydes.voz.nueva.lip_sync import prepare_pcm_timeline


EventSink = Callable[[VoiceEvent], None]


@dataclass(frozen=True)
class QueuedAudioBlock:
    index: int
    audio: AudioBuffer
    timeline: tuple[VisemeCue, ...]
    text: str = ""
    boundary: str = "none"


@dataclass
class AudioPipeline:
    sample_rate: int = 24000
    channels: int = 1
    block_frames: int = 960
    _cancelled: threading.Event = field(default_factory=threading.Event, init=False)
    _play_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stream_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stream: object | None = field(default=None, init=False)
    _playback_serial: int = field(default=0, init=False)

    def normalize(self, source: AudioBuffer, *, fade_edges: bool = True) -> AudioBuffer:
        if not source.pcm_s16le or source.frame_count <= 0:
            raise ValueError("empty_audio_buffer")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy_required_for_audio_pipeline") from exc
        samples = np.frombuffer(source.pcm_s16le, dtype="<i2").astype(np.float64)
        if source.channels > 1:
            samples = samples.reshape(-1, source.channels).mean(axis=1)
        if source.sample_rate != self.sample_rate and len(samples) > 1:
            target_count = max(1, round(len(samples) * self.sample_rate / source.sample_rate))
            source_axis = np.linspace(0.0, 1.0, len(samples), endpoint=True)
            target_axis = np.linspace(0.0, 1.0, target_count, endpoint=True)
            samples = np.interp(target_axis, source_axis, samples)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples *= min(1.0, (0.94 * 32767.0) / peak)
        fade_frames = min(len(samples) // 2, max(1, int(self.sample_rate * 0.004)))
        if fade_edges and fade_frames > 1:
            fade = np.linspace(0.0, 1.0, fade_frames)
            samples[:fade_frames] *= fade
            samples[-fade_frames:] *= fade[::-1]
        rendered = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
        return AudioBuffer(
            pcm_s16le=rendered.tobytes(),
            sample_rate=self.sample_rate,
            channels=1,
            frame_count=len(rendered),
        )

    def play(
        self,
        source: AudioBuffer,
        timeline: tuple[VisemeCue, ...],
        *,
        event_sink: EventSink | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PlaybackResult:
        audio = self.normalize(source)
        with self._play_lock:
            self._cancelled = cancel_event if cancel_event is not None else threading.Event()
            if self._cancelled.is_set():
                return PlaybackResult("interrupted", 0, audio.frame_count, 0.0, 0.0)
            return self._play_locked(audio, timeline, event_sink)

    def play_sequence(
        self,
        blocks: Iterable[QueuedAudioBlock],
        *,
        event_sink: EventSink | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PlaybackResult:
        with self._play_lock:
            self._cancelled = cancel_event if cancel_event is not None else threading.Event()
            if self._cancelled.is_set():
                return PlaybackResult("interrupted", 0, 0, 0.0, 0.0)
            return self._play_sequence_locked(iter(blocks), event_sink)

    def cancel(self) -> None:
        self._cancelled.set()
        with self._stream_lock:
            stream = self._stream
        if stream is not None:
            abort = getattr(stream, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass

    def _play_locked(
        self,
        audio: AudioBuffer,
        timeline: tuple[VisemeCue, ...],
        event_sink: EventSink | None,
    ) -> PlaybackResult:
        playback_id = self._next_playback_id()
        try:
            import sounddevice as sd
        except ImportError:
            return PlaybackResult(
                status="failed",
                played_frames=0,
                total_frames=audio.frame_count,
                playback_ms=0.0,
                spoken_fraction=0.0,
                error=EngineError("sounddevice_unavailable", "playback", "audio_pipeline"),
            )
        started = time.perf_counter()
        played = 0
        stream = None
        try:
            stream = DacOutputStream(sd,
                cancel_event=self._cancelled,
                samplerate=audio.sample_rate,
                channels=audio.channels,
                blocksize=self.block_frames,
                event_sink=lambda payload: self._emit(event_sink, "playback_started", payload),
            )
            with self._stream_lock:
                self._stream = stream
            stream.start()
            visual_timeline, active_start, active_end = prepare_pcm_timeline(audio, timeline)
            stream.set_segment({
                    "duration_ms": audio.duration_seconds * 1000.0,
                    "sample_rate": audio.sample_rate,
                    "frame_count": audio.frame_count,
                    "playback_id": playback_id,
                    "timeline": visual_timeline,
                    "timeline_source": "measured_pcm",
                    "active_start_ms": active_start,
                    "active_end_ms": active_end,
            })
            frame_bytes = audio.channels * 2
            while played < audio.frame_count and not self._cancelled.is_set():
                count = min(self.block_frames, audio.frame_count - played)
                begin = played * frame_bytes
                end = begin + count * frame_bytes
                stream.write(audio.pcm_s16le[begin:end])
                played += count
            status = "interrupted" if self._cancelled.is_set() else "completed"
            self._finish_stream_playback(stream, interrupted=status == "interrupted")
            diagnostics = stream.diagnostics()
            played = stream.played_frames
            if self._cancelled.is_set():
                status = "interrupted"
            stream = None
            self._emit(
                event_sink,
                "playback_interrupted" if status == "interrupted" else "playback_stopped",
                {
                    "playback_id": playback_id,
                    "played_frames": played,
                    "total_frames": audio.frame_count,
                    **diagnostics,
                },
            )
            return PlaybackResult(
                status=status,
                played_frames=played,
                total_frames=audio.frame_count,
                playback_ms=(time.perf_counter() - started) * 1000.0,
                spoken_fraction=played / max(1, audio.frame_count),
                metadata=diagnostics,
            )
        except Exception as exc:
            self._emit(
                event_sink,
                "voice_failed",
                {"playback_id": playback_id, "error_code": type(exc).__name__},
            )
            return PlaybackResult(
                status="failed",
                played_frames=played,
                total_frames=audio.frame_count,
                playback_ms=(time.perf_counter() - started) * 1000.0,
                spoken_fraction=played / max(1, audio.frame_count),
                error=EngineError(
                    f"playback_{type(exc).__name__}",
                    "playback",
                    "audio_pipeline",
                    recoverable=True,
                ),
            )
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            with self._stream_lock:
                self._stream = None

    def _play_sequence_locked(
        self,
        blocks,
        event_sink: EventSink | None,
    ) -> PlaybackResult:
        playback_id = self._next_playback_id()
        try:
            import sounddevice as sd
        except ImportError:
            return PlaybackResult(
                status="failed",
                played_frames=0,
                total_frames=0,
                playback_ms=0.0,
                spoken_fraction=0.0,
                error=EngineError("sounddevice_unavailable", "playback", "audio_pipeline"),
            )

        started = time.perf_counter()
        played = 0
        total_frames = 0
        segment_count = 0
        underruns = 0
        max_gap_ms = 0.0
        played_segment_indices: list[int] = []
        stream = None
        try:
            wait_started = time.perf_counter()
            first = next(blocks, None)
            initial_wait_ms = (time.perf_counter() - wait_started) * 1000.0
            if first is None:
                if self._cancelled.is_set():
                    return PlaybackResult("interrupted", 0, 0, 0.0, 0.0)
                return PlaybackResult(
                    status="failed",
                    played_frames=0,
                    total_frames=0,
                    playback_ms=(time.perf_counter() - started) * 1000.0,
                    spoken_fraction=0.0,
                    error=EngineError("empty_audio_sequence", "playback", "audio_pipeline"),
                )
            stream = DacOutputStream(sd,
                cancel_event=self._cancelled,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_frames,
                event_sink=lambda payload: self._emit(event_sink, "playback_started", payload),
            )
            with self._stream_lock:
                self._stream = stream
            stream.start()
            current = first
            while current is not None and not self._cancelled.is_set():
                audio, trimmed_frames = self._trim_sequence_audio(
                    self.normalize(current.audio, fade_edges=current.boundary != "native"),
                    boundary=current.boundary,
                    first=current.index == 0,
                )
                total_frames += audio.frame_count
                segment_count += 1
                played_segment_indices.append(current.index)
                trim_ms = trimmed_frames / audio.sample_rate * 1000.0
                shifted = tuple(VisemeCue(max(0, c.start_ms - trim_ms), max(0, c.end_ms - trim_ms), c.viseme, c.weight) for c in current.timeline if c.end_ms > trim_ms)
                visual_timeline, active_start, active_end = prepare_pcm_timeline(audio, shifted)
                stream.set_segment({
                        "duration_ms": audio.duration_seconds * 1000.0,
                        "sample_rate": audio.sample_rate,
                        "frame_count": audio.frame_count,
                        "playback_id": playback_id,
                        "timeline": visual_timeline,
                        "timeline_source": "measured_pcm",
                        "active_start_ms": active_start,
                        "active_end_ms": active_end,
                        "segment_index": current.index,
                })
                frame_bytes = audio.channels * 2
                segment_played = 0
                while segment_played < audio.frame_count and not self._cancelled.is_set():
                    count = min(self.block_frames, audio.frame_count - segment_played)
                    begin = segment_played * frame_bytes
                    end = begin + count * frame_bytes
                    stream.write(audio.pcm_s16le[begin:end])
                    segment_played += count
                    played += count
                if self._cancelled.is_set():
                    break
                wait_started = time.perf_counter()
                current = next(blocks, None)
                gap_ms = (time.perf_counter() - wait_started) * 1000.0
                if current is not None and gap_ms > 20.0:
                    underruns += 1
                    max_gap_ms = max(max_gap_ms, gap_ms)
                    self._emit(
                        event_sink,
                        "queue_underrun",
                        {
                            "gap_ms": gap_ms,
                            "before_segment": current.index,
                            "text": current.text,
                            "chunk_bytes": len(current.audio.pcm_s16le),
                            "frame_count": current.audio.frame_count,
                            "duration_ms": current.audio.duration_seconds * 1000.0,
                        },
                    )
            status = "interrupted" if self._cancelled.is_set() else "completed"
            self._finish_stream_playback(stream, interrupted=status == "interrupted")
            diagnostics = stream.diagnostics()
            played = stream.played_frames
            if self._cancelled.is_set():
                status = "interrupted"
            stream = None
            self._emit(
                event_sink,
                "playback_interrupted" if status == "interrupted" else "playback_stopped",
                {
                    "playback_id": playback_id,
                    "played_frames": played,
                    "total_frames": total_frames,
                    "segment_count": segment_count,
                    "max_gap_ms": max_gap_ms,
                    **diagnostics,
                },
            )
            return PlaybackResult(
                status=status,
                played_frames=played,
                total_frames=total_frames,
                playback_ms=(time.perf_counter() - started) * 1000.0,
                spoken_fraction=played / max(1, total_frames),
                metadata={
                    **diagnostics,
                    "segment_count": segment_count,
                    "queue_underruns": underruns,
                    "max_gap_ms": round(max_gap_ms, 3),
                    "initial_queue_wait_ms": round(initial_wait_ms, 3),
                    "played_segment_indices": played_segment_indices,
                },
            )
        except Exception as exc:
            self._emit(
                event_sink,
                "voice_failed",
                {"playback_id": playback_id, "error_code": type(exc).__name__},
            )
            return PlaybackResult(
                status="failed",
                played_frames=played,
                total_frames=total_frames,
                playback_ms=(time.perf_counter() - started) * 1000.0,
                spoken_fraction=played / max(1, total_frames),
                error=EngineError(
                    f"playback_{type(exc).__name__}",
                    "playback",
                    "audio_pipeline",
                    recoverable=True,
                ),
                metadata={
                    "segment_count": segment_count,
                    "queue_underruns": underruns,
                    "max_gap_ms": round(max_gap_ms, 3),
                    "played_segment_indices": played_segment_indices,
                },
            )
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            with self._stream_lock:
                self._stream = None

    @staticmethod
    def _output_latency_seconds(stream: object) -> float:
        """Return the device's reported output latency without guessing from text."""

        value = getattr(stream, "latency", 0.0)
        if isinstance(value, (tuple, list)):
            value = value[-1] if value else 0.0
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def _next_playback_id(self) -> int:
        self._playback_serial += 1
        return self._playback_serial

    @staticmethod
    def _finish_stream_playback(stream: object, *, interrupted: bool) -> None:
        """Drain normal playback; abort only when the user explicitly interrupts."""

        method_name = "abort" if interrupted else "stop"
        method = getattr(stream, method_name, None)
        if callable(method):
            method()
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _prepare_sequence_audio(
        self,
        audio: AudioBuffer,
        *,
        boundary: str,
        first: bool,
    ) -> AudioBuffer:
        return self._trim_sequence_audio(audio, boundary=boundary, first=first)[0]

    def _trim_sequence_audio(
        self, audio: AudioBuffer, *, boundary: str, first: bool,
    ) -> tuple[AudioBuffer, int]:
        """Trim only excessive edge silence while preserving linguistic pauses."""

        if boundary == "native":
            return audio, 0

        try:
            import numpy as np
        except ImportError:
            return audio, 0
        samples = np.frombuffer(audio.pcm_s16le, dtype="<i2")
        if samples.size < 2:
            return audio, 0
        peak = float(np.max(np.abs(samples)))
        if peak <= 0:
            return audio, 0
        threshold = max(180.0, peak * 0.018)
        active = np.flatnonzero(np.abs(samples) >= threshold)
        if not active.size:
            return audio, 0
        leading_keep_ms = 35 if first else 18
        trailing_keep_ms = {"major": 120, "minor": 70, "none": 35}.get(boundary, 70)
        leading_keep = round(audio.sample_rate * leading_keep_ms / 1000.0)
        trailing_keep = round(audio.sample_rate * trailing_keep_ms / 1000.0)
        begin = max(0, int(active[0]) - leading_keep)
        end = min(len(samples), int(active[-1]) + trailing_keep + 1)
        rendered = samples[begin:end].copy()
        fade_frames = min(len(rendered) // 2, max(1, int(audio.sample_rate * 0.003)))
        if fade_frames > 1:
            fade = np.linspace(0.0, 1.0, fade_frames)
            rendered[:fade_frames] = np.rint(rendered[:fade_frames] * fade).astype("<i2")
            rendered[-fade_frames:] = np.rint(rendered[-fade_frames:] * fade[::-1]).astype("<i2")
        return AudioBuffer(
            pcm_s16le=rendered.astype("<i2", copy=False).tobytes(),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            frame_count=len(rendered),
        ), begin

    @staticmethod
    def _emit(event_sink: EventSink | None, kind: str, payload: dict[str, object]) -> None:
        if event_sink is not None:
            event_sink(VoiceEvent(kind=kind, created_monotonic=time.perf_counter(), payload=payload))
