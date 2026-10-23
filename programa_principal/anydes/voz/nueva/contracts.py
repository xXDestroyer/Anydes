from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EngineKind = Literal[
    "qwen3-tts-base",
    "qwen3-tts-faster",
    "xtts",
    "kokoro",
    "piper",
    "emergency",
]


@dataclass(frozen=True)
class EngineError:
    code: str
    stage: str
    engine_id: str
    message: str = ""
    recoverable: bool = False


@dataclass(frozen=True)
class VoiceDirection:
    text: str
    conversation_intent: str
    authorized_emotion: str
    speed: float
    pitch: float
    energy: float
    warmth: float
    expressiveness: float
    volume: float
    pause_before_ms: int
    pause_after_ms: int
    emphasis: tuple[str, ...] = ()
    direction_id: str = ""


@dataclass(frozen=True)
class VoiceSynthesisRequest:
    spoken_text: str
    direction: VoiceDirection
    identity_id: str
    language: str
    locale: str
    reference_audio_path: str = ""


@dataclass(frozen=True)
class AudioBuffer:
    pcm_s16le: bytes
    sample_rate: int
    channels: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / float(self.sample_rate)


@dataclass(frozen=True)
class VisemeCue:
    start_ms: float
    end_ms: float
    viseme: str
    weight: float = 0.72

    def to_dict(self) -> dict[str, float | str]:
        return {
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "viseme": self.viseme,
            "weight": round(self.weight, 3),
        }


@dataclass(frozen=True)
class SynthesisResult:
    status: Literal["ready", "failed", "cancelled"]
    engine_id: str
    voice_id: str
    identity_id: str
    audio: AudioBuffer | None = None
    alignment: tuple[VisemeCue, ...] = ()
    synthesis_ms: float = 0.0
    first_audio_ms: float = 0.0
    error: EngineError | None = None
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaybackResult:
    status: Literal["completed", "interrupted", "failed"]
    played_frames: int
    total_frames: int
    playback_ms: float
    spoken_fraction: float
    error: EngineError | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceEvent:
    kind: Literal[
        "synthesis_started",
        "audio_ready",
        "playback_started",
        "playback_stopped",
        "playback_interrupted",
        "voice_failed",
        "emergency_voice_started",
        "segment_synthesis_started",
        "segment_audio_ready",
        "segment_enqueued",
        "segment_dequeued",
        "segment_dropped",
        "segment_discarded",
        "stream_producer_finished",
        "queue_underrun",
        "tts_idle_release_started",
        "tts_temporary_cache_released",
        "tts_predictive_reclaim",
    ]
    created_monotonic: float
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoicePipelineTrace:
    status: str = "idle"
    phase: str = ""
    error_code: str = ""
    input_characters: int = 0
    spoken_characters: int = 0
    synthesis_ms: float = 0.0
    processing_ms: float = 0.0
    playback_ms: float = 0.0
    first_audio_ms: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    frame_count: int = 0
    cancelled: bool = False
    output_mode: str = "single_pcm_stream"
    engine_id: str = ""
    voice_id: str = ""
    identity_id: str = ""
    spoken_text: str = ""
    segment_count: int = 1
    max_gap_ms: float = 0.0
    queue_underruns: int = 0
    generated_segment_count: int = 0
    generated_frame_count: int = 0
    discarded_segment_count: int = 0
    discarded_frame_count: int = 0
    stream_integrity_ok: bool = True
