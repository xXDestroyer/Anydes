from __future__ import annotations

from dataclasses import dataclass, field

from anydes.cognition.contracts import JsonValue, SpeechIntention, clamp01
from anydes.voz.identity import VoiceIdentity, VoiceProfile


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass(frozen=True)
class VoiceSynchronizationPlan:
    """Backend-neutral signals shared by voice and the future VRM body."""

    lip_sync: bool = True
    smile: float = 0.0
    head_motion: float = 0.0
    gaze_softness: float = 0.5
    breathing: float = 0.2
    micro_expression: str = "calm"
    timeline_cues: tuple[dict[str, JsonValue], ...] = ()

    def __post_init__(self) -> None:
        for name in ("smile", "head_motion", "gaze_softness", "breathing"):
            object.__setattr__(self, name, clamp01(getattr(self, name)))
        object.__setattr__(self, "timeline_cues", tuple(self.timeline_cues))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "lip_sync": self.lip_sync,
            "smile": self.smile,
            "head_motion": self.head_motion,
            "gaze_softness": self.gaze_softness,
            "breathing": self.breathing,
            "micro_expression": self.micro_expression,
            "timeline_cues": list(self.timeline_cues),
        }


@dataclass(frozen=True)
class VoiceRenderPlan:
    """A portable vocal performance, independent from any TTS backend."""

    text: str
    backend_text: str
    speech_rate: float
    pitch_variation: float
    volume: float
    pause_before: float
    pause_after: float
    energy: float
    intensity: float
    warmth: float
    confidence: float
    curiosity: float
    thinking: float
    smile: float
    breathiness: float
    micro_pause_scale: float
    emotion: str
    identity: VoiceIdentity
    synchronization: VoiceSynchronizationPlan = field(default_factory=VoiceSynchronizationPlan)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "text": self.text,
            "backend_text": self.backend_text,
            "speech_rate": self.speech_rate,
            "pitch_variation": self.pitch_variation,
            "volume": self.volume,
            "pause_before": self.pause_before,
            "pause_after": self.pause_after,
            "energy": self.energy,
            "intensity": self.intensity,
            "warmth": self.warmth,
            "confidence": self.confidence,
            "curiosity": self.curiosity,
            "thinking": self.thinking,
            "smile": self.smile,
            "breathiness": self.breathiness,
            "micro_pause_scale": self.micro_pause_scale,
            "emotion": self.emotion,
            "identity": self.identity.to_dict(),
            "synchronization": self.synchronization.to_dict(),
        }


@dataclass(frozen=True)
class VoiceInterpreter:
    """Blends Anydes' identity and full speech state into one performance.

    No single emotion label selects a preset. Every rendered dimension is a
    bounded blend of several continuous signals, so the same sentence can be
    performed differently as Anydes' internal state changes.
    """

    profile: VoiceProfile = field(default_factory=VoiceProfile)

    def interpret(self, speech: SpeechIntention) -> VoiceRenderPlan:
        if not isinstance(speech, SpeechIntention):
            raise TypeError("VoiceInterpreter accepts only SpeechIntention.")

        identity = self.profile.identity
        uncertainty = 1.0 - speech.confidence
        reflective_load = clamp01(
            speech.thinking * 0.52
            + speech.curiosity * 0.23
            + uncertainty * 0.25
        )
        activation = clamp01(
            speech.energy * 0.58
            + identity.base_energy * 0.27
            + speech.confidence * 0.15
        )
        connection = clamp01(
            speech.warmth * 0.52
            + identity.warmth * 0.28
            + speech.smile * 0.20
        )

        rate_factor = (
            1.0
            + (activation - 0.5) * 0.10
            - reflective_load * 0.07
            - speech.curiosity * 0.025
            - uncertainty * 0.035
        )
        speech_rate = _clamp(speech.speech_rate * rate_factor, 0.72, 1.12)
        pause_before = _clamp(
            speech.pause_before
            + reflective_load * 0.22
            + speech.curiosity * 0.05
            + uncertainty * 0.07,
            0.0,
            0.9,
        )
        pause_after = _clamp(
            speech.pause_after
            + reflective_load * 0.06
            + uncertainty * 0.08
            + (1.0 - activation) * 0.025,
            0.0,
            0.65,
        )
        pitch_variation = _clamp(
            speech.pitch_variation * 0.68
            + speech.curiosity * 0.08
            + activation * 0.055
            + connection * 0.035,
            0.06,
            0.34,
        )
        intensity = _clamp(
            activation * 0.46
            + speech.volume * 0.22
            + speech.confidence * 0.17
            + connection * 0.15,
            0.25,
            0.9,
        )
        volume = _clamp(
            speech.volume
            * (0.88 + activation * 0.08 + speech.confidence * 0.025 + connection * 0.02),
            0.62,
            1.0,
        )
        smile = _clamp(
            speech.smile * 0.58
            + identity.voice_smile * 0.22
            + connection * 0.14
            + activation * 0.06,
            0.0,
            0.68,
        )
        breathiness = _clamp(
            identity.breathiness * 0.58
            + reflective_load * 0.17
            + connection * 0.12
            + (1.0 - activation) * 0.13,
            0.05,
            0.52,
        )
        micro_pause_scale = _clamp(
            0.22
            + reflective_load * 0.34
            + uncertainty * 0.20
            + speech.curiosity * 0.14
            + (1.0 - activation) * 0.10,
            0.15,
            0.78,
        )

        synchronization = VoiceSynchronizationPlan(
            lip_sync=bool(speech.text.strip()),
            smile=smile,
            head_motion=_clamp(
                activation * 0.34
                + speech.curiosity * 0.18
                + connection * 0.12
                + speech.pitch_variation * 0.16,
                0.05,
                0.5,
            ),
            gaze_softness=_clamp(
                connection * 0.58
                + reflective_load * 0.17
                + speech.confidence * 0.15
                + identity.naturalness * 0.10,
                0.35,
                0.92,
            ),
            breathing=_clamp(
                breathiness * 0.42
                + reflective_load * 0.22
                + (1.0 - activation) * 0.18
                + identity.naturalness * 0.10,
                0.12,
                0.58,
            ),
            micro_expression=speech.emotion,
            timeline_cues=(),
        )

        return VoiceRenderPlan(
            text=speech.text,
            backend_text=speech.text,
            speech_rate=speech_rate,
            pitch_variation=pitch_variation,
            volume=volume,
            pause_before=pause_before,
            pause_after=pause_after,
            energy=activation,
            intensity=intensity,
            warmth=connection,
            confidence=speech.confidence,
            curiosity=speech.curiosity,
            thinking=speech.thinking,
            smile=smile,
            breathiness=breathiness,
            micro_pause_scale=micro_pause_scale,
            emotion=speech.emotion,
            identity=identity,
            synchronization=synchronization,
        )
