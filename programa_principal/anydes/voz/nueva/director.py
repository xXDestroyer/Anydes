from __future__ import annotations

from dataclasses import dataclass

from anydes.cognition.contracts import SpeechIntention
from anydes.voz.identity import VoiceIdentity
from anydes.voz.nueva.contracts import VoiceDirection


def _bounded(value: float, limits: tuple[float, float]) -> float:
    return max(limits[0], min(limits[1], float(value)))


@dataclass(frozen=True)
class VoiceDirector:
    """Projects authorized turn state into bounded vocal direction.

    It never rewrites speech and never calls an LLM. Multiple continuous
    dimensions contribute to every parameter; an emotion label never selects a
    separate voice.
    """

    identity: VoiceIdentity

    def direct(self, speech: SpeechIntention) -> VoiceDirection:
        if not isinstance(speech, SpeechIntention):
            raise TypeError("VoiceDirector accepts only SpeechIntention.")
        limits = self.identity.variation_limits
        uncertainty = 1.0 - speech.confidence
        reflective = min(
            1.0,
            speech.thinking * 0.54 + speech.curiosity * 0.2 + uncertainty * 0.26,
        )
        activation = min(
            1.0,
            speech.energy * 0.58
            + self.identity.base_energy * 0.27
            + speech.confidence * 0.15,
        )
        connection = min(
            1.0,
            speech.warmth * 0.5
            + self.identity.warmth * 0.3
            + speech.smile * 0.2,
        )
        speed = _bounded(
            self.identity.base_speed
            * (0.96 + activation * 0.08 - reflective * 0.08 - uncertainty * 0.03)
            * (speech.speech_rate / 0.9),
            limits["speed"],
        )
        pitch = _bounded(
            self.identity.base_pitch
            + speech.pitch_variation * 0.08
            + activation * 0.035
            + connection * 0.025
            - reflective * 0.025,
            limits["pitch"],
        )
        energy = _bounded(
            activation * 0.78 + connection * 0.12 + speech.volume * 0.1,
            limits["energy"],
        )
        warmth = _bounded(
            connection * 0.82 + self.identity.warmth * 0.18,
            limits["warmth"],
        )
        expressiveness = _bounded(
            self.identity.expressiveness * 0.46
            + speech.pitch_variation * 0.2
            + activation * 0.18
            + connection * 0.16,
            limits["expressiveness"],
        )
        return VoiceDirection(
            text=speech.text,
            conversation_intent=str(speech.metadata.get("conversation_intent") or "respond"),
            authorized_emotion=speech.emotion,
            speed=speed,
            pitch=pitch,
            energy=energy,
            warmth=warmth,
            expressiveness=expressiveness,
            volume=max(0.55, min(1.0, speech.volume)),
            pause_before_ms=round(min(900.0, speech.pause_before * 1000.0)),
            pause_after_ms=round(min(650.0, speech.pause_after * 1000.0)),
            emphasis=tuple(
                str(item)
                for item in speech.metadata.get("voice_emphasis", [])
                if isinstance(item, str)
            ),
            direction_id=str(speech.metadata.get("intention_id") or speech.plan_id or "voice:turn"),
        )
