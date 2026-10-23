from __future__ import annotations

from anydes.cognition.contracts import EmotionSnapshot
from anydes.nucleo.config import Config
from anydes.nucleo.estado_emocional import MotorEmocional, obtener_motor


class EmotionProvider:
    """Adapts the current emotional motor state into an EmotionSnapshot."""

    def __init__(self, config: Config, motor: MotorEmocional | None = None) -> None:
        self.config = config
        self.motor = motor or obtener_motor()

    def snapshot(self) -> EmotionSnapshot:
        state = self.motor.obtener_estado()
        summary = state.como_texto()
        return EmotionSnapshot(
            energy=state.energia,
            mood=state.animo,
            interest=_normalize_signed_to_unit(state.interes),
            tension=state.tension,
            affection=_normalize_signed_to_unit(state.afecto),
            dominant_label=state.emocion_dominante,
            summary=f"{summary} Contexto emocional actual: {state.contexto}.",
        )


def _normalize_signed_to_unit(value: float) -> float:
    return (max(-1.0, min(1.0, float(value))) + 1.0) / 2.0
