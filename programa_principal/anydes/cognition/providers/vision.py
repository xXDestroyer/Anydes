from __future__ import annotations

from datetime import datetime, timezone

from anydes.cognition.contracts import Observation
from anydes.nucleo.config import Config
from anydes.vision.state import ScreenState, get_screen_state
from anydes.vision.observation_store import VisionObservationStore, get_vision_observation_store


class VisionProvider:
    """Converts current screen/vision state into observations.

    It does not capture the screen, call a vision model, generate comments, or
    decide whether visual context should be used. It only reports what visual
    state is already available for the Reasoning Engine.
    """

    def __init__(
        self,
        config: Config,
        screen_state: ScreenState | None = None,
        observation_store: VisionObservationStore | None = None,
    ) -> None:
        self.config = config
        self.screen_state = screen_state or get_screen_state()
        self.observation_store = observation_store or get_vision_observation_store()

    def observations(self) -> tuple[Observation, ...]:
        if not self.config.vision_enabled:
            return (
                Observation(
                    id="vision:disabled",
                    source="vision",
                    summary="Vision is disabled in configuration.",
                    confidence=1.0,
                    salience=0.1,
                    data={
                        "vision_enabled": False,
                        "watching": False,
                        "has_recent_frame": False,
                    },
                ),
            )

        watching = self.screen_state.is_watching()
        jpeg = self.screen_state.latest_jpeg()
        captured_at = self.screen_state.last_capture_at
        has_frame = bool(jpeg)

        if has_frame:
            summary = "Vision has a recent screen frame available."
            if watching:
                summary = "Vision is actively watching the screen and has a recent frame."
        elif watching:
            summary = "Vision is active but no screen frame is available yet."
        else:
            summary = "Vision is enabled but has no active or recent screen frame."

        state_observation = Observation(
            id="vision:screen_state",
            source="vision",
            summary=summary,
            confidence=0.9 if has_frame or watching else 0.6,
            salience=0.7 if has_frame or watching else 0.25,
            data={
                "vision_enabled": True,
                "watching": watching,
                "has_recent_frame": has_frame,
                "frame_size_bytes": len(jpeg) if jpeg else 0,
                "last_capture_at": captured_at.isoformat() if captured_at else "",
                "age_seconds": _age_seconds(captured_at) if captured_at else None,
                "provider": self.config.resolved_vision_provider(),
                "model": self.config.resolved_vision_model(),
            },
        )
        described = tuple(
            Observation(
                id=item.id,
                source="vision",
                summary=item.summary,
                timestamp=item.observed_at.isoformat(),
                confidence=item.confidence,
                salience=item.salience,
                data={
                    "evidence_type": "visual_description",
                    "observed_at": item.observed_at.isoformat(),
                    **dict(item.metadata),
                },
            )
            for item in self.observation_store.snapshot()
        )
        return (state_observation, *described)


def _age_seconds(captured_at: datetime) -> float:
    return max(0.0, (datetime.now(timezone.utc) - captured_at).total_seconds())
