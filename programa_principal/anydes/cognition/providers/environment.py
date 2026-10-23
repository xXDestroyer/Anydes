from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from anydes.cognition.contracts import Observation
from anydes.interfaz.pyside_companion.room_environment import RoomEnvironment, synced_environment
from anydes.nucleo.config import Config


@dataclass
class EnvironmentProvider:
    """Reports the living room environment as cognitive observations.

    This provider only contributes context. It never decides whether Anydes
    should mention the weather, look at the window, or change behavior.
    """

    config: Config

    def observations(self, moment: datetime | None = None) -> tuple[Observation, ...]:
        local_moment = moment or datetime.now().astimezone()
        environment = synced_environment(local_moment)
        return (environment_observation(environment, local_moment),)


def environment_observation(
    environment: RoomEnvironment,
    moment: datetime | None = None,
) -> Observation:
    local_moment = moment or datetime.now().astimezone()
    return Observation(
        id="environment:room",
        source="environment",
        summary=(
            "Room environment: "
            f"{environment.time_of_day}, weather {environment.weather}, season {environment.season}."
        ),
        confidence=0.82,
        salience=_salience(environment),
        data={
            "time_of_day": environment.time_of_day,
            "weather": environment.weather,
            "season": environment.season,
            "local_time": local_moment.isoformat(),
            "local_date": local_moment.date().isoformat(),
            "timezone": str(local_moment.tzinfo or "local"),
            "sunrise": environment.sunrise,
            "sunset": environment.sunset,
            "warm_light": round(environment.warm_light, 3),
            "cool_light": round(environment.cool_light, 3),
            "lamp_glow": round(environment.lamp_glow, 3),
            "window_glow": round(environment.window_glow, 3),
            "particle_density": round(environment.particle_density, 3),
        },
    )


def _salience(environment: RoomEnvironment) -> float:
    if environment.weather in {"storm", "snow"}:
        return 0.72
    if environment.time_of_day in {"dawn", "sunset"}:
        return 0.62
    if environment.weather == "rain":
        return 0.55
    return 0.42
