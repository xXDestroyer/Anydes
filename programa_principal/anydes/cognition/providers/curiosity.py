from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from anydes.cognition.contracts import CuriosityModel
from anydes.cognition.curiosity_store import CuriosityStateStore
from anydes.nucleo.config import Config


class CuriosityProvider:
    """Projects persisted curiosity state into context without deciding or writing."""

    def __init__(
        self,
        config: Config,
        memory=None,
        screen_state=None,
        *,
        store: CuriosityStateStore | None = None,
    ) -> None:
        # Positional legacy dependencies remain accepted while callers migrate.
        _ = memory, screen_state
        self.store = store or CuriosityStateStore.from_config(config)

    def snapshot(self, *, now: datetime | None = None) -> tuple[CuriosityModel, ...]:
        moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        document = self.store.read()
        return tuple(_with_age(item, moment) for item in document.curiosities)


def _with_age(item: CuriosityModel, moment: datetime) -> CuriosityModel:
    try:
        appeared = datetime.fromisoformat(item.appeared_at.replace("Z", "+00:00"))
        if appeared.tzinfo is None:
            appeared = appeared.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (moment - appeared.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        age_seconds = 0.0
    return replace(item, age_seconds=age_seconds)
