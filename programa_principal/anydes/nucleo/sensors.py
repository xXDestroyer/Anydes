from __future__ import annotations

import threading
from collections.abc import Callable


class SensorHub:
    """Coordinates emergency shutdown without owning concrete sensor engines."""

    def __init__(self) -> None:
        self._stoppers: dict[tuple[str, str], Callable[[], None]] = {}
        self._lock = threading.RLock()

    def register(self, kind: str, key: str, stopper: Callable[[], None]) -> None:
        with self._lock:
            self._stoppers[(kind, key)] = stopper

    def unregister(self, kind: str, key: str) -> None:
        with self._lock:
            self._stoppers.pop((kind, key), None)

    def active(self, kind: str | None = None) -> bool:
        with self._lock:
            return any(kind is None or item_kind == kind for item_kind, _ in self._stoppers)

    def stop_all(self) -> tuple[str, ...]:
        with self._lock:
            callbacks = tuple(self._stoppers.items())
            self._stoppers.clear()
        stopped: list[str] = []
        for (kind, key), stopper in callbacks:
            try:
                stopper()
                stopped.append(f"{kind}:{key}")
            except Exception:
                continue
        return tuple(stopped)


sensor_hub = SensorHub()
