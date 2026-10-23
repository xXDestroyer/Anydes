from __future__ import annotations

import threading
from pathlib import Path


_registry_guard = threading.Lock()
_locks: dict[Path, threading.RLock] = {}


def shared_path_lock(path: Path) -> threading.RLock:
    """Return one process-wide reentrant lock for a resolved local path."""

    normalized = path.resolve(strict=False)
    with _registry_guard:
        return _locks.setdefault(normalized, threading.RLock())
