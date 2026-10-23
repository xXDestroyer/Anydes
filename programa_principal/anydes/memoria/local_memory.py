from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from anydes.nucleo.path_locks import shared_path_lock


class JsonMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = shared_path_lock(path)

    def _load(self) -> dict[str, str]:
        with self._lock:
            if not self.path.exists():
                return {}
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return {}
            return {str(key): str(value) for key, value in data.items()}

    def _save(self, data: dict[str, str]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, self.path)

    def get(self, key: str) -> str | None:
        return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._load()
            data[key] = value
            self._save(data)

    def clear(self) -> None:
        self._save({})

    def all(self) -> dict[str, str]:
        return self._load()
