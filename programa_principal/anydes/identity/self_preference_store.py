from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from anydes.cognition.contracts import SelfPreference
from anydes.nucleo.config import Config
from anydes.nucleo.path_locks import shared_path_lock


@dataclass
class SelfPreferenceStore:
    """Canonical persistence boundary for Anydes' mutable preferences."""

    path: Path

    @classmethod
    def from_config(cls, config: Config) -> "SelfPreferenceStore":
        return cls(config.self_state_path)

    def preferences(self) -> dict[str, SelfPreference]:
        preferences, _ = self.snapshot()
        return preferences

    def get(self, key: str) -> SelfPreference | None:
        return self.preferences().get(key)

    def revision(self) -> int:
        with shared_path_lock(self.path):
            return int(self._read_payload().get("revision", 0))

    def snapshot(self) -> tuple[dict[str, SelfPreference], int]:
        with shared_path_lock(self.path):
            payload = self._read_payload()
            values = payload.get("preferences", {})
            preferences = (
                {
                    str(key): SelfPreference.from_dict(value)
                    for key, value in values.items()
                    if isinstance(value, dict)
                }
                if isinstance(values, dict)
                else {}
            )
            return preferences, int(payload.get("revision", 0))

    def put_preference(self, preference: SelfPreference) -> int:
        with shared_path_lock(self.path):
            payload = self._read_payload()
            preferences = payload.setdefault("preferences", {})
            if not isinstance(preferences, dict):
                preferences = {}
                payload["preferences"] = preferences
            preferences[preference.key] = preference.to_dict()
            payload["schema_version"] = "2.0"
            payload["revision"] = int(payload.get("revision", 0)) + 1
            self._write_payload(payload)
            return int(payload["revision"])

    def _read_payload(self) -> dict[str, object]:
        default = {"schema_version": "2.0", "revision": 0, "preferences": {}}
        if not self.path.exists():
            return default
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        if not isinstance(payload, dict):
            return default
        payload.setdefault("schema_version", "1.0")
        payload.setdefault("revision", 0)
        payload.setdefault("preferences", {})
        return payload

    def _write_payload(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, self.path)
