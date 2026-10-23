from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from anydes.memoria.data_protection import DataProtector, default_data_protector
from anydes.nucleo.path_locks import shared_path_lock


_PRIVATE_DETAIL_KEYS = frozenset({"argument_preview", "preview", "text_preview"})


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = shared_path_lock(path)

    def record(self, event: str, **details: Any) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "details": details,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")

    def sanitize_legacy_content(
        self,
        backup_dir: Path,
        *,
        protector: DataProtector | None = None,
    ) -> int:
        """Remove legacy message previews after creating a protected backup."""

        with self._lock:
            if not self.path.exists():
                return 0
            raw = self.path.read_bytes()
            entries: list[dict[str, Any]] = []
            redacted = 0
            for line in raw.decode("utf-8", errors="replace").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                details = entry.get("details")
                if isinstance(details, dict):
                    removed = sorted(_PRIVATE_DETAIL_KEYS.intersection(details))
                    if removed:
                        redacted += 1
                        details = {key: value for key, value in details.items() if key not in removed}
                        details["redacted_legacy_fields"] = removed
                        entry["details"] = details
                entries.append(entry)
            if redacted == 0:
                return 0

            active_protector = protector or default_data_protector()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = backup_dir / f"audit-{stamp}.protected.json"
            suffix = 1
            while backup.exists():
                backup = backup_dir / f"audit-{stamp}-{suffix}.protected.json"
                suffix += 1
            _atomic_write(
                backup,
                json.dumps(
                    {
                        "version": 1,
                        "source_name": self.path.name,
                        "scheme": active_protector.scheme,
                        "payload": active_protector.protect(raw),
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            sanitized = "".join(
                json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n"
                for entry in entries
            )
            _atomic_write(self.path, sanitized)
            return redacted


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
