from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from anydes.memoria.data_protection import (
    DataProtectionError,
    DataProtector,
    default_data_protector,
)
from anydes.nucleo.config import Config
from anydes.nucleo.path_locks import shared_path_lock


@dataclass(frozen=True)
class CalendarEventRecord:
    entry_id: str
    draft_id: str
    draft_version: int
    title: str
    start_iso: str
    end_iso: str
    timezone: str
    backend: str
    task_id: str
    status: str
    recorded_at: str


class CalendarEventStore:
    """Encrypted index of calendar changes verified by the real backend."""

    _version = 1

    def __init__(
        self,
        path: Path,
        *,
        protector: DataProtector | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._protector = protector or default_data_protector(
            passphrase=os.getenv("ANYDES_MEMORY_PASSPHRASE")
        )
        self._path_lock = shared_path_lock(self.path)
        self._lock = threading.RLock()
        self._records = list(self._load())

    @classmethod
    def from_config(cls, config: Config) -> "CalendarEventStore":
        return cls(config.realtime_calendar_events_path)

    def record_verified_creation(
        self,
        *,
        receipt: dict[str, object],
        parameters: dict[str, object],
        task_id: str,
    ) -> CalendarEventRecord | None:
        data = dict(receipt.get("data") or {})
        if (
            str(receipt.get("capability") or "") != "calendar_create"
            or str(receipt.get("status") or "") != "success"
            or not bool(receipt.get("verified"))
            or not str(data.get("entry_id") or "").strip()
        ):
            return None
        record = CalendarEventRecord(
            entry_id=str(data["entry_id"]).strip(),
            draft_id=str(parameters.get("draft_id") or ""),
            draft_version=int(parameters.get("draft_version") or 1),
            title=str(data.get("title") or parameters.get("title") or "").strip(),
            start_iso=str(
                data.get("start_iso") or parameters.get("start_iso") or ""
            ),
            end_iso=str(data.get("end_iso") or parameters.get("end_iso") or ""),
            timezone=str(data.get("timezone") or parameters.get("timezone") or ""),
            backend=str(data.get("backend") or "connected_calendar"),
            task_id=task_id,
            status="created",
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._records = [
                item for item in self._records if item.entry_id != record.entry_id
            ]
            self._records.append(record)
            self._persist_locked()
        return record

    def records(self) -> tuple[CalendarEventRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def get(self, entry_id: str) -> CalendarEventRecord | None:
        with self._lock:
            return next(
                (item for item in self._records if item.entry_id == entry_id),
                None,
            )

    def _persist_locked(self) -> None:
        raw = json.dumps(
            [asdict(item) for item in self._records],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        envelope = {
            "schema_version": self._version,
            "scheme": self._protector.scheme,
            "payload": self._protector.protect(raw),
        }
        with self._path_lock:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(envelope, handle, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, self.path)

    def _load(self) -> tuple[CalendarEventRecord, ...]:
        if not self.path.exists():
            return ()
        try:
            with self._path_lock:
                envelope = json.loads(self.path.read_text(encoding="utf-8"))
            if str(envelope.get("scheme") or "") != self._protector.scheme:
                return ()
            raw = self._protector.unprotect(str(envelope.get("payload") or ""))
            payload = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            DataProtectionError,
        ):
            return ()
        records: list[CalendarEventRecord] = []
        for item in payload if isinstance(payload, list) else ():
            if not isinstance(item, dict):
                continue
            try:
                records.append(CalendarEventRecord(**item))
            except (TypeError, ValueError):
                continue
        return tuple(records)
