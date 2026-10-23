from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from anydes.cognition.contracts import CuriosityModel
from anydes.nucleo.config import Config
from anydes.nucleo.path_locks import shared_path_lock


CURRENT_CURIOSITY_SCHEMA_VERSION = 1


class CuriosityStateError(RuntimeError):
    """Base error for persistent curiosity state."""


class CuriosityStateCorruptionError(CuriosityStateError):
    """The stored document is unreadable or violates its schema."""


@dataclass(frozen=True)
class CuriosityStateDocument:
    schema_version: int
    revision: int
    updated_at: str
    curiosities: tuple[CuriosityModel, ...]

    @classmethod
    def empty(cls) -> "CuriosityStateDocument":
        return cls(
            schema_version=CURRENT_CURIOSITY_SCHEMA_VERSION,
            revision=0,
            updated_at="",
            curiosities=(),
        )


CuriosityStateMutation = Callable[[dict[str, CuriosityModel]], None]


class CuriosityStateStore:
    """Atomic, versioned persistence for contextual curiosity objects."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = shared_path_lock(path)

    @classmethod
    def from_config(cls, config: Config) -> "CuriosityStateStore":
        return cls(config.curiosity_state_path)

    def read(self) -> CuriosityStateDocument:
        with self._lock:
            return self._read_unlocked()

    def transact(self, mutation: CuriosityStateMutation) -> CuriosityStateDocument:
        if not callable(mutation):
            raise TypeError("Curiosity state mutation must be callable.")
        with self._lock:
            current = self._read_unlocked()
            records = {item.id: item for item in current.curiosities}
            mutation(records)
            normalized = _normalize_records(records)
            if normalized == current.curiosities:
                return current
            document = CuriosityStateDocument(
                schema_version=CURRENT_CURIOSITY_SCHEMA_VERSION,
                revision=current.revision + 1,
                updated_at=datetime.now(timezone.utc).isoformat(),
                curiosities=normalized,
            )
            self._write_atomic(document)
            return document

    def _read_unlocked(self) -> CuriosityStateDocument:
        if not self.path.exists():
            return CuriosityStateDocument.empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CuriosityStateCorruptionError(
                f"Curiosity state could not be read: {type(exc).__name__}"
            ) from exc
        return _document_from_payload(payload)

    def _write_atomic(self, document: CuriosityStateDocument) -> None:
        payload = {
            "schema_version": document.schema_version,
            "revision": document.revision,
            "updated_at": document.updated_at,
            "curiosities": {item.id: item.to_dict() for item in document.curiosities},
        }
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise CuriosityStateError("Curiosity state is not valid JSON.") from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except OSError as exc:
            raise CuriosityStateError("Curiosity state could not be committed atomically.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _document_from_payload(payload: object) -> CuriosityStateDocument:
    if not isinstance(payload, dict):
        raise CuriosityStateCorruptionError("Curiosity state root must be an object.")
    if payload.get("schema_version") != CURRENT_CURIOSITY_SCHEMA_VERSION:
        raise CuriosityStateCorruptionError("Unsupported curiosity state schema version.")
    raw_records = payload.get("curiosities", {})
    if not isinstance(raw_records, dict):
        raise CuriosityStateCorruptionError("Curiosity records must be an object.")
    try:
        records = {
            str(key): CuriosityModel.from_dict(value)
            for key, value in raw_records.items()
            if isinstance(value, dict)
        }
        if len(records) != len(raw_records):
            raise ValueError("A curiosity record is not an object.")
        normalized = _normalize_records(records)
        revision = max(0, int(payload.get("revision", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise CuriosityStateCorruptionError("Curiosity state contains invalid records.") from exc
    return CuriosityStateDocument(
        schema_version=CURRENT_CURIOSITY_SCHEMA_VERSION,
        revision=revision,
        updated_at=str(payload.get("updated_at", "")),
        curiosities=normalized,
    )


def _normalize_records(records: dict[str, CuriosityModel]) -> tuple[CuriosityModel, ...]:
    normalized: list[CuriosityModel] = []
    for key, item in records.items():
        if not isinstance(item, CuriosityModel):
            raise CuriosityStateError("Curiosity mutations must store CuriosityModel values.")
        if not key.strip() or key != item.id:
            raise CuriosityStateError("Curiosity record keys must match non-empty model ids.")
        normalized.append(item)
    return tuple(sorted(normalized, key=lambda item: (item.appeared_at, item.id)))
