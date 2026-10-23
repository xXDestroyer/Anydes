from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from anydes.nucleo.config import Config


CURRENT_RELATIONSHIP_SCHEMA_VERSION = 2
SUPPORTED_RELATIONSHIP_SCHEMA_VERSIONS = (1, 2)


class RelationshipStateError(RuntimeError):
    """Base error for relationship-state persistence failures."""


class RelationshipStateCorruptionError(RelationshipStateError):
    """The stored document is unreadable or violates its storage schema."""


class RelationshipStateVersionError(RelationshipStateError):
    """The stored schema version is not supported by this runtime."""


class RelationshipStateRevisionError(RelationshipStateError):
    """An optimistic transaction used a stale revision."""


class RelationshipStateValidationError(RelationshipStateError):
    """A transaction attempted to persist an invalid document."""


class RelationshipStateLockError(RelationshipStateError):
    """The store could not obtain its cross-process write lock in time."""


@dataclass(frozen=True)
class RelationshipStateDocument:
    """Versioned persistence envelope, independent from cognitive contracts."""

    schema_version: int
    revision: int
    updated_at: str
    collections: dict[str, dict[str, dict[str, Any]]]

    @classmethod
    def empty(cls) -> "RelationshipStateDocument":
        return cls(
            schema_version=CURRENT_RELATIONSHIP_SCHEMA_VERSION,
            revision=0,
            updated_at="",
            collections={},
        )

    def collection(self, name: str) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.collections.get(name, {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "collections": copy.deepcopy(self.collections),
        }


RelationshipMutation = Callable[[dict[str, dict[str, dict[str, Any]]]], None]


_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCKS_BY_PATH: dict[Path, threading.RLock] = {}


class RelationshipStateStore:
    """Atomic local persistence for relationship state.

    This layer knows nothing about threads, loops, promises or reasoning. It only
    protects a versioned set of named JSON collections. Semantic writes will be
    owned by RelationshipWriter in a later stage.
    """

    def __init__(self, path: Path, *, lock_timeout_seconds: float = 10.0) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")
        self.lock_timeout_seconds = max(0.1, float(lock_timeout_seconds))
        self._thread_lock = _lock_for_path(path)

    @classmethod
    def from_config(cls, config: Config) -> "RelationshipStateStore":
        return cls(config.relationship_state_path)

    def read(self) -> RelationshipStateDocument:
        with self._thread_lock:
            return self._read_unlocked()

    def transact(
        self,
        mutation: RelationshipMutation,
        *,
        expected_revision: int | None = None,
    ) -> RelationshipStateDocument:
        if not callable(mutation):
            raise TypeError("Relationship mutation must be callable.")

        with self._thread_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._interprocess_write_lock():
                current = self._read_unlocked()
                if expected_revision is not None and expected_revision != current.revision:
                    raise RelationshipStateRevisionError(
                        f"Expected relationship revision {expected_revision}, found {current.revision}."
                    )

                collections = copy.deepcopy(current.collections)
                mutation(collections)
                normalized = _validate_collections(collections)
                if normalized == current.collections:
                    return current

                updated = RelationshipStateDocument(
                    schema_version=CURRENT_RELATIONSHIP_SCHEMA_VERSION,
                    revision=current.revision + 1,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    collections=normalized,
                )
                self._write_atomic(updated)
                return updated

    def _read_unlocked(self) -> RelationshipStateDocument:
        if not self.path.exists():
            return RelationshipStateDocument.empty()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RelationshipStateCorruptionError(
                f"Relationship state could not be read: {exc}"
            ) from exc
        return _document_from_payload(payload)

    def _write_atomic(self, document: RelationshipStateDocument) -> None:
        try:
            serialized = json.dumps(
                document.to_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise RelationshipStateValidationError(
                f"Relationship state is not valid JSON: {exc}"
            ) from exc

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        except OSError as exc:
            raise RelationshipStateError(
                f"Relationship state could not be committed atomically: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @contextmanager
    def _interprocess_write_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            _ensure_lock_byte(handle)
            deadline = time.monotonic() + self.lock_timeout_seconds
            while True:
                try:
                    _lock_file(handle)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise RelationshipStateLockError(
                            f"Timed out waiting for relationship-state lock: {self.lock_path}"
                        ) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                _unlock_file(handle)


def _lock_for_path(path: Path) -> threading.RLock:
    normalized = path.resolve(strict=False)
    with _LOCK_REGISTRY_GUARD:
        return _LOCKS_BY_PATH.setdefault(normalized, threading.RLock())


def _document_from_payload(payload: Any) -> RelationshipStateDocument:
    if not isinstance(payload, dict):
        raise RelationshipStateCorruptionError("Relationship state root must be a JSON object.")

    version = payload.get("schema_version")
    if version not in SUPPORTED_RELATIONSHIP_SCHEMA_VERSIONS:
        raise RelationshipStateVersionError(
            f"Unsupported relationship schema version: {version!r}."
        )

    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise RelationshipStateCorruptionError("Relationship revision must be a non-negative integer.")

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        raise RelationshipStateCorruptionError("Relationship updated_at must be a string.")

    try:
        collections = _validate_collections(payload.get("collections"))
    except RelationshipStateValidationError as exc:
        raise RelationshipStateCorruptionError(str(exc)) from exc

    return RelationshipStateDocument(
        # Version 1 records are interpreted lazily and written as version 2 only
        # after a real transaction, so reads remain side-effect free.
        schema_version=CURRENT_RELATIONSHIP_SCHEMA_VERSION,
        revision=revision,
        updated_at=updated_at,
        collections=collections,
    )


def _validate_collections(value: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(value, dict):
        raise RelationshipStateValidationError("Relationship collections must be an object.")

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for collection_name, records in value.items():
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise RelationshipStateValidationError("Collection names must be non-empty strings.")
        if not isinstance(records, dict):
            raise RelationshipStateValidationError(
                f"Relationship collection '{collection_name}' must be an object."
            )
        normalized_records: dict[str, dict[str, Any]] = {}
        for record_id, record in records.items():
            if not isinstance(record_id, str) or not record_id.strip():
                raise RelationshipStateValidationError("Record ids must be non-empty strings.")
            if not isinstance(record, dict):
                raise RelationshipStateValidationError(
                    f"Relationship record '{record_id}' must be an object."
                )
            normalized_records[record_id] = copy.deepcopy(record)
        normalized[collection_name] = normalized_records

    try:
        json.dumps(normalized, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RelationshipStateValidationError(
            f"Relationship collections contain non-JSON data: {exc}"
        ) from exc
    return normalized


def _ensure_lock_byte(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)


def _lock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
