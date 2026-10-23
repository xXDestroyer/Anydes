from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from anydes.memoria.data_protection import DataProtectionError, DataProtector, default_data_protector
from anydes.nucleo.path_locks import shared_path_lock


_SCHEMA_VERSION = 2


class ProtectedMemoryStore:
    """Local protected values encrypted by the operating system DPAPI.

    The passphrase remains an explicit feature gate for backward-compatible
    configuration. Cryptographic protection is provided by Windows, not by a
    project-specific cipher.
    """

    def __init__(
        self,
        path: Path,
        passphrase: str | None,
        *,
        protector: DataProtector | None = None,
    ) -> None:
        self.path = path
        self.passphrase = passphrase
        self._protector = protector
        self._lock = shared_path_lock(path)

    @property
    def enabled(self) -> bool:
        return bool(self.passphrase)

    def set(self, key: str, value: str) -> None:
        if not self.passphrase:
            raise ValueError("La memoria protegida necesita ANYDES_MEMORY_PASSPHRASE.")
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("La clave de memoria protegida no puede estar vacia.")
        with self._lock:
            data = self._load_document()
            protector = self._get_protector()
            data["entries"][clean_key] = {
                "scheme": protector.scheme,
                "payload": protector.protect(value.encode("utf-8")),
            }
            self._save_document(data)

    def get(self, key: str) -> str | None:
        if not self.passphrase:
            raise ValueError("La memoria protegida necesita ANYDES_MEMORY_PASSPHRASE.")
        with self._lock:
            document = self._load_document()
            entry = document["entries"].get(key)
            if not isinstance(entry, dict):
                return None
            protector = self._get_protector()
            if str(entry.get("scheme", "")) != protector.scheme:
                raise DataProtectionError("Protected memory uses an unsupported protection scheme.")
            plaintext = protector.unprotect(str(entry.get("payload", "")))
            return plaintext.decode("utf-8")

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._load_document()["entries"].keys())

    def _get_protector(self) -> DataProtector:
        if self._protector is None:
            self._protector = default_data_protector()
        return self._protector

    def _load_document(self) -> dict[str, object]:
        if not self.path.exists():
            return {"version": _SCHEMA_VERSION, "entries": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise DataProtectionError("Protected memory has an invalid document format.")
        if data.get("version") != _SCHEMA_VERSION:
            raise DataProtectionError(
                "Protected memory uses a legacy format. Restore it from backup before migration."
            )
        entries = data.get("entries")
        if not isinstance(entries, dict):
            raise DataProtectionError("Protected memory entries have an invalid format.")
        return {"version": _SCHEMA_VERSION, "entries": dict(entries)}

    def _save_document(self, data: dict[str, object]) -> None:
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
                json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink(missing_ok=True)
