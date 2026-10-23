from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Callable

from ..path_policy import FilesystemBoundary


class WorkspaceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SettingsSnapshot:
    internal_workspace: str
    permanent_workspace: str
    temporary_workspace: str
    active_workspace: str
    active_workspace_kind: str
    authorized_external_scopes: tuple[str, ...]
    protected_core: bool = True


class SettingsStore:
    """Staging settings with explicit persistent/session separation."""

    schema = "anydes.tools.settings.v001_5"

    def __init__(
        self,
        project_root: Path,
        tools_root: Path | None = None,
        *,
        settings_path: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.tools_root = (tools_root or self.project_root / "anydes_tools").resolve()
        self.boundary = FilesystemBoundary(self.project_root, self.tools_root)
        self.internal_workspace = (self.tools_root / "workspaces" / "manual_v001").resolve()
        self.internal_workspace.mkdir(parents=True, exist_ok=True)
        self.settings_path = (
            settings_path
            or self.tools_root / "config" / "settings-v001_5.json"
        ).resolve()
        self._lock = threading.RLock()
        self._listeners: set[Callable[[SettingsSnapshot], None]] = set()
        self._permanent_workspace = ""
        self._temporary_workspace = ""
        self._session_scopes: list[str] = []
        self.last_load_error = ""
        self._load()

    def snapshot(self) -> SettingsSnapshot:
        with self._lock:
            permanent = self._permanent_workspace
            temporary = self._temporary_workspace
            active = temporary or permanent or str(self.internal_workspace)
            kind = "temporary" if temporary else "permanent" if permanent else "internal"
            scopes = tuple(dict.fromkeys(
                item
                for item in (permanent, temporary, *self._session_scopes)
                if item and Path(item).resolve() != self.internal_workspace
            ))
            return SettingsSnapshot(
                internal_workspace=str(self.internal_workspace),
                permanent_workspace=permanent,
                temporary_workspace=temporary,
                active_workspace=active,
                active_workspace_kind=kind,
                authorized_external_scopes=scopes,
            )

    def set_permanent_workspace(self, raw_path: str | Path) -> SettingsSnapshot:
        workspace = self._validate_workspace(raw_path)
        with self._lock:
            self._permanent_workspace = str(workspace)
            self._save_locked()
        return self._notify()

    def clear_permanent_workspace(self) -> SettingsSnapshot:
        with self._lock:
            self._permanent_workspace = ""
            self._save_locked()
        return self._notify()

    def set_temporary_workspace(self, raw_path: str | Path) -> SettingsSnapshot:
        workspace = self._validate_workspace(raw_path)
        with self._lock:
            self._temporary_workspace = str(workspace)
        return self._notify()

    def clear_temporary_workspace(self) -> SettingsSnapshot:
        with self._lock:
            self._temporary_workspace = ""
        return self._notify()

    def authorize_session_scope(self, raw_path: str | Path) -> SettingsSnapshot:
        workspace = self._validate_workspace(raw_path)
        with self._lock:
            value = str(workspace)
            self._session_scopes = [item for item in self._session_scopes if item != value]
            self._session_scopes.append(value)
        return self._notify()

    def revoke_session_scope(self, raw_path: str | Path) -> SettingsSnapshot:
        value = str(Path(raw_path).resolve(strict=False))
        with self._lock:
            self._session_scopes = [item for item in self._session_scopes if item != value]
        return self._notify()

    def subscribe(self, listener: Callable[[SettingsSnapshot], None]) -> None:
        with self._lock:
            self._listeners.add(listener)

    def unsubscribe(self, listener: Callable[[SettingsSnapshot], None]) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def safe_public_payload(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "schema": self.schema,
            "active_workspace": snapshot.active_workspace,
            "active_workspace_kind": snapshot.active_workspace_kind,
            "permanent_workspace": snapshot.permanent_workspace,
            "temporary_workspace": snapshot.temporary_workspace,
            "authorized_external_scopes": list(snapshot.authorized_external_scopes),
            "protected_core": True,
        }

    def _load(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if payload.get("schema") != self.schema:
                raise ValueError("settings_schema_mismatch")
            value = str(payload.get("permanent_workspace") or "").strip()
            if value:
                self._permanent_workspace = str(self._validate_workspace(value))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._permanent_workspace = ""
            self.last_load_error = f"{type(exc).__name__}: {exc}"

    def _save_locked(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "permanent_workspace": self._permanent_workspace,
        }
        temporary = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.settings_path)

    def _notify(self) -> SettingsSnapshot:
        snapshot = self.snapshot()
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(snapshot)
        return snapshot

    def _validate_workspace(self, raw_path: str | Path) -> Path:
        value = str(raw_path).strip()
        if not value:
            raise WorkspaceValidationError("workspace_path_empty")
        candidate = Path(value)
        if not candidate.is_absolute():
            raise WorkspaceValidationError("workspace_path_must_be_absolute")
        if self.boundary._contains_reparse_point(candidate):
            raise WorkspaceValidationError("DENIED_REPARSE_POINT")
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceValidationError("workspace_path_unavailable") from exc
        if not candidate.is_dir():
            raise WorkspaceValidationError("workspace_must_be_directory")
        decision = self.boundary.classify(
            candidate,
            operation="READ",
            authorized_scopes=(candidate,),
        )
        if not decision.allowed:
            raise WorkspaceValidationError(decision.reason)
        return decision.path


_STORES: dict[tuple[str, str], SettingsStore] = {}
_STORES_LOCK = threading.Lock()


def settings_for(project_root: Path, tools_root: Path | None = None) -> SettingsStore:
    project = project_root.resolve()
    tools = (tools_root or project / "anydes_tools").resolve()
    key = (str(project), str(tools))
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = SettingsStore(project, tools)
            _STORES[key] = store
        return store
