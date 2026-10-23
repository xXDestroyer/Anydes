from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from anydes.memoria.data_protection import DataProtectionError, DataProtector, default_data_protector
from anydes.nucleo.path_locks import shared_path_lock


Category = str
_SCHEMA_VERSION = 2
_PROTECTED_CATEGORIES = frozenset({"Personal", "Secretos"})


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str
    time: str = ""
    category: str = "Normal"
    cognitively_retrievable: bool = True
    record_id: str = ""
    causal_status: str = "resolved"
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationRetentionPolicy:
    max_age_days: int | None = None
    max_turns: int | None = None

    def __post_init__(self) -> None:
        if self.max_age_days is not None and self.max_age_days < 1:
            raise ValueError("max_age_days must be positive when configured.")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be positive when configured.")

    @property
    def enabled(self) -> bool:
        return self.max_age_days is not None or self.max_turns is not None


@dataclass(frozen=True)
class ConversationMigrationResult:
    status: str
    migrated_turns: int = 0
    backup_path: str = ""


@dataclass(frozen=True)
class ConversationRepairResult:
    status: str
    repaired_turns: int = 0
    safety_backup_path: str = ""


class ConversationMemory:
    """Private historical archive plus a separate non-secret cognitive index.

    Normal turns remain readable local records. Personal and Secret turns are
    protected at rest. Secret turns never enter the cognitive retrieval index.
    """

    def __init__(
        self,
        log_path: Path,
        classified_path: Path,
        *,
        cognitive_path: Path | None = None,
        pending_path: Path | None = None,
        legacy_log_path: Path | None = None,
        legacy_classified_path: Path | None = None,
        migration_backup_dir: Path | None = None,
        migration_marker_path: Path | None = None,
        protector: DataProtector | None = None,
        retention_policy: ConversationRetentionPolicy | None = None,
    ) -> None:
        self.log_path = log_path
        self.archive_path = log_path
        self.classified_path = classified_path
        self.cognitive_path = cognitive_path or log_path.with_name(f"{log_path.stem}.cognitive.jsonl")
        self.pending_path = pending_path or self.cognitive_path.with_name(
            f"{self.cognitive_path.stem}.pending.jsonl"
        )
        self.legacy_log_path = legacy_log_path
        self.legacy_classified_path = legacy_classified_path
        self.migration_backup_dir = migration_backup_dir or log_path.parent / "migration_backups"
        self.migration_marker_path = migration_marker_path or log_path.with_suffix(".migration.json")
        self._protector = protector
        self.retention_policy = retention_policy or ConversationRetentionPolicy()
        self._lock = shared_path_lock(log_path)
        self._unavailable_protected_entries = 0

    @property
    def unavailable_protected_entries(self) -> int:
        return self._unavailable_protected_entries

    @_synchronized
    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        cognitively_retrievable: bool = True,
    ) -> None:
        moment = datetime.now(timezone.utc).isoformat()
        category = classify_turn(user_text, assistant_text)
        archive_entry = self._encode_entry(
            ConversationTurn(
                user=user_text,
                assistant=assistant_text,
                time=moment,
                category=category,
                cognitively_retrievable=cognitively_retrievable,
            )
        )
        self._append_json_line(self.archive_path, archive_entry)

        if category != "Secretos" and cognitively_retrievable:
            self._append_json_line(self.cognitive_path, archive_entry)

        self._append_classified(category, moment)
        if self.retention_policy.enabled:
            self._apply_retention_locked(self.retention_policy)

    @_synchronized
    def record_rejected_turn(
        self,
        user_text: str,
        visible_recovery: str,
        *,
        rejection_reasons: tuple[str, ...] = (),
    ) -> None:
        """Archive a rejection while keeping only the original input pending."""

        moment = datetime.now(timezone.utc).isoformat()
        category = classify_text(user_text)
        record_id = f"rejected:{uuid4().hex}"
        archive_turn = ConversationTurn(
            user=user_text,
            assistant=visible_recovery,
            time=moment,
            category=category,
            cognitively_retrievable=False,
            record_id=record_id,
            causal_status="rejected",
            rejection_reasons=tuple(rejection_reasons),
        )
        self._append_json_line(
            self.archive_path,
            self._encode_entry(archive_turn),
        )
        if category != "Secretos":
            pending_turn = ConversationTurn(
                user=user_text,
                assistant="",
                time=moment,
                category=category,
                cognitively_retrievable=True,
                record_id=record_id,
                causal_status="pending",
                rejection_reasons=tuple(rejection_reasons),
            )
            self._append_json_line(
                self.pending_path,
                self._encode_entry(pending_turn),
            )
        self._append_classified(category, moment)
        if self.retention_policy.enabled:
            self._apply_retention_locked(self.retention_policy)

    @_synchronized
    def pending_turns(self, limit: int = 6) -> list[ConversationTurn]:
        turns = self._read_turns(self.pending_path, include_secrets=False)
        if not limit:
            return []
        return list(reversed(turns[-max(0, limit) :]))

    @_synchronized
    def counts(self) -> dict[str, int]:
        data = self._load_classified()
        return {name: len(items) for name, items in data.items()}

    @_synchronized
    def first_user_turn(self) -> dict[str, str] | None:
        for turn in self._read_turns(self.cognitive_path, include_secrets=False):
            user = turn.user.strip()
            if not user or user == "[iniciativa automatica]":
                continue
            return {
                "user": user,
                "assistant": turn.assistant,
                "time": turn.time,
                "category": turn.category,
            }
        return None

    @_synchronized
    def recent_turns(
        self,
        limit: int = 10,
        *,
        include_secrets: bool = False,
    ) -> list[ConversationTurn]:
        source = self.archive_path if include_secrets else self.cognitive_path
        turns = self._read_turns(source, include_secrets=include_secrets)
        return turns[-max(0, limit) :] if limit else []

    @_synchronized
    def search(
        self,
        query: str,
        limit: int = 8,
        *,
        include_secrets: bool = False,
    ) -> list[dict[str, str]]:
        query_words = _tokenize(query)
        if not query_words:
            return []

        source = self.archive_path if include_secrets else self.cognitive_path
        matches: list[tuple[int, ConversationTurn]] = []
        for turn in self._read_turns(source, include_secrets=include_secrets):
            if turn.category == "Secretos" and not include_secrets:
                continue
            haystack_words = set(_tokenize(f"{turn.user} {turn.assistant}"))
            score = sum(1 for word in query_words if word in haystack_words)
            if score:
                matches.append((score, turn))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "category": turn.category,
                "time": turn.time,
                "user": turn.user,
                "assistant": turn.assistant,
            }
            for _, turn in matches[:limit]
        ]

    @_synchronized
    def relevant_context(
        self,
        query: str,
        *,
        limit: int = 5,
        include_secrets: bool = False,
    ) -> str:
        matches = self.search(query, limit=limit, include_secrets=include_secrets)
        if not matches:
            return "Recuerdos relevantes para este mensaje: sin coincidencias claras."

        lines = ["Recuerdos relevantes para este mensaje:"]
        for item in matches:
            lines.append(f"- [{item.get('category', 'Normal')}] Tu: {item['user'][:240]}")
            lines.append(f"  Anydes: {item['assistant'][:240]}")
        return "\n".join(lines)

    @_synchronized
    def recent_summary(self, *, limit: int = 12, include_secrets: bool = False) -> str:
        turns = self.recent_turns(limit=limit, include_secrets=include_secrets)
        if not turns:
            return "Todavia no hay conversaciones recientes guardadas."
        lines = ["Conversaciones recientes guardadas:"]
        for turn in turns:
            lines.append(f"- Tu: {turn.user[:260]}")
            lines.append(f"  Anydes: {turn.assistant[:260]}")
        return "\n".join(lines)

    @_synchronized
    def context(self, *, include_secrets: bool = False, limit: int = 12) -> str:
        counts = self.counts()
        lines = [
            "Memoria conversacional clasificada:",
            f"- Normal: {counts.get('Normal', 0)} dato(s).",
            f"- Personal: {counts.get('Personal', 0)} dato(s).",
            (
                f"- Secretos: {counts.get('Secretos', 0)} dato(s)."
                if include_secrets
                else f"- Secretos: {counts.get('Secretos', 0)} dato(s), ocultos por seguridad."
            ),
        ]
        recent = self.recent_turns(limit=min(limit, 6), include_secrets=include_secrets)
        if recent:
            lines.append("Conversaciones recientes disponibles:")
            for turn in recent:
                lines.append(f"- Usuario: {turn.user[:300]}")
                lines.append(f"  Anydes: {turn.assistant[:300]}")
        return "\n".join(lines)

    @_synchronized
    def migrate_legacy(self) -> ConversationMigrationResult:
        if self.migration_marker_path.exists():
            self._retire_legacy_plaintext()
            return ConversationMigrationResult(status="already_migrated")
        legacy = self.legacy_log_path
        if legacy is None or not legacy.exists() or legacy.resolve() == self.archive_path.resolve():
            return ConversationMigrationResult(status="not_needed")

        backup = self._create_versioned_backup(
            tuple(
                path
                for path in (self.legacy_log_path, self.legacy_classified_path)
                if path is not None and path.exists()
            )
        )
        legacy_entries = self._read_raw_jsonl(legacy)
        archive_entries: list[dict[str, object]] = []
        cognitive_entries: list[dict[str, object]] = []
        categories = _empty_categories()
        for raw in legacy_entries:
            user = str(raw.get("user", ""))
            assistant = str(raw.get("assistant", ""))
            category = _normalize_category(str(raw.get("category", ""))) or classify_turn(user, assistant)
            turn = ConversationTurn(
                user=user,
                assistant=assistant,
                time=str(raw.get("time", "")) or datetime.now(timezone.utc).isoformat(),
                category=category,
            )
            encoded = self._encode_entry(turn)
            archive_entries.append(encoded)
            categories[category].append(_category_metadata(turn.time, category))
            if category != "Secretos":
                cognitive_entries.append(encoded)

        try:
            self._commit_migration(archive_entries, cognitive_entries, categories, backup)
        except Exception:
            self._remove_incomplete_migration()
            raise
        self._retire_legacy_plaintext()
        return ConversationMigrationResult(
            status="migrated",
            migrated_turns=len(archive_entries),
            backup_path=str(backup),
        )

    @_synchronized
    def repair_from_protected_legacy_backup(
        self,
        protected_backup: Path,
        *,
        source_protector: DataProtector | None = None,
    ) -> ConversationRepairResult:
        """Re-encrypt a legacy backup after preserving the current v2 files.

        Plaintext exists only in memory. The source backup is never modified and
        every protected entry is verified before and after the atomic commit.
        """

        envelope = json.loads(protected_backup.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise DataProtectionError("Conversation repair backup has an invalid format.")
        protector = self._get_protector()
        backup_protector = source_protector or protector
        if str(envelope.get("scheme", "")) != backup_protector.scheme:
            raise DataProtectionError("Conversation repair backup uses an unsupported scheme.")
        legacy_payload = backup_protector.unprotect(str(envelope.get("payload", "")))
        archive_entries, cognitive_entries, categories = self._prepare_legacy_payload(
            legacy_payload
        )
        self._verify_protected_entries(archive_entries)

        targets = tuple(
            path
            for path in (
                self.archive_path,
                self.cognitive_path,
                self.classified_path,
                self.migration_marker_path,
            )
            if path.exists()
        )
        safety_backup = self._create_versioned_backup(targets)
        try:
            self._commit_repair(
                archive_entries,
                cognitive_entries,
                categories,
                protected_backup,
                safety_backup,
            )
            verified = self._read_turns(
                self.archive_path,
                include_secrets=True,
                strict_protection=True,
            )
            if len(verified) != len(archive_entries):
                raise DataProtectionError("Conversation repair verification was incomplete.")
        except Exception:
            self._restore_versioned_backup(safety_backup, targets)
            raise
        return ConversationRepairResult(
            status="repaired",
            repaired_turns=len(archive_entries),
            safety_backup_path=str(safety_backup),
        )

    @_synchronized
    def export(self, destination: Path, *, include_secrets: bool = False) -> int:
        """Export a readable copy. Secrets are excluded unless explicitly requested."""

        source = self.archive_path if include_secrets else self.cognitive_path
        turns = self._read_turns(
            source,
            include_secrets=include_secrets,
            strict_protection=True,
        )
        payload = "".join(
            json.dumps(
                {
                    "category": turn.category,
                    "time": turn.time,
                    "user": turn.user,
                    "assistant": turn.assistant,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
            for turn in turns
        )
        _atomic_write_text(destination, payload)
        return len(turns)

    @_synchronized
    def export_protected_archive(self, destination: Path) -> int:
        """Copy the encrypted archive without decrypting protected records."""

        entries = self._read_raw_jsonl(self.archive_path)
        payload = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in entries)
        _atomic_write_text(destination, payload)
        return len(entries)

    @_synchronized
    def delete_all(self, *, create_backup: bool = False) -> str:
        backup = ""
        paths = tuple(
            path
            for path in (
                self.archive_path,
                self.cognitive_path,
                self.pending_path,
                self.classified_path,
            )
            if path.exists()
        )
        if create_backup and paths:
            backup = str(self._create_versioned_backup(paths))
        for path in paths:
            path.unlink(missing_ok=True)
        self.migration_marker_path.unlink(missing_ok=True)
        return backup

    @_synchronized
    def apply_retention(self, policy: ConversationRetentionPolicy | None = None) -> int:
        return self._apply_retention_locked(policy or self.retention_policy)

    def _apply_retention_locked(self, policy: ConversationRetentionPolicy) -> int:
        if not policy.enabled:
            return 0
        archive_entries = self._filter_by_retention(self._read_raw_jsonl(self.archive_path), policy)
        cognitive_entries = self._filter_by_retention(self._read_raw_jsonl(self.cognitive_path), policy)
        pending_entries = self._filter_by_retention(
            self._read_raw_jsonl(self.pending_path),
            policy,
        )
        previous_count = len(self._read_raw_jsonl(self.archive_path))
        categories = _empty_categories()
        for entry in archive_entries:
            category = _normalize_category(str(entry.get("category", ""))) or "Normal"
            categories[category].append(_category_metadata(str(entry.get("time", "")), category))
        _write_jsonl_atomic(self.archive_path, archive_entries)
        _write_jsonl_atomic(self.cognitive_path, cognitive_entries)
        _write_jsonl_atomic(self.pending_path, pending_entries)
        _atomic_write_json(self.classified_path, {"version": _SCHEMA_VERSION, "categories": categories})
        return max(0, previous_count - len(archive_entries))

    def _filter_by_retention(
        self,
        entries: list[dict[str, object]],
        policy: ConversationRetentionPolicy,
    ) -> list[dict[str, object]]:
        retained = entries
        if policy.max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=policy.max_age_days)
            retained = [entry for entry in retained if _entry_datetime(entry) >= cutoff]
        if policy.max_turns is not None:
            retained = retained[-policy.max_turns :]
        return retained

    def _encode_entry(self, turn: ConversationTurn) -> dict[str, object]:
        entry: dict[str, object] = {
            "version": _SCHEMA_VERSION,
            "category": turn.category,
            "time": turn.time,
            "cognitively_retrievable": turn.cognitively_retrievable,
            "record_id": turn.record_id,
            "causal_status": turn.causal_status,
            "rejection_reasons": list(turn.rejection_reasons),
        }
        if turn.category not in _PROTECTED_CATEGORIES:
            entry["user"] = turn.user
            entry["assistant"] = turn.assistant
            return entry

        payload = json.dumps(
            {"user": turn.user, "assistant": turn.assistant},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
        protector = self._get_protector()
        entry["protected"] = {
            "scheme": protector.scheme,
            "payload": protector.protect(payload),
        }
        return entry

    def _decode_entry(
        self,
        entry: dict[str, object],
        *,
        include_secrets: bool,
    ) -> ConversationTurn | None:
        category = _normalize_category(str(entry.get("category", ""))) or classify_turn(
            str(entry.get("user", "")),
            str(entry.get("assistant", "")),
        )
        if category == "Secretos" and not include_secrets:
            return None

        protected = entry.get("protected")
        if isinstance(protected, dict):
            scheme = str(protected.get("scheme", ""))
            protector = self._get_protector()
            if scheme != protector.scheme:
                raise DataProtectionError("Conversation data uses an unsupported protection scheme.")
            raw = protector.unprotect(str(protected.get("payload", "")))
            decoded = json.loads(raw.decode("utf-8"))
            user = str(decoded.get("user", ""))
            assistant = str(decoded.get("assistant", ""))
        else:
            user = str(entry.get("user", ""))
            assistant = str(entry.get("assistant", ""))

        return ConversationTurn(
            user=user,
            assistant=assistant,
            time=str(entry.get("time", "")),
            category=category,
            cognitively_retrievable=bool(entry.get("cognitively_retrievable", True)),
            record_id=str(entry.get("record_id", "")),
            causal_status=str(entry.get("causal_status", "resolved")),
            rejection_reasons=tuple(
                str(item) for item in entry.get("rejection_reasons", [])
            ),
        )

    def _get_protector(self) -> DataProtector:
        if self._protector is None:
            self._protector = default_data_protector()
        return self._protector

    def _read_turns(
        self,
        path: Path,
        *,
        include_secrets: bool,
        strict_protection: bool | None = None,
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        unavailable = 0
        strict = include_secrets if strict_protection is None else strict_protection
        for entry in self._read_raw_jsonl(path):
            try:
                turn = self._decode_entry(entry, include_secrets=include_secrets)
            except DataProtectionError:
                unavailable += 1
                if strict:
                    raise
                continue
            if turn is not None:
                turns.append(turn)
        self._unavailable_protected_entries = unavailable
        return turns

    def _read_raw_jsonl(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        entries: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                entries.append(item)
        return entries

    def _append_json_line(self, path: Path, entry: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")

    def _append_classified(self, category: Category, moment: str) -> None:
        data = self._load_classified()
        data[category].append(_category_metadata(moment, category))
        _atomic_write_json(
            self.classified_path,
            {"version": _SCHEMA_VERSION, "categories": data},
        )

    def _load_classified(self) -> dict[str, list[dict[str, object]]]:
        if not self.classified_path.exists():
            return _empty_categories()
        try:
            data = json.loads(self.classified_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return _empty_categories()
        if not isinstance(data, dict):
            return _empty_categories()
        categories = data.get("categories") if data.get("version") == _SCHEMA_VERSION else data
        if not isinstance(categories, dict):
            return _empty_categories()
        return {
            name: [item for item in categories.get(name, []) if isinstance(item, dict)]
            for name in ("Secretos", "Personal", "Normal")
        }

    def _create_versioned_backup(self, paths: tuple[Path, ...]) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = self.migration_backup_dir / f"{stamp}-{uuid4().hex[:8]}"
        destination.mkdir(parents=True, exist_ok=False)
        manifest: list[dict[str, object]] = []
        protector = self._get_protector()
        for path in paths:
            raw = path.read_bytes()
            target = destination / f"{path.name}.protected.json"
            _atomic_write_json(
                target,
                {
                    "version": 1,
                    "source_name": path.name,
                    "scheme": protector.scheme,
                    "payload": protector.protect(raw),
                },
            )
            manifest.append(
                {
                    "file": path.name,
                    "protected_file": target.name,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        _atomic_write_json(
            destination / "manifest.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": manifest,
            },
        )
        return destination

    def _commit_migration(
        self,
        archive_entries: list[dict[str, object]],
        cognitive_entries: list[dict[str, object]],
        categories: dict[str, list[dict[str, object]]],
        backup: Path,
    ) -> None:
        _write_jsonl_atomic(self.archive_path, archive_entries)
        _write_jsonl_atomic(self.cognitive_path, cognitive_entries)
        _atomic_write_json(
            self.classified_path,
            {"version": _SCHEMA_VERSION, "categories": categories},
        )
        _atomic_write_json(
            self.migration_marker_path,
            {
                "version": _SCHEMA_VERSION,
                "migrated_at": datetime.now(timezone.utc).isoformat(),
                "backup": backup.name,
                "turn_count": len(archive_entries),
            },
        )

    def _commit_repair(
        self,
        archive_entries: list[dict[str, object]],
        cognitive_entries: list[dict[str, object]],
        categories: dict[str, list[dict[str, object]]],
        protected_backup: Path,
        safety_backup: Path,
    ) -> None:
        _write_jsonl_atomic(self.archive_path, archive_entries)
        _write_jsonl_atomic(self.cognitive_path, cognitive_entries)
        _atomic_write_json(
            self.classified_path,
            {"version": _SCHEMA_VERSION, "categories": categories},
        )
        marker: dict[str, object] = {}
        if self.migration_marker_path.exists():
            try:
                loaded = json.loads(self.migration_marker_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    marker = loaded
            except (OSError, UnicodeError, json.JSONDecodeError):
                marker = {}
        marker.update(
            {
                "version": _SCHEMA_VERSION,
                "last_repair_at": datetime.now(timezone.utc).isoformat(),
                "repair_source": protected_backup.parent.name,
                "repair_safety_backup": safety_backup.name,
                "turn_count": len(archive_entries),
            }
        )
        _atomic_write_json(self.migration_marker_path, marker)

    def _prepare_legacy_payload(
        self,
        payload: bytes,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, list[dict[str, object]]],
    ]:
        archive_entries: list[dict[str, object]] = []
        cognitive_entries: list[dict[str, object]] = []
        categories = _empty_categories()
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            user = str(raw.get("user", ""))
            assistant = str(raw.get("assistant", ""))
            category = _normalize_category(str(raw.get("category", ""))) or classify_turn(
                user,
                assistant,
            )
            turn = ConversationTurn(
                user=user,
                assistant=assistant,
                time=str(raw.get("time", "")) or datetime.now(timezone.utc).isoformat(),
                category=category,
            )
            encoded = self._encode_entry(turn)
            archive_entries.append(encoded)
            categories[category].append(_category_metadata(turn.time, category))
            if category != "Secretos":
                cognitive_entries.append(encoded)
        return archive_entries, cognitive_entries, categories

    def _verify_protected_entries(self, entries: list[dict[str, object]]) -> None:
        for entry in entries:
            if isinstance(entry.get("protected"), dict):
                self._decode_entry(entry, include_secrets=True)

    def _restore_versioned_backup(
        self,
        backup: Path,
        targets: tuple[Path, ...],
    ) -> None:
        protected_by_name: dict[str, Path] = {}
        for candidate in backup.glob("*.protected.json"):
            envelope = json.loads(candidate.read_text(encoding="utf-8"))
            source_name = str(envelope.get("source_name", ""))
            if source_name:
                protected_by_name[source_name] = candidate
        protector = self._get_protector()
        for target in targets:
            candidate = protected_by_name.get(target.name)
            if candidate is None:
                continue
            envelope = json.loads(candidate.read_text(encoding="utf-8"))
            if str(envelope.get("scheme", "")) != protector.scheme:
                raise DataProtectionError("Conversation safety backup uses an unsupported scheme.")
            _atomic_write_bytes(
                target,
                protector.unprotect(str(envelope.get("payload", ""))),
            )

    def _retire_legacy_plaintext(self) -> None:
        for path in (self.legacy_log_path, self.legacy_classified_path):
            if path is None or path.resolve() in {
                self.archive_path.resolve(),
                self.classified_path.resolve(),
            }:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # The encrypted v2 archive and protected backup are already durable.
                continue

    def _remove_incomplete_migration(self) -> None:
        for path in (
            self.archive_path,
            self.cognitive_path,
            self.classified_path,
            self.migration_marker_path,
        ):
            path.unlink(missing_ok=True)


def classify_turn(user_text: str, assistant_text: str) -> Category:
    categories = (classify_text(user_text), classify_text(assistant_text))
    if "Secretos" in categories:
        return "Secretos"
    if "Personal" in categories:
        return "Personal"
    return "Normal"


def classify_text(text: str) -> Category:
    lowered = text.lower()
    secret_markers = [
        "secreto",
        "no le digas",
        "contraseña",
        "contrasena",
        "password",
        "token",
        "api key",
        "apikey",
        "clave",
        "privado",
    ]
    if any(marker in lowered for marker in secret_markers):
        return "Secretos"

    personal_markers = [
        "me llamo",
        "mi nombre",
        "soy ",
        "yo ",
        "me gusta",
        "prefiero",
        "preferencia",
        "odio",
        "quiero",
        "mi ",
        "mis ",
        "familia",
        "mamá",
        "mama",
        "papá",
        "papa",
        "polita",
    ]
    if any(marker in lowered for marker in personal_markers):
        return "Personal"
    return "Normal"


def _normalize_category(value: str) -> Category | None:
    normalized = value.strip().casefold()
    return {
        "normal": "Normal",
        "personal": "Personal",
        "secretos": "Secretos",
        "secreto": "Secretos",
        "secret": "Secretos",
    }.get(normalized)


def _empty_categories() -> dict[str, list[dict[str, object]]]:
    return {"Secretos": [], "Personal": [], "Normal": []}


def _category_metadata(moment: str, category: str) -> dict[str, object]:
    return {
        "time": moment,
        "protected": category in _PROTECTED_CATEGORIES,
    }


def _entry_datetime(entry: dict[str, object]) -> datetime:
    value = str(entry.get("time", ""))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.max.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_jsonl_atomic(path: Path, entries: list[dict[str, object]]) -> None:
    payload = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in entries)
    _atomic_write_text(path, payload)


def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    payload = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, payload)


def _atomic_write_text(path: Path, payload: str) -> None:
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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
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


def _tokenize(text: str) -> list[str]:
    return [
        part
        for part in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(part) > 2
    ]
