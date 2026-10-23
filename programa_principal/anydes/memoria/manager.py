from __future__ import annotations

import os
from dataclasses import dataclass

from anydes.memoria.local_memory import JsonMemoryStore
from anydes.memoria.conversation_memory import (
    ConversationMemory,
    ConversationRetentionPolicy,
    classify_text,
)
from anydes.memoria.secure_store import ProtectedMemoryStore
from anydes.memoria.data_protection import default_data_protector
from anydes.nucleo.config import Config

SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "clave_api",
    "contrasena",
    "contraseña",
    "password",
    "secret",
    "secreto",
    "token",
}


@dataclass(frozen=True)
class MemoryResult:
    ok: bool
    message: str


class MemoryManager:
    def __init__(
        self,
        permanent: JsonMemoryStore,
        temporary: JsonMemoryStore,
        protected: ProtectedMemoryStore,
        conversation: ConversationMemory,
    ) -> None:
        self.permanent = permanent
        self.temporary = temporary
        self.protected = protected
        self.conversation = conversation
        self._backfill_origin_from_log()

    def _backfill_origin_from_log(self) -> None:
        if self.permanent.get("primera_palabra_usuario"):
            return
        first = self.conversation.first_user_turn()
        if not first:
            return
        self.permanent.set("primera_palabra_usuario", first["user"][:500])
        if first.get("time"):
            self.permanent.set("nacimiento_conversacion", first["time"])

    @classmethod
    def from_config(cls, config: Config) -> "MemoryManager":
        passphrase = os.getenv("ANYDES_MEMORY_PASSPHRASE")
        protector = default_data_protector(passphrase=passphrase)
        conversation = ConversationMemory(
            config.conversation_archive_path,
            config.conversation_categories_path,
            cognitive_path=config.conversation_cognitive_path,
            legacy_log_path=config.conversation_log_path,
            legacy_classified_path=config.classified_memory_path,
            migration_backup_dir=config.conversation_migration_backup_dir,
            migration_marker_path=config.conversation_migration_marker_path,
            retention_policy=ConversationRetentionPolicy(
                max_age_days=config.conversation_retention_days or None,
                max_turns=config.conversation_retention_max_turns or None,
            ),
            protector=protector,
        )
        conversation.migrate_legacy()
        return cls(
            permanent=JsonMemoryStore(config.permanent_memory_path),
            temporary=JsonMemoryStore(config.temporary_memory_path),
            protected=ProtectedMemoryStore(
                config.protected_memory_path,
                passphrase,
                protector=protector,
            ),
            conversation=conversation,
        )

    def set_permanent(self, key: str, value: str) -> MemoryResult:
        if self._is_sensitive(key) or self._is_sensitive_value(value):
            return MemoryResult(False, "Ese dato parece sensible. No lo guardo en memoria permanente normal.")
        self.permanent.set(key, value)
        return MemoryResult(True, f"Guardado en memoria permanente: {key}")

    def set_temporary(self, key: str, value: str) -> MemoryResult:
        if self._is_sensitive(key) or self._is_sensitive_value(value):
            return MemoryResult(False, "Ese dato parece sensible. No lo guardo en memoria temporal normal.")
        self.temporary.set(key, value)
        return MemoryResult(True, f"Guardado en memoria temporal: {key}")

    def set_protected(self, key: str, value: str) -> MemoryResult:
        if not self.protected.enabled:
            return MemoryResult(
                False,
                "Memoria protegida desactivada. Define ANYDES_MEMORY_PASSPHRASE solo en tu sesion si quieres usarla.",
            )
        self.protected.set(key, value)
        return MemoryResult(True, f"Guardado en memoria protegida: {key}")

    def get_protected(self, key: str) -> MemoryResult:
        if not self.protected.enabled:
            return MemoryResult(
                False,
                "Memoria protegida desactivada. Define ANYDES_MEMORY_PASSPHRASE solo en tu sesion si quieres usarla.",
            )
        value = self.protected.get(key)
        if value is None:
            return MemoryResult(False, f"No tengo memoria protegida para '{key}'.")
        return MemoryResult(True, value)

    def profile_summary(self) -> str:
        permanent = self.permanent.all()
        temporary = self.temporary.all()
        protected_keys = self.protected.keys()
        classified = self.conversation.counts()
        return "\n".join(
            [
                "Memoria de Anydes:",
                f"- Permanente: {len(permanent)} dato(s).",
                f"- Temporal: {len(temporary)} dato(s).",
                f"- Protegida: {len(protected_keys)} clave(s), valores ocultos.",
                f"- Conversacion Normal: {classified.get('Normal', 0)} dato(s).",
                f"- Conversacion Personal: {classified.get('Personal', 0)} dato(s).",
                f"- Conversacion Secretos: {classified.get('Secretos', 0)} dato(s), separados.",
            ]
        )

    def safe_context(self) -> str:
        permanent = self.permanent.all()
        temporary = self.temporary.all()
        lines = [self.profile_summary()]

        if permanent:
            lines.append("Datos permanentes no sensibles:")
            for key, value in sorted(permanent.items()):
                if not self._is_sensitive(key) and not self._is_sensitive_value(value):
                    lines.append(f"- {key}: {value}")

        if temporary:
            lines.append("Contexto temporal no sensible:")
            for key, value in sorted(temporary.items()):
                if not self._is_sensitive(key) and not self._is_sensitive_value(value):
                    lines.append(f"- {key}: {value}")

        lines.append(self.conversation.context(include_secrets=False))
        if self.conversation.unavailable_protected_entries:
            lines.append(
                "Some protected personal conversation records are temporarily unavailable. "
                "Do not infer or disclose their contents."
            )

        return "\n".join(lines)

    def relevant_context(self, query: str) -> str:
        return self.conversation.relevant_context(query, include_secrets=False)

    def search_conversation(self, query: str, limit: int = 8) -> str:
        matches = self.conversation.search(query, limit=limit, include_secrets=False)
        if not matches:
            return "No encontre conversaciones anteriores que coincidan."
        lines = ["Coincidencias en conversaciones anteriores:"]
        for item in matches:
            lines.append(f"- {item['time']} [{item.get('category', 'Normal')}]")
            lines.append(f"  Tu: {item['user'][:300]}")
            lines.append(f"  Anydes: {item['assistant'][:300]}")
        return "\n".join(lines)

    def recent_conversation_summary(self) -> str:
        return self.conversation.recent_summary()

    def origin_summary(self) -> str:
        primera = self.permanent.get("primera_palabra_usuario")
        if primera and self._is_sensitive_value(primera):
            primera = None
        nacimiento = self.permanent.get("nacimiento_conversacion")
        first_turn = self.conversation.first_user_turn()
        lines: list[str] = []
        if primera:
            lines.append(f"La primera palabra o mensaje que recuerdo de ti fue: «{primera}».")
            if nacimiento:
                lines.append(f"Lo registré el {nacimiento}.")
        elif first_turn:
            lines.append(
                f"En el historial, lo primero que dijiste fue: «{first_turn['user']}» "
                f"({first_turn.get('time', 'sin fecha')})."
            )
        else:
            lines.append("Todavia no tengo registrada nuestra primera conversacion.")
        return "\n".join(lines)

    def conversation_history_pairs(self, limit: int = 10) -> list[tuple[str, str]]:
        turns = self.conversation.recent_turns(limit=limit)
        pairs: list[tuple[str, str]] = []
        for turn in turns:
            user = turn.user.strip()
            if not user or user == "[iniciativa automatica]":
                continue
            pairs.append((user, turn.assistant))
        return pairs

    def pending_conversation_inputs(self, limit: int = 6):
        return self.conversation.pending_turns(limit=limit)

    def record_conversation_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        cognitively_retrievable: bool = True,
    ) -> None:
        self.conversation.record_turn(
            user_text,
            assistant_text,
            cognitively_retrievable=cognitively_retrievable,
        )
        self._ensure_origin_milestone(user_text)

    def record_rejected_conversation_turn(
        self,
        user_text: str,
        visible_recovery: str,
        *,
        rejection_reasons: tuple[str, ...] = (),
    ) -> None:
        self.conversation.record_rejected_turn(
            user_text,
            visible_recovery,
            rejection_reasons=rejection_reasons,
        )
        self._ensure_origin_milestone(user_text)

    def export_conversation_history(self, destination, *, include_secrets: bool = False) -> int:
        return self.conversation.export(destination, include_secrets=include_secrets)

    def export_protected_conversation_archive(self, destination) -> int:
        return self.conversation.export_protected_archive(destination)

    def delete_conversation_history(self, *, create_backup: bool = False) -> str:
        return self.conversation.delete_all(create_backup=create_backup)

    def apply_conversation_retention(self) -> int:
        return self.conversation.apply_retention()

    def _ensure_origin_milestone(self, user_text: str) -> None:
        user = user_text.strip()
        if not user or user == "[iniciativa automatica]":
            return
        if self._is_sensitive_value(user):
            return
        if self.permanent.get("primera_palabra_usuario"):
            return
        from datetime import datetime, timezone

        self.permanent.set("primera_palabra_usuario", user[:500])
        self.permanent.set("nacimiento_conversacion", datetime.now(timezone.utc).isoformat())

    def _is_sensitive(self, key: str) -> bool:
        normalized = key.strip().lower()
        return any(part in normalized for part in SENSITIVE_KEY_PARTS)

    def _is_sensitive_value(self, value: object) -> bool:
        return classify_text(str(value)) == "Secretos"
