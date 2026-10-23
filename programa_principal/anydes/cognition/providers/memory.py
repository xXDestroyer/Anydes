from __future__ import annotations

from anydes.cognition.contracts import MemoryCategory, MemoryItem, MemorySnapshot
from anydes.memoria.conversation_memory import classify_text
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.config import Config


class MemoryProvider:
    """Adapts the existing MemoryManager state into the new memory contracts."""

    def __init__(self, config: Config, manager: MemoryManager | None = None) -> None:
        self.config = config
        self.manager = manager or MemoryManager.from_config(config)

    def snapshot(self, *, recent_limit: int = 8) -> MemorySnapshot:
        temporary = self.manager.temporary.all()
        permanent = self.manager.permanent.all()
        protected_keys = self.manager.protected.keys()
        history = self.manager.conversation_history_pairs(limit=recent_limit)
        pending = self.manager.pending_conversation_inputs(
            limit=min(4, recent_limit)
        )

        working = tuple(
            MemoryItem(
                id=f"working:{_slug(key)}",
                tier="working",
                category=_category_for_text(f"{key}: {value}"),
                content=f"{key}: {value}",
                importance=0.55,
                metadata={"key": key, "source": "temporary_memory"},
            )
            for key, value in sorted(temporary.items())
        )

        conversation_short = tuple(
            MemoryItem(
                id=f"short:conversation:{index + 1}",
                tier="short",
                category="normal",
                content=f"Usuario: {user}\nAnydes: {assistant}",
                importance=0.45,
                metadata={"source": "conversation_history", "position": index},
            )
            for index, (user, assistant) in enumerate(history)
        )
        pending_short = tuple(
            MemoryItem(
                id=f"short:pending:{_slug(turn.record_id or str(index + 1))}",
                tier="short",
                category=_category_for_text(turn.user),
                content=turn.user,
                importance=max(0.7, 0.96 - index * 0.05),
                last_seen_at=turn.time,
                metadata={
                    "source": "pending_user_observation",
                    "position": index,
                    "record_id": turn.record_id,
                    "causal_status": "pending",
                    "subject": "current_user",
                    "predicate": "unresolved_user_input",
                },
            )
            for index, turn in enumerate(pending)
        )
        short = (*pending_short, *conversation_short)

        long = tuple(
            MemoryItem(
                id=f"long:{_slug(key)}",
                tier="long",
                category=_category_for_text(f"{key}: {value}"),
                content=f"{key}: {value}",
                importance=0.75,
                metadata={"key": key, "source": "permanent_memory"},
            )
            for key, value in sorted(permanent.items())
        )

        protected_summary = (
            f"{len(protected_keys)} protected key(s) hidden."
            if protected_keys
            else "No protected memory keys are stored."
        )

        return MemorySnapshot(
            working=working,
            short=short,
            long=long,
            protected_available=bool(protected_keys),
            protected_summary=protected_summary,
        )


def _slug(value: str) -> str:
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned or "item"


def _category_for_text(text: str) -> MemoryCategory:
    old_category = classify_text(text)
    if old_category == "Personal":
        return "personal"
    if old_category == "Secretos":
        return "secret"
    return "normal"
