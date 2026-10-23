from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Literal

from anydes.memoria.data_protection import (
    DataProtectionError,
    DataProtector,
    default_data_protector,
)
from anydes.nucleo.path_locks import shared_path_lock


EventAuthor = Literal["user", "anydes", "system", "tool"]


@dataclass(frozen=True)
class ConversationEvent:
    """One literal, chronological event. It is separate from cognitive memory."""

    event_id: str
    author: EventAuthor
    exact_text: str
    occurred_at: str
    timezone: str
    local_date: str
    session_id: str
    event_type: str
    topic: str = ""
    entities: tuple[str, ...] = ()
    related_event_ids: tuple[str, ...] = ()
    tool_result: dict[str, object] = field(default_factory=dict)
    provenance: str = "realtime-visible-conversation"
    category: str = "Normal"

    @classmethod
    def create(
        cls,
        *,
        author: EventAuthor,
        exact_text: str,
        session_id: str,
        event_type: str,
        topic: str = "",
        entities: Iterable[str] = (),
        related_event_ids: Iterable[str] = (),
        tool_result: dict[str, object] | None = None,
        provenance: str = "realtime-visible-conversation",
        category: str = "Normal",
        moment: datetime | None = None,
        local_timezone: tzinfo | None = None,
    ) -> "ConversationEvent":
        local = moment or datetime.now().astimezone()
        if local.tzinfo is None:
            local = local.replace(tzinfo=local_timezone or timezone.utc)
        elif local_timezone is not None:
            local = local.astimezone(local_timezone)
        zone_name = getattr(local.tzinfo, "key", None) or str(local.tzinfo or "UTC")
        return cls(
            event_id=f"conversation-event:{uuid.uuid4().hex}",
            author=author,
            exact_text=exact_text.strip(),
            occurred_at=local.astimezone(timezone.utc).isoformat(),
            timezone=zone_name,
            local_date=local.date().isoformat(),
            session_id=session_id,
            event_type=event_type,
            topic=topic.strip(),
            entities=tuple(item.strip() for item in entities if item.strip()),
            related_event_ids=tuple(
                item.strip() for item in related_event_ids if item.strip()
            ),
            tool_result=dict(tool_result or {}),
            provenance=provenance,
            category=category if category in {"Normal", "Personal", "Secretos"} else "Normal",
        )


@dataclass(frozen=True)
class JournalQuery:
    kind: str
    session_id: str = ""
    local_date: str = ""
    author: EventAuthor | str = "user"
    topic: str = ""
    limit: int = 8
    include_personal: bool = False
    include_secrets: bool = False


class ConversationJournal:
    """Encrypted append-only history with deterministic, bounded queries."""

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
        self._lock = shared_path_lock(self.path)
        self._state_lock = threading.RLock()
        self._events = list(self._load())

    def remember(self, events: Iterable[ConversationEvent]) -> tuple[ConversationEvent, ...]:
        items = tuple(events)
        with self._state_lock:
            known = {event.event_id for event in self._events}
            self._events.extend(event for event in items if event.event_id not in known)
        return items

    def persist(self, events: Iterable[ConversationEvent]) -> None:
        items = tuple(events)
        if not items:
            return
        lines = [self._encode(event) for event in items]
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(json.dumps(line, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def append(self, events: Iterable[ConversationEvent]) -> tuple[ConversationEvent, ...]:
        items = self.remember(events)
        self.persist(items)
        return items

    def events(
        self,
        *,
        include_personal: bool = False,
        include_secrets: bool = False,
    ) -> tuple[ConversationEvent, ...]:
        with self._state_lock:
            items = tuple(self._events)
        return tuple(
            event
            for event in items
            if _category_allowed(
                event.category,
                include_personal=include_personal,
                include_secrets=include_secrets,
            )
        )

    def query(self, query: JournalQuery) -> tuple[ConversationEvent, ...]:
        items = list(
            self.events(
                include_personal=query.include_personal,
                include_secrets=query.include_secrets,
            )
        )
        if query.author:
            items = [event for event in items if event.author == query.author]
        if query.session_id:
            items = [event for event in items if event.session_id == query.session_id]
        if query.local_date:
            items = [event for event in items if event.local_date == query.local_date]
        if query.topic:
            items = [
                event
                for event in items
                if _topic_matches(
                    query.topic,
                    " ".join((event.topic, event.exact_text, *event.entities)),
                )
            ]

        kind = query.kind
        if kind in {"first", "first_session", "first_day"}:
            return tuple(items[:1])
        if kind == "last":
            return tuple(items[-1:])
        if kind == "recent":
            return tuple(items[-max(1, query.limit) :])
        if kind == "questions":
            items = [event for event in items if event.event_type == "question"]
        elif kind == "assertions":
            items = [event for event in items if event.event_type == "assertion"]
        elif kind == "user_facts":
            items = [
                event
                for event in items
                if event.event_type in {"assertion", "user_preference", "correction"}
            ]
        elif kind == "corrections":
            items = [event for event in items if event.event_type == "correction"]
        elif kind == "preferences":
            items = [event for event in items if event.event_type == "user_preference"]
        elif kind == "commitments":
            items = [
                event
                for event in items
                if event.event_type in {"commitment", "promise", "task_proposed"}
            ]
        elif kind == "repeated_questions":
            questions = [event for event in items if event.event_type == "question"]
            counts: dict[str, int] = {}
            for event in questions:
                key = _normalize(event.exact_text)
                counts[key] = counts.get(key, 0) + 1
            items = [
                event
                for event in questions
                if counts.get(_normalize(event.exact_text), 0) > 1
            ]
        elif kind == "topics":
            seen: set[str] = set()
            selected: list[ConversationEvent] = []
            for event in reversed(items):
                key = _normalize(event.topic)
                if not key or key in seen:
                    continue
                seen.add(key)
                selected.append(event)
            items = list(reversed(selected))
        return tuple(items[-max(1, query.limit) :])

    def session_ids(
        self,
        *,
        include_personal: bool,
        include_secrets: bool = False,
    ) -> tuple[str, ...]:
        """Return distinct session IDs in chronological order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for event in self.events(
            include_personal=include_personal,
            include_secrets=include_secrets,
        ):
            if not event.session_id or event.session_id in seen:
                continue
            seen.add(event.session_id)
            ordered.append(event.session_id)
        return tuple(ordered)

    def previous_session_id(
        self,
        current_session_id: str,
        *,
        include_personal: bool,
        include_secrets: bool = False,
    ) -> str:
        """Find the latest recorded session before the current one."""

        ordered = self.session_ids(
            include_personal=include_personal,
            include_secrets=include_secrets,
        )
        previous = ""
        for session_id in ordered:
            if session_id == current_session_id:
                return previous
            previous = session_id
        return previous if previous != current_session_id else ""

    def previous_session_events(
        self,
        current_session_id: str,
        *,
        limit: int = 8,
        include_personal: bool,
        include_secrets: bool = False,
    ) -> tuple[ConversationEvent, ...]:
        session_id = self.previous_session_id(
            current_session_id,
            include_personal=include_personal,
            include_secrets=include_secrets,
        )
        if not session_id:
            return ()
        return self.query(
            JournalQuery(
                kind="recent",
                session_id=session_id,
                author="",
                limit=limit,
                include_personal=include_personal,
                include_secrets=include_secrets,
            )
        )

    def first_user_message_of_day(
        self,
        target_date: date,
        *,
        include_personal: bool,
        include_secrets: bool = False,
    ) -> ConversationEvent | None:
        matches = self.query(
            JournalQuery(
                kind="first_day",
                local_date=target_date.isoformat(),
                author="user",
                include_personal=include_personal,
                include_secrets=include_secrets,
            )
        )
        return matches[0] if matches else None

    def first_user_message_of_session(
        self,
        session_id: str,
        *,
        include_personal: bool,
        include_secrets: bool = False,
    ) -> ConversationEvent | None:
        matches = self.query(
            JournalQuery(
                kind="first_session",
                session_id=session_id,
                author="user",
                include_personal=include_personal,
                include_secrets=include_secrets,
            )
        )
        return matches[0] if matches else None

    def _encode(self, event: ConversationEvent) -> dict[str, object]:
        raw = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {
            "version": self._version,
            "scheme": self._protector.scheme,
            "event_id": event.event_id,
            "occurred_at": event.occurred_at,
            "payload": self._protector.protect(raw),
        }

    def _load(self) -> tuple[ConversationEvent, ...]:
        if not self.path.exists():
            return ()
        loaded: list[ConversationEvent] = []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                envelope = json.loads(line)
                raw = self._protector.unprotect(str(envelope["payload"]))
                payload = json.loads(raw.decode("utf-8"))
                payload["entities"] = tuple(payload.get("entities") or ())
                payload["related_event_ids"] = tuple(
                    payload.get("related_event_ids") or ()
                )
                payload["tool_result"] = dict(payload.get("tool_result") or {})
                loaded.append(ConversationEvent(**payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, DataProtectionError):
                # A damaged trailing event never invalidates the earlier journal.
                continue
        return tuple(loaded)


def _category_allowed(
    category: str,
    *,
    include_personal: bool,
    include_secrets: bool,
) -> bool:
    if category == "Secretos":
        return include_secrets
    if category == "Personal":
        return include_personal
    return True


def _normalize(text: str) -> str:
    import unicodedata

    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(folded.split())


_TOPIC_STOP_WORDS = frozenset(
    {
        "a",
        "acerca",
        "al",
        "algo",
        "de",
        "del",
        "el",
        "en",
        "la",
        "las",
        "lo",
        "los",
        "mi",
        "mis",
        "que",
        "sobre",
        "te",
        "un",
        "una",
        "y",
        "yo",
    }
)


def _topic_matches(query: str, candidate: str) -> bool:
    needle = _normalize(query)
    haystack = _normalize(candidate)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    query_tokens = {
        token
        for token in needle.split()
        if len(token) > 2 and token not in _TOPIC_STOP_WORDS
    }
    if not query_tokens:
        return False
    candidate_tokens = set(haystack.split())
    matched = len(query_tokens & candidate_tokens)
    required = 1 if len(query_tokens) == 1 else max(2, len(query_tokens) - 1)
    return matched >= required
