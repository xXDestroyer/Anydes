from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Literal

from anydes.memoria.data_protection import (
    DataProtectionError,
    DataProtector,
    default_data_protector,
)
from anydes.nucleo.config import Config
from anydes.nucleo.path_locks import shared_path_lock
from anydes.realtime.journal import ConversationJournal


InsightKind = Literal["pattern", "inference", "hypothesis"]
InsightStatus = Literal["provisional", "confirmed", "contradicted"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DerivedInsight:
    """A traceable, revisable interpretation about the user."""

    insight_id: str
    subject: str
    kind: InsightKind
    statement: str
    evidence_event_ids: tuple[str, ...]
    confidence: float
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    status: InsightStatus = "provisional"
    contradiction_event_ids: tuple[str, ...] = ()
    provenance: str = "authorized-conversation-evidence"


class DerivedInsightStore:
    """Encrypted store; it never creates interpretations by itself."""

    _version = 1

    def __init__(
        self,
        path: Path,
        journal: ConversationJournal,
        *,
        protector: DataProtector | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = journal
        self._protector = protector or default_data_protector(
            passphrase=os.getenv("ANYDES_MEMORY_PASSPHRASE")
        )
        self._lock = shared_path_lock(self.path)
        self._state_lock = threading.RLock()
        self._items = list(self._load())

    @classmethod
    def from_config(
        cls,
        config: Config,
        journal: ConversationJournal,
    ) -> "DerivedInsightStore":
        return cls(config.realtime_derived_insights_path, journal)

    def propose(
        self,
        *,
        statement: str,
        kind: InsightKind,
        evidence_event_ids: Iterable[str],
        confidence: float,
        subject: str = "user",
    ) -> DerivedInsight:
        evidence = tuple(dict.fromkeys(item for item in evidence_event_ids if item))
        self._validate_evidence(evidence)
        if kind not in {"pattern", "inference", "hypothesis"}:
            raise ValueError("Unsupported derived insight kind.")
        if not statement.strip():
            raise ValueError("A derived insight needs a statement.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        insight = DerivedInsight(
            insight_id=f"derived-insight:{uuid.uuid4().hex}",
            subject=subject.strip() or "user",
            kind=kind,
            statement=statement.strip(),
            evidence_event_ids=evidence,
            confidence=confidence,
        )
        with self._state_lock:
            self._items.append(insight)
        self.persist()
        return insight

    def contradict(
        self,
        insight_id: str,
        *,
        evidence_event_ids: Iterable[str],
    ) -> DerivedInsight:
        evidence = tuple(dict.fromkeys(item for item in evidence_event_ids if item))
        self._validate_evidence(evidence)
        with self._state_lock:
            for index, item in enumerate(self._items):
                if item.insight_id != insight_id:
                    continue
                updated = replace(
                    item,
                    status="contradicted",
                    contradiction_event_ids=tuple(
                        dict.fromkeys((*item.contradiction_event_ids, *evidence))
                    ),
                    updated_at=_utc_now(),
                )
                self._items[index] = updated
                break
            else:
                raise KeyError(insight_id)
        self.persist()
        return updated

    def query(
        self,
        *,
        subject: str = "user",
        status: InsightStatus | str = "provisional",
        limit: int = 6,
    ) -> tuple[DerivedInsight, ...]:
        with self._state_lock:
            items = tuple(self._items)
        selected = [
            item
            for item in items
            if (not subject or item.subject == subject)
            and (not status or item.status == status)
        ]
        return tuple(selected[-max(1, limit) :])

    def persist(self) -> None:
        with self._state_lock:
            raw = json.dumps(
                [asdict(item) for item in self._items],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        envelope = {
            "schema_version": self._version,
            "scheme": self._protector.scheme,
            "payload": self._protector.protect(raw),
        }
        with self._lock:
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

    def _validate_evidence(self, evidence_event_ids: tuple[str, ...]) -> None:
        if not evidence_event_ids:
            raise ValueError("Derived insights require evidence.")
        available = {
            event.event_id
            for event in self.journal.events(
                include_personal=True,
                include_secrets=False,
            )
        }
        missing = set(evidence_event_ids) - available
        if missing:
            raise ValueError("Derived insight evidence is missing or unauthorized.")

    def _load(self) -> tuple[DerivedInsight, ...]:
        if not self.path.exists():
            return ()
        try:
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
        loaded: list[DerivedInsight] = []
        for item in payload if isinstance(payload, list) else ():
            if not isinstance(item, dict):
                continue
            try:
                loaded.append(
                    DerivedInsight(
                        insight_id=str(item["insight_id"]),
                        subject=str(item.get("subject") or "user"),
                        kind=str(item["kind"]),
                        statement=str(item["statement"]),
                        evidence_event_ids=tuple(item.get("evidence_event_ids") or ()),
                        confidence=float(item["confidence"]),
                        created_at=str(item.get("created_at") or _utc_now()),
                        updated_at=str(item.get("updated_at") or _utc_now()),
                        status=str(item.get("status") or "provisional"),
                        contradiction_event_ids=tuple(
                            item.get("contradiction_event_ids") or ()
                        ),
                        provenance=str(
                            item.get("provenance")
                            or "authorized-conversation-evidence"
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(loaded)
