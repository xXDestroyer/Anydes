from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import NamedTemporaryFile

from anydes.memoria.data_protection import (
    DataProtectionError,
    DataProtector,
    default_data_protector,
)
from anydes.nucleo.config import Config
from anydes.nucleo.path_locks import shared_path_lock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LanguageConvention:
    expression: str
    meaning: str
    tone: str = ""
    examples: tuple[str, ...] = ()
    confidence: float = 1.0
    source: str = "user_confirmed"
    created_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True)
class RelationalPreference:
    key: str
    value: str
    confidence: float = 1.0
    source: str = "user_confirmed"
    created_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True)
class SpatialIdentity:
    digital_home: str = "la habitación digital de Anydes"
    execution_place: str = "el dispositivo donde Anydes está ejecutándose"
    physical_location_available: bool = False
    user_location_is_separate: bool = True
    mobility: str = (
        "puede habitar distintos dispositivos en el futuro sin inventar una dirección física"
    )


@dataclass(frozen=True)
class LanguageResolution:
    matched: bool
    expression: str = ""
    meaning: str = ""
    confidence: float = 0.0


class RealtimeIdentityProfileStore:
    """Protected persistence for lived identity preferences used by Realtime."""

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
        self._payload = self._load()

    @classmethod
    def from_config(cls, config: Config) -> "RealtimeIdentityProfileStore":
        return cls(config.realtime_identity_profile_path)

    def ensure_defaults(self) -> None:
        changed = False
        with self._state_lock:
            conventions = self._payload.setdefault("language_conventions", {})
            if "chi" not in conventions:
                conventions["chi"] = asdict(
                    LanguageConvention(
                        expression="chi",
                        meaning="sí",
                        tone="tierno y cotidiano",
                        examples=("chi", "chiii"),
                        source="user_confirmed_project_requirement",
                    )
                )
                changed = True
            relationship = self._payload.setdefault("relationship_preferences", {})
            if "warm_closeness" not in relationship:
                relationship["warm_closeness"] = asdict(
                    RelationalPreference(
                        key="warm_closeness",
                        value=(
                            "DemoUser prefiere que Anydes sea más dulce, cercana y "
                            "cariñosa con él, de forma natural y sin exagerar."
                        ),
                        source="user_confirmed_project_requirement",
                    )
                )
                changed = True
        if changed:
            self.persist()

    def conventions(self) -> tuple[LanguageConvention, ...]:
        with self._state_lock:
            raw = dict(self._payload.get("language_conventions", {}))
        values = []
        for item in raw.values():
            if not isinstance(item, dict):
                continue
            values.append(
                LanguageConvention(
                    expression=str(item.get("expression") or ""),
                    meaning=str(item.get("meaning") or ""),
                    tone=str(item.get("tone") or ""),
                    examples=tuple(str(value) for value in item.get("examples", [])),
                    confidence=float(item.get("confidence", 1.0)),
                    source=str(item.get("source") or "user_confirmed"),
                    created_at=str(item.get("created_at") or _utc_now()),
                )
            )
        return tuple(value for value in values if value.expression and value.meaning)

    def relationship_preferences(self) -> tuple[RelationalPreference, ...]:
        with self._state_lock:
            raw = dict(self._payload.get("relationship_preferences", {}))
        values = []
        for key, item in raw.items():
            if not isinstance(item, dict):
                continue
            values.append(
                RelationalPreference(
                    key=str(item.get("key") or key),
                    value=str(item.get("value") or ""),
                    confidence=float(item.get("confidence", 1.0)),
                    source=str(item.get("source") or "user_confirmed"),
                    created_at=str(item.get("created_at") or _utc_now()),
                )
            )
        return tuple(value for value in values if value.key and value.value)

    def spatial_identity(self) -> SpatialIdentity:
        return SpatialIdentity()

    def register_language_convention(
        self,
        convention: LanguageConvention,
    ) -> None:
        key = _normalize(convention.expression)
        if not key or not convention.meaning.strip():
            return
        with self._state_lock:
            conventions = self._payload.setdefault("language_conventions", {})
            conventions[key] = asdict(convention)
        self.persist()

    def resolve_language(self, text: str) -> LanguageResolution:
        target = _normalize(text)
        if not target or len(target.split()) > 5:
            return LanguageResolution(False)
        best: tuple[float, LanguageConvention] | None = None
        for convention in self.conventions():
            candidates = (convention.expression, *convention.examples)
            for candidate in candidates:
                normalized = _normalize(candidate)
                score = SequenceMatcher(None, target, normalized).ratio()
                if target == normalized:
                    score = 1.0
                if best is None or score > best[0]:
                    best = (score, convention)
        if best is None or best[0] < 0.82:
            return LanguageResolution(False)
        return LanguageResolution(
            True,
            expression=best[1].expression,
            meaning=best[1].meaning,
            confidence=min(best[0], best[1].confidence),
        )

    def relational_prompt(self) -> str:
        values = self.relationship_preferences()
        return " ".join(item.value for item in values[:2])

    def persist(self) -> None:
        with self._state_lock:
            payload = json.dumps(
                self._payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        envelope = {
            "schema_version": self._version,
            "scheme": self._protector.scheme,
            "payload": self._protector.protect(payload),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def _load(self) -> dict[str, object]:
        default = {
            "schema_version": self._version,
            "language_conventions": {},
            "relationship_preferences": {},
        }
        if not self.path.exists():
            return default
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                return default
            if str(envelope.get("scheme") or "") != self._protector.scheme:
                return default
            raw = self._protector.unprotect(str(envelope.get("payload") or ""))
            payload = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            DataProtectionError,
        ):
            return default
        return payload if isinstance(payload, dict) else default


def extract_language_convention(text: str) -> LanguageConvention | None:
    clean = " ".join(text.strip().split())
    if not clean or len(clean) > 220:
        return None
    patterns = (
        r"(?i)\bcuando\s+digo\s+[\"']?(.{1,35}?)[\"']?\s+"
        r"(?:significa|quiero\s+decir)\s+[\"']?(.{1,55}?)[\"']?[.!]?$",
        r"(?i)^[\"']?(.{1,35}?)[\"']?\s+es\s+"
        r"(?:una\s+)?forma(?:\s+\w+){0,4}\s+de\s+decir\s+"
        r"[\"']?(.{1,55}?)[\"']?[.!]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match is None:
            continue
        expression = match.group(1).strip(" \"'")
        meaning = match.group(2).strip(" \"'")
        if expression and meaning and _normalize(expression) != _normalize(meaning):
            return LanguageConvention(
                expression=expression,
                meaning=meaning,
                source="user_confirmed_conversation",
            )
    return None


def _normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", folded).split())
