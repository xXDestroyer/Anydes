from __future__ import annotations

import json
import re
import threading
import unicodedata
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(text).casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9+#._ -]+", " ", folded).split())


@dataclass(frozen=True)
class MemoryItemV2:
    id: str
    content: str
    source: str
    speaker: str
    timestamp: str = field(default_factory=utc_now)
    topic: str = ""
    entities: tuple[str, ...] = ()
    importance: float = 0.5
    confidence: float = 1.0
    scope: str = "session"
    retention: str = "working"
    memory_type: str = "working"
    expiry: str = "session_end"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Referent:
    id: str
    label: str
    canonical: str
    kind: str
    source: str
    speaker: str
    topic: str
    domain: str
    turn: int
    confidence: float = 1.0


@dataclass(frozen=True)
class ActiveProblem:
    kind: str
    label: str
    values: dict[str, str]
    pending_operation: str
    source: str
    turn: int


@dataclass(frozen=True)
class PendingQuestionV2:
    text: str
    expected: str
    target: str
    turn: int


@dataclass(frozen=True)
class TopicFrame:
    topic: str
    domain: str
    entities: tuple[str, ...]
    summary: str
    turn: int
    importance: float


@dataclass(frozen=True)
class ContinuityResolution:
    kind: str = ""
    original_text: str = ""
    canonical_text: str = ""
    target: str = ""
    confidence: float = 0.0
    source: str = ""
    speaker: str = ""
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkingSnapshotV2:
    active_topic: str = ""
    active_subtopic: str = ""
    current_domain: str = ""
    active_entities: tuple[str, ...] = ()
    active_example: str = ""
    pending_question: str = ""
    pending_task: str = ""
    recent_referents: tuple[str, ...] = ()
    conversation_focus: str = ""
    recent_user_intent: str = ""
    topic_stack: tuple[str, ...] = ()
    resolved_referent: str = ""
    resolved_source: str = ""
    resolved_speaker: str = ""
    turn_count: int = 0


_DOMAIN_PATTERNS = (
    ("Monster Hunter", re.compile(r"\bmonster\s+hunter\b|\bcharge\s+blade\b|\bswitch\s+axe\b", re.I)),
    ("Pokémon", re.compile(r"\bpok[eé]mon\b|\btipo\s+(?:fuego|agua|planta|hielo|roca|tierra|acero|hada|dragon|fantasma)\b", re.I)),
    ("matemáticas", re.compile(r"\b(?:proyecci[oó]n|vector(?:es)?|matriz|derivada|integral|ecuaci[oó]n)\b", re.I)),
    ("física", re.compile(r"\b(?:gravitaci[oó]n|fuerza|masa|aceleraci[oó]n|newton|energ[ií]a)\b", re.I)),
    ("programación", re.compile(r"\b(?:python|c\+\+|c[oó]digo|funci[oó]n|clase|bug|debug)\b", re.I)),
)

_TOPIC_PATTERNS = (
    ("weapons", re.compile(r"\b(?:armas?|hacha\s+cargada|charge\s+blade|switch\s+axe)\b", re.I)),
    ("type weaknesses", re.compile(r"\b(?:d[eé]bil|debilidad(?:es)?|resiste|tipo\s+fuego)\b", re.I)),
    ("vector projection", re.compile(r"\bproyecci[oó]n\b", re.I)),
    ("universal gravitation", re.compile(r"\b(?:ley\s+de\s+)?gravitaci[oó]n\s+universal\b", re.I)),
    ("formula", re.compile(r"\bf[oó]rmula\b", re.I)),
    ("code", re.compile(r"\b(?:c[oó]digo|funci[oó]n|clase|debug|error)\b", re.I)),
)

_ALIASES = {
    "hacha cargada": "Charge Blade",
    "tipo fuego": "tipo Fuego",
    "gravitacion universal": "ley de gravitación universal",
}

_REFERENCE_SIGNAL = re.compile(
    r"\b(?:eso|esto|esa|ese|esas|esos|aquello|lo\s+anterior|lo\s+de\s+antes|"
    r"cu[aá]nto\s+ser[ií]a|la\s+f[oó]rmula|esa\s+f[oó]rmula|esa\s+arma|"
    r"el\s+segundo|la\s+segunda|cu[aá]nt[oa]s?(?:\s+ser[ií]a|\s+tiempo)?|"
    r"ah[ií]|lo\s+que\s+dijiste|"
    r"no\s+es\s+d[eé]bil\s+a\s+nada\s+m[aá]s|volviendo\s+a)\b",
    re.I,
)


class V2SidecarStore:
    """Append-only diagnostic sidecar. It never writes to memory v1."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = {"timestamp": utc_now(), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class WorkingConversationStateV2:
    """Bounded, provenance-aware state for immediate conversational continuity."""

    def __init__(
        self,
        *,
        topic_capacity: int = 6,
        referent_capacity: int = 24,
        telemetry_path: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._topics: deque[TopicFrame] = deque(maxlen=max(3, topic_capacity))
        self._referents: deque[Referent] = deque(maxlen=max(12, referent_capacity))
        self._memory_items: deque[MemoryItemV2] = deque(maxlen=64)
        self._turn = 0
        self._active_topic = ""
        self._active_subtopic = ""
        self._domain = ""
        self._active_problem: ActiveProblem | None = None
        self._pending_question: PendingQuestionV2 | None = None
        self._pending_task = ""
        self._focus = ""
        self._recent_user_intent = ""
        self._last_resolution = ContinuityResolution()
        self.store = V2SidecarStore(telemetry_path)

    @property
    def last_resolution(self) -> ContinuityResolution:
        with self._lock:
            return self._last_resolution

    def snapshot(self) -> WorkingSnapshotV2:
        with self._lock:
            problem = self._active_problem
            return WorkingSnapshotV2(
                active_topic=self._active_topic,
                active_subtopic=self._active_subtopic,
                current_domain=self._domain,
                active_entities=tuple(item.canonical for item in list(self._referents)[-8:]),
                active_example=self._render_problem(problem) if problem else "",
                pending_question=self._pending_question.text if self._pending_question else "",
                pending_task=self._pending_task,
                recent_referents=tuple(item.canonical for item in list(self._referents)[-6:]),
                conversation_focus=self._focus,
                recent_user_intent=self._recent_user_intent,
                topic_stack=tuple(frame.topic for frame in self._topics),
                resolved_referent=self._last_resolution.target,
                resolved_source=self._last_resolution.source,
                resolved_speaker=self._last_resolution.speaker,
                turn_count=self._turn,
            )

    def memory_items(self) -> tuple[MemoryItemV2, ...]:
        with self._lock:
            return tuple(self._memory_items)

    def observe_turn(self, user_text: str, assistant_text: str) -> None:
        with self._lock:
            self._turn += 1
            turn = self._turn
            combined = f"{user_text}\n{assistant_text}"
            self._recent_user_intent = self._infer_intent(user_text)
            requested_topic = self._return_target(user_text)
            if requested_topic:
                restored = self._find_topic(requested_topic)
                if restored is not None:
                    self._activate_frame(restored)

            domain = self._detect_domain(combined) or self._domain
            topic = self._detect_topic(combined)
            explicit_shift = bool(re.search(r"\b(?:cambiemos|otro\s+tema|ahora\s+hablemos|pasemos\s+a)\b", normalize(user_text)))
            if not topic and not explicit_shift:
                topic = self._active_topic
            if not topic and not self._active_topic:
                topic = self._topic_from_surface(user_text)
            if explicit_shift and not topic:
                topic = self._topic_from_surface(user_text)
                domain = self._detect_domain(user_text)

            entities = self._extract_entities(combined, domain=domain, topic=topic, turn=turn)
            for entity in entities:
                self._append_referent(entity)
            problem = self._extract_problem(combined, topic=topic, turn=turn)
            if problem is not None:
                self._active_problem = problem
            elif explicit_shift:
                self._active_problem = None

            question = self._last_question(assistant_text)
            self._pending_question = (
                PendingQuestionV2(
                    question,
                    "answer",
                    self._pending_target(question) or self._best_target(),
                    turn,
                )
                if question
                else None
            )
            self._pending_task = self._extract_pending_task(assistant_text)
            if topic:
                self._active_topic = topic
            if domain:
                self._domain = domain
            self._focus = self._best_target() or self._active_topic or self._domain
            self._active_subtopic = next(
                (item.canonical for item in reversed(entities) if item.kind == "subject"),
                "",
            )
            self._remember_user_facts(user_text, topic=self._active_topic, entities=entities)
            self._upsert_topic(
                TopicFrame(
                    topic=self._active_topic,
                    domain=self._domain,
                    entities=tuple(item.canonical for item in entities) or (self._active_topic,),
                    summary=" ".join(combined.split())[:360],
                    turn=turn,
                    importance=1.0,
                )
            )
            self._last_resolution = ContinuityResolution()
            self.store.append({"event": "turn_committed", "state": asdict(self.snapshot())})

    def resolve(self, text: str) -> ContinuityResolution:
        with self._lock:
            result = self._resolve_locked(text)
            self._last_resolution = result
            self.store.append({"event": "reference_resolution", "resolution": asdict(result)})
            return result

    def context_note(self) -> str:
        snapshot = self.snapshot()
        lines = ["WORKING CONVERSATION STATE V2 (evidencia estructurada, no texto para repetir):"]
        if snapshot.current_domain:
            lines.append(f"- WORLD_CONTEXT domain: {snapshot.current_domain}")
        if snapshot.active_topic:
            lines.append(f"- active_topic: {snapshot.active_topic}")
        if snapshot.active_subtopic:
            lines.append(f"- active_subtopic: {snapshot.active_subtopic}")
        if snapshot.recent_user_intent:
            lines.append(f"- recent_user_intent: {snapshot.recent_user_intent}")
        if snapshot.active_entities:
            lines.append("- active_entities: " + " | ".join(snapshot.active_entities[-6:]))
        if snapshot.active_example:
            lines.append(f"- active_problem: {snapshot.active_example}")
        if snapshot.pending_question:
            lines.append(f"- pending_question: {snapshot.pending_question}")
        if snapshot.pending_task:
            lines.append(f"- pending_task: {snapshot.pending_task}")
        if snapshot.topic_stack:
            lines.append("- topic_stack recent-first: " + " > ".join(reversed(snapshot.topic_stack)))
        if snapshot.resolved_referent:
            lines.append(
                f"- resolved_referent: {snapshot.resolved_referent} "
                f"[source={snapshot.resolved_source}; speaker={snapshot.resolved_speaker}]"
            )
            lines.append("- Responde usando ese referente; no pidas aclaración salvo conflicto real entre candidatos.")
        lines.extend(
            [
                "- USER_SAID pertenece al usuario; SELF pertenece solo a Anydes; WORLD_CONTEXT no es SELF.",
                "- Una inferencia del modelo no se convierte por sí sola en recuerdo ni experiencia.",
                "- Conserva el dominio activo al interpretar elipsis, pronombres y nombres abreviados.",
            ]
        )
        return "\n".join(lines)

    def _resolve_locked(self, text: str) -> ContinuityResolution:
        normalized = normalize(text)
        if not normalized:
            return ContinuityResolution()

        return_target = self._return_target(text)
        if return_target:
            frame = self._find_topic(return_target)
            if frame is not None:
                self._activate_frame(frame)
                return ContinuityResolution(
                    "topic_stack", text, text, frame.topic, 0.96,
                    "WORKING", "WORLD", tuple(item.topic for item in self._topics if item != frame),
                )

        if re.search(r"\b(?:cuanto\s+seria|cual\s+seria\s+el\s+resultado|y\s+cuanto)\b", normalized):
            if self._active_problem is not None:
                target = self._render_problem(self._active_problem)
                return ContinuityResolution(
                    "active_problem", text,
                    f"Resuelve el problema activo ({target}). Petición original: {text}",
                    target, 0.98, self._active_problem.source, "WORLD",
                )

        if "formula" in normalized:
            target = self._find_formula_target()
            if target:
                return ContinuityResolution(
                    "active_formula", text,
                    f"Da la fórmula correspondiente a {target}.",
                    target, 0.97, "WORKING", "WORLD",
                )

        if self._pending_question is not None and self._is_short_answer(normalized):
            return ContinuityResolution(
                "pending_question", text, text, self._pending_question.target or self._pending_question.text,
                0.95, "WORKING", "USER",
            )

        if re.search(r"\b(?:no\s+es\s+debil|debil\s+a\s+nada\s+mas|nada\s+mas)\b", normalized):
            target = self._best_referent(kinds={"subject", "entity", "domain"})
            if target is not None:
                return self._referent_resolution(text, target, "predicate_subject", 0.97)

        ordinal = re.search(r"\b(?:el|la)\s+(segund[oa]|tercer[oa]|primer[oa])\b", normalized)
        if ordinal:
            index = {"primero": 0, "primera": 0, "segundo": 1, "segunda": 1, "tercero": 2, "tercera": 2}[ordinal.group(1)]
            candidates = self._topic_referents()
            if len(candidates) > index:
                return self._referent_resolution(text, candidates[index], "ordinal", 0.92)

        explicit = self._canonical_explicit_entity(normalized)
        if explicit and self._domain:
            target = f"{explicit} dentro de {self._domain}"
            return ContinuityResolution(
                "domain_entity", text, f"{text} (dominio activo: {self._domain}; entidad: {explicit})",
                target, 0.95, "WORLD_CONTEXT", "WORLD",
            )

        if _REFERENCE_SIGNAL.search(text):
            target = self._best_referent()
            if target is not None:
                return self._referent_resolution(text, target, "recent_referent", 0.88)
            if self._active_topic:
                return ContinuityResolution(
                    "recent_referent", text, text, self._active_topic, 0.78, "WORKING", "WORLD",
                )
        return ContinuityResolution()

    def _referent_resolution(self, text: str, referent: Referent, kind: str, confidence: float) -> ContinuityResolution:
        return ContinuityResolution(
            kind,
            text,
            f"{text} (referente resuelto: {referent.canonical}; dominio: {referent.domain or self._domain})",
            referent.canonical,
            confidence,
            referent.source,
            referent.speaker,
        )

    def _detect_domain(self, text: str) -> str:
        for label, pattern in _DOMAIN_PATTERNS:
            if pattern.search(text):
                return label
        return ""

    def _detect_topic(self, text: str) -> str:
        for label, pattern in _TOPIC_PATTERNS:
            if pattern.search(text):
                if label == "formula" and self._active_topic:
                    return self._active_topic
                return label
        return ""

    @staticmethod
    def _topic_from_surface(text: str) -> str:
        normalized = normalize(text)
        normalized = re.sub(r"^.*?\b(?:cambiemos|otro tema|ahora hablemos|pasemos a)\b", "", normalized).strip()
        return " ".join(normalized.split()[:6])

    def _extract_entities(self, text: str, *, domain: str, topic: str, turn: int) -> list[Referent]:
        found: list[tuple[str, str]] = []
        normalized = normalize(text)
        for raw, canonical in _ALIASES.items():
            if raw in normalized:
                kind = "subject" if raw.startswith("tipo ") else "entity"
                found.append((canonical, kind))
        for match in re.finditer(r"(?:`([^`]+)`|\"([^\"]+)\")", text):
            label = next(group for group in match.groups() if group)
            found.append((label.strip(), "subject"))
        for match in re.finditer(
            r"\b((?:Monster Hunter|Pok[eé]mon|Charge Blade|Switch Axe|"
            r"ley de gravitaci[oó]n universal|tipo\s+(?:Fuego|Agua|Planta|Hielo|Roca|Tierra)))\b",
            text,
            re.I,
        ):
            found.append((match.group(1).strip(), "domain" if "monster" in match.group(1).casefold() or "pok" in match.group(1).casefold() else "subject"))
        numbered = re.findall(
            r"(?:^|\n)\s*(?:\d+[.)-]|[-*])\s*"
            r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ +#-]{2,30})",
            text,
        )
        for label in numbered:
            cleaned = label.strip(" ,.;:-")
            if len(cleaned.split()) <= 4:
                found.append((cleaned, "entity"))
        unique: list[Referent] = []
        seen: set[str] = set()
        for label, kind in found:
            key = normalize(label)
            if not key or key in seen:
                continue
            seen.add(key)
            canonical = _ALIASES.get(key, label)
            unique.append(
                Referent(
                    id=f"referent:{uuid.uuid4().hex}", label=label, canonical=canonical,
                    kind=kind, source="WORLD_CONTEXT", speaker="WORLD", topic=topic,
                    domain=domain, turn=turn,
                )
            )
        return unique

    def _extract_problem(self, text: str, *, topic: str, turn: int) -> ActiveProblem | None:
        assignments = {
            name: values.replace(" ", "")
            for name, values in re.findall(r"\b([a-zA-Z])\s*=\s*(\([^)]{1,60}\))", text)
        }
        normalized = normalize(text)
        if assignments:
            operation = "projection" if "proyeccion" in normalized or topic == "vector projection" else "calculate"
            return ActiveProblem("math", topic or "active calculation", assignments, operation, "WORLD_CONTEXT", turn)
        if "gravitacion universal" in normalized:
            return ActiveProblem("formula", "ley de gravitación universal", {}, "provide_formula", "WORLD_CONTEXT", turn)
        if re.search(r"```|\b(?:funcion|clase|codigo|bug|error)\b", normalized):
            return ActiveProblem("code", topic or "code", {}, "continue_or_debug", "WORLD_CONTEXT", turn)
        return None

    def _remember_user_facts(self, text: str, *, topic: str, entities: list[Referent]) -> None:
        normalized = normalize(text)
        match = re.search(r"\b(?:me\s+gusta|me\s+encanta|prefiero)\s+([^,.!?;]{2,80})", normalized)
        if not match:
            return
        content = match.group(1).strip()
        item = MemoryItemV2(
            id=f"memory-v2:{uuid.uuid4().hex}", content=content,
            source="USER_SAID", speaker="USER", topic=topic,
            entities=tuple(entity.canonical for entity in entities),
            importance=0.7, confidence=1.0, scope="user", retention="short",
            memory_type="preference", expiry="review_before_long_term",
        )
        self._memory_items.append(item)
        self.store.append({"event": "memory_candidate", "item": asdict(item)})

    def _find_formula_target(self) -> str:
        if self._active_problem and self._active_problem.kind == "formula":
            return self._active_problem.label
        for item in reversed(self._referents):
            if "ley" in normalize(item.canonical) or "formula" in item.kind:
                return item.canonical
        if self._active_topic not in {"", "formula"}:
            return self._active_topic
        return ""

    def _best_referent(self, kinds: set[str] | None = None) -> Referent | None:
        if kinds is None:
            priorities = ("subject", "entity", "domain")
        else:
            priorities = tuple(kind for kind in ("subject", "entity", "domain") if kind in kinds)
        for kind in priorities:
            for item in reversed(self._referents):
                if item.kind == kind:
                    return item
        return None

    def _best_target(self) -> str:
        item = self._best_referent()
        return item.canonical if item else self._active_topic

    def _topic_referents(self) -> list[Referent]:
        rows = [item for item in self._referents if not self._active_topic or item.topic == self._active_topic]
        deduped: list[Referent] = []
        seen: set[str] = set()
        for item in rows:
            key = normalize(item.canonical)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _canonical_explicit_entity(self, normalized: str) -> str:
        for alias, canonical in _ALIASES.items():
            if alias in normalized:
                return canonical
        return ""

    def _append_referent(self, referent: Referent) -> None:
        key = normalize(referent.canonical)
        retained = [item for item in self._referents if normalize(item.canonical) != key]
        self._referents.clear()
        self._referents.extend(retained)
        self._referents.append(referent)

    def _upsert_topic(self, frame: TopicFrame) -> None:
        if not frame.topic:
            return
        key = normalize(f"{frame.domain} {frame.topic}")
        retained = [item for item in self._topics if normalize(f"{item.domain} {item.topic}") != key]
        self._topics.clear()
        self._topics.extend(retained)
        self._topics.append(frame)

    def _find_topic(self, target: str) -> TopicFrame | None:
        normalized_target = normalize(target)
        for alias, canonical in _ALIASES.items():
            if alias in normalized_target:
                normalized_target = normalized_target.replace(alias, normalize(canonical))
        terms = set(normalized_target.split())
        scored: list[tuple[float, TopicFrame]] = []
        for frame in self._topics:
            haystack = set(normalize(f"{frame.topic} {frame.domain} {' '.join(frame.entities)} {frame.summary}").split())
            lexical = len(terms & haystack)
            if lexical:
                age = max(0, self._turn - frame.turn)
                decayed_importance = max(0.08, frame.importance * (0.86 ** age))
                scored.append((lexical * 10.0 + decayed_importance, frame))
        return max(scored, key=lambda item: (item[0], item[1].turn))[1] if scored else None

    def _activate_frame(self, frame: TopicFrame) -> None:
        self._active_topic = frame.topic
        self._domain = frame.domain
        self._focus = next((item for item in reversed(frame.entities) if item), frame.topic)
        if self._active_problem is not None and normalize(self._active_problem.label) != normalize(frame.topic):
            self._active_problem = None

    @staticmethod
    def _return_target(text: str) -> str:
        match = re.search(r"\bvolviendo\s+a\s+(?:lo\s+de\s+)?(.+?)(?:[,.!?]|$)", normalize(text))
        return match.group(1).strip() if match else ""

    @staticmethod
    def _last_question(text: str) -> str:
        matches = re.findall(r"[^.!?¿]*¿?[^.!?]*\?", text)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _extract_pending_task(text: str) -> str:
        match = re.search(r"\b(?:puedo|quieres\s+que)\s+([^?]{3,120})\?", text, re.I)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _pending_target(question: str) -> str:
        normalized = normalize(question)
        match = re.search(r"\b(?:quieres\s+que|puedo)\s+(.+)$", normalized)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _is_short_answer(normalized: str) -> bool:
        words = normalized.split()
        return len(words) <= 7 and bool(set(words) & {"si", "no", "claro", "dale", "hazlo", "eso", "exacto"})

    @staticmethod
    def _infer_intent(text: str) -> str:
        normalized = normalize(text)
        if "?" in text or text.lstrip().startswith("¿"):
            return "question"
        if re.search(r"\b(?:haz|abre|busca|calcula|explica|dime|muestra|revisa)\b", normalized):
            return "request"
        if re.search(r"\b(?:me gusta|me encanta|prefiero)\b", normalized):
            return "user_preference"
        return "statement"

    @staticmethod
    def _render_problem(problem: ActiveProblem) -> str:
        values = ", ".join(f"{key}={value}" for key, value in problem.values.items())
        suffix = f"; {values}" if values else ""
        return f"{problem.label}; operación pendiente={problem.pending_operation}{suffix}"
