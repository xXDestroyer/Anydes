from __future__ import annotations

import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field

from anydes.realtime.router import RealtimeAct, normalize


@dataclass(frozen=True)
class SessionTurn:
    user: str
    assistant: str


@dataclass(frozen=True)
class PendingOffer:
    text: str
    topic: str
    expected_response: str = "accept_or_reject"


@dataclass(frozen=True)
class PendingAction:
    capability: str
    argument: str = ""
    entity: str = ""
    status: str = "proposed"
    task_id: str = ""
    origin_turn: int = 0
    parameters: dict[str, object] = field(default_factory=dict)
    source: str = "phase1"
    requires_confirmation: bool = False


@dataclass(frozen=True)
class PendingCommitment:
    objective: str
    task_id: str
    status: str
    origin_turn: int = 0


@dataclass(frozen=True)
class PendingCalendarDraft:
    draft_id: str
    version: int
    title: str
    activity: str = ""
    subject: str = ""
    participants: tuple[str, ...] = ()
    start_iso: str = ""
    end_iso: str = ""
    timezone: str = ""
    location: str = ""
    notes: str = ""
    missing_fields: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    created_turn: int = 0
    updated_turn: int = 0

    def as_parameters(self) -> dict[str, object]:
        return {
            "draft_id": self.draft_id,
            "draft_version": self.version,
            "title": self.title,
            "activity": self.activity,
            "subject": self.subject,
            "participants": list(self.participants),
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "timezone": self.timezone,
            "location": self.location,
            "notes": self.notes,
            "missing_fields": list(self.missing_fields),
            "changed_fields": list(self.changed_fields),
        }


@dataclass(frozen=True)
class DialogueResolution:
    kind: str = ""
    canonical_text: str = ""
    target: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class RealtimeSessionSnapshot:
    active_topic: str = ""
    last_assistant_question: str = ""
    pending_offer: PendingOffer | None = None
    expected_response: str = ""
    recent_correction: str = ""
    pending_instruction: str = ""
    main_reference: str = ""
    recently_confirmed_fact: str = ""
    last_preference_owner: str = ""
    last_preference_topic: str = ""
    proposed_action: PendingAction | None = None
    pending_commitment: PendingCommitment | None = None
    pending_calendar_draft: PendingCalendarDraft | None = None
    principal_entity: str = ""
    pronoun_references: tuple[str, ...] = ()
    recent_language_convention: str = ""
    last_task_result: str = ""
    resume_turn: int | None = None
    turn_count: int = 0


class RealtimeSessionState:
    """Compact immediate continuity plus a local, bounded visible-session log."""

    def __init__(
        self,
        *,
        history_capacity: int = 240,
        session_id: str | None = None,
    ) -> None:
        self._turns: deque[SessionTurn] = deque(maxlen=max(20, history_capacity))
        self._lock = threading.RLock()
        self._snapshot = RealtimeSessionSnapshot()
        self.session_id = session_id or f"realtime-session:{uuid.uuid4().hex}"
        self._language_conventions: dict[str, str] = {}

    def snapshot(self) -> RealtimeSessionSnapshot:
        with self._lock:
            return self._snapshot

    def register_language_convention(self, expression: str, meaning: str) -> None:
        normalized = _compact(expression)
        semantic = normalize(meaning)
        if not normalized or not semantic:
            return
        with self._lock:
            self._language_conventions[normalized] = semantic
            self._snapshot = _replace_snapshot(
                self._snapshot,
                recent_language_convention=f"{expression.strip()}={meaning.strip()}",
            )

    def register_action_proposal(
        self,
        capability: str,
        argument: str = "",
        *,
        entity: str = "",
        task_id: str = "",
        parameters: dict[str, object] | None = None,
        source: str = "phase1",
        requires_confirmation: bool = False,
    ) -> PendingAction:
        action = PendingAction(
            capability=capability.strip(),
            argument=argument.strip(),
            entity=(entity or argument or capability).strip(),
            task_id=task_id,
            origin_turn=self.snapshot().turn_count,
            parameters=dict(parameters or {}),
            source=source,
            requires_confirmation=requires_confirmation,
        )
        with self._lock:
            self._snapshot = _replace_snapshot(
                self._snapshot,
                proposed_action=action,
                principal_entity=action.entity,
                resume_turn=action.origin_turn,
            )
        return action

    def register_commitment(
        self,
        *,
        objective: str,
        task_id: str,
        status: str,
    ) -> PendingCommitment:
        commitment = PendingCommitment(
            objective=objective.strip(),
            task_id=task_id,
            status=status,
            origin_turn=self.snapshot().turn_count,
        )
        with self._lock:
            self._snapshot = _replace_snapshot(
                self._snapshot,
                pending_commitment=commitment,
                resume_turn=commitment.origin_turn,
            )
        return commitment

    def register_calendar_draft(
        self,
        parameters: dict[str, object],
    ) -> PendingCalendarDraft:
        with self._lock:
            previous = self._snapshot.pending_calendar_draft
            draft_id = str(parameters.get("draft_id") or "")
            same_draft = bool(
                previous is not None and draft_id and draft_id == previous.draft_id
            )
            turn = self._snapshot.turn_count
            version = int(
                parameters.get("draft_version")
                or ((previous.version + 1) if same_draft else 1)
            )
            draft = PendingCalendarDraft(
                draft_id=draft_id or f"calendar-draft:{uuid.uuid4().hex}",
                version=max(1, version),
                title=str(parameters.get("title") or "Actividad").strip(),
                activity=str(parameters.get("activity") or "").strip(),
                subject=str(parameters.get("subject") or "").strip(),
                participants=tuple(
                    str(item).strip()
                    for item in parameters.get("participants", ())
                    if str(item).strip()
                ),
                start_iso=str(parameters.get("start_iso") or ""),
                end_iso=str(parameters.get("end_iso") or ""),
                timezone=str(parameters.get("timezone") or ""),
                location=str(parameters.get("location") or "").strip(),
                notes=str(parameters.get("notes") or "").strip(),
                missing_fields=tuple(parameters.get("missing_fields") or ()),
                changed_fields=tuple(parameters.get("changed_fields") or ()),
                created_turn=previous.created_turn if same_draft else turn,
                updated_turn=turn,
            )
            self._snapshot = _replace_snapshot(
                self._snapshot,
                pending_calendar_draft=draft,
                principal_entity=draft.title,
                resume_turn=draft.created_turn,
            )
            return draft

    def mark_action_resolved(self, *, task_id: str = "") -> None:
        with self._lock:
            commitment = self._snapshot.pending_commitment
            if commitment is not None and task_id and commitment.task_id != task_id:
                commitment = self._snapshot.pending_commitment
            else:
                commitment = None
            self._snapshot = _replace_snapshot(
                self._snapshot,
                proposed_action=None,
                pending_commitment=commitment,
                pending_calendar_draft=None,
                resume_turn=None,
            )

    def commit_task_result(self, text: str, *, objective: str) -> None:
        with self._lock:
            self._snapshot = _replace_snapshot(
                self._snapshot,
                active_topic=objective.strip() or self._snapshot.active_topic,
                proposed_action=None,
                pending_commitment=None,
                pending_calendar_draft=None,
                principal_entity=objective.strip() or self._snapshot.principal_entity,
                recently_confirmed_fact=text.strip(),
                last_task_result=text.strip(),
                resume_turn=None,
            )

    def resolve_reference(self, text: str) -> DialogueResolution:
        compact = _compact(text)
        if not compact:
            return DialogueResolution()
        with self._lock:
            state = self._snapshot
            convention = self._language_conventions.get(compact, "")
        signal = _reply_signal(compact, convention=convention)
        if state.proposed_action is not None:
            action = state.proposed_action
            if signal in {"affirmative", "proceed", "reference"}:
                target = " ".join(
                    item for item in (action.capability, action.argument) if item
                )
                return DialogueResolution(
                    kind="action_confirmation",
                    canonical_text="sí, hazlo",
                    target=target,
                    confidence=0.96,
                )
            if signal == "negative":
                return DialogueResolution(
                    kind="action_rejection",
                    canonical_text="no",
                    target=action.entity,
                    confidence=0.96,
                )
        if state.pending_offer is not None:
            if signal in {"affirmative", "proceed"}:
                return DialogueResolution(
                    kind="offer_acceptance",
                    canonical_text="sí",
                    target=state.pending_offer.text,
                    confidence=0.95,
                )
            if signal == "negative":
                return DialogueResolution(
                    kind="offer_rejection",
                    canonical_text="no",
                    target=state.pending_offer.text,
                    confidence=0.95,
                )
        if state.last_assistant_question and signal in {"affirmative", "negative"}:
            return DialogueResolution(
                kind="question_answer",
                canonical_text="sí" if signal == "affirmative" else "no",
                target=state.last_assistant_question,
                confidence=0.92,
            )
        if signal in {"reference", "proceed"} and state.main_reference:
            return DialogueResolution(
                kind="previous_reference",
                canonical_text=text.strip(),
                target=state.main_reference,
                confidence=0.84,
            )
        return DialogueResolution()

    def context_note(self, act: RealtimeAct, user_text: str = "") -> str:
        with self._lock:
            state = self._snapshot
        current_topic = _topic_hint(user_text)
        continues_topic = bool(
            act.references_previous
            or act.accepts_pending_offer
            or act.rejects_pending_offer
            or (
                current_topic
                and state.active_topic
                and _topics_related(current_topic, state.active_topic)
            )
        )
        lines: list[str] = []
        if act.accepts_pending_offer and state.pending_offer is not None:
            lines.append(
                "El usuario acepta la oferta pendiente de Anydes: "
                f'"{state.pending_offer.text}". Cumple esa oferta ahora con contenido concreto; '
                "no hagas otra oferta ni preguntes otra vez que desea."
            )
        elif act.rejects_pending_offer and state.pending_offer is not None:
            lines.append(
                "El usuario rechaza de forma natural la oferta pendiente: "
                f'"{state.pending_offer.text}".'
            )
        elif act.references_previous and state.main_reference:
            lines.append(f'Referencia inmediata principal: "{state.main_reference}".')
            if _is_short_continuation(user_text):
                lines.append(
                    "El usuario te pide continuar el tema anterior. Aporta ahora el contenido "
                    "que falta. La respuesta debe entregar ese contenido, no prometerlo "
                    "ni terminar en otra pregunta."
                )
        if (
            act.references_previous
            and state.last_assistant_question
            and not act.accepts_pending_offer
            and not act.rejects_pending_offer
        ):
            lines.append(
                "El mensaje actual responde brevemente a esta pregunta anterior de Anydes: "
                f'"{state.last_assistant_question}". No la repitas.'
            )
        if state.active_topic and continues_topic:
            lines.append(
                f"Tema activo inmediato: {state.active_topic}. "
                "Interpreta los pronombres y omisiones del mensaje actual como referencias "
                "a este tema y responde sobre el de forma explicita en la primera oracion."
            )
        preference_relevant = bool(
            act.preference_owner
            or "favorit" in normalize(user_text)
            or "preferencia" in normalize(user_text)
        )
        if state.recent_correction and preference_relevant:
            lines.append(f"Corrección reciente no resuelta: {state.recent_correction}.")
        if state.recently_confirmed_fact and preference_relevant:
            lines.append(f"Dato confirmado recientemente: {state.recently_confirmed_fact}.")
        if state.proposed_action is not None:
            lines.append(
                "Accion propuesta pendiente: "
                f"{state.proposed_action.capability} {state.proposed_action.argument}. "
                "No afirmes que fue ejecutada sin un resultado real."
            )
        if state.pending_calendar_draft is not None:
            draft = state.pending_calendar_draft
            lines.append(
                "Borrador de calendario pendiente: "
                f"{draft.title} (versión {draft.version}). "
                "No lo presentes como creado hasta recibir un resultado real."
            )
        if state.pending_commitment is not None:
            lines.append(
                "Compromiso respaldado por tarea real: "
                f"{state.pending_commitment.objective} "
                f"(estado {state.pending_commitment.status})."
            )
        if act.references_previous and state.last_task_result:
            lines.append(
                f'Resultado real más reciente de una tarea: "{state.last_task_result}".'
            )
        if act.history_target == "session_topics":
            fragments = self._representative_user_messages(limit=9)
            if fragments:
                lines.append(
                    "Fragmentos reales recuperados de esta sesion para resumir sus temas: "
                    + " | ".join(f'"{item}"' for item in fragments)
                    + ". No anadas temas que no aparezcan aqui."
                )
        return "\n".join(lines)

    def commit(self, user_text: str, assistant_text: str, act: RealtimeAct) -> None:
        turn = SessionTurn(user=user_text.strip(), assistant=assistant_text.strip())
        question = _last_question(turn.assistant)
        offer = _pending_offer(question, turn.assistant)
        with self._lock:
            previous = self._snapshot
            self._turns.append(turn)
            correction = user_text.strip() if act.correction else previous.recent_correction
            confirmed = (
                user_text.strip()
                if act.primary == "user_preference"
                or (
                    act.correction
                    and "favorit" in normalize(user_text)
                )
                else previous.recently_confirmed_fact
            )
            new_topic = _topic_hint(user_text)
            if act.accepts_pending_offer or act.rejects_pending_offer:
                pending_offer = offer
            else:
                pending_offer = offer or previous.pending_offer
            if offer is None and (act.accepts_pending_offer or act.rejects_pending_offer):
                pending_offer = None
            elif (
                offer is None
                and previous.pending_offer is not None
                and new_topic
                and not _topics_related(new_topic, previous.pending_offer.topic)
                and not act.references_previous
            ):
                pending_offer = None
            active_topic = (
                previous.active_topic
                if (
                    act.references_previous
                    or act.accepts_pending_offer
                    or act.rejects_pending_offer
                )
                else new_topic or previous.active_topic
            )
            proposed_action = previous.proposed_action
            if (
                proposed_action is not None
                and new_topic
                and proposed_action.entity
                and not act.references_previous
                and not _topics_related(new_topic, proposed_action.entity)
            ):
                proposed_action = None
            self._snapshot = RealtimeSessionSnapshot(
                active_topic=active_topic,
                last_assistant_question=(
                    question
                    if question
                    else "" if act.references_previous else previous.last_assistant_question
                ),
                pending_offer=pending_offer,
                expected_response=offer.expected_response if offer else "",
                recent_correction=correction,
                pending_instruction=(
                    user_text.strip() if act.primary == "request" else ""
                ),
                main_reference=_main_reference(turn),
                recently_confirmed_fact=confirmed,
                last_preference_owner=(
                    act.preference_owner or previous.last_preference_owner
                ),
                last_preference_topic=(
                    act.preference_topic or previous.last_preference_topic
                ),
                proposed_action=proposed_action,
                pending_commitment=previous.pending_commitment,
                pending_calendar_draft=(
                    previous.pending_calendar_draft
                    if proposed_action is not None
                    and proposed_action.capability == "calendar_create"
                    else None
                ),
                principal_entity=(
                    previous.principal_entity
                    if act.references_previous
                    else new_topic or previous.principal_entity
                ),
                pronoun_references=(
                    (previous.principal_entity,)
                    if act.references_previous and previous.principal_entity
                    else ()
                ),
                recent_language_convention=previous.recent_language_convention,
                last_task_result=previous.last_task_result,
                resume_turn=previous.resume_turn,
                turn_count=previous.turn_count + 1,
            )

    def resolve_history(self, target: str) -> str | None:
        with self._lock:
            turns = tuple(self._turns)
        if not turns:
            return "Esta es nuestra primera intervención de la sesión."
        if target == "first_user_message":
            return f'Tu primer mensaje de esta sesión fue: "{turns[0].user}".'
        if target == "first_user_question":
            first = next((turn.user for turn in turns if _has_question(turn.user)), None)
            return (
                f'Tu primera pregunta de esta sesión fue: "{first}".'
                if first
                else "Todavía no habías hecho una pregunta en esta sesión."
            )
        if target == "last_user_message":
            return f'Tu mensaje anterior fue: "{turns[-1].user}".'
        if target == "previous_topic":
            topic = _topic_hint(turns[-1].user)
            return (
                f"Justo antes estábamos hablando de {topic}."
                if topic
                else f'Lo último que mencionaste fue: "{turns[-1].user}".'
            )
        if target == "recent_messages":
            recent = turns[-3:]
            rendered = "; ".join(f'"{turn.user}"' for turn in recent)
            return f"Tus mensajes recientes fueron: {rendered}."
        if target == "repeated_question":
            questions = [turn.user for turn in turns if _has_question(turn.user)]
            if not questions:
                return "No encuentro una pregunta anterior repetida en esta sesión."
            normalized_last = normalize(questions[-1])
            count = sum(normalize(item) == normalized_last for item in questions)
            return (
                f'Esa pregunta aparece {count} veces en esta sesión: "{questions[-1]}".'
            )
        if target == "session_topics":
            return None
        return None

    def turns(self) -> tuple[SessionTurn, ...]:
        with self._lock:
            return tuple(self._turns)

    def confirmed_user_preference(self, topic: str) -> str:
        normalized_topic = normalize(topic)
        with self._lock:
            turns = tuple(self._turns)
        for turn in reversed(turns):
            match = re.search(
                r"\bmi\s+(.{1,45}?)\s+favorit[oa]s?\s+"
                r"(?:ahora\s+)?(?:es|son)\s+([^,.!?;]{1,70})",
                normalize(turn.user),
            )
            if match is None:
                continue
            candidate_topic = match.group(1).strip()
            if normalized_topic and not _topics_related(
                normalized_topic,
                candidate_topic,
            ):
                continue
            return match.group(2).strip()
        return ""

    def _representative_user_messages(self, *, limit: int) -> tuple[str, ...]:
        with self._lock:
            turns = tuple(self._turns)
        selected: list[str] = []
        covered_terms: set[str] = set()
        for turn in turns:
            text = turn.user.strip()
            normalized = normalize(text)
            if len(normalized.split()) < 3 or _is_history_meta_question(normalized):
                continue
            terms = set(_topic_hint(text).split())
            if not terms or (terms <= covered_terms and len(selected) >= 3):
                continue
            selected.append(" ".join(text.split()[:18]))
            covered_terms.update(terms)
        if len(selected) <= limit:
            return tuple(selected)
        indexes = {
            round(index * (len(selected) - 1) / (limit - 1))
            for index in range(limit)
        }
        return tuple(selected[index] for index in sorted(indexes))


def _has_question(text: str) -> bool:
    return "?" in text or "¿" in text


def _last_question(text: str) -> str:
    matches = re.findall(r"[^.!?¿]*[¿]?[^.!?]*\?", text)
    return matches[-1].strip() if matches else ""


def _pending_offer(question: str, assistant_text: str) -> PendingOffer | None:
    if not question:
        return None
    normalized = normalize(question)
    if (
        re.match(r"^(?:hay|tienes|conoces)\s+(?:algun|alguna|algo)\b", normalized)
        and not re.search(r"\b(?:prefieres|quieres)\s+que\b|\bpuedo\b", normalized)
    ):
        return None
    if not re.search(
        r"\b(?:quieres\s+que|te\s+gustaria\s+que|"
        r"te\s+gustaria\s+(?:un|una|otro|otra|algo|saber|ver)|"
        r"deseas\s+que|prefieres\s+que|puedo|te cuento|te explico|"
        r"te muestro|vemos|seguimos)\b",
        normalized,
    ):
        return None
    return PendingOffer(
        text=question,
        topic=_topic_hint(assistant_text),
    )


def _topic_hint(text: str) -> str:
    words = [
        token
        for token in re.findall(r"[a-záéíóúüñ0-9]+", normalize(text))
        if len(token) >= 4
        and token
        not in {
            "como",
            "cual",
            "cuando",
            "donde",
            "para",
            "pero",
            "porque",
            "quieres",
            "gustaria",
            "puedo",
            "sobre",
            "algo",
            "esto",
            "esta",
            "tengo",
            "sabes",
            "dime",
            "explica",
        }
    ]
    return " ".join(words[:5])


def _main_reference(turn: SessionTurn) -> str:
    return _topic_hint(f"{turn.user} {turn.assistant}")


def _topics_related(left: str, right: str) -> bool:
    left_terms = set(normalize(left).split())
    right_terms = set(normalize(right).split())
    return bool(left_terms & right_terms)


def _is_history_meta_question(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:primer|primera|ultimo|ultima|anterior|recientes?|"
            r"sesion|conversacion|mensajes?|preguntas?|temas?)\b",
            normalized,
        )
        and re.search(r"\b(?:cual|cuales|que|de que)\b", normalized)
    )


def _is_short_continuation(text: str) -> bool:
    return _reply_signal(_compact(text)) in {
        "affirmative",
        "proceed",
        "reference",
    }


def _compact(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", normalize(text)).split())


def _reply_signal(compact: str, *, convention: str = "") -> str:
    semantic = normalize(convention)
    if semantic in {"si", "afirmativo", "affirmative", "aceptacion"}:
        return "affirmative"
    if semantic in {"no", "negativo", "negative", "rechazo"}:
        return "negative"
    tokens = set(compact.split())
    if len(tokens) <= 7 and tokens & {
        "si",
        "claro",
        "vale",
        "ok",
        "okay",
        "perfecto",
        "bueno",
    } and not tokens & {"no", "mejor"}:
        return "affirmative"
    if len(tokens) <= 6 and tokens & {
        "dale",
        "dime",
        "cuentame",
        "continua",
        "adelante",
        "hazlo",
        "intentalo",
        "abrelo",
        "muestrame",
    }:
        return "proceed"
    if len(tokens) <= 6 and tokens & {
        "esa",
        "ese",
        "eso",
        "anterior",
        "ejemplo",
    }:
        return "reference"
    if len(tokens) <= 6 and "no" in tokens:
        return "negative"
    return ""


def _replace_snapshot(
    snapshot: RealtimeSessionSnapshot,
    **changes,
) -> RealtimeSessionSnapshot:
    from dataclasses import replace

    return replace(snapshot, **changes)


DialogueState = RealtimeSessionState
