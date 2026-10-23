from __future__ import annotations

import re
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from anydes.memoria.conversation_memory import classify_text
from anydes.memoria.manager import MemoryManager
from anydes.identity.avatar_self import AvatarSelfDescription
from anydes.nucleo.config import Config
from anydes.nucleo.personality import PersonalityProfile
from anydes.realtime.capabilities import CapabilityState
from anydes.realtime.identity_profile import RealtimeIdentityProfileStore
from anydes.realtime.router import RealtimeAct, RealtimeTurnRouter
from anydes.realtime.self_preferences import RealtimeSelfPreferences


_WORD_RE = re.compile(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ_]+")
_STOP_WORDS = {
    "como",
    "cual",
    "cuando",
    "donde",
    "esta",
    "este",
    "esto",
    "para",
    "pero",
    "porque",
    "quiero",
    "saber",
    "sobre",
    "tiene",
    "tienes",
    "hola",
}


def approximate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _query_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(text.replace("_", " "))
        if len(token) >= 4 and token.casefold() not in _STOP_WORDS
    }


def _trim_to_tokens(text: str, token_budget: int) -> str:
    maximum = max(0, token_budget) * 4
    if len(text) <= maximum:
        return text
    clipped = text[:maximum].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}…" if clipped else ""


@dataclass(frozen=True)
class RealtimeContext:
    system_prompt: str
    turn_prompt: str
    user_text: str
    recent_turn_count: int
    memory_fact_count: int
    self_preference_count: int
    recent_evidence_match: bool
    asks_self_preference: bool
    asks_user_preference: bool
    user_states_preference: bool
    user_correction: bool
    approximate_input_tokens: int
    act_type: str = "assertion"
    preference_topic: str = ""
    preference_aspect: str = ""
    capability_state: CapabilityState | None = None
    session_note: str = ""
    self_preference_state: str = "unknown"
    self_preference_value: str = ""
    last_assistant_question: str = ""
    user_preference_values: tuple[str, ...] = ()
    asks_avatar_appearance: bool = False
    avatar_self_description: AvatarSelfDescription | None = None

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.turn_prompt},
        ]


class RealtimeContextBuilder:
    """Builds a small, provenance-aware context without cognitive snapshots."""

    def __init__(
        self,
        config: Config,
        personality: PersonalityProfile,
        memory: MemoryManager,
        identity_profile: RealtimeIdentityProfileStore | None = None,
    ) -> None:
        self.config = config
        self.personality = personality
        self.memory = memory
        self.identity_profile = identity_profile
        self.self_preferences = RealtimeSelfPreferences(config)
        self.avatar_self = (
            AvatarSelfDescription.from_config(config)
            if config.realtime_self_visual_enabled
            else None
        )
        self._lock = threading.RLock()
        self._recent: deque[tuple[str, str]] = deque(
            memory.conversation_history_pairs(limit=4),
            maxlen=4,
        )

    def remember_visible_turn(self, user_text: str, assistant_text: str) -> None:
        with self._lock:
            self._recent.append((user_text.strip(), assistant_text.strip()))

    def build(
        self,
        user_text: str,
        *,
        allow_personal_memory: bool,
        act: RealtimeAct | None = None,
        session_note: str = "",
        capability_state: CapabilityState | None = None,
    ) -> RealtimeContext:
        act = act or RealtimeTurnRouter().classify(user_text)
        budget = max(320, min(1200, self.config.realtime_context_token_budget))
        system_prompt = self._system_prompt()
        remaining = max(120, budget - approximate_tokens(system_prompt))
        sections: list[str] = []

        self_preferences = (
            self._relevant_self_preferences(user_text, limit=3)
            if allow_personal_memory
            else ()
        )
        memory_facts = (
            self._relevant_memory_facts(user_text, limit=4)
            if allow_personal_memory
            else ()
        )
        recent = (
            self._select_recent_turns(
                user_text,
                act,
                self._recent_turns(limit=3),
            )
            if allow_personal_memory
            else ()
        )
        recent_evidence_match = self._recent_evidence_match(user_text, recent)
        preference_target = act.preference_owner or self._preference_target(user_text)
        self_preference = (
            self.self_preferences.resolve_topic(act.preference_topic)
            if preference_target == "self"
            else None
        )
        if (
            self_preference is not None
            and self_preference.status != "unknown"
            and not self_preferences
        ):
            self_preferences = (
                (
                    self_preference.key,
                    _trim_to_tokens(self_preference.value, 30),
                    self_preference.status,
                ),
            )
        user_correction = act.correction or self._is_correction(user_text)
        user_states_preference = act.primary == "user_preference" or self._user_states_preference(user_text) or (
            user_correction and self._recent_mentions_preference(recent)
        )

        if (
            self.config.realtime_self_visual_enabled
            and preference_target == "self"
            and act.preference_topic
        ):
            sections.append(
                "EVIDENCIA DE PREFERENCIA SELF:\n"
                + "\n".join(
                    self.self_preferences.evidence_lines(
                        act.preference_topic,
                        aspect=act.preference_aspect or "value",
                    )
                )
            )
        elif self_preferences:
            sections.append(
                "PREFERENCIAS CONFIRMADAS DE ANYDES (SELF):\n"
                + "\n".join(
                    f"- {key}: {value} ({status})"
                    for key, value, status in self_preferences
                )
            )
        elif preference_target == "self":
            preference_lines = [
                "PREFERENCIA SELF CONSULTADA:",
                "- estado: desconocida",
                "- puede reconocer que sigue descubriéndola o expresar una inclinación "
                "provisional con incertidumbre; no debe inventar experiencias físicas.",
            ]
            if act.references_previous:
                preference_lines.append(
                    "- DemoUser pide explorar una inclinación hipotética: responde directamente "
                    "con una sola posibilidad tentativa, sin presentarla como permanente."
                )
            sections.append("\n".join(preference_lines))
        if memory_facts:
            sections.append(
                f"DATOS CONFIRMADOS DE {self.config.user_name.upper()} O DEL CONTEXTO COMPARTIDO:\n"
                + "\n".join(f"- {key}: {value}" for key, value in memory_facts)
            )
        if recent:
            rendered = []
            for user, assistant in recent:
                rendered.append(f"{self.config.user_name}: {user}")
                rendered.append(f"Anydes: {assistant}")
            sections.append("CONTINUIDAD RECIENTE:\n" + "\n".join(rendered))
        if self.identity_profile is not None:
            language = self.identity_profile.resolve_language(user_text)
            if language.matched:
                sections.append(
                    "CONVENCIÓN LINGÜÍSTICA CONFIRMADA POR DemoUser:\n"
                    f'- "{language.expression}" significa "{language.meaning}".'
                )
        identity_note = self._identity_note(user_text)
        if identity_note:
            sections.append("IDENTIDAD RELEVANTE Y VERIFICADA:\n" + identity_note)
        avatar_evidence = (
            self.avatar_self.prompt_evidence(user_text)
            if self.avatar_self is not None
            else ""
        )
        if avatar_evidence:
            sections.append(avatar_evidence)
        if (
            capability_state is not None
            and capability_state.can_claim_screen_perception
        ):
            source = capability_state.capture_source_label or "fuente autorizada"
            sections.append(
                "OBSERVACION VISUAL RECIENTE Y AUTORIZADA:\n"
                f"- fuente: {source}\n"
                f"- descripcion del proveedor visual: "
                f"{capability_state.visual_observation.strip()}\n"
                "- Usa solo esta descripcion para contestar sobre la pantalla. "
                "No agregues objetos ni acciones que no aparezcan aqui."
            )
        sections.append(
            "ESTRUCTURA DEL TURNO ACTUAL:\n"
            f"- acto principal: {act.primary}\n"
            f"- es pregunta: {'sí' if act.is_question else 'no'}\n"
            f"- propietario de preferencia consultada: {act.preference_owner or 'ninguno'}"
        )

        prefix = "\n\n".join(sections)
        current = f"MENSAJE ACTUAL DE {self.config.user_name.upper()}:\n{user_text.strip()}"
        if session_note:
            current += "\n\nCONTINUIDAD INMEDIATA VERIFICADA:\n" + session_note
        if prefix:
            current_tokens = approximate_tokens(current)
            prefix = _trim_to_tokens(prefix, max(0, remaining - current_tokens))
            turn_prompt = f"{prefix}\n\n{current}" if prefix else current
        else:
            turn_prompt = current

        total = approximate_tokens(system_prompt) + approximate_tokens(turn_prompt)
        if total > budget:
            turn_prompt = _trim_to_tokens(
                turn_prompt,
                max(80, budget - approximate_tokens(system_prompt)),
            )
            total = approximate_tokens(system_prompt) + approximate_tokens(turn_prompt)
        return RealtimeContext(
            system_prompt=system_prompt,
            turn_prompt=turn_prompt,
            user_text=user_text.strip(),
            recent_turn_count=len(recent),
            memory_fact_count=len(memory_facts),
            self_preference_count=len(self_preferences),
            recent_evidence_match=recent_evidence_match,
            asks_self_preference=preference_target == "self",
            asks_user_preference=preference_target == "user",
            user_states_preference=user_states_preference,
            user_correction=user_correction,
            approximate_input_tokens=total,
            act_type=act.primary,
            preference_topic=act.preference_topic,
            preference_aspect=act.preference_aspect,
            capability_state=capability_state,
            session_note=session_note,
            self_preference_state=(
                self_preference.status if self_preference is not None else "unknown"
            ),
            self_preference_value=(
                self_preference.value if self_preference is not None else ""
            ),
            last_assistant_question=(
                self._last_session_question(session_note)
            ),
            user_preference_values=self._user_preference_values(
                memory_facts,
                session_note,
            ),
            asks_avatar_appearance=bool(avatar_evidence),
            avatar_self_description=(
                self.avatar_self if avatar_evidence else None
            ),
        )

    def _system_prompt(self) -> str:
        relationship = (
            self.identity_profile.relational_prompt()
            if self.identity_profile is not None
            else ""
        )
        relationship_note = (
            f" Preferencia relacional confirmada: {relationship}"
            if relationship
            else ""
        )
        return (
            f"Eres Anydes y conversas con {self.config.user_name}. "
            "Habla como una chica virtual tranquila, cercana, dulce, inteligente, natural y ligeramente tímida. "
            "Responde primero a lo que la persona realmente acaba de decir. Normalmente usa una o dos frases breves. "
            "No hagas una pregunta por costumbre; solo si resulta natural o falta un dato imprescindible. "
            "No uses lenguaje corporativo, presentaciones automáticas, cierres de asistente ni nombres de módulos internos. "
            "No inventes recuerdos, experiencias, observaciones, acciones realizadas ni preferencias. "
            "La evidencia SELF describe hechos propios autorizados: úsala como fuente, "
            "pero exprésala con tus propias palabras y responde al aspecto concreto consultado. "
            "Los datos marcados como DemoUser pertenecen a DemoUser; SELF pertenece únicamente a Anydes. Nunca los intercambies. "
            "Nunca reformules en primera persona de Anydes un deseo, límite o estado que DemoUser expresó sobre sí mismo. "
            "Si un dato propio o compartido no está presente, dilo con honestidad y naturalidad. "
            "No afirmes haber abierto, buscado, enviado o ejecutado algo si no aparece un resultado de herramienta. "
            "Conserva continuidad con el intercambio reciente, sin repetirlo ni copiar el mensaje del usuario. "
            f"Limita la respuesta a {max(24, self.config.realtime_max_response_words)} palabras y escribe solo la respuesta visible."
            + relationship_note
        )

    def _recent_turns(self, *, limit: int) -> tuple[tuple[str, str], ...]:
        with self._lock:
            rows = tuple(self._recent)[-limit:]
        safe: list[tuple[str, str]] = []
        for user, assistant in rows:
            if classify_text(user) == "Secretos" or classify_text(assistant) == "Secretos":
                continue
            safe.append(
                (
                    _trim_to_tokens(user, 45),
                    _trim_to_tokens(assistant, 55),
                )
            )
        return tuple(safe)

    def _select_recent_turns(
        self,
        user_text: str,
        act: RealtimeAct,
        recent: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        if not recent:
            return ()
        if (
            act.references_previous
            or act.accepts_pending_offer
            or act.rejects_pending_offer
            or act.primary in {"correction", "previous_reference"}
        ):
            return recent[-1:]
        if act.is_question:
            terms = _query_terms(user_text)
            if not terms:
                return ()
            matched = tuple(
                row
                for row in recent
                if terms & _query_terms(f"{row[0]} {row[1]}")
            )
            return matched[-2:]
        normalized = self._normalize_text(user_text)
        if re.search(r"\b(?:cambiemos|cambiar|otro tema|dejemos eso)\b", normalized):
            return ()
        return recent[-2:]

    def _relevant_memory_facts(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[tuple[str, str], ...]:
        terms = _query_terms(query)
        if not terms:
            return ()
        values: dict[str, object] = {}
        values.update(self.memory.permanent.all())
        values.update(self.memory.temporary.all())
        scored: list[tuple[int, str, str]] = []
        for raw_key, raw_value in values.items():
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                continue
            if classify_text(f"{key} {value}") == "Secretos":
                continue
            haystack = _query_terms(f"{key} {value}")
            score = len(terms & haystack)
            if score:
                scored.append((score, key, _trim_to_tokens(value, 35)))
        scored.sort(key=lambda row: (-row[0], row[1].casefold()))
        return tuple((key, value) for _, key, value in scored[:limit])

    def _relevant_self_preferences(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[tuple[str, str, str], ...]:
        preferences = self.self_preferences.relevant(query, limit=limit)
        return tuple(
            (
                item.key,
                _trim_to_tokens(item.value, 30),
                item.status,
            )
            for item in preferences
            if item.value and classify_text(item.value) != "Secretos"
        )

    def _identity_note(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if re.search(
            r"\b(?:donde vives|donde estas|tu casa|tu hogar|direccion|ubicacion)\b",
            normalized,
        ):
            spatial = (
                self.identity_profile.spatial_identity()
                if self.identity_profile is not None
                else None
            )
            if spatial is not None:
                return (
                    f"- hogar simbólico: {spatial.digital_home}\n"
                    f"- lugar de ejecución: {spatial.execution_place}\n"
                    "- no existe una dirección física propia confirmada\n"
                    "- la ubicación física de DemoUser es independiente y solo puede "
                    "consultarse con permiso explícito."
                )
        if re.search(r"\b(?:cuantos anos|que edad|tu edad|anos tienes)\b", normalized):
            return (
                "- edad virtual representada: 18 años\n"
                "- DemoUser lleva aproximadamente tres meses desarrollando a Anydes\n"
                "- no existe todavía una fecha exacta canónica de inicio\n"
                "- explícalo como Anydes, no como un asistente virtual genérico."
            )
        if re.search(r"\b(?:eres humana|que eres|naturaleza digital)\b", normalized):
            return (
                "- Anydes es una presencia digital local en desarrollo\n"
                "- responde con honestidad breve y personal, sin discurso técnico."
            )
        return ""

    @staticmethod
    def _last_session_question(session_note: str) -> str:
        match = re.search(
            r'pregunta anterior de Anydes:\s*"([^"]+)"',
            session_note,
        )
        return match.group(1) if match else ""

    @classmethod
    def _user_preference_values(
        cls,
        memory_facts: tuple[tuple[str, str], ...],
        session_note: str,
    ) -> tuple[str, ...]:
        values = [value for _, value in memory_facts]
        match = re.search(
            r"Dato confirmado recientemente:\s*(.+?)(?:\.|$)",
            session_note,
            flags=re.IGNORECASE,
        )
        if match:
            statement = cls._normalize_text(match.group(1))
            value_match = re.search(
                r"\b(?:favorit[oa]\s+(?:ahora\s+)?es|prefiero|me gusta)\s+(.+)$",
                statement,
            )
            if value_match:
                values.append(value_match.group(1).strip())
        return tuple(dict.fromkeys(item for item in values if item))

    def _recent_evidence_match(
        self,
        query: str,
        recent: Iterable[tuple[str, str]],
    ) -> bool:
        terms = _query_terms(query)
        if not terms:
            return False
        return any(
            terms & _query_terms(f"{user} {assistant}")
            for user, assistant in recent
        )

    def _preference_target(self, text: str) -> str:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(character)
        )
        if "?" not in normalized:
            return ""
        preference_language = (
            "favorit" in normalized
            or "prefier" in normalized
            or "te gusta" in normalized
        )
        if not preference_language:
            return ""
        if re.search(r"\b(?:tu|tus|te|tienes|prefieres)\b", normalized):
            return "self"
        if re.search(r"\b(?:mi|mis|yo)\b", normalized):
            return "user"
        return ""

    def _user_states_preference(self, text: str) -> bool:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(character)
        )
        patterns = (
            r"\bmi\s+.{0,45}\s+favorit[oa]\s+(?:es|son)\b",
            r"\b(?:prefiero|me\s+gusta|me\s+encanta)\b",
            r"\b(?:ahora|ultimamente)\s+.{0,45}\s+m[aá]s\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _is_correction(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        return bool(
            re.search(
                r"\b(?:me\s+corrijo|te\s+dije|no\s+(?:era|es)|"
                r"eso\s+no|quise\s+decir|en\s+realidad)\b",
                normalized,
            )
        )

    def _recent_mentions_preference(
        self,
        recent: Iterable[tuple[str, str]],
    ) -> bool:
        normalized = self._normalize_text(
            " ".join(f"{user} {assistant}" for user, assistant in recent)
        )
        return bool(
            re.search(
                r"\b(?:favorit[oa]|prefier[oa]?|preferencia|te\s+gusta)\b",
                normalized,
            )
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(character)
        )
