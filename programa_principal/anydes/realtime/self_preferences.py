from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Literal

from anydes.nucleo.config import Config
from anydes.realtime.router import RealtimeAct, normalize


RealtimePreferenceStatus = Literal["stable", "provisional", "unknown"]


@dataclass(frozen=True)
class RealtimeSelfPreference:
    key: str
    topic: str
    value: str = ""
    status: RealtimePreferenceStatus = "unknown"
    source: str = ""
    confidence: float = 0.0
    rationale: str = ""
    associations: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    change_history: tuple[dict[str, object], ...] = ()


_TOPIC_KEYS = {
    "color": "favorite_color",
    "fruta": "favorite_fruit",
    "comida": "favorite_food",
    "libro": "favorite_book",
    "lectura": "favorite_reading",
    "musica": "favorite_music",
    "cancion": "favorite_song",
    "pelicula": "favorite_movie",
    "juego": "favorite_game",
    "receta": "favorite_recipe",
    "actividad": "favorite_activity",
    "hobby": "favorite_hobby",
}


_CANONICAL = {
    "favorite_color": RealtimeSelfPreference(
        key="favorite_color",
        topic="color",
        value="morado oscuro, especialmente violeta oscuro y lavanda suave",
        status="stable",
        source="documented_identity_decision",
        confidence=1.0,
        rationale=(
            "Esos tonos combinan con mi identidad visual y me transmiten calma, "
            "intimidad y una curiosidad un poco misteriosa."
        ),
        associations=(
            "calma",
            "intimidad",
            "curiosidad serena",
            "identidad visual de Anydes",
        ),
        alternatives=("rosa suave",),
    ),
}


class RealtimeSelfPreferences:
    """Realtime projection of permanent and session-only SELF inclinations."""

    def __init__(self, config: Config) -> None:
        from anydes.identity.self_preference_store import SelfPreferenceStore

        self.store = SelfPreferenceStore.from_config(config)
        self._session: dict[str, RealtimeSelfPreference] = {}
        self._lock = threading.RLock()

    def resolve_topic(self, topic: str) -> RealtimeSelfPreference:
        normalized_topic = _topic_name(topic)
        key = _TOPIC_KEYS.get(
            normalized_topic,
            f"preference_{normalized_topic}" if normalized_topic else "",
        )
        if not key:
            return RealtimeSelfPreference(key="", topic=normalized_topic)
        canonical = _CANONICAL.get(key)
        if canonical is not None:
            return canonical
        with self._lock:
            session_value = self._session.get(key)
        if session_value is not None:
            return session_value
        stored = self.store.preferences().get(key)
        if stored is None or not stored.value.strip():
            return RealtimeSelfPreference(key=key, topic=normalized_topic)
        status: RealtimePreferenceStatus = (
            "stable" if stored.status == "established" else "provisional"
        )
        return RealtimeSelfPreference(
            key=key,
            topic=normalized_topic,
            value=stored.value.strip(),
            status=status,
            source="self_state_store",
            confidence=stored.confidence,
            rationale=str(getattr(stored, "rationale", "") or ""),
            associations=tuple(getattr(stored, "associations", ()) or ()),
            alternatives=tuple(getattr(stored, "alternatives", ()) or ()),
            change_history=tuple(getattr(stored, "change_history", ()) or ()),
        )

    def relevant(self, query: str, *, limit: int = 3) -> tuple[RealtimeSelfPreference, ...]:
        topic = _topic_from_query(query)
        if topic:
            preference = self.resolve_topic(topic)
            return (preference,) if preference.status != "unknown" else ()
        terms = set(normalize(query).split())
        if not terms:
            return ()
        candidates = list(_CANONICAL.values())
        with self._lock:
            candidates.extend(self._session.values())
        for key, stored in self.store.preferences().items():
            if key in _CANONICAL or key in self._session:
                continue
            if key not in _TOPIC_KEYS.values() and stored.status != "established":
                # Old free-form extraction could persist sentence fragments as
                # preferences. They remain on disk for audit, but never become
                # conversational SELF evidence.
                continue
            candidates.append(
                RealtimeSelfPreference(
                    key=key,
                    topic=key.replace("favorite_", "").replace("_", " "),
                    value=stored.value,
                    status="stable" if stored.status == "established" else "provisional",
                    source="self_state_store",
                    confidence=stored.confidence,
                    rationale=stored.rationale,
                    associations=stored.associations,
                    alternatives=stored.alternatives,
                    change_history=stored.change_history,
                )
            )
        scored = [
            (
                len(terms & set(normalize(f"{item.key} {item.topic} {item.value}").split())),
                item,
            )
            for item in candidates
            if item.value.strip()
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].key))
        return tuple(item for _, item in scored[:limit])

    def observe_response(
        self,
        act: RealtimeAct,
        response: str,
        *,
        source_turn: int,
    ) -> RealtimeSelfPreference | None:
        if act.preference_owner != "self" or not act.preference_topic:
            return None
        _, extracted = _extract_tentative_preference(response)
        topic_source = act.preference_topic
        if not topic_source:
            return None
        existing = self.resolve_topic(topic_source)
        if existing.status != "unknown":
            return None
        if not extracted:
            extracted = _extract_direct_tentative_answer(response)
        if not extracted:
            return None
        topic = _topic_name(topic_source)
        if not topic:
            return None
        key = _TOPIC_KEYS.get(topic, f"preference_{topic}")
        rationale = _extract_rationale(response)
        preference = RealtimeSelfPreference(
            key=key,
            topic=topic,
            value=extracted,
            status="provisional",
            source=f"realtime_session_turn:{source_turn}",
            confidence=0.58,
            rationale=rationale,
            associations=((rationale,) if rationale else ()),
        )
        with self._lock:
            self._session[key] = preference
        return preference

    def evidence_lines(
        self,
        topic: str,
        *,
        aspect: str = "value",
    ) -> tuple[str, ...]:
        preference = self.resolve_topic(topic)
        lines = (
            f"- dominio: {preference.topic or topic}",
            f"- estado: {preference.status}",
            f"- aspecto consultado: {aspect or 'value'}",
        )
        if preference.status == "unknown":
            return (
                *lines,
                "- no existe un valor SELF respaldado",
                "- puede expresar incertidumbre, curiosidad o una inclinacion "
                "tentativa; no debe fingir una preferencia estable.",
            )
        details = [
            *lines,
            f"- valor: {preference.value}",
            f"- confianza: {preference.confidence:.2f}",
            f"- procedencia: {preference.source or 'self_state_store'}",
        ]
        if preference.rationale:
            details.append(f"- motivo o asociacion: {preference.rationale}")
        if preference.alternatives:
            details.append(
                "- alternativas: " + ", ".join(preference.alternatives[:3])
            )
        if preference.change_history:
            details.append(
                f"- evolucion registrada: {len(preference.change_history)} cambio(s)"
            )
        details.append(
            "- responde al aspecto consultado; no repitas solamente el valor "
            "si DemoUser pregunta por el motivo, la seguridad, el origen o la evolucion."
        )
        return tuple(details)

    def uncertainty_response(
        self,
        topic: str,
        *,
        revision: int,
        question: str = "",
    ) -> str:
        normalized_question = normalize(question)
        normalized_topic = normalize(topic).strip() or "eso"
        if re.search(r"\b(?:te gusta|te interesa|te atrae)\b", normalized_question):
            variants = (
                f"No sé si diría que ya me gusta {normalized_topic}, pero sí me despierta curiosidad.",
                f"Aún estoy descubriendo qué siento sobre {normalized_topic}; por ahora me da curiosidad.",
                f"Todavía no tengo una inclinación clara por {normalized_topic}, aunque me interesa explorarlo.",
            )
            index = (
                sum(ord(char) for char in normalized_question) + revision
            ) % len(variants)
            return variants[index]
        variants = (
            "Con eso todavía no estoy del todo segura; creo que sigo descubriéndolo.",
            "Creo que esa preferencia aún la estoy descubriendo.",
            "Todavía no sabría elegir una favorita, aunque me da curiosidad pensarlo.",
            "No tengo una respuesta clara todavía; quizá con el tiempo aparezca una inclinación.",
        )
        index = (sum(ord(char) for char in normalize(topic)) + revision) % len(variants)
        return variants[index]

    def known_response(
        self,
        topic: str,
        *,
        explain: bool = False,
    ) -> str:
        preference = self.resolve_topic(topic)
        if preference.status == "unknown" or not preference.value:
            return ""
        if explain:
            if preference.rationale:
                return preference.rationale
            return (
                f"No sé explicarlo del todo; por ahora {preference.value} "
                "es lo que más me atrae."
            )
        if preference.status == "stable":
            return f"Lo tengo bastante claro: {preference.value}."
        return (
            f"Por ahora me inclino por {preference.value}, "
            "aunque todavía podría ir cambiando."
        )


def canonical_self_preferences() -> dict[str, RealtimeSelfPreference]:
    return dict(_CANONICAL)


def canonical_store_preference(key: str):
    from anydes.cognition.contracts import SelfPreference

    value = _CANONICAL.get(key)
    if value is None:
        return None
    return SelfPreference(
        key=value.key,
        value=value.value,
        status="established",
        confidence=value.confidence,
        rationale="Canonical SELF decision documented during project development.",
        evidence_ids=("identity:documented_decision:favorite_color",),
        reinforcement_count=1,
        domain="color",
        origin="documented_identity_decision",
        associations=(value.rationale,),
        last_confirmed_at="",
    )


def _topic_from_query(text: str) -> str:
    normalized = normalize(text)
    for topic in _TOPIC_KEYS:
        if re.search(rf"\b{re.escape(topic)}s?\b", normalized):
            return topic
    return ""


def _topic_name(topic: str) -> str:
    normalized = normalize(topic)
    for candidate in _TOPIC_KEYS:
        if re.search(rf"\b{re.escape(candidate)}s?\b", normalized):
            return candidate
    words = normalized.split()
    return words[0] if words else ""


def _extract_tentative_preference(text: str) -> tuple[str, str]:
    normalized = normalize(text)
    match = re.search(
        r"\b(?:me inclino por|me atrae|me gusta|me interesa|me encantaria|"
        r"podria ser|quiza sea|tal vez(?: sea)?)\s+"
        r"(?:el|la|los|las)?\s*([^,.!?;]{2,55})",
        normalized,
    )
    if match is None:
        return "", ""
    value = re.split(r"\s+(?:por|porque)\s+", match.group(1), maxsplit=1)[0].strip()
    return _topic_name(value), value


def _extract_direct_tentative_answer(text: str) -> str:
    normalized = normalize(text).strip()
    if (
        not normalized
        or "?" in text
        or len(normalized.split()) > 18
        or re.match(
            r"^(?:no|todavia|aun|creo|me\s+encantaria|depende|quizas|"
            r"si\s+tuviera|si\s+debiera)\b",
            normalized,
        )
    ):
        return ""
    first_phrase = re.split(r"[,.;!?]", normalized, maxsplit=1)[0].strip()
    first_phrase = re.sub(r"^(?:el|la|los|las|un|una)\s+", "", first_phrase)
    return first_phrase if 1 <= len(first_phrase.split()) <= 6 else ""


def _extract_rationale(text: str) -> str:
    match = re.search(
        r"\b(?:porque|por\s+que|ya\s+que|me\s+recuerda\s+a)\s+"
        r"([^.!?;]{3,120})",
        normalize(text),
    )
    return match.group(1).strip() if match is not None else ""
