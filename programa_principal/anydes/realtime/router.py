from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


RealtimeActType = Literal[
    "question",
    "assertion",
    "user_preference",
    "correction",
    "request",
    "offer_acceptance",
    "offer_rejection",
    "previous_reference",
    "history_query",
]


@dataclass(frozen=True)
class RealtimeAct:
    """Small structural reading used by the product path, not a cognitive plan."""

    primary: RealtimeActType
    is_question: bool = False
    preference_owner: Literal["user", "self", ""] = ""
    preference_topic: str = ""
    preference_aspect: str = ""
    correction: bool = False
    references_previous: bool = False
    accepts_pending_offer: bool = False
    rejects_pending_offer: bool = False
    history_target: str = ""


class RealtimeTurnRouter:
    """Classifies surface structure without deciding Anydes' response."""

    def classify(
        self,
        text: str,
        *,
        has_pending_offer: bool = False,
        has_pending_question: bool = False,
    ) -> RealtimeAct:
        normalized = normalize(text)
        compact = normalized.strip(" .,!?:;")
        history_target = self._history_target(normalized)
        if history_target:
            return RealtimeAct(
                primary="history_query",
                is_question=True,
                history_target=history_target,
            )

        if has_pending_offer and self._is_acceptance(compact):
            return RealtimeAct(
                primary="offer_acceptance",
                accepts_pending_offer=True,
                references_previous=True,
            )
        if has_pending_offer and self._is_rejection(compact):
            return RealtimeAct(
                primary="offer_rejection",
                rejects_pending_offer=True,
                references_previous=True,
            )
        if has_pending_question and (
            self._is_acceptance(compact) or self._is_rejection(compact)
        ):
            return RealtimeAct(
                primary="previous_reference",
                references_previous=True,
            )
        if self._is_acceptance(compact):
            return RealtimeAct(
                primary="previous_reference",
                references_previous=True,
            )

        question = self._is_question(text, normalized)
        preference_owner, preference_topic = self._preference_query(
            normalized,
            question=question,
        )
        preference_aspect = self._preference_aspect(
            normalized,
            question=question,
        )
        correction = self._is_correction(normalized)
        references_previous = self._references_previous(normalized)
        if correction:
            primary: RealtimeActType = "correction"
        elif question:
            primary = "question"
        elif self._states_user_preference(normalized):
            primary = "user_preference"
            preference_owner = "user"
            preference_topic = self._stated_preference_topic(normalized)
        elif self._is_request(normalized):
            primary = "request"
        elif references_previous:
            primary = "previous_reference"
        else:
            primary = "assertion"
        return RealtimeAct(
            primary=primary,
            is_question=question,
            preference_owner=preference_owner,
            preference_topic=preference_topic,
            preference_aspect=preference_aspect,
            correction=correction,
            references_previous=references_previous,
        )

    @staticmethod
    def _is_question(text: str, normalized: str) -> bool:
        if "?" in text or "¿" in text:
            return True
        return bool(
            re.match(
                r"^(?:que|cual|cuales|como|cuando|donde|por que|quien|"
                r"cuanto|cuantos|sabes|puedes|podrias|tienes|eres|te gusta)\b",
                normalized,
            )
        )

    @staticmethod
    def _states_user_preference(normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(?:mi|mis)\s+.{0,55}\s+favorit[oa]s?\s+(?:es|son)\b|"
                r"\b(?:yo\s+)?(?:prefiero|me gusta|me encanta|me interesan?)\b",
                normalized,
            )
        )

    @staticmethod
    def _is_correction(normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(?:me corrijo|te dije|no era|eso no|quise decir|"
                r"en realidad|mejor dicho|rectifico)\b",
                normalized,
            )
        )

    @staticmethod
    def _is_request(normalized: str) -> bool:
        return bool(
            re.match(
                r"^(?:por favor\s+)?(?:abre|busca|muestra|dime|cuentame|"
                r"explicame|haz|pon|recuerda|guarda|ayudame|podrias|puedes)\b",
                normalized,
            )
        )

    @staticmethod
    def _references_previous(normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(?:eso|esa|ese|lo anterior|el anterior|la anterior|"
                r"el ejemplo|lo que dijiste|lo de antes|ese tema)\b|"
                r"\b(?:por|probar|comido|cocinado)\s+un[oa]\b|"
                r"^(?:y\s+)?(?:para\s+que|por\s+que|como)\s+"
                r"(?:sirve|funciona|se usa)\b|"
                r"^(?:y\s+)?(?:por\s+que|de\s+donde|desde\s+cuando|"
                r"estas\s+segura|segura|que\s+tan\s+segura|"
                r"podria\s+cambiar|ha\s+cambiado)\s*[?]?$",
                normalized,
            )
        )

    @staticmethod
    def _is_acceptance(compact: str) -> bool:
        compact = " ".join(re.sub(r"[^a-z0-9 ]+", " ", compact).split())
        tokens = set(compact.split())
        direct = {
            "si",
            "claro",
            "dale",
            "va",
            "bueno",
            "ok",
            "okay",
            "perfecto",
            "porfa",
            "adelante",
        }
        return (
            bool(tokens & direct)
            and len(tokens) <= 7
            and not tokens.intersection({"no", "mejor"})
        ) or compact in {"a ver", "a ver dale", "hazlo", "cuentame"}

    @staticmethod
    def _is_rejection(compact: str) -> bool:
        compact = " ".join(re.sub(r"[^a-z0-9 ]+", " ", compact).split())
        return bool(
            re.fullmatch(
                r"(?:no|no gracias|mejor no|ahora no|dejalo|dejemoslo)",
                compact,
            )
        )

    @staticmethod
    def _preference_query(
        normalized: str,
        *,
        question: bool,
    ) -> tuple[Literal["user", "self", ""], str]:
        if not question or not re.search(
            r"\b(?:favorit[oa]s?|preferid[oa]s?|prefieres?|te gusta|te interesa|preferencia)\b",
            normalized,
        ):
            return "", ""
        owner: Literal["user", "self", ""] = ""
        if re.search(r"\b(?:tu|tus|te|prefieres|tienes)\b", normalized):
            owner = "self"
        elif re.search(r"\b(?:mi|mis|yo)\b", normalized):
            owner = "user"
        topic_match = re.search(
            r"\b(?:cual|que)\s+es\s+(?:tu|mi)\s+(.+?)\s+favorit[oa]\b",
            normalized,
        )
        if topic_match is None:
            topic_match = re.search(
                r"\b(?:fruta|color|comida|libro|musica|cancion|pelicula|"
                r"actividad|hobby|juego|receta)s?\b",
                normalized,
            )
        if topic_match is None:
            topic_match = re.search(
                r"\b(?:te gusta|te interesa|prefieres)\s+([^,.!?;]{2,45})",
                normalized,
            )
        topic = topic_match.group(1 if topic_match.lastindex else 0) if topic_match else ""
        return owner, topic.strip()

    @staticmethod
    def _preference_aspect(normalized: str, *, question: bool) -> str:
        if not question:
            return ""
        if re.search(r"\b(?:por\s+que|motivo|razon|que\s+te\s+atrae)\b", normalized):
            return "motive"
        if re.search(
            r"\b(?:estas\s+segura|que\s+tan\s+segura|seguridad|certeza)\b",
            normalized,
        ):
            return "confidence"
        if re.search(r"\b(?:de\s+donde|desde\s+cuando|origen)\b", normalized):
            return "origin"
        if re.search(r"\b(?:alternativa|otra\s+opcion|cual\s+mas)\b", normalized):
            return "alternatives"
        if re.search(r"\b(?:ha\s+cambiado|podria\s+cambiar|evolucion)\b", normalized):
            return "evolution"
        return "value"

    @staticmethod
    def _history_target(normalized: str) -> str:
        if not re.search(
            r"\b(?:sesion|conversacion|historial|mensajes?|preguntas?|"
            r"dije|dijiste|recuerdas?|sabes|temas?|dia|fecha|"
            r"preferencias?|gustos?|hobbies|aficiones|intereses|"
            r"inferiste|dedujiste|patrones?)\b",
            normalized,
        ):
            return ""
        if re.search(
            r"\b(?:conversacion|sesion)\s+anterior\b|"
            r"\bultima\s+(?:conversacion|sesion)\b|"
            r"\bque\s+hablamos\s+(?:la\s+)?(?:vez|sesion)\s+anterior\b",
            normalized,
        ):
            return "previous_session"
        if re.search(
            r"\bprimer(?:a|o)?\s+(?:mensaje|cosa)\s+(?:de|del|en)\s+(?:el\s+)?dia\b|"
            r"\blo\s+primero\s+que\s+(?:te\s+)?dije\s+hoy\b",
            normalized,
        ):
            return "first_user_message_day"
        if re.search(
            r"\b(?:mensajes?|que\s+(?:te\s+)?dije|que\s+hablamos)\s+"
            r"(?:de|del|en)?\s*(?:la\s+)?(?:fecha\s+)?"
            r"(?:hoy|ayer|\d{1,4}[-/]\d{1,2}[-/]\d{1,4})\b",
            normalized,
        ):
            return "messages_on_date"
        if re.search(
            r"\b(?:de\s+que|que|cuales)\s+temas?\s+(?:hablamos|tocamos|vimos)\b|"
            r"\bde\s+que\s+(?:cosas|temas?)\s+hablamos\b",
            normalized,
        ):
            return "session_topics"
        if re.search(r"\bprimer(?:a|o)?\s+pregunta\b", normalized):
            return "first_user_question"
        if re.search(r"\bprimer(?:a|o)?\s+(?:mensaje|cosa)\b|lo primero que", normalized):
            return "first_user_message"
        if re.search(
            r"\bultim[oa]\s+(?:mensaje|cosa|pregunta)\b|"
            r"\b(?:mensaje|pregunta)\s+anterior\b",
            normalized,
        ):
            return "last_user_message"
        if re.search(r"\btema\s+anterior\b|de que hablabamos", normalized):
            return "previous_topic"
        if re.search(r"\b(?:mensajes|conversacion)\s+recientes?\b", normalized):
            return "recent_messages"
        if re.search(r"\b(?:repeti|repetido|ya te pregunte)\b", normalized):
            return "repeated_question"
        if re.search(r"\b(?:corregi|correcciones)\b", normalized):
            return "corrections"
        if re.search(r"\b(?:preferencias?|gustos?)\s+que\s+(?:te\s+)?dije\b", normalized):
            return "preferences"
        if re.search(r"\b(?:promesas?|compromisos?)\b", normalized):
            return "commitments"
        if re.search(
            r"\b(?:que\s+has|que\s+habias)\s+(?:inferido|deducido)\b|"
            r"\bpatrones?\s+(?:has\s+)?(?:visto|notado|observado)\b",
            normalized,
        ):
            return "derived_insights"
        if re.search(
            r"\b(?:que\s+(?:te\s+)?dije|que\s+sabes|recuerdas?)\s+"
            r"(?:sobre|acerca\s+de|de)\b|"
            r"\bque\s+sabes\s+de\s+mi\b|"
            r"\b(?:mis\s+)?(?:hobbies|aficiones|intereses|gustos)\b",
            normalized,
        ):
            return "user_topic"
        return ""

    @staticmethod
    def _stated_preference_topic(normalized: str) -> str:
        match = re.search(
            r"\bmi\s+(.{1,45}?)\s+favorit[oa]s?\s+(?:ahora\s+)?(?:es|son)\b",
            normalized,
        )
        return match.group(1).strip() if match else ""


def normalize(text: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", folded).strip()
