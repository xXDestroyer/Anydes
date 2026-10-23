from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from anydes.realtime.context import RealtimeContext


_TECHNICAL_LEAKS = re.compile(
    r"\b(?:reasoning engine|internalcontext|evidencebundle|provider|stack trace|groundedresponseplan)\b",
    re.IGNORECASE,
)
_UNVERIFIED_ACTION = re.compile(
    r"\b(?:ya\s+(?:lo\s+)?(?:abr[ií]|busqu[eé]|envi[eé]|ejecut[eé])|"
    r"he\s+(?:abierto|buscado|enviado|ejecutado)|"
    r"listo[,;:]?\s+(?:abr[ií]|busqu[eé]|envi[eé]|ejecut[eé]))\b",
    re.IGNORECASE,
)
_UNBACKED_TASK_COMMITMENT = re.compile(
    r"\b(?:(?:voy|vamos)\s+a\s+"
    r"(?:buscar|abrir|enviar|averiguar|ejecutar|reproducir|investigar)|"
    r"(?:lo|eso)\s+(?:hare|haré|buscare|buscaré|abrire|abriré|"
    r"enviare|enviaré|averiguare|averiguaré)|"
    r"te\s+avis(?:o|are|aré)\s+(?:cuando|en cuanto)|"
    r"ya\s+se\s+esta\s+(?:abriendo|reproduciendo|enviando|ejecutando))\b",
    re.IGNORECASE,
)
_SPEAKER_PREFIX = re.compile(
    r"^\s*(?:(?:anydes|assistant|asistente)\s*:\s*)+",
    re.IGNORECASE,
)
_VISIBLE_INTERNAL_LABEL = re.compile(r"(?:^|\s)(?:SELF|USER)\s*:", re.IGNORECASE)
_STABLE_SELF_PREFERENCE = re.compile(
    r"\bmi\s+.{0,45}\s+(?:favorit[oa]|preferid[oa])\s+(?:es|son)\b",
    re.IGNORECASE,
)
_SELF_LIKING = re.compile(
    r"\b(?:me\s+(?:gusta|gustan|encanta|encantan)|prefiero)\b",
    re.IGNORECASE,
)
_TENTATIVE_SELF = re.compile(
    r"\b(?:creo|quiz[aá]|tal\s+vez|podr[ií]a|me\s+inclino|"
    r"me\s+atrae|todav[ií]a\s+estoy|no\s+estoy\s+segura|"
    r"no\s+tengo(?:\s+a[uú]n)?\s+.{0,30}\s+favorit[oa])\b",
    re.IGNORECASE,
)
_FIRST_PERSON_UNKNOWN = re.compile(
    r"\b(?:no\s+(?:lo\s+)?s[eé]|no\s+recuerdo|no\s+lo\s+tengo|"
    r"no\s+me\s+lo\s+has|a[uú]n\s+no|todav[ií]a\s+no)\b",
    re.IGNORECASE,
)
_GENERIC_ASSISTANT_TAIL = re.compile(
    r"(?:en\s+qu[eé]\s+puedo\s+ayudarte|"
    r"en\s+qu[eé]\s+puedo\s+asistirte|"
    r"(?:puedo|pueda|podr[ií]a)\s+(?:ayudarte|asistirte|apoyarte)|"
    r"en\s+lo\s+que\s+pueda\s+(?:ayudarte|asistirte)|"
    r"ayudar\s+en\s+lo\s+que\s+pueda|"
    r"algo\s+m[aá]s\s+en\s+lo\s+que\s+pueda\s+(?:ayudarte|asistirte|apoyarte)|"
    r"estoy\s+aqu[ií]\s+para\s+ayudarte|"
    r"(?:si|cuando)\s+necesitas?\s+algo|"
    r"[¿]?\s*necesitas?\s+algo(?:\s+m[aá]s)?)",
    re.IGNORECASE,
)
_KNOWN_USER_NAME_QUESTION = re.compile(
    r"[¿?]?\s*(?:y\s+t[uú][,;:]?\s*)?c[oó]mo\s+te\s+llamas\s*[?]?",
    re.IGNORECASE,
)
_AFFIRMATIVE_VISUAL_CLAIM = re.compile(
    r"\b(?:estoy\s+viendo|en\s+tu\s+pantalla\s+hay|"
    r"(?:veo|observo|puedo\s+ver)\s+(?:tu\s+pantalla|la\s+pantalla|"
    r"tu\s+escritorio|el\s+escritorio|la\s+imagen|tu\s+imagen|la\s+foto|"
    r"una\s+playa|un\s+escritorio|una\s+ventana|lo\s+que\s+tienes\s+abierto))\b",
    re.IGNORECASE,
)
_MEMORY_ACKNOWLEDGEMENT = re.compile(
    r"\b(?:lo\s+tendre\s+presente|lo\s+recordare|me\s+lo\s+guardare|anotado)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_PHYSICAL_EXPERIENCE = re.compile(
    r"\b(?:he\s+(?:comido|probado|cocinado|olido|tocado)|"
    r"cuando\s+(?:comi|probe|cocine)|recuerdo\s+haber\s+(?:comido|probado|cocinado))\b",
    re.IGNORECASE,
)
_FIRST_PERSON_STATE = re.compile(
    r"\b(?:no\s+)?(?:quiero|necesito|prefiero|me\s+preocupa|me\s+molesta)\s+"
    r"([^,.!?;]{2,55})",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_HISTORY = re.compile(
    r"\s+(?:que\s+)?(?:desde\s+siempre\s+)?siempre\s+me\s+ha(?:n)?\s+"
    r"(?:gustado|interesado|encantado)(?:\s+mucho)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    text: str
    issues: tuple[str, ...] = ()


class RealtimeOutputValidator:
    """Fast deterministic boundary; it never asks a model to repair output."""

    def __init__(self, *, max_words: int = 72) -> None:
        self.max_words = max(16, int(max_words))

    def sanitize_prefix(self, text: str) -> str:
        return _SPEAKER_PREFIX.sub("", text, count=1).lstrip()

    def validate(
        self,
        text: str,
        context: RealtimeContext,
        *,
        tool_executed: bool = False,
        task_created: bool = False,
    ) -> ValidationResult:
        _ = context
        clean = self.sanitize_prefix(str(text or "")).strip()
        issues: list[str] = []
        clean, user_echo_removed = self._remove_leading_user_echo(
            clean,
            context.user_text,
        )
        if user_echo_removed:
            issues.append("user_echo_removed")
        clean, echo_removed = self._remove_question_echo(clean, context.user_text)
        if echo_removed:
            issues.append("question_echo_removed")
        previous_question = getattr(context, "last_assistant_question", "")
        if previous_question:
            clean, repeated_removed = self._remove_question_echo(
                clean,
                previous_question,
            )
            if repeated_removed:
                issues.append("resolved_question_repetition_removed")
        clean = self.sanitize_prefix(clean)
        if int(getattr(context, "self_preference_count", 0) or 0) == 0:
            clean, unsupported_history_removed = _UNSUPPORTED_SELF_HISTORY.subn(
                "",
                clean,
            )
            if unsupported_history_removed:
                clean = re.sub(r"\s+([,.!?])", r"\1", clean)
                issues.append("unsupported_self_history_removed")
        clean, generic_removed = self._remove_generic_tail(clean)
        if generic_removed:
            issues.append("generic_tail_removed")
            if not clean:
                issues.append("generic_response_only")
        if not clean:
            issues.append("empty_output")
        if _TECHNICAL_LEAKS.search(clean):
            issues.append("technical_leak")
        if _VISIBLE_INTERNAL_LABEL.search(clean):
            issues.append("internal_label")
        if not tool_executed and _UNVERIFIED_ACTION.search(clean):
            issues.append("unverified_action_claim")
        if not task_created and _UNBACKED_TASK_COMMITMENT.search(clean):
            issues.append("unbacked_task_commitment")
        capability = getattr(context, "capability_state", None)
        if (
            _AFFIRMATIVE_VISUAL_CLAIM.search(clean)
            and not bool(
                capability is not None
                and capability.can_claim_screen_perception
            )
        ):
            issues.append("unverified_visual_claim")
        avatar_self = getattr(context, "avatar_self_description", None)
        if (
            bool(getattr(context, "asks_avatar_appearance", False))
            and avatar_self is not None
        ):
            issues.extend(avatar_self.validation_issues(clean))
        if (
            getattr(context, "act_type", "") == "question"
            and _MEMORY_ACKNOWLEDGEMENT.search(self._normalize(clean))
        ):
            issues.append("question_treated_as_memory")
        if self._claims_physical_experience(clean):
            issues.append("unsupported_physical_experience")
        normalized_clean = self._normalize(clean)
        self_value = self._normalize(
            str(getattr(context, "self_preference_value", "") or "")
        )
        for value in getattr(context, "user_preference_values", ()):
            normalized_value = self._normalize(str(value))
            if (
                normalized_value
                and normalized_value in normalized_clean
                and normalized_value not in self_value
                and (
                    _SELF_LIKING.search(clean)
                    or _STABLE_SELF_PREFERENCE.search(clean)
                )
            ):
                issues.append("user_self_preference_mixing")
                break
        self_preference_state = getattr(context, "self_preference_state", "unknown")
        self_preference_aspect = getattr(context, "preference_aspect", "")
        if (
            self_preference_state != "stable"
            and _STABLE_SELF_PREFERENCE.search(clean)
        ):
            issues.append("self_preference_unverified")
        elif (
            context.asks_self_preference
            and self_preference_state == "stable"
            and self_preference_aspect in {"", "value"}
            and self_value
            and not self._preference_value_is_present(clean, self_value)
            and not _FIRST_PERSON_UNKNOWN.search(clean)
        ):
            issues.append("self_preference_contradiction")
        elif (
            context.asks_self_preference
            and self_preference_state == "unknown"
            and _SELF_LIKING.search(clean)
            and not _TENTATIVE_SELF.search(clean)
        ):
            issues.append("self_preference_unverified")
        if (
            context.user_states_preference
            and context.self_preference_count == 0
            and _SELF_LIKING.search(clean)
        ):
            issues.append("mirrors_user_preference_as_self")
        if self._mirrors_user_state(clean, context.user_text):
            issues.append("mirrors_user_state_as_self")
        if (
            context.asks_user_preference
            and context.memory_fact_count == 0
            and not context.recent_evidence_match
            and not _FIRST_PERSON_UNKNOWN.search(clean)
        ):
            issues.append("user_preference_unverified")

        words = clean.split()
        if len(words) > self.max_words:
            clean = self._shorten(clean)
            issues.append("length_trimmed")

        blocking = tuple(
            issue
            for issue in issues
            if issue
            not in {
                "length_trimmed",
                "question_echo_removed",
                "user_echo_removed",
                "resolved_question_repetition_removed",
                "unsupported_self_history_removed",
                "generic_tail_removed",
            }
        )
        return ValidationResult(not blocking, clean, tuple(issues))

    def prefix_is_safe(self, text: str) -> bool:
        clean = self.sanitize_prefix(text)
        return (
            bool(clean.strip())
            and not _TECHNICAL_LEAKS.search(clean)
            and not _VISIBLE_INTERNAL_LABEL.search(clean)
            and not _UNVERIFIED_ACTION.search(clean)
            and not _UNBACKED_TASK_COMMITMENT.search(clean)
        )

    @classmethod
    def _preference_value_is_present(cls, text: str, value: str) -> bool:
        ignored = {
            "especialmente",
            "oscuro",
            "oscura",
            "suave",
            "favorito",
            "favorita",
            "prefiero",
        }
        anchors = {
            token
            for token in cls._normalize(value).split()
            if len(token) >= 4 and token not in ignored
        }
        return bool(anchors & set(cls._normalize(text).split()))

    def _shorten(self, text: str) -> str:
        words = text.split()
        clipped = " ".join(words[: self.max_words]).rstrip(" ,;:")
        sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if sentence_end >= max(20, len(clipped) // 2):
            return clipped[: sentence_end + 1]
        return f"{clipped}…"

    def _remove_question_echo(self, response: str, user_text: str) -> tuple[str, bool]:
        response_parts = re.findall(r"[^.!?]+[.!?]?", response)
        user_questions = [
            item
            for item in re.findall(r"[^.!?]+[?]", user_text)
            if item.strip()
        ]
        if not response_parts or not user_questions:
            return response, False
        normalized_questions = [self._normalize(item) for item in user_questions]
        kept: list[str] = []
        removed = False
        for part in response_parts:
            normalized = self._normalize(part)
            is_echo = any(
                normalized
                and SequenceMatcher(None, normalized, question).ratio() >= 0.92
                for question in normalized_questions
            )
            if is_echo:
                removed = True
                continue
            kept.append(part.strip())
        return (" ".join(kept).strip(), removed)

    def _remove_leading_user_echo(
        self,
        response: str,
        user_text: str,
    ) -> tuple[str, bool]:
        candidate = response.strip()
        source = user_text.strip()
        if not candidate or not source:
            return response, False
        if "?" in source or "Â¿" in source:
            return response, False
        if not candidate.casefold().startswith(source.casefold()):
            return response, False
        remainder = candidate[len(source) :].lstrip(" \t\r\n,;:-")
        return remainder, True

    def _claims_physical_experience(self, text: str) -> bool:
        normalized = self._normalize(text)
        for match in _UNSUPPORTED_PHYSICAL_EXPERIENCE.finditer(normalized):
            prefix = normalized[max(0, match.start() - 55) : match.start()]
            if re.search(
                r"(?:\bno\b|\bnunca\b|\bsin\b).{0,35}$|"
                r"\b(?:fingir|afirmar|decir)\s+que.{0,20}$",
                prefix,
            ):
                continue
            return True
        return False

    def _mirrors_user_state(self, response: str, user_text: str) -> bool:
        user_match = _FIRST_PERSON_STATE.search(self._normalize(user_text))
        response_match = _FIRST_PERSON_STATE.search(self._normalize(response))
        if user_match is None or response_match is None:
            return False
        ignored = {"ahora", "algo", "eso", "esto", "aqui", "ahi"}
        user_terms = set(user_match.group(1).split()) - ignored
        response_terms = set(response_match.group(1).split()) - ignored
        return bool(
            user_terms
            and len(user_terms & response_terms) >= min(2, len(user_terms))
        )

    def _remove_generic_tail(self, response: str) -> tuple[str, bool]:
        parts = [
            item.strip()
            for item in re.findall(r"[^.!?]+[.!?]?", response)
            if item.strip()
        ]
        if not parts:
            return response, False
        kept = [
            item
            for item in parts
            if not _GENERIC_ASSISTANT_TAIL.search(item)
            and not _KNOWN_USER_NAME_QUESTION.search(item)
        ]
        if len(kept) == len(parts):
            return response, False
        return (" ".join(kept).strip(), True)

    @staticmethod
    def _normalize(text: str) -> str:
        folded = "".join(
            character
            for character in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(character)
        )
        return " ".join(re.findall(r"[a-z0-9]+", folded))
