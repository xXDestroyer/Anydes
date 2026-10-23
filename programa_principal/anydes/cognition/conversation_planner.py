from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from anydes.cognition.contracts import (
    ConversationAct,
    ConversationIntent,
    ConversationPlan,
    InternalContext,
    Intention,
)


_QUESTION_OPENERS = frozenset(
    {
        "como",
        "cual",
        "cuales",
        "cuando",
        "cuanto",
        "cuanta",
        "cuantos",
        "cuantas",
        "donde",
        "por que",
        "que",
        "quien",
        "quienes",
    }
)


@dataclass(frozen=True)
class ConversationPlanner:
    """Orders conversational acts without generating speech or taking actions.

    The planner protects obligations created by the user's current turn. Open
    curiosity remains an opportunity and cannot displace a direct answer.
    """

    def protect_intent(
        self,
        user_text: str,
        proposed: ConversationIntent,
        *,
        interaction_kind: str = "user_turn",
    ) -> ConversationIntent:
        if interaction_kind != "user_turn" or not _is_direct_question(user_text):
            return proposed
        if _is_request_for_a_question(user_text, proposed):
            return proposed
        if proposed.mode == "respond" and not proposed.curiosity_id:
            return replace(
                proposed,
                target_topic=_compact_topic(user_text),
                reason=proposed.reason or "La pregunta directa del turno actual es la obligacion principal.",
            )
        return replace(
            proposed,
            mode="respond",
            target_topic=_compact_topic(user_text),
            reason="La pregunta directa del turno actual debe responderse antes de explorar otros temas.",
            curiosity_id="",
        )

    def plan(self, context: InternalContext, intention: Intention) -> ConversationPlan:
        observation = next(
            (item for item in reversed(context.observations) if item.source in {"user_text", "voice"}),
            None,
        )
        user_text = str(observation.data.get("text", observation.summary)) if observation else ""
        protected = self.protect_intent(user_text, intention.conversation_intent)
        direct_question = bool(user_text and _is_direct_question(user_text))
        deferred = ()
        if direct_question and intention.conversation_intent.curiosity_id and not protected.curiosity_id:
            deferred = (intention.conversation_intent.curiosity_id,)
        primary = ConversationAct(
            id=f"act:{intention.id}:primary",
            mode=protected.mode,
            target_topic=protected.target_topic,
            priority=1.0 if direct_question else 0.78,
            required=direct_question,
            source="user_request" if direct_question else "reasoning",
            curiosity_id=protected.curiosity_id,
            reason=protected.reason,
        )
        secondary = ()
        if (
            protected.mode not in {"ask", "clarify"}
            and "?" in intention.speech
            and not looks_like_only_question(intention.speech)
        ):
            secondary = (
                ConversationAct(
                    id=f"act:{intention.id}:secondary-question",
                    mode="ask",
                    target_topic=protected.target_topic,
                    priority=0.35,
                    required=False,
                    source="reasoning",
                    reason="The reasoned response contains an optional follow-up after its answer.",
                ),
            )
        return ConversationPlan(
            observation_id=observation.id if observation else "",
            primary=primary,
            secondary=secondary,
            deferred_curiosity_ids=deferred,
            direct_question=direct_question,
            rationale=(
                "A direct question owns the primary conversational act."
                if direct_question
                else "The reasoned conversational act remains primary."
            ),
        )

    def apply(self, intention: Intention, plan: ConversationPlan) -> Intention:
        current = intention.conversation_intent
        conversation_intent = replace(
            current,
            mode=plan.primary.mode,
            target_topic=plan.primary.target_topic,
            reason=plan.primary.reason,
            curiosity_id=plan.primary.curiosity_id,
        )
        metadata = dict(intention.speech_intention.metadata)
        metadata["conversation_plan"] = {
            "observation_id": plan.observation_id,
            "primary_mode": plan.primary.mode,
            "direct_question": plan.direct_question,
            "answer_required": plan.primary.required,
            "secondary_modes": [item.mode for item in plan.secondary],
            "deferred_curiosity_ids": list(plan.deferred_curiosity_ids),
        }
        speech = replace(intention.speech_intention, metadata=metadata)
        return replace(
            intention,
            conversation_intent=conversation_intent,
            speech=intention.speech,
            speech_intention=speech,
        )


def is_direct_question(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if "?" in text or "¿" in text:
        return True
    if "?" in text or "\u00bf" in text:
        return True
    words = normalized.split()
    for width in (2, 1):
        if " ".join(words[:width]) in _QUESTION_OPENERS:
            return True
    return False


_is_direct_question = is_direct_question


def _is_request_for_a_question(text: str, proposed: ConversationIntent) -> bool:
    if proposed.mode != "ask":
        return False
    normalized = _normalize(text)
    request_patterns = (
        r"\b(hazme|haceme|formulame)\s+(una\s+)?pregunta\b",
        r"\b(preguntame|pregunta me)\b",
        r"\bquiero\s+que\s+me\s+preguntes\b",
        r"\bpuedes\s+(hacerme|preguntarme)\b",
    )
    return any(re.search(pattern, normalized) for pattern in request_patterns)


def _compact_topic(text: str) -> str:
    compact = " ".join(text.strip().split())
    return compact[:240]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("\u00bf", " ").replace("\u00c2", " ")
    return re.sub(r"[^a-z0-9?¿]+", " ", ascii_text).strip()


def looks_like_only_question(text: str) -> bool:
    """Return whether speech appears to contain a question without an answer first."""
    had_inverted_question = text.lstrip().startswith("\u00bf")
    text = text.lstrip("\u00bf\u00c2").strip()
    stripped = text.strip().lstrip("¿").strip()
    if not stripped.endswith("?"):
        return False
    normalized = _normalize(stripped).replace("?", "").replace("¿", "").strip()
    if had_inverted_question:
        return True
    openers = (
        "como ", "cual ", "cuales ", "cuando ", "cuanto ", "donde ",
        "por que ", "que ", "quien ", "te gustaria ", "quieres ",
        "y tu", "que tal",
    )
    if normalized.startswith(openers):
        return True

    clauses = [
        _normalize(item).replace("?", "").strip()
        for item in re.split(r"[,;:\n]+", stripped)
    ]
    clauses = [item for item in clauses if item]
    if not clauses or not clauses[-1].startswith(openers):
        return False
    conversational_prefaces = (
        "ah",
        "aja",
        "bueno",
        "hey",
        "hola",
        "oye",
    )
    return all(
        clause in conversational_prefaces
        or clause.startswith(tuple(f"{prefix} " for prefix in conversational_prefaces))
        for clause in clauses[:-1]
    )
