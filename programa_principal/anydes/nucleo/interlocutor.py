from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal


InterlocutorState = Literal[
    "typed_owner",
    "verified_voice",
    "unknown_voice",
    "verification_unavailable",
]
InputChannel = Literal["typed", "voice"]


@dataclass(frozen=True)
class InterlocutorContext:
    """Identity boundary for one interaction, independent from speaker biometrics."""

    state: InterlocutorState
    channel: InputChannel
    display_name: str = ""
    confidence: float = 0.0
    owner_verified: bool = False
    reason: str = ""

    @property
    def allows_personal_memory(self) -> bool:
        return self.owner_verified and self.state in {"typed_owner", "verified_voice"}

    @property
    def allows_owner_state_mutation(self) -> bool:
        return self.allows_personal_memory

    @property
    def prompt_context(self) -> str:
        if self.state == "typed_owner":
            return "Interlocutor: entrada escrita local tratada como propietario por politica explicita."
        if self.state == "verified_voice" and self.owner_verified:
            return (
                "Interlocutor: voz reconocida como el propietario configurado. "
                "La coincidencia acustica permite contexto personal, pero no autoriza acciones sensibles."
            )
        if self.state == "verified_voice":
            return (
                "Interlocutor: voz conocida, pero no corresponde al propietario configurado. "
                "No uses memoria personal ni relacional de DemoUser."
            )
        if self.state == "unknown_voice":
            return "Interlocutor: voz desconocida. No uses memoria personal, protegida ni relacional."
        return (
            "Interlocutor: la verificacion de identidad no esta disponible. "
            "No uses memoria personal, protegida ni relacional."
        )

    @classmethod
    def typed(cls, owner_name: str, *, trusted_by_policy: bool) -> "InterlocutorContext":
        if trusted_by_policy:
            return cls(
                state="typed_owner",
                channel="typed",
                display_name=owner_name,
                confidence=1.0,
                owner_verified=True,
                reason="local_typed_input_policy",
            )
        return cls(
            state="verification_unavailable",
            channel="typed",
            reason="typed_input_owner_policy_disabled",
        )

    @classmethod
    def from_voice_match(cls, match, *, owner_name: str) -> "InterlocutorContext":
        if match is None:
            return cls.verification_unavailable(channel="voice")
        if not bool(getattr(match, "matched", False)):
            return cls(
                state="unknown_voice",
                channel="voice",
                confidence=_confidence(match),
                reason="speaker_profile_not_matched",
            )
        name = str(getattr(match, "name", "") or "").strip()
        is_owner = bool(name) and _normalize_name(name) == _normalize_name(owner_name)
        return cls(
            state="verified_voice",
            channel="voice",
            display_name=name,
            confidence=_confidence(match),
            owner_verified=is_owner,
            reason="owner_voice_profile_matched" if is_owner else "non_owner_voice_profile_matched",
        )

    @classmethod
    def verification_unavailable(
        cls,
        *,
        channel: InputChannel = "voice",
        reason: str = "speaker_verification_unavailable",
    ) -> "InterlocutorContext":
        return cls(
            state="verification_unavailable",
            channel=channel,
            reason=reason,
        )


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char)).strip()


def _confidence(match) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(match, "confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0
