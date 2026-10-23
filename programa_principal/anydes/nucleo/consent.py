from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from anydes.nucleo.config import Config


ExternalCapability = Literal[
    "conversation_ai",
    "external_vision",
    "weather",
    "external_tts",
    "external_stt",
]
SessionCapability = Literal["screen_observation", "external_vision"]

EXTERNAL_PROVIDERS = frozenset({"openai", "openai_vision", "groq", "edge"})


class SessionConsentStore:
    """Volatile grants that are intentionally lost when the process closes."""

    def __init__(self) -> None:
        self._grants: set[str] = set()
        self._lock = threading.RLock()

    def grant(self, capability: SessionCapability) -> None:
        with self._lock:
            self._grants.add(capability)

    def revoke(self, capability: SessionCapability) -> None:
        with self._lock:
            self._grants.discard(capability)

    def revoke_all(self) -> None:
        with self._lock:
            self._grants.clear()

    def allows(self, capability: SessionCapability) -> bool:
        with self._lock:
            return capability in self._grants

    def snapshot(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._grants)


@dataclass(frozen=True)
class ExternalAccessDecision:
    allowed: bool
    reason: str


def external_access_decision(
    config: Config,
    capability: ExternalCapability,
    *,
    require_session: bool = False,
    consents: SessionConsentStore | None = None,
) -> ExternalAccessDecision:
    if not config.network_enabled:
        return ExternalAccessDecision(False, "external_network_disabled")
    configured = {
        "conversation_ai": config.external_ai_allowed,
        "external_vision": config.external_vision_allowed,
        "weather": config.weather_allowed,
        "external_tts": config.external_tts_allowed,
        "external_stt": config.external_stt_allowed,
    }[capability]
    if not configured:
        return ExternalAccessDecision(False, f"{capability}_not_allowed")
    if require_session and not (consents or session_consents).allows("external_vision"):
        return ExternalAccessDecision(False, "external_vision_session_consent_required")
    return ExternalAccessDecision(True, "allowed")


def is_external_provider(provider: str) -> bool:
    return provider.strip().lower() in EXTERNAL_PROVIDERS


session_consents = SessionConsentStore()
