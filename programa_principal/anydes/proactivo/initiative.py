from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from anydes.cognition import Intention
from anydes.nucleo.config import Config

if TYPE_CHECKING:
    from anydes.nucleo.assistant import AnydesAssistant


@dataclass(frozen=True)
class ProactiveMessage:
    text: str
    reason: str
    intention: Intention | None = None


class ProactiveEngine:
    """Offers a quiet moment to the cognitive pipeline; it owns no messages."""

    def __init__(self, config: Config, assistant: AnydesAssistant | None = None) -> None:
        self.config = config
        self._assistant = assistant

    def build_message(self, now: datetime | None = None) -> ProactiveMessage:
        intention = self._runtime_assistant().initiate(now)
        if intention is None:
            return ProactiveMessage(
                text="",
                reason="Reasoning Engine chose silence or Reflection rejected the initiative.",
            )
        return ProactiveMessage(
            text=intention.speech,
            reason=intention.conversation_intent.reason or intention.reasoning_summary,
            intention=intention,
        )

    def _runtime_assistant(self) -> AnydesAssistant:
        if self._assistant is None:
            from anydes.cerebro.llm import build_brain, build_vision_brain
            from anydes.funciones_extra.registry import ToolRegistry
            from anydes.nucleo.assistant import AnydesAssistant
            from anydes.nucleo.audit import AuditLog
            from anydes.nucleo.permissions import PermissionManager
            from anydes.nucleo.personality import PersonalityProfile

            self._assistant = AnydesAssistant(
                config=self.config,
                tools=ToolRegistry.default(self.config),
                permissions=PermissionManager(),
                audit=AuditLog(self.config.audit_log_path),
                personality=PersonalityProfile.load(self.config.personality_prompt_path),
                brain=build_brain(self.config),
                vision_brain=build_vision_brain(self.config),
            )
        return self._assistant
