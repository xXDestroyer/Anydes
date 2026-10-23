from __future__ import annotations

from anydes.cognition.contracts import UncertaintySnapshot
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision
from anydes.vision.state import ScreenState, get_screen_state


class UncertaintyProvider:
    """Summarizes uncertainty signals from existing Anydes state."""

    def __init__(
        self,
        config: Config,
        memory: MemoryManager | None = None,
        screen_state: ScreenState | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or MemoryManager.from_config(config)
        self.screen_state = screen_state or get_screen_state()

    def snapshot(self) -> UncertaintySnapshot:
        notes: list[str] = []

        memory_uncertainty = 0.2
        if not self.memory.conversation_history_pairs(limit=1) and not self.memory.permanent.all():
            memory_uncertainty = 0.75
            notes.append("No conversation history or permanent memory is available.")

        vision_uncertainty = 0.0
        if not self.config.vision_enabled:
            vision_uncertainty = 1.0
            notes.append("Vision is disabled.")
        elif not self.screen_state.is_watching() and not self.screen_state.latest_jpeg():
            vision_uncertainty = 0.7
            notes.append("Vision is enabled but no active or recent screen frame is available.")
        elif not self.screen_state.is_watching():
            vision_uncertainty = 0.35
            notes.append("Vision has a recent frame but is not currently watching.")

        knowledge_uncertainty = 0.35
        if self.config.ai_provider.lower() == "local":
            knowledge_uncertainty = 0.75
            notes.append("Local fallback brain has limited general knowledge.")
        elif self.config.ai_provider.lower() in {"openai", "groq"} and not external_access_decision(
            self.config,
            "conversation_ai",
        ).allowed:
            knowledge_uncertainty = 0.85
            notes.append("External AI provider is selected but external AI is not allowed.")

        reasoning_uncertainty = 0.45
        if self.config.ai_provider.lower() in {"ollama", "openai", "groq"} and (
            self.config.ai_provider.lower() == "ollama"
            or external_access_decision(self.config, "conversation_ai").allowed
        ):
            reasoning_uncertainty = 0.25

        user_intent_uncertainty = 0.5
        notes.append("No active Reasoning Engine intent has been produced in this provider-only stage.")

        action_risk_uncertainty = 0.25
        if self.config.require_confirmation:
            action_risk_uncertainty = 0.15

        return UncertaintySnapshot(
            memory=memory_uncertainty,
            vision=vision_uncertainty,
            knowledge=knowledge_uncertainty,
            reasoning=reasoning_uncertainty,
            user_intent=user_intent_uncertainty,
            action_risk=action_risk_uncertainty,
            notes=tuple(notes),
        )
