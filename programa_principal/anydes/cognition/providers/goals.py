from __future__ import annotations

from anydes.cognition.contracts import Goal, GoalSnapshot
from anydes.nucleo.config import Config


class GoalProvider:
    """Builds permanent and temporary goal snapshots from current Anydes configuration."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def snapshot(self) -> GoalSnapshot:
        return GoalSnapshot(
            permanent=self.permanent_goals(),
            temporary=self.temporary_goals(),
        )

    def permanent_goals(self) -> tuple[Goal, ...]:
        return (
            Goal(
                id="permanent:privacy",
                tier="permanent",
                description="Protect DemoUser's privacy and keep sensitive data local unless explicitly allowed.",
                priority=1.0,
                success_criteria=(
                    "Do not expose protected memory without authorization.",
                    "Do not send data to external providers unless configured.",
                ),
            ),
            Goal(
                id="permanent:transparency",
                tier="permanent",
                description=(
                    "Answer honestly when DemoUser explicitly asks about Anydes' nature, limitations or capabilities, "
                    "without volunteering identity disclaimers in ordinary conversation."
                ),
                priority=0.95,
                success_criteria=(
                    "Distinguish transparency from self-presentation.",
                    "Do not interrupt natural conversation with generic AI disclaimers.",
                ),
            ),
            Goal(
                id="permanent:usefulness",
                tier="permanent",
                description="Help DemoUser with practical, context-aware and conversational support.",
                priority=0.9,
            ),
            Goal(
                id="permanent:safety",
                tier="permanent",
                description="Require confirmation for sensitive, external or device-affecting actions.",
                priority=1.0,
            ),
            Goal(
                id="permanent:continuity",
                tier="permanent",
                description="Preserve continuity through memory while respecting privacy boundaries.",
                priority=0.85,
            ),
        )

    def temporary_goals(self) -> tuple[Goal, ...]:
        goals: list[Goal] = [
            Goal(
                id="temporary:current_architecture_migration",
                tier="temporary",
                description="Make Anydes converse naturally from Identity, relationship, memory and current internal state.",
                priority=0.95,
                success_criteria=(
                    "Opening the room never emits an automatic presentation.",
                    "Greetings are reasoned from current context rather than fixed templates.",
                    "Relationship continuity influences conversation without overriding privacy.",
                ),
            )
        ]

        if self.config.proactive_enabled:
            goals.append(
                Goal(
                    id="temporary:proactive_mode",
                    tier="temporary",
                    description="Allow Anydes to initiate useful interactions while active and configured to do so.",
                    priority=0.65,
                    metadata={
                        "interval_seconds": self.config.proactive_interval_seconds,
                        "minute_marks": self.config.proactive_minute_marks,
                    },
                )
            )

        if self.config.vision_enabled:
            goals.append(
                Goal(
                    id="temporary:vision_context",
                    tier="temporary",
                    description="Use screen or visual context as information for future reasoning when vision is enabled.",
                    priority=0.7,
                    metadata={
                        "provider": self.config.resolved_vision_provider(),
                        "model": self.config.resolved_vision_model(),
                        "live_mode": self.config.screen_live_mode,
                    },
                )
            )

        if self.config.voice_activation.lower() != "button":
            goals.append(
                Goal(
                    id="temporary:voice_activation",
                    tier="temporary",
                    description="Maintain the currently selected voice activation mode.",
                    priority=0.55,
                    metadata={
                        "activation": self.config.voice_activation,
                        "wake_word": self.config.wake_word,
                    },
                )
            )

        return tuple(goals)
