from __future__ import annotations

from dataclasses import dataclass

from anydes.cognition.contracts import InternalContext, Observation
from anydes.cognition.body import BodyStateProvider
from anydes.cognition.providers import (
    CuriosityProvider,
    EmotionProvider,
    EnvironmentProvider,
    GoalProvider,
    MemoryProvider,
    RelationshipProvider,
    SelfProvider,
    UncertaintyProvider,
    VisionProvider,
)
from anydes.nucleo.config import Config
from anydes.cognition.live_runtime.metrics import NullMetricsCollector, PipelineProfiler


@dataclass
class ContextProviders:
    memory: MemoryProvider
    self_model: SelfProvider
    goals: GoalProvider
    emotion: EmotionProvider
    uncertainty: UncertaintyProvider
    curiosity: CuriosityProvider
    vision: VisionProvider
    environment: EnvironmentProvider
    relationship: RelationshipProvider
    body: BodyStateProvider


class ContextBuilder:
    """Builds InternalContext by composing context providers.

    This class does not reason, answer, execute actions or modify Assistant.respond().
    It only gathers provider snapshots into the agreed cognitive contract.
    """

    def __init__(
        self,
        config: Config,
        providers: ContextProviders | None = None,
        *,
        profiler: PipelineProfiler | None = None,
    ) -> None:
        self.config = config
        self.profiler = profiler or PipelineProfiler(NullMetricsCollector())
        self.providers = providers or ContextProviders(
            memory=MemoryProvider(config),
            self_model=SelfProvider(config),
            goals=GoalProvider(config),
            emotion=EmotionProvider(config),
            uncertainty=UncertaintyProvider(config),
            curiosity=CuriosityProvider(config),
            vision=VisionProvider(config),
            environment=EnvironmentProvider(config),
            relationship=RelationshipProvider(config),
            body=BodyStateProvider.from_config(config),
        )

    def build(self, observations: tuple[Observation, ...] | list[Observation] = ()) -> InternalContext:
        observation_tuple = (
            tuple(observations)
            + self.providers.vision.observations()
            + self.providers.environment.observations()
        )
        return InternalContext(
            observations=observation_tuple,
            memory=self.profiler.call(
                "pipeline.memory_search",
                self.providers.memory.snapshot,
            ),
            goals=self.providers.goals.snapshot(),
            self_model=self.providers.self_model.snapshot(),
            emotion=self.providers.emotion.snapshot(),
            uncertainty=self.providers.uncertainty.snapshot(),
            curiosity=self.providers.curiosity.snapshot(),
            relationship=self.providers.relationship.snapshot(),
            body=self.providers.body.snapshot(),
            available_actions=(),
        )


def user_text_observation(
    text: str,
    *,
    observation_id: str = "obs:user_text",
    silence_allowed: bool | None = None,
) -> Observation:
    data: dict[str, object] = {"text": text}
    if silence_allowed is not None:
        data["silence_allowed"] = silence_allowed
    return Observation(
        id=observation_id,
        source="user_text",
        summary=text,
        confidence=1.0,
        salience=0.8 if text.strip() else 0.0,
        data=data,
    )
