from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from anydes.cognition.contracts import (
    BodyStateSnapshot,
    EmotionSnapshot,
    GoalSnapshot,
    MemorySnapshot,
    Observation,
    SelfModel,
    UncertaintySnapshot,
)
from anydes.cognition.body import BodyStateProvider
from anydes.cognition.live_runtime.contracts import ProviderSample, ProviderUpdatePolicy
from anydes.cognition.live_runtime.providers import FunctionalLiveProvider
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
from anydes.identity.relationship_contracts import RelationshipSnapshot
from anydes.nucleo.config import Config


def build_current_provider_adapters(
    config: Config,
) -> tuple[FunctionalLiveProvider[Any], ...]:
    """Adapt current Context Providers without changing their implementations.

    This is a migration boundary only. ``Assistant.respond`` continues using
    the legacy ContextBuilder until the live pipeline reaches parity.
    """

    # Concrete providers are constructed lazily so simulation never touches
    # user stores, hardware state or network-capable implementations.
    memory_read = _lazy_method(lambda: MemoryProvider(config), "snapshot")
    self_read = _lazy_method(lambda: SelfProvider(config), "snapshot")
    goals_read = _lazy_method(lambda: GoalProvider(config), "snapshot")
    emotion_read = _lazy_method(lambda: EmotionProvider(config), "snapshot")
    uncertainty_read = _lazy_method(lambda: UncertaintyProvider(config), "snapshot")
    curiosity_read = _lazy_method(lambda: CuriosityProvider(config), "snapshot")
    vision_read = _lazy_method(lambda: VisionProvider(config), "observations")
    environment_read = _lazy_method(lambda: EnvironmentProvider(config), "observations")
    relationship_read = _lazy_method(lambda: RelationshipProvider(config), "snapshot")
    body_read = _lazy_method(lambda: BodyStateProvider.from_config(config), "snapshot")

    return (
        _adapter(
            "memory",
            policy=ProviderUpdatePolicy(update_mode="event"),
            live_read=memory_read,
            simulated_read=lambda: MemorySnapshot(
                protected_summary="Simulated protected memory is unavailable."
            ),
        ),
        _adapter(
            "self",
            policy=ProviderUpdatePolicy(update_mode="event"),
            live_read=self_read,
            simulated_read=lambda: SelfModel(
                name=config.profile_name,
                kind="simulated digital entity",
                identity_summary="Simulated SELF state for local runtime verification.",
                personality_traits=("calm", "warm", "curious"),
                mutable_state={"simulation": True},
                body_state={"simulation": True, "expression": "neutral"},
            ),
        ),
        _adapter(
            "goals",
            policy=ProviderUpdatePolicy(update_mode="event"),
            live_read=goals_read,
            simulated_read=GoalSnapshot,
        ),
        _adapter(
            "emotion",
            policy=ProviderUpdatePolicy(
                update_mode="hybrid",
                interval_seconds=30.0,
                fresh_for_seconds=45.0,
                usable_for_seconds=120.0,
            ),
            live_read=emotion_read,
            simulated_read=lambda: EmotionSnapshot(
                energy=0.62,
                mood=0.2,
                interest=0.58,
                tension=0.08,
                affection=0.64,
                dominant_label="calm",
                summary="Stable simulated emotional state.",
            ),
        ),
        _adapter(
            "uncertainty",
            policy=ProviderUpdatePolicy(
                update_mode="hybrid",
                interval_seconds=30.0,
                fresh_for_seconds=60.0,
                usable_for_seconds=180.0,
            ),
            live_read=uncertainty_read,
            simulated_read=lambda: UncertaintySnapshot(
                memory=0.2,
                vision=0.3,
                knowledge=0.3,
                reasoning=0.25,
                user_intent=0.4,
                action_risk=0.1,
                notes=("All uncertainty values are simulated.",),
            ),
        ),
        _adapter(
            "curiosity",
            policy=ProviderUpdatePolicy(
                update_mode="hybrid",
                interval_seconds=300.0,
                fresh_for_seconds=600.0,
                usable_for_seconds=1800.0,
            ),
            live_read=curiosity_read,
            simulated_read=tuple,
        ),
        _adapter(
            "vision",
            policy=ProviderUpdatePolicy(
                update_mode="periodic",
                interval_seconds=config.resolved_screen_watch_interval_seconds(),
                fresh_for_seconds=max(2.0, config.resolved_screen_watch_interval_seconds() * 2.0),
                usable_for_seconds=30.0,
                timeout_seconds=5.0,
                priority=70,
            ),
            live_read=vision_read,
            simulated_read=lambda: (_simulated_vision_observation(),),
        ),
        _adapter(
            "environment",
            policy=ProviderUpdatePolicy(
                update_mode="periodic",
                interval_seconds=600.0,
                fresh_for_seconds=900.0,
                usable_for_seconds=14400.0,
                timeout_seconds=5.0,
            ),
            # The safe initial value avoids network I/O during runtime startup.
            live_initial=lambda: (_simulated_environment_observation(initial_fallback=True),),
            live_read=environment_read,
            simulated_read=lambda: (_simulated_environment_observation(),),
        ),
        _adapter(
            "relationship",
            policy=ProviderUpdatePolicy(update_mode="event"),
            live_read=relationship_read,
            simulated_read=RelationshipSnapshot,
        ),
        _adapter(
            "body",
            policy=ProviderUpdatePolicy(update_mode="event"),
            live_read=body_read,
            simulated_read=lambda: BodyStateSnapshot(
                available=True,
                state="neutral",
                expression="neutral",
                gesture="idle",
                gaze="user",
                model_3d={"simulated": True},
            ),
        ),
    )


def _adapter(
    provider_id: str,
    *,
    policy: ProviderUpdatePolicy,
    live_read: Callable[[], Any],
    simulated_read: Callable[[], Any],
    live_initial: Callable[[], Any] | None = None,
) -> FunctionalLiveProvider[Any]:
    initial_reader = live_initial or live_read
    return FunctionalLiveProvider(
        provider_id=provider_id,
        policy=policy,
        live_initial=lambda: ProviderSample(value=initial_reader()),
        live_update=lambda request: ProviderSample(value=live_read()),
        simulated_initial=lambda: ProviderSample(
            value=simulated_read(),
            metadata={"simulated": True},
        ),
        simulated_update=lambda request: ProviderSample(
            value=simulated_read(),
            metadata={"simulated": True},
        ),
    )


def _lazy_method(factory: Callable[[], Any], method_name: str) -> Callable[[], Any]:
    instance: Any | None = None

    def read() -> Any:
        nonlocal instance
        if instance is None:
            instance = factory()
        return getattr(instance, method_name)()

    return read


def _simulated_vision_observation() -> Observation:
    return Observation(
        id="vision:simulated",
        source="vision",
        summary="Simulated visual context is available without a camera or screen capture.",
        confidence=0.75,
        salience=0.2,
        data={
            "simulated": True,
            "watching": False,
            "has_recent_frame": False,
        },
    )


def _simulated_environment_observation(*, initial_fallback: bool = False) -> Observation:
    moment = datetime.now().astimezone()
    return Observation(
        id="environment:initial_fallback" if initial_fallback else "environment:simulated",
        source="environment",
        summary="A calm simulated room environment is available.",
        confidence=0.65 if initial_fallback else 0.75,
        salience=0.15,
        data={
            "simulated": not initial_fallback,
            "initial_fallback": initial_fallback,
            "time_of_day": "neutral",
            "weather": "calm",
            "local_time": moment.isoformat(),
        },
    )
