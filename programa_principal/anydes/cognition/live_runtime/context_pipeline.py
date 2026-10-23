from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from anydes.cognition.contracts import (
    CuriosityModel,
    CuriosityPrompt,
    BodyStateSnapshot,
    EmotionSnapshot,
    GoalSnapshot,
    InternalContext,
    MemorySnapshot,
    Observation,
    SelfModel,
    UncertaintySnapshot,
)
from anydes.cognition.live_runtime.contracts import LiveCognitiveSnapshot
from anydes.identity.relationship_contracts import RelationshipSnapshot
from anydes.nucleo.config import Config
from anydes.cognition.live_runtime.metrics import record_current_stage


ContextProfile = Literal["standard", "relationship", "visual", "environment", "tool"]

_CORE_PROVIDERS = ("memory", "self", "goals", "emotion", "uncertainty", "curiosity", "body")
_PROFILE_PROVIDERS: dict[ContextProfile, tuple[str, ...]] = {
    "standard": _CORE_PROVIDERS + ("relationship",),
    "relationship": _CORE_PROVIDERS + ("relationship",),
    "visual": _CORE_PROVIDERS + ("relationship", "vision"),
    "environment": _CORE_PROVIDERS + ("relationship", "environment"),
    "tool": _CORE_PROVIDERS + ("relationship",),
}

# Routing vocabulary selects useful evidence only. It never selects a response or action.
_PROFILE_SIGNALS: dict[ContextProfile, tuple[str, ...]] = {
    "visual": (
        "pantalla",
        "ves",
        "viendo",
        "imagen",
        "foto",
        "camara",
        "monitor",
        "ventana abierta",
    ),
    "environment": (
        "clima",
        "lluvia",
        "lloviendo",
        "tormenta",
        "soleado",
        "nublado",
        "nieve",
        "temperatura",
        "amanecer",
        "atardecer",
    ),
    "relationship": (
        "recuerdas",
        "acuerdas",
        "promesa",
        "pendiente",
        "nosotros",
        "relacion",
        "hablamos",
        "proyecto anydes",
    ),
}


@dataclass(frozen=True)
class ContextSelection:
    """A deterministic evidence-selection request, not a cognitive decision."""

    profile: ContextProfile
    provider_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ContextAssembly:
    """InternalContext plus non-sensitive provenance about its live snapshot."""

    context: InternalContext
    root_revision: int
    selected_provider_ids: tuple[str, ...]
    used_provider_ids: tuple[str, ...]
    missing_provider_ids: tuple[str, ...]
    stale_provider_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContextShadowDiff:
    """Structural parity result. It intentionally contains no context values."""

    matches: bool
    changed_sections: tuple[str, ...] = ()


class ContextSelectionPolicy:
    """Chooses the smallest useful snapshot profile from interaction evidence."""

    def select(self, observations: tuple[Observation, ...]) -> ContextSelection:
        if any(item.source == "tool" for item in observations):
            return self._selection("tool", "structured_tool_observation")

        text = _normalized_user_text(observations)
        scores = {
            profile: sum(_signal_matches(text, signal) for signal in signals)
            for profile, signals in _PROFILE_SIGNALS.items()
        }
        best_profile = max(scores, key=scores.get)
        if scores[best_profile] > 0:
            return self._selection(best_profile, f"interaction_signal:{best_profile}")
        return self._selection("standard", "default_interaction_context")

    @staticmethod
    def _selection(profile: ContextProfile, reason: str) -> ContextSelection:
        return ContextSelection(
            profile=profile,
            provider_ids=_PROFILE_PROVIDERS[profile],
            reason=reason,
        )


class InternalContextAssembler:
    """Builds InternalContext from one immutable root without calling Providers."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def assemble(
        self,
        snapshot: LiveCognitiveSnapshot,
        selection: ContextSelection,
        observations: tuple[Observation, ...],
    ) -> ContextAssembly:
        selected = set(selection.provider_ids)
        used: list[str] = []
        missing: list[str] = []
        stale: list[str] = []

        def read(provider_id: str, expected_type: type | tuple[type, ...], default: Any) -> Any:
            if provider_id not in selected:
                return default
            item = snapshot.get(provider_id)
            if item is None or not isinstance(item.value, expected_type):
                missing.append(provider_id)
                return default
            used.append(provider_id)
            if item.freshness_at() == "stale":
                stale.append(provider_id)
            return item.value

        memory_started = time.perf_counter()
        try:
            memory = read("memory", MemorySnapshot, MemorySnapshot())
        finally:
            record_current_stage(
                "pipeline.memory_search",
                time.perf_counter() - memory_started,
            )
        self_model = read(
            "self",
            SelfModel,
            SelfModel(
                name=self.config.profile_name,
                kind="digital entity",
                identity_summary="Identity snapshot is temporarily unavailable.",
            ),
        )
        goals = read("goals", GoalSnapshot, GoalSnapshot())
        emotion = read("emotion", EmotionSnapshot, EmotionSnapshot())
        uncertainty = read(
            "uncertainty",
            UncertaintySnapshot,
            UncertaintySnapshot(
                memory=1.0,
                vision=1.0,
                knowledge=0.5,
                reasoning=0.5,
                user_intent=0.5,
                notes=("Live uncertainty snapshot unavailable.",),
            ),
        )
        curiosity_value = read("curiosity", (tuple, list), ())
        curiosity = tuple(
            item
            for item in curiosity_value
            if isinstance(item, (CuriosityModel, CuriosityPrompt))
        )
        relationship = read(
            "relationship",
            RelationshipSnapshot,
            RelationshipSnapshot(
                available=False,
                warnings=("Live relationship snapshot unavailable.",),
            ),
        )
        body = read("body", BodyStateSnapshot, BodyStateSnapshot())
        live_observations = (
            _observation_slice(read("vision", (tuple, list), ()))
            + _observation_slice(read("environment", (tuple, list), ()))
        )
        assembled_observations = _deduplicate_observations(
            tuple(observations) + live_observations
        )
        context = InternalContext(
            observations=assembled_observations,
            memory=memory,
            goals=goals,
            self_model=self_model,
            emotion=emotion,
            uncertainty=uncertainty,
            curiosity=curiosity,
            relationship=relationship,
            body=body,
            available_actions=(),
            timestamp=snapshot.published_at.isoformat(),
        )
        return ContextAssembly(
            context=context,
            root_revision=snapshot.revision,
            selected_provider_ids=selection.provider_ids,
            used_provider_ids=tuple(used),
            missing_provider_ids=tuple(missing),
            stale_provider_ids=tuple(stale),
        )


def compare_contexts(candidate: InternalContext, legacy: InternalContext) -> ContextShadowDiff:
    """Compare sections without retaining or returning their potentially private values."""

    candidate_data = candidate.to_dict()
    legacy_data = legacy.to_dict()
    sections = sorted(set(candidate_data) | set(legacy_data))
    changed = tuple(
        section
        for section in sections
        if _fingerprint(candidate_data.get(section)) != _fingerprint(legacy_data.get(section))
    )
    return ContextShadowDiff(matches=not changed, changed_sections=changed)


def _normalized_user_text(observations: tuple[Observation, ...]) -> str:
    text = " ".join(
        str(item.data.get("text") or item.summary)
        for item in observations
        if item.source == "user_text"
    )
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _signal_matches(text: str, signal: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(signal)}(?!\w)", text) is not None


def _observation_slice(value: tuple[Any, ...] | list[Any]) -> tuple[Observation, ...]:
    return tuple(item for item in value if isinstance(item, Observation))


def _deduplicate_observations(observations: tuple[Observation, ...]) -> tuple[Observation, ...]:
    unique: dict[str, Observation] = {}
    for observation in observations:
        unique.setdefault(observation.id, observation)
    return tuple(unique.values())


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
