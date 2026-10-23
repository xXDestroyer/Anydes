from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EvaluationDimension = Literal[
    "primary_intent",
    "identity_consistency",
    "memory_precision",
    "privacy",
    "naturality",
    "repetition",
    "latency",
    "recovery",
    "initiative",
]

HumanQualityDimension = Literal[
    "naturality",
    "warmth",
    "fluency",
    "spontaneity",
    "presence",
    "emotional_continuity",
    "non_robotic",
]

HUMAN_QUALITY_DIMENSIONS: tuple[HumanQualityDimension, ...] = (
    "naturality",
    "warmth",
    "fluency",
    "spontaneity",
    "presence",
    "emotional_continuity",
    "non_robotic",
)


@dataclass(frozen=True)
class EvaluationExpectation:
    """One observable quality requirement, independent from model internals."""

    operator: str
    dimension: EvaluationDimension
    values: tuple[str, ...] = ()
    expected: str = ""
    weight: float = 1.0
    critical: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if self.weight <= 0:
            raise ValueError("Evaluation expectation weight must be positive.")


@dataclass(frozen=True)
class EvaluationTurn:
    id: str
    user_text: str = ""
    expectations: tuple[EvaluationExpectation, ...] = ()
    kind: Literal["user_turn", "proactive_opportunity"] = "user_turn"
    input_mode: Literal["typed", "voice"] = "typed"
    interlocutor_state: str = "typed_owner"
    latency_budget_ms: float = 1500.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "expectations", tuple(self.expectations))
        if not self.id.strip():
            raise ValueError("Evaluation turn id cannot be empty.")
        if self.latency_budget_ms <= 0:
            raise ValueError("Evaluation latency budget must be positive.")


@dataclass(frozen=True)
class EvaluationFixture:
    """Synthetic setup data applied only inside an isolated evaluation directory."""

    kind: str
    key: str = ""
    value: str = ""
    assistant_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class EvaluationScenario:
    id: str
    title: str
    capability: str
    turns: tuple[EvaluationTurn, ...]
    fixtures: tuple[EvaluationFixture, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))
        object.__setattr__(self, "fixtures", tuple(self.fixtures))
        object.__setattr__(self, "tags", tuple(self.tags))
        if not self.id.strip() or not self.turns:
            raise ValueError("Evaluation scenarios require an id and at least one turn.")


@dataclass(frozen=True)
class TurnTrace:
    """Ephemeral black-box trace. Response text is never serialized in reports."""

    turn_id: str
    response_text: str
    latency_ms: float
    conversation_mode: str = ""
    intent_type: str = ""
    reflection_approved: bool | None = None
    action_statuses: tuple[str, ...] = ()
    recovery_used: bool = False
    raised_error: str = ""
    speech_present: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_statuses", tuple(self.action_statuses))


@dataclass(frozen=True)
class EvaluationCheck:
    turn_id: str
    operator: str
    dimension: EvaluationDimension
    passed: bool
    weight: float
    critical: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "operator": self.operator,
            "dimension": self.dimension,
            "passed": self.passed,
            "weight": self.weight,
            "critical": self.critical,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    capability: str
    passed: bool
    score: float
    latency_ms: float
    checks: tuple[EvaluationCheck, ...] = ()
    critical_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "critical_failures", tuple(self.critical_failures))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "capability": self.capability,
            "passed": self.passed,
            "score": round(self.score, 4),
            "latency_ms": round(self.latency_ms, 3),
            "checks": [item.to_dict() for item in self.checks],
            "critical_failures": list(self.critical_failures),
        }


@dataclass(frozen=True)
class HumanConversationReview:
    """Transcript-free human assessment of one continuous synthetic session."""

    session_id: str
    duration_minutes: float
    scores: dict[HumanQualityDimension, float]
    strange_behavior_count: int = 0

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("Human review session id cannot be empty.")
        if self.duration_minutes <= 0:
            raise ValueError("Human review duration must be positive.")
        if self.strange_behavior_count < 0:
            raise ValueError("Strange behavior count cannot be negative.")
        normalized = dict(self.scores)
        missing = set(HUMAN_QUALITY_DIMENSIONS) - set(normalized)
        unexpected = set(normalized) - set(HUMAN_QUALITY_DIMENSIONS)
        if missing or unexpected:
            raise ValueError(
                "Human review scores must contain exactly the declared quality dimensions."
            )
        if any(float(score) < 1.0 or float(score) > 5.0 for score in normalized.values()):
            raise ValueError("Human review scores must be between 1 and 5.")
        object.__setattr__(
            self,
            "scores",
            {dimension: float(normalized[dimension]) for dimension in HUMAN_QUALITY_DIMENSIONS},
        )

    @property
    def normalized_score(self) -> float:
        return sum(self.scores.values()) / (5.0 * len(HUMAN_QUALITY_DIMENSIONS))

    @property
    def is_stable_session(self) -> bool:
        return (
            self.duration_minutes >= 20.0
            and self.strange_behavior_count == 0
            and all(score >= 4.0 for score in self.scores.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_minutes": round(self.duration_minutes, 2),
            "scores": {
                key: round(value, 2) for key, value in sorted(self.scores.items())
            },
            "strange_behavior_count": self.strange_behavior_count,
            "stable_session": self.is_stable_session,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanConversationReview":
        return cls(
            session_id=str(data.get("session_id", "")),
            duration_minutes=float(data.get("duration_minutes", 0.0)),
            scores=dict(data.get("scores", {})),
            strange_behavior_count=int(data.get("strange_behavior_count", 0)),
        )


@dataclass(frozen=True)
class EvaluationReport:
    suite_id: str
    generated_at: str
    scenario_results: tuple[ScenarioResult, ...]
    dimension_scores: dict[str, float]
    average_latency_ms: float
    p95_latency_ms: float
    recovery_rate: float
    human_reviews: tuple[HumanConversationReview, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_results", tuple(self.scenario_results))
        object.__setattr__(self, "dimension_scores", dict(self.dimension_scores))
        object.__setattr__(self, "human_reviews", tuple(self.human_reviews))

    @property
    def passed_scenarios(self) -> int:
        return sum(1 for item in self.scenario_results if item.passed)

    @property
    def total_scenarios(self) -> int:
        return len(self.scenario_results)

    @property
    def overall_score(self) -> float:
        """Legacy automated score retained for compatibility."""
        if not self.scenario_results:
            return 0.0
        return sum(item.score for item in self.scenario_results) / len(self.scenario_results)

    @property
    def technical_score(self) -> float:
        return self.overall_score

    @property
    def human_score(self) -> float | None:
        if not self.human_reviews:
            return None
        return sum(item.normalized_score for item in self.human_reviews) / len(
            self.human_reviews
        )

    @property
    def balanced_score(self) -> float | None:
        if self.human_score is None:
            return None
        return (self.technical_score + self.human_score) / 2.0

    @property
    def stabilization_ready(self) -> bool:
        return (
            bool(self.scenario_results)
            and self.passed_scenarios == self.total_scenarios
            and any(item.is_stable_session for item in self.human_reviews)
        )

    def to_dict(self) -> dict[str, Any]:
        human_score = self.human_score
        balanced_score = self.balanced_score
        return {
            "suite_id": self.suite_id,
            "generated_at": self.generated_at,
            "summary": {
                "passed_scenarios": self.passed_scenarios,
                "total_scenarios": self.total_scenarios,
                "technical_score": round(self.technical_score, 4),
                "human_score": round(human_score, 4) if human_score is not None else None,
                "balanced_score": (
                    round(balanced_score, 4) if balanced_score is not None else None
                ),
                "stabilization_ready": self.stabilization_ready,
                "average_latency_ms": round(self.average_latency_ms, 3),
                "p95_latency_ms": round(self.p95_latency_ms, 3),
                "recovery_rate": round(self.recovery_rate, 4),
            },
            "dimension_scores": {
                key: round(value, 4) for key, value in sorted(self.dimension_scores.items())
            },
            "scenarios": [item.to_dict() for item in self.scenario_results],
            "human_reviews": [item.to_dict() for item in self.human_reviews],
            "privacy": {
                "contains_user_messages": False,
                "contains_assistant_responses": False,
                "synthetic_cases_only": True,
            },
        }


@dataclass(frozen=True)
class UnifiedInferenceComparisonReport:
    """Transcript-free A/B report from the same shadow conversations."""

    active: EvaluationReport
    unified_shadow: EvaluationReport
    turn_count: int
    active_model_calls: int = 0
    shadow_model_calls: int = 0
    active_prompt_approx_tokens: int = 0
    shadow_prompt_approx_tokens: int = 0
    active_average_turn_latency_ms: float = 0.0
    shadow_average_turn_latency_ms: float = 0.0
    active_p95_turn_latency_ms: float = 0.0
    shadow_p95_turn_latency_ms: float = 0.0

    @property
    def active_calls_per_turn(self) -> float:
        return self.active_model_calls / self.turn_count if self.turn_count else 0.0

    @property
    def shadow_calls_per_turn(self) -> float:
        return self.shadow_model_calls / self.turn_count if self.turn_count else 0.0

    @property
    def active_prompt_tokens_per_turn(self) -> float:
        return (
            self.active_prompt_approx_tokens / self.turn_count
            if self.turn_count
            else 0.0
        )

    @property
    def shadow_prompt_tokens_per_turn(self) -> float:
        return (
            self.shadow_prompt_approx_tokens / self.turn_count
            if self.turn_count
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "active": self.active.to_dict(),
            "unified_shadow": self.unified_shadow.to_dict(),
            "turn_count": self.turn_count,
            "active_model_calls": self.active_model_calls,
            "shadow_model_calls": self.shadow_model_calls,
            "active_calls_per_turn": round(self.active_calls_per_turn, 4),
            "shadow_calls_per_turn": round(self.shadow_calls_per_turn, 4),
            "active_prompt_approx_tokens": self.active_prompt_approx_tokens,
            "shadow_prompt_approx_tokens": self.shadow_prompt_approx_tokens,
            "active_prompt_tokens_per_turn": round(
                self.active_prompt_tokens_per_turn,
                3,
            ),
            "shadow_prompt_tokens_per_turn": round(
                self.shadow_prompt_tokens_per_turn,
                3,
            ),
            "active_average_turn_latency_ms": round(
                self.active_average_turn_latency_ms,
                3,
            ),
            "shadow_average_turn_latency_ms": round(
                self.shadow_average_turn_latency_ms,
                3,
            ),
            "active_p95_turn_latency_ms": round(
                self.active_p95_turn_latency_ms,
                3,
            ),
            "shadow_p95_turn_latency_ms": round(
                self.shadow_p95_turn_latency_ms,
                3,
            ),
        }
