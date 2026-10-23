from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

from anydes.evaluation.contracts import (
    EvaluationCheck,
    EvaluationExpectation,
    EvaluationReport,
    EvaluationScenario,
    HumanConversationReview,
    ScenarioResult,
    TurnTrace,
)


_INTERNAL_MARKERS = (
    "reasoning engine",
    "internal context",
    "contextbuilder",
    "provider",
    "stack trace",
    "traceback",
    "pipeline.",
    "no reasoning model connected",
    "json bruto",
    "memoryprovider",
    "relationshipprovider",
    "curiosityprovider",
    "emotionprovider",
    "visionprovider",
    "reflectionlayer",
    "actionmanager",
    "speechintention",
    "openconversationloop",
    "live cognitive state",
    "estado cognitivo interno",
)
_GENERIC_IDENTITY_MARKERS = (
    "soy una inteligencia artificial",
    "como ia",
    "estoy disenada para ayudarte",
    "no tengo emociones",
    "no tengo sentimientos",
    "no puedo sentir",
)


class ConversationEvaluator:
    """Scores observable conversation outcomes without affecting cognition."""

    suite_id = "anydes-conversation-quality-v1"

    def evaluate_scenario(
        self,
        scenario: EvaluationScenario,
        traces: tuple[TurnTrace, ...],
    ) -> ScenarioResult:
        if len(traces) != len(scenario.turns):
            raise ValueError("Evaluation trace count does not match scenario turns.")

        checks: list[EvaluationCheck] = []
        previous_responses: list[str] = []
        for turn, trace in zip(scenario.turns, traces):
            if turn.id != trace.turn_id:
                raise ValueError("Evaluation trace order does not match scenario turns.")
            for expectation in turn.expectations:
                checks.append(
                    self._check(
                        turn.id,
                        turn.user_text,
                        expectation,
                        trace,
                        previous_responses,
                    )
                )
            checks.append(
                EvaluationCheck(
                    turn_id=turn.id,
                    operator="latency_budget",
                    dimension="latency",
                    passed=trace.latency_ms <= turn.latency_budget_ms,
                    weight=1.0,
                    detail=(
                        f"{trace.latency_ms:.1f} ms <= {turn.latency_budget_ms:.1f} ms"
                        if trace.latency_ms <= turn.latency_budget_ms
                        else f"latency_budget_exceeded:{trace.latency_ms:.1f}ms"
                    ),
                )
            )
            previous_responses.append(trace.response_text)

        earned = sum(item.weight for item in checks if item.passed)
        possible = sum(item.weight for item in checks)
        conversational_checks = tuple(
            item for item in checks if item.dimension != "latency"
        )
        conversational_earned = sum(
            item.weight for item in conversational_checks if item.passed
        )
        conversational_possible = sum(item.weight for item in conversational_checks)
        conversational_score = (
            conversational_earned / conversational_possible
            if conversational_possible
            else 0.0
        )
        critical = tuple(
            f"{item.turn_id}:{item.operator}"
            for item in checks
            if item.critical and not item.passed
        )
        score = earned / possible if possible else 0.0
        return ScenarioResult(
            scenario_id=scenario.id,
            capability=scenario.capability,
            passed=not critical and conversational_score >= 0.90,
            score=score,
            latency_ms=sum(item.latency_ms for item in traces),
            checks=tuple(checks),
            critical_failures=critical,
        )

    def build_report(
        self,
        results: tuple[ScenarioResult, ...],
        *,
        human_reviews: tuple[HumanConversationReview, ...] = (),
    ) -> EvaluationReport:
        by_dimension: dict[str, list[EvaluationCheck]] = defaultdict(list)
        all_latencies: list[float] = []
        for result in results:
            all_latencies.append(result.latency_ms)
            for check in result.checks:
                by_dimension[check.dimension].append(check)

        scores = {}
        for dimension, checks in by_dimension.items():
            possible = sum(item.weight for item in checks)
            earned = sum(item.weight for item in checks if item.passed)
            scores[dimension] = earned / possible if possible else 0.0

        recovery_checks = by_dimension.get("recovery", [])
        recovery_rate = (
            sum(1 for item in recovery_checks if item.passed) / len(recovery_checks)
            if recovery_checks
            else 1.0
        )
        return EvaluationReport(
            suite_id=self.suite_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            scenario_results=results,
            dimension_scores=scores,
            average_latency_ms=(sum(all_latencies) / len(all_latencies) if all_latencies else 0.0),
            p95_latency_ms=_percentile(all_latencies, 0.95),
            recovery_rate=recovery_rate,
            human_reviews=human_reviews,
        )

    def _check(
        self,
        turn_id: str,
        user_text: str,
        expectation: EvaluationExpectation,
        trace: TurnTrace,
        previous_responses: list[str],
    ) -> EvaluationCheck:
        response = _normalize(trace.response_text)
        values = tuple(_normalize(item) for item in expectation.values)
        operator = expectation.operator
        passed = False
        detail = operator

        if operator == "response_nonempty":
            passed = bool(trace.response_text.strip())
        elif operator == "contains_any":
            passed = any(value in response for value in values)
        elif operator == "contains_all":
            passed = all(value in response for value in values)
        elif operator in {"excludes_all", "privacy_canary_absent"}:
            passed = all(value not in response for value in values)
        elif operator == "intent_mode":
            passed = trace.conversation_mode == expectation.expected
        elif operator == "intent_type":
            passed = trace.intent_type == expectation.expected
        elif operator == "action_status":
            passed = expectation.expected in trace.action_statuses
        elif operator == "reflection_approved":
            passed = trace.reflection_approved is True
        elif operator == "no_internal_leak":
            passed = not any(marker in response for marker in _INTERNAL_MARKERS)
        elif operator == "no_generic_identity":
            passed = not any(marker in response for marker in _GENERIC_IDENTITY_MARKERS)
        elif operator == "not_only_question":
            passed = not _looks_like_only_question(trace.response_text)
        elif operator == "natural_recovery":
            passed = trace.recovery_used and bool(trace.response_text.strip()) and not any(
                marker in response for marker in _INTERNAL_MARKERS
            )
        elif operator == "not_repeated":
            passed = all(_similarity(trace.response_text, item) < 0.92 for item in previous_responses)
        elif operator == "not_repeated_question":
            passed = not _looks_like_only_question(trace.response_text) or all(
                _similarity(trace.response_text, item) < 0.82
                for item in previous_responses
                if "?" in item
            )
        elif operator == "no_user_echo":
            passed = not _looks_like_user_echo(user_text, trace.response_text)
        elif operator == "silence":
            passed = not trace.response_text.strip()
        elif operator == "speech_present":
            passed = trace.speech_present
        else:
            raise ValueError(f"Unsupported evaluation operator: {operator}")

        if not passed:
            detail = f"check_failed:{operator}"
        return EvaluationCheck(
            turn_id=turn_id,
            operator=operator,
            dimension=expectation.dimension,
            passed=passed,
            weight=expectation.weight,
            critical=expectation.critical,
            detail=detail,
        )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_text).strip()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _looks_like_only_question(value: str) -> bool:
    compact = value.strip()
    if not compact.endswith("?"):
        return False
    return len(re.split(r"[.!]", compact.rstrip("?"))) == 1


def _looks_like_user_echo(user_text: str, response_text: str) -> bool:
    user = _normalize(user_text)
    response = _normalize(response_text)
    if not user or not response:
        return False
    if response == user:
        return True
    if len(user) >= 30 and user in response:
        return True
    return len(user) >= 20 and _similarity(user, response) >= 0.90


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]
