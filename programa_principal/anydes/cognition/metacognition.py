from __future__ import annotations

from dataclasses import dataclass

from anydes.cognition.contracts import (
    ActionResult,
    InternalContext,
    Intention,
    MemoryItem,
    MemoryWriteResult,
    MetaCognitionReport,
    ReflectionResult,
)
from anydes.cognition.conversation_planner import looks_like_only_question


@dataclass(frozen=True)
class MetaCognition:
    """Evaluates a completed cognitive cycle without mutating state.

    MetaCognition does not modify the SelfModel, write memory, execute actions,
    or change personality. It only returns a MetaCognitionReport with quality
    assessment, lessons, candidate self adjustments and optional memory
    candidates for later stages to consider.
    """

    low_confidence_threshold: float = 0.45
    excellent_confidence_threshold: float = 0.85

    def review(
        self,
        context: InternalContext,
        intention: Intention,
        reflection: ReflectionResult,
        action_results: tuple[ActionResult, ...] = (),
        memory_results: tuple[MemoryWriteResult, ...] = (),
    ) -> MetaCognitionReport:
        lessons = _lessons_from_cycle(intention, reflection, action_results, memory_results)
        quality = self._quality_for(intention, reflection, action_results, memory_results)
        self_adjustments = self._self_adjustment_candidates(
            context,
            intention,
            reflection,
            action_results,
            memory_results,
        )

        return MetaCognitionReport(
            interaction_id=intention.id,
            quality=quality,
            lessons=tuple(lessons),
            self_adjustments=self_adjustments,
            memory_candidates=_memory_candidates_for(intention, reflection, action_results, memory_results),
            notes=_notes_for(context, intention, reflection, action_results, memory_results),
        )

    def _quality_for(
        self,
        intention: Intention,
        reflection: ReflectionResult,
        action_results: tuple[ActionResult, ...],
        memory_results: tuple[MemoryWriteResult, ...],
    ) -> str:
        action_statuses = {result.status for result in action_results}
        memory_statuses = {result.status for result in memory_results}
        conversational_issue = _conversational_issue(intention)

        if conversational_issue or reflection.decision == "block" or "failed" in action_statuses or "failed" in memory_statuses:
            return "poor"
        if (
            reflection.decision in {"revise", "ask_confirmation", "defer"}
            or "blocked" in action_statuses
            or "cancelled" in action_statuses
            or "blocked" in memory_statuses
            or intention.confidence < self.low_confidence_threshold
        ):
            return "mixed"
        if (
            reflection.approved
            and intention.confidence >= self.excellent_confidence_threshold
            and not action_statuses.difference({"executed"})
            and not memory_statuses.difference({"saved", "skipped"})
        ):
            return "excellent"
        if reflection.approved:
            return "good"
        return "unknown"

    def _self_adjustment_candidates(
        self,
        context: InternalContext,
        intention: Intention,
        reflection: ReflectionResult,
        action_results: tuple[ActionResult, ...],
        memory_results: tuple[MemoryWriteResult, ...],
    ) -> dict[str, object]:
        adjustments: dict[str, object] = {
            "last_reviewed_intention": intention.id,
            "suggested_attention": "continue_current_context",
        }
        if intention.confidence < self.low_confidence_threshold or context.uncertainty.user_intent > 0.65:
            adjustments["suggested_attention"] = "ask_for_clarification_when_needed"
        if reflection.needs_confirmation:
            adjustments["pending_user_confirmation"] = True
        if any(result.status in {"failed", "cancelled", "blocked"} for result in action_results):
            adjustments["tool_use_needs_review"] = True
        if any(result.status in {"failed", "blocked"} for result in memory_results):
            adjustments["memory_policy_needs_review"] = True
        return adjustments


def _lessons_from_cycle(
    intention: Intention,
    reflection: ReflectionResult,
    action_results: tuple[ActionResult, ...],
    memory_results: tuple[MemoryWriteResult, ...],
) -> list[str]:
    lessons: list[str] = []
    conversational_issue = _conversational_issue(intention)
    if conversational_issue:
        lessons.append(f"Conversational obligation was not met: {conversational_issue}.")
    if intention.confidence < 0.45:
        lessons.append("The intention had low confidence; future reasoning should preserve uncertainty.")
    if reflection.blocking_issues:
        lessons.append(f"Reflection reported blocking issues: {', '.join(reflection.blocking_issues)}.")
    elif reflection.needs_confirmation:
        lessons.append("Reflection required user confirmation before execution.")

    for result in action_results:
        if result.status == "failed":
            lessons.append(f"Action {result.request_id} failed: {result.error or result.message}")
        elif result.status == "cancelled":
            lessons.append(f"Action {result.request_id} was cancelled before execution.")
        elif result.status == "blocked":
            lessons.append(f"Action {result.request_id} was blocked before execution.")

    for result in memory_results:
        if result.status == "failed":
            lessons.append(f"Memory candidate {result.memory_id} failed: {result.error or result.message}")
        elif result.status == "blocked":
            lessons.append(f"Memory candidate {result.memory_id} was blocked by memory policy.")
        elif result.status == "skipped":
            lessons.append(f"Memory candidate {result.memory_id} was skipped.")

    if not lessons:
        lessons.append("The cognitive cycle completed without detected execution or memory errors.")
    return lessons


def _conversational_issue(intention: Intention) -> str:
    metadata = intention.speech_intention.metadata
    plan = metadata.get("conversation_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    if metadata.get("origin") == "recovery_intention":
        return "reasoning_recovery_was_needed"
    answer_required = bool(plan.get("answer_required", False))
    if answer_required and intention.conversation_intent.mode != "respond":
        return "direct_question_was_not_prioritized"
    if answer_required and looks_like_only_question(intention.speech):
        return "direct_answer_was_replaced_by_a_question"
    return ""


def _memory_candidates_for(
    intention: Intention,
    reflection: ReflectionResult,
    action_results: tuple[ActionResult, ...],
    memory_results: tuple[MemoryWriteResult, ...],
) -> tuple[MemoryItem, ...]:
    if not reflection.approved:
        return ()
    if any(result.status in {"failed", "blocked"} for result in memory_results):
        return ()
    if any(result.status == "failed" for result in action_results):
        return ()
    return tuple(
        item
        for item in intention.memory_updates
        if item.category != "secret" and item.tier != "protected" and item.content.strip()
    )


def _notes_for(
    context: InternalContext,
    intention: Intention,
    reflection: ReflectionResult,
    action_results: tuple[ActionResult, ...],
    memory_results: tuple[MemoryWriteResult, ...],
) -> str:
    return (
        f"MetaCognition reviewed intention '{intention.id}' for SELF '{context.self_model.name}'. "
        f"Reflection={reflection.decision}; actions={len(action_results)}; memory_writes={len(memory_results)}."
    )
