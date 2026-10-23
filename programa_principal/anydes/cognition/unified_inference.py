from __future__ import annotations

import time
from dataclasses import dataclass, replace

from anydes.cognition.contracts import (
    InternalContext,
    Intention,
    ReflectionResult,
)
from anydes.cognition.conversation_planner import ConversationPlanner
from anydes.cognition.evidence_builder import EvidenceBundleBuilder
from anydes.cognition.reasoning import ReasoningEngine, ReasoningModel
from anydes.cognition.reflection import ReflectionLayer


@dataclass(frozen=True)
class UnifiedInferenceShadowReport:
    """Payload-free comparison between the active and unified intentions."""

    status: str
    cycle: str
    duration_ms: float = 0.0
    inference_calls: int = 0
    matches_structure: bool = False
    changed_sections: tuple[str, ...] = ()
    reflection_decision: str = ""
    reflection_issue_codes: tuple[str, ...] = ()
    candidate_has_speech: bool = False
    candidate_action_count: int = 0
    candidate_memory_update_count: int = 0
    candidate_relationship_update_count: int = 0
    candidate_self_update_count: int = 0
    candidate_curiosity_update_count: int = 0
    candidate_has_grounded_plan: bool = False
    evidence_item_count: int = 0
    evidence_truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cycle": self.cycle,
            "duration_ms": round(self.duration_ms, 3),
            "inference_calls": self.inference_calls,
            "matches_structure": self.matches_structure,
            "changed_sections": list(self.changed_sections),
            "reflection_decision": self.reflection_decision,
            "reflection_issue_codes": list(self.reflection_issue_codes),
            "candidate_has_speech": self.candidate_has_speech,
            "candidate_action_count": self.candidate_action_count,
            "candidate_memory_update_count": self.candidate_memory_update_count,
            "candidate_relationship_update_count": self.candidate_relationship_update_count,
            "candidate_self_update_count": self.candidate_self_update_count,
            "candidate_curiosity_update_count": self.candidate_curiosity_update_count,
            "candidate_has_grounded_plan": self.candidate_has_grounded_plan,
            "evidence_item_count": self.evidence_item_count,
            "evidence_truncated": self.evidence_truncated,
        }


@dataclass
class UnifiedInferenceShadowRunner:
    """Evaluates one complete model thought without applying any side effect."""

    model: ReasoningModel
    planner: ConversationPlanner | None = None
    reflection: ReflectionLayer | None = None
    evidence_builder: EvidenceBundleBuilder | None = None
    error_reporter: object | None = None

    def run(
        self,
        context: InternalContext,
        baseline: Intention,
        *,
        cycle: str,
    ) -> tuple[Intention, ReflectionResult, UnifiedInferenceShadowReport]:
        started = time.perf_counter()
        grounded_context = replace(
            context,
            evidence_bundle=(self.evidence_builder or EvidenceBundleBuilder()).build(
                context
            ),
        )
        engine = ReasoningEngine(
            model=self.model,
            error_reporter=(
                self.error_reporter if callable(self.error_reporter) else None
            ),
            require_grounded_plan=True,
        )
        candidate = engine.reason(grounded_context)
        candidate = _mark_unified_candidate(candidate)

        reflection = (self.reflection or ReflectionLayer()).reflect(
            candidate,
            grounded_context,
        )
        changed = _changed_sections(baseline, candidate)
        recovered = candidate.intent_type in {
            "reasoning_parse_error",
            "reasoning_error",
        }
        report = UnifiedInferenceShadowReport(
            status="recovered" if recovered else "completed",
            cycle=cycle,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            inference_calls=1,
            matches_structure=not changed,
            changed_sections=changed,
            reflection_decision=reflection.decision,
            reflection_issue_codes=tuple(
                _safe_issue_code(item) for item in reflection.blocking_issues
            ),
            candidate_has_speech=bool(candidate.speech.strip()),
            candidate_action_count=len(candidate.actions),
            candidate_memory_update_count=len(candidate.memory_updates),
            candidate_relationship_update_count=len(candidate.relationship_updates),
            candidate_self_update_count=len(candidate.self_updates),
            candidate_curiosity_update_count=len(candidate.curiosity_updates),
            candidate_has_grounded_plan=candidate.grounded_plan is not None,
            evidence_item_count=len(grounded_context.evidence_bundle.items),
            evidence_truncated=grounded_context.evidence_bundle.selection_budget.truncated,
        )
        return candidate, reflection, report


def unavailable_shadow_report(cycle: str) -> UnifiedInferenceShadowReport:
    return UnifiedInferenceShadowReport(status="unavailable", cycle=cycle)


def _mark_unified_candidate(intention: Intention) -> Intention:
    metadata = dict(intention.speech_intention.metadata)
    metadata["origin"] = "native_reasoning"
    metadata["inference_architecture"] = "unified"
    speech = replace(intention.speech_intention, metadata=metadata)
    return replace(
        intention,
        speech=speech.text,
        speech_intention=speech,
    )


def _changed_sections(
    baseline: Intention,
    candidate: Intention,
) -> tuple[str, ...]:
    changed: list[str] = []
    if baseline.intent_type != candidate.intent_type:
        changed.append("intent_type")
    if baseline.conversation_intent.mode != candidate.conversation_intent.mode:
        changed.append("conversation_mode")
    if tuple(item.action_type for item in baseline.actions) != tuple(
        item.action_type for item in candidate.actions
    ):
        changed.append("action_types")
    for name in (
        "avatar",
        "memory_updates",
        "relationship_updates",
        "self_updates",
        "curiosity_updates",
    ):
        if bool(getattr(baseline, name)) != bool(getattr(candidate, name)):
            changed.append(name)
    if bool(baseline.speech.strip()) != bool(candidate.speech.strip()):
        changed.append("speech_presence")
    return tuple(changed)


def _safe_issue_code(issue: str) -> str:
    return str(issue).partition(":")[0][:80]
