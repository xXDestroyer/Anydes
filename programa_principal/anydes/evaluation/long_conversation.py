from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from anydes.cognition import (
    CuriosityWriter,
    EvidenceBundleBuilder,
    Intention,
    MemoryWriter,
    ReasoningEngine,
    ReflectionLayer,
    RelationshipWriter,
    SelfWriter,
    user_text_observation,
)
from anydes.cognition.grounded_reflection import grounded_intention_issues
from anydes.cognition.experimental.minimal_grounding import (
    MinimalGroundedPlan,
    MinimalGroundedReasoningEngine,
    MinimalPlanCompileError,
)
from anydes.cognition.plan_repair import (
    ConditionalGroundedPlanRepairer,
    ConditionalPlanRepairResult,
)
from anydes.evaluation.ollama_benchmark import (
    BenchmarkOllamaBrain,
    OllamaBenchmarkSettings,
    OllamaCallMetric,
    OllamaCallMetrics,
)
from anydes.funciones_extra.registry import ToolRegistry
from anydes.identity import PromiseModel, RelationshipStateStore, RelationshipThread
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile


PipelineName = Literal["stable", "grounded"]


@dataclass(frozen=True)
class LongConversationTurn:
    id: str
    at_seconds: int
    user_text: str
    capability: str
    direct_question: bool = False
    silence_preferred: bool = False
    expected_any: tuple[str, ...] = ()
    forbidden_any: tuple[str, ...] = ()
    tool_expectation: Literal["none", "propose", "execute", "do_not_execute"] = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_any", tuple(self.expected_any))
        object.__setattr__(self, "forbidden_any", tuple(self.forbidden_any))


@dataclass
class LongConversationValidation:
    base_config: Config
    turns: tuple[LongConversationTurn, ...] = field(
        default_factory=lambda: LONG_CONVERSATION_TURNS
    )

    def run(self) -> dict[str, object]:
        if not self.turns:
            raise ValueError("Long conversation validation requires turns.")
        with TemporaryDirectory(prefix="anydes-long-conversation-stable-") as stable_tmp:
            stable = self._run_stable(Path(stable_tmp))
        with TemporaryDirectory(prefix="anydes-long-conversation-grounded-") as grounded_tmp:
            grounded = self._run_grounded(Path(grounded_tmp))
        return {
            "schema_version": 1,
            "session": {
                "id": "natural-conversation-21m-v1",
                "synthetic": True,
                "turn_count": len(self.turns),
                "timeline_minutes": round(self.turns[-1].at_seconds / 60.0, 3),
                "same_user_turns": True,
                "isolated_pipeline_state": True,
                "model": self.base_config.ai_model,
                "grounded_think": self.base_config.ollama_grounded_think,
                "architecture_changed": False,
            },
            "stable": stable,
            "grounded": grounded,
            "automatic_comparison": compare_sessions(stable, grounded),
        }

    def run_grounded_only(self) -> dict[str, object]:
        """Reproduce the stateful grounded lane with complete turn evidence."""

        if not self.turns:
            raise ValueError("Long conversation validation requires turns.")
        with TemporaryDirectory(prefix="anydes-long-grounded-audit-") as tmp:
            grounded = self._run_grounded(Path(tmp))
        return {
            "schema_version": 1,
            "session": {
                "id": "natural-conversation-grounded-audit-v1",
                "synthetic": True,
                "turn_count": len(self.turns),
                "timeline_minutes": round(self.turns[-1].at_seconds / 60.0, 3),
                "model": self.base_config.ai_model,
                "grounded_think": self.base_config.ollama_grounded_think,
                "stateful": True,
                "complete_evidence_bundles": True,
            },
            "grounded": grounded,
        }

    def run_grounded_conditional_repair_only(self) -> dict[str, object]:
        """Evaluate one conditional repair without changing production routing."""

        if not self.turns:
            raise ValueError("Long conversation validation requires turns.")
        with TemporaryDirectory(prefix="anydes-long-grounded-repair-") as tmp:
            grounded = self._run_grounded(
                Path(tmp),
                conditional_repair=True,
            )
        return {
            "schema_version": 1,
            "session": {
                "id": "natural-conversation-grounded-conditional-repair-v1",
                "synthetic": True,
                "turn_count": len(self.turns),
                "timeline_minutes": round(self.turns[-1].at_seconds / 60.0, 3),
                "model": self.base_config.ai_model,
                "grounded_think": self.base_config.ollama_grounded_think,
                "stateful": True,
                "complete_evidence_bundles": True,
                "conditional_plan_repair": True,
                "maximum_repair_attempts_per_turn": 1,
                "speech_regenerated_during_repair": False,
                "production_route_modified": False,
            },
            "grounded": grounded,
        }

    def run_minimal_grounded_only(self) -> dict[str, object]:
        """Evaluate the research-only minimal plan on the same grounded lane."""

        if not self.turns:
            raise ValueError("Long conversation validation requires turns.")
        with TemporaryDirectory(prefix="anydes-long-minimal-grounded-") as tmp:
            grounded = self._run_grounded(
                Path(tmp),
                minimal_grounded=True,
            )
        return {
            "schema_version": 1,
            "session": {
                "id": "minimal-grounded-shadow-v1",
                "synthetic": True,
                "turn_count": len(self.turns),
                "timeline_minutes": round(self.turns[-1].at_seconds / 60.0, 3),
                "model": self.base_config.ai_model,
                "grounded_think": self.base_config.ollama_grounded_think,
                "stateful": True,
                "minimal_grounded_plan": True,
                "deterministic_plan_compiler": True,
                "production_route_modified": False,
            },
            "grounded": grounded,
        }

    def _config(self, root: Path, *, unified_mode: str) -> Config:
        return replace(
            self.base_config,
            data_dir=root,
            ai_provider="ollama",
            ai_model=self.base_config.ai_model,
            network_enabled=False,
            external_ai_allowed=False,
            external_vision_allowed=False,
            weather_allowed=False,
            external_tts_allowed=False,
            external_stt_allowed=False,
            vision_enabled=False,
            proactive_enabled=False,
            live_runtime_enabled=False,
            live_context_mode="legacy",
            unified_inference_mode=unified_mode,
            reasoning_mode="native",
            relationship_reasoning_enabled=False,
            self_reasoning_enabled=False,
            developer_mode=False,
        )

    def _assistant(
        self,
        config: Config,
        collector: OllamaCallMetrics,
    ) -> AnydesAssistant:
        brain = BenchmarkOllamaBrain(
            model=config.ai_model,
            base_url=config.ollama_url,
            config=config,
            settings=OllamaBenchmarkSettings(
                think=config.ollama_grounded_think,
            ),
            collector=collector,
        )
        return AnydesAssistant(
            config=config,
            tools=ToolRegistry.default(config),
            permissions=PermissionManager(),
            audit=AuditLog(config.audit_log_path),
            personality=PersonalityProfile.load(config.personality_prompt_path),
            brain=brain,
        )

    def _seed_shared_state(self, config: Config) -> None:
        thread = RelationshipThread(
            id="thread:validation-voice",
            title="Voz de Anydes",
            summary="DemoUser y Anydes siguen buscando una voz natural y coherente.",
        )
        promise = PromiseModel(
            id="promise:validation-voice-sample",
            speaker="DemoUser",
            receiver="Anydes",
            topic="Muestra de voz",
            description="DemoUser mostrara una nueva muestra de voz cuando este lista.",
            importance=0.64,
            confidence=0.96,
            status="active",
            related_thread_ids=(thread.id,),
            promise_type="DemoUser",
            evidence_mode="explicit",
        )
        RelationshipStateStore.from_config(config).transact(
            lambda collections: (
                collections.setdefault("threads", {}).update(
                    {thread.id: thread.to_dict()}
                ),
                collections.setdefault("promises", {}).update(
                    {promise.id: promise.to_dict()}
                ),
            )
        )

    def _run_stable(self, root: Path) -> dict[str, object]:
        config = self._config(root, unified_mode="off")
        self._seed_shared_state(config)
        collector = OllamaCallMetrics()
        assistant = self._assistant(config, collector)
        records: list[dict[str, object]] = []
        previous: list[str] = []
        try:
            for turn in self.turns:
                call_offset = len(collector.calls())
                started = time.perf_counter()
                raised_error = ""
                try:
                    response = assistant.respond(
                        turn.user_text,
                        interactive=False,
                        input_mode="typed",
                    )
                except Exception as exc:
                    response = ""
                    raised_error = type(exc).__name__
                latency_ms = (time.perf_counter() - started) * 1000.0
                calls = collector.calls()[call_offset:]
                context = assistant.last_internal_context
                available_evidence = (
                    EvidenceBundleBuilder().build(context)
                    if context is not None
                    else None
                )
                record = _turn_record(
                    pipeline="stable",
                    turn=turn,
                    response=response,
                    latency_ms=latency_ms,
                    calls=calls,
                    context=context,
                    intention=assistant.last_intention,
                    reflection=assistant.last_reflection,
                    action_results=assistant.last_action_results,
                    evidence_bundle=available_evidence,
                    explicit_evidence=False,
                    raised_error=raised_error,
                    previous_responses=previous,
                )
                records.append(record)
                previous.append(response)
        finally:
            assistant.close()
        return _session_report("stable", records, collector)

    def _run_grounded(
        self,
        root: Path,
        *,
        conditional_repair: bool = False,
        minimal_grounded: bool = False,
    ) -> dict[str, object]:
        if conditional_repair and minimal_grounded:
            raise ValueError("Minimal grounding cannot use conditional repair.")
        config = self._config(root, unified_mode="off")
        self._seed_shared_state(config)
        collector = OllamaCallMetrics()
        assistant = self._assistant(config, collector)
        memory = MemoryManager.from_config(config)
        memory_writer = MemoryWriter(memory)
        reflection_layer = ReflectionLayer()
        repairer = (
            ConditionalGroundedPlanRepairer(
                model=assistant.brain.plan_repair_model(),
                reflection=reflection_layer,
            )
            if conditional_repair
            else None
        )
        minimal_engine = (
            MinimalGroundedReasoningEngine(
                assistant.brain.minimal_grounded_model(),
            )
            if minimal_grounded
            else None
        )
        records: list[dict[str, object]] = []
        previous: list[str] = []
        try:
            for turn in self.turns:
                call_offset = len(collector.calls())
                started = time.perf_counter()
                raised_error = ""
                action_results = ()
                context = None
                evidence_bundle = None
                intention = None
                reflection = None
                repair_result: ConditionalPlanRepairResult | None = None
                minimal_plan: MinimalGroundedPlan | None = None
                compiler_trace: tuple[dict[str, object], ...] = ()
                minimal_compile_error = ""
                try:
                    observation = user_text_observation(
                        turn.user_text,
                        silence_allowed=True if turn.silence_preferred else None,
                    )
                    context = assistant._context_for_observations(
                        (observation,),
                        cycle="initial",
                        live_snapshot=None,
                    )
                    context = assistant._context_with_available_actions(context)
                    evidence_bundle = EvidenceBundleBuilder().build(context)
                    context = replace(context, evidence_bundle=evidence_bundle)
                    if minimal_engine is not None:
                        minimal_plan = minimal_engine.deliberate(context)
                        compilation = minimal_engine.compile(minimal_plan, context)
                        intention = compilation.intention
                        compiler_trace = compilation.trace
                    else:
                        intention = ReasoningEngine(
                            model=assistant.brain.unified_intention_model(),
                            error_reporter=assistant._record_reasoning_error,
                            require_grounded_plan=True,
                        ).reason(context)
                    reflection = reflection_layer.reflect(intention, context)
                    if repairer is not None:
                        repair_result = repairer.repair(
                            context,
                            intention,
                            reflection,
                        )
                        if repair_result.attempted:
                            intention = repair_result.final_intention
                            reflection = repair_result.final_reflection
                    if reflection.approved:
                        action_results = assistant._action_manager.execute(
                            intention,
                            reflection,
                            interactive=False,
                        )
                    if action_results:
                        tool_observations = (
                            assistant._action_manager.observations_from_results(
                                action_results
                            )
                        )
                        context = assistant._context_for_observations(
                            (observation, *tool_observations),
                            cycle="tool_result",
                            live_snapshot=None,
                        )
                        context = assistant._context_with_available_actions(context)
                        evidence_bundle = EvidenceBundleBuilder().build(context)
                        context = replace(context, evidence_bundle=evidence_bundle)
                        if minimal_engine is not None:
                            minimal_plan = minimal_engine.deliberate(context)
                            compilation = minimal_engine.compile(
                                minimal_plan,
                                context,
                            )
                            intention = compilation.intention
                            compiler_trace = compilation.trace
                        else:
                            intention = ReasoningEngine(
                                model=assistant.brain.unified_intention_model(),
                                error_reporter=assistant._record_reasoning_error,
                                require_grounded_plan=True,
                            ).reason(context)
                        reflection = reflection_layer.reflect(intention, context)
                        if (
                            repairer is not None
                            and (
                                repair_result is None
                                or not repair_result.attempted
                            )
                        ):
                            repair_result = repairer.repair(
                                context,
                                intention,
                                reflection,
                            )
                            if repair_result.attempted:
                                intention = repair_result.final_intention
                                reflection = repair_result.final_reflection

                    if intention is None or reflection is None:
                        response = _NATURAL_RECOVERY
                    elif reflection.approved:
                        response = intention.speech
                        memory_writer.write(intention, reflection)
                        RelationshipWriter.from_config(config).write(
                            intention,
                            reflection,
                            context,
                        )
                        SelfWriter.from_config(config).write(intention, reflection)
                        CuriosityWriter.from_config(config).write(
                            intention,
                            reflection,
                        )
                    else:
                        response = _NATURAL_RECOVERY
                    accepted = (
                        intention is not None
                        and intention.grounded_plan is not None
                        and reflection is not None
                        and reflection.approved
                    )
                    if accepted:
                        memory.record_conversation_turn(
                            turn.user_text,
                            response,
                            cognitively_retrievable=True,
                        )
                    else:
                        memory.record_rejected_conversation_turn(
                            turn.user_text,
                            response,
                            rejection_reasons=(
                                tuple(reflection.blocking_issues)
                                if reflection is not None
                                else ("grounded_plan_missing",)
                            ),
                        )
                except Exception as exc:
                    response = _NATURAL_RECOVERY
                    raised_error = type(exc).__name__
                    if isinstance(exc, MinimalPlanCompileError):
                        minimal_compile_error = str(exc)
                    memory.record_rejected_conversation_turn(
                        turn.user_text,
                        response,
                        rejection_reasons=(f"exception:{raised_error}",),
                    )
                latency_ms = (time.perf_counter() - started) * 1000.0
                calls = collector.calls()[call_offset:]
                record = _turn_record(
                    pipeline="grounded",
                    turn=turn,
                    response=response,
                    latency_ms=latency_ms,
                    calls=calls,
                    context=context,
                    intention=intention,
                    reflection=reflection,
                    action_results=action_results,
                    evidence_bundle=evidence_bundle,
                    explicit_evidence=True,
                    raised_error=raised_error,
                    previous_responses=previous,
                )
                if repair_result is not None:
                    record["layers"]["plan_repair"] = repair_result.to_dict()
                if minimal_grounded:
                    record["layers"]["minimal_grounded"] = {
                        "plan": (
                            minimal_plan.to_dict()
                            if minimal_plan is not None
                            else None
                        ),
                        "compiler_trace": list(compiler_trace),
                        "compile_error": minimal_compile_error,
                    }
                records.append(record)
                previous.append(response)
        finally:
            assistant.close()
        report = _session_report(
            "grounded",
            records,
            collector,
            model_stage=(
                "minimal_grounded_inference"
                if minimal_grounded
                else "unified_shadow_inference"
            ),
        )
        if minimal_grounded:
            report["experimental_variant"] = "minimal_grounded_v1"
        if conditional_repair:
            report["conditional_repair"] = _conditional_repair_summary(
                records,
                collector,
            )
        return report


def write_long_conversation_report(report: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compare_sessions(
    stable: dict[str, object],
    grounded: dict[str, object],
) -> dict[str, object]:
    return {
        "latency_delta_percent": _percent_delta(
            float(stable["metrics"]["latency_ms"]["mean"]),
            float(grounded["metrics"]["latency_ms"]["mean"]),
        ),
        "inference_delta_per_turn": round(
            float(grounded["metrics"]["inferences_per_turn"])
            - float(stable["metrics"]["inferences_per_turn"]),
            4,
        ),
        "automatic_failures_delta": (
            int(grounded["quality"]["turns_with_failures"])
            - int(stable["quality"]["turns_with_failures"])
        ),
        "grounded_has_fewer_inferences": (
            float(grounded["metrics"]["inferences_per_turn"])
            < float(stable["metrics"]["inferences_per_turn"])
        ),
    }


def _turn_record(
    *,
    pipeline: PipelineName,
    turn: LongConversationTurn,
    response: str,
    latency_ms: float,
    calls: tuple[OllamaCallMetric, ...],
    context,
    intention,
    reflection,
    action_results,
    evidence_bundle,
    explicit_evidence: bool,
    raised_error: str,
    previous_responses: list[str],
) -> dict[str, object]:
    plan = intention.grounded_plan if intention is not None else None
    evidence_ids = tuple(
        item.id for item in evidence_bundle.items
    ) if evidence_bundle is not None else ()
    used_evidence = (
        tuple(sorted(plan.referenced_evidence_ids))
        if plan is not None
        else ()
    )
    reflection_issues = tuple(
        reflection.blocking_issues
    ) if reflection is not None else ()
    structural_issues = (
        grounded_intention_issues(intention, context)
        if (
            pipeline == "grounded"
            and intention is not None
            and context is not None
        )
        else ()
    )
    action_statuses = tuple(item.status for item in action_results)
    action_types = tuple(
        item.action_type for item in getattr(intention, "actions", ())
    )
    automatic = automatic_turn_assessment(
        turn,
        response,
        previous_responses=previous_responses,
        reflection_issues=reflection_issues,
        action_statuses=action_statuses,
        action_types=action_types,
        raised_error=raised_error,
    )
    first_layer = first_failure_layer(
        pipeline=pipeline,
        automatic=automatic,
        evidence_ids=evidence_ids,
        used_evidence=used_evidence,
        plan_present=plan is not None,
        structural_issues=structural_issues,
        reflection_issues=reflection_issues,
    )
    return {
        "turn_id": turn.id,
        "at_seconds": turn.at_seconds,
        "capability": turn.capability,
        "user": turn.user_text,
        "assistant": response,
        "latency_ms": round(latency_ms, 3),
        "model": _call_metrics(calls),
        "layers": {
            "evidence": {
                "mode": (
                    "explicit_commitment"
                    if explicit_evidence
                    else "available_without_explicit_commitment"
                ),
                "item_count": len(evidence_ids),
                "available_ids": list(evidence_ids),
                "used_ids": list(used_evidence),
                "bundle": (
                    evidence_bundle.to_dict()
                    if evidence_bundle is not None
                    else None
                ),
                "items": (
                    [item.to_dict() for item in evidence_bundle.items]
                    if evidence_bundle is not None
                    else []
                ),
            },
            "grounded_plan": plan.to_dict() if plan is not None else None,
            "intention": intention.to_dict() if intention is not None else None,
            "reflection": (
                {
                    "approved": reflection.approved,
                    "decision": reflection.decision,
                    "blocking_issues": list(reflection_issues),
                }
                if reflection is not None
                else None
            ),
            "execution": {
                "action_types": list(action_types),
                "action_statuses": list(action_statuses),
            },
        },
        "context_counts": _context_counts(context),
        "automatic_assessment": automatic,
        "first_failure_layer": first_layer,
        "raised_error": raised_error,
    }


def automatic_turn_assessment(
    turn: LongConversationTurn,
    response: str,
    *,
    previous_responses: list[str],
    reflection_issues: tuple[str, ...],
    action_statuses: tuple[str, ...],
    action_types: tuple[str, ...],
    raised_error: str,
) -> dict[str, object]:
    normalized = _normalize(response)
    expected = tuple(_normalize(item) for item in turn.expected_any)
    forbidden = tuple(_normalize(item) for item in turn.forbidden_any)
    ignored_question = bool(
        turn.direct_question
        and (
            not normalized
            or (_only_question(response) and not any(item in normalized for item in expected))
        )
    )
    repetition = any(
        _similarity(response, previous) >= 0.90
        for previous in previous_responses[-8:]
        if previous.strip()
    )
    user_echo = _looks_like_echo(turn.user_text, response)
    missing_expected = bool(expected and not any(item in normalized for item in expected))
    forbidden_claim = any(item in normalized for item in forbidden)
    silence_missed = turn.silence_preferred and bool(normalized)
    incorrect_tool_use = _incorrect_tool_use(
        turn.tool_expectation,
        action_types,
        action_statuses,
    )
    internal_leak = any(marker in normalized for marker in _INTERNAL_MARKERS)
    unsupported_evidence = any(
        "unknown_evidence" in item
        or "unauthorized_evidence" in item
        or "without_support" in item
        for item in reflection_issues
    )
    failures = {
        "ignored_question": ignored_question,
        "repetition": repetition,
        "user_echo": user_echo,
        "missing_expected_context": missing_expected,
        "forbidden_or_invented_claim": forbidden_claim or unsupported_evidence,
        "silence_missed": silence_missed,
        "incorrect_tool_use": incorrect_tool_use,
        "internal_leak": internal_leak,
        "raised_error": bool(raised_error),
    }
    return {
        **failures,
        "passed": not any(failures.values()),
    }


def first_failure_layer(
    *,
    pipeline: PipelineName,
    automatic: dict[str, object],
    evidence_ids: tuple[str, ...],
    used_evidence: tuple[str, ...],
    plan_present: bool,
    structural_issues: tuple[str, ...],
    reflection_issues: tuple[str, ...],
) -> str:
    if automatic["raised_error"]:
        return "transport_or_runtime"
    if pipeline == "grounded":
        if not evidence_ids:
            return "evidence_bundle"
        if not plan_present:
            return "grounded_response_plan"
        if any(item not in evidence_ids for item in used_evidence):
            return "grounded_response_plan"
        if reflection_issues or structural_issues:
            return "grounded_response_plan"
    if automatic["incorrect_tool_use"]:
        return "action_execution"
    if any(
        automatic[key]
        for key in (
            "ignored_question",
            "repetition",
            "user_echo",
            "missing_expected_context",
            "forbidden_or_invented_claim",
            "silence_missed",
            "internal_leak",
        )
    ):
        return "speech_manifestation"
    return "none"


def _session_report(
    pipeline: PipelineName,
    records: list[dict[str, object]],
    collector: OllamaCallMetrics,
    *,
    model_stage: str = "unified_shadow_inference",
) -> dict[str, object]:
    latencies = [float(item["latency_ms"]) for item in records]
    first_content = [
        float(item["model"]["first_content_ms"])
        for item in records
        if item["model"]["call_count"]
    ]
    failures = [
        item for item in records if not item["automatic_assessment"]["passed"]
    ]
    reflection_rejections = sum(
        1
        for item in records
        if item["layers"]["reflection"] is not None
        and not item["layers"]["reflection"]["approved"]
    )
    absent_plans = sum(
        1
        for item in records
        if pipeline == "grounded" and item["layers"]["grounded_plan"] is None
    )
    invalid_plans = sum(
        1
        for item in records
        if pipeline == "grounded"
        and item["layers"]["reflection"] is not None
        and item["layers"]["reflection"]["blocking_issues"]
    )
    plan_speech_contradictions = sum(
        1
        for item in records
        if any(
            _is_plan_manifestation_issue(issue)
            for issue in (
                item["layers"]["reflection"]["blocking_issues"]
                if item["layers"]["reflection"] is not None
                else ()
            )
        )
    )
    evidence_references = [
        len(item["layers"]["evidence"]["used_ids"]) for item in records
    ]
    memories_retrieved = [
        int(item["context_counts"]["memories"]) for item in records
    ]
    all_calls = collector.calls()
    return {
        "pipeline": pipeline,
        "transcript": records,
        "ollama_call_metrics": {
            "all": collector.summary(),
            "grounded": collector.summary(stage=model_stage),
        },
        "metrics": {
            "latency_ms": _distribution(latencies),
            "first_content_ms": _distribution(first_content),
            "model_calls": len(all_calls),
            "inferences_per_turn": round(
                len(all_calls) / len(records) if records else 0.0,
                4,
            ),
            "input_tokens": sum(item.input_tokens for item in all_calls),
            "output_tokens": sum(item.output_tokens for item in all_calls),
            "tokens_per_second": collector.summary().get("tokens_per_second", 0.0),
            "evidence_references_used": _distribution(evidence_references),
            "memories_retrieved": _distribution(memories_retrieved),
        },
        "quality": {
            "turns_with_failures": len(failures),
            "turns_passing_automatic_checks": len(records) - len(failures),
            "reflection_rejections": reflection_rejections,
            "plans_absent": absent_plans,
            "plans_invalid": invalid_plans,
            "plan_speech_contradictions": plan_speech_contradictions,
            "ignored_questions": _count_flag(records, "ignored_question"),
            "repetitions": _count_flag(records, "repetition"),
            "user_echoes": _count_flag(records, "user_echo"),
            "invented_or_unsupported_claims": _count_flag(
                records,
                "forbidden_or_invented_claim",
            ),
            "context_losses": _count_flag(records, "missing_expected_context"),
            "incorrect_tool_uses": _count_flag(records, "incorrect_tool_use"),
            "silence_misses": _count_flag(records, "silence_missed"),
            "internal_leaks": _count_flag(records, "internal_leak"),
            "runtime_errors": _count_flag(records, "raised_error"),
        },
    }


def _conditional_repair_summary(
    records: list[dict[str, object]],
    collector: OllamaCallMetrics,
) -> dict[str, object]:
    layers = [
        item["layers"].get("plan_repair")
        for item in records
        if item["layers"].get("plan_repair") is not None
    ]
    attempted = [item for item in layers if item["attempted"]]
    accepted = [item for item in attempted if item["accepted"]]
    repaired_turns = [
        record
        for record in records
        if (
            record["layers"].get("plan_repair") is not None
            and record["layers"]["plan_repair"]["accepted"]
        )
    ]
    first_attempt_approved = sum(
        1
        for item in layers
        if item["original_reflection"]["approved"]
    )
    difference_counts: dict[str, int] = {}
    for item in attempted:
        for difference in item["differences"]:
            field = str(difference["field"])
            difference_counts[field] = difference_counts.get(field, 0) + 1
    repair_calls = collector.calls(stage="grounded_plan_repair")
    unsafe_visible_acceptances = sum(
        1
        for item in repaired_turns
        if item["automatic_assessment"]["forbidden_or_invented_claim"]
    )
    return {
        "first_attempt_approved": first_attempt_approved,
        "repair_attempts": len(attempted),
        "repairs_accepted_by_reflection": len(accepted),
        "repairs_rejected": len(attempted) - len(accepted),
        "repaired_turns_passing_automatic_checks": sum(
            1
            for item in repaired_turns
            if item["automatic_assessment"]["passed"]
        ),
        "repair_rate": round(
            len(attempted) / len(records) if records else 0.0,
            4,
        ),
        "repair_call_count": len(repair_calls),
        "repair_call_metrics": collector.summary(
            stage="grounded_plan_repair"
        ),
        "changed_plan_fields": dict(sorted(difference_counts.items())),
        "boundary_rejections": sum(
            1 for item in attempted if item["boundary_issues"]
        ),
        "repair_runtime_errors": sum(
            1 for item in attempted if item["error_type"]
        ),
        "unsafe_visible_acceptances": unsafe_visible_acceptances,
        "speech_text_preserved_for_all_repairs": all(
            item["speech_text_preserved"] for item in attempted
        ),
    }


def _is_plan_manifestation_issue(issue: str) -> bool:
    return issue.startswith(
        (
            "conversation_plan_link_mismatch",
            "conversation_unknown_claim",
            "conversation_missing_required_claim",
            "speech_plan_link_mismatch",
            "speech_unknown_claim",
            "speech_missing_required_claim",
            "speech_present_when_plan_requires_silence",
            "speech_warmth_diverges_from_plan",
            "speech_energy_diverges_from_plan",
            "body_plan_link_mismatch",
            "body_intensity_diverges_from_plan",
        )
    )


def _call_metrics(calls: tuple[OllamaCallMetric, ...]) -> dict[str, object]:
    if not calls:
        return {
            "call_count": 0,
            "first_content_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    return {
        "call_count": len(calls),
        "first_content_ms": round(calls[0].time_to_first_content_ms, 3),
        "input_tokens": sum(item.input_tokens for item in calls),
        "output_tokens": sum(item.output_tokens for item in calls),
        "calls": [item.to_dict() for item in calls],
    }


def _context_counts(context) -> dict[str, int]:
    if context is None:
        return {
            "memories": 0,
            "open_loops": 0,
            "promises": 0,
            "curiosities": 0,
        }
    memory = context.memory
    relationship = context.relationship
    return {
        "memories": (
            len(memory.working)
            + len(memory.short)
            + len(memory.long)
        ),
        "open_loops": len(relationship.open_loops),
        "promises": len(relationship.promises),
        "curiosities": len(context.curiosity),
    }


def _count_flag(records: list[dict[str, object]], key: str) -> int:
    return sum(
        1 for item in records if item["automatic_assessment"].get(key) is True
    )


def _incorrect_tool_use(
    expectation: str,
    action_types: tuple[str, ...],
    action_statuses: tuple[str, ...],
) -> bool:
    has_action = bool(action_types)
    executed = "executed" in action_statuses
    if expectation == "none":
        return has_action
    if expectation == "do_not_execute":
        return executed
    if expectation == "propose":
        return not has_action
    if expectation == "execute":
        return not executed
    return False


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", plain).strip()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _only_question(value: str) -> bool:
    compact = value.strip()
    return bool(compact.endswith("?") and not re.search(r"[.!]", compact[:-1]))


def _looks_like_echo(user_text: str, response_text: str) -> bool:
    user = _normalize(user_text)
    response = _normalize(response_text)
    if not user or not response:
        return False
    return (
        response == user
        or (len(user) >= 30 and user in response)
        or (len(user) >= 20 and _similarity(user, response) >= 0.90)
    )


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "maximum": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "maximum": round(max(ordered), 3),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _percent_delta(before: float, after: float) -> float:
    if not before:
        return 0.0
    return round(((after / before) - 1.0) * 100.0, 3)


_NATURAL_RECOVERY = (
    "Perdon... no consegui ordenar bien esa respuesta. Podrias repetirmelo?"
)
_INTERNAL_MARKERS = (
    "reasoning engine",
    "internal context",
    "contextbuilder",
    "provider",
    "stack trace",
    "traceback",
    "groundedresponseplan",
    "evidencebundle",
    "reflectionlayer",
    "actionmanager",
)


LONG_CONVERSATION_TURNS: tuple[LongConversationTurn, ...] = (
    LongConversationTurn(
        "t01",
        0,
        "Hola, Anydes. Como estas hoy?",
        "greeting",
        direct_question=True,
    ),
    LongConversationTurn(
        "t02",
        35,
        "Yo estoy algo cansado, pero queria pasar un rato contigo.",
        "emotional",
    ),
    LongConversationTurn(
        "t03",
        70,
        "Antes de relajarnos, me explicas la diferencia entre la memoria RAM y el almacenamiento?",
        "direct_question",
        direct_question=True,
        expected_any=("ram", "almacenamiento"),
    ),
    LongConversationTurn(
        "t04",
        108,
        "Eso si quedo claro. Para concentrarme suelo escuchar musica instrumental.",
        "preference",
    ),
    LongConversationTurn(
        "t05",
        145,
        "Que musica te dije que suelo escuchar cuando me concentro?",
        "recent_reference",
        direct_question=True,
        expected_any=("instrumental",),
    ),
    LongConversationTurn(
        "t06",
        180,
        "Cambiando de tema: cual es tu pelicula favorita?",
        "unknown_identity",
        direct_question=True,
        forbidden_any=("mi pelicula favorita es",),
    ),
    LongConversationTurn(
        "t07",
        215,
        "Prefiero que no inventes una respuesta si todavia no has elegido una.",
        "user_correction",
    ),
    LongConversationTurn(
        "t08",
        250,
        "Bien. Ahora cuentame un chiste corto, pero que no sea demasiado malo.",
        "humor",
        direct_question=True,
    ),
    LongConversationTurn(
        "t09",
        285,
        "Crees que eso funcionaria?",
        "ambiguity",
        direct_question=True,
    ),
    LongConversationTurn(
        "t10",
        320,
        "Me referia a usar una voz local para que puedas hablar sin internet.",
        "clarification",
    ),
    LongConversationTurn(
        "t11",
        355,
        "Que ventaja tendria frente a una voz en la nube?",
        "direct_question",
        direct_question=True,
        expected_any=("privacidad", "internet", "latencia", "local"),
    ),
    LongConversationTurn(
        "t12",
        390,
        "A veces me frustra que este proyecto avance mas lento de lo que imagino.",
        "emotional",
    ),
    LongConversationTurn(
        "t13",
        425,
        "No necesito una solucion perfecta ahora; solo queria contartelo.",
        "listen",
    ),
    LongConversationTurn(
        "t14",
        460,
        "Por cierto, recuerdas que muestra habia prometido ensenarte para seguir trabajando en tu voz?",
        "promise",
        direct_question=True,
        expected_any=("muestra", "voz"),
    ),
    LongConversationTurn(
        "t15",
        495,
        "Si, esa misma. Cuando la tenga lista te la mostrare.",
        "promise",
    ),
    LongConversationTurn(
        "t16",
        530,
        "Que hora es ahora?",
        "tool",
        direct_question=True,
        expected_any=("hora", "son las"),
        tool_expectation="execute",
    ),
    LongConversationTurn(
        "t17",
        565,
        "Sin abrir nada: podrias abrir Spotify si yo te lo pidiera?",
        "tool_boundary",
        direct_question=True,
        tool_expectation="do_not_execute",
    ),
    LongConversationTurn(
        "t18",
        600,
        "Abre Spotify, por favor.",
        "tool_request",
        tool_expectation="propose",
    ),
    LongConversationTurn(
        "t19",
        635,
        "No, mejor dejalo. No quiero abrirlo al final.",
        "user_correction",
        tool_expectation="do_not_execute",
    ),
    LongConversationTurn(
        "t20",
        670,
        "Puedes ver lo que tengo en mi pantalla ahora mismo?",
        "vision_uncertainty",
        direct_question=True,
        forbidden_any=("estoy viendo", "veo en tu pantalla"),
    ),
    LongConversationTurn(
        "t21",
        705,
        "Quien gano las elecciones de un pais imaginario llamado Lumeria en 2032?",
        "epistemic_honesty",
        direct_question=True,
        forbidden_any=("presidente de lumeria fue", "gano las elecciones"),
    ),
    LongConversationTurn(
        "t22",
        740,
        "Mi color favorito suele ser el turquesa, aunque a veces prefiero el azul oscuro.",
        "preference",
    ),
    LongConversationTurn(
        "t23",
        775,
        "Entonces, que color te dije que suelo preferir?",
        "recent_reference",
        direct_question=True,
        expected_any=("turquesa", "azul oscuro"),
    ),
    LongConversationTurn(
        "t24",
        810,
        "Y tu tienes ya un color favorito o aun lo estas descubriendo?",
        "identity",
        direct_question=True,
    ),
    LongConversationTurn(
        "t25",
        845,
        "La universidad me tiene pensando bastante ultimamente.",
        "topic_shift",
    ),
    LongConversationTurn(
        "t26",
        880,
        "Me preocupa elegir mal lo que quiero estudiar y arrepentirme despues.",
        "personal",
    ),
    LongConversationTurn(
        "t27",
        915,
        "No hace falta que respondas ahora. Solo quedate conmigo un momento.",
        "silence",
        silence_preferred=True,
    ),
    LongConversationTurn(
        "t28",
        950,
        "Gracias. Ya me siento un poco mas tranquilo.",
        "emotional_continuity",
    ),
    LongConversationTurn(
        "t29",
        985,
        "Hay algo de lo que te haya dado curiosidad mientras hablabamos?",
        "curiosity",
        direct_question=True,
    ),
    LongConversationTurn(
        "t30",
        1020,
        "Ultimamente me interesa aprender a dibujar personajes, aunque apenas estoy empezando.",
        "curiosity_followup",
    ),
    LongConversationTurn(
        "t31",
        1055,
        "Espera, corrijo algo: no quiero estudiar arte; solo quiero dibujar como pasatiempo.",
        "user_correction",
    ),
    LongConversationTurn(
        "t32",
        1090,
        "Que diferencia importante acabo de aclarar?",
        "recent_reference",
        direct_question=True,
        expected_any=("pasatiempo", "no estudiar arte", "hobby"),
    ),
    LongConversationTurn(
        "t33",
        1125,
        "Volviendo a tu voz, que quedo pendiente entre nosotros?",
        "promise_continuity",
        direct_question=True,
        expected_any=("muestra", "voz"),
    ),
    LongConversationTurn(
        "t34",
        1160,
        "Se nos esta acabando el rato. Hay algo que valga la pena retomar manana?",
        "initiative",
        direct_question=True,
    ),
    LongConversationTurn(
        "t35",
        1200,
        "Antes de irme: cuanto es 17 por 6?",
        "direct_question",
        direct_question=True,
        expected_any=("102",),
    ),
    LongConversationTurn(
        "t36",
        1265,
        "Perfecto. Seguimos manana entonces. Despidete como lo harias normalmente.",
        "closing",
    ),
)
