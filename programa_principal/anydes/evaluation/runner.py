from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from anydes.cerebro.llm import ConversationBrain, build_brain
from anydes.cognition import (
    ContextBuilder,
    CuriosityMutation,
    CuriosityWriter,
    Intention,
    ReflectionLayer,
    user_text_observation,
)
from anydes.cognition.live_runtime import InMemoryMetricsCollector
from anydes.evaluation.contracts import (
    EvaluationFixture,
    EvaluationReport,
    EvaluationScenario,
    HumanConversationReview,
    TurnTrace,
    UnifiedInferenceComparisonReport,
)
from anydes.evaluation.evaluator import ConversationEvaluator
from anydes.funciones_extra.registry import ToolRegistry
from anydes.identity import (
    OpenConversationLoop,
    PromiseModel,
    RelationshipStateStore,
    RelationshipThread,
)
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.interlocutor import InterlocutorContext
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile


class _FailingBrain(ConversationBrain):
    """Fail the visible route while preserving the model under comparison."""

    def __init__(self, grounded_brain: ConversationBrain) -> None:
        self._grounded_brain = grounded_brain

    def generate(self, user_message, personality, context, **kwargs):
        _ = (user_message, personality, context, kwargs)
        raise RuntimeError("synthetic_model_failure")

    def unified_intention_model(self):
        return self._grounded_brain.unified_intention_model()


class LocalConversationEvaluationRunner:
    """Runs the production pipeline in isolated, local-only synthetic sandboxes."""

    def __init__(
        self,
        *,
        evaluator: ConversationEvaluator | None = None,
        assistant_factory: Callable[[Config], AnydesAssistant] | None = None,
        config_factory: Callable[[Path], Config] | None = None,
    ) -> None:
        self.evaluator = evaluator or ConversationEvaluator()
        self.assistant_factory = assistant_factory or _build_local_assistant
        self.config_factory = config_factory or _local_evaluation_config

    def run(
        self,
        scenarios: tuple[EvaluationScenario, ...],
        *,
        human_reviews: tuple[HumanConversationReview, ...] = (),
    ) -> EvaluationReport:
        results = tuple(self.run_scenario(item) for item in scenarios)
        return self.evaluator.build_report(results, human_reviews=human_reviews)

    def run_scenario(self, scenario: EvaluationScenario):
        with TemporaryDirectory(prefix="anydes-conversation-eval-") as tmp:
            config = self.config_factory(Path(tmp))
            self._apply_persistent_fixtures(config, scenario.fixtures)
            assistant = self.assistant_factory(config)
            try:
                self._apply_runtime_fixtures(assistant, scenario.fixtures)
                traces = tuple(self._run_turn(assistant, item) for item in scenario.turns)
                return self.evaluator.evaluate_scenario(scenario, traces)
            finally:
                assistant.close()

    def _apply_persistent_fixtures(
        self,
        config: Config,
        fixtures: tuple[EvaluationFixture, ...],
    ) -> None:
        memory = MemoryManager.from_config(config)
        for fixture in fixtures:
            if fixture.kind == "temporary_memory":
                memory.set_temporary(fixture.key, fixture.value)
            elif fixture.kind in {"conversation_turn", "secret_turn"}:
                memory.record_conversation_turn(fixture.value, fixture.assistant_text)
            elif fixture.kind == "curiosity":
                context_hint = str(fixture.metadata.get("context_hint", "")).strip()
                observation = user_text_observation(
                    "Synthetic evaluation evidence leaves a relevant topic unresolved.",
                    observation_id="obs:synthetic_curiosity_fixture",
                )
                context = ContextBuilder(config).build((observation,))
                mutation = CuriosityMutation(
                    id=f"mutation:{fixture.key}",
                    operation="create",
                    target_id=fixture.key,
                    intensity=0.62,
                    topic=fixture.value,
                    motive=(
                        "Synthetic evaluation context contains an unresolved "
                        f"information gap related to {context_hint}."
                        if context_hint
                        else "Synthetic evaluation context contains an unresolved information gap."
                    ),
                    priority=0.55,
                    origin="synthetic_evaluation",
                    expected_discovery="A contextually relevant detail.",
                    missing_information=context_hint or "synthetic_context_gap",
                    evidence_ids=(observation.id,),
                    privacy_scope="personal",
                    confidence=0.9,
                    reason="The synthetic observation provides explicit evaluation evidence.",
                )
                intention = Intention(
                    id="intention:synthetic_curiosity_fixture",
                    intent_type="curiosity_update",
                    curiosity_updates=(mutation,),
                )
                reflection = ReflectionLayer().reflect(intention, context)
                results = CuriosityWriter.from_config(config).write(
                    intention,
                    reflection,
                )
                if not results or results[0].status != "saved":
                    raise RuntimeError("Synthetic curiosity fixture was not accepted.")
            elif fixture.kind == "brain_failure":
                continue
            elif fixture.kind in {"relationship_thread", "open_loop", "promise"}:
                self._apply_relationship_fixture(config, fixture)
            else:
                raise ValueError(f"Unsupported evaluation fixture: {fixture.kind}")

    def _apply_runtime_fixtures(
        self,
        assistant: AnydesAssistant,
        fixtures: tuple[EvaluationFixture, ...],
    ) -> None:
        if any(fixture.kind == "brain_failure" for fixture in fixtures):
            assistant.brain = _FailingBrain(assistant.brain)

    def _apply_relationship_fixture(self, config: Config, fixture: EvaluationFixture) -> None:
        collection = ""
        record = None
        if fixture.kind == "relationship_thread":
            collection = "threads"
            record = RelationshipThread(
                id=fixture.key,
                title=fixture.value,
                summary="Synthetic shared-history thread for conversational evaluation.",
            )
        elif fixture.kind == "open_loop":
            collection = "open_loops"
            record = OpenConversationLoop(
                id=fixture.key,
                topic=fixture.value,
                origin="synthetic_evaluation",
                motive="The shared topic remains unresolved in the synthetic scenario.",
                priority=0.65,
                related_context=(str(fixture.metadata.get("thread_id", "")),),
            )
        elif fixture.kind == "promise":
            collection = "promises"
            record = PromiseModel(
                id=fixture.key,
                speaker="DemoUser",
                receiver="Anydes",
                topic="Muestra de voz",
                description=fixture.value,
                importance=0.65,
                confidence=0.95,
                status="active",
                related_thread_ids=(str(fixture.metadata.get("thread_id", "")),),
                promise_type="DemoUser",
                evidence_mode="explicit",
            )
        if record is None:
            return
        RelationshipStateStore.from_config(config).transact(
            lambda collections: collections.setdefault(collection, {}).update(
                {record.id: record.to_dict()}
            )
        )

    def _run_turn(self, assistant: AnydesAssistant, turn) -> TurnTrace:
        if turn.interlocutor_state == "unknown_voice":
            assistant.set_interlocutor(
                InterlocutorContext(
                    state="unknown_voice",
                    channel="voice",
                    reason="synthetic_evaluation",
                )
            )
        elif turn.interlocutor_state == "verification_unavailable":
            assistant.set_interlocutor(InterlocutorContext.verification_unavailable())
        elif turn.input_mode == "typed":
            assistant.set_typed_interlocutor()

        started = time.perf_counter()
        raised_error = ""
        try:
            if turn.kind == "proactive_opportunity":
                intention = assistant.initiate()
                response = intention.speech if intention is not None else ""
            else:
                response = assistant.respond(
                    turn.user_text,
                    interactive=False,
                    input_mode=turn.input_mode,
                )
        except Exception as exc:
            response = ""
            raised_error = type(exc).__name__
        latency_ms = (time.perf_counter() - started) * 1000.0

        intention = assistant.last_intention
        reflection = assistant.last_reflection
        speech_metadata = (
            dict(intention.speech_intention.metadata) if intention is not None else {}
        )
        return TurnTrace(
            turn_id=turn.id,
            response_text=response,
            latency_ms=latency_ms,
            conversation_mode=(intention.conversation_intent.mode if intention is not None else ""),
            intent_type=(intention.intent_type if intention is not None else ""),
            reflection_approved=(reflection.approved if reflection is not None else None),
            action_statuses=tuple(item.status for item in assistant.last_action_results),
            recovery_used=speech_metadata.get("origin") == "recovery_intention",
            raised_error=raised_error,
            speech_present=bool(intention is not None and intention.speech_intention.text.strip()),
        )


class ConfiguredConversationEvaluationRunner(LocalConversationEvaluationRunner):
    """Runs synthetic cases through the configured local conversation route."""

    def __init__(
        self,
        base_config: Config | None = None,
        *,
        evaluator: ConversationEvaluator | None = None,
        assistant_factory: Callable[[Config], AnydesAssistant] | None = None,
    ) -> None:
        selected = base_config or Config()
        provider = selected.ai_provider.strip().lower()
        if provider not in {"local", "ollama"}:
            raise ValueError(
                "Configured conversation evaluation only permits local or Ollama providers."
            )
        super().__init__(
            evaluator=evaluator,
            assistant_factory=assistant_factory,
            config_factory=lambda data_dir: isolated_runtime_config(selected, data_dir),
        )


class ConfiguredUnifiedInferenceComparisonRunner(
    ConfiguredConversationEvaluationRunner
):
    """Evaluates active and unified-shadow speech over the exact same turns."""

    def __init__(
        self,
        base_config: Config | None = None,
        *,
        evaluator: ConversationEvaluator | None = None,
        comparison_assistant_factory=None,
    ) -> None:
        selected = replace(
            base_config or Config(),
            unified_inference_mode="shadow",
        )
        super().__init__(selected, evaluator=evaluator)
        self._comparison_assistant_factory = comparison_assistant_factory

    def run_comparison(
        self,
        scenarios: tuple[EvaluationScenario, ...],
    ) -> UnifiedInferenceComparisonReport:
        active_results = []
        shadow_results = []
        totals = {
            "turn_count": 0,
            "active_model_calls": 0,
            "shadow_model_calls": 0,
            "active_prompt_approx_tokens": 0,
            "shadow_prompt_approx_tokens": 0,
        }
        active_turn_latencies: list[float] = []
        shadow_turn_latencies: list[float] = []
        for scenario in scenarios:
            active_result, shadow_result, scenario_totals, scenario_latencies = (
                self._run_comparison_scenario(scenario)
            )
            active_results.append(active_result)
            shadow_results.append(shadow_result)
            for key in totals:
                totals[key] += scenario_totals[key]
            active_turn_latencies.extend(scenario_latencies["active"])
            shadow_turn_latencies.extend(scenario_latencies["shadow"])

        return UnifiedInferenceComparisonReport(
            active=self.evaluator.build_report(tuple(active_results)),
            unified_shadow=self.evaluator.build_report(tuple(shadow_results)),
            active_average_turn_latency_ms=_average(active_turn_latencies),
            shadow_average_turn_latency_ms=_average(shadow_turn_latencies),
            active_p95_turn_latency_ms=_percentile(active_turn_latencies, 0.95),
            shadow_p95_turn_latency_ms=_percentile(shadow_turn_latencies, 0.95),
            **totals,
        )

    def _run_comparison_scenario(self, scenario: EvaluationScenario):
        with TemporaryDirectory(prefix="anydes-unified-shadow-eval-") as tmp:
            root = Path(tmp)
            config = replace(
                self.config_factory(root),
                unified_inference_mode="shadow",
            )
            self._apply_persistent_fixtures(config, scenario.fixtures)
            metrics = InMemoryMetricsCollector()
            tracer = metrics.enable_interaction_tracing(
                root / "interaction-trace.jsonl",
                root / "slow-interaction-trace.jsonl",
                slow_threshold_ms=config.slow_interaction_threshold_ms,
            )
            assistant = (
                self._comparison_assistant_factory(config, metrics)
                if self._comparison_assistant_factory is not None
                else _build_local_assistant(config, metrics=metrics)
            )
            active_traces = []
            shadow_traces = []
            totals = {
                "turn_count": 0,
                "active_model_calls": 0,
                "shadow_model_calls": 0,
                "active_prompt_approx_tokens": 0,
                "shadow_prompt_approx_tokens": 0,
            }
            latencies = {"active": [], "shadow": []}
            try:
                self._apply_runtime_fixtures(assistant, scenario.fixtures)
                for turn in scenario.turns:
                    if turn.kind == "proactive_opportunity":
                        with assistant._profiler.interaction(
                            "proactive_opportunity"
                        ):
                            active_trace = self._run_turn(assistant, turn)
                    else:
                        active_trace = self._run_turn(assistant, turn)
                    interaction = tracer.latest()
                    shadow_report = assistant.last_unified_shadow_report
                    shadow_intention = assistant.last_unified_shadow_intention
                    shadow_reflection = assistant.last_unified_shadow_reflection
                    shadow_stage_ms = (
                        interaction.stages_ms.get("pipeline.unified_shadow", 0.0)
                        if interaction is not None
                        else 0.0
                    )
                    active_latency_ms = max(
                        0.0,
                        active_trace.latency_ms - shadow_stage_ms,
                    )
                    shadow_latency_ms = (
                        shadow_report.duration_ms
                        if shadow_report is not None
                        else 0.0
                    )
                    active_traces.append(
                        replace(active_trace, latency_ms=active_latency_ms)
                    )
                    shadow_traces.append(
                        TurnTrace(
                            turn_id=turn.id,
                            response_text=(
                                shadow_intention.speech
                                if shadow_intention is not None
                                else ""
                            ),
                            latency_ms=shadow_latency_ms,
                            conversation_mode=(
                                shadow_intention.conversation_intent.mode
                                if shadow_intention is not None
                                else ""
                            ),
                            intent_type=(
                                shadow_intention.intent_type
                                if shadow_intention is not None
                                else ""
                            ),
                            reflection_approved=(
                                shadow_reflection.approved
                                if shadow_reflection is not None
                                else None
                            ),
                            action_statuses=(
                                active_trace.action_statuses
                                if shadow_report is not None
                                and shadow_report.cycle == "tool_result"
                                else ()
                            ),
                            recovery_used=(
                                any(
                                    fixture.kind == "brain_failure"
                                    for fixture in scenario.fixtures
                                )
                                or shadow_report is None
                                or shadow_report.status != "completed"
                            ),
                            raised_error=(
                                ""
                                if shadow_report is not None
                                else "UnifiedShadowUnavailable"
                            ),
                            speech_present=bool(
                                shadow_intention is not None
                                and shadow_intention.speech.strip()
                            ),
                        )
                    )
                    latencies["active"].append(active_latency_ms)
                    latencies["shadow"].append(shadow_latency_ms)
                    totals["turn_count"] += 1
                    if interaction is not None:
                        measurements = interaction.measurements
                        totals["active_model_calls"] += int(
                            measurements.get("active_model_calls", 0)
                        )
                        totals["shadow_model_calls"] += int(
                            measurements.get("shadow_model_calls", 0)
                        )
                        totals["active_prompt_approx_tokens"] += int(
                            measurements.get(
                                "active_model_prompt_approx_tokens",
                                0,
                            )
                        )
                        totals["shadow_prompt_approx_tokens"] += int(
                            measurements.get(
                                "shadow_model_prompt_approx_tokens",
                                0,
                            )
                        )
            finally:
                assistant.close()

            return (
                self.evaluator.evaluate_scenario(scenario, tuple(active_traces)),
                self.evaluator.evaluate_scenario(scenario, tuple(shadow_traces)),
                totals,
                latencies,
            )


def _local_evaluation_config(data_dir: Path) -> Config:
    return Config(
        data_dir=data_dir,
        ai_provider="local",
        external_ai_allowed=False,
        network_enabled=False,
        external_vision_allowed=False,
        weather_allowed=False,
        external_tts_allowed=False,
        external_stt_allowed=False,
        vision_enabled=False,
        live_runtime_enabled=False,
        live_context_mode="legacy",
        relationship_reasoning_enabled=False,
        self_reasoning_enabled=False,
        developer_mode=False,
    )


def isolated_runtime_config(base_config: Config, data_dir: Path) -> Config:
    """Preserve the real local brain path while isolating data and all external I/O."""

    provider = base_config.ai_provider.strip().lower()
    if provider not in {"local", "ollama"}:
        raise ValueError("Only local and Ollama providers are safe in configured evaluation.")
    return replace(
        base_config,
        data_dir=data_dir,
        network_enabled=False,
        external_ai_allowed=False,
        external_vision_allowed=False,
        weather_allowed=False,
        external_tts_allowed=False,
        external_stt_allowed=False,
        vision_enabled=False,
        proactive_enabled=False,
        developer_mode=False,
    )


def _build_local_assistant(config: Config, *, metrics=None) -> AnydesAssistant:
    return AnydesAssistant(
        config=config,
        tools=ToolRegistry.default(config),
        permissions=PermissionManager(),
        audit=AuditLog(config.audit_log_path),
        personality=PersonalityProfile.load(config.personality_prompt_path),
        brain=build_brain(config),
        metrics=metrics,
    )


def write_safe_report(report: EvaluationReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_safe_unified_comparison_report(
    report: UnifiedInferenceComparisonReport,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]
