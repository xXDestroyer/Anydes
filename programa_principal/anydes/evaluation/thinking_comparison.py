from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from anydes.cognition import (
    EvidenceBundleBuilder,
    ReasoningEngine,
    ReflectionLayer,
    user_text_observation,
)
from anydes.evaluation.long_conversation import (
    LONG_CONVERSATION_TURNS,
    LongConversationTurn,
    LongConversationValidation,
)
from anydes.evaluation.ollama_benchmark import (
    BenchmarkOllamaBrain,
    OllamaBenchmarkSettings,
    OllamaCallMetrics,
)
from anydes.funciones_extra.registry import ToolRegistry
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.assistant import AnydesAssistant
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile


class GroundedThinkingComparison:
    """Paired long-session comparison with one shared context per turn."""

    def __init__(
        self,
        base_config: Config,
        turns: tuple[LongConversationTurn, ...] = LONG_CONVERSATION_TURNS,
    ) -> None:
        self.base_config = base_config
        self.turns = tuple(turns)

    def run(self) -> dict[str, object]:
        with TemporaryDirectory(prefix="anydes-grounded-thinking-paired-") as tmp:
            root = Path(tmp)
            config = self._config(root)
            LongConversationValidation(config)._seed_shared_state(config)
            collectors = {
                "think_false": OllamaCallMetrics(),
                "think_true": OllamaCallMetrics(),
            }
            brains = {
                name: BenchmarkOllamaBrain(
                    model=config.ai_model,
                    base_url=config.ollama_url,
                    config=replace(
                        config,
                        ollama_grounded_think=(name == "think_true"),
                    ),
                    settings=OllamaBenchmarkSettings(
                        think=(name == "think_true"),
                    ),
                    collector=collectors[name],
                )
                for name in collectors
            }
            context_assistant = self._assistant(config, brains["think_false"])
            memory = MemoryManager.from_config(config)
            rows = []
            try:
                for turn in self.turns:
                    observation = user_text_observation(
                        turn.user_text,
                        silence_allowed=True if turn.silence_preferred else None,
                    )
                    context = context_assistant._context_for_observations(
                        (observation,),
                        cycle="initial",
                        live_snapshot=None,
                    )
                    context = context_assistant._context_with_available_actions(
                        context
                    )
                    bundle = EvidenceBundleBuilder().build(context)
                    context = replace(context, evidence_bundle=bundle)
                    bundle_dict = bundle.to_dict()
                    row = {
                        "turn_id": turn.id,
                        "at_seconds": turn.at_seconds,
                        "capability": turn.capability,
                        "user": turn.user_text,
                        "context": {
                            "serialized_characters": len(context.to_json()),
                            "memory_items": (
                                len(context.memory.working)
                                + len(context.memory.short)
                                + len(context.memory.long)
                            ),
                            "relationship_revision": context.relationship.revision,
                            "open_loops": len(context.relationship.open_loops),
                            "promises": len(context.relationship.promises),
                            "curiosities": len(context.curiosity),
                        },
                        "evidence_bundle_digest": _digest(bundle_dict),
                        "evidence_bundle": bundle_dict,
                        "variants": {},
                    }
                    for name, brain in brains.items():
                        row["variants"][name] = self._run_variant(
                            context,
                            brain,
                            collectors[name],
                        )
                    rows.append(row)
                    memory.record_conversation_turn(
                        turn.user_text,
                        "",
                        cognitively_retrievable=True,
                    )
            finally:
                context_assistant.close()

        return {
            "schema_version": 1,
            "experiment": {
                "id": "qwen3-8b-grounded-thinking-paired-21m-v1",
                "model": "qwen3:8b",
                "turn_count": len(self.turns),
                "timeline_minutes": round(self.turns[-1].at_seconds / 60.0, 3),
                "single_context_per_turn": True,
                "identical_evidence_bundle": True,
                "responses_persisted": False,
                "recoveries_persisted": False,
                "tools_executed": False,
                "reason": (
                    "Keep thinking mode as the only causal variable; actions and "
                    "variant responses cannot mutate the canonical context."
                ),
            },
            "turns": rows,
            "summary": {
                name: _variant_summary(rows, name, collectors[name])
                for name in collectors
            },
        }

    def _config(self, root: Path) -> Config:
        return replace(
            self.base_config,
            data_dir=root,
            ai_provider="ollama",
            ai_model="qwen3:8b",
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
            unified_inference_mode="off",
            reasoning_mode="native",
            relationship_reasoning_enabled=False,
            self_reasoning_enabled=False,
            developer_mode=False,
        )

    @staticmethod
    def _assistant(
        config: Config,
        brain: BenchmarkOllamaBrain,
    ) -> AnydesAssistant:
        return AnydesAssistant(
            config=config,
            tools=ToolRegistry.default(config),
            permissions=PermissionManager(),
            audit=AuditLog(config.audit_log_path),
            personality=PersonalityProfile.load(config.personality_prompt_path),
            brain=brain,
        )

    @staticmethod
    def _run_variant(
        context,
        brain: BenchmarkOllamaBrain,
        collector: OllamaCallMetrics,
    ) -> dict[str, object]:
        offset = len(collector.calls())
        started = time.perf_counter()
        raised_error = ""
        try:
            intention = ReasoningEngine(
                model=brain.unified_intention_model(),
                require_grounded_plan=True,
            ).reason(context)
            reflection = ReflectionLayer().reflect(intention, context)
        except Exception as exc:
            intention = None
            reflection = None
            raised_error = type(exc).__name__
        latency_ms = (time.perf_counter() - started) * 1000.0
        calls = collector.calls()[offset:]
        plan = intention.grounded_plan if intention is not None else None
        approved = bool(reflection is not None and reflection.approved)
        candidate_speech = intention.speech if intention is not None else ""
        return {
            "latency_ms": round(latency_ms, 3),
            "model_calls": len(calls),
            "model_metrics": [item.to_dict() for item in calls],
            "plan": plan.to_dict() if plan is not None else None,
            "intention": intention.to_dict() if intention is not None else None,
            "candidate_speech": candidate_speech,
            "visible_response": (
                candidate_speech if approved else _NATURAL_RECOVERY
            ),
            "reflection": (
                {
                    "approved": reflection.approved,
                    "decision": reflection.decision,
                    "blocking_issues": list(reflection.blocking_issues),
                }
                if reflection is not None
                else None
            ),
            "raised_error": raised_error,
        }


def write_thinking_comparison(
    report: dict[str, object],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _variant_summary(
    rows: list[dict[str, object]],
    name: str,
    collector: OllamaCallMetrics,
) -> dict[str, object]:
    variants = [row["variants"][name] for row in rows]
    latencies = [float(item["latency_ms"]) for item in variants]
    rejected = [
        item
        for item in variants
        if item["reflection"] is None or not item["reflection"]["approved"]
    ]
    issue_counts: dict[str, int] = {}
    for item in rejected:
        reflection = item["reflection"]
        for issue in (
            reflection["blocking_issues"] if reflection is not None else ()
        ):
            code = str(issue).split(":", 1)[0]
            issue_counts[code] = issue_counts.get(code, 0) + 1
    calls = collector.calls()
    return {
        "accepted_plans": len(variants) - len(rejected),
        "rejected_plans": len(rejected),
        "absent_plans": sum(1 for item in variants if item["plan"] is None),
        "latency_ms": _distribution(latencies),
        "model_calls": len(calls),
        "inferences_per_turn": round(len(calls) / len(rows), 4),
        "input_tokens": sum(item.input_tokens for item in calls),
        "output_tokens": sum(item.output_tokens for item in calls),
        "tokens_per_second": collector.summary().get("tokens_per_second", 0.0),
        "thinking_calls": collector.summary().get("thinking_calls", 0),
        "issue_categories": dict(sorted(issue_counts.items())),
    }


def _digest(data: dict[str, object]) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "maximum": round(max(ordered), 3),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


_NATURAL_RECOVERY = (
    "Perdon... no consegui ordenar bien esa respuesta. Podrias repetirmelo?"
)
