from __future__ import annotations

from anydes.realtime.cancellation import ResponseSuperseded

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from anydes.cognition.self_writer import SelfWriter
from anydes.funciones_extra.registry import ToolRegistry
from anydes.memoria.conversation_memory import classify_text
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.intent import Intent
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile
from anydes.realtime.capabilities import CapabilityState, capture_capability_state
from anydes.realtime.calendar_state import CalendarEventStore
from anydes.realtime.context import RealtimeContextBuilder
from anydes.realtime.derived_insights import DerivedInsightStore
from anydes.realtime.identity_profile import (
    RealtimeIdentityProfileStore,
    extract_language_convention,
)
from anydes.realtime.journal import ConversationEvent, ConversationJournal
from anydes.realtime.local_routes import RealtimeLocalRoutes
from anydes.realtime.router import RealtimeAct, RealtimeTurnRouter
from anydes.realtime.session import RealtimeSessionState
from anydes.realtime.self_preferences import canonical_store_preference
from anydes.realtime.tasks import RealtimeTask, TaskManager
from anydes.realtime.tooling import (
    ActionExecutor,
    CapabilityRegistry,
    ExecutionReceipt,
    RealtimeActionRouter,
    SemanticAction,
)
from anydes.realtime.transport import OllamaRealtimeTransport
from anydes.realtime.validation import RealtimeOutputValidator, ValidationResult


@dataclass(frozen=True)
class RealtimeMetrics:
    model: str
    route: str
    context_preparation_ms: float
    time_to_first_token_ms: float
    time_to_first_text_ms: float
    response_complete_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    inference_count: int
    recent_turn_count: int
    memory_fact_count: int
    self_preference_count: int
    approximate_context_tokens: int
    validation_issues: tuple[str, ...]
    timed_out: bool
    failure_code: str = ""
    failure_layer: str = ""

    def safe_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation_issues"] = list(self.validation_issues)
        return payload


@dataclass(frozen=True)
class RealtimeResult:
    text: str
    metrics: RealtimeMetrics
    deferred: bool = False
    validation_fallback: bool = False


@dataclass(frozen=True)
class RealtimeStreamTrace:
    received_text: str = ""
    displayed_text: str = ""
    done_reason: str = ""
    interruption_reason: str = ""
    error_type: str = ""
    connection_state: str = ""
    validation_issues: tuple[str, ...] = ()
    partial_preserved: bool = False
    failure_code: str = ""
    failure_layer: str = ""


@dataclass(frozen=True)
class TurnClassification:
    route: str
    reasons: tuple[str, ...] = ()


class RealtimeConversationService:
    """One-call conversational path with post-response persistence."""

    def __init__(
        self,
        config: Config,
        personality: PersonalityProfile,
        *,
        memory: MemoryManager | None = None,
        transport: OllamaRealtimeTransport | None = None,
        tools: ToolRegistry | None = None,
        permissions: PermissionManager | None = None,
        audit: AuditLog | None = None,
        session: RealtimeSessionState | None = None,
        journal: ConversationJournal | None = None,
        task_manager: TaskManager | None = None,
        capability_registry: CapabilityRegistry | None = None,
        action_executor: ActionExecutor | None = None,
        action_router: RealtimeActionRouter | None = None,
        identity_profile: RealtimeIdentityProfileStore | None = None,
        derived_insights: DerivedInsightStore | None = None,
        calendar_events: CalendarEventStore | None = None,
    ) -> None:
        self.config = config
        self.memory = memory or MemoryManager.from_config(config)
        self.self_visual_enabled = bool(config.realtime_self_visual_enabled)
        self.phase3_enabled = bool(
            config.realtime_phase3_enabled
            or config.realtime_phase4_enabled
            or identity_profile is not None
        )
        self.identity_profile = (
            identity_profile
            if identity_profile is not None
            else (
                RealtimeIdentityProfileStore.from_config(config)
                if self.phase3_enabled
                else None
            )
        )
        if self.identity_profile is not None:
            self.identity_profile.ensure_defaults()
        self.context_builder = RealtimeContextBuilder(
            config,
            personality,
            self.memory,
            identity_profile=self.identity_profile,
        )
        self.self_writer = SelfWriter.from_config(config)
        if self.phase3_enabled:
            canonical = canonical_store_preference("favorite_color")
            if canonical is not None:
                self.self_writer.ensure_preference(canonical)
        self.validator = RealtimeOutputValidator(
            max_words=config.realtime_max_response_words
        )
        self.transport = transport or OllamaRealtimeTransport(
            base_url=config.ollama_url,
            model=config.realtime_model,
            keep_alive=config.realtime_keep_alive,
            timeout_seconds=config.realtime_timeout_seconds,
            max_output_tokens=config.realtime_max_output_tokens,
            num_ctx=config.realtime_num_ctx,
        )
        self.tools = tools or ToolRegistry.default(config)
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLog(config.audit_log_path)
        self.session = session or RealtimeSessionState()
        if self.identity_profile is not None:
            for convention in self.identity_profile.conventions():
                self.session.register_language_convention(
                    convention.expression,
                    convention.meaning,
                )
                for example in convention.examples:
                    self.session.register_language_convention(
                        example,
                        convention.meaning,
                    )
        self.phase2_enabled = bool(
            config.realtime_phase2_enabled
            or self.phase3_enabled
            or capability_registry is not None
            or action_executor is not None
            or action_router is not None
        )
        self.phase1_enabled = bool(
            config.realtime_phase1_enabled
            or self.phase2_enabled
            or journal is not None
            or task_manager is not None
        )
        self.journal = (
            journal
            if journal is not None
            else (
                ConversationJournal(config.realtime_journal_path)
                if self.phase1_enabled
                else None
            )
        )
        self.task_manager = (
            task_manager
            if task_manager is not None
            else (
                TaskManager(config.realtime_tasks_path)
                if self.phase1_enabled
                else None
            )
        )
        self.derived_insights = (
            derived_insights
            if derived_insights is not None
            else (
                DerivedInsightStore.from_config(config, self.journal)
                if self.phase1_enabled and self.journal is not None
                else None
            )
        )
        self.calendar_events = (
            calendar_events
            if calendar_events is not None
            else (
                CalendarEventStore.from_config(config)
                if self.phase2_enabled
                else None
            )
        )
        self.capability_registry = (
            capability_registry
            if capability_registry is not None
            else (
                CapabilityRegistry(config, self.tools)
                if self.phase2_enabled
                else None
            )
        )
        self.action_executor = (
            action_executor
            if action_executor is not None
            else (
                ActionExecutor(
                    self.capability_registry,
                    self.tools,
                    self.permissions,
                    self.audit,
                )
                if self.phase2_enabled and self.capability_registry is not None
                else None
            )
        )
        self.action_router = (
            action_router
            if action_router is not None
            else RealtimeActionRouter() if self.phase2_enabled else None
        )
        self.router = RealtimeTurnRouter()
        self.local_routes = RealtimeLocalRoutes(
            self.tools,
            self.permissions,
            self.audit,
            self.session,
            self.journal,
            derived_insights=self.derived_insights,
        )
        self._persistence = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="anydes-realtime-memory",
        )
        self._closed = False
        self._trace_lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._warmed = False
        self.last_result: RealtimeResult | None = None
        self.last_stream_trace = RealtimeStreamTrace()
        self.last_capability_state: CapabilityState | None = None
        self._last_journal_event_id = ""

    def warmup(self) -> float:
        with self._warm_lock:
            if self._warmed:
                return 0.0
            elapsed = self.transport.warmup()
            self._warmed = True
            return elapsed

    def close(self) -> None:
        self._closed = True
        self._persistence.shutdown(wait=True, cancel_futures=False)
        if self.task_manager is not None:
            self.task_manager.close()

    def classify(self, text: str, *, has_tool_intent: bool = False) -> TurnClassification:
        stripped = text.strip()
        reasons: list[str] = []
        if has_tool_intent:
            reasons.append("tool_intent")
        if len(stripped) > 900 or len(stripped.split()) > 170:
            reasons.append("large_request")
        if "```" in stripped or stripped.count("\n") >= 7:
            reasons.append("structured_work")
        return TurnClassification(
            route="complex" if reasons else "everyday",
            reasons=tuple(reasons),
        )

    def respond(
        self,
        user_text: str,
        *,
        allow_personal_memory: bool,
        allow_persistence: bool,
        has_tool_intent: bool = False,
        intent=None,
        on_text: Callable[[str], None] | None = None,
    ) -> RealtimeResult:
        started = time.perf_counter()
        session_snapshot = self.session.snapshot()
        pending_before_resolution = session_snapshot.proposed_action
        language_resolution = (
            self.identity_profile.resolve_language(user_text)
            if self.identity_profile is not None
            else None
        )
        reference_text = (
            language_resolution.meaning
            if language_resolution is not None and language_resolution.matched
            else user_text
        )
        resolution = (
            self.session.resolve_reference(reference_text)
            if self.phase1_enabled
            else None
        )
        structural_text = (
            resolution.canonical_text
            if resolution is not None and resolution.canonical_text
            else user_text
        )
        act = self.router.classify(
            structural_text,
            has_pending_offer=session_snapshot.pending_offer is not None,
            has_pending_question=bool(session_snapshot.last_assistant_question),
        )
        if resolution is not None:
            if resolution.kind == "offer_acceptance":
                act = replace(
                    act,
                    primary="offer_acceptance",
                    accepts_pending_offer=True,
                    references_previous=True,
                )
            elif resolution.kind == "offer_rejection":
                act = replace(
                    act,
                    primary="offer_rejection",
                    rejects_pending_offer=True,
                    references_previous=True,
                )
            elif (
                resolution.kind in {"question_answer", "previous_reference"}
                and act.primary not in {
                    "history_query",
                    "question",
                    "correction",
                    "user_preference",
                    "request",
                }
            ):
                act = replace(
                    act,
                    primary="previous_reference",
                    references_previous=True,
                )
            elif (
                resolution.kind == "action_confirmation"
                and not (
                    self.phase2_enabled
                    and pending_before_resolution is not None
                    and pending_before_resolution.source == "phase2"
                )
            ):
                intent = Intent("tool", resolution.target)
                has_tool_intent = True
        if (
            not act.preference_topic
            and session_snapshot.last_preference_topic
            and _references_preference(user_text)
        ):
            act = replace(
                act,
                references_previous=True,
                preference_owner=session_snapshot.last_preference_owner,
                preference_topic=session_snapshot.last_preference_topic,
            )
        requested_tool = _requested_tool_name(intent)
        capability = capture_capability_state(
            self.config,
            self.tools,
            requested_tool=requested_tool,
            frame_max_age_seconds=self.config.realtime_frame_max_age_seconds,
        )
        if resolution is not None and resolution.kind == "action_rejection":
            self.session.mark_action_resolved()
            return self._finish_local(
                user_text,
                "Entendido, no lo hago.",
                "pending_action_cancelled",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        phase2_action = None
        if (
            self.phase2_enabled
            and pending_before_resolution is not None
            and pending_before_resolution.source == "phase2"
            and resolution is not None
            and resolution.kind == "action_confirmation"
        ):
            phase2_action = SemanticAction(
                capability=pending_before_resolution.capability,
                mode="confirm",
                entity=pending_before_resolution.entity,
                parameters=dict(pending_before_resolution.parameters),
                origin_text=user_text,
            )
        elif self.phase2_enabled and self.action_router is not None:
            if (
                pending_before_resolution is not None
                and pending_before_resolution.source == "phase2"
            ):
                phase2_action = self.action_router.complete_pending(
                    user_text,
                    pending_before_resolution,
                )
            if phase2_action is None:
                phase2_action = self.action_router.match(user_text, intent)
        if phase2_action is not None:
            phase2_result = self._handle_phase2_action(
                phase2_action,
                user_text=user_text,
                act=act,
                started=started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
            if phase2_result is not None:
                return phase2_result
        if (
            not self.self_visual_enabled
            and act.preference_owner == "self"
            and act.preference_topic
        ):
            self_preference = self.context_builder.self_preferences.resolve_topic(
                act.preference_topic
            )
            if self_preference.status != "unknown":
                explain = any(
                    marker in _normalized_text(user_text)
                    for marker in ("por que", "motivo", "razon", "atrae")
                )
                return self._finish_local(
                    user_text,
                    self.context_builder.self_preferences.known_response(
                        act.preference_topic,
                        explain=explain,
                    ),
                    "self_preference_known",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
            if act.references_previous:
                pass
            else:
                return self._finish_local(
                    user_text,
                    self.context_builder.self_preferences.uncertainty_response(
                        act.preference_topic,
                        revision=session_snapshot.turn_count,
                        question=user_text,
                    ),
                    "self_preference_unknown",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
        local = self.local_routes.resolve(
            user_text,
            act,
            intent,
            capability,
            allow_personal_memory=allow_personal_memory,
        )
        if local.handled:
            if local.tool_result is not None:
                self.session.mark_action_resolved()
            return self._finish_local(
                user_text,
                local.text,
                local.route,
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=local.capability_state or capability,
                tool_result=local.tool_result,
            )
        classification = self.classify(
            user_text,
            has_tool_intent=(
                has_tool_intent
                if intent is None
                else str(getattr(intent, "name", "") or "") == "tool"
            ),
        )
        if classification.route == "complex":
            text = (
                "Eso necesita un poco más de atención para hacerlo bien. "
                "En este modo rápido prefiero no fingir que ya está resuelto."
            )
            complete_ms = (time.perf_counter() - started) * 1000.0
            metrics = RealtimeMetrics(
                model=self.config.realtime_model,
                route="complex_deferred",
                context_preparation_ms=0.0,
                time_to_first_token_ms=0.0,
                time_to_first_text_ms=complete_ms,
                response_complete_ms=complete_ms,
                input_tokens=0,
                output_tokens=0,
                tokens_per_second=0.0,
                inference_count=0,
                recent_turn_count=0,
                memory_fact_count=0,
                self_preference_count=0,
                approximate_context_tokens=0,
                validation_issues=(),
                timed_out=False,
            )
            if on_text is not None:
                on_text(text)
            result = RealtimeResult(text=text, metrics=metrics, deferred=True)
            self.last_result = result
            self._record_visible_turn(user_text, text, allow_persistence, act)
            self._trace(metrics)
            return result

        prepared_at = time.perf_counter()
        context = self.context_builder.build(
            user_text,
            allow_personal_memory=allow_personal_memory,
            act=act,
            session_note=self.session.context_note(act, user_text),
            capability_state=capability,
        )
        context_ms = (time.perf_counter() - prepared_at) * 1000.0
        first_visible_at: float | None = None
        hold_for_preference_validation = (
            context.asks_self_preference
            and context.self_preference_count == 0
        ) or (
            context.asks_user_preference
            and context.memory_fact_count == 0
            and not context.recent_evidence_match
        )

        def stream_visible(cumulative: str) -> None:
            nonlocal first_visible_at
            if hold_for_preference_validation:
                return
            clean = self.validator.sanitize_prefix(cumulative)
            if len(clean.strip()) < 4 or not self.validator.prefix_is_safe(clean):
                return
            if first_visible_at is None:
                first_visible_at = time.perf_counter()
            if on_text is not None:
                on_text(clean)

        stream_error_type = ""
        partial_preserved = False
        try:
            self._warmed = True
            outcome = self.transport.stream(
                context.messages(),
                on_content=stream_visible,
                max_output_tokens=(
                    min(40, self.config.realtime_max_output_tokens)
                    if hold_for_preference_validation
                    else None
                ),
            )
            validation = self.validator.validate(
                outcome.text,
                context,
                task_created=False,
            )
            if (
                (outcome.done_reason == "length" or outcome.connection_state == "interrupted")
                and validation.text
                and validation.text[-1] not in ".!?"
            ):
                completed = _last_complete_sentence(validation.text)
                if completed:
                    validation = self.validator.validate(
                        completed,
                        context,
                        task_created=False,
                    )
                    validation = ValidationResult(
                        valid=validation.valid,
                        text=validation.text,
                        issues=(*validation.issues, "partial_output_preserved"),
                    )
                    partial_preserved = validation.valid
                else:
                    fragment = validation.text.rstrip(" ,;:")
                    closed = (
                        self.validator.validate(
                            f"{fragment}.",
                            context,
                            task_created=False,
                        )
                        if len(fragment.split()) >= 5
                        and self.validator.prefix_is_safe(fragment)
                        else ValidationResult(False, fragment, ())
                    )
                    if closed.valid:
                        validation = ValidationResult(
                            valid=True,
                            text=closed.text,
                            issues=(
                                *closed.issues,
                                "partial_fragment_closed",
                                "partial_output_preserved",
                            ),
                        )
                        partial_preserved = True
                    else:
                        validation = ValidationResult(
                            valid=False,
                            text=validation.text,
                            issues=(*validation.issues, "truncated_output"),
                        )
        except ResponseSuperseded:
            raise
        except Exception as exc:
            outcome = None
            stream_error_type = type(exc).__name__
            validation = self.validator.validate(
                "",
                context,
                task_created=False,
            )

        fallback = not validation.valid
        failure_code, failure_layer = _classify_realtime_failure(
            outcome,
            validation,
            stream_error_type=stream_error_type,
        )
        if fallback:
            if (
                context.asks_self_preference
                and context.self_preference_count == 0
            ):
                final_text = self.context_builder.self_preferences.uncertainty_response(
                    context.preference_topic,
                    revision=self.session.snapshot().turn_count,
                    question=user_text,
                )
            elif (
                context.asks_user_preference
                and context.memory_fact_count == 0
                and not context.recent_evidence_match
            ):
                final_text = "No lo tengo presente todavía."
            elif (
                "self_preference_unverified" in validation.issues
                or "self_preference_contradiction" in validation.issues
            ):
                preference = self.context_builder.self_preferences.resolve_topic(
                    context.preference_topic
                )
                if context.asks_self_preference and preference.status != "unknown":
                    final_text = self.context_builder.self_preferences.known_response(
                        context.preference_topic,
                        explain=context.preference_aspect == "motive",
                    )
                else:
                    final_text = (
                        self.context_builder.self_preferences.uncertainty_response(
                            context.preference_topic,
                            revision=self.session.snapshot().turn_count,
                            question=user_text,
                        )
                        if context.asks_self_preference
                        else "Lo tendré presente."
                    )
            elif (
                "unsupported_avatar_self_claim" in validation.issues
                and context.avatar_self_description is not None
            ):
                final_text = context.avatar_self_description.safe_summary(user_text)
            elif "mirrors_user_preference_as_self" in validation.issues:
                final_text = (
                    "Gracias por corregirme; lo tendré presente."
                    if context.user_correction
                    else "Lo tendré presente."
                )
            elif "mirrors_user_state_as_self" in validation.issues:
                final_text = "Entiendo. Voy a respetar eso."
            elif "user_preference_unverified" in validation.issues:
                final_text = "No lo tengo presente todavía."
            elif "generic_response_only" in validation.issues:
                final_text = "Claro."
            else:
                final_text = (
                    _recovery_text_for_failure(
                        failure_code,
                        validation.issues,
                    )
                    if self.config.realtime_audio_stabilization_enabled
                    else (
                        "Perdón, me quedé a medias un momento. "
                        "Sigo teniendo presente lo que me dijiste."
                    )
                )
        else:
            final_text = validation.text
        if on_text is not None and (fallback or first_visible_at is None):
            on_text(final_text)
            if first_visible_at is None:
                first_visible_at = time.perf_counter()

        complete_ms = (time.perf_counter() - started) * 1000.0
        if outcome is None:
            first_token_ms = complete_ms
            input_tokens = 0
            output_tokens = 0
            tokens_per_second = 0.0
            timed_out = False
        else:
            first_token_ms = context_ms + outcome.time_to_first_token_ms
            input_tokens = outcome.input_tokens or context.approximate_input_tokens
            output_tokens = outcome.output_tokens
            tokens_per_second = outcome.tokens_per_second
            timed_out = outcome.timed_out
        first_text_ms = (
            (first_visible_at - started) * 1000.0
            if first_visible_at is not None
            else complete_ms
        )
        metrics = RealtimeMetrics(
            model=self.config.realtime_model,
            route="everyday",
            context_preparation_ms=context_ms,
            time_to_first_token_ms=first_token_ms,
            time_to_first_text_ms=first_text_ms,
            response_complete_ms=complete_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            inference_count=1,
            recent_turn_count=context.recent_turn_count,
            memory_fact_count=context.memory_fact_count,
            self_preference_count=context.self_preference_count,
            approximate_context_tokens=context.approximate_input_tokens,
            validation_issues=validation.issues,
            timed_out=timed_out,
            failure_code=failure_code,
            failure_layer=failure_layer,
        )
        result = RealtimeResult(
            text=final_text,
            metrics=metrics,
            validation_fallback=fallback,
        )
        self.last_result = result
        observed_preference = self.context_builder.self_preferences.observe_response(
            act,
            final_text,
            source_turn=self.session.snapshot().turn_count + 1,
        )
        if (
            observed_preference is not None
            and self.phase3_enabled
            and not self._closed
        ):
            self._persistence.submit(
                self.self_writer.record_realtime_inclination,
                key=observed_preference.key,
                domain=observed_preference.topic,
                value=observed_preference.value,
                origin=observed_preference.source,
                rationale=(
                    observed_preference.rationale
                    or "Inclinación provisional expresada por Anydes en conversación."
                ),
                evidence_ids=(
                    f"realtime:{self.session.session_id}:"
                    f"turn:{self.session.snapshot().turn_count + 1}",
                ),
                confidence=observed_preference.confidence,
            )
        self.last_capability_state = capability
        self.last_stream_trace = RealtimeStreamTrace(
            received_text=outcome.text if outcome is not None else "",
            displayed_text=final_text,
            done_reason=outcome.done_reason if outcome is not None else "",
            interruption_reason=(
                outcome.interruption_reason if outcome is not None else "service_exception"
            ),
            error_type=outcome.error_type if outcome is not None else stream_error_type,
            connection_state=outcome.connection_state if outcome is not None else "failed",
            validation_issues=validation.issues,
            partial_preserved=partial_preserved,
            failure_code=failure_code,
            failure_layer=failure_layer,
        )
        if failure_code:
            self.audit.record(
                "realtime_recovery",
                failure_code=failure_code,
                failure_layer=failure_layer,
                partial_preserved=partial_preserved,
            )
        self._record_visible_turn(user_text, final_text, allow_persistence, act)
        self._trace(metrics)
        return result

    def _record_visible_turn(
        self,
        user_text: str,
        assistant_text: str,
        allow_persistence: bool,
        act: RealtimeAct,
        *,
        include_in_generation_context: bool = True,
        tool_result=None,
    ) -> None:
        if self.identity_profile is not None:
            convention = extract_language_convention(user_text)
            if convention is not None:
                self.session.register_language_convention(
                    convention.expression,
                    convention.meaning,
                )
                if allow_persistence and not self._closed:
                    self._persistence.submit(
                        self.identity_profile.register_language_convention,
                        convention,
                    )
        if include_in_generation_context:
            self.context_builder.remember_visible_turn(user_text, assistant_text)
        self.session.commit(user_text, assistant_text, act)
        if self.journal is not None and allow_persistence:
            category = _more_private_category(
                classify_text(user_text),
                classify_text(assistant_text),
            )
            topic = self.session.snapshot().active_topic
            user_event = ConversationEvent.create(
                author="user",
                exact_text=user_text,
                session_id=self.session.session_id,
                event_type=act.primary,
                topic=topic,
                related_event_ids=(
                    (self._last_journal_event_id,)
                    if self._last_journal_event_id
                    else ()
                ),
                category=category,
            )
            assistant_event = ConversationEvent.create(
                author="anydes",
                exact_text=assistant_text,
                session_id=self.session.session_id,
                event_type=(
                    "tool_result"
                    if tool_result is not None
                    else "response"
                ),
                topic=topic,
                related_event_ids=(user_event.event_id,),
                tool_result=_safe_tool_result(tool_result),
                category=category,
            )
            events = self.journal.remember((user_event, assistant_event))
            self._last_journal_event_id = assistant_event.event_id
            if not self._closed:
                self._persistence.submit(self.journal.persist, events)
        if not allow_persistence or self._closed:
            return
        self._persistence.submit(
            self.memory.record_conversation_turn,
            user_text,
            assistant_text,
            cognitively_retrievable=True,
        )

    def _trace(self, metrics: RealtimeMetrics) -> None:
        if not self.config.developer_mode:
            return
        path = Path(self.config.realtime_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(metrics.safe_dict(), ensure_ascii=True, sort_keys=True)
        with self._trace_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _finish_local(
        self,
        user_text: str,
        text: str,
        route: str,
        act: RealtimeAct,
        started: float,
        *,
        allow_persistence: bool,
        on_text: Callable[[str], None] | None,
        capability: CapabilityState,
        tool_result=None,
    ) -> RealtimeResult:
        complete_ms = (time.perf_counter() - started) * 1000.0
        if on_text is not None:
            on_text(text)
        metrics = RealtimeMetrics(
            model=self.config.realtime_model,
            route=route,
            context_preparation_ms=0.0,
            time_to_first_token_ms=0.0,
            time_to_first_text_ms=complete_ms,
            response_complete_ms=complete_ms,
            input_tokens=0,
            output_tokens=0,
            tokens_per_second=0.0,
            inference_count=0,
            recent_turn_count=0,
            memory_fact_count=0,
            self_preference_count=0,
            approximate_context_tokens=0,
            validation_issues=(),
            timed_out=False,
        )
        result = RealtimeResult(text=text, metrics=metrics)
        self.last_result = result
        self.last_capability_state = capability
        self.last_stream_trace = RealtimeStreamTrace(
            displayed_text=text,
            done_reason="local",
            connection_state="local",
        )
        self._record_visible_turn(
            user_text,
            text,
            allow_persistence,
            act,
            include_in_generation_context=False,
            tool_result=tool_result,
        )
        self._trace(metrics)
        return result

    def schedule_task(
        self,
        *,
        objective: str,
        origin_event_id: str,
        tool: str,
        operation,
        parameters: dict[str, object] | None = None,
        permissions: tuple[str, ...] = (),
        requires_confirmation: bool = False,
    ) -> RealtimeTask:
        if self.task_manager is None:
            raise RuntimeError("Realtime Phase 1 task management is disabled.")
        task = self.task_manager.propose(
            objective=objective,
            origin_event_id=origin_event_id,
            tool=tool,
            parameters=parameters,
            permissions=permissions,
            requires_confirmation=requires_confirmation,
        )
        self.session.register_commitment(
            objective=objective,
            task_id=task.task_id,
            status=task.status,
        )
        if not requires_confirmation:
            self.task_manager.queue(task.task_id, operation)
        return task

    def drain_notifications(self) -> tuple[str, ...]:
        if self.task_manager is None:
            return ()
        notifications = self.task_manager.drain_notifications()
        messages: list[str] = []
        for notification in notifications:
            text = notification.visible_text()
            messages.append(text)
            self.session.mark_action_resolved(task_id=notification.task_id)
            self.session.commit_task_result(
                text,
                objective=notification.objective,
            )
            if self.calendar_events is not None:
                task = self.task_manager.get(notification.task_id)
                receipt = dict(notification.result.get("receipt") or {})
                if task is not None and receipt:
                    self.calendar_events.record_verified_creation(
                        receipt=receipt,
                        parameters=task.parameters,
                        task_id=notification.task_id,
                    )
            if self.journal is not None:
                event = ConversationEvent.create(
                    author="anydes",
                    exact_text=text,
                    session_id=self.session.session_id,
                    event_type="task_result",
                    topic=notification.objective,
                    related_event_ids=(
                        (self._last_journal_event_id,)
                        if self._last_journal_event_id
                        else ()
                    ),
                    tool_result={
                        "task_id": notification.task_id,
                        "status": notification.status,
                        "result": notification.result,
                        "error_type": notification.error_type,
                    },
                    category="Personal",
                )
                self.journal.remember((event,))
                self._last_journal_event_id = event.event_id
                if not self._closed:
                    self._persistence.submit(self.journal.persist, (event,))
        return tuple(messages)

    def _handle_phase2_action(
        self,
        action: SemanticAction,
        *,
        user_text: str,
        act: RealtimeAct,
        started: float,
        allow_persistence: bool,
        on_text: Callable[[str], None] | None,
        capability: CapabilityState,
    ) -> RealtimeResult | None:
        registry = self.capability_registry
        executor = self.action_executor
        if registry is None or executor is None:
            return None
        resolution = registry.resolve(action)
        if action.mode == "inquire":
            return self._finish_phase2_inquiry(
                action,
                resolution,
                user_text=user_text,
                act=act,
                started=started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        if action.capability == "calendar_create":
            draft = self.session.register_calendar_draft(
                dict(action.parameters)
            )
            action = replace(
                action,
                entity=draft.title,
                parameters=draft.as_parameters(),
            )
            missing = tuple(action.parameters.get("missing_fields") or ())
            if resolution.availability == "unavailable":
                return self._finish_local(
                    user_text,
                    "No encuentro un calendario local conectado en el que pueda crear eventos.",
                    "phase2:calendar_unavailable",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
            if missing:
                labels = {
                    "title": "qué actividad es",
                    "date": "la fecha",
                    "time": "la hora",
                }
                needed = ", ".join(labels.get(item, item) for item in missing)
                self.session.register_action_proposal(
                    action.capability,
                    entity=action.entity,
                    parameters=action.parameters,
                    source="phase2",
                    requires_confirmation=True,
                )
                return self._finish_local(
                    user_text,
                    f"Para prepararlo bien me falta {needed}.",
                    "phase2:calendar_incomplete",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
            if action.mode != "confirm":
                summary = _calendar_action_summary(action)
                self.session.register_action_proposal(
                    action.capability,
                    entity=action.entity,
                    parameters=action.parameters,
                    source="phase2",
                    requires_confirmation=True,
                )
                return self._finish_local(
                    user_text,
                    f"Preparé este evento: {summary}. ¿Quieres que lo cree así?",
                    "phase2:calendar_confirmation",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
        if (
            action.capability == "open_application"
            and resolution.reason == "application_ambiguous"
        ):
            alternatives = tuple(resolution.alternatives[:4])
            self.session.register_action_proposal(
                action.capability,
                entity=action.entity,
                parameters={
                    **dict(action.parameters),
                    "alternatives": list(alternatives),
                },
                source="phase2",
                requires_confirmation=False,
            )
            choices = ", ".join(alternatives)
            return self._finish_local(
                user_text,
                f"Encontré varias aplicaciones parecidas: {choices}. ¿Cuál quieres abrir?",
                "phase2:open_application:ambiguous",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        if resolution.availability == "unavailable":
            if resolution.fallback:
                fallback = SemanticAction(
                    capability=resolution.fallback,
                    mode="confirm",
                    entity=action.entity,
                    parameters=dict(action.parameters),
                    origin_text=user_text,
                )
                self.session.register_action_proposal(
                    fallback.capability,
                    entity=fallback.entity,
                    parameters=fallback.parameters,
                    source="phase2",
                    requires_confirmation=True,
                )
                return self._finish_local(
                    user_text,
                    _fallback_offer(action, fallback),
                    "phase2:fallback_confirmation",
                    act,
                    started,
                    allow_persistence=allow_persistence,
                    on_text=on_text,
                    capability=capability,
                )
            receipt = ExecutionReceipt(
                capability=action.capability,
                status="unavailable",
                verified=False,
                error=resolution.reason,
            )
            return self._finish_local(
                user_text,
                receipt.visible_text(),
                f"phase2:{action.capability}:unavailable",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        if resolution.asynchronous:
            objective = _phase2_objective(action)

            def operation() -> dict[str, object]:
                receipt = executor.execute(action)
                return {
                    "message": receipt.visible_text(),
                    "receipt": receipt.safe_dict(),
                    "_task_success": receipt.status == "success",
                    "_task_error": receipt.error or receipt.status,
                }

            task = self.schedule_task(
                objective=objective,
                origin_event_id=(
                    f"pending:{self.session.session_id}:"
                    f"{self.session.snapshot().turn_count + 1}"
                ),
                tool=resolution.tool_name,
                operation=operation,
                parameters=action.parameters,
            )
            self.session.register_action_proposal(
                action.capability,
                entity=action.entity,
                task_id=task.task_id,
                parameters=action.parameters,
                source="phase2",
            )
            return self._finish_local(
                user_text,
                _phase2_started_text(action),
                f"phase2:{action.capability}:queued",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        receipt = executor.execute(action)
        if receipt.status != "success" and resolution.fallback:
            fallback = SemanticAction(
                capability=resolution.fallback,
                mode="confirm",
                entity=action.entity,
                parameters=dict(action.parameters),
                origin_text=user_text,
            )
            self.session.register_action_proposal(
                fallback.capability,
                entity=fallback.entity,
                parameters=fallback.parameters,
                source="phase2",
                requires_confirmation=True,
            )
            return self._finish_local(
                user_text,
                _fallback_offer(action, fallback),
                "phase2:fallback_confirmation",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
                tool_result=receipt.tool_result,
            )
        self.session.mark_action_resolved()
        return self._finish_local(
            user_text,
            receipt.visible_text(),
            f"phase2:{action.capability}:{receipt.status}",
            act,
            started,
            allow_persistence=allow_persistence,
            on_text=on_text,
            capability=(
                capability.with_action_result(receipt.tool_result)
                if receipt.tool_result is not None
                else capability
            ),
            tool_result=receipt.tool_result,
        )

    def _finish_phase2_inquiry(
        self,
        action: SemanticAction,
        resolution,
        *,
        user_text: str,
        act: RealtimeAct,
        started: float,
        allow_persistence: bool,
        on_text,
        capability: CapabilityState,
    ) -> RealtimeResult:
        if (
            action.capability == "open_application"
            and resolution.reason == "application_ambiguous"
        ):
            alternatives = tuple(resolution.alternatives[:4])
            self.session.register_action_proposal(
                action.capability,
                entity=action.entity,
                parameters={
                    **dict(action.parameters),
                    "alternatives": list(alternatives),
                },
                source="phase2",
                requires_confirmation=False,
            )
            text = (
                "Encontré varias aplicaciones parecidas: "
                f"{', '.join(alternatives)}. ¿Cuál quieres usar?"
            )
            return self._finish_local(
                user_text,
                text,
                "phase2:open_application:ambiguous",
                act,
                started,
                allow_persistence=allow_persistence,
                on_text=on_text,
                capability=capability,
            )
        if resolution.availability == "unavailable":
            if resolution.fallback:
                fallback = SemanticAction(
                    capability=resolution.fallback,
                    mode="confirm",
                    entity=action.entity,
                    parameters=dict(action.parameters),
                    origin_text=user_text,
                )
                self.session.register_action_proposal(
                    fallback.capability,
                    entity=fallback.entity,
                    parameters=fallback.parameters,
                    source="phase2",
                    requires_confirmation=True,
                )
                text = _fallback_offer(action, fallback)
            else:
                text = "No encuentro esa capacidad disponible ahora mismo."
        else:
            text = _phase2_capability_available_text(action, resolution=resolution)
            self.session.register_action_proposal(
                action.capability,
                entity=action.entity,
                parameters=action.parameters,
                source="phase2",
                requires_confirmation=False,
            )
        return self._finish_local(
            user_text,
            text,
            f"phase2:{action.capability}:inquiry",
            act,
            started,
            allow_persistence=allow_persistence,
            on_text=on_text,
            capability=capability,
        )


def _requested_tool_name(intent) -> str:
    if str(getattr(intent, "name", "") or "") != "tool":
        return ""
    argument = str(getattr(intent, "argument", "") or "").strip()
    return argument.split(maxsplit=1)[0] if argument else ""


def _calendar_action_summary(action: SemanticAction) -> str:
    title = str(action.parameters.get("title") or "Actividad")
    raw_start = str(action.parameters.get("start_iso") or "")
    try:
        start = datetime.fromisoformat(raw_start)
    except ValueError:
        return title
    participants = tuple(action.parameters.get("participants") or ())
    companion = (
        f" con {participants[0]}"
        if participants
        and _normalized_text(str(participants[0])) not in _normalized_text(title)
        else ""
    )
    return (
        f"{title}{companion}, el {start.strftime('%d/%m/%Y')} "
        f"a las {start.strftime('%H:%M')}"
    )


def _phase2_objective(action: SemanticAction) -> str:
    if action.capability == "weather_current":
        return "consultar el clima real"
    if action.capability == "calendar_create":
        return f"crear el evento {action.parameters.get('title') or 'solicitado'}"
    return f"ejecutar {action.capability.replace('_', ' ')}"


def _phase2_started_text(action: SemanticAction) -> str:
    if action.capability == "weather_current":
        return "Voy a consultar el clima real y te aviso aquí apenas tenga el resultado."
    if action.capability == "calendar_create":
        return "Voy a guardarlo en el calendario y te confirmo aquí el resultado."
    return "Ya empecé esa tarea; te aviso aquí cuando termine."


def _fallback_offer(
    original: SemanticAction,
    fallback: SemanticAction,
) -> str:
    _ = fallback
    if original.capability in {"spotify_open", "spotify_search"}:
        return (
            "No pude confirmar Spotify como aplicación local. "
            "Puedo abrir la búsqueda en Spotify Web, ¿quieres que lo haga?"
        )
    return "Esa opción local no está disponible. Puedo intentar una alternativa web, ¿quieres?"


def _phase2_capability_available_text(
    action: SemanticAction,
    *,
    resolution=None,
) -> str:
    if action.capability == "calendar_create":
        if getattr(resolution, "reason", "") == "microsoft365_web_requires_user_save":
            return (
                "Sí. Puedo preparar el evento en tu calendario de Microsoft 365; "
                "antes de guardarlo tendrás que revisarlo y pulsar Guardar."
            )
        return "Sí, puedo crear eventos en el calendario conectado después de confirmarlos contigo."
    if action.capability == "spotify_open":
        return "Sí, puedo intentar abrir Spotify cuando me lo indiques."
    if action.capability == "spotify_search":
        return (
            "Sí, puedo abrir esa búsqueda en Spotify. "
            "Solo confirmaré reproducción si Spotify realmente la reporta."
        )
    if action.capability == "open_application":
        label = str(action.parameters.get("name") or action.entity)
        return f"Sí, puedo intentar abrir {label}."
    if action.capability == "open_browser_tab":
        if not action.parameters.get("url"):
            return "Sí. Dime qué página quieres que abra en la nueva pestaña."
        return "Sí, puedo abrir una pestaña nueva en tu navegador."
    if action.capability == "open_web_service":
        label = str(action.parameters.get("service") or action.entity)
        return f"Sí, puedo abrir {label} en el navegador."
    if action.capability == "open_url" and not action.parameters.get("url"):
        return "Sí. Dime qué página quieres que abra en la nueva pestaña."
    return "Sí, esa capacidad está disponible."


def _more_private_category(left: str, right: str) -> str:
    rank = {"Normal": 0, "Personal": 1, "Secretos": 2}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _safe_tool_result(result) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "tool_name": str(getattr(result, "tool_name", "") or ""),
        "status": str(getattr(result, "status", "") or ""),
        "data": dict(getattr(result, "data", {}) or {}),
        "error_type": (
            type(getattr(result, "error", None)).__name__
            if getattr(result, "error", None)
            else ""
        ),
    }


def _classify_realtime_failure(
    outcome,
    validation: ValidationResult,
    *,
    stream_error_type: str,
) -> tuple[str, str]:
    if validation.valid:
        return "", ""
    if outcome is None:
        code = stream_error_type or "unknown_exception"
        return f"transport_exception_{code}", "transport"
    if outcome.connection_state == "interrupted":
        reason = outcome.interruption_reason or outcome.error_type or "interrupted"
        return f"stream_{reason}", "transport"
    if "truncated_output" in validation.issues or outcome.done_reason == "length":
        return "generation_truncated", "generation"
    if not outcome.text.strip() or "empty_output" in validation.issues:
        return "model_empty_response", "generation"
    blocking = next(
        (
            issue
            for issue in validation.issues
            if issue
            not in {
                "length_trimmed",
                "question_echo_removed",
                "user_echo_removed",
                "resolved_question_repetition_removed",
                "unsupported_self_history_removed",
                "generic_tail_removed",
            }
        ),
        "rejected_output",
    )
    return f"validation_{blocking}", "validation"


def _recovery_text_for_failure(
    failure_code: str,
    validation_issues: tuple[str, ...],
) -> str:
    issues = set(validation_issues)
    if {
        "unverified_action_claim",
        "unbacked_task_commitment",
    } & issues:
        return "Todavía no tengo un resultado real de esa acción, así que no voy a fingir que ya está hecha."
    if "unverified_visual_claim" in issues:
        return "No tengo una observación visual fiable para responderte eso ahora mismo."
    if "question_treated_as_memory" in issues:
        return "Entendí que me estabas preguntando algo, pero esa respuesta no quedó bien formulada. Repítemela, porfa."
    if failure_code == "generation_truncated":
        return "La respuesta se cortó antes de quedar completa. Puedo intentarla otra vez."
    if failure_code == "model_empty_response":
        return "Esta vez no alcancé a formular una respuesta. Repítemelo, porfa."
    if failure_code.startswith("stream_") or failure_code.startswith(
        "transport_exception_"
    ):
        return "Mi respuesta se interrumpió antes de terminar. Puedo intentarla otra vez."
    if {"technical_leak", "internal_label"} & issues:
        return "No conseguí expresar esa respuesta de una forma clara. Repítemelo, porfa."
    return "No pude darte una respuesta fiable esta vez. Repítemelo, porfa."


def _last_complete_sentence(text: str) -> str:
    endings = [text.rfind("."), text.rfind("?"), text.rfind("!")]
    end = max(endings)
    return text[: end + 1].strip() if end >= 0 else ""


def _normalized_text(text: str) -> str:
    import unicodedata

    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(folded.split())


def _references_preference(text: str) -> bool:
    import re

    normalized = _normalized_text(text)
    return bool(
        re.search(
            r"\b(?:por\s+que|motivo|razon|estas\s+segura|que\s+tan\s+segura|"
            r"de\s+donde|desde\s+cuando|alternativa|otra\s+opcion|"
            r"podria\s+cambiar|ha\s+cambiado|por\s+un[oa]|"
            r"(?:ese|esa)\s+(?:color|fruta|comida|cancion|pelicula|receta)|"
            r"cual\s+(?:imaginarias|elegirias|preferirias))\b",
            normalized,
        )
    )
