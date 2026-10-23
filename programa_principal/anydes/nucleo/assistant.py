from __future__ import annotations

import base64
import json
import re
import threading
from dataclasses import replace
from datetime import datetime

from anydes.cognition import (
    ActionManager,
    ActionRequest,
    BodyController,
    BodyUpdateResult,
    ConversationIntent,
    ConversationPlanner,
    ContextBuilder,
    CuriosityWriter,
    Intention,
    MemoryWriter,
    MetaCognition,
    Observation,
    ReasoningEngine,
    RelationshipWriter,
    SelfWriter,
    SpeechIntention,
    UnifiedInferenceShadowRunner,
    user_text_observation,
)
from anydes.cognition.live_runtime.events import (
    CONVERSATION_RECORDED,
    DomainEvent,
    EventPublisher,
    NullEventPublisher,
)
from anydes.cognition.live_runtime.interfaces import MetricsCollector, RuntimeControl
from anydes.cognition.live_runtime.context_pipeline import (
    ContextSelectionPolicy,
    InternalContextAssembler,
    compare_contexts,
)
from anydes.cognition.live_runtime.metrics import (
    NullMetricsCollector,
    PipelineProfiler,
    approximate_token_count,
    current_interaction_id,
    set_current_interaction_measurement,
)
from anydes.cognition.privacy import context_for_interlocutor, identity_for_interlocutor
from anydes.cognition.reasoning import build_revision_prompt
from anydes.cerebro.llm import ConversationBrain, build_vision_brain
from anydes.cerebro.prompt_builder import BrainContext
from anydes.funciones_extra.registry import ToolRegistry
from anydes.identity import build_identity_core
from anydes.memoria.conversation_memory import classify_text
from anydes.memoria.manager import MemoryManager
from anydes.nucleo.audit import AuditLog
from anydes.nucleo.config import Config
from anydes.nucleo.consent import (
    external_access_decision,
    is_external_provider,
    session_consents,
)
from anydes.nucleo.intent import IntentMatcher
from anydes.nucleo.interlocutor import InterlocutorContext
from anydes.nucleo.permissions import PermissionManager
from anydes.nucleo.personality import PersonalityProfile
from anydes.vision.state import get_screen_state
from anydes.vision.observation_store import get_vision_observation_store
from anydes.vision.tools import ensure_screen_watching, latest_screen_base64
from anydes.vision.watcher import active_screen_frame_lease


class AnydesAssistant:
    def __init__(
        self,
        config: Config,
        tools: ToolRegistry,
        permissions: PermissionManager,
        audit: AuditLog,
        personality: PersonalityProfile,
        brain: ConversationBrain,
        vision_brain: ConversationBrain | None = None,
        metrics: MetricsCollector | None = None,
        events: EventPublisher | None = None,
        live_runtime: RuntimeControl | None = None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.permissions = permissions
        self.audit = audit
        self.personality = personality
        self.intent_matcher = IntentMatcher()
        self.brain = brain
        self.vision_brain = vision_brain or build_vision_brain(config)
        self.metrics = metrics or NullMetricsCollector()
        self.events = events or NullEventPublisher()
        self.live_runtime = live_runtime
        self._profiler = PipelineProfiler(self.metrics)
        self._context_selection_policy = ContextSelectionPolicy()
        self._internal_context_assembler = InternalContextAssembler(config)
        self._interaction_lock = threading.RLock()
        self.interlocutor = InterlocutorContext.typed(
            config.user_name,
            trusted_by_policy=config.typed_input_is_owner,
        )
        self.permissions.set_interlocutor(self.interlocutor)
        self.cognitive_self_state: dict[str, object] = {}
        self.last_internal_context = None
        self.last_intention = None
        self.last_conversation_plan = None
        self.last_reflection = None
        self.last_reflection_attempts = ()
        self.last_action_results = ()
        self.last_body_result = None
        self.last_memory_results = ()
        self.last_relationship_results = ()
        self.last_curiosity_results = ()
        self.last_self_results = ()
        self.last_cognitive_report = None
        self.last_live_snapshot_revision = None
        self.last_context_selection = None
        self.last_context_assembly = None
        self.last_live_context_candidate = None
        self.last_context_shadow_diff = None
        self.last_reasoning_route = ""
        self.last_reasoning_shadow_diff = None
        self.last_unified_shadow_intention = None
        self.last_unified_shadow_reflection = None
        self.last_unified_shadow_report = None
        self.last_realtime_result = None
        self._realtime_service = None

    def close(self) -> None:
        if self._realtime_service is not None:
            self._realtime_service.close()
        if self.live_runtime is not None:
            self.live_runtime.stop()

    def set_current_speaker(self, speaker_match) -> None:
        self.set_interlocutor(
            InterlocutorContext.from_voice_match(
                speaker_match,
                owner_name=self.config.user_name,
            )
        )
        verifier = getattr(self.permissions, "verifier", None)
        if verifier is not None and hasattr(verifier, "set_voice_match"):
            verifier.set_voice_match(speaker_match)

    def set_typed_interlocutor(self) -> None:
        self.set_interlocutor(
            InterlocutorContext.typed(
                self.config.user_name,
                trusted_by_policy=self.config.typed_input_is_owner,
            )
        )

    def set_interlocutor(self, interlocutor: InterlocutorContext) -> None:
        self.interlocutor = interlocutor
        self.permissions.set_interlocutor(interlocutor)

    @property
    def speaker_context(self) -> str:
        return self.interlocutor.prompt_context

    @property
    def speaker_allows_memory_context(self) -> bool:
        return self.interlocutor.allows_personal_memory

    def greet(self) -> str:
        """Backward-compatible boundary; opening the room never emits speech."""
        self.audit.record("session_ready", automatic_speech=False)
        return ""

    def initiate(self, moment: datetime | None = None) -> Intention | None:
        """Reason about a proactive opportunity without fabricating a user turn."""
        memory = MemoryManager.from_config(self.config)
        current = moment or datetime.now().astimezone()
        observation = Observation(
            id=f"obs:proactive_opportunity:{current.isoformat()}",
            source="system",
            summary="A quiet moment may allow Anydes to initiate something meaningful or remain silent.",
            confidence=1.0,
            salience=0.48,
            data={
                "event": "proactive_opportunity",
                "local_time": current.isoformat(),
                "silence_is_allowed": True,
            },
        )
        live_snapshot = self._capture_live_snapshot()
        context = self._context_for_observations(
            (observation,),
            cycle="proactive",
            live_snapshot=live_snapshot,
        )
        intention = self._reason_intention(
            text="",
            memory=memory,
            context=context,
            interactive=False,
            compatibility_model=_ProactiveCompatibilityReasoningModel(
                assistant=self,
                memory=memory,
                context=context,
            ),
            cycle="proactive",
        )
        self._observe_unified_inference_shadow(
            context,
            intention,
            cycle="proactive",
        )
        intention, _ = self._plan_intention(context, intention)
        allowed_modes = {"ask", "recall", "shift_topic", "share_observation", "listen"}
        if intention.conversation_intent.mode not in allowed_modes:
            self.audit.record(
                "proactive_silence",
                conversation_mode=intention.conversation_intent.mode,
            )
            return None

        reflection = self._reflection_layer.reflect(intention, context)
        if not reflection.approved:
            self.audit.record(
                "proactive_rejected",
                issues=list(reflection.blocking_issues),
            )
            return None

        if intention.conversation_intent.mode == "listen":
            report = MetaCognition().review(context, intention, reflection)
            self.last_internal_context = context
            self.last_reflection = reflection
            self.last_cognitive_report = report
            self.audit.record("proactive_silence", conversation_mode="listen")
            return None

        body_result = BodyController.from_config(
            self.config,
            events=self.events,
        ).apply(intention)
        relationship_results = RelationshipWriter.from_config(
            self.config,
            events=self.events,
        ).write(
            intention,
            reflection,
            context,
        )
        self_results = SelfWriter.from_config(
            self.config,
            events=self.events,
        ).write(
            intention,
            reflection,
        )
        curiosity_results = CuriosityWriter.from_config(
            self.config,
            events=self.events,
        ).write(
            intention,
            reflection,
        )
        self.last_self_results = self_results
        report = MetaCognition().review(context, intention, reflection)
        self._update_self_model_from_meta(report)
        self._remember_last_cycle(
            context,
            intention,
            reflection,
            (),
            body_result,
            (),
            relationship_results,
            curiosity_results,
            report,
        )
        self.audit.record(
            "proactive_intention",
            conversation_mode=intention.conversation_intent.mode,
            curiosity_id=intention.conversation_intent.curiosity_id,
        )
        return intention

    def respond(
        self,
        text: str,
        *,
        interactive: bool = True,
        input_mode: str | None = None,
    ) -> str:
        with self._interaction_lock:
            if input_mode == "typed":
                self.set_typed_interlocutor()
            elif input_mode not in {None, "voice"}:
                raise ValueError(f"Unsupported input mode: {input_mode}")
            with self._profiler.interaction("user_turn"):
                set_current_interaction_measurement(
                    "input_characters",
                    len(text),
                )
                return self._profiler.call(
                    "pipeline.total",
                    lambda: self._respond(text, interactive=interactive),
                    interaction="user_turn",
                )

    def respond_realtime(
        self,
        text: str,
        *,
        input_mode: str | None = None,
        on_text=None,
    ) -> str:
        """Use the reversible low-latency path without changing respond()."""
        with self._interaction_lock:
            if input_mode == "typed":
                self.set_typed_interlocutor()
            elif input_mode not in {None, "voice"}:
                raise ValueError(f"Unsupported input mode: {input_mode}")
            if self._realtime_service is None:
                from anydes.realtime import RealtimeConversationService

                self._realtime_service = RealtimeConversationService(
                    self.config,
                    self.personality,
                    tools=self.tools,
                    permissions=self.permissions,
                    audit=self.audit,
                )
            tool_match = self.intent_matcher.match(text)
            if self.config.realtime_phase4_enabled:
                self._refresh_realtime_visual_observation(text, tool_match)
            result = self._realtime_service.respond(
                text,
                allow_personal_memory=self.interlocutor.allows_personal_memory,
                allow_persistence=self.interlocutor.allows_owner_state_mutation,
                has_tool_intent=tool_match is not None,
                intent=tool_match,
                on_text=on_text,
            )
            self.last_realtime_result = result
            self.last_intention = Intention(
                id=f"intention:realtime:{datetime.now().astimezone().isoformat()}",
                intent_type=(
                    "realtime_deferred" if result.deferred else "realtime_respond"
                ),
                conversation_intent=ConversationIntent(mode="respond"),
                speech_intention=SpeechIntention(
                    text=result.text,
                    emotion="calm",
                    energy=0.38,
                    warmth=0.82,
                    confidence=0.72,
                    thinking=0.12,
                    curiosity=0.35,
                    pause_before=0.08,
                    pause_after=0.14,
                    speech_rate=0.92,
                    pitch_variation=0.14,
                    volume=0.9,
                    smile=0.22,
                    metadata={"origin": "realtime_conversation"},
                ),
                avatar={
                    "expression": "calm",
                    "gesture": "speaking",
                    "gaze": "user",
                },
                confidence=0.72,
                reasoning_summary="",
            )
            self.audit.record(
                "realtime_response",
                route=result.metrics.route,
                model=result.metrics.model,
                response_complete_ms=round(
                    result.metrics.response_complete_ms,
                    3,
                ),
                inference_count=result.metrics.inference_count,
                validation_fallback=result.validation_fallback,
            )
            return result.text

    def _refresh_realtime_visual_observation(self, text: str, intent) -> None:
        if str(getattr(intent, "name", "") or "") != "screen_analyze":
            return
        if not self.config.vision_enabled:
            return
        if not session_consents.allows("screen_observation"):
            return
        from anydes.vision.residency import get_active_vision_residency_manager

        residency = get_active_vision_residency_manager()
        if residency is not None:
            residency.prefetch(reason="authorized_vision_intent")
        with active_screen_frame_lease(
            refresh=self.config.screen_fresh_on_analyze,
        ) as frame_ready:
            if self.config.screen_fresh_on_analyze and not frame_ready:
                self.audit.record(
                    "realtime_vision_frame_refresh_failed",
                    route="authorized_capture",
                )
                return
            AnydesAssistant._analyze_realtime_visual_frame(self)

    def _analyze_realtime_visual_frame(self) -> None:
        state = get_screen_state()
        watching, _captured_at, _frame_available = state.perception_snapshot()
        frame = state.latest_frame()
        if not watching or frame is None:
            return
        age = max(
            0.0,
            (
                datetime.now().astimezone()
                - frame.captured_at.astimezone()
            ).total_seconds(),
        )
        if age > self.config.realtime_frame_max_age_seconds:
            return
        provider = self.config.resolved_vision_provider()
        if is_external_provider(provider) and not session_consents.allows(
            "external_vision"
        ):
            return
        context = BrainContext(
            user_name="usuario local",
            profile_name="Anydes",
            memory_summary="",
            memory_context="",
            external_ai_allowed=(
                is_external_provider(provider)
                and session_consents.allows("external_vision")
            ),
            speaker_context="Interlocutor local; sin memoria compartida.",
            screen_sharing_active=True,
            screen_analysis=True,
            conversation_history=(),
            identity_context="Anydes observa únicamente el frame autorizado.",
            internal_state_context="",
            situational_context="",
            relationship_context={},
            evidence_context=(),
        )
        response = self.vision_brain.generate(
            "Describe this image.",
            self.personality,
            context,
            screen_image_base64=base64.b64encode(frame.jpeg_bytes).decode("ascii"),
        )
        provider_status = response.provider.casefold()
        if any(
            marker in provider_status
            for marker in (
                "missing",
                "unavailable",
                "error",
                "required",
                "disabled",
                "no-vision",
            )
        ):
            self.audit.record(
                "realtime_vision_description_unavailable",
                provider=response.provider,
                frame_hash=frame.frame_hash,
                frame_source_id=frame.source_id,
                frame_session_id=frame.session_id,
            )
            return
        summary = " ".join(response.text.strip().split())
        if (
            not summary
            or len(summary) > 500
            or sum(character.isalpha() for character in summary) < 4
        ):
            self.audit.record(
                "realtime_vision_description_invalid",
                frame_hash=frame.frame_hash,
                frame_source_id=frame.source_id,
                frame_session_id=frame.session_id,
                provider=response.provider,
            )
            return
        current_frame = state.latest_frame()
        if (
            not session_consents.allows("screen_observation")
            or current_frame is None
            or current_frame.frame_hash != frame.frame_hash
            or current_frame.source_id != frame.source_id
            or current_frame.session_id != frame.session_id
        ):
            self.audit.record(
                "realtime_vision_frame_superseded",
                frame_hash=frame.frame_hash,
                frame_source_id=frame.source_id,
                frame_session_id=frame.session_id,
                provider=response.provider,
            )
            return
        get_vision_observation_store().publish(
            summary,
            confidence=0.78,
            salience=0.7,
            metadata={
                "frame_hash": frame.frame_hash,
                "frame_captured_at": frame.captured_at.isoformat(),
                "frame_width": frame.width,
                "frame_height": frame.height,
                "frame_source_id": frame.source_id,
                "frame_source_kind": frame.source_kind,
                "frame_session_id": frame.session_id,
                "vision_provider": response.provider,
                "vision_route": "authorized_realtime_frame",
            },
        )
        self.audit.record(
            "realtime_vision_observation_published",
            frame_hash=frame.frame_hash,
            frame_source_id=frame.source_id,
            frame_source_kind=frame.source_kind,
            frame_session_id=frame.session_id,
            frame_width=frame.width,
            frame_height=frame.height,
            frame_captured_at=frame.captured_at.isoformat(),
            observation_at=datetime.now().astimezone().isoformat(),
            provider=response.provider,
            route="authorized_realtime_frame",
        )

    def warmup_realtime(self) -> float:
        if self.config.resolved_conversation_mode() != "realtime":
            return 0.0
        with self._interaction_lock:
            if self._realtime_service is None:
                from anydes.realtime import RealtimeConversationService

                self._realtime_service = RealtimeConversationService(
                    self.config,
                    self.personality,
                    tools=self.tools,
                    permissions=self.permissions,
                    audit=self.audit,
                )
            return self._realtime_service.warmup()

    def drain_realtime_notifications(self) -> tuple[str, ...]:
        """Return verified asynchronous results without entering the response pipeline."""
        service = self._realtime_service
        if service is None:
            return ()
        return service.drain_notifications()

    def _respond(self, text: str, *, interactive: bool = True) -> str:
        self.last_reflection_attempts = ()
        self.last_unified_shadow_intention = None
        self.last_unified_shadow_reflection = None
        self.last_unified_shadow_report = None
        memory = self._profiler.call(
            "pipeline.memory_manager",
            lambda: MemoryManager.from_config(self.config),
        )
        self.audit.record(
            "user_message",
            category=classify_text(text),
            character_count=len(text),
        )

        observation = self._profiler.call(
            "pipeline.observation_reception",
            lambda: user_text_observation(text),
        )
        live_snapshot = self._capture_live_snapshot()
        context = self._context_for_observations(
            (observation,),
            cycle="initial",
            live_snapshot=live_snapshot,
        )
        context = self._context_with_available_actions(context)
        self._record_context_measurements(context)
        intention = self._profiler.call(
            "pipeline.reasoning",
            lambda: self._reason_intention(
                text=text,
                memory=memory,
                context=context,
                interactive=interactive,
                compatibility_model=_CompatibilityReasoningModel(
                    assistant=self,
                    text=text,
                    memory=memory,
                    context=context,
                    interactive=interactive,
                ),
                cycle="initial",
            ),
            cycle="initial",
        )
        self._profiler.call(
            "pipeline.unified_shadow",
            lambda: self._observe_unified_inference_shadow(
                context,
                intention,
                cycle="initial",
            ),
            cycle="initial",
        )
        intention, conversation_plan = self._profiler.call(
            "pipeline.conversation_planner",
            lambda: self._plan_intention(context, intention),
            cycle="initial",
        )
        reflection = self._profiler.call(
            "pipeline.reflection",
            lambda: self._reflection_layer.reflect(intention, context),
            cycle="initial",
        )
        intention, conversation_plan, reflection = self._resolve_reflection(
            text=text,
            memory=memory,
            context=context,
            intention=intention,
            conversation_plan=conversation_plan,
            reflection=reflection,
            interactive=interactive,
            cycle="initial",
        )
        action_results = self._profiler.call(
            "pipeline.action_manager",
            lambda: self._action_manager.execute(
                intention,
                reflection,
                interactive=interactive,
            ),
            cycle="initial",
        )
        response_context = context
        response_intention = intention
        response_reflection = reflection
        if action_results:
            tool_observations = self._action_manager.observations_from_results(action_results)
            response_context = self._context_for_observations(
                (observation, *tool_observations),
                cycle="tool_result",
                live_snapshot=live_snapshot,
            )
            response_context = self._context_with_available_actions(response_context)
            self._record_context_measurements(response_context)
            response_intention = self._profiler.call(
                "pipeline.reasoning",
                lambda: self._reason_intention(
                    text=text,
                    memory=memory,
                    context=response_context,
                    interactive=interactive,
                    compatibility_model=_ToolObservationReasoningModel(
                        assistant=self,
                        original_text=text,
                        context=response_context,
                        memory=memory,
                    ),
                    cycle="tool_result",
                ),
                cycle="tool_result",
            )
            self._profiler.call(
                "pipeline.unified_shadow",
                lambda: self._observe_unified_inference_shadow(
                    response_context,
                    response_intention,
                    cycle="tool_result",
                ),
                cycle="tool_result",
            )
            response_intention, conversation_plan = self._profiler.call(
                "pipeline.conversation_planner",
                lambda: self._plan_intention(response_context, response_intention),
                cycle="tool_result",
            )
            response_reflection = self._profiler.call(
                "pipeline.reflection",
                lambda: self._reflection_layer.reflect(
                    response_intention,
                    response_context,
                ),
                cycle="tool_result",
            )
            (
                response_intention,
                conversation_plan,
                response_reflection,
            ) = self._resolve_reflection(
                text=text,
                memory=memory,
                context=response_context,
                intention=response_intention,
                conversation_plan=conversation_plan,
                reflection=response_reflection,
                interactive=interactive,
                cycle="tool_result",
            )

        response_intention = self._attach_interaction_trace(response_intention)

        if response_reflection.approved:
            body_result = self._profiler.call(
                "pipeline.body",
                lambda: BodyController.from_config(
                    self.config,
                    events=self.events,
                ).apply(response_intention),
            )
        else:
            body_result = BodyUpdateResult(
                intention_id=response_intention.id,
                status="skipped",
                message="reflection_not_approved",
            )
        if (
            self.interlocutor.allows_owner_state_mutation
            and response_reflection.approved
        ):
            memory_results = self._profiler.call(
                "pipeline.memory_writer",
                lambda: MemoryWriter(memory, events=self.events).write(
                    response_intention,
                    response_reflection,
                ),
            )
            relationship_results = self._profiler.call(
                "pipeline.relationship_writer",
                lambda: RelationshipWriter.from_config(
                    self.config,
                    events=self.events,
                ).write(
                    response_intention,
                    response_reflection,
                    response_context,
                ),
            )
            self_results = self._profiler.call(
                "pipeline.self_writer",
                lambda: SelfWriter.from_config(
                    self.config,
                    events=self.events,
                ).write(
                    response_intention,
                    response_reflection,
                ),
            )
            curiosity_results = self._profiler.call(
                "pipeline.curiosity_writer",
                lambda: CuriosityWriter.from_config(
                    self.config,
                    events=self.events,
                ).write(
                    response_intention,
                    response_reflection,
                ),
            )
        else:
            memory_results = ()
            relationship_results = ()
            self_results = ()
            curiosity_results = ()
            self.audit.record(
                (
                    "owner_state_write_blocked"
                    if not self.interlocutor.allows_owner_state_mutation
                    else "owner_state_write_skipped_by_reflection"
                ),
                interlocutor_state=self.interlocutor.state,
                input_channel=self.interlocutor.channel,
            )
        self.last_self_results = self_results
        report = self._profiler.call(
            "pipeline.metacognition",
            lambda: MetaCognition().review(
                response_context,
                response_intention,
                response_reflection,
                action_results=action_results,
                memory_results=memory_results,
            ),
        )
        response_text = self._response_from_cognitive_cycle(response_intention)
        if self.interlocutor.allows_owner_state_mutation:
            self._update_self_model_from_meta(report)
        self._remember_last_cycle(
            response_context,
            response_intention,
            response_reflection,
            action_results,
            body_result,
            memory_results,
            relationship_results,
            curiosity_results,
            report,
        )
        if relationship_results:
            self.audit.record(
                "relationship_updates",
                statuses=[item.status for item in relationship_results],
                operations=[item.operation for item in relationship_results],
            )
        if self_results:
            self.audit.record(
                "self_updates",
                statuses=[item.status for item in self_results],
                operations=[item.operation for item in self_results],
                keys=[item.key for item in self_results],
            )
        if curiosity_results:
            self.audit.record(
                "curiosity_updates",
                statuses=[item.status for item in curiosity_results],
                operations=[item.operation for item in curiosity_results],
            )
        self.audit.record("metacognition_report", quality=report.quality)
        cognitively_retrievable = (
            response_intention.speech_intention.metadata.get("origin")
            != "recovery_intention"
            and response_intention.intent_type
            not in {"reasoning_parse_error", "reflection_recovery"}
        )
        rejection_reasons = tuple(
            dict.fromkeys(
                issue
                for attempt in self.last_reflection_attempts
                if not attempt.approved
                for issue in attempt.blocking_issues
            )
        )
        return self._profiler.call(
            "pipeline.finalize",
            lambda: self._finalize(
                text,
                response_text,
                memory,
                cognitively_retrievable=cognitively_retrievable,
                rejection_reasons=rejection_reasons,
            ),
        )

    def _capture_live_snapshot(self):
        mode = self.config.resolved_live_context_mode()
        if mode == "legacy" or self.live_runtime is None:
            return None
        try:
            return self._profiler.call(
                "pipeline.live_snapshot",
                self.live_runtime.snapshot,
                mode=mode,
            )
        except Exception as exc:
            if self.config.developer_mode:
                self.audit.record(
                    "live_context_snapshot_failed",
                    error_type=type(exc).__name__,
                )
            return None

    def _context_for_observations(
        self,
        observations: tuple[Observation, ...],
        *,
        cycle: str,
        live_snapshot,
    ):
        mode = self.config.resolved_live_context_mode()
        if mode == "legacy" or live_snapshot is None:
            return self._legacy_context(observations, cycle=cycle)

        selection = self._profiler.call(
            "pipeline.context_planning",
            lambda: self._context_selection_policy.select(observations),
            cycle=cycle,
        )
        set_current_interaction_measurement(
            "selected_providers",
            len(selection.provider_ids),
        )
        try:
            assembly = self._profiler.call(
                "pipeline.internal_context_assembler",
                lambda: self._internal_context_assembler.assemble(
                    live_snapshot,
                    selection,
                    observations,
                ),
                cycle=cycle,
                profile=selection.profile,
            )
        except Exception as exc:
            if self.config.developer_mode:
                self.audit.record(
                    "live_context_assembly_failed",
                    cycle=cycle,
                    profile=selection.profile,
                    root_revision=live_snapshot.revision,
                    error_type=type(exc).__name__,
                )
            return self._legacy_context(observations, cycle=cycle)

        candidate = context_for_interlocutor(assembly.context, self.interlocutor)
        self.last_live_snapshot_revision = assembly.root_revision
        self.last_context_selection = selection
        self.last_context_assembly = assembly
        self.last_live_context_candidate = candidate
        self.metrics.gauge(
            "live_context_root_revision",
            float(assembly.root_revision),
            cycle=cycle,
        )
        self.metrics.gauge(
            "live_context_selected_providers",
            float(len(selection.provider_ids)),
            profile=selection.profile,
        )
        if mode == "live":
            return candidate

        legacy = self._legacy_context(observations, cycle=cycle)
        difference = compare_contexts(candidate, legacy)
        self.last_context_shadow_diff = difference
        if self.config.developer_mode:
            self.audit.record(
                "live_context_shadow_comparison",
                cycle=cycle,
                profile=selection.profile,
                root_revision=assembly.root_revision,
                matches=difference.matches,
                changed_sections=list(difference.changed_sections),
                missing_provider_ids=list(assembly.missing_provider_ids),
                stale_provider_ids=list(assembly.stale_provider_ids),
            )
        return legacy

    def _legacy_context(
        self,
        observations: tuple[Observation, ...],
        *,
        cycle: str,
    ):
        context = self._profiler.call(
            "pipeline.context_builder",
            lambda: ContextBuilder(
                self.config,
                profiler=self._profiler,
            ).build(observations),
            cycle=cycle,
        )
        return context_for_interlocutor(context, self.interlocutor)

    @staticmethod
    def _record_context_measurements(context) -> None:
        serialized = context.to_json()
        memory_count = (
            len(context.memory.working)
            + len(context.memory.short)
            + len(context.memory.long)
        )
        set_current_interaction_measurement(
            "internal_context_characters",
            len(serialized),
        )
        set_current_interaction_measurement(
            "internal_context_approx_tokens",
            approximate_token_count(len(serialized)),
        )
        set_current_interaction_measurement(
            "memories_retrieved",
            memory_count,
        )
        set_current_interaction_measurement(
            "open_loops_considered",
            len(context.relationship.open_loops),
        )
        set_current_interaction_measurement(
            "promises_considered",
            len(context.relationship.promises),
        )
        set_current_interaction_measurement(
            "curiosities_considered",
            len(context.curiosity),
        )
        set_current_interaction_measurement(
            "observations_considered",
            len(context.observations),
        )

    @staticmethod
    def _attach_interaction_trace(intention: Intention) -> Intention:
        interaction_id = current_interaction_id()
        if not interaction_id:
            return intention
        metadata = dict(intention.speech_intention.metadata)
        metadata["interaction_trace_id"] = interaction_id
        speech = replace(intention.speech_intention, metadata=metadata)
        return replace(
            intention,
            speech=speech.text,
            speech_intention=speech,
        )

    def _plan_intention(self, context, intention: Intention):
        planner = ConversationPlanner()
        plan = planner.plan(context, intention)
        planned = planner.apply(intention, plan)
        self.last_conversation_plan = plan
        self.audit.record(
            "conversation_plan",
            primary_mode=plan.primary.mode,
            direct_question=plan.direct_question,
            deferred_curiosity_ids=list(plan.deferred_curiosity_ids),
        )
        return planned, plan

    def _reason_intention(
        self,
        *,
        text: str,
        memory: MemoryManager,
        context,
        interactive: bool,
        compatibility_model,
        cycle: str,
    ) -> Intention:
        _ = (text, memory, interactive)
        mode = self.config.resolved_reasoning_mode()
        model_factory = getattr(self.brain, "intention_model", None)
        native_model = model_factory() if callable(model_factory) else None
        compatibility_engine = ReasoningEngine(
            model=compatibility_model,
            error_reporter=self._record_reasoning_error,
        )
        if mode == "legacy" or native_model is None:
            self.last_reasoning_route = "legacy"
            if mode != "legacy" and self.config.developer_mode:
                self.audit.record(
                    "native_reasoning_unavailable",
                    cycle=cycle,
                    provider=type(self.brain).__name__,
                )
            return compatibility_engine.reason(context)

        native = ReasoningEngine(
            model=native_model,
            error_reporter=self._record_reasoning_error,
        ).reason(context)
        self.last_reasoning_route = "native"
        if mode == "native":
            return native

        legacy = compatibility_engine.reason(context)
        difference = _compare_intention_structure(native, legacy)
        self.last_reasoning_route = "shadow"
        self.last_reasoning_shadow_diff = difference
        if self.config.developer_mode:
            self.audit.record(
                "reasoning_shadow_comparison",
                cycle=cycle,
                **difference,
            )
        return legacy

    def _context_with_available_actions(self, context):
        if any(item.source == "tool" for item in context.observations):
            return replace(context, available_actions=())
        capabilities = []
        for tool in self.tools.list_tools():
            policy = self.permissions.policy_for(tool.name)
            if policy.level == "blocked" or not self.permissions.interlocutor_allows(tool.name):
                continue
            capabilities.append(
                ActionRequest(
                    id=f"capability:{tool.name}",
                    action_type=tool.name,
                    requires_confirmation=policy.level == "confirm",
                    risk_level="medium" if policy.level == "confirm" else "low",
                    reason=tool.description,
                )
            )
        return replace(context, available_actions=tuple(capabilities))

    def _observe_unified_inference_shadow(
        self,
        context,
        baseline: Intention,
        *,
        cycle: str,
    ) -> None:
        if self.config.resolved_unified_inference_mode() != "shadow":
            return
        model_factory = getattr(self.brain, "unified_intention_model", None)
        model = model_factory() if callable(model_factory) else None
        if model is None:
            if self.config.developer_mode:
                self.audit.record(
                    "unified_inference_shadow_unavailable",
                    cycle=cycle,
                    provider=type(self.brain).__name__,
                )
            return

        candidate, reflection, report = UnifiedInferenceShadowRunner(
            model=model,
            error_reporter=self._record_reasoning_error,
        ).run(
            context,
            baseline,
            cycle=cycle,
        )
        self.last_unified_shadow_intention = candidate
        self.last_unified_shadow_reflection = reflection
        self.last_unified_shadow_report = report
        if self.config.developer_mode:
            self.audit.record(
                "unified_inference_shadow_comparison",
                **report.to_dict(),
            )

    def _resolve_reflection(
        self,
        *,
        text: str,
        memory: MemoryManager,
        context,
        intention: Intention,
        conversation_plan,
        reflection,
        interactive: bool,
        cycle: str,
    ):
        attempts = [reflection]
        if reflection.approved:
            self.last_reflection_attempts = (
                *self.last_reflection_attempts,
                *attempts,
            )
            return intention, conversation_plan, reflection

        if reflection.decision == "revise":
            revision_limit = 1
            for revision_number in range(1, revision_limit + 1):
                revision_engine = ReasoningEngine(
                    model=_ReflectionRevisionReasoningModel(
                        assistant=self,
                        text=text,
                        memory=memory,
                        context=context,
                        original=intention,
                        reflection=reflection,
                        interactive=interactive,
                    ),
                    error_reporter=self._record_reasoning_error,
                )
                revision_cycle = f"{cycle}_revision_{revision_number}"
                revised = self._profiler.call(
                    "pipeline.reasoning",
                    lambda: revision_engine.reason(context),
                    cycle=revision_cycle,
                )
                revised, revised_plan = self._profiler.call(
                    "pipeline.conversation_planner",
                    lambda: self._plan_intention(context, revised),
                    cycle=revision_cycle,
                )
                revised_reflection = self._profiler.call(
                    "pipeline.reflection",
                    lambda: self._reflection_layer.reflect(revised, context),
                    cycle=revision_cycle,
                )
                attempts.append(revised_reflection)
                if revised_reflection.approved:
                    self.last_reflection_attempts = (
                        *self.last_reflection_attempts,
                        *attempts,
                    )
                    self.audit.record(
                        "reflection_reformulation_approved",
                        cycle=cycle,
                        attempts=len(attempts),
                    )
                    return revised, revised_plan, revised_reflection
                intention = revised
                conversation_plan = revised_plan
                reflection = revised_reflection
                if reflection.decision != "revise":
                    break

        recovery_speech = {
            "ask_confirmation": "Necesito que me confirmes eso antes de continuar.",
            "block": "No puedo continuar con eso de esa forma.",
        }.get(
            reflection.decision,
            "Perdon... no consegui ordenar bien esa respuesta. Intentemoslo otra vez.",
        )
        recovery = ReasoningEngine(error_reporter=self._record_reasoning_error).recover(
            context,
            reason="Reflection rejected the proposed intention.",
            intent_type="reflection_recovery",
            failure_scope="reflection",
            speech=recovery_speech,
        )
        recovery, recovery_plan = self._profiler.call(
            "pipeline.conversation_planner",
            lambda: self._plan_intention(context, recovery),
            cycle=f"{cycle}_recovery",
        )
        recovery_reflection = self._profiler.call(
            "pipeline.reflection",
            lambda: self._reflection_layer.reflect(recovery, context),
            cycle=f"{cycle}_recovery",
        )
        attempts.append(recovery_reflection)
        self.last_reflection_attempts = (
            *self.last_reflection_attempts,
            *attempts,
        )
        self.audit.record(
            "reflection_recovery",
            cycle=cycle,
            approved=recovery_reflection.approved,
            attempts=len(attempts),
        )
        return recovery, recovery_plan, recovery_reflection

    def _reformulate_after_reflection(
        self,
        *,
        text: str,
        memory: MemoryManager,
        context,
        original: Intention,
        reflection,
    ) -> Intention:
        sanitized = _remove_rejected_side_effects(original, reflection.blocking_issues)
        if sanitized is not None:
            return sanitized

        brain_context = self._build_brain_context(
            text,
            memory,
            internal_context=context,
        )
        feedback = ", ".join(reflection.blocking_issues) or reflection.reason
        revision_context = replace(
            brain_context,
            situational_context=(
                f"{brain_context.situational_context}\n"
                "Internal reformulation requirement: answer the current user request "
                "naturally and correct these validation failures without mentioning them: "
                f"{feedback}"
            ).strip(),
        )
        response = self.brain.generate(text, self.personality, revision_context)
        self.audit.record("brain_response_revision", provider=response.provider)
        return _speech_intention(
            "reflection:reformulated",
            response.text,
            conversation_intent=response.conversation_intent,
            relationship_updates=response.relationship_updates,
            self_updates=response.self_updates,
            reasoning_summary="Reasoning Engine reformulated an intention after Reflection.",
        )

    def _record_reasoning_error(self, error: Exception) -> None:
        self.audit.record(
            "reasoning_failure",
            error_type=type(error).__name__,
            detail=str(error)[:500],
        )

    @property
    def _action_manager(self) -> ActionManager:
        return ActionManager(self.tools, self.permissions, self.audit)

    @property
    def _reflection_layer(self):
        from anydes.cognition import ReflectionLayer

        return ReflectionLayer()

    def _compatibility_intention(
        self,
        text: str,
        memory: MemoryManager,
        *,
        interactive: bool,
        internal_context=None,
    ) -> Intention:
        normalized = text.strip().lower()

        if not normalized:
            return _speech_intention("compatibility:empty", self.personality.soften("Estoy aqui. Escribe 'ayuda' para ver comandos."))

        if not self.speaker_allows_memory_context and (
            "memoria" in normalized or normalized.startswith("ejecutar memory")
        ):
            return _speech_intention(
                "compatibility:speaker_memory_block",
                "No puedo mostrar recuerdos con una voz que no reconozco bien. Verifica tu identidad o escribe desde la consola.",
            )

        intent = self.intent_matcher.match(text)
        if intent is not None:
            if intent.name == "screen_analyze":
                return self._screen_analysis_intention(text)
            intention = self._intention_for_matched_intent(intent.name, intent.argument)
            if intention is not None:
                return intention

        if self._should_answer_with_live_screen(text):
            return self._screen_analysis_intention(text)

        if normalized.startswith("ejecutar ") or normalized.startswith("run "):
            command = text.split(maxsplit=2)
            tool_name = command[1] if len(command) > 1 else ""
            argument = command[2] if len(command) > 2 else ""
            return _tool_intention(tool_name, argument)
        if re.search(r"\b(?:hora|fecha)\b", normalized):
            return _tool_intention("current_time", "")
        if "sistema" in normalized or "computadora" in normalized:
            return _tool_intention("system_info", "")

        context = self._build_brain_context(text, memory, internal_context=internal_context)
        response = self.brain.generate(text, self.personality, context)
        self.audit.record("brain_response", provider=response.provider)
        return _speech_intention(
            "compatibility:brain",
            response.text,
            conversation_intent=response.conversation_intent,
            relationship_updates=response.relationship_updates,
            self_updates=response.self_updates,
            reasoning_summary=f"Compatibility ConversationBrain provider: {response.provider}.",
        )

    def _intention_for_matched_intent(self, name: str, argument: str) -> Intention | None:
        if name == "empty":
            return _speech_intention("compatibility:empty", self.personality.soften("Estoy aqui. Escribe 'ayuda' para ver comandos."))
        if name == "help":
            return _speech_intention("compatibility:help", HELP_TEXT)
        if name == "tools":
            return _speech_intention("compatibility:tools", self._tool_list())
        if name == "personality":
            return _speech_intention("compatibility:personality", self.personality.describe())
        if name == "memory_origin":
            return _speech_intention("compatibility:memory_origin", MemoryManager.from_config(self.config).origin_summary())
        if name == "memory_profile":
            return _tool_intention("memory_profile", "")
        if name == "memory_recent":
            return _tool_intention("memory_recent", "")
        if name == "tool":
            tool_name, _, tool_argument = argument.partition(" ")
            return _tool_intention(tool_name, tool_argument)
        return None

    def _screen_analysis_intention(self, question: str) -> Intention:
        actions = [
            ActionRequest(
                id="action:screen_capture",
                action_type="screen_capture",
                parameters={"argument": question},
                requires_confirmation=False,
                risk_level="low",
                reason="A current frame is required as structured visual evidence.",
            )
        ]
        if (
            is_external_provider(self.config.resolved_vision_provider())
            and external_access_decision(self.config, "external_vision").allowed
        ):
            actions.append(
                ActionRequest(
                    id="action:external_vision_session",
                    action_type="external_vision_session",
                    requires_confirmation=False,
                    risk_level="low",
                    reason="External visual interpretation requires separate session consent.",
                )
            )
        return Intention(
            id="intention:screen_analysis",
            intent_type="analyze_screen",
            speech="",
            actions=tuple(actions),
            confidence=0.9,
            reasoning_summary="Screen analysis proposed as reflected actions; no capture occurred during reasoning.",
        )

    def _response_from_cognitive_cycle(self, intention: Intention) -> str:
        return intention.speech

    def _remember_last_cycle(
        self,
        context,
        intention,
        reflection,
        action_results,
        body_result,
        memory_results,
        relationship_results,
        curiosity_results,
        report,
    ) -> None:
        self.last_internal_context = context
        self.last_intention = intention
        self.last_reflection = reflection
        self.last_action_results = tuple(action_results)
        self.last_body_result = body_result
        self.last_memory_results = tuple(memory_results)
        self.last_relationship_results = tuple(relationship_results)
        self.last_curiosity_results = tuple(curiosity_results)
        self.last_cognitive_report = report

    def _update_self_model_from_meta(self, report) -> None:
        self.cognitive_self_state.update(report.self_adjustments)

    def _respond_with_screen(self, text: str, *, interactive: bool) -> str:
        if not self.config.vision_enabled:
            return "La vision de pantalla esta desactivada. Activa ANYDES_VISION_ENABLED=true."

        screen_state = get_screen_state()
        if not session_consents.allows("screen_observation"):
            allowed = self.permissions.authorize(
                "screen_capture",
                "Captura la pantalla para que Anydes pueda ver lo que estas haciendo.",
                interactive=interactive,
            )
            if not allowed:
                return "Accion cancelada o bloqueada: screen_capture"

        provider = self.config.resolved_vision_provider()
        if is_external_provider(provider):
            configured = external_access_decision(self.config, "external_vision")
            if not configured.allowed:
                return "La vision externa esta desactivada por la configuracion de privacidad."
            if not session_consents.allows("external_vision"):
                allowed = self.permissions.authorize(
                    "external_vision_session",
                    f"Envia esta captura a {provider}; no incluye memoria ni historial.",
                    interactive=interactive,
                )
                if not allowed:
                    return "No envie la captura al proveedor externo."
                session_consents.grant("external_vision")

        session_consents.grant("screen_observation")

        ensure_screen_watching(self.config)

        screen_b64 = latest_screen_base64(self.config, fresh=self.config.screen_fresh_on_analyze)
        if not screen_b64:
            return (
                "No pude capturar tu pantalla. Instala dependencias con "
                "`pip install mss Pillow` y vuelve a intentarlo."
            )

        memory = MemoryManager.from_config(self.config)
        context = self._build_brain_context(
            text,
            memory,
            screen_sharing_active=screen_state.is_watching(),
            screen_analysis=True,
            screen_gaming_coach=self.intent_matcher.wants_screen_gaming_advice(text),
        )
        self.audit.record(
            "screen_analyze_requested",
            category=classify_text(text),
            character_count=len(text),
        )
        response = self.vision_brain.generate(
            text,
            self.personality,
            context,
            screen_image_base64=screen_b64,
        )
        self.audit.record("brain_response", provider=response.provider)
        return response.text

    def _finalize(
        self,
        user_text: str,
        response_text: str,
        memory: MemoryManager,
        *,
        cognitively_retrievable: bool = True,
        rejection_reasons: tuple[str, ...] = (),
    ) -> str:
        if not self.interlocutor.allows_owner_state_mutation:
            self.audit.record(
                "conversation_not_persisted",
                interlocutor_state=self.interlocutor.state,
                input_channel=self.interlocutor.channel,
            )
            return response_text
        if cognitively_retrievable:
            memory.record_conversation_turn(
                user_text,
                response_text,
                cognitively_retrievable=True,
            )
        else:
            memory.record_rejected_conversation_turn(
                user_text,
                response_text,
                rejection_reasons=rejection_reasons,
            )
        try:
            self.events.publish(
                DomainEvent(
                    event_type=CONVERSATION_RECORDED,
                    source="AnydesAssistant",
                    aggregate_id="conversation_memory",
                    payload={"recorded": True},
                )
            )
        except Exception:
            pass
        return response_text

    def _tool_list(self) -> str:
        rows = []
        for tool in self.tools.list_tools():
            policy = self.permissions.policy_for(tool.name)
            rows.append(f"{tool.name} [{policy.level}] - {tool.description}")
        return "\n".join(rows)

    def _should_answer_with_live_screen(self, text: str) -> bool:
        if not self.config.vision_enabled:
            return False
        if not get_screen_state().is_watching():
            return False
        normalized = text.strip().lower()
        if not normalized:
            return False
        if self.intent_matcher.match(text) is not None:
            return False
        live_hints = (
            "que ves",
            "que veo",
            "que hay",
            "que pasa",
            "que esta pasando",
            "que estoy",
            "que hago",
            "que deberia",
            "que debo",
            "me ayudas",
            "ayudame",
            "recomiendas",
            "opinas",
            "piensas",
            "en pantalla",
            "aqui",
            "aca",
            "ahora",
            "esto",
            "esta pantalla",
        )
        return any(hint in normalized for hint in live_hints)

    def _build_brain_context(
        self,
        text: str,
        memory: MemoryManager,
        *,
        screen_sharing_active: bool = False,
        screen_analysis: bool = False,
        screen_gaming_coach: bool = False,
        internal_context=None,
        interaction_kind: str = "user_turn",
    ) -> BrainContext:
        history = (
            tuple(memory.conversation_history_pairs(limit=12))
            if self.interlocutor.allows_personal_memory
            else ()
        )
        identity_core = build_identity_core(
            self.config,
            personality_profile=self.personality,
            memory=memory,
        )
        identity_core = identity_for_interlocutor(identity_core, self.interlocutor)
        memory_summary = (
            memory.profile_summary()
            if self.interlocutor.allows_personal_memory
            else "Memoria privada no disponible para este interlocutor."
        )
        return BrainContext(
            user_name=self.config.user_name,
            profile_name=self.config.profile_name,
            memory_summary=memory_summary,
            memory_context=(
                f"{memory.safe_context()}\n\n{memory.relevant_context(text)}"
                if self.speaker_allows_memory_context
                else "Memoria personal y protegida no disponible para este interlocutor."
            ),
            external_ai_allowed=(
                external_access_decision(
                    self.config,
                    "external_vision" if screen_analysis else "conversation_ai",
                    require_session=screen_analysis and is_external_provider(
                        self.config.resolved_vision_provider()
                    ),
                ).allowed
            ),
            speaker_context=self.speaker_context,
            screen_sharing_active=screen_sharing_active,
            screen_analysis=screen_analysis,
            screen_gaming_coach=screen_gaming_coach,
            conversation_history=history,
            identity_context=json.dumps(identity_core.to_dict(), ensure_ascii=False, sort_keys=True),
            internal_state_context=_internal_state_json(internal_context),
            situational_context=_situational_brief(internal_context),
            curiosity_context=(
                tuple(item.to_dict() for item in internal_context.curiosity)
                if internal_context is not None
                else ()
            ),
            relationship_context=_relationship_brain_snapshot(internal_context),
            evidence_context=_brain_evidence_context(internal_context),
            interaction_kind=interaction_kind,
        )

    def _reasoned_speech_from_tool_observations(self, text: str, context, memory: MemoryManager) -> str:
        tool_observations = [observation for observation in context.observations if observation.source == "tool"]
        if not tool_observations:
            return ""

        successful_tools = {
            str(observation.data.get("tool_name", ""))
            for observation in tool_observations
            if observation.data.get("status") == "executed"
        }
        if "screen_capture" in successful_tools:
            image = latest_screen_base64(self.config, fresh=False)
            if image:
                brain_context = self._build_brain_context(
                    text,
                    memory,
                    screen_sharing_active=get_screen_state().is_watching(),
                    screen_analysis=True,
                    internal_context=context,
                )
                response = self.vision_brain.generate(
                    text,
                    self.personality,
                    brain_context,
                    screen_image_base64=image,
                )
                self.audit.record("brain_response", provider=response.provider)
                return response.text

        if self.config.ai_provider.lower() != "local" and external_access_decision(
            self.config,
            "conversation_ai",
        ).allowed:
            tool_payload = [observation.to_dict() for observation in tool_observations]
            brain_context = self._build_brain_context(
                text,
                memory,
                internal_context=context,
            )
            response = self.brain.generate(
                (
                    "El usuario dijo esto:\n"
                    f"{text}\n\n"
                    "Estas son observaciones estructuradas producidas por herramientas. "
                    "No muestres JSON bruto; responde natural, breve y segun la personalidad:\n"
                    f"{json.dumps(tool_payload, ensure_ascii=False, sort_keys=True)}"
                ),
                self.personality,
                brain_context,
            )
            self.audit.record("brain_response", provider=response.provider)
            return response.text

        return self.personality.soften(_local_tool_observation_speech(tool_observations, self.config.user_name))


class _CompatibilityReasoningModel:
    def __init__(
        self,
        assistant: AnydesAssistant,
        text: str,
        memory: MemoryManager,
        context,
        *,
        interactive: bool,
    ) -> None:
        self.assistant = assistant
        self.text = text
        self.memory = memory
        self.context = context
        self.interactive = interactive

    def generate(self, prompt: str) -> str:
        _ = prompt
        intention = self.assistant._compatibility_intention(
            self.text,
            self.memory,
            interactive=self.interactive,
            internal_context=self.context,
        )
        return _intention_with_contextual_speech(intention, self.context).to_json()


class _ProactiveCompatibilityReasoningModel:
    """Keeps the legacy proactive route available while native reasoning migrates."""

    def __init__(
        self,
        assistant: AnydesAssistant,
        memory: MemoryManager,
        context,
    ) -> None:
        self.assistant = assistant
        self.memory = memory
        self.context = context

    def generate(self, prompt: str) -> str:
        _ = prompt
        brain_context = self.assistant._build_brain_context(
            "oportunidad de iniciativa autonoma",
            self.memory,
            internal_context=self.context,
            interaction_kind="proactive_opportunity",
        )
        response = self.assistant.brain.generate(
            (
                "No hay un mensaje nuevo del usuario. Esta es una oportunidad interna de "
                "iniciativa, no una orden. Compara la memoria, la relacion y el contexto. "
                "Elige una pregunta, recuerdo u observacion solo si aporta valor real; usa "
                "listen cuando interrumpir no valga la pena."
            ),
            self.assistant.personality,
            brain_context,
        )
        intention = _speech_intention(
            "cognitive:proactive",
            response.text,
            conversation_intent=response.conversation_intent,
            relationship_updates=response.relationship_updates,
            self_updates=response.self_updates,
            reasoning_summary=f"Proactive opportunity reasoned by {response.provider}.",
        )
        return _intention_with_contextual_speech(intention, self.context).to_json()


class _ReflectionRevisionReasoningModel:
    def __init__(
        self,
        assistant: AnydesAssistant,
        text: str,
        memory: MemoryManager,
        context,
        original: Intention,
        reflection,
        *,
        interactive: bool,
    ) -> None:
        self.assistant = assistant
        self.text = text
        self.memory = memory
        self.context = context
        self.original = original
        self.reflection = reflection
        self.interactive = interactive

    def generate(self, prompt: str) -> str:
        _ = self.interactive
        sanitized = _remove_rejected_side_effects(
            self.original,
            self.reflection.blocking_issues,
        )
        if sanitized is not None:
            return _intention_with_contextual_speech(sanitized, self.context).to_json()

        if self.assistant.config.resolved_reasoning_mode() == "native":
            model_factory = getattr(self.assistant.brain, "intention_model", None)
            native_model = model_factory() if callable(model_factory) else None
            if native_model is not None:
                feedback = ", ".join(self.reflection.blocking_issues) or self.reflection.reason
                failures = self.reflection.blocking_issues or (feedback,)
                native_revision = getattr(
                    native_model,
                    "generate_revision_for_context",
                    None,
                )
                if callable(native_revision):
                    return native_revision(
                        self.context,
                        self.original,
                        tuple(failures),
                    )
                return native_model.generate(
                    build_revision_prompt(
                        self.context,
                        self.original,
                        tuple(failures),
                    )
                )

        intention = self.assistant._reformulate_after_reflection(
            text=self.text,
            memory=self.memory,
            context=self.context,
            original=self.original,
            reflection=self.reflection,
        )
        return _intention_with_contextual_speech(intention, self.context).to_json()


class _ToolObservationReasoningModel:
    def __init__(
        self,
        assistant: AnydesAssistant,
        original_text: str,
        context,
        memory: MemoryManager,
    ) -> None:
        self.assistant = assistant
        self.original_text = original_text
        self.context = context
        self.memory = memory

    def generate(self, prompt: str) -> str:
        _ = prompt
        speech = self.assistant._reasoned_speech_from_tool_observations(
            self.original_text,
            self.context,
            self.memory,
        )
        intention = _speech_intention(
            "reasoned:tool_observation_response",
            speech,
            reasoning_summary="Tool outputs were converted into Observations before generating user-facing speech.",
        )
        return _intention_with_contextual_speech(intention, self.context).to_json()


def _local_tool_observation_speech(observations, user_name: str) -> str:
    observation = observations[-1]
    data = observation.data
    status = str(data.get("status", ""))
    tool_name = str(data.get("tool_name", ""))
    result = data.get("result", {})
    result_data = result if isinstance(result, dict) else {}
    error = data.get("error")

    if status != "executed":
        return _tool_failure_speech(tool_name, str(error or "permission_denied"))

    if tool_name == "current_time":
        return _time_speech(result_data)
    if tool_name == "word_count":
        return _word_count_speech(result_data)
    if tool_name == "system_info":
        return _system_info_speech(result_data)
    if tool_name == "format_json":
        return _format_json_speech(result_data)
    if tool_name in {"memory_profile", "memory_recent", "memory_conversation_search"}:
        return str(result_data.get("summary") or "Tengo esa informacion en memoria, pero no encontre detalles utiles.")
    if tool_name in {"memory_permanent_read", "memory_temporary_read", "memory_read"}:
        return _memory_read_speech(result_data)
    if tool_name in {"memory_permanent_write", "memory_temporary_write", "memory_write", "memory_protected_write"}:
        return _memory_write_speech(result_data)
    if tool_name == "memory_temporary_clear":
        return "Listo, borre la memoria temporal."
    if tool_name in {"open_url", "chrome_search", "open_youtube", "youtube_search", "open_spotify", "open_app"}:
        return _app_tool_speech(tool_name, result_data)
    if tool_name in {"screen_capture", "screen_watch_start", "screen_watch_stop", "screen_status"}:
        return _screen_tool_speech(tool_name, result_data)
    if result_data.get("blocked"):
        return "No pude hacerlo porque esa capacidad esta bloqueada en esta base segura."
    if "text" in result_data:
        return f"Recibi el resultado de {tool_name}, pero lo voy a tratar como dato interno antes de responderte."
    return f"Listo, ejecute {tool_name} y ya incorpore el resultado a mi contexto."


def _tool_failure_speech(tool_name: str, error: str) -> str:
    if error == "permission_denied":
        return f"Accion cancelada: no pude ejecutar {tool_name} porque necesita confirmacion o esta bloqueada."
    if error == "unknown_tool":
        return f"No conozco esa herramienta todavia: {tool_name}."
    return f"No pude ejecutar {tool_name}. El error quedo registrado como dato interno: {error}."


def _time_speech(data: dict[str, object]) -> str:
    raw_time = str(data.get("current_time", ""))
    try:
        moment = datetime.fromisoformat(raw_time)
        hour = moment.hour
        minute = moment.minute
        suffix = "de la madrugada"
        if 6 <= hour < 12:
            suffix = "de la manana"
        elif 12 <= hour < 18:
            suffix = "de la tarde"
        elif 18 <= hour < 24:
            suffix = "de la noche"
        display_hour = hour % 12 or 12
        return f"Ahora mismo son las {display_hour}:{minute:02d} {suffix}."
    except ValueError:
        return "Ya consulte la hora, pero no pude convertirla a un formato bonito."


def _word_count_speech(data: dict[str, object]) -> str:
    words = int(data.get("words", 0) or 0)
    characters = int(data.get("characters", 0) or 0)
    lines = int(data.get("lines", 0) or 0)
    return f"Conte {words} palabra(s), {characters} caracter(es) y {lines} linea(s)."


def _system_info_speech(data: dict[str, object]) -> str:
    os_name = data.get("os", "sistema desconocido")
    python_version = data.get("python_version", "version no detectada")
    machine = data.get("machine", "arquitectura no detectada")
    return f"Tu equipo esta corriendo {os_name} en {machine}, con Python {python_version}."


def _format_json_speech(data: dict[str, object]) -> str:
    formatted = str(data.get("formatted_json", "")).strip()
    if not formatted:
        return "El texto no produjo un JSON formateado usable."
    return f"El JSON es valido. Formateado queda asi:\n{formatted}"


def _memory_read_speech(data: dict[str, object]) -> str:
    tier = _memory_tier_label(str(data.get("tier", "memoria")))
    if not data.get("found"):
        return f"No encontre datos en la memoria {tier} para esa consulta."
    if data.get("key"):
        return f"En memoria {tier} tengo guardado {data.get('key')}: {data.get('value')}."
    items = data.get("items", {})
    if not isinstance(items, dict) or not items:
        return f"La memoria {tier} esta vacia."
    rendered = ", ".join(f"{key}: {value}" for key, value in sorted(items.items()))
    return f"En memoria {tier} tengo esto: {rendered}."


def _memory_write_speech(data: dict[str, object]) -> str:
    tier = _memory_tier_label(str(data.get("tier", "memoria")))
    key = str(data.get("key", "ese dato"))
    if data.get("saved"):
        return f"Listo, guarde {key} en memoria {tier}."
    if data.get("sensitive_rejected"):
        return "Ese dato parece sensible, asi que no lo guarde en memoria normal."
    if data.get("protected_store_enabled") is False:
        return "La memoria protegida no esta activada todavia, asi que no guarde ese dato sensible."
    return f"No pude guardar {key} en memoria {tier}."


def _memory_tier_label(tier: str) -> str:
    return {
        "temporary": "temporal",
        "permanent": "permanente",
        "protected": "protegida",
    }.get(tier, tier)


def _app_tool_speech(tool_name: str, data: dict[str, object]) -> str:
    if not data.get("opened"):
        if data.get("error") == "unknown_or_unapproved_app":
            return "Por seguridad no abri esa aplicacion porque no esta en la lista conocida."
        return "No pude abrir eso ahora."
    if tool_name in {"chrome_search", "youtube_search"}:
        return f"Ya abri la busqueda de {data.get('query', '')}."
    if tool_name == "open_spotify":
        return "Ya intente abrir Spotify."
    if tool_name == "open_app":
        return f"Ya intente abrir {data.get('app', 'la aplicacion')}."
    return "Ya lo abri."


def _screen_tool_speech(tool_name: str, data: dict[str, object]) -> str:
    if tool_name == "screen_capture":
        return f"Tome una captura de pantalla de unos {data.get('size_kb', '?')} KB y la tengo como contexto visual."
    if tool_name == "screen_watch_start":
        if data.get("live_mode"):
            return f"Vision en vivo activada; voy a observar cada {data.get('interval_seconds')} segundos."
        return "Ya active la vision de pantalla en segundo plano."
    if tool_name == "screen_watch_stop":
        return "Listo, deje de observar tu pantalla."
    if data.get("watching"):
        return "Ahora mismo tengo la vision de pantalla activa."
    if data.get("has_recent_capture"):
        return "No estoy mirando en continuo, pero tengo una captura reciente."
    return "Ahora mismo no estoy viendo tu pantalla."


def _remove_rejected_side_effects(
    intention: Intention,
    blocking_issues: tuple[str, ...],
) -> Intention | None:
    """Remove only structurally rejected effects; never rewrite conversational content."""

    fields: set[str] = set()
    prefixes = {
        "action_": "actions",
        "memory_": "memory_updates",
        "relationship_": "relationship_updates",
        "self_": "self_updates",
        "curiosity_": "curiosity_updates",
    }
    for issue in blocking_issues:
        field = next(
            (candidate for prefix, candidate in prefixes.items() if issue.startswith(prefix)),
            None,
        )
        if field is None or not getattr(intention, field):
            return None
        fields.add(field)
    if not fields:
        return None

    revised_id = f"{intention.id}:effects_removed"
    metadata = dict(intention.speech_intention.metadata)
    metadata["intention_id"] = revised_id
    changes = {
        field: ()
        for field in fields
    }
    return replace(
        intention,
        id=revised_id,
        speech_intention=replace(intention.speech_intention, metadata=metadata),
        reasoning_summary="Rejected side effects were removed before execution.",
        **changes,
    )


def _compare_intention_structure(
    native: Intention,
    legacy: Intention,
) -> dict[str, object]:
    """Compare decision shape without retaining user-facing speech or private values."""

    changed = []
    if native.intent_type != legacy.intent_type:
        changed.append("intent_type")
    if native.conversation_intent.mode != legacy.conversation_intent.mode:
        changed.append("conversation_mode")
    if tuple(item.action_type for item in native.actions) != tuple(
        item.action_type for item in legacy.actions
    ):
        changed.append("action_types")
    if bool(native.memory_updates) != bool(legacy.memory_updates):
        changed.append("memory_updates")
    if bool(native.relationship_updates) != bool(legacy.relationship_updates):
        changed.append("relationship_updates")
    if bool(native.self_updates) != bool(legacy.self_updates):
        changed.append("self_updates")
    if bool(native.curiosity_updates) != bool(legacy.curiosity_updates):
        changed.append("curiosity_updates")
    return {
        "matches": not changed,
        "changed_sections": tuple(changed),
    }


def _speech_intention(
    intent_type: str,
    speech: str,
    *,
    conversation_intent: ConversationIntent | None = None,
    relationship_updates=(),
    self_updates=(),
    reasoning_summary: str = "Compatibility response adapted into cognitive Intention.",
) -> Intention:
    speech_plan = _speech_plan_from_text(speech)
    return Intention(
        id=f"intention:{intent_type.replace(':', '_')}",
        intent_type=intent_type,
        conversation_intent=conversation_intent or ConversationIntent(),
        speech=speech,
        speech_intention=speech_plan,
        actions=(),
        avatar={
            "expression": "neutral",
            "gesture": "talk" if speech.strip() else "idle",
        },
        memory_updates=(),
        relationship_updates=tuple(relationship_updates),
        self_updates=tuple(self_updates),
        confidence=0.9,
        reasoning_summary=reasoning_summary,
    )


def _speech_plan_from_text(text: str) -> SpeechIntention:
    return SpeechIntention(
        text=text,
        emotion="calm",
        energy=0.38,
        warmth=0.84,
        confidence=0.72,
        thinking=0.12,
        curiosity=0.45,
        pause_before=0.18,
        pause_after=0.18,
        speech_rate=0.9,
        pitch_variation=0.14,
        volume=0.9,
        smile=0.24,
        metadata={"origin": "legacy_compatibility_boundary"},
    )


def _intention_with_contextual_speech(intention: Intention, context) -> Intention:
    text = intention.speech.strip()
    if not text:
        return intention

    uncertainty = context.uncertainty
    emotion = context.emotion
    active_curiosity = [
        item.importance
        for item in context.curiosity
        if not getattr(item, "satisfied", False)
    ]
    curiosity_signal = (
        sum(active_curiosity) / len(active_curiosity)
        if active_curiosity
        else emotion.interest
    )
    confidence = max(
        0.1,
        min(
            0.95,
            1.0
            - uncertainty.reasoning * 0.34
            - uncertainty.knowledge * 0.24
            - uncertainty.user_intent * 0.22
            - uncertainty.memory * 0.20,
        ),
    )
    speech = SpeechIntention(
        text=text,
        emotion=emotion.dominant_label,
        energy=emotion.energy,
        warmth=emotion.affection * 0.62 + emotion.interest * 0.38,
        confidence=confidence,
        thinking=uncertainty.reasoning * 0.62 + uncertainty.user_intent * 0.38,
        curiosity=emotion.interest * 0.58 + curiosity_signal * 0.42,
        pause_before=0.14 + uncertainty.reasoning * 0.18 + uncertainty.user_intent * 0.08,
        pause_after=0.14 + uncertainty.reasoning * 0.07 + uncertainty.knowledge * 0.04,
        speech_rate=0.94 - uncertainty.reasoning * 0.07 - uncertainty.user_intent * 0.04,
        pitch_variation=0.11 + emotion.interest * 0.07 + emotion.energy * 0.025,
        volume=0.82 + emotion.energy * 0.10 + confidence * 0.04,
        smile=emotion.affection * 0.30 + max(0.0, emotion.mood) * 0.18,
        metadata={
            "origin": "internal_context",
            "context_timestamp": context.timestamp,
        },
    )
    return replace(intention, speech=text, speech_intention=speech)


def _internal_state_json(context) -> str:
    if context is None:
        return "No cognitive state snapshot was attached to this legacy boundary."
    payload = {
        "context_timestamp": context.timestamp,
        "emotion": context.emotion.to_dict(),
        "uncertainty": context.uncertainty.to_dict(),
        "curiosity": [item.to_dict() for item in context.curiosity],
        "goals": context.goals.to_dict(),
        "body": context.body.to_dict(),
        "self_preferences": context.self_model.mutable_state.get("self_preferences", {}),
        "self_state_revision": context.self_model.mutable_state.get("self_state_revision", 0),
        "environment": [
            observation.to_dict()
            for observation in context.observations
            if observation.source == "environment"
        ],
        "relationship": (
            context.self_model.mutable_state
            .get("identity_core", {})
            .get("relationship_model", {})
            if isinstance(context.self_model.mutable_state.get("identity_core"), dict)
            else {}
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _relationship_brain_snapshot(context) -> dict[str, object]:
    if context is None:
        return {}
    data = context.relationship.to_dict()
    # Keep prompts bounded while preserving the complete snapshot in InternalContext.
    for collection in (
        "threads",
        "associations",
        "open_loops",
        "promises",
        "shared_experiences",
        "milestones",
    ):
        items = data.get(collection, [])
        if isinstance(items, list) and len(items) > 24:
            data[collection] = items[-24:]
    return data


def _brain_evidence_context(context) -> tuple[dict[str, object], ...]:
    if context is None:
        return ()

    evidence: list[dict[str, object]] = []
    identity_core = context.self_model.mutable_state.get("identity_core", {})
    if isinstance(identity_core, dict):
        for evidence_id, key in (
            ("self:identity", "identity"),
            ("self:personality", "personality"),
            ("self:body_identity", "body_identity"),
            ("self:thinking_style", "thinking_style"),
        ):
            value = identity_core.get(key, {})
            evidence.append(
                {
                    "id": evidence_id,
                    "kind": "self_identity",
                    "summary": json.dumps(value, ensure_ascii=False, sort_keys=True)[:500],
                }
            )
    for item in context.observations:
        evidence.append(
            {
                "id": item.id,
                "kind": "observation",
                "source": item.source,
                "summary": item.summary[:360],
            }
        )
    for tier_name, tier in (
        ("working", context.memory.working),
        ("short", context.memory.short),
        ("long", context.memory.long),
    ):
        for item in tier[-12:]:
            evidence.append(
                {
                    "id": item.id,
                    "kind": "memory",
                    "tier": tier_name,
                    "summary": item.content[:360],
                }
            )
    for item in context.curiosity[-12:]:
        evidence.append(
            {
                "id": item.id,
                "kind": "curiosity",
                "summary": getattr(item, "topic", getattr(item, "prompt", ""))[:360],
            }
        )
    relationship_groups = (
        ("thread", context.relationship.threads, "title"),
        ("open_loop", context.relationship.open_loops, "topic"),
        ("promise", context.relationship.promises, "promise_text"),
        ("shared_experience", context.relationship.shared_experiences, "summary"),
        ("milestone", context.relationship.milestones, "description"),
    )
    for kind, items, summary_field in relationship_groups:
        for item in items[-24:]:
            evidence.append(
                {
                    "id": item.id,
                    "kind": kind,
                    "summary": str(getattr(item, summary_field, ""))[:360],
                }
            )
    for item in context.relationship.associations[-24:]:
        evidence.append(
            {
                "id": item.id,
                "kind": "relationship_association",
                "summary": f"{item.thread_id} -> {item.target_type}:{item.target_id}",
            }
        )
    return tuple(evidence)


def _situational_brief(context) -> str:
    if context is None:
        return "No hay una lectura situacional adjunta."

    emotion = context.emotion
    identity_core = context.self_model.mutable_state.get("identity_core", {})
    relationship = (
        identity_core.get("relationship_model", {})
        if isinstance(identity_core, dict)
        else {}
    )
    environment = next(
        (item.data for item in context.observations if item.source == "environment"),
        {},
    )
    active_goals = [
        goal.description
        for goal in (*context.goals.temporary, *context.goals.permanent)
        if goal.status == "active"
    ]

    lines = [
        (
            f"Estado predominante de Anydes: {emotion.dominant_label}. "
            "Energia, curiosidad y calidez deben modular sutilmente la expresion; no las enumeres ni las recites."
        ),
    ]
    if environment:
        lines.append(
            "Momento y ambiente: "
            f"{environment.get('local_time', context.timestamp)}, periodo {environment.get('time_of_day', 'unknown')}, "
            f"clima {environment.get('weather', 'unknown')}."
        )
    if isinstance(relationship, dict) and relationship:
        lines.append(
            "Relacion con DemoUser: "
            f"{relationship.get('familiarity', 'sin descripcion')}; "
            f"ultimo contacto {relationship.get('last_interaction_at') or 'sin registro previo'}."
        )
        cues = relationship.get("continuity_cues", [])
        if isinstance(cues, list) and cues:
            lines.append("Continuidad reciente sugerida por memoria: " + " | ".join(str(item) for item in cues[-3:]))
    if active_goals:
        lines.append("Objetivos activos relevantes: " + " | ".join(active_goals[:3]))
    lines.append(
        "Usa esta lectura como contexto, no como texto para recitar. Responde solo a lo que DemoUser acaba de decir."
    )
    return "\n".join(lines)


def _tool_intention(tool_name: str, argument: str) -> Intention:
    return Intention(
        id=f"intention:tool:{tool_name or 'unknown'}",
        intent_type="compatibility:tool",
        speech="",
        actions=(
            ActionRequest(
                id=f"action:{tool_name or 'unknown'}",
                action_type=tool_name,
                parameters={"argument": argument},
                requires_confirmation=False,
                risk_level="low",
                reason="Compatibility tool intent adapted into ActionRequest.",
            ),
        ),
        memory_updates=(),
        confidence=0.9,
        reasoning_summary="Compatibility tool routing adapted into cognitive ActionRequest.",
    )


HELP_TEXT = """Comandos principales:
ayuda
herramientas
ejecutar current_time
ejecutar system_info
ejecutar word_count tu texto aqui
ejecutar format_json {"nombre":"Anydes"}
personalidad
memoria
ejecutar memory_permanent_read
ejecutar memory_permanent_write nombre=DemoUser
ejecutar memory_temporary_read
ejecutar memory_temporary_write tema_actual=Fase 1
ejecutar memory_temporary_clear
ejecutar memory_recent
ejecutar screen_capture
ejecutar screen_watch_start
ejecutar screen_watch_stop
ejecutar memory_protected_write token=valor_sensible
salir

Tambien puedes hablar mas natural:
mira mi pantalla
te gusta lo que estoy haciendo?
comparte pantalla
deja de ver mi pantalla
que deberia comprar aqui?
que deberia hacer aqui?
cual es tu memoria?
que recuerdas de mi?
que hablamos recien?
muestrame tu personalidad
que hora es?
recuerda que mi color favorito es azul
por ahora recuerda que estamos probando fase 1
borra tu memoria temporal
"""
