from __future__ import annotations

from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary
from dataclasses import replace

from .controller import ToolConversationController
from .conversational_bridge import resolved_conversation


_INSTALLED = False
_CONTEXT_PATCHED = False
_CONTROLLERS: WeakKeyDictionary[Any, ToolConversationController] = WeakKeyDictionary()


def install(project_root: Path | None = None) -> None:
    """Install the staging-only conversational adapter on AnydesAssistant."""

    global _INSTALLED
    if _INSTALLED:
        return
    from anydes.nucleo.assistant import AnydesAssistant
    from .presentation import install_math_path_preservation

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    from anydes_tools.ui_lifecycle import install as install_lifecycle
    install_lifecycle()
    install_math_path_preservation()
    from .typed_content import install_literal_presentation
    install_literal_presentation()
    _install_capability_context_bridge()
    original_realtime = AnydesAssistant.respond_realtime
    original_stable = AnydesAssistant.respond
    original_close = AnydesAssistant.close

    def respond_realtime(self, text, *, input_mode=None, on_text=None):
        controller = _controller(self, root)
        prior_failure = controller.state.last_tool_failure
        prior_failure_id = prior_failure.failure_id if prior_failure else ""
        response = controller.handle(text, turn_id=_turn_id())
        if response is None:
            _update_capability_context(self, controller)
            guarded_callback = _guarded_callback(self, on_text, text, controller)
            with resolved_conversation(self, controller):
                raw = original_realtime(
                    self,
                    text,
                    input_mode=input_mode,
                    on_text=guarded_callback,
                )
            clean = _compose_final(self, raw, text, controller, prior_failure_id=prior_failure_id)
            if controller.last_resolution is not None and controller.last_resolution.kind.value == "CONTENT_RESPONSE":
                controller.recent_content_response = clean[:12000]
            if guarded_callback is not None:
                guarded_callback.finish(clean)
            return clean
        clean = _compose_final(self, response.text, text, controller, response=response, prior_failure_id=prior_failure_id)
        response = replace(response, text=clean)
        _publish(self, text, response)
        _sync_final_payload(self, clean)
        if on_text is not None:
            on_text(clean)
        return clean

    def respond(self, text, *, interactive=True, input_mode=None):
        controller = _controller(self, root)
        prior_failure = controller.state.last_tool_failure
        prior_failure_id = prior_failure.failure_id if prior_failure else ""
        response = controller.handle(text, turn_id=_turn_id())
        if response is None:
            _update_capability_context(self, controller)
            with resolved_conversation(self, controller):
                raw = original_stable(self, text, interactive=interactive, input_mode=input_mode)
            clean = _compose_final(self, raw, text, controller, prior_failure_id=prior_failure_id)
            if controller.last_resolution is not None and controller.last_resolution.kind.value == "CONTENT_RESPONSE":
                controller.recent_content_response = clean[:12000]
            return clean
        clean = _compose_final(self, response.text, text, controller, response=response, prior_failure_id=prior_failure_id)
        response = replace(response, text=clean)
        _publish(self, text, response)
        _sync_final_payload(self, clean)
        return clean

    def close(self):
        controller = _CONTROLLERS.pop(self, None)
        if controller is not None:
            controller.close()
        return original_close(self)

    from .diagnostics import observed_entry
    AnydesAssistant.respond_realtime = _publication_guard(observed_entry(respond_realtime, root, _turn_id))
    AnydesAssistant.respond = _publication_guard(observed_entry(respond, root, _turn_id))
    AnydesAssistant.close = close
    AnydesAssistant.respond_realtime._anydes_tool_conversation_v001_5 = True
    AnydesAssistant.respond._anydes_tool_conversation_v001_5 = True
    _INSTALLED = True


def controller_for(assistant) -> ToolConversationController | None:
    return _CONTROLLERS.get(assistant)


def ensure_controller_for(assistant, project_root: Path) -> ToolConversationController:
    return _controller(assistant, project_root.resolve())


def _controller(assistant, root: Path) -> ToolConversationController:
    controller = _CONTROLLERS.get(assistant)
    if controller is None:
        from .semantic_runtime import (
            AssistantActionMemoryResponder,
            AssistantCodeGenerator,
            AssistantCodeModifier,
            AssistantGeneralTaskReasoner,
            AssistantInteractionClassifier,
            AssistantSafeContextProvider,
            AssistantSemanticContentGenerator,
            AssistantSemanticLineEvaluator,
            AssistantSemanticProseGenerator,
            AssistantTaskAdherenceEvaluator,
        )

        controller = ToolConversationController(
            root,
            semantic_evaluator=AssistantSemanticLineEvaluator(assistant),
            semantic_content_generator=AssistantSemanticContentGenerator(assistant),
            semantic_prose_generator=AssistantSemanticProseGenerator(assistant),
            task_adherence_evaluator=AssistantTaskAdherenceEvaluator(assistant),
            action_memory_responder=AssistantActionMemoryResponder(assistant),
            interaction_classifier=AssistantInteractionClassifier(assistant),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=AssistantCodeGenerator(assistant),
            code_modifier=AssistantCodeModifier(assistant),
            safe_context_provider=AssistantSafeContextProvider(assistant),
        )
        _CONTROLLERS[assistant] = controller
    return controller


def _install_capability_context_bridge() -> None:
    global _CONTEXT_PATCHED
    if _CONTEXT_PATCHED:
        return
    import anydes.realtime.service as service_module

    session_class = service_module.RealtimeSessionState
    original = session_class.context_note

    def context_note_with_tools(self, act, user_text=""):
        base = original(self, act, user_text)
        tools = str(getattr(self, "tool_capability_note", "") or "").strip()
        return f"{base}\n{tools}".strip() if tools else base

    session_class.context_note = context_note_with_tools
    _CONTEXT_PATCHED = True


def _update_capability_context(assistant, controller: ToolConversationController) -> None:
    service = getattr(assistant, "_realtime_service", None)
    session = getattr(service, "session", None)
    if session is not None:
        session.tool_capability_note = controller.capability_context_note()


def _guarded_callback(assistant, callback, user_text: str, controller):
    if callback is None:
        return None

    class BufferedPublisher:
        def __init__(self) -> None:
            self.latest = ""

        def __call__(self, partial: str) -> None:
            self.latest = str(partial or "")

        def finish(self, text: str) -> None:
            if text:
                callback(text)

    return BufferedPublisher()


def _sanitize_model_response(assistant, text: str, user_text: str, controller) -> str:
    """Deprecated compatibility entry; no independent gate."""
    return _compose_final(assistant, text, user_text, controller)


def _sync_final_payload(assistant, text):
    intention = getattr(assistant, "last_intention", None)
    if intention is None:
        from anydes.cognition import Intention, ConversationIntent, SpeechIntention
        intention = Intention(id="intention:publication", intent_type="publication",
            conversation_intent=ConversationIntent(mode="respond"),
            speech_intention=SpeechIntention.from_text(text))
        assistant.last_intention = intention
    speech = getattr(intention, "speech_intention", None)
    if intention is not None and speech is not None:
        assistant.last_intention = replace(
            intention,
            speech=text,
            speech_intention=replace(speech, text=text),
        )
    realtime = getattr(assistant, "last_realtime_result", None)
    if realtime is not None:
        assistant.last_realtime_result = replace(realtime, text=text)


def _publication_guard(method):
    """Unexpected runtime errors get safe publication; cancellation still propagates."""
    from functools import wraps

    @wraps(method)
    def guarded(assistant, text, **kwargs):
        before = getattr(assistant, "last_final_response", None)
        try:
            return method(assistant, text, **kwargs)
        except Exception as exc:
            from .publication_boundary import PublicationFailure
            if isinstance(exc, PublicationFailure):
                raise  # Recovery was already bounded; no second publication attempt.
            from anydes.realtime.cancellation import ResponseSuperseded
            if isinstance(exc, ResponseSuperseded):
                raise
            final = getattr(assistant, "last_final_response", None)
            if final is not None and final is not before:
                return final.visible_text  # Never publish a second outcome after a callback failure.
            from .publication import RuntimeOutcome, OutcomeStatus, outcome_from_controller
            controller = _CONTROLLERS.get(assistant)
            if controller is not None:
                controller._internal_pipeline_failure(_turn_id(), exc=exc)
            observed = outcome_from_controller(controller) if controller else RuntimeOutcome()
            outcome = replace(observed,
                status=OutcomeStatus.PARTIAL_SUCCESS if observed.committed_artifacts else OutcomeStatus.FAILED,
                safe_evidence={**observed.safe_evidence, "failure_message":
                    observed.safe_evidence.get("failure_message") or "No pude terminar esta respuesta por un fallo interno registrado."})
            clean = _compose_final(assistant, "", text, controller, outcome_override=outcome)
            callback = kwargs.get("on_text")
            if callback is not None:
                callback(clean)
            return clean
    return guarded


def _compose_final(assistant, text, user_text, controller, **kwargs):
    from .diagnostics import current_or_new
    diagnostic = current_or_new(assistant, getattr(controller, 'project_root', Path(__file__).resolve().parents[2]),
        getattr(getattr(controller, '_pipeline_trace', None), 'turn_id', ''))
    with diagnostic.stage('PUBLICATION'):
        return _compose_final_observed(assistant, text, user_text, controller, **kwargs)


def _compose_final_observed(assistant, text, user_text, controller, *, response=None, prior_failure_id=None, outcome_override=None):
    from .publication import FinalResponseComposer, outcome_from_controller, OutcomeStatus
    from .intent_normalization import asks_for_full_path
    outcome = outcome_override or (controller.last_runtime_outcome if response is not None else None) or outcome_from_controller(controller, response, user_goal=user_text, prior_failure_id=prior_failure_id)
    realtime = getattr(assistant, 'last_realtime_result', None)
    failure_code = getattr(getattr(realtime, 'metrics', None), 'failure_code', '')
    if (response is None and outcome_override is None
            and outcome.status == OutcomeStatus.CONVERSATIONAL
            and failure_code in {'natural_response_local_invariant_rejected',
                'natural_response_grounding_rejected','natural_response_grounding_inconclusive',
                'natural_response_review_unavailable'}):
        outcome = replace(outcome, status=OutcomeStatus.FAILED,
            safe_evidence={**outcome.safe_evidence,
                'reason_code':failure_code, 'failure_stage':'LOCAL_RESPONSE_VALIDATION',
                'failure_message':'La respuesta candidata fue rechazada antes de publicarse.'})
    if controller is not None:
        controller.audit.record("ui_bridge_outcome", outcome_id=outcome.outcome_id,
            controller_outcome_id=controller.last_runtime_outcome.outcome_id if controller.last_runtime_outcome else "",
            status=outcome.status.value, task_id=outcome.task_id, failure_ref=outcome.failure_ref)
        controller._pipeline_trace.record("PUBLICATION", entered=True, status="ENTERED")
    final = FinalResponseComposer.compose(outcome, conversational_text=text,
        reveal_paths=asks_for_full_path(user_text))
    from .outcome_voice import naturalize_final
    from .publication_boundary import PublicationFailure, require_visible, recover_publication
    try:
        final = require_visible(naturalize_final(assistant,controller,outcome,final,user_text))
    except PublicationFailure as failure:
        final = recover_publication(assistant,controller,outcome,final,user_text,failure)
    from .typed_content import approve_literal_publication, ContentKind
    approve_literal_publication(final)
    for segment in final.content_segments:
        if segment.kind == ContentKind.CODE:
            import hashlib
            assistant.audit.record('literal_code_publication',outcome_id=outcome.outcome_id,
                characters=len(segment.text),sha256=hashlib.sha256(segment.text.encode()).hexdigest(),
                language=segment.language,math_transform_allowed=False)
    if controller is not None:
        controller.discourse.record(user_text, final.visible_text)
        offer = getattr(controller, '_pending_offer_candidate', None)
        if offer and final.visible_text.endswith(offer['question']):
            controller.discourse.pending_offer = offer
            controller.state.previous_assistant_act = 'SUPPORTED_ACTION_OFFER'
        controller._pending_offer_candidate = None
    assistant.last_runtime_outcome = outcome
    assistant.last_final_response = final
    require_visible(final)
    _sync_final_payload(assistant, final.spoken_text)
    import hashlib
    fingerprint = {"characters": len(final.visible_text),
        "sha256": hashlib.sha256(final.visible_text.encode("utf-8")).hexdigest()}
    assistant.audit.record("final_publication", outcome_id=outcome.outcome_id,
        interaction_kind=outcome.interaction_kind, task_id=outcome.task_id,
        outcome_status=outcome.status.value, publication_status=final.decision.status.value,
        allowed_claims=list(final.decision.allowed_claims),
        artifact_ids=list(final.decision.artifact_ids), completed_steps=list(outcome.completed_steps),
        failed_step=outcome.failed_step, blocked_steps=list(outcome.blocked_steps),
        failure_ref=outcome.failure_ref, attempt_number=outcome.attempt_number,
        verification_statuses=[v.get("status") for v in outcome.verification_summary],
        visible_text=fingerprint, spoken_text=fingerprint,
        sanitized=final.sanitized, chat_speech_equal=final.visible_text == final.spoken_text)
    if controller is not None:
        controller._pipeline_trace.record("PUBLICATION", exited=True, status=final.decision.status.value,
            reason_code=final.decision.reason)
    return final.visible_text


def _finalize_response(
    assistant,
    text: str,
    *,
    verified: bool,
    user_text: str,
    controller,
) -> str:
    """Deprecated: the boolean can no longer authorize operational claims."""
    return _compose_final(assistant, text, user_text, controller)


def _turn_id() -> str:
    try:
        from anydes.interfaz.turn_interaction import current_turn

        turn = current_turn()
        if turn is not None:
            return str(turn.turn_id)
    except Exception:
        pass
    from uuid import uuid4

    return f"tool:{uuid4().hex}"


def _publish(assistant, user_text: str, response) -> None:
    from anydes.cognition import ActionResult, ConversationIntent, Intention, SpeechIntention

    controller = _CONTROLLERS[assistant]
    action_results = tuple(
        ActionResult(
            request_id=item.request_id,
            status="executed" if item.status.value == "SUCCESS" else "failed",
            message="tool_result_available" if item.status.value == "SUCCESS" else "tool_not_executed",
            output={
                "tool_id": item.tool_id,
                "capability": item.capability,
                "data": item.output,
            },
            error=item.error or None,
        )
        for item in controller.last_results
    )
    assistant.last_action_results = action_results
    assistant.last_intention = Intention(
        id=f"intention:tool_conversation:{_turn_id()}",
        intent_type="tool_conversation",
        conversation_intent=ConversationIntent(mode="respond"),
        speech_intention=SpeechIntention.from_text(response.text),
        avatar={"expression": "calm", "gesture": "speaking", "gaze": "user"},
        confidence=0.95,
        reasoning_summary="Staging publication derived from runtime evidence; execution authority remained local.",
    )
    assistant.audit.record(
        "tool_conversation_response",
        handled=True,
        resolver_kind=response.resolver_kind.value if response.resolver_kind else "pending_or_local",
        executed=response.executed,
        selected_tools=list(response.selected_tools),
    )
    service = getattr(assistant, "_realtime_service", None)
    continuity = getattr(getattr(service, "session", None), "continuity_v2", None)
    if continuity is not None:
        continuity.observe_turn(user_text, response.text)
