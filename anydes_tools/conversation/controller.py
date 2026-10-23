from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import re
import threading
import time
import hashlib
import json
import zipfile
from typing import Any
from uuid import uuid4

from ..audit import ToolAuditLog
from ..composition import _resolve_references
from ..contracts import (
    CapabilityIntent,
    CapabilityResolution,
    CapabilityResolutionKind,
    ExecutionStatus,
    PermissionEffect,
    ToolExecutionResult,
    ToolRequest,
)
from ..executor import ToolExecutor
from ..path_policy import FilesystemBoundary
from ..permissions import PermissionBridge
from ..registry import build_official_registry
from ..resolver import CapabilityResolver
from ..settings import CapabilityAwareness, SettingsSnapshot, SettingsStore, settings_for
from .contracts import (
    ConversationToolResponse,
    ConversationToolState,
    PendingToolAction,
    PlannedToolAction,
    StructuredActionPlan,
    UserTaskSpec,
)
from .action_memory import ActionExperienceMemory
from .failure_state import (
    AttemptOutcome,
    FailureResponseRenderer,
    RemediationAssessment,
    LastToolFailure,
    VerificationEvidence,
    artifact_type_for,
    plan_reference,
    is_failure_explanation_request,
)
from .document_transform import (
    DocumentTransformationSpec,
    apply_patch_plan,
    build_semantic_patch_plan,
    verify_patch_plan,
)
from .content_generation import (
    ContentValidator,
    ContentGenerationSpec,
    PostActionVerifier,
)
from .delete_plan import (
    BatchActionResult,
    CollectionResolver,
    DeleteArtifactKind,
    DeletePlan,
    DeleteScope,
    DeleteSelectionRequest,
    target_matches_snapshot,
)
from .document_conversion import DocumentConversionSpec, verify_conversion
from .document_model import (
    DocumentEditSpec,
    DocumentModel,
    apply_document_edit,
    verify_document_edit,
)
from .document_spec import (
    DocumentRenderer,
    build_document_spec,
    document_spec_from_extracted_text,
)
from .planner import (
    plan_from_text,
    validate_structured_action_plan,
)
from .reflection import ConversationToolReflection
from .response_gate import gate_action_response
from .intent_normalization import (
    ConfirmationDecision,
    classify_confirmation,
    sanitize_windows_stem,
)
from .structured_text import (
    ContentBuilder,
    TextTransformSpec,
    transform_text,
    verify_creation,
    verify_transform,
)
from .code_generation import (
    CodeGenerationSpec,
    CodeModificationSpec,
    CodeValidator,
    GeneratedCodeArtifact,
    verify_code_test,
)
from .typed_verification import VerificationDispatcher
from .oracle_registry import OracleRegistry, OracleSelector
from .verification_types import VerificationStatus, TestSpec
from .grounding_evidence import verify_grounding
from .task_adherence import evaluate_adherence, aggregate_adherence
from .operation_semantics import OperationSemantics
_LEGACY_CODE_TEST_GATE = verify_code_test  # Compatibility for the existing isolated C test hook.
from .general_reasoning import (
    GeneralTaskPlan,
    GeneralTaskPlanValidator,
    compact_capability_vocabulary,
)
from .interaction_router import (
    GoalConsistencyValidator,
    InteractionResolver,
    InteractionKind,
    TaskControl,
    EXPLANATION_CONTROLS, FAILURE_RELATIONS, TurnRelation, ResolutionContext,
)
from .executable_plan import ExecutablePlanAdapter, ExecutableTaskPlan, StepResult, BrainPlanProposal
from .dataflow import ArtifactBindings, ArtifactState, DataflowError, file_hash, validate_dataflow
from .pipeline_trace import PipelineTrace, traced_stage, exception_code, exception_metadata


_CONFIRM_EFFECTS = frozenset({
    PermissionEffect.OVERWRITE,
    PermissionEffect.MOVE,
    PermissionEffect.DELETE,
    PermissionEffect.EXECUTE_CODE,
    PermissionEffect.INSTALL_DEPENDENCY,
})


class ToolConversationController:
    """Session-scoped conversational authority for the approved v001 tools."""

    def __init__(
        self,
        project_root: Path,
        tools_root: Path | None = None,
        *,
        settings: SettingsStore | None = None,
        semantic_evaluator=None,
        semantic_content_generator=None,
        semantic_prose_generator=None,
        action_memory_responder=None,
        interaction_classifier=None,
        general_task_reasoner=None,
        code_generator=None,
        code_modifier=None,
        safe_context_provider=None,
        task_adherence_evaluator=None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.tools_root = (tools_root or self.project_root / "anydes_tools").resolve()
        self.settings = settings or settings_for(self.project_root, self.tools_root)
        self.registry = build_official_registry(self.tools_root)
        self.boundary = FilesystemBoundary(self.project_root, self.tools_root)
        self.audit = ToolAuditLog(self.tools_root / "logs" / "conversation-v001_5.jsonl")
        self.executor = ToolExecutor(
            self.project_root,
            self.tools_root,
            self.registry,
            PermissionBridge(),
            self.audit,
        )
        self.resolver = CapabilityResolver(self.registry, self.boundary)
        self.general_plan_validator = GeneralTaskPlanValidator(
            self.registry,
            self.resolver,
            self.boundary,
            max_steps=8,
        )
        self.reflection = ConversationToolReflection()
        self.semantic_evaluator = semantic_evaluator
        self.content_builder = ContentBuilder(semantic_content_generator, semantic_evaluator)
        self.semantic_prose_generator = semantic_prose_generator
        self.task_adherence_evaluator = task_adherence_evaluator
        self.action_memory_responder = action_memory_responder
        self.interaction_resolver = InteractionResolver(interaction_classifier)
        self._operational_artifacts = {}
        self.plan_adapter = ExecutablePlanAdapter(self.registry, path_authorizer=lambda path:
            self.boundary.classify(path, operation="READ", authorized_scopes=self.state.authorized_scopes).allowed,
            artifact_lookup=lambda path: self.state.recent_artifact_state.ref_by_path(path)
                or self._operational_artifacts.get(str(Path(path).resolve(strict=False))),
            requested_effects_provider=lambda: self.last_resolution.requested_effects if self.last_resolution else ())
        self.last_resolution = None
        from .discourse_context import DiscourseContext
        self.discourse = DiscourseContext()
        self.last_step_results: list[StepResult] = []
        self._plan_source = "LEGACY_ADAPTER"
        self._executing_plan = None
        self._artifact_bindings = None
        self._dataflow_step = None
        self.last_dataflow_summary = {}
        self._archive_transaction = False
        self.recent_content_response = ""
        self.general_task_reasoner = general_task_reasoner
        self.code_generator = code_generator
        self.code_modifier = code_modifier
        self.code_oracle_provider = None  # Trusted host extension; never populated from a brain proposal.
        self.oracle_registry = OracleRegistry()
        self.oracle_selector = OracleSelector(self.oracle_registry)
        self.safe_context_provider = safe_context_provider
        settings_snapshot = self.settings.snapshot()
        self.state = ConversationToolState(
            active_workspace=settings_snapshot.active_workspace,
            authorized_scopes=settings_snapshot.authorized_external_scopes,
        )
        official_settings = (self.tools_root / "config" / "settings-v001_5.json").resolve()
        memory_root = (
            self.tools_root
            if self.settings.settings_path == official_settings
            else self.settings.settings_path.parent / ".action-memory-test"
        )
        self.action_memory = ActionExperienceMemory(
            memory_root,
            recent_state=self.state.recent_action_state,
        )
        self.awareness = CapabilityAwareness(self.registry, self.settings)
        self.last_results: tuple[ToolExecutionResult, ...] = ()
        self.last_runtime_outcome = None
        self._retained_draft = None
        self._turn_reference = None
        self._adherence_results = {}
        self._pipeline_trace = PipelineTrace(self.audit, "integration",
            lambda: (self.state.pending_task.task_id, self.state.pending_task.plan_revision))
        self.oracle_selector._pipeline_trace = self._pipeline_trace
        self._lock = threading.RLock()
        self.settings.subscribe(self._apply_settings_snapshot)

    def handle(self, text: str, *, turn_id: str) -> ConversationToolResponse | None:
        with self._lock:
            self._reasoner_called = False
            self._executor_called = False
            self._plan_source = "NONE"
            self.last_resolution = None
            self.last_runtime_outcome = None
            # Failure availability is evidence, not ownership of a new turn.
            self._explanation_root = None
            self._pipeline_trace = PipelineTrace(self.audit, turn_id,
                lambda: (self.state.pending_task.task_id, self.state.pending_task.plan_revision))
            self.oracle_selector._pipeline_trace = self._pipeline_trace
            try:
                try:
                    response = self._handle_resolved_turn(text, turn_id=turn_id)
                    if response is None and self.last_resolution and self.last_resolution.kind == InteractionKind.TOOL_ACTION:
                        response = self._internal_pipeline_failure(turn_id,
                            reason_code="TOOL_ACTION_WITHOUT_TERMINAL_RESPONSE")
                except Exception as exc:
                    # Cancellation is not a task failure and must retain its existing UX.
                    try:
                        from anydes.realtime.cancellation import ResponseSuperseded
                    except ImportError:
                        ResponseSuperseded = ()
                    if isinstance(exc, ResponseSuperseded):
                        raise
                    response = self._internal_pipeline_failure(turn_id, exc=exc)
                from .publication import outcome_from_controller, OutcomeStatus
                outcome = outcome_from_controller(self, response, user_goal=text)
                if (self.last_resolution and self.last_resolution.source == 'LOCAL_FAIL_CLOSED'
                        and self._explanation_root is None and self.state.failure_topic_record()
                        and self.last_resolution.kind == InteractionKind.CLARIFICATION):
                    # Ambiguity is not a new action failure or an explanation of
                    # the old one. Keep history while asking about this request.
                    outcome = replace(outcome, status=OutcomeStatus.CLARIFICATION,
                        failure_ref='', clarification=response.text if response else
                        'Necesito un poco mas de contexto sobre tu pregunta actual.', safe_evidence={})
                if outcome.status == OutcomeStatus.FAILED and not outcome.failure_ref:
                    reason = (response.debug.get("reason_code") or response.debug.get("reason")
                              or response.debug.get("response_source") or "ACTION_NOT_EXECUTED") if response else "ACTION_NOT_EXECUTED"
                    reason = str(reason)
                    response = self._internal_pipeline_failure(turn_id,
                        reason_code=reason if re.fullmatch(r"[A-Z0-9_:.]{1,160}", reason) else "ACTION_NOT_EXECUTED")
                    outcome = outcome_from_controller(self, response, user_goal=text)
                self.last_runtime_outcome = outcome
                if not outcome.safe_evidence.get("reason_code"):
                    outcome = replace(outcome, safe_evidence={**outcome.safe_evidence,
                        "reason_code": {"SUCCESS":"EVIDENCE_CONFIRMED", "PARTIAL_SUCCESS":"PARTIAL_EFFECTS_CONFIRMED",
                            "PENDING_CONFIRMATION":"USER_APPROVAL_REQUIRED", "CLARIFICATION":"REFERENCE_OR_INTENT_UNRESOLVED",
                            "UNSUPPORTED":"CAPABILITY_UNAVAILABLE", "INCONCLUSIVE":"VERIFICATION_EVIDENCE_INSUFFICIENT",
                            "CONVERSATIONAL":"CONVERSATIONAL_DELEGATION"}.get(outcome.status.value, "ACTION_FAILED")})
                    self.last_runtime_outcome = outcome
                if outcome.status == OutcomeStatus.PENDING_CONFIRMATION:
                    self._pipeline_trace.record("PENDING_CONFIRMATION", entered=True, exited=True,
                        status="PENDING_CONFIRMATION", reason_code="USER_APPROVAL_REQUIRED", next_stage="EXECUTION_DISPATCH")
                self._update_resolution_context(outcome, turn_id)
                self._pipeline_trace.terminal(outcome)
                return response
            finally:
                pending = self.state.pending_task
                resolution = self.last_resolution
                self.audit.record("phase_c_turn", turn_id=turn_id,
                    interaction_kind=resolution.kind.value if resolution else "UNRESOLVED",
                    task_control=resolution.task_control.value if resolution else "NONE",
                    plan_source=self._plan_source, task_id=pending.task_id,
                    plan_revision=pending.plan_revision, pending_task=pending.status,
                    attempt_number=pending.attempt_number,
                    failure_id=pending.failure_ref.failure_id if pending.failure_ref else "",
                    reasoner_called=self._reasoner_called, executor_called=self._executor_called)

    def _context_snapshot(self):
        state = self.state
        pending = state.pending_task
        failure = state.failure_topic_record()
        topic = state.active_failure_topic
        index = state.recent_artifact_state
        focus = tuple({'artifact_id':item.artifact_id, 'artifact_type':item.runtime_ref.artifact_type,
            'version':item.runtime_ref.version, 'state':item.runtime_ref.state.value,
            'display_name':item.filename, 'task_id':item.task_id}
            for item in index.artifacts if item.artifact_id in index.focus_artifact_ids
            and item.runtime_ref and index._selectable(item))
        recent = state.recent_action_state.events[-1] if state.recent_action_state.events else None
        approval = state.pending_tool_action
        plan = pending.plan
        confirmation = {'available':approval is not None,
            'task_id':pending.task_id,'status':pending.status}
        if approval is not None:
            confirmation.update(plan_id=approval.plan_id, pending_request_id=approval.pending_request_id,
                plan_revision=pending.plan_revision, confirmation_type=approval.reason,
                valid_until=approval.expires_at,
                requested_action={'intent':approval.action.intent,'operation':approval.action.operation},
                artifact_names=[Path(approval.action.target_path).name] if approval.action.target_path else [],
                permission_effects=[e.value for e in approval.confirmed_effects],
                risk_levels=sorted({s.risk_class for s in plan.steps}) if plan else [],
                steps=[{'step_id':s.step_id,'produces':s.produces_artifacts,'consumes':s.consumes_artifacts}
                    for s in plan.steps] if plan else [],
                verification_requirement=approval.action.arguments.get('code_generation_spec',{}).get('verification_requirement',''))
        return ResolutionContext(
            pending_confirmation=confirmation,
            active_task={'task_id':pending.task_id, 'status':pending.status,
                'plan_revision':pending.plan_revision},
            active_failure_topic={'failure_id':failure.failure_id, 'task_id':failure.task_id,
                'root_failure_id':topic.root_failure_id, 'stage':failure.stage,
                'reason_code':failure.error_code, 'explanation_depth':topic.explanation_depth,
                'retryable':failure.retryable} if failure else {},
            conversation_artifact_focus=focus,
            recent_action={'task_id':recent.task_id, 'status':recent.result_status,
                'artifact_ids':recent.artifact_ids} if recent else {},
            previous_assistant_act=state.previous_assistant_act,
            discourse=self.discourse.snapshot())

    def _update_resolution_context(self, outcome, turn_id):
        decision = self.last_resolution
        relation = decision.turn_relation if decision else TurnRelation.UNSPECIFIED
        if decision and decision.source != 'LOCAL_FAIL_CLOSED':
            self.discourse.resolve(decision)
        topic = self.state.active_failure_topic
        # Only a resolved discourse transition can dismiss an old topic. A new
        # failure created by this action already has its own root and stays active.
        from .discourse_context import ConversationPurpose
        distinct_domain = bool(decision and decision.source != 'LOCAL_FAIL_CLOSED'
            and decision.kind in {InteractionKind.CONVERSATION, InteractionKind.CONTENT_RESPONSE}
            and decision.task_control == TaskControl.NONE
            and decision.conversation_purpose in {ConversationPurpose.SELF_PREFERENCE,
                ConversationPurpose.SELF_RELATIONAL, ConversationPurpose.PERSONAL_IDENTITY,
                ConversationPurpose.PERSONAL_MEMORY, ConversationPurpose.ENTITY_QUERY})
        if topic and (distinct_domain or relation in {TurnRelation.NEW_TOPIC, TurnRelation.TOPIC_CLOSURE} or (
                relation in {TurnRelation.NEW_ACTION, TurnRelation.NEW_CORRECTIVE_ACTION, TurnRelation.ARTIFACT_FOLLOWUP}
                and not outcome.failure_ref)):
            self.state.failure_topic_active = False
            topic.topic_status = 'DISMISSED'
            topic.superseded_by_user_action = turn_id
        control = decision.task_control if decision else TaskControl.NONE
        status = outcome.status.value
        if control == TaskControl.REMEDIATION_QUERY:
            act = 'FAILURE_REMEDIATION_RESPONSE'
        elif control in EXPLANATION_CONTROLS:
            act = 'FAILURE_EXPLANATION'
        elif outcome.safe_evidence.get('reason_code') and getattr(self, '_explanation_root', None) and status == 'FAILED':
            act = 'FAILURE_EXPLANATION_UNAVAILABLE'
        else:
            act = {'PENDING_CONFIRMATION':'CONFIRMATION_REQUEST', 'FAILED':'ACTION_FAILED',
                'INCONCLUSIVE':'ACTION_INCONCLUSIVE', 'SUCCESS':'ACTION_COMPLETED',
                'PARTIAL_SUCCESS':'ACTION_PARTIAL', 'CLARIFICATION':'CLARIFICATION_REQUEST',
                'UNSUPPORTED':'UNSUPPORTED', 'CONVERSATIONAL':'CONTENT_RESPONSE'}.get(status,'NONE')
        self.state.previous_assistant_act = act
        self.audit.record('resolution_context_after', turn_id=turn_id,
            active_context_after=self._context_snapshot().to_dict(), assistant_act=act,
            final_task_control=control.value, turn_relation=relation.value,
            failure_ref=outcome.failure_ref)

    def _internal_pipeline_failure(self, turn_id, *, exc=None, reason_code=""):
        root = getattr(self, '_explanation_root', None)
        if root is not None:
            reason = reason_code or (exception_code(exc) if exc else 'EXPLANATION_INTERNAL_FAILURE')
            secondary = LastToolFailure.create(task_id=root.task_id, stage='FAILURE_EXPLANATION',
                error_code=reason, safe_user_message='No pude ampliar la explicacion en este turno.',
                retryable=False, caused_during_explanation_of=root.failure_id)
            self.state.record_explanation_error(secondary)
            self.audit.record('explanation_secondary_failure', turn_id=turn_id,
                failure_record=asdict(secondary), root_failure_id=root.failure_id)
            return ConversationToolResponse(True,
                'No pude ampliar la explicacion en este turno. El fallo que estabamos revisando sigue siendo el de la tarea anterior.',
                executed=False, debug={'response_source':'EXPLANATION_SECONDARY_FAILURE',
                    'reason_code':reason, 'failure_ref':root.failure_id, 'task_id':root.task_id})
        pending = self.state.pending_task
        validation_detail = {}
        if reason_code == 'INTERACTION_CLARIFICATION':
            from .contracts import PendingTaskState
            trials = getattr(self.interaction_resolver,'validation_history',())
            errors = [e for trial in trials for e in trial.get('errors',())]
            validation_detail = {'validator':'InteractionResolver', 'validation_stage':'INTERACTION_RESOLUTION',
                'failed_constraint':errors[0] if errors else '', 'errors':list(dict.fromkeys(errors)),
                'repair_attempted':len(trials)>1, 'repair_result':'REJECTED'}
            self.state.pending_task = pending = PendingTaskState(task_id='task:'+uuid4().hex)
        if not pending.task_id:
            pending.task_id = "task:" + uuid4().hex
        stage = self._pipeline_trace.failed_stage if self._pipeline_trace else ""
        stage = ('INTERACTION_RESOLUTION' if validation_detail else stage) or (self._pipeline_trace.current[-1] if self._pipeline_trace and self._pipeline_trace.current else "PIPELINE_CONTROL")
        reason = reason_code or (exception_code(exc) if exc else "INTERNAL_PIPELINE_FAILURE")
        # No message/locals/source lines from unexpected exceptions enter the logs.
        if exc:
            self.audit.record("pipeline_exception", stage=stage, task_id=pending.task_id,
                reason_code=reason, **exception_metadata(exc))
        message = ('No pude interpretar esa solicitud de forma coherente; no hice cambios.' if validation_detail else
            "No pude continuar porque una etapa interna del plan no quedó válida.")
        self.state.last_tool_failure = LastToolFailure.create(task_id=pending.task_id,
            plan_revision=pending.plan_revision, attempt_number=pending.attempt_number,
            stage=stage, error_code=reason, technical_message=reason,
            safe_user_message=message, retryable=False, original_plan_ref=plan_reference(pending.plan),
            validation_detail=validation_detail)
        pending.status = "FAILED"
        pending.required_confirmation = None
        self.audit.record("conversation_tool_failure_state", turn_id=turn_id,
            failure_record=asdict(self.state.last_tool_failure), stage=stage, error_code=reason,
            task_id=pending.task_id)
        self._record_attempt_outcome("FAILED", turn_id=turn_id, failure=self.state.last_tool_failure)
        return ConversationToolResponse(True, message, executed=False,
            debug={"response_source":"INTERNAL_PIPELINE_FAILURE", "reason_code":reason, "stage":stage})

    def _handle_resolved_turn(self, text: str, *, turn_id: str) -> ConversationToolResponse | None:
        with self._lock:
            self.last_results = ()
            self.last_step_results = []
            pending = self.state.pending_tool_action
            topic_failure = self.state.failure_topic_record()
            self._resolution_context_before = self._context_snapshot()
            self.audit.record('resolution_context_before', turn_id=turn_id,
                active_context_before=self._resolution_context_before.to_dict())
            decision = self._pipeline_trace.invoke("INTERACTION_RESOLUTION", self.interaction_resolver.resolve, text, state={
                "pending_action": pending is not None,
                "resolution_context": self._resolution_context_before.to_dict(),
                "recent_dialogue": self.discourse.turns[-6:],
                "pending_status": self.state.pending_task.status,
                "failure_available": topic_failure is not None,
                "active_failure": {"relevant":self.state.failure_topic_active,
                    "summary":topic_failure.safe_user_message if topic_failure else '',
                    "failure_id":topic_failure.failure_id if topic_failure else '',
                    "reason_code":topic_failure.error_code if topic_failure else '',
                    "validation_detail":topic_failure.validation_detail if topic_failure else {},
                    "explanation_depth":self.state.failure_explanation_depth,
                    "explained":self.state.failure_explanation_depth > 0},
                "retention_available": self._retained_draft is not None,
                "unpublished_draft": {"task_id":self._retained_draft['task_id'],
                    "display_name":Path(self._retained_draft['destination']).name,
                    "artifact_type":"CODE", "verification_status":self._retained_draft['verification']['status'],
                    "state":"PRIVATE_UNPUBLISHED", "available_control":"RETAIN_UNVERIFIED"
                    } if self._retained_draft else None,
                "retry_available": self.state.last_failed_action is not None,
                "destructive_confirmation": bool(pending and PermissionEffect.DELETE in pending.confirmed_effects),
                "active_artifact": bool(self.state.active_document or self.state.active_code_file),
                "recent_artifacts": [Path(p).name for p in self.state.recent_files[-8:]],
                "recent_content_available": bool(self.recent_content_response),
                "capabilities": list(self.registry.list_capabilities(enabled_only=True)),
            })
            self.last_resolution = decision
            if (decision.task_control in EXPLANATION_CONTROLS or any(
                    trial.get('turn_relation') in {r.value for r in FAILURE_RELATIONS}
                    and trial.get('kind') == 'CONTENT_RESPONSE'
                    for trial in getattr(self.interaction_resolver, 'validation_history', ()))):
                self._explanation_root = topic_failure
            self.audit.record('resolution_control_validation', turn_id=turn_id,
                classifier_results=getattr(self.interaction_resolver, 'validation_history', []),
                repair_used=len(getattr(self.interaction_resolver,'validation_history',[])) > 1,
                final_task_control=decision.task_control.value, failure_ref=decision.failure_ref,
                turn_relation=decision.turn_relation.value)
            if (decision.source != 'LOCAL_FAIL_CLOSED' and decision.kind != InteractionKind.CLARIFICATION
                    and decision.task_control not in EXPLANATION_CONTROLS):
                self._explanation_root = None
            self.audit.record("interaction_resolution", turn_id=turn_id,
                interaction_kind=decision.kind.value, task_control=decision.task_control.value,
                source=decision.source, confidence=decision.confidence,
                operation_intent=decision.operation_intent.value,
                effect_intent=decision.effect_intent.to_dict(),
                task_id=self.state.pending_task.task_id,
                plan_revision=self.state.pending_task.plan_revision,
                pending_task=self.state.pending_task.status,
                reasoner_called=False, executor_called=False)
            if decision.kind == InteractionKind.CLARIFICATION:
                if decision.source == 'IDENTITY_ATTRIBUTE_UNRESOLVED':
                    return ConversationToolResponse(True,
                        'No pude determinar que dato de tu identidad estabas preguntando. Puedes precisar cual?',
                        executed=False, debug={'response_source':'IDENTITY_ATTRIBUTE_CLARIFICATION'})
                if (any('explain_without_failure' in trial.get('errors', ())
                        for trial in getattr(self.interaction_resolver, 'validation_history', ()))
                        and not self.state.failure_topic_record()):
                    return ConversationToolResponse(True,
                        'No tengo un fallo identificado para explicar. A que tarea te refieres?',
                        executed=False, debug={'response_source':'FAILURE_REFERENCE_CLARIFICATION'})
                return ConversationToolResponse(True,
                    "Necesito que aclares qué quieres hacer antes de continuar.", executed=False,
                    debug={"response_source": "INTERACTION_CLARIFICATION"})
            if decision.kind == InteractionKind.UNSUPPORTED:
                return ConversationToolResponse(True,
                    "No puedo resolver esa solicitud con las capacidades disponibles.", executed=False)
            if decision.turn_relation == TurnRelation.OFFER_ACCEPTANCE:
                offer = self.discourse.pending_offer
                if not offer:
                    raise ValueError('accepted_offer_not_available')
                text = offer['request']
                self.discourse.pending_offer = None
                self.audit.record('assistant_offer_accepted', turn_id=turn_id,
                    offer_id=offer['offer_id'], operation_kind=offer['operation_kind'],
                    permission_granted=False)
            if decision.task_control in EXPLANATION_CONTROLS:
                return self._handle_failure_explanation(text, turn_id, force=True)
            if decision.task_control == TaskControl.RETAIN_UNVERIFIED:
                return self._retain_unverified(turn_id, text)
            if decision.task_control == TaskControl.RETRY:
                return self._handle_retry(text, turn_id, force=True)
            if decision.task_control == TaskControl.CANCEL:
                had_pending = bool(self.state.pending_tool_action or self.discourse.pending_offer
                    or self.state.pending_task.status in {'PENDING', 'PENDING_CONFIRMATION', 'FAILED'})
                self.state.pending_tool_action = None
                self.discourse.pending_offer = None
                self._finish_task('CANCELLED', clear_action=True)
                self.audit.record('semantic_cancel', turn_id=turn_id, had_pending=had_pending,
                    executor_called=False, history_preserved=True)
                return ConversationToolResponse(True,
                    'Entendido, no lo hago.' if had_pending else 'No habia ninguna accion pendiente; no ejecutare nada.',
                    executed=False, debug={'response_source':'CANCELLED'})
            if decision.task_control in {TaskControl.CONFIRM, TaskControl.CANCEL}:
                current = self.state.pending_task.plan
                self._plan_source = current.plan_source if current else "LEGACY_ADAPTER"
                return self._handle_pending(text, turn_id, control=decision.task_control)
            if decision.kind in {InteractionKind.CONVERSATION, InteractionKind.CONTENT_RESPONSE}:
                if decision.capability_query:
                    awareness = self.awareness.answer(text)
                    if awareness is not None:
                        return awareness
                memory_response = self._handle_action_memory_query(text, turn_id)
                if memory_response is not None:
                    return memory_response
                self.audit.record("conversation_reasoning_only", turn_id=turn_id,
                    interaction_kind=decision.kind.value, reasoner_called=False, executor_called=False)
                return None
            if pending is not None:
                if pending.reason == "missing_rename_name" and decision.references:
                    return self._handle_pending(text, turn_id)
                self.state.pending_tool_action = None
                self._finish_task("SUPERSEDED", clear_action=True)
            self._plan_source = "FAST_PATH"
            self._turn_reference = None
            from .operation_intent import ArtifactOperation
            if decision.operation_intent == ArtifactOperation.READ:
                return self._read_artifact_content(text, decision, turn_id)
            existing_intent = decision.operation_intent not in {ArtifactOperation.NONE, ArtifactOperation.CREATE}
            if existing_intent or decision.operation_intent != ArtifactOperation.CREATE and (decision.references or 'EDIT_ARTIFACT' in decision.requested_operations):
                target_type = {'CODE_ARTIFACT':'CODE', 'DOCUMENT_ARTIFACT':'DOCUMENT'}.get(decision.output_kind.value, '')
                self._turn_reference = self.state.recent_artifact_state.resolve_ref(text,
                    require_unique=True, artifact_type=target_type)
                self.audit.record('context_artifact_resolution', turn_id=turn_id,
                    **self.state.recent_artifact_state.last_resolution,
                    binding_source='CONVERSATION_ARTIFACT_FOCUS')
                if existing_intent and self._turn_reference is None:
                    return ConversationToolResponse(True, 'Que archivo quieres usar para esa accion?',
                        executed=False, debug={'response_source':'ARTIFACT_REFERENCE_CLARIFICATION'})
                if 'EDIT_ARTIFACT' in decision.requested_operations and target_type == 'CODE' and self._turn_reference is None:
                    choices = [r.filename for r in self.state.recent_artifact_state.artifacts
                        if self.state.recent_artifact_state._selectable(r) and r.runtime_ref and r.runtime_ref.artifact_type == 'CODE']
                    question = ('El programa reciente sigue sin publicarse. Puedes pedirme conservarlo como borrador para revisarlo.'
                        if self._retained_draft else '¿Que programa quieres modificar: ' + ' o '.join(choices[-3:]) + '?')
                    return ConversationToolResponse(True, question, executed=False,
                        debug={'response_source':'ARTIFACT_REFERENCE_CLARIFICATION'})
            action = plan_from_text(
                text,
                self.state,
                content_builder=ContentBuilder(),
                requested_operations=decision.requested_operations,
            )
            if action is not None and action.intent == "needs_artifact_selection":
                self.audit.record("artifact_reference_resolution", turn_id=turn_id,
                    status="MISSING_OR_AMBIGUOUS", executor_called=False)
                return ConversationToolResponse(True,
                    "Necesito que indiques exactamente qué archivo o carpeta quieres usar.",
                    executed=False, debug={"response_source": "ARTIFACT_REFERENCE_CLARIFICATION"})
            deterministic_errors = (
                validate_structured_action_plan(action, self.registry)
                if action is not None
                else ()
            )
            deterministic_unresolved = bool(
                action is not None
                and (
                    deterministic_errors
                    or action.intent.startswith("invalid_")
                    or action.intent in {"needs_patch_details", "copy_missing_target"}
                )
            )
            self.audit.record(
                "conversation_interaction_route",
                turn_id=turn_id,
                category=decision.category.value,
                confidence=decision.confidence,
                primary_goal=decision.primary_goal,
                requested_outcome=decision.requested_outcome.value,
            )

            goal_errors = (
                GoalConsistencyValidator.validate_action(decision, action)
                if action is not None
                else ()
            )
            deterministic_unresolved = deterministic_unresolved or bool(goal_errors)
            deterministic_errors = tuple(deterministic_errors) + tuple(goal_errors)
            if action is not None and not deterministic_unresolved:
                try:
                    self.plan_adapter.compile(action, source="FAST_PATH")
                except ValueError as exc:
                    deterministic_unresolved = True
                    deterministic_errors = (*deterministic_errors, str(exc))
                    self.audit.record("plan_completeness", turn_id=turn_id,
                        plan_source="FAST_PATH", valid=False, errors=(str(exc),), executor_called=False)
            if action is not None and action.explicit_request and not deterministic_unresolved:
                self._trace_dispatch(turn_id, action, planner_selected=True)
                self.state.retry_count = 0
                self._remember_task_plan(action)
                return self._resolve_and_prepare(action, turn_id=turn_id)

            general_response = self._handle_general_reasoning(
                text,
                turn_id=turn_id,
                deterministic_action=action,
                deterministic_errors=tuple(deterministic_errors),
                interaction_decision=decision,
            )
            if general_response is not None:
                return general_response

            if action is None or deterministic_unresolved:
                if decision.kind == InteractionKind.TOOL_ACTION:
                    self.audit.record(
                        "conversation_runtime_trace",
                        turn_id=turn_id,
                        planner_selected=False,
                        executor_called=False,
                        action_result_status="NOT_EXECUTED",
                        response_source="FAIL_CLOSED_UNRESOLVED_TOOL_REQUEST",
                    )
                    return ConversationToolResponse(
                        True,
                        "No pude convertir esa solicitud en un plan de herramienta válido, así que no ejecuté ninguna acción.",
                        executed=False,
                        debug={"response_source": "FAIL_CLOSED_UNRESOLVED_TOOL_REQUEST"},
                    )

            self._trace_dispatch(turn_id, action, planner_selected=True)
            self.state.retry_count = 0
            self._remember_task_plan(action)
            return self._resolve_and_prepare(action, turn_id=turn_id)

    def _read_artifact_content(self, text, decision, turn_id):
        """One exact immutable reference, normal B1 executor, no output artifact."""
        ref = self.state.recent_artifact_state.resolve_ref(
            ' '.join(decision.references) or text, require_unique=True)
        if ref is None:
            return ConversationToolResponse(True, 'Que archivo quieres que te muestre?', executed=False,
                debug={'response_source':'ARTIFACT_REFERENCE_CLARIFICATION'})
        capability = {'CODE':'developer.read_code', 'DOCUMENT':'document.read_document',
                      'TEXT':'filesystem.read'}.get(ref.artifact_type)
        if not capability or not self.registry.find_by_capability(capability):
            return ConversationToolResponse(True, 'No tengo una herramienta autorizada para mostrar ese tipo de archivo.',
                executed=False, debug={'response_source':'READ_UNSUPPORTED'})
        ledger = ArtifactBindings(ref.task_id, {'selected':ref})
        ledger.require('selected', verified=True, allow_unverified=True)
        old = self._executing_plan, self._dataflow_step, self._artifact_bindings
        try:
            self._executing_plan = self._dataflow_step = self._artifact_bindings = None
            result = self._execute_capability(capability, {'path':ref.path},
                turn_id=turn_id, confirmed_effects=())
        finally:
            self._executing_plan, self._dataflow_step, self._artifact_bindings = old
        self.last_results = (result,)
        self._remember_results((result,))
        ledger.require('selected', verified=True, allow_unverified=True)
        if result.status != ExecutionStatus.SUCCESS:
            return self._internal_pipeline_failure(turn_id, reason_code='ARTIFACT_READ_FAILED')
        content = result.output.get('content', result.output.get('text', result.output.get('code')))
        if not isinstance(content, str):
            raise ValueError('READ_CONTENT_OUTPUT_MISSING')
        self.audit.record('artifact_content_read', turn_id=turn_id, artifact_id=ref.artifact_id,
            version=ref.version, hash=ref.hash, capability=capability, mutation=False,
            content_length=len(content), output_kind='CONTENT')
        return ConversationToolResponse(True, content[:12000], executed=True,
            debug={'response_source':'READ_CONTENT', 'artifact_id':ref.artifact_id,
                'version':ref.version, 'hash':ref.hash, 'display_name':ref.display_name,
                'artifact_type':ref.artifact_type, 'format':ref.format})

    def _handle_general_reasoning(
        self,
        text: str,
        *,
        turn_id: str,
        deterministic_action: PlannedToolAction | None,
        deterministic_errors: tuple[str, ...],
        interaction_decision,
    ) -> ConversationToolResponse | None:
        if self.general_task_reasoner is None:
            return None
        if interaction_decision.kind != InteractionKind.TOOL_ACTION:
            raise ValueError("reasoner_requires_tool_action")
        self._plan_source = "GENERAL_REASONER"
        self._reasoner_called = True
        self.audit.record("task_reasoner_dispatch", turn_id=turn_id,
            interaction_kind=interaction_decision.kind.value, reasoner_called=True)
        request = self._general_reasoning_request(text)
        ref = self._turn_reference
        request['artifact_focus'] = ({'logical_name':'selected_artifact', 'display_name':ref.display_name,
            'artifact_type':ref.artifact_type, 'format':ref.format, 'version':ref.version,
            'source_goal':ref.metadata.get('task_goal','')} if ref else None)
        from .semantic_proposal import semantic_vocabulary
        request["semantic_operations"] = semantic_vocabulary(self.registry)
        request["interaction_decision"] = {
            "category": interaction_decision.category.value,
            "confidence": interaction_decision.confidence,
            "primary_goal": interaction_decision.primary_goal,
            "requested_outcome": interaction_decision.requested_outcome.value,
            "needs_content_generation": interaction_decision.needs_content_generation,
            "references_existing_artifact": interaction_decision.references_existing_artifact,
        }
        request["interaction_resolution"] = {
            "kind": interaction_decision.kind.value,
            "output_kind": interaction_decision.output_kind.value,
            "task_control": interaction_decision.task_control.value,
            "requested_effects": interaction_decision.requested_effects,
            "artifact_intent": interaction_decision.artifact_intent,
            "references": interaction_decision.references,
            "primary_goal": interaction_decision.primary_goal,
            "requested_operations": interaction_decision.requested_operations,
            "requires_execution": interaction_decision.requires_execution,
            "verification_requirement": interaction_decision.verification_requirement.value,
            "operation_intent": interaction_decision.operation_intent.value,
            "effect_intent": interaction_decision.effect_intent.to_dict(),
        }
        from .semantic_proposal import operation_constraint
        request['operation_constraint'] = operation_constraint(request['interaction_resolution'])
        allowed = request['operation_constraint']['allowed_semantic_operations']
        if allowed is not None:
            request['semantic_operations'] = [op for op in request['semantic_operations']
                if op['operation_kind'] in allowed]
        try:
            proposal = self._pipeline_trace.invoke("SEMANTIC_PLANNING", self.general_task_reasoner, request)
            plan = proposal.proposal if isinstance(proposal, BrainPlanProposal) else proposal
            brain_candidate = plan
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            pending = self.state.pending_task
            pending.task_id = f"task:{uuid4().hex}"
            pending.status = "FAILED"
            pending.plan = pending.failed_plan = None
            pending.failed_action = None
            self.state.last_tool_failure = LastToolFailure.create(
                task_id=pending.task_id, stage="GENERAL_REASONER",
                error_code=("PLAN_PROTOCOL_ERROR" if type(exc).__name__ == "PlanProtocolError"
                    else f"GENERAL_REASONER_FAILED:{type(exc).__name__}"),
                technical_message=(str(exc) if type(exc).__name__ == "PlanProtocolError" else type(exc).__name__),
                safe_user_message="No pude preparar la ejecucion en este momento; no ejecute ninguna accion.",
                retryable=False)
            self.audit.record(
                "general_task_reasoner_failed",
                turn_id=turn_id,
                error_type=type(exc).__name__,
                error_code=str(exc)[:240],
                deterministic_errors=deterministic_errors,
            )
            return ConversationToolResponse(True, "No pude formar un plan válido, así que no ejecuté ninguna acción.",
                executed=False, debug={"reason_code":self.state.last_tool_failure.error_code})
        from .plan_protocol import budget, consume_repair
        from .capability_grounding import CapabilityGroundingValidator
        def validate_grounded(candidate):
            from .semantic_proposal import SemanticProposal, operation_mismatches
            from .plan_lowering import ExecutablePlanLowerer, PlanLoweringError
            if isinstance(candidate, SemanticProposal):
                original = candidate
                self.state.pending_task.task_id = original.task_id
                try:
                    mismatches = operation_mismatches(original, request['operation_constraint'])
                    if mismatches:
                        raise PlanLoweringError(mismatches[0])
                    candidate = self._pipeline_trace.invoke("PLAN_LOWERING", ExecutablePlanLowerer(self.registry,
                        self.state.active_workspace, external_artifacts={'selected_artifact':ref} if ref else {}).lower_requirements, original)
                    if ref and ref.artifact_type == 'CODE' and candidate.task_type != 'MODIFY_CODE' and (
                            'EDIT_ARTIFACT' in interaction_decision.requested_operations):
                        raise PlanLoweringError('CONTEXT_ARTIFACT_BINDING_REQUIRED')
                    self.audit.record("semantic_plan_lowering", task_id=original.task_id,
                        status="LOWERED", capabilities_selected_locally=candidate.required_capabilities,
                        operations=[{"id":o["operation_id"],"kind":o["operation_kind"],
                            "inputs":o["inputs"],"outputs":o["outputs"]} for o in original.payload["operations"]])
                except PlanLoweringError as exc:
                    candidate = GeneralTaskPlan.from_dict({"task_id":original.task_id,
                        "goal":original.payload["goal"],"task_type":"TOOL_COMPOSITION"})
                    result = self.general_plan_validator.validate(candidate,
                        authorized_scopes=self.state.authorized_scopes)
                    self.audit.record("semantic_plan_lowering", task_id=original.task_id,
                        status="REJECTED", error_code=str(exc), executor_called=False)
                    return candidate, replace(result, valid=False, errors=(str(exc),))
            try:
                candidate = self._pipeline_trace.invoke("PLAN_VALIDATION", CapabilityGroundingValidator(self.registry).ground, candidate)
                grounding_errors = ()
            except ValueError as exc:
                grounding_errors = (str(exc),)
            if grounding_errors:
                result = self.general_plan_validator.validate(candidate,
                    authorized_scopes=self.state.authorized_scopes)
                result = replace(result, valid=False, errors=grounding_errors + result.errors)
            else:
                from .verification_types import test_spec_diagnostics
                with test_spec_diagnostics(self.audit.record, task_id=candidate.task_id,
                        source_step=next((s.step_id for s in candidate.steps
                            if s.capability=='developer.write_code'), ''),
                        proposal_origin='REPAIR' if budget(request)['used'] else 'ORIGINAL'):
                    result = self._validate_complete_proposal(candidate, text, interaction_decision)
            self.audit.record("capability_grounding", task_id=candidate.task_id,
                capability_grounding_status="REJECTED" if grounding_errors else "VALID",
                errors=grounding_errors, executor_called=False)
            return candidate, result
        plan, validation = validate_grounded(plan)
        repair_count = int(budget(request)["used"])
        if not validation.valid:
            self.audit.record("plan_completeness_rejected", turn_id=turn_id,
                task_id=plan.task_id, errors=validation.errors, executor_called=False,
                step_bindings=[{"step_id": s.step_id, "produces": s.produces_artifacts,
                    "consumes": s.consumes_artifacts} for s in plan.steps])
            request["required_artifact_bindings"] = {
                "requested_effects": interaction_decision.requested_effects,
                "requested_outputs": plan.requested_final_outputs,
                "declared_outputs": plan.desired_outputs,
                "steps": [{"step_id": s.step_id, "produces": s.produces_artifacts,
                    "consumes": s.consumes_artifacts} for s in plan.steps],
            }
            self.state.pending_task.task_id = plan.task_id
            self.state.pending_task.status = "FAILED"
            self.state.pending_task.plan = self.state.pending_task.failed_plan = None
            self.state.last_failed_action = None
            repair = getattr(self.general_task_reasoner, "repair", None)
            if callable(repair) and not budget(request)["used"]:
                repair_count = 1
                try:
                    if not getattr(self.general_task_reasoner, "handles_shared_repair", False):
                        consume_repair(request, "PLAN_VALIDATION")
                    proposal = repair(request, brain_candidate, validation.errors)
                    plan = proposal.proposal if isinstance(proposal, BrainPlanProposal) else proposal
                    brain_candidate = plan
                    plan, validation = validate_grounded(plan)
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    self.audit.record(
                        "general_task_repair_failed",
                        turn_id=turn_id,
                        error_type=type(exc).__name__,
                    )
        if validation.valid:
            self._pipeline_trace.failed_stage = self._pipeline_trace.failed_reason = ""
        self.audit.record(
            "general_task_plan",
            turn_id=turn_id,
            task_id=plan.task_id,
            task_type=plan.task_type,
            deterministic_planner_selected=deterministic_action is not None,
            general_reasoner_selected=True,
            capabilities_selected=plan.required_capabilities,
            plan_valid=validation.valid,
            validation_errors=validation.errors,
            repair_count=repair_count,
            executor_called=False,
            verification_passed=False,
        )
        self.audit.record("plan_completeness", turn_id=turn_id, valid=validation.valid,
            errors=validation.errors, repair_count=repair_count, executor_called=False,
            task_id=plan.task_id, steps=[{"step_id": s.step_id, "capability": s.capability,
                "produces": s.produces_artifacts, "consumes": s.consumes_artifacts,
                "produces_values": s.produces_values, "consumes_values": s.consumes_values,
                "argument_keys": tuple(sorted(s.arguments))} for s in plan.steps],
            declared_outputs=plan.desired_outputs, requested_final_outputs=plan.requested_final_outputs)
        if not validation.valid:
            precise = validation.errors[0] if validation.errors else 'GENERAL_PLAN_INVALID'
            self.state.last_tool_failure = LastToolFailure.create(
                task_id=plan.task_id,
                stage="PLAN_VALIDATION",
                error_code=precise,
                technical_message=precise,
                safe_user_message=("No pude preparar la ejecucion del programa."
                    if precise.startswith('SEMANTIC_OPERATION_MISMATCH') else
                    "No pude preparar una secuencia de acciones valida para esa peticion."),
                validation_detail={'validator':'LOCAL_PLAN_CONTRACT', 'reason_code':precise,
                    'validation_stage':'PLAN_LOWERING' if precise.startswith(('PLAN_LOWERING_FAILED:','PLAN_AMBIGUOUS:')) else 'PLAN_VALIDATION',
                    'failed_constraint':precise.split(':')[1] if precise.startswith(('PLAN_LOWERING_FAILED:', 'PLAN_AMBIGUOUS:', 'PLAN_INCOMPLETE:')) else precise.split(':')[0],
                    'repair_attempted':bool(repair_count), 'repair_result':'REJECTED',
                    'errors':list(validation.errors)[:8]},
                retryable=False,
            )
            self._record_attempt_outcome("FAILED", turn_id=turn_id, failure=self.state.last_tool_failure)
            self.audit.record('conversation_tool_failure_state', turn_id=turn_id,
                failure_record=asdict(self.state.last_tool_failure))
            return ConversationToolResponse(
                True,
                "No pude formar un plan seguro con las capacidades disponibles, así que no ejecuté ninguna acción.",
                executed=False,
                debug={
                    "response_source": "GENERAL_PLAN_INVALID",
                    "errors": validation.errors,
                    "repair_count": repair_count,
                },
            )
        if plan.task_type == "REASONING_ONLY":
            return ConversationToolResponse(True,
                "El plan no incluye la acción solicitada. Necesito aclarar el resultado antes de continuar.",
                executed=False)
        if plan.task_type in {"MISSING_CAPABILITY", "UNSUPPORTED"}:
            return ConversationToolResponse(
                True,
                "No tengo todavía una capacidad verificada para completar esa tarea.",
                CapabilityResolutionKind.MISSING_CAPABILITY,
                executed=False,
                debug={"response_source": "GENERAL_PLAN_MISSING_CAPABILITY"},
            )
        if plan.task_type == "FORBIDDEN":
            return ConversationToolResponse(
                True,
                "No puedo ejecutar esa tarea porque afecta una zona protegida de Anydes.",
                CapabilityResolutionKind.FORBIDDEN,
                executed=False,
                debug={"response_source": "GENERAL_PLAN_FORBIDDEN"},
            )
        if plan.ambiguities:
            return ConversationToolResponse(
                True,
                f"Necesito aclarar algo antes de continuar: {plan.ambiguities[0]}",
                executed=False,
                debug={"response_source": "GENERAL_PLAN_NEEDS_CLARIFICATION"},
            )
        if plan.task_type == "GENERATE_CONVERSATIONAL_CONTENT":
            return ConversationToolResponse(True,
                "El plan propone una respuesta textual, pero pediste una acción. Necesito aclarar el resultado que quieres.",
                executed=False, debug={"response_source": "PROPOSAL_OUTCOME_CONTRADICTION"})
        action = self._action_from_general_plan(plan, text)
        if action is None:
            self.state.pending_task.task_id = plan.task_id
            self.state.pending_task.status = "FAILED"
            self.state.pending_task.plan = self.state.pending_task.failed_plan = None
            self.state.last_failed_action = None
            self.state.last_tool_failure = LastToolFailure.create(
                task_id=plan.task_id,
                stage="GENERAL_PLAN_LOWERING",
                error_code="GENERAL_PLAN_NOT_EXECUTABLE",
                safe_user_message="el plan conceptual no pudo traducirse a una ejecución local verificada",
                retryable=False,
            )
            return ConversationToolResponse(
                True,
                "No pude traducir el plan conceptual a una ejecución local verificada.",
                executed=False,
                debug={"response_source": "GENERAL_PLAN_NOT_EXECUTABLE"},
            )
        self._trace_dispatch(turn_id, action, planner_selected=False)
        self.state.retry_count = 0
        self._remember_task_plan(action)
        return self._resolve_and_prepare(action, turn_id=turn_id)

    def _validate_complete_proposal(self, plan, text, decision):
        from .general_reasoning import GeneralTaskPlanValidation
        try:
            return self._pipeline_trace.invoke("PLAN_VALIDATION", self._validate_complete_proposal_checked, plan, text, decision)
        except ValueError as exc:
            # Expected strict sub-schema rejection participates in the existing one repair.
            return GeneralTaskPlanValidation(False, (exception_code(exc),), "INVALID", plan.required_capabilities)

    def _validate_complete_proposal_checked(self, plan, text, decision):
        validation = self.general_plan_validator.validate(plan,
            authorized_scopes=self.state.authorized_scopes,
            external_artifacts=('selected_artifact',) if self._turn_reference else ())
        errors = list(validation.errors)
        errors.extend(GoalConsistencyValidator.validate_plan(decision, plan))
        from .verification_types import VerificationRequirement
        required = decision.verification_requirement
        if required in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE}:
            raw = plan.inputs.get("code_spec", {})
            if plan.task_type == "GENERATE_CODE":
                if not raw.get("execution_required"):
                    errors.append("PLAN_INCOMPLETE:requested_execution_missing")
                if required == VerificationRequirement.TEST_SUITE and not (raw.get("tests") or raw.get('test_cases')):
                    errors.append("PLAN_INCOMPLETE:requested_test_suite_missing")
                if not any(s.capability in {"developer.run_python", "developer.run_cpp", "developer.test"} for s in plan.steps):
                    errors.append("PLAN_INCOMPLETE:requested_verifier_missing")
        if not errors and plan.needs_tools and not plan.ambiguities:
            action = self._action_from_general_plan(plan, text)
            if action is None:
                errors.append("PLAN_INCOMPLETE:proposal_not_lowerable")
            else:
                try:
                    compiled = self.plan_adapter.compile(action, source="GENERAL_REASONER", task_id=plan.task_id)
                    from .plan_completeness import PlanCompletenessValidator
                    PlanCompletenessValidator().validate(replace(compiled,
                        typed_inputs={**compiled.typed_inputs, "verification_requirement":required.value}))
                except ValueError as exc:
                    errors.append(str(exc))
        return replace(validation, valid=not errors, errors=tuple(dict.fromkeys(errors)))

    def _general_reasoning_request(self, text: str) -> dict[str, Any]:
        safe_context = ""
        if self.safe_context_provider is not None:
            try:
                safe_context = str(self.safe_context_provider("USER_CONTEXT", text))[:40_000]
            except (RuntimeError, TypeError, ValueError):
                safe_context = ""
        artifacts = [
            {
                "artifact_id": item.get("artifact_id", ""),
                "filename": item.get("display_name", ""),
                "format": item.get("format", ""),
                "topic": item.get("semantic_topic", ""),
                "relationship": item.get("relationship", {}),
            }
            for item in self.state.recent_artifact_state.snapshot()[-12:]
        ]
        return {
            "user_request": text[:12_000],
            "tool_registry": compact_capability_vocabulary(self.registry),
            "workspace": {
                "display_name": Path(self.state.active_workspace).name,
                "authorized": True,
                "protected_core": True,
            },
            "recent_artifacts": artifacts,
            "recent_code_artifacts": [
                {
                    "filename": Path(path).name,
                    "active": path == self.state.active_code_file,
                }
                for path in self.state.recent_files[-12:]
                if Path(path).suffix.casefold() in {".py", ".cpp", ".cc", ".cxx"}
            ],
            "recent_actions": self.state.recent_action_state.snapshot()[-8:],
            "safe_relevant_context": safe_context,
            "recent_content_response": self.recent_content_response[:12000],
            "policy": {
                "brain_has_execution_authority": False,
                "tool_builder_enabled": False,
                "max_steps": 8,
                "one_repair_only": True,
            },
        }

    def _execute_general_conversational_content(
        self,
        plan: GeneralTaskPlan,
        text: str,
        turn_id: str,
    ) -> ConversationToolResponse:
        if self.semantic_prose_generator is None:
            return ConversationToolResponse(
                True,
                "No pude generar el contenido porque el generador no está disponible.",
                executed=False,
                debug={"response_source": "GENERAL_CONTENT_GENERATOR_UNAVAILABLE"},
            )
        raw_spec = dict(plan.inputs.get("content_spec", {}))
        raw_spec.setdefault("content_kind", str(plan.inputs.get("content_kind", "expository_text")))
        raw_spec.setdefault("topic", str(plan.inputs.get("topic", plan.goal)))
        raw_spec.setdefault("purpose", "conversation")
        raw_spec.setdefault("target_length_words", int(plan.inputs.get("target_length_words", 200)))
        raw_spec.setdefault("language", str(plan.inputs.get("language", "es")))
        raw_spec.setdefault("tone", str(plan.inputs.get("tone", "natural")))
        raw_spec.setdefault("structure", str(plan.inputs.get("structure", "paragraphs")))
        raw_spec.setdefault("constraints", tuple(plan.constraints))
        raw_spec.setdefault("source_mode", str(plan.inputs.get("source_mode", "GENERATIVE")))
        try:
            specification = ContentGenerationSpec.from_dict(raw_spec)
            if specification.source_mode != "GENERATIVE" and not specification.source_context:
                context = ""
                if self.safe_context_provider is not None:
                    context = str(self.safe_context_provider(specification.source_mode, text))
                specification = specification.with_source_context(context)
            generated = str(self.semantic_prose_generator(specification)).strip()
            validation = ContentValidator.validate(generated, specification)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.state.last_tool_failure = LastToolFailure.create(
                task_id=plan.task_id,
                stage="CONTENT_GENERATION",
                error_code=f"GENERAL_CONTENT_FAILED:{type(exc).__name__}",
                safe_user_message="la generación de contenido no terminó correctamente",
                retryable=True,
            )
            return ConversationToolResponse(
                True,
                "No pude completar el contenido solicitado.",
                executed=False,
                debug={"response_source": "GENERAL_CONTENT_FAILED"},
            )
        if validation.status != "VALID":
            self.state.last_tool_failure = LastToolFailure.create(
                task_id=plan.task_id,
                stage="CONTENT_VALIDATION",
                error_code=validation.reason or "GENERAL_CONTENT_INVALID",
                safe_user_message="el contenido generado no superó la validación local",
                retryable=True,
            )
            return ConversationToolResponse(
                True,
                "No pude validar el contenido generado.",
                executed=False,
                debug={"response_source": "GENERAL_CONTENT_INVALID", "validation": asdict(validation)},
            )
        self.audit.record(
            "general_task_completed",
            turn_id=turn_id,
            task_id=plan.task_id,
            task_type=plan.task_type,
            executor_called=False,
            verification_passed=True,
            word_count=validation.word_count,
        )
        return ConversationToolResponse(
            True,
            generated,
            CapabilityResolutionKind.REASONING_ONLY,
            executed=False,
            debug={"response_source": "GENERAL_REASONER_CONTENT", "verification_passed": True},
        )

    def _action_from_general_plan(
        self,
        plan: GeneralTaskPlan,
        source_text: str,
    ) -> PlannedToolAction | None:
        if plan.task_type == "MODIFY_CODE":
            try:
                specification = CodeModificationSpec.from_dict(
                    dict(plan.inputs.get("code_modification_spec", {}))
                )
            except (TypeError, ValueError):
                return None
            selected = self._turn_reference
            target = Path(selected.path) if selected else None
            if target is None or not target.is_file():
                return None
            expected_language = "python" if target.suffix.casefold() == ".py" else "cpp"
            if specification.language not in {expected_language, "c++" if expected_language == "cpp" else expected_language}:
                return None
            if not {"developer.read_code", "developer.write_code"}.issubset(
                set(plan.required_capabilities)
            ):
                return None
            arguments = {
                "path": str(target),
                "code_modification_spec": specification.to_dict(include_source=False),
                "general_task_plan": plan.to_dict(),
            }
            task = UserTaskSpec(
                task_id=plan.task_id,
                goal="modify_code",
                target_type="code",
                output_format=expected_language,
                filename=target.name,
                content_requirements={"code_modification_spec": arguments["code_modification_spec"]},
                constraints=tuple(plan.constraints),
                destination=str(target.parent),
            )
            tool = self.registry.find_by_capability("developer.write_code")[0]
            structured = StructuredActionPlan(
                plan_id=f"plan:{uuid4().hex}",
                action_type="COMPOSABLE",
                tool_id=tool.manifest.tool_id,
                capability="developer.write_code",
                arguments=arguments,
                workspace=str(target.parent),
                task_id=task.task_id,
            )
            return PlannedToolAction(
                intent="modify_code",
                capabilities=plan.required_capabilities,
                operation="general_modify_code",
                arguments=arguments,
                target_path=str(target),
                source_text=source_text,
                task_spec=task,
                structured_action_plan=structured,
                resolved_artifacts=(selected,),
                resolved_artifact_bindings={'selected_artifact':selected},
            )
        if plan.task_type == "GENERATE_CODE":
            try:
                specification = CodeGenerationSpec.from_dict(dict(plan.inputs.get("code_spec", {})))
            except (TypeError, ValueError):
                return None
            path = (Path(self.state.active_workspace) / specification.target_filename).resolve(strict=False)
            tool = self.registry.find_by_capability("developer.write_code")
            if not tool or "developer.write_code" not in plan.required_capabilities:
                return None
            if specification.execution_required and not any(
                capability in plan.required_capabilities
                for capability in ("developer.run_python", "developer.run_cpp", "developer.test")
            ):
                return None
            arguments = {
                "path": str(path),
                "code_generation_spec": specification.to_dict(),
                "general_task_plan": plan.to_dict(),
            }
            task = UserTaskSpec(
                task_id=plan.task_id,
                goal="generate_code",
                target_type="code",
                output_format=specification.language,
                filename=path.name,
                content_requirements={"code_generation_spec": specification.to_dict()},
                constraints=tuple(plan.constraints),
                destination=str(path.parent),
            )
            structured = StructuredActionPlan(
                plan_id=f"plan:{uuid4().hex}",
                action_type="COMPOSABLE" if len(plan.steps) > 1 else "EXACT_TOOL",
                tool_id=tool[0].manifest.tool_id,
                capability="developer.write_code",
                arguments=arguments,
                workspace=str(path.parent),
                task_id=task.task_id,
            )
            return PlannedToolAction(
                intent="generate_code",
                capabilities=plan.required_capabilities,
                operation="general_generate_code",
                arguments=arguments,
                target_path=str(path),
                source_text=source_text,
                task_spec=task,
                structured_action_plan=structured,
            )
        if plan.task_type == "GENERATE_DOCUMENT":
            artifact = dict(plan.artifact_requirements)
            format_id = str(artifact.get("format", "")).casefold()
            capability = {
                "txt": "document.create_txt",
                "md": "document.create_markdown",
                "markdown": "document.create_markdown",
                "docx": "document.create_docx",
                "pdf": "document.create_pdf",
            }.get(format_id)
            if not capability or capability not in plan.required_capabilities:
                return None
            suffix = {"txt": ".txt", "md": ".md", "markdown": ".md", "docx": ".docx", "pdf": ".pdf"}[format_id]
            filename = sanitize_windows_stem(Path(str(artifact.get("filename", "documento"))).stem) + suffix
            path = (Path(self.state.active_workspace) / filename).resolve(strict=False)
            raw_spec = dict(plan.inputs.get("content_spec", {}))
            raw_spec.setdefault("topic", plan.goal)
            raw_spec.setdefault("purpose", "document")
            raw_spec.setdefault("target_length_words", 300)
            raw_spec.setdefault("language", "es")
            raw_spec.setdefault("tone", "natural")
            raw_spec.setdefault("structure", "paragraphs")
            raw_spec.setdefault("constraints", tuple(plan.constraints))
            raw_spec.setdefault("content_kind", "expository_text")
            specification = ContentGenerationSpec.from_dict(raw_spec)
            if (specification.source_mode == "CONVERSATION_CONTEXT"
                    and self.recent_content_response and self.last_resolution.references):
                specification = specification.with_source_context(
                    "PROVENANCE: DIRECT_CONVERSATION\n" + self.recent_content_response)
            source_reference = str(
                artifact.get("source_filename")
                or artifact.get("source_reference")
                or plan.inputs.get("source_filename", "")
            ).strip()
            source_path = ""
            if source_reference:
                source_path = self.state.recent_artifact_state.resolve(
                    source_text,
                    explicit_filename=Path(source_reference).name,
                )
                if not source_path:
                    candidate = (Path(self.state.active_workspace) / Path(source_reference).name).resolve(strict=False)
                    source_path = str(candidate) if candidate.is_file() else ""
                if not source_path:
                    return None
            elif str(specification.source_mode).upper() == "TRANSFORM_EXISTING":
                source_path = self.state.active_document
                if not source_path:
                    return None
            if source_path and "document.read_document" not in plan.required_capabilities:
                return None
            if specification.source_mode != "GENERATIVE" and not specification.source_context and self.safe_context_provider is not None:
                specification = specification.with_source_context(
                    str(self.safe_context_provider(specification.source_mode, source_text))
                )
            arguments = {
                "path": str(path),
                "overwrite": path.exists(),
                "content_generation_spec": specification.to_dict(),
                "document_title": str(artifact.get("title", path.stem)),
                "source_path": source_path,
                "general_task_plan": plan.to_dict(),
            }
            task = UserTaskSpec(
                task_id=plan.task_id,
                goal="generate_document",
                target_type="document",
                output_format="markdown" if format_id == "md" else format_id,
                filename=path.name,
                content_requirements={"content_generation_spec": specification.to_dict()},
                constraints=tuple(plan.constraints),
                destination=str(path.parent),
            )
            tool = self.registry.find_by_capability(capability)[0]
            structured = StructuredActionPlan(
                plan_id=f"plan:{uuid4().hex}",
                action_type="EXACT_TOOL",
                tool_id=tool.manifest.tool_id,
                capability=capability,
                arguments=arguments,
                workspace=str(path.parent),
                task_id=task.task_id,
            )
            return PlannedToolAction(
                intent="generate_document",
                capabilities=plan.required_capabilities,
                operation=(
                    "general_generated_document"
                    if source_path
                    else "generated_create_document"
                ),
                arguments=arguments,
                target_path=str(path),
                source_text=source_text,
                task_spec=task,
                structured_action_plan=structured,
            )
        if plan.task_type == "TOOL_COMPOSITION" and plan.steps:
            first = plan.steps[0]
            tools = self.registry.find_by_capability(first.capability)
            if not tools:
                return None
            task = UserTaskSpec(
                task_id=plan.task_id,
                goal="general_tool_composition",
                target_type="composition",
                output_format=str(plan.artifact_requirements.get("format", "")),
                filename=str(plan.artifact_requirements.get("filename", "")),
                content_requirements={"general_task_plan": plan.to_dict()},
                constraints=tuple(plan.constraints),
                destination=self.state.active_workspace,
            )
            structured = StructuredActionPlan(
                plan_id=f"plan:{uuid4().hex}",
                action_type="COMPOSABLE",
                tool_id=tools[0].manifest.tool_id,
                capability=first.capability,
                arguments={"general_task_plan": plan.to_dict()},
                workspace=self.state.active_workspace,
                task_id=task.task_id,
            )
            return PlannedToolAction(
                intent="general_tool_composition",
                capabilities=plan.required_capabilities,
                operation="general_tool_composition",
                arguments={"general_task_plan": plan.to_dict()},
                source_text=source_text,
                task_spec=task,
                structured_action_plan=structured,
                resolved_artifact_bindings={'selected_artifact':self._turn_reference}
                    if self._turn_reference and any('selected_artifact' in step.consumes_artifacts for step in plan.steps) else {},
            )
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = self.state.pending_tool_action
            return {
                "active_workspace": self.state.active_workspace,
                "authorized_scopes": list(self.state.authorized_scopes),
                "active_document": self.state.active_document,
                "active_document_ref": asdict(self.state.active_document_ref) if self.state.active_document_ref else None,
                "source_document": self.state.source_document,
                "related_artifacts": list(self.state.related_artifacts),
                "recent_artifacts": self.state.recent_artifact_state.snapshot(),
                "active_artifact_id": self.state.recent_artifact_state.active_artifact_id,
                "recent_actions": self.state.recent_action_state.snapshot(),
                "active_code_file": self.state.active_code_file,
                "recent_files": list(self.state.recent_files),
                "recent_tool_results": list(self.state.recent_tool_results),
                "pending_task": {
                    "task_id": self.state.pending_task.task_id,
                    "plan_revision": self.state.pending_task.plan_revision,
                    "status": self.state.pending_task.status,
                    "pending_control": self.state.pending_task.pending_control,
                    "failure_ref": self.state.last_tool_failure.error_code if self.state.last_tool_failure else "",
                },
                "pending_tool_action": asdict(pending) if pending else None,
                "retry_count": self.state.retry_count,
                "last_verification": dict(self.state.last_verification),
                "last_tool_failure": (
                    asdict(self.state.last_tool_failure)
                    if self.state.last_tool_failure is not None
                    else None
                ),
                "pending_user_task": asdict(self.state.pending_user_task) if self.state.pending_user_task else None,
                "pending_structured_plan": asdict(self.state.pending_structured_plan) if self.state.pending_structured_plan else None,
                "replan_count": self.state.replan_count,
                "settings": self.settings.safe_public_payload(),
                "capability_snapshot": asdict(self.registry.capability_snapshot()),
            }

    def capability_context_note(self) -> str:
        return self.awareness.context_note()

    def use_isolated_action_memory(self, root: Path) -> None:
        """Redirect action evidence for deterministic staging tests only."""

        self.action_memory = ActionExperienceMemory(
            root.resolve(),
            recent_state=self.state.recent_action_state,
        )

    def close(self) -> None:
        with self._lock:
            self.state.pending_tool_action = None
            self._drop_retained_draft()
        self.settings.unsubscribe(self._apply_settings_snapshot)

    def _drop_retained_draft(self):
        draft = self._retained_draft
        if draft:
            path = Path(draft['source'])
            root = Path(draft.get('workspace_root',self.state.active_workspace))
            owned = root/'.anydes-drafts'
            if owned.exists() and owned.resolve(strict=True) != owned:
                self.audit.record('quality_cleanup_denied', reason_code='REPARSE_DRAFT_ROOT')
                return
            if (path.parent == owned and path.name.startswith('draft_')) or (
                    path.parent == Path(self.state.active_workspace) and path.name.startswith('.anydes-code-')):
                path.unlink(missing_ok=True)
                if owned.is_dir() and not any(owned.iterdir()):
                    owned.rmdir()
            self._retained_draft = None
            self.state.recent_artifact_state.focus_unavailable = False

    def _retain_unverified(self, turn_id, source_request):
        draft = self._retained_draft
        if not draft or time.time()-draft['created_at'] > 1800:
            raise ValueError('QUALITY_RETENTION_NOT_AVAILABLE')
        path = Path(draft['destination'])
        from .dataflow import path_version
        if path_version(path) != draft['expected_destination_hash']:
            raise DataflowError('STALE_ARTIFACT:quality_destination')
        task = UserTaskSpec(task_id=draft['task_id'], goal='retain_unverified', target_type='code',
            output_format=path.suffix.lstrip('.'), filename=path.name, destination=str(path.parent))
        action = PlannedToolAction('retain_unverified', ('developer.write_code',), operation='retain_unverified',
            arguments={'path':str(path), 'overwrite':path.exists(), 'general_task_plan':{
                'steps':[{'step_id':'retain_source', 'capability':'developer.write_code',
                    'produces_artifacts':['retained_source'], 'arguments':{'path':str(path)}}],
                'desired_outputs':[{'binding':'retained_source','artifact_type':'CODE','format':path.suffix.lstrip('.')}]}},
            target_path=str(path), task_spec=task, source_text=source_request)
        previous = draft.get('previous_ref')
        if previous:
            action.arguments['general_task_plan']['steps'][0]['consumes_artifacts'] = ['previous_source']
            action = replace(action, resolved_artifact_bindings={
                'previous_source':replace(previous,producer_step_id='INPUT')})
        action = replace(action, structured_action_plan=StructuredActionPlan(
            plan_id='plan:'+uuid4().hex, action_type='EXACT_TOOL', tool_id='official.developer',
            capability='developer.write_code', arguments=action.arguments,
            workspace=str(path.parent), task_id=task.task_id))
        self.state.pending_task.plan = None
        self.state.pending_task.plan_revision += 1
        draft['failure_id'] = self.state.last_tool_failure.failure_id if self.state.last_tool_failure else ''
        self.state.last_tool_failure = None
        self._remember_task_plan(action)
        return self._resolve_and_prepare(action, turn_id=turn_id)

    def _apply_settings_snapshot(self, snapshot: SettingsSnapshot) -> None:
        with self._lock:
            self.state.active_workspace = snapshot.active_workspace
            self.state.authorized_scopes = snapshot.authorized_external_scopes

    def _handle_action_memory_query(
        self,
        text: str,
        turn_id: str,
    ) -> ConversationToolResponse | None:
        if not self.action_memory.is_retrieval_query(text):
            return None
        events = self.action_memory.retrieve(text, limit=5)
        if self.action_memory_responder is not None:
            try:
                answer = str(self.action_memory_responder(text, events)).strip()
            except (RuntimeError, TypeError, ValueError):
                answer = ""
        else:
            answer = ""
        if not answer:
            if not events:
                answer = "No encontré una acción registrada que confirme eso."
            else:
                answer = " ".join(item.action_summary + "." for item in events[:3])
        self.audit.record(
            "conversation_action_memory_retrieved",
            turn_id=turn_id,
            evidence_count=len(events),
            evidence_ids=[item.event_id for item in events],
            provenance=[item.provenance for item in events],
        )
        return ConversationToolResponse(
            True,
            answer,
            executed=False,
            debug={
                "response_source": "ACTION_EXPERIENCE_MEMORY",
                "evidence_ids": [item.event_id for item in events],
                "verified_action_evidence": bool(events),
            },
        )

    def _handle_failure_explanation(
        self,
        text: str,
        turn_id: str,
        *,
        force: bool = False,
    ) -> ConversationToolResponse | None:
        failure = self.state.failure_topic_record()
        if failure is None or (not force and not is_failure_explanation_request(text)):
            return None
        self.state.failure_topic_active = True
        if self.last_resolution is not None and self.last_resolution.task_control == TaskControl.REMEDIATION_QUERY:
            pending = self.state.pending_task
            assessment = RemediationAssessment.assess(failure,
                retry_available=(pending.failed_action is not None and pending.failed_plan is not None
                    and self.state.retry_count < 1 and not pending.committed_effects))
            self.audit.record('conversation_failure_remediation', turn_id=turn_id,
                task_id=failure.task_id, root_failure_id=failure.failure_id,
                assessment=asdict(assessment), executor_called=False, reasoner_called=False)
            return ConversationToolResponse(True, assessment.safe_reason, executed=False,
                debug={'response_source':'STRUCTURED_FAILURE_RESULT', 'failure_stage':failure.stage,
                    'error_code':failure.error_code, 'remediation_assessment':asdict(assessment)})
        self.state.failure_explanation_depth = min(3, self.state.failure_explanation_depth + 1)
        self.state.active_failure_topic.explanation_depth = self.state.failure_explanation_depth
        answer = FailureResponseRenderer.render(failure, depth=self.state.failure_explanation_depth)
        self.audit.record(
            "conversation_failure_explained",
            turn_id=turn_id,
            task_id=failure.task_id,
            failure_id=failure.failure_id, plan_revision=failure.plan_revision,
            attempt_number=failure.attempt_number, step_id=failure.step_id,
            original_plan_ref=failure.original_plan_ref,
            verification_ref=failure.verification_ref, artifact_type=failure.artifact_type,
            stage=failure.stage,
            error_code=failure.error_code,
            retryable=failure.retryable,
            response_source="STRUCTURED_FAILURE_RESULT",
            explanation_depth=self.state.failure_explanation_depth,
            root_failure_id=self.state.active_failure_topic.root_failure_id,
        )
        return ConversationToolResponse(
            True,
            answer,
            executed=False,
            debug={
                "response_source": "STRUCTURED_FAILURE_RESULT",
                "failure_stage": failure.stage,
                "error_code": failure.error_code,
            },
        )

    def _handle_pending(self, text: str, turn_id: str, *, control=None) -> ConversationToolResponse | None:
        pending = self.state.pending_tool_action
        if pending is None:
            return None
        if time.time() > pending.expires_at:
            self.state.pending_tool_action = None
            self._finish_task("EXPIRED", clear_action=True)
            self.audit.record(
                "pending_tool_expired",
                turn_id=turn_id,
                pending_action_id=pending.pending_request_id,
                task_id=pending.task_id,
                plan_id=pending.plan_id,
            )
            return ConversationToolResponse(
                True,
                "La confirmación anterior expiró y no ejecuté ninguna acción.",
                executed=False,
            )
        if pending.reason == "missing_rename_name":
            new_name = _filename_from_followup(text, pending.action.target_path)
            if not new_name:
                return ConversationToolResponse(
                    True,
                    "Necesito un nombre de archivo válido para continuar.",
                    executed=False,
                )
            resumed = replace(
                pending.action,
                intent="rename",
                arguments={**pending.action.arguments, "new_name": new_name},
            )
            self.state.pending_tool_action = None
            return self._prepare_or_execute(resumed, turn_id=turn_id)

        destructive = PermissionEffect.DELETE in pending.confirmed_effects
        decision = (
            ConfirmationDecision.CONFIRM if control == TaskControl.CONFIRM else
            ConfirmationDecision.REJECT if control == TaskControl.CANCEL else
            classify_confirmation(text, destructive=destructive)
        )
        if decision == ConfirmationDecision.AMBIGUOUS and _is_task_resume_request(text) and not destructive:
            decision = ConfirmationDecision.CONFIRM
        if decision == ConfirmationDecision.REJECT:
            self.state.pending_tool_action = None
            self._finish_task("CANCELLED", clear_action=True)
            self.audit.record(
                "pending_tool_cancelled",
                turn_id=turn_id,
                intent=pending.action.intent,
                reason=pending.reason,
            )
            return ConversationToolResponse(True, "Entendido, no lo hago.", executed=False)
        if decision != ConfirmationDecision.CONFIRM:
            if destructive:
                self.audit.record(
                    "pending_tool_confirmation_ambiguous",
                    turn_id=turn_id,
                    pending_action_id=pending.pending_request_id,
                    risk="DELETE",
                )
                return ConversationToolResponse(
                    True,
                    "Necesito una confirmación clara antes de eliminar archivos. ¿Confirmas que quieres borrarlos?",
                    executed=False,
                )
            self.state.pending_tool_action = None
            self._finish_task("SUPERSEDED", clear_action=True)
            self.audit.record(
                "pending_tool_superseded",
                turn_id=turn_id,
                intent=pending.action.intent,
            )
            return None

        self.state.pending_tool_action = None
        self.audit.record(
            "pending_tool_resumed",
            original_turn_id=pending.origin_turn_id,
            confirmation_turn_id=turn_id,
            pending_request_id=pending.pending_request_id,
            pending_action_id=pending.pending_request_id,
            task_id=pending.task_id,
            plan_id=pending.plan_id,
            confirmation_expected=pending.confirmation_expected,
            pending_state="CONSUMED",
            intent=pending.action.intent,
            reason=pending.reason,
            preserved_content_characters=len(str(pending.action.arguments.get("content", ""))),
        )
        if pending.scopes_to_authorize:
            for scope in pending.scopes_to_authorize:
                self.settings.authorize_session_scope(scope)
            merged = self.state.authorized_scopes
            self.audit.record(
                "workspace_scope_authorized",
                turn_id=turn_id,
                scopes=merged,
            )
        return self._prepare_or_execute(
            pending.action,
            turn_id=turn_id,
            confirmed_effects=pending.confirmed_effects,
        )

    def _handle_retry(self, text: str, turn_id: str, *, force=False) -> ConversationToolResponse | None:
        if (not force and not _is_retry_request(text)) or self.state.last_failed_action is None:
            return None
        pending_task = self.state.pending_task
        if pending_task.failed_plan is None or (
            pending_task.failure_ref and not pending_task.failure_ref.retryable
        ):
            return ConversationToolResponse(True,
                "El fallo registrado no permite reintentar ese plan de forma segura.", executed=False)
        if pending_task.committed_effects:
            return ConversationToolResponse(True,
                "La tarea ya aplicó parte de sus cambios. No repetiré esos efectos; necesitamos un plan nuevo.", executed=False)
        if self.state.retry_count >= 1:
            self.audit.record(
                "conversation_retry_limited",
                turn_id=turn_id,
                retry_count=self.state.retry_count,
                failure_reason=self.state.last_error.get("error", ""),
            )
            return ConversationToolResponse(
                True,
                "La acción volvió a fallar y no haré más reintentos automáticos. Puedo explicarte el error o preparar un plan distinto.",
                executed=False,
                debug={"retry_count": self.state.retry_count, "retry_limited": True},
            )
        action = self.state.last_failed_action
        failure = pending_task.failure_ref
        if failure and failure.original_plan_ref:
            original = self.state.failure_plans.get(failure.original_plan_ref)
            if (original is None or plan_reference(original) != failure.original_plan_ref
                or original.task_id != failure.task_id or original.revision != failure.plan_revision):
                return ConversationToolResponse(True,
                    "El plan del fallo anterior ya no está disponible de forma verificable.", executed=False)
            pending_task.failed_plan = original
        pending_task.plan = pending_task.failed_plan
        pending_task.task_id = pending_task.plan.task_id
        pending_task.plan_revision = pending_task.plan.revision
        try:
            self.plan_adapter.revalidate(pending_task.plan)
        except ValueError:
            return ConversationToolResponse(True,
                "La tarea anterior ya no está autorizada y no la reintenté.", executed=False)
        self._plan_source = pending_task.plan.plan_source
        self.state.retry_count += 1
        pending_task.attempt_number += 1
        self.audit.record(
            "conversation_retry",
            turn_id=turn_id,
            retry_count=self.state.retry_count,
            task_id=pending_task.task_id, plan_revision=pending_task.plan_revision,
            attempt_number=pending_task.attempt_number,
            failure_id=failure.failure_id if failure else "",
            original_plan_ref=failure.original_plan_ref if failure else "",
            intent=action.intent,
            target_path=action.target_path,
        )
        # Re-enter authority/permission checks; a previous confirmation is not a grant.
        return self._resolve_and_prepare(action, turn_id=turn_id)

    @traced_stage("PERMISSION_CHECK")
    def _resolve_and_prepare(self, action: PlannedToolAction, *, turn_id: str) -> ConversationToolResponse:
        plan_errors = validate_structured_action_plan(action, self.registry)
        if plan_errors:
            self._finish_task("PLAN_INVALID", clear_action=False)
            self._record_failure(
                action,
                turn_id=turn_id,
                stage="PLANNING",
                error_code="PLAN_INVALID",
                safe_user_message="el plan estructurado no superó su validación local",
                retryable=False,
            )
            self.audit.record(
                "conversation_action_plan_invalid",
                turn_id=turn_id,
                task_id=action.task_spec.task_id if action.task_spec else "",
                plan_id=action.structured_action_plan.plan_id if action.structured_action_plan else "",
                errors=plan_errors,
                replan_count=self.state.replan_count,
            )
            return ConversationToolResponse(
                True,
                "No ejecuté la acción porque el plan no coincide con el formato o destino que pediste.",
                executed=False,
                debug={"plan_status": "PLAN_INVALID", "errors": plan_errors},
            )
        intention = CapabilityIntent(
            required_capabilities=action.capabilities,
            operation=action.operation,
            target_path=action.target_path,
        )
        resolution = self.resolver.resolve(intention)
        self.audit.record(
            "conversation_tool_intent",
            turn_id=turn_id,
            intent=action.intent,
            required_capability=action.capabilities,
            resolver_result=resolution.kind.value,
            selected_tool=resolution.tool_ids,
            arguments=action.arguments,
        )
        if resolution.kind == CapabilityResolutionKind.FORBIDDEN:
            return ConversationToolResponse(
                True,
                "No puedo modificar el núcleo protegido de Anydes.",
                resolution.kind,
                debug={"reason": resolution.reason},
            )
        if resolution.kind in {
            CapabilityResolutionKind.MISSING_CAPABILITY,
            CapabilityResolutionKind.TEMPORARY_SCRIPT,
            CapabilityResolutionKind.UNSUPPORTED,
        }:
            return ConversationToolResponse(
                True,
                _missing_capability_text(action.capabilities),
                resolution.kind,
                debug={"reason": resolution.reason},
            )
        reflection = self.reflection.reflect(action, resolution)
        self.audit.record(
            "conversation_reflection",
            turn_id=turn_id,
            intent=action.intent,
            approved=reflection.approved,
            reason=reflection.reason,
        )
        if not reflection.approved:
            return ConversationToolResponse(
                True,
                "No puedo ejecutar esa acción de forma segura.",
                resolution.kind,
                debug={"reason": reflection.reason},
            )
        return self._prepare_or_execute(action, turn_id=turn_id, resolution=resolution)

    def _prepare_or_execute(
        self,
        action: PlannedToolAction,
        *,
        turn_id: str,
        resolution: CapabilityResolution | None = None,
        confirmed_effects: tuple[PermissionEffect, ...] = (),
    ) -> ConversationToolResponse:
        if resolution is None:
            resolution = self.resolver.resolve(
                CapabilityIntent(
                    required_capabilities=action.capabilities,
                    operation=action.operation,
                    target_path=action.target_path,
                )
            )
        plan_errors = validate_structured_action_plan(action, self.registry)
        if plan_errors:
            self._finish_task("PLAN_INVALID", clear_action=False)
            self._record_failure(
                action,
                turn_id=turn_id,
                stage="PLANNING",
                error_code="PLAN_INVALID",
                safe_user_message="el plan estructurado dejó de ser válido",
                retryable=False,
            )
            return ConversationToolResponse(
                True,
                "No ejecuté la acción porque el plan estructurado dejó de ser válido.",
                resolution.kind,
                resolution.tool_ids,
                executed=False,
                debug={"plan_status": "PLAN_INVALID", "errors": plan_errors},
            )
        if action.intent == "rename_missing_name":
            self.state.pending_tool_action = PendingToolAction(
                action=action,
                reason="missing_rename_name",
                origin_turn_id=turn_id,
                pending_request_id=f"pending:{turn_id}:{uuid4().hex[:10]}",
                confirmation_expected=False,
                state="WAITING_ARGUMENT",
            )
            return ConversationToolResponse(
                True,
                "Claro. ¿Qué nombre quieres ponerle?",
                resolution.kind,
                resolution.tool_ids,
                executed=False,
            )
        missing = _missing_required_detail(action)
        if missing:
            stage = (
                "CONTENT_SPECIFICATION"
                if action.intent == "invalid_semantic_creation"
                else "PLANNING"
            )
            error_code = (
                "CONTENT_SPEC_UNRESOLVED"
                if action.intent == "invalid_semantic_creation"
                else "MISSING_REQUIRED_DETAIL"
            )
            self._record_failure(
                action,
                turn_id=turn_id,
                stage=stage,
                error_code=error_code,
                safe_user_message=missing,
                retryable=action.intent == "invalid_semantic_creation",
            )
            return ConversationToolResponse(True, missing, resolution.kind, resolution.tool_ids)

        forbidden, scopes = self._scope_requirements(action)
        if forbidden:
            return ConversationToolResponse(
                True,
                "No puedo acceder a esa ruta porque pertenece al núcleo protegido de Anydes.",
                CapabilityResolutionKind.FORBIDDEN,
                resolution.tool_ids,
                debug={"reason": forbidden},
            )
        if scopes:
            self.state.pending_tool_action = PendingToolAction(
                action=action,
                reason="authorize_scope",
                confirmed_effects=tuple(dict.fromkeys((
                    *confirmed_effects,
                    *(
                        (PermissionEffect.CREATE,)
                        if PermissionEffect.CREATE in self._confirmation_effects(action)
                        else ()
                    ),
                ))),
                scopes_to_authorize=scopes,
                origin_turn_id=turn_id,
                pending_request_id=f"pending:{turn_id}:{uuid4().hex[:10]}",
                task_id=action.task_spec.task_id if action.task_spec else "",
                plan_id=action.structured_action_plan.plan_id if action.structured_action_plan else "",
            )
            label = scopes[0]
            self.audit.record(
                "conversation_permission_pending",
                turn_id=turn_id,
                permission_decision="authorize_scope",
                scopes=scopes,
            )
            return ConversationToolResponse(
                True,
                f"Necesito acceso a esta carpeta para hacerlo: {_display_path(label)}. "
                "¿Quieres autorizarla para esta sesión y continuar?",
                resolution.kind,
                resolution.tool_ids,
            )

        if action.operation == "delete_collection_request":
            prepared = self._prepare_delete_collection(
                action,
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            if isinstance(prepared, ConversationToolResponse):
                return prepared
            return self._prepare_or_execute(
                prepared,
                turn_id=turn_id,
                resolution=resolution,
                confirmed_effects=confirmed_effects,
            )

        required_effects = tuple(
            effect for effect in self._confirmation_effects(action)
            if effect in _CONFIRM_EFFECTS and effect not in confirmed_effects
        )
        if required_effects:
            try:
                self.state.pending_task.plan = self.plan_adapter.compile(action,
                    source=self._plan_source, task_id=self.state.pending_task.task_id,
                    revision=self.state.pending_task.plan_revision)
            except ValueError:
                return ConversationToolResponse(True,
                    "No puedo pedir confirmación porque el plan ya no está autorizado.", executed=False)
            effect = required_effects[0]
            pending_confirmed = tuple(dict.fromkeys((*confirmed_effects, *required_effects)))
            self.state.pending_tool_action = PendingToolAction(
                action=action,
                reason=f"confirm_{effect.value.casefold()}",
                confirmed_effects=pending_confirmed,
                origin_turn_id=turn_id,
                pending_request_id=f"pending:{turn_id}:{uuid4().hex[:10]}",
                task_id=action.task_spec.task_id if action.task_spec else "",
                plan_id=action.structured_action_plan.plan_id if action.structured_action_plan else "",
            )
            prompt = _confirmation_text(effect, action)
            self.audit.record(
                "conversation_permission_pending",
                turn_id=turn_id,
                permission_decision=effect.value,
                intent=action.intent,
            )
            return ConversationToolResponse(True, prompt, resolution.kind, resolution.tool_ids)

        try:
            current = self.state.pending_task.plan
            same_action = current is not None and current.legacy_action == action
            plan = self.plan_adapter.compile(action,
                source=current.plan_source if same_action else self._plan_source,
                task_id=current.task_id if same_action else self.state.pending_task.task_id,
                revision=current.revision if same_action else self.state.pending_task.plan_revision)
            self.state.pending_task.plan = plan
            self.state.pending_task.task_id = plan.task_id
            results = self._execute(plan, turn_id=turn_id, confirmed_effects=confirmed_effects)
        except ValueError as exc:
            self._record_failure(action, turn_id=turn_id, stage="PLANNING",
                error_code=str(exc), safe_user_message="el plan ejecutable no superó la autoridad local",
                retryable=False)
            return ConversationToolResponse(True,
                "No ejecuté la acción porque su plan dejó de estar autorizado.", executed=False)
        self.last_results = tuple(results)
        response = self._response_from_results(action, results, resolution, turn_id=turn_id)
        if results:
            try:
                self.action_memory.record(
                    action,
                    results,
                    turn_id=turn_id,
                    verification=self.state.last_verification,
                    performed_success=response.executed,
                    workspace=self.state.active_workspace,
                    artifacts=self.state.recent_artifact_state,
                )
            except (OSError, TypeError, ValueError) as exc:
                self.audit.record(
                    "action_experience_memory_error",
                    turn_id=turn_id,
                    error_type=type(exc).__name__,
                )
        self.audit.record(
            "conversation_action_result",
            turn_id=turn_id,
            intent=action.intent,
            composition_plan=action.operation,
            action_result=[
                {
                    "tool_id": item.tool_id,
                    "capability": item.capability,
                    "status": item.status.value,
                    "error": item.error,
                    "duration_ms": item.duration_ms,
                }
                for item in results
            ],
            referent_resolution={
                "active_document": self.state.active_document,
                "active_document_ref": asdict(self.state.active_document_ref) if self.state.active_document_ref else None,
                "active_code_file": self.state.active_code_file,
            },
            retry_count=self.state.retry_count,
            verification_status=self.state.last_verification.get("status", ""),
            executor_called=bool(results),
            response_source=response.debug.get("response_source", "TOOL_RESULT"),
        )
        return response

    def _trace_dispatch(
        self,
        turn_id: str,
        action: PlannedToolAction,
        *,
        planner_selected: bool,
    ) -> None:
        plan = action.structured_action_plan
        self.audit.record(
            "conversation_runtime_trace",
            turn_id=turn_id,
            planner_selected=planner_selected,
            task_id=action.task_spec.task_id if action.task_spec else "",
            plan_id=plan.plan_id if plan else "",
            tool_id=plan.tool_id if plan else "",
            capability=plan.capability if plan else (action.capabilities[0] if action.capabilities else ""),
            executor_called=False,
            action_result_status="PENDING",
            response_source="STRUCTURED_PLANNER",
        )

    @traced_stage("EXECUTION_DISPATCH")
    def _execute(
        self,
        plan: ExecutableTaskPlan,
        *,
        turn_id: str,
        confirmed_effects: tuple[PermissionEffect, ...],
    ) -> tuple[ToolExecutionResult, ...]:
        self.plan_adapter.revalidate(plan)
        validate_dataflow(plan)
        self._executing_plan = plan
        self._artifact_bindings = ArtifactBindings(plan.task_id,
            plan.typed_inputs.get("artifact_bindings", {}))
        self.last_dataflow_summary = {}
        for path, operation in self._paths_for_action(plan.legacy_action):
            if path and not str(path).startswith("$"):
                decision = self.boundary.classify(path, operation=operation,
                    authorized_scopes=self.state.authorized_scopes)
                if decision.allowed:
                    self._artifact_bindings.snapshot(path)
        self.audit.record("executable_plan_dispatch", turn_id=turn_id,
            task_id=plan.task_id, plan_revision=plan.revision, plan_source=plan.plan_source,
            step_ids=[s.step_id for s in plan.steps], executor_called=True)
        try:
            results = self._execute_legacy_operation(plan.legacy_action,
                turn_id=turn_id, confirmed_effects=confirmed_effects)
            if plan.legacy_action.operation in {"generated_create_document", "general_generated_document"} and all(
                    r.status == ExecutionStatus.SUCCESS for r in results):
                completed = {r.step_id for r in self.last_step_results}
                for step in plan.steps:
                    if step.step_id not in completed:
                        extra = self._execute_flow_step(step, turn_id, confirmed_effects)
                        results = (*results, *extra)
                        if any(r.status != ExecutionStatus.SUCCESS for r in extra):
                            break
                        completed.add(step.step_id)
            known = {r.values.get("request_id") for r in self.last_step_results}
            for result in results:
                if result.request_id not in known:
                    match = next((s for s in plan.steps if s.capability_id == result.capability), None)
                    generation = next((s for s in plan.steps if s.step_type == "CONTENT_GENERATION"), None)
                    if generation and result.status != ExecutionStatus.SUCCESS and not all(
                            name in self._artifact_bindings.values for name in generation.produces_values):
                        match = generation
                    self.last_step_results.append(StepResult(
                        step_id=match.step_id if match else "legacy_verification",
                        status=result.status.value, values={"request_id": result.request_id},
                        error=result.error, verification={
                            **dict(self.state.last_verification),
                            **({"verification_evidence": result.output["verification_evidence"]}
                               if "verification_evidence" in result.output else {})}))
            self._complete_dataflow_results(plan, results, turn_id)
            missing = next((r for r in self.last_step_results if r.status == "NOT_EXECUTED"), None)
            if missing:
                results = (*results, self._synthetic_failure(turn_id, "", "DATAFLOW_STEP_NOT_EXECUTED:" + missing.step_id,
                    status=ExecutionStatus.INVALID_INPUT))
            if results:
                results = (*results[:-1], replace(results[-1], output={
                    **results[-1].output, "task_execution": self.last_dataflow_summary}))
            return results
        finally:
            self._executing_plan = None
            self._dataflow_step = None

    def _execute_flow_step(self, step, turn_id, confirmed_effects):
        self._dataflow_step = step
        try:
            args, refs = self._artifact_bindings.resolve_arguments(step)
            if step.capability_id.startswith("document.create") and refs:
                if len(refs) != 1:
                    raise DataflowError("DATAFLOW_CONVERSION_SINGLE_SOURCE_REQUIRED")
                source = self._artifact_bindings.require(step.consumes_artifacts[0], verified=True)
                read = self._execute_capability("document.read_document", {"path": source.path},
                    turn_id=turn_id, confirmed_effects=confirmed_effects)
                if read.status != ExecutionStatus.SUCCESS:
                    return (read,)
                text = str(read.output.get("content", ""))
                if not text.strip():
                    raise DataflowError("DATAFLOW_CONVERSION_SOURCE_EMPTY")
                format_id = {"document.create_pdf": "pdf", "document.create_docx": "docx",
                    "document.create_txt": "txt", "document.create_markdown": "markdown"}[step.capability_id]
                suffix = ".md" if format_id == "markdown" else "." + format_id
                final = str(Path(args.get("path") or Path(source.path).with_suffix(suffix)).resolve(strict=False))
                rendered = DocumentRenderer.render(document_spec_from_extracted_text(text), format_id)
                converted = self._stage_and_commit_document(final_path=final,
                    capability=step.capability_id, content=rendered.content, title=rendered.title,
                    turn_id=turn_id, confirmed_effects=confirmed_effects,
                    verify=lambda actual: asdict(verify_conversion(text, actual, final, format_id)),
                    require_nonempty=True, conversion_source=source)
                for binding in step.produces_artifacts:
                    ref = self._artifact_bindings.bindings.get(binding)
                    if ref:
                        related = replace(ref, metadata={**ref.metadata, "converted_from": source.artifact_id})
                        self._artifact_bindings.bindings[binding] = related
                        self._artifact_bindings.artifacts[ref.artifact_id] = related
                if all(r.status == ExecutionStatus.SUCCESS for r in converted):
                    self.state.remember_conversion(source.path, final)
                self._remember_results((read, *converted))
                return (read, *converted)
            if step.capability_id.startswith("document.create"):
                content = str(args.get("content", ""))
                if not content.strip():
                    raise DataflowError("DATAFLOW_CONTENT_EMPTY")
                final = str(Path(args["path"]).resolve(strict=False))
                title = str(args.get("title", ""))
                expected = (title + "\n" if title else "") + content
                target_format = "md" if step.capability_id == "document.create_markdown" else Path(final).suffix.lstrip(".")
                written = self._stage_and_commit_document(final_path=final,
                    capability=step.capability_id, content=content, title=title,
                    turn_id=turn_id, confirmed_effects=confirmed_effects,
                    verify=lambda actual: asdict(verify_conversion(expected, actual, final, target_format)),
                    require_nonempty=True)
                return tuple(written)
            if step.verifier_type == 'CODE' and step.capability_id in {'developer.run_python','developer.run_cpp','developer.test'}:
                from .verification_types import build_runtime_tests
                canonical = self._pipeline_trace.invoke('TESTSPEC_BUILD', build_runtime_tests,
                    [{'stdin':args.get('stdin','')}], semantic_cases=True, plan=self._executing_plan)
                self.audit.record('canonical_test_spec', builder='build_runtime_tests',
                    task_id=self._executing_plan.task_id, source_step=step.step_id,
                    artifact_id=refs[0].artifact_id if refs else '',
                    test_id=canonical[0].test_id, verification_requirement='EXECUTION')
            result = self._execute_capability(step.capability_id, args,
                turn_id=turn_id, confirmed_effects=confirmed_effects)
            if step.verifier_type == 'CODE' and step.capability_id in {'developer.run_python','developer.run_cpp','developer.test'}:
                typed = self._pipeline_trace.invoke('VERIFICATION_DISPATCH',VerificationDispatcher.verify,
                    'CODE',execution_only=True,exit_code=result.output.get('returncode'),
                    execution_succeeded=result.status == ExecutionStatus.SUCCESS,
                    artifact_id=refs[0].artifact_id if refs else '')
                self._remember_typed_verification(typed)
                self.audit.record('typed_verification',verification=typed.to_dict(),
                    verification_requirement='EXECUTION',oracle_called=False)
                for binding in step.consumes_artifacts:
                    self._artifact_bindings.require(binding,verified=True,allow_unverified=True)
                if typed.status != VerificationStatus.PASS:
                    return (result,self._synthetic_failure(turn_id,step.capability_id,typed.error_code,
                        status=ExecutionStatus.VERIFY_FAILED,output=typed.legacy()))
            return (result,)
        except (DataflowError, KeyError) as exc:
            reason = str(exc) if isinstance(exc, DataflowError) else "DATAFLOW_REQUIRED_ARGUMENT_MISSING"
            failed = self._synthetic_failure(turn_id, step.capability_id, reason, status=ExecutionStatus.INVALID_INPUT)
            self.last_step_results.append(StepResult(step.step_id, failed.status.value,
                error=reason, values={"request_id": failed.request_id}))
            return (failed,)
        finally:
            self._dataflow_step = None

    def _complete_dataflow_results(self, plan, results, turn_id):
        failed = next((r for r in self.last_step_results if r.status != "SUCCESS"), None)
        for step in plan.steps:
            if not any(r.step_id == step.step_id for r in self.last_step_results):
                status = "BLOCKED_DEPENDENCY" if failed else "NOT_EXECUTED"
                self.last_step_results.append(StepResult(step.step_id, status,
                    error="DATAFLOW_BLOCKED" if failed else "DATAFLOW_STEP_NOT_EXECUTED",
                    blocked_by_step_id=failed.step_id if failed else "",
                    blocked_artifact_id=(self._artifact_bindings.bindings[step.consumes_artifacts[0]].artifact_id
                        if step.consumes_artifacts and step.consumes_artifacts[0] in self._artifact_bindings.bindings
                        else step.consumes_artifacts[0] if step.consumes_artifacts else "")))
        for binding, ref in list(self._artifact_bindings.bindings.items()):
            if ref.state == ArtifactState.TEMPORARY and not Path(ref.path).exists():
                bad = replace(ref, state=ArtifactState.FAILED)
                self._artifact_bindings.bindings[binding] = bad
                self._artifact_bindings.artifacts[bad.artifact_id] = bad
        for index, result in enumerate(self.last_step_results):
            if result.status == "BLOCKED_DEPENDENCY":
                step = next(s for s in plan.steps if s.step_id == result.step_id)
                ref = self._artifact_bindings.artifacts.get(result.blocked_artifact_id)
                self.last_step_results[index] = replace(result, values={**result.values,
                    "required_state": "VERIFIED" if step.requires_verified_inputs else "AVAILABLE",
                    "actual_state": ref.state.value if ref else "MISSING"})
        refs = self._artifact_bindings.artifacts.values()
        for ref in refs:
            if ref.state in {ArtifactState.COMMITTED, ArtifactState.COMMITTED_UNVERIFIED} and ref.producer_step_id != "INPUT":
                self._operational_artifacts[ref.path] = ref
                recent = self.state.recent_artifact_state.register(ref)
                if recent:
                    self.audit.record("recent_artifact_registered", turn_id=turn_id,
                        artifact_id=ref.artifact_id, artifact_type=ref.artifact_type,
                        task_id=ref.task_id, producer_step_id=ref.producer_step_id,
                        version=ref.version, hash=ref.hash, state=ref.state.value,
                        binding_source="RUNTIME_COMMIT")
        while len(self._operational_artifacts) > 32:
            self._operational_artifacts.pop(next(iter(self._operational_artifacts)))
        structured = []
        for step in plan.steps:
            rows = [r for r in self.last_step_results if r.step_id == step.step_id]
            unsuccessful = next((r for r in rows if r.status != "SUCCESS"), None)
            selected = unsuccessful or rows[-1]
            structured.append(replace(selected,
                produced_artifacts=tuple(dict.fromkeys(a for r in rows for a in r.produced_artifacts)),
                consumed_artifacts=tuple(dict.fromkeys(a for r in rows for a in r.consumed_artifacts)),
                typed_values={k: v for r in rows for k, v in r.typed_values.items()},
                observed_effects=tuple(e for r in rows for e in r.observed_effects),
                values={**selected.values, "request_ids": [r.values.get("request_id") for r in rows if r.values.get("request_id")]},
                verification=unsuccessful.verification if unsuccessful else rows[-1].verification))
        self.last_dataflow_summary = {
            "task_id": plan.task_id,
            "completed_steps": [r.step_id for r in structured if r.status == "SUCCESS"],
            "failed_step": failed.step_id if failed else "",
            "blocked_steps": [r.step_id for r in self.last_step_results if r.status == "BLOCKED_DEPENDENCY"],
            "committed_artifacts": [asdict(a) for a in refs if a.state in {ArtifactState.COMMITTED, ArtifactState.COMMITTED_UNVERIFIED} and a.producer_step_id != "INPUT"],
            "step_results": [asdict(r) for r in structured],
            "artifact_versions": {a.artifact_id: a.version for a in self._artifact_bindings.artifacts.values()},
            "artifact_states": {a.artifact_id: a.state.value for a in self._artifact_bindings.artifacts.values()},
            "temporary_artifacts": [asdict(a) for a in refs if a.state == ArtifactState.TEMPORARY],
            "failed_artifacts": [asdict(a) for a in refs if a.state == ArtifactState.FAILED],
            "removed_artifact_ids": [a.artifact_id for a in refs
                if a.state == ArtifactState.FAILED and not Path(a.path).exists()],
            "quarantined_artifacts": [],  # This transaction path removes failed temps, not quarantine.
        }
        self.audit.record("task_dataflow_outcome", turn_id=turn_id, task_id=plan.task_id,
            completed_steps=self.last_dataflow_summary["completed_steps"],
            failed_step=self.last_dataflow_summary["failed_step"],
            blocked_steps=self.last_dataflow_summary["blocked_steps"],
            committed_artifact_ids=[a["artifact_id"] for a in self.last_dataflow_summary["committed_artifacts"]])
        self.last_step_results = structured

    def _execute_legacy_operation(self, action, *, turn_id, confirmed_effects):
        """Temporary operation adapter; existing verification semantics are unchanged."""
        self.state.last_verification = {}
        if action.operation == 'retain_unverified':
            draft = self._retained_draft
            if not draft or draft['task_id'] != self.state.pending_task.task_id or time.time()-draft['created_at'] > 1800:
                raise ValueError('QUALITY_RETENTION_EXPIRED')
            binding = self._executing_plan.steps[0].produces_artifacts[0]
            ref = self._artifact_bindings.produce(binding, draft['source'], self._executing_plan.steps[0].step_id,
                artifact_type='CODE', metadata={'user_accepted_unverified':True,
                    'verification_result':draft['verification'], 'source_artifact_id':draft['artifact_id'],
                    'task_goal':draft.get('task_goal',''), 'failure_ref':draft.get('failure_id','')})
            if ref.hash != draft['hash']:
                raise DataflowError('STALE_ARTIFACT:quality_retention')
            from .dataflow import path_version
            if path_version(draft['destination']) != draft['expected_destination_hash']:
                raise DataflowError('STALE_ARTIFACT:quality_destination')
            self._artifact_bindings.path_snapshots[str(Path(draft['destination']).resolve(strict=False))] = draft['expected_destination_hash']
            result = self._execute_capability('filesystem.atomic_replace', {
                'source':draft['source'], 'destination':draft['destination'],
                'commit_effect':'OVERWRITE' if Path(draft['destination']).exists() else 'CREATE'},
                turn_id=turn_id, confirmed_effects=confirmed_effects)
            if result.status == ExecutionStatus.SUCCESS:
                ref = self._artifact_bindings.bindings[binding]
                ref = replace(ref, state=ArtifactState.COMMITTED_UNVERIFIED)
                self._artifact_bindings.bindings[binding] = ref
                self._artifact_bindings.artifacts[ref.artifact_id] = ref
                self._retained_draft = None
                owned = Path(draft['source']).parent
                if owned.name == '.anydes-drafts' and not any(owned.iterdir()):
                    owned.rmdir()
                self.state.last_verification = {'status':'USER_ACCEPTED_UNVERIFIED'}
            return (result,)
        if action.operation == "document_conversion":
            return self._execute_document_conversion(action, turn_id, confirmed_effects)
        if action.operation == "structural_document_edit":
            return self._execute_structural_document_edit(action, turn_id, confirmed_effects)
        if action.operation == "rename_existing":
            return self._execute_rename(action, turn_id, confirmed_effects)
        if action.operation == "copy_existing":
            return self._execute_copy(action, turn_id, confirmed_effects)
        if action.operation == "text_files_to_pdf":
            return self._execute_text_files_to_pdf(action, turn_id, confirmed_effects)
        if action.operation == "append_document":
            return self._execute_append(action, turn_id, confirmed_effects)
        if action.operation == "verified_create_document":
            return self._execute_verified_create(action, turn_id, confirmed_effects)
        if action.operation == "generated_create_document":
            return self._execute_generated_create(action, turn_id, confirmed_effects)
        if action.operation == "general_generated_document":
            return self._execute_general_generated_document(action, turn_id, confirmed_effects)
        if action.operation == "generated_document_edit":
            return self._execute_generated_edit(action, turn_id, confirmed_effects)
        if action.operation == "structured_edit_text":
            return self._execute_structured_edit(action, turn_id, confirmed_effects)
        if action.operation == "document_transform":
            return self._execute_document_transform(action, turn_id, confirmed_effects)
        if action.operation == "delete_collection":
            return self._execute_delete_collection(action, turn_id, confirmed_effects)
        if action.operation == "run_inline_python":
            return self._execute_inline_python(action, turn_id, confirmed_effects)
        if action.operation == "general_generate_code":
            return self._execute_general_code(action, turn_id, confirmed_effects)
        if action.operation == "general_modify_code":
            return self._execute_general_code_modification(action, turn_id, confirmed_effects)
        if action.operation == "general_tool_composition":
            return self._execute_general_composition(action, turn_id, confirmed_effects)
        capability = action.capabilities[0]
        result = self._execute_capability(
            capability,
            action.arguments,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        self._remember_results((result,))
        return (result,)

    def _execute_copy(self, action, turn_id, confirmed_effects):
        source = Path(str(action.arguments["source"])).resolve(strict=False)
        destination = Path(str(action.arguments["destination"])).resolve(strict=False)
        try:
            before_hash = _file_hash(source)
        except OSError as exc:
            failure = self._synthetic_failure(
                turn_id,
                "filesystem.copy",
                f"copy_source_unreadable:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        copied = self._execute_capability(
            "filesystem.copy",
            action.arguments,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results = [copied]
        if copied.status == ExecutionStatus.SUCCESS:
            try:
                after_hash = _file_hash(destination)
                verified = destination.is_file() and before_hash == after_hash
            except OSError:
                after_hash = ""
                verified = False
            verification = {
                "status": "VERIFIED" if verified else "VERIFY_FAILED",
                "source_hash": before_hash,
                "target_hash": after_hash,
                "target_exists": destination.is_file(),
            }
            self.state.last_verification = verification
            results[-1] = replace(copied, output={**copied.output, **verification})
            if not verified:
                results.append(self._synthetic_failure(
                    turn_id,
                    "filesystem.copy",
                    "copy_verification_failed",
                    status=ExecutionStatus.VERIFY_FAILED,
                    output=verification,
                ))
        self._remember_results(results)
        return tuple(results)

    def _execute_verified_create(self, action, turn_id, confirmed_effects):
        expected = str(action.arguments.get("content", ""))
        file_type = Path(action.arguments["path"]).suffix.casefold().removeprefix(".")
        specification = dict(action.arguments.get("creation_spec", {}))
        results = self._stage_and_commit_document(
            final_path=str(action.arguments["path"]),
            capability=action.capabilities[0],
            content=expected,
            title="",
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: verify_creation(actual, expected, file_type, specification),
            allow_repair=True,
        )
        self._remember_results(results)
        return tuple(results)

    def _prepare_delete_collection(self, action, *, turn_id, confirmed_effects):
        root = str(action.arguments.get("root", self.state.active_workspace))
        pattern = str(action.arguments.get("pattern", "*"))
        search = self._execute_capability(
            "filesystem.search",
            {"root": root, "pattern": pattern, "limit": 2000},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        if search.status != ExecutionStatus.SUCCESS:
            self.last_results = (search,)
            return ConversationToolResponse(
                True,
                _friendly_failure(search),
                executed=False,
                debug={"response_source": "DELETE_COLLECTION_ENUMERATION_FAILED"},
            )
        request = DeleteSelectionRequest(
            scope=DeleteScope(str(action.arguments.get("delete_scope", DeleteScope.COLLECTION.value))),
            workspace=root,
            pattern=pattern,
            recursive=bool(action.arguments.get("recursive", False)),
        )
        try:
            delete_plan = CollectionResolver.resolve(
                request,
                tuple(str(item) for item in search.output.get("matches", ())),
            )
        except (OSError, TypeError, ValueError) as exc:
            failure = self._synthetic_failure(
                turn_id,
                "filesystem.search",
                f"delete_collection_resolution_failed:{type(exc).__name__}",
            )
            self.last_results = (search, failure)
            return ConversationToolResponse(
                True,
                "No pude preparar de forma segura la lista de archivos que se eliminarían.",
                executed=False,
                debug={"response_source": "DELETE_COLLECTION_PLAN_FAILED"},
            )
        if not delete_plan.targets:
            self.last_results = (search,)
            return ConversationToolResponse(
                True,
                "No encontré archivos que coincidan con esa selección en el workspace activo.",
                executed=False,
                debug={"response_source": "DELETE_COLLECTION_EMPTY"},
            )
        prepared = replace(
            action,
            operation="delete_collection",
            arguments={"delete_plan": delete_plan.to_dict()},
            target_path=delete_plan.workspace,
        )
        self.audit.record(
            "conversation_delete_plan_prepared",
            turn_id=turn_id,
            delete_plan_id=delete_plan.plan_id,
            scope=delete_plan.scope.value,
            selection_rule=delete_plan.selection_rule,
            target_count=len(delete_plan.targets),
            targets=[item.absolute_canonical_path for item in delete_plan.targets],
        )
        return prepared

    def _execute_delete_collection(self, action, turn_id, confirmed_effects):
        try:
            plan = DeletePlan.from_dict(dict(action.arguments["delete_plan"]))
        except (KeyError, TypeError, ValueError) as exc:
            failure = self._synthetic_failure(
                turn_id,
                "filesystem.delete",
                f"delete_plan_invalid:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        if PermissionEffect.DELETE not in confirmed_effects:
            failure = self._synthetic_failure(
                turn_id,
                "filesystem.delete",
                "delete_collection_confirmation_missing",
                status=ExecutionStatus.DENIED,
            )
            self._remember_results((failure,))
            return (failure,)
        plan = replace(plan, confirmed=True)
        results: list[ToolExecutionResult] = []
        deleted: list[str] = []
        failed: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        for target in plan.targets:
            if target.artifact_kind in {
                DeleteArtifactKind.APPLICATION_LOCK,
                DeleteArtifactKind.SYSTEM_TEMP,
                DeleteArtifactKind.TRANSACTION_TEMP,
            }:
                skipped.append({
                    "path": target.absolute_canonical_path,
                    "reason": target.artifact_kind.value,
                })
                continue
            if not target_matches_snapshot(target):
                failure = self._synthetic_failure(
                    turn_id,
                    "filesystem.delete",
                    "delete_target_changed_since_confirmation",
                    status=ExecutionStatus.VERIFY_FAILED,
                    output={"path": target.absolute_canonical_path},
                )
                results.append(failure)
                failed.append({
                    "path": target.absolute_canonical_path,
                    "reason": failure.error,
                })
                continue
            result = self._execute_capability(
                "filesystem.delete",
                {"path": target.absolute_canonical_path},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(result)
            if result.status == ExecutionStatus.SUCCESS and not Path(target.absolute_canonical_path).exists():
                deleted.append(target.absolute_canonical_path)
            else:
                failed.append({
                    "path": target.absolute_canonical_path,
                    "reason": result.error or "delete_verification_failed",
                })
        accounted = len(deleted) + len(failed) + len(skipped)
        status = "VERIFY_FAILED"
        if accounted == len(plan.targets):
            status = "VERIFIED" if not failed and not skipped else "VERIFIED_PARTIAL"
        batch = BatchActionResult(
            deleted=tuple(deleted),
            failed=tuple(failed),
            skipped=tuple(skipped),
            status=status,
        )
        verification = {
            "status": status,
            "delete_plan_id": plan.plan_id,
            "expected_deletions": len(plan.targets),
            "actual_deletions": len(deleted),
            "deleted": deleted,
            "failed": failed,
            "skipped": skipped,
            "targets_accounted": accounted == len(plan.targets),
            "batch_result": batch.to_dict(),
        }
        self.state.last_verification = verification
        self.audit.record(
            "conversation_delete_verification",
            turn_id=turn_id,
            **verification,
        )
        if status == "VERIFY_FAILED":
            results.append(self._synthetic_failure(
                turn_id,
                "filesystem.delete",
                "delete_collection_verification_failed",
                status=ExecutionStatus.VERIFY_FAILED,
                output=verification,
            ))
        self._remember_results(results)
        return tuple(results)

    def _execute_generated_create(self, action, turn_id, confirmed_effects):
        trace = {
            "planner_selected": True,
            "task_type": action.task_spec.goal if action.task_spec else "",
            "content_kind": "",
            "source_context_type": "",
            "self_context_attached": False,
            "brain_called": False,
            "brain_result_length": 0,
            "content_validator_result": "NOT_RUN",
            "content_validator_reason": "",
            "document_spec_created": False,
            "renderer_called": False,
            "action_result": "NOT_CALLED",
        }

        def record(stage: str) -> None:
            self.audit.record(
                "conversation_content_pipeline_trace",
                turn_id=turn_id,
                stage=stage,
                task_id=action.task_spec.task_id if action.task_spec else "",
                **trace,
            )

        if self.semantic_prose_generator is None:
            trace["action_result"] = "CONTENT_GENERATOR_UNAVAILABLE"
            record("CONTENT_GENERATION")
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                "semantic_prose_generator_unavailable",
            )
            self._remember_results((failure,))
            return (failure,)
        try:
            specification = ContentGenerationSpec.from_dict(
                dict(action.arguments["content_generation_spec"])
            )
            trace["content_kind"] = specification.content_kind
            trace["source_context_type"] = specification.source_mode
        except (KeyError, TypeError, ValueError) as exc:
            trace["action_result"] = f"CONTENT_SPECIFICATION_FAILED:{type(exc).__name__}"
            record("CONTENT_SPECIFICATION")
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                f"content_generation_failed:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        try:
            prepare = getattr(self.semantic_prose_generator, "prepare_specification", None)
            if callable(prepare):
                specification = prepare(specification)
            trace["self_context_attached"] = bool(
                specification.source_mode == "SELF_CONTEXT"
                and specification.source_context.strip()
            )
            if (
                specification.source_mode == "SELF_CONTEXT"
                and callable(prepare)
                and not trace["self_context_attached"]
            ):
                raise RuntimeError("self_context_unavailable")
            record("SELF_CONTEXT")
        except (TypeError, ValueError, RuntimeError) as exc:
            trace["action_result"] = f"SELF_CONTEXT_UNAVAILABLE:{type(exc).__name__}"
            record("SELF_CONTEXT")
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                "self_context_unavailable",
            )
            self._remember_results((failure,))
            return (failure,)
        try:
            trace["brain_called"] = True
            content = str(self.semantic_prose_generator(specification)).strip()
            trace["brain_result_length"] = len(content)
            record("CONTENT_GENERATION")
        except (TypeError, ValueError, RuntimeError) as exc:
            generation_error = _semantic_generation_error(exc)
            trace["action_result"] = generation_error
            record("CONTENT_GENERATION")
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                generation_error,
            )
            self._remember_results((failure,))
            return (failure,)
        validation = ContentValidator.validate(content, specification)
        grounding = verify_grounding([], {"source_context": specification.source_context},
            required=specification.source_mode in {"SELF_CONTEXT", "USER_CONTEXT",
                "FACTUAL_FROM_AVAILABLE_CONTEXT", "FACTUAL_REQUIRES_RETRIEVAL"}
                or "require_factual_grounding" in specification.constraints)
        self.audit.record("content_grounding_verification", **grounding.to_dict())
        if grounding.status == VerificationStatus.INCONCLUSIVE:
            failure = self._synthetic_failure(turn_id, action.capabilities[0], grounding.error_code,
                status=ExecutionStatus.VERIFY_FAILED, output=grounding.legacy())
            self._remember_results((failure,))
            return (failure,)
        trace["content_validator_result"] = validation.status
        trace["content_validator_reason"] = validation.reason
        record("CONTENT_VALIDATION")
        self.audit.record(
            "conversation_generated_content_validation",
            turn_id=turn_id,
            **asdict(validation),
        )
        if validation.status != "VALID":
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                validation.reason,
                status=ExecutionStatus.VERIFY_FAILED,
                output=asdict(validation),
            )
            self._remember_results((failure,))
            return (failure,)
        if self._artifact_bindings is not None:
            generation = next((s for s in self._executing_plan.steps if s.step_type == "CONTENT_GENERATION"), None)
            value_ids = generation.produces_values if generation else ("content.main",)
            for value_id in value_ids:
                self._artifact_bindings.values[value_id] = content
            self.last_step_results.append(StepResult("content_generation", "SUCCESS",
                typed_values={value_id: {"type": "CONTENT", "characters": len(content),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()} for value_id in value_ids}))
        try:
            document = build_document_spec(
                content,
                specification,
                requested_title=str(action.arguments.get("document_title", "")),
                request_text=action.source_text,
            )
            if action.task_spec and action.task_spec.output_format in {"txt", "markdown"}:
                document = replace(document, include_title=False)
            trace["document_spec_created"] = True
            record("DOCUMENT_SPEC")
            trace["renderer_called"] = True
            rendered = DocumentRenderer.render(
                document,
                action.task_spec.output_format if action.task_spec else "txt",
            )
            record("RENDER")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            trace["action_result"] = f"RENDER_FAILED:{type(exc).__name__}"
            record("RENDER")
            failure = self._synthetic_failure(
                turn_id,
                action.capabilities[0],
                f"document_render_failed:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        adherence = lambda: self._content_adherence(action.source_text, specification, content)
        results = self._stage_and_commit_document(
            final_path=str(action.arguments["path"]),
            capability=action.capabilities[0],
            content=rendered.content,
            title=rendered.title,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: {**PostActionVerifier.generated_creation(
                actual,
                rendered.expected_extractable_text,
                specification,
                generated_body=content,
            ), "task_adherence": adherence},
            require_nonempty=True,
        )
        trace["action_result"] = ",".join(item.status.value for item in results)
        record("ACTION_RESULT")
        self._remember_results(results)
        return tuple(results)

    def _execute_general_generated_document(self, action, turn_id, confirmed_effects):
        source_path = str(action.arguments.get("source_path", ""))
        if not source_path:
            failure = self._synthetic_failure(
                turn_id,
                "document.read_document",
                "general_document_source_missing",
            )
            self._remember_results((failure,))
            return (failure,)
        source = self._execute_capability(
            "document.read_document",
            {"path": source_path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        if source.status != ExecutionStatus.SUCCESS:
            self._remember_results((source,))
            return (source,)
        source_content = str(source.output.get("content", "")).strip()
        if not source_content:
            failure = self._synthetic_failure(
                turn_id,
                "document.read_document",
                "general_document_source_empty",
                status=ExecutionStatus.VERIFY_FAILED,
            )
            self._remember_results((source, failure))
            return (source, failure)
        specification = ContentGenerationSpec.from_dict(
            dict(action.arguments.get("content_generation_spec", {}))
        ).with_source_context(
            "PROVENANCE: TRANSFORM_EXISTING\n" + source_content[:40_000]
        )
        lowered = replace(
            action,
            operation="generated_create_document",
            capabilities=(action.structured_action_plan.capability,),
            arguments={
                **action.arguments,
                "content_generation_spec": specification.to_dict(),
            },
        )
        generated = self._execute_generated_create(lowered, turn_id, confirmed_effects)
        return (source, *generated)

    def _execute_generated_edit(self, action, turn_id, confirmed_effects):
        path = str(action.arguments["path"])
        read = self._execute_capability(
            "document.read_document",
            {"path": path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read]
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)
        if self.semantic_prose_generator is None:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                "semantic_prose_generator_unavailable",
            ))
            self._remember_results(results)
            return tuple(results)
        before = str(read.output.get("content", ""))
        try:
            specification = ContentGenerationSpec.from_dict(
                dict(action.arguments["content_generation_spec"])
            ).with_source_context(before)
            generated = str(self.semantic_prose_generator(specification)).strip()
            validation = ContentValidator.validate(generated, specification)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                _semantic_generation_error(exc),
            ))
            self._remember_results(results)
            return tuple(results)
        if validation.status != "VALID":
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                validation.reason,
                status=ExecutionStatus.VERIFY_FAILED,
                output=asdict(validation),
            ))
            self._remember_results(results)
            return tuple(results)
        adherence = lambda: self._content_adherence(action.source_text, specification, generated)
        mode = str(action.arguments.get("mode", "APPEND"))
        expected = generated if mode == "REPLACE" else before.rstrip() + "\n\n" + generated + "\n"
        staged = self._stage_edit_and_commit_document(
            final_path=path,
            before=before,
            expected=expected,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: {
                **PostActionVerifier.generated_edit(
                    before,
                    actual,
                    expected,
                    generated,
                ),
                "word_count": validation.word_count,
                "task_adherence": adherence,
            },
        )
        results.extend(staged)
        self._remember_results(results)
        return tuple(results)

    def _execute_structured_edit(self, action, turn_id, confirmed_effects):
        path = str(action.arguments["path"])
        read = self._execute_capability(
            "document.read_document",
            {"path": path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read]
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)

        original = str(read.output.get("content", ""))
        try:
            specification = TextTransformSpec.from_payload(dict(action.arguments["transform"]))
            outcome = transform_text(original, specification)
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                f"structured_transform_invalid:{type(exc).__name__}",
            ))
            self._remember_results(results)
            return tuple(results)

        if outcome.expected_matches == 0 or not outcome.changed:
            verification = verify_transform(original, original, outcome)
            self.state.last_verification = verification
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                "structured_transform_no_changes",
                status=ExecutionStatus.VERIFY_FAILED,
                output=verification,
            ))
            self._remember_results(results)
            return tuple(results)

        staged = self._stage_edit_and_commit_document(
            final_path=path,
            before=original,
            expected=outcome.content,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: verify_transform(actual, outcome.content, outcome),
        )
        results.extend(staged)
        self._remember_results(results)
        return tuple(results)

    def _execute_structural_document_edit(self, action, turn_id, confirmed_effects):
        path = str(action.arguments["path"])
        read = self._execute_capability(
            "document.read_document",
            {"path": path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read]
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)
        before = str(read.output.get("content", ""))
        try:
            specification = DocumentEditSpec.from_dict(dict(action.arguments["edit_spec"]))
            model = DocumentModel.from_extracted_text(before)
            expected = apply_document_edit(model, specification).render_text()
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                f"document_edit_spec_invalid:{type(exc).__name__}",
            ))
            self._remember_results(results)
            return tuple(results)
        staged = self._stage_edit_and_commit_document(
            final_path=path,
            before=before,
            expected=expected,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: verify_document_edit(
                before, actual, expected, specification
            ),
            document_edit_spec=specification.to_dict(),
        )
        results.extend(staged)
        self._remember_results(results)
        return tuple(results)

    def _execute_rename(self, action, turn_id, confirmed_effects):
        source = Path(str(action.arguments["path"])).resolve(strict=False)
        destination = source.with_name(str(action.arguments["new_name"]))
        renamed = self._execute_capability(
            "filesystem.rename",
            action.arguments,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [renamed]
        if renamed.status == ExecutionStatus.SUCCESS:
            inspected = self._execute_capability(
                "filesystem.inspect",
                {"path": str(destination)},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(inspected)
            passed = (
                inspected.status == ExecutionStatus.SUCCESS
                and destination.is_file()
                and not source.exists()
                and destination.suffix.casefold() == source.suffix.casefold()
            )
            self.state.last_verification = {
                "status": "VERIFIED" if passed else "VERIFY_FAILED",
                "source_absent": not source.exists(),
                "destination_exists": destination.is_file(),
                "format_preserved": destination.suffix.casefold() == source.suffix.casefold(),
            }
            if not passed:
                results.append(self._synthetic_failure(
                    turn_id,
                    "filesystem.rename",
                    "rename_verification_failed",
                    status=ExecutionStatus.VERIFY_FAILED,
                    output=self.state.last_verification,
                ))
        self._remember_results(results)
        return tuple(results)

    def _execute_document_transform(self, action, turn_id, confirmed_effects):
        path = str(action.arguments["path"])
        read = self._execute_capability(
            "document.read_document",
            {"path": path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read]
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)
        if self.semantic_evaluator is None:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                "semantic_evaluator_unavailable",
            ))
            self._remember_results(results)
            return tuple(results)
        before = str(read.output.get("content", ""))
        try:
            specification = DocumentTransformationSpec.from_dict(
                dict(action.arguments["transformation"])
            )
            patch_plan = build_semantic_patch_plan(
                before,
                specification,
                self.semantic_evaluator,
            )
            expected = apply_patch_plan(before, patch_plan)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                f"semantic_transform_plan_failed:{type(exc).__name__}",
            ))
            self._remember_results(results)
            return tuple(results)
        self.audit.record(
            "conversation_patch_plan",
            turn_id=turn_id,
            target_path=path,
            before_hash=patch_plan.before_hash,
            expected_changes=patch_plan.expected_changes,
            selected_line_ids=patch_plan.selected_line_ids,
        )
        staged = self._stage_edit_and_commit_document(
            final_path=path,
            before=before,
            expected=expected,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: asdict(verify_patch_plan(
                before,
                actual,
                expected,
                patch_plan,
            )),
        )
        results.extend(staged)
        self._remember_results(results)
        return tuple(results)

    def _execute_document_conversion(self, action, turn_id, confirmed_effects):
        try:
            specification = DocumentConversionSpec.from_dict(
                dict(action.arguments["conversion_spec"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            failure = self._synthetic_failure(
                turn_id,
                "document.read_document",
                f"document_conversion_invalid:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        read = self._execute_capability(
            "document.read_document",
            {"path": specification.source_path},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read]
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)
        source_text = str(read.output.get("content", ""))
        if not source_text.strip():
            results.append(self._synthetic_failure(
                turn_id,
                action.structured_action_plan.capability,
                "document_conversion_source_empty",
                status=ExecutionStatus.VERIFY_FAILED,
            ))
            self._remember_results(results)
            return tuple(results)
        rendered = DocumentRenderer.render(
            document_spec_from_extracted_text(source_text),
            specification.target_format,
        )
        source_ref = next((r for r in self._artifact_bindings.bindings.values()
            if r.path == specification.source_path), None) if self._artifact_bindings else None
        if source_ref is None and self._artifact_bindings is not None:
            source_ref = self._artifact_bindings.produce("conversion.source", specification.source_path,
                "INPUT", state=ArtifactState.COMMITTED, artifact_type="DOCUMENT")
        converted = self._stage_and_commit_document(
            final_path=specification.target_path,
            capability=action.structured_action_plan.capability,
            content=rendered.content,
            title=rendered.title,
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            verify=lambda actual: asdict(verify_conversion(
                source_text,
                actual,
                specification.target_path,
                specification.target_format,
            )),
            require_nonempty=True, conversion_source=source_ref,
        )
        results.extend(converted)
        if self.state.last_verification.get("status") == "VERIFIED":
            self.state.remember_conversion(
                specification.source_path,
                specification.target_path,
            )
            self.audit.record(
                "conversation_document_conversion",
                turn_id=turn_id,
                converted_from=specification.source_path,
                converted_to=specification.target_path,
                content_similarity=self.state.last_verification.get("content_similarity", 0.0),
            )
        self._remember_results(results)
        return tuple(results)

    def _remember_typed_verification(self, typed):
        evidence = typed.evidence
        self.state.verification_evidence[typed.verification_id] = VerificationEvidence(
            typed.verification_id, typed.verifier_type, typed.failed_check, typed.error_code,
            evidence.get("input", {}), (), evidence.get("actual", evidence.get("text", {})),
            evidence.get("exit_code"), {}, {}, "", typed.to_dict())
        while len(self.state.verification_evidence) > 64:
            self.state.verification_evidence.pop(next(iter(self.state.verification_evidence)))

    def _content_adherence(self, goal, specification, content):
        context = ""
        if self.safe_context_provider is not None:
            try:
                context = str(self.safe_context_provider("CONVERSATION_CONTEXT", goal))[:3000]
                context += "\nASSISTANT_CONTEXT:\n" + str(self.safe_context_provider("SELF_CONTEXT", goal))[:3000]
            except (RuntimeError, ValueError, TypeError, AttributeError):
                context = ""
        task_id = self._executing_plan.task_id if self._executing_plan else ""
        # Pre/post-commit checks inspect the same immutable content contract. Only
        # physical checks run again; semantic judgment must not change by chance.
        key = hashlib.sha256(json.dumps({"task_id":task_id, "goal":goal,
            "specification":asdict(specification), "content":content, "context":context},
            sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        if key in self._adherence_results:
            result = self._adherence_results[key]
            self.audit.record('task_adherence_reused', task_id=task_id,
                adherence_id=result.adherence_id, content_hash=result.content_hash)
            return result
        result = evaluate_adherence(self.task_adherence_evaluator, goal=goal,
            specification=specification, content=content,
            task_id=task_id, context=context)
        self._adherence_results[key] = result
        while len(self._adherence_results) > 64:
            self._adherence_results.pop(next(iter(self._adherence_results)))
        self.audit.record("task_adherence", **result.to_dict())
        return result

    @traced_stage("VERIFICATION_DISPATCH")
    def _verify_staged_artifact(self, path, capability, legacy_verify, actual, conversion_source=None):
        """Semantic constraints become an explicit check beside typed physical checks."""
        legacy = dict(legacy_verify(actual))
        formats = {"document.create_docx": "docx", "document.create_pdf": "pdf",
            "document.create_txt": "txt", "document.create_markdown": "md"}
        format_id = formats.get(capability)
        ref = next((r for r in self._artifact_bindings.artifacts.values() if r.path == str(path)), None) if self._artifact_bindings else None
        if capability == "document.edit_document" and ref is not None:
            format_id = ref.format
        if conversion_source is not None and ref is not None:
            typed = VerificationDispatcher.verify("CONVERSION", source_ref=conversion_source, target_ref=ref,
                expected_source_id=conversion_source.artifact_id)
        else:
            typed = VerificationDispatcher.verify("PDF" if format_id == "pdf" else "DOCUMENT",
                path=path, format_id=format_id, artifact_id=ref.artifact_id if ref else "")
        layers = [{"validator_type": "STRUCTURAL", "status": typed.status.value,
            "evidence": dict(typed.checks), "reason": typed.error_code}]
        layers.append({"validator_type": "QUANTITATIVE_CONTRACT", "status":
            "PASS" if legacy.get("status") == "VERIFIED" else "FAIL",
            "evidence": {k: legacy[k] for k in ("word_count", "minimum_words", "maximum_words", "content_match") if k in legacy},
            "reason": legacy.get("reason", "")})
        if typed.status != VerificationStatus.FAIL and legacy.get("status") != "VERIFIED":
            typed = replace(typed, status=VerificationStatus.FAIL, failed_check="semantic_constraints",
                checks={**typed.checks, "semantic_constraints": False}, error_code="SEMANTIC_CONSTRAINT_FAILED")
        adherence = legacy.pop("task_adherence", None)
        if callable(adherence):
            adherence = adherence() if typed.status == VerificationStatus.PASS else None
        if adherence is not None:
            layers.append({"validator_type": "TASK_ADHERENCE", "status": adherence.status.value,
                "evidence": adherence.to_dict(), "reason": adherence.reason_code})
            typed = aggregate_adherence(typed, adherence)
        typed = replace(typed, evidence={**typed.evidence, "validation_layers": layers})
        self.audit.record("typed_verification", task_id=self._executing_plan.task_id if self._executing_plan else "",
            **typed.to_dict())
        self._remember_typed_verification(typed)
        return {**legacy, **typed.legacy()}

    def _stage_and_commit_document(
        self,
        *,
        final_path: str,
        capability: str,
        content: str,
        title: str,
        turn_id: str,
        confirmed_effects: tuple[PermissionEffect, ...],
        verify,
        allow_repair: bool = False,
        commit_effect: PermissionEffect | None = None,
        require_nonempty: bool = False,
        conversion_source=None,
    ) -> list[ToolExecutionResult]:
        final = Path(final_path).resolve(strict=False)
        if require_nonempty and not str(content).strip():
            self.state.last_verification = {
                "status": "VERIFY_FAILED",
                "reason": "staged_content_empty",
                "atomic_commit": False,
            }
            return [self._synthetic_failure(
                turn_id,
                capability,
                "generated_content_empty",
                status=ExecutionStatus.VERIFY_FAILED,
            )]
        temporary = final.with_name(f".{final.stem}.anydes-{uuid4().hex[:12]}{final.suffix}")
        internal_create = tuple(dict.fromkeys((*confirmed_effects, PermissionEffect.CREATE)))
        results: list[ToolExecutionResult] = []
        create = self._execute_capability(
            capability,
            {"path": str(temporary), "content": content, "title": title, "overwrite": False},
            turn_id=turn_id,
            confirmed_effects=internal_create,
        )
        results.append(create)
        if create.status != ExecutionStatus.SUCCESS:
            return results
        if not any(
            effect.effect == PermissionEffect.CREATE
            and Path(effect.path).resolve(strict=False) == temporary
            for effect in create.side_effects
        ):
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            self.state.last_verification = {
                "status": "VERIFY_FAILED",
                "reason": "staged_create_evidence_missing",
                "atomic_commit": False,
            }
            return results
        read = self._execute_capability(
            "document.read_document",
            {"path": str(temporary)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            transaction_io=True,
        )
        results.append(read)
        if read.status != ExecutionStatus.SUCCESS:
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            return results
        verification = self._verify_staged_artifact(temporary, capability, verify,
            str(read.output.get("content", "")), conversion_source)
        repair_count = 0
        if verification.get("status") != "VERIFIED" and allow_repair:
            repair_count = 1
            actual = str(read.output.get("content", ""))
            repair = self._execute_capability(
                "document.edit_document",
                {"path": str(temporary), "old_text": actual, "new_text": content},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(repair)
            if repair.status == ExecutionStatus.SUCCESS:
                read = self._execute_capability(
                    "document.read_document",
                    {"path": str(temporary)},
                    turn_id=turn_id,
                    confirmed_effects=confirmed_effects,
                    transaction_io=True,
                )
                results.append(read)
                if read.status == ExecutionStatus.SUCCESS:
                    verification = self._verify_staged_artifact(temporary, capability, verify,
                        str(read.output.get("content", "")), conversion_source)
        verification["repair_count"] = repair_count
        verification["atomic_commit"] = False
        self.state.last_verification = verification
        self.audit.record(
            "conversation_staged_write_verification",
            turn_id=turn_id,
            temporary_path=str(temporary),
            final_path=str(final),
            **verification,
        )
        if verification.get("status") != "VERIFIED":
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            results.append(self._synthetic_failure(
                turn_id,
                capability,
                "staged_document_verification_failed",
                status=ExecutionStatus.VERIFY_FAILED,
                output=verification,
            ))
            return results
        if self._artifact_bindings is not None:
            for binding, ref in list(self._artifact_bindings.bindings.items()):
                if ref.path == str(temporary):
                    self._artifact_bindings.transition(binding, ArtifactState.VERIFIED)
        commit_effect = commit_effect or (
            PermissionEffect.OVERWRITE if final.exists() else PermissionEffect.CREATE
        )
        commit = self._execute_capability(
            "filesystem.atomic_replace",
            {
                "source": str(temporary),
                "destination": str(final),
                "commit_effect": commit_effect.value,
            },
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(commit)
        if commit.status != ExecutionStatus.SUCCESS:
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            return results
        final_read = self._execute_capability(
            "document.read_document",
            {"path": str(final)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            transaction_io=True,
        )
        results.append(final_read)
        if final_read.status == ExecutionStatus.SUCCESS:
            final_verification = self._verify_staged_artifact(final, capability, verify,
                str(final_read.output.get("content", "")), conversion_source)
            final_verification["repair_count"] = repair_count
            final_verification["atomic_commit"] = final_verification.get("status") == "VERIFIED"
            self.state.last_verification = final_verification
            results[-1] = replace(
                final_read,
                output={**final_read.output, **final_verification},
            )
            self.audit.record(
                "conversation_atomic_commit_verification",
                turn_id=turn_id,
                final_path=str(final),
                **final_verification,
            )
            if final_verification.get("status") != "VERIFIED":
                results.append(self._synthetic_failure(
                    turn_id,
                    capability,
                    "atomic_commit_verification_failed",
                    status=ExecutionStatus.VERIFY_FAILED,
                    output=final_verification,
                ))
        return results

    def _discard_staged(self, path, turn_id, confirmed_effects, results):
        if not Path(path).exists():
            return
        discard = self._execute_capability(
            "filesystem.delete",
            {"path": str(path)},
            turn_id=turn_id,
            confirmed_effects=tuple(dict.fromkeys((*confirmed_effects, PermissionEffect.DELETE))),
            runtime_maintenance=True,
        )
        results.append(discard)

    def _stage_edit_and_commit_document(
        self,
        *,
        final_path: str,
        before: str,
        expected: str,
        turn_id: str,
        confirmed_effects: tuple[PermissionEffect, ...],
        verify,
        document_edit_spec: dict[str, Any] | None = None,
    ) -> list[ToolExecutionResult]:
        final = Path(final_path).resolve(strict=False)
        temporary = final.with_name(f".{final.stem}.anydes-{uuid4().hex[:12]}{final.suffix}")
        internal_create = tuple(dict.fromkeys((*confirmed_effects, PermissionEffect.CREATE)))
        results: list[ToolExecutionResult] = []
        copied = self._execute_capability(
            "filesystem.copy",
            {"source": str(final), "destination": str(temporary), "overwrite": False},
            turn_id=turn_id,
            confirmed_effects=internal_create,
        )
        results.append(copied)
        if copied.status != ExecutionStatus.SUCCESS:
            return results
        if self._artifact_bindings is not None and self._executing_plan is not None:
            edit_step = next((s for s in self._executing_plan.steps if s.capability_id == "document.edit_document"), None)
            if edit_step:
                for binding in edit_step.produces_artifacts:
                    if binding not in self._artifact_bindings.bindings:
                        self._artifact_bindings.produce(binding, temporary, edit_step.step_id)
        edit = self._execute_capability(
            "document.edit_document",
            (
                {"path": str(temporary), "document_edit_spec": document_edit_spec}
                if document_edit_spec is not None
                else {"path": str(temporary), "old_text": before, "new_text": expected}
            ),
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            transaction_io=True,
        )
        results.append(edit)
        if edit.status != ExecutionStatus.SUCCESS:
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            return results
        read = self._execute_capability(
            "document.read_document",
            {"path": str(temporary)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            transaction_io=True,
        )
        results.append(read)
        if read.status != ExecutionStatus.SUCCESS:
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            return results
        verification = self._verify_staged_artifact(temporary, "document.edit_document", verify,
            str(read.output.get("content", "")))
        repair_count = 0
        if verification.get("status") != "VERIFIED" and document_edit_spec is None:
            repair_count = 1
            actual = str(read.output.get("content", ""))
            repair = self._execute_capability(
                "document.edit_document",
                {"path": str(temporary), "old_text": actual, "new_text": expected},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(repair)
            if repair.status == ExecutionStatus.SUCCESS:
                read = self._execute_capability(
                    "document.read_document",
                    {"path": str(temporary)},
                    turn_id=turn_id,
                    confirmed_effects=confirmed_effects,
                    transaction_io=True,
                )
                results.append(read)
                if read.status == ExecutionStatus.SUCCESS:
                    verification = self._verify_staged_artifact(temporary, "document.edit_document", verify,
                        str(read.output.get("content", "")))
        verification["repair_count"] = repair_count
        verification["atomic_commit"] = False
        self.state.last_verification = verification
        self.audit.record(
            "conversation_staged_edit_verification",
            turn_id=turn_id,
            temporary_path=str(temporary),
            final_path=str(final),
            **verification,
        )
        if verification.get("status") != "VERIFIED":
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            results.append(self._synthetic_failure(
                turn_id,
                "document.edit_document",
                "staged_document_edit_verification_failed",
                status=ExecutionStatus.VERIFY_FAILED,
                output=verification,
            ))
            return results
        if self._artifact_bindings is not None:
            for binding, ref in list(self._artifact_bindings.bindings.items()):
                if ref.path == str(temporary):
                    self._artifact_bindings.transition(binding, ArtifactState.VERIFIED)
        commit = self._execute_capability(
            "filesystem.atomic_replace",
            {
                "source": str(temporary),
                "destination": str(final),
                "commit_effect": PermissionEffect.EDIT.value,
            },
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(commit)
        if commit.status != ExecutionStatus.SUCCESS:
            self._discard_staged(temporary, turn_id, confirmed_effects, results)
            return results
        final_read = self._execute_capability(
            "document.read_document",
            {"path": str(final)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
            transaction_io=True,
        )
        results.append(final_read)
        if final_read.status == ExecutionStatus.SUCCESS:
            final_verification = self._verify_staged_artifact(final, "document.edit_document", verify,
                str(final_read.output.get("content", "")))
            final_verification["repair_count"] = repair_count
            final_verification["atomic_commit"] = final_verification.get("status") == "VERIFIED"
            self.state.last_verification = final_verification
            results[-1] = replace(final_read, output={**final_read.output, **final_verification})
            if final_verification.get("status") != "VERIFIED":
                results.append(self._synthetic_failure(
                    turn_id,
                    "document.edit_document",
                    "atomic_document_edit_verification_failed",
                    status=ExecutionStatus.VERIFY_FAILED,
                    output=final_verification,
                ))
        return results

    def _execute_text_files_to_pdf(self, action, turn_id, confirmed_effects):
        results: list[ToolExecutionResult] = []
        search = self._execute_capability(
            "filesystem.search",
            {"root": action.arguments["root"], "pattern": "*.txt", "limit": 200},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(search)
        if search.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)
        matches = list(search.output.get("matches", []))
        if not matches:
            empty = self._synthetic_failure(turn_id, "filesystem.read", "no_text_files_found")
            results.append(empty)
            self._remember_results(results)
            return tuple(results)
        sections: list[str] = []
        for path in matches:
            read = self._execute_capability(
                "filesystem.read",
                {"path": path},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(read)
            if read.status != ExecutionStatus.SUCCESS:
                self._remember_results(results)
                return tuple(results)
            sections.append(f"{Path(path).name}\n\n{read.output.get('content', '')}")
        destination = action.arguments["destination"]
        create = self._execute_capability(
            "document.create_pdf",
            {
                "path": destination,
                "content": "\n\n".join(sections),
                "overwrite": Path(destination).exists(),
                "title": "Textos combinados",
            },
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(create)
        self._remember_results(results)
        return tuple(results)

    def _execute_append(self, action, turn_id, confirmed_effects):
        read = self._execute_capability(
            "document.read_document",
            {"path": action.arguments["path"]},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results((read,))
            return (read,)
        original = str(read.output.get("content", ""))
        if not original:
            failed = self._synthetic_failure(turn_id, "document.edit_document", "empty_document_edit_requires_explicit_content")
            self._remember_results((read, failed))
            return (read, failed)
        addition = str(action.arguments.get("addition", "")).strip()
        updated = original.rstrip() + "\n" + addition + "\n"
        edit = self._execute_capability(
            "document.edit_document",
            {"path": action.arguments["path"], "old_text": original, "new_text": updated},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results: list[ToolExecutionResult] = [read, edit]
        if edit.status == ExecutionStatus.SUCCESS:
            read_back = self._execute_capability(
                "document.read_document",
                {"path": action.arguments["path"]},
                turn_id=turn_id,
                confirmed_effects=confirmed_effects,
            )
            results.append(read_back)
            if read_back.status == ExecutionStatus.SUCCESS:
                verification = verify_creation(
                    str(read_back.output.get("content", "")), updated, "txt"
                )
                self.state.last_verification = verification
                results[-1] = replace(read_back, output={**read_back.output, **verification})
                if verification["status"] != "VERIFIED":
                    results.append(self._synthetic_failure(
                        turn_id,
                        "document.edit_document",
                        "write_verification_failed",
                        status=ExecutionStatus.VERIFY_FAILED,
                        output=verification,
                    ))
        self._remember_results(results)
        return tuple(results)

    def _execute_inline_python(self, action, turn_id, confirmed_effects):
        path = _next_code_path(Path(self.state.active_workspace) / "codigo.py")
        write = self._execute_capability(
            "developer.write_code",
            {"path": str(path), "code": action.arguments["code"]},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        if write.status != ExecutionStatus.SUCCESS:
            self._remember_results((write,))
            return (write,)
        run = self._execute_capability(
            "developer.run_python",
            {"path": str(path)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        self._remember_results((write, run))
        return (write, run)

    def _execute_general_code(self, action, turn_id, confirmed_effects):
        prepared_artifact = action.arguments.get("generated_code_artifact")
        if self.code_generator is None and not isinstance(prepared_artifact, dict):
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                "code_generator_unavailable",
            )
            self._remember_results((failure,))
            return (failure,)
        try:
            specification = CodeGenerationSpec.from_dict(
                dict(action.arguments.get("code_generation_spec", {}))
            )
            artifact = (
                GeneratedCodeArtifact.from_dict(prepared_artifact)
                if isinstance(prepared_artifact, dict)
                else self.code_generator(specification)
            )
            if isinstance(artifact, dict):
                artifact = GeneratedCodeArtifact.from_dict(artifact)
            validation = CodeValidator.validate(artifact, specification)
            from .verification_types import build_runtime_tests
            tests = self._pipeline_trace.invoke("TESTSPEC_BUILD", build_runtime_tests, specification.tests or artifact.tests,
                specification=specification, plan=self._executing_plan)
        except (RuntimeError, TypeError, ValueError, SyntaxError) as exc:
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                f"code_generation_failed:{type(exc).__name__}",
            )
            self._remember_results((failure,))
            return (failure,)
        if not validation.valid:
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                "code_validation_failed:" + ",".join(validation.errors),
                status=ExecutionStatus.VERIFY_FAILED,
                output={"validation_errors": list(validation.errors)},
            )
            self._remember_results((failure,))
            return (failure,)

        final_path = Path(str(action.arguments["path"])).resolve(strict=False)
        temp_path = final_path.with_name(
            f".anydes-code-{uuid4().hex[:12]}{final_path.suffix}"
        )
        results: list[ToolExecutionResult] = []
        write = self._execute_capability(
            "developer.write_code",
            {"path": str(temp_path), "code": artifact.code, "overwrite": False},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(write)
        if write.status != ExecutionStatus.SUCCESS:
            self._remember_results(results)
            return tuple(results)

        self._drop_retained_draft()

        self.audit.record("verification_semantics_propagation",
            task_id=self._executing_plan.task_id if self._executing_plan else "",
            descriptor=specification.operation_descriptor,
            test_specs=[asdict(test) for test in tests],
            semantics_status="PRESENT" if any(t.operation_semantics for t in tests) else "ABSENT",
            binding_source="EXECUTABLE_PLAN" if self._executing_plan else "CODE_SPEC")
        preferred_run = "developer.run_python" if specification.language == "python" else "developer.run_cpp"
        run_capability = (
            preferred_run
            if preferred_run in action.capabilities
            else "developer.test"
        )
        verification_rows: list[dict[str, Any]] = []
        from .verification_types import VerificationRequirement, code_verification_requirement
        verification_requirement = code_verification_requirement(specification)
        if specification.execution_required:
            if not tests and verification_requirement == VerificationRequirement.EXECUTION:
                run = self._execute_capability(run_capability,
                    {"path":str(temp_path),"stdin":"","timeout":8.0,"language":specification.language},
                    turn_id=turn_id,confirmed_effects=confirmed_effects)
                results.append(run)
                verified_run = VerificationDispatcher.verify("CODE", execution_only=True,
                    exit_code=run.output.get("returncode"), execution_succeeded=run.status==ExecutionStatus.SUCCESS)
                self._remember_typed_verification(verified_run)
                self.audit.record("typed_verification", verification=verified_run.to_dict(),
                    verification_requirement=verification_requirement.value, oracle_called=False)
                if verified_run.status != VerificationStatus.PASS:
                    results.append(self._synthetic_failure(turn_id,run_capability,"code_execution_verification_failed",
                        status=ExecutionStatus.VERIFY_FAILED,output=verified_run.legacy()))
            elif not tests:
                results.append(self._synthetic_failure(
                    turn_id,
                    run_capability,
                    "code_execution_requires_tests",
                    status=ExecutionStatus.VERIFY_FAILED,
                ))
            else:
                for test in tests:
                    run = self._execute_capability(
                        run_capability,
                        {
                            "path": str(temp_path),
                            "stdin": test.stdin,
                            "timeout": test.timeout,
                            "language": specification.language,
                        },
                        turn_id=turn_id,
                        confirmed_effects=confirmed_effects,
                    )
                    results.append(run)
                    code_ref = next((r for r in self._artifact_bindings.artifacts.values()
                        if r.path == str(temp_path)), None) if self._artifact_bindings else None
                    oracle_test = test
                    trusted_oracle = isinstance(prepared_artifact, dict) and action.arguments.get("oracle_authority") != "MODEL_GENERATED"
                    if callable(self.code_oracle_provider):
                        approved = self.code_oracle_provider(specification, test)
                        if isinstance(approved, TestSpec) and approved.stdin == test.stdin and approved.test_id == test.test_id:
                            oracle_test, trusted_oracle = approved, True
                    verification = (VerificationDispatcher.verify('CODE', execution_only=True,
                        exit_code=run.output.get('returncode'), execution_succeeded=False)
                        if run.status != ExecutionStatus.SUCCESS else self._pipeline_trace.invoke("VERIFICATION_DISPATCH", VerificationDispatcher.verify, "CODE", test=oracle_test,
                        stdout=str(run.output.get("stdout", "")), exit_code=run.output.get("returncode"),
                        artifact_id=code_ref.artifact_id if code_ref else "",
                        trusted_oracle=trusted_oracle, oracle_selector=self.oracle_selector))
                    self.audit.record("oracle_selection", test_id=test.test_id,
                        task_id=self._executing_plan.task_id if self._executing_plan else "",
                        verification=verification.to_dict())
                    if (verification.status == VerificationStatus.PASS
                            and verify_code_test is not _LEGACY_CODE_TEST_GATE
                            and not verify_code_test(str(run.output.get("stdout", "")), test)):
                        verification = replace(verification, status=VerificationStatus.FAIL,
                            failed_check=test.test_id, error_code="TEST_ASSERTION_FAILED",
                            safe_message=f"El programa no superó la prueba {test.test_id}.")
                    passed = run.status == ExecutionStatus.SUCCESS and verification.status == VerificationStatus.PASS
                    evidence = VerificationEvidence.code_test(test, run, passed, verification)
                    for index, step_result in enumerate(self.last_step_results):
                        if step_result.values.get("request_id") == run.request_id:
                            self.last_step_results[index] = replace(step_result,
                                verification={"verification_result": verification.to_dict()})
                    self.state.verification_evidence[evidence.verification_id] = evidence
                    evidence_audit = asdict(evidence)
                    evidence_audit["error_code"] = evidence_audit.pop("cause_code")
                    self.audit.record("conversation_verification_evidence", turn_id=turn_id,
                        task_id=self.state.pending_task.task_id,
                        attempt_number=self.state.pending_task.attempt_number, **evidence_audit)
                    while len(self.state.verification_evidence) > 64:
                        self.state.verification_evidence.pop(next(iter(self.state.verification_evidence)))
                    verification_rows.append({
                        "test_id": test.test_id,
                        "passed": passed,
                        "returncode": run.output.get("returncode"),
                        "verification_ref": evidence.verification_id,
                        "verification_result": verification.to_dict(),
                    })
                    if not passed:
                        results.append(self._synthetic_failure(
                            turn_id,
                            run_capability,
                            f"code_test_failed:{test.test_id}",
                            status=ExecutionStatus.VERIFY_FAILED,
                            output={"test_id": test.test_id, "verification_evidence": asdict(evidence)},
                        ))
                        break
        failed = any(
            item.status != ExecutionStatus.SUCCESS
            or (
                item.capability.startswith("developer.run_")
                and int(item.output.get("returncode", 0) or 0) != 0
            )
            for item in results
        )
        if failed:
            quality_only = bool(verification_rows) and run.status == ExecutionStatus.SUCCESS
            if quality_only and verification_rows[-1]['verification_result']['status'] in {'INCONCLUSIVE','FAIL'}:
                self._drop_retained_draft()
                from .dataflow import file_hash
                owned = Path(self.state.active_workspace)/'.anydes-drafts'
                owned.mkdir(exist_ok=True)
                if owned.resolve(strict=True) != owned:
                    raise ValueError('DENIED_REPARSE_QUALITY_DRAFT_ROOT')
                held = owned/('draft_'+uuid4().hex+temp_path.suffix)
                moved = self._execute_capability('filesystem.move',{'source':str(temp_path),
                    'destination':str(held)},turn_id=turn_id,
                    confirmed_effects=tuple((*confirmed_effects,PermissionEffect.MOVE)),runtime_maintenance=True)
                results.append(moved)
                if moved.status != ExecutionStatus.SUCCESS:
                    temp_path.unlink(missing_ok=True)
                    self._remember_results(results)
                    return tuple(results)
                for binding, item in list(self._artifact_bindings.bindings.items()):
                    if item.path == str(temp_path):
                        updated = replace(item,temporary_path_ref=str(held))
                        self._artifact_bindings.bindings[binding] = updated
                        self._artifact_bindings.artifacts[item.artifact_id] = updated
                self._retained_draft = {'source':str(held), 'destination':str(final_path),
                    'workspace_root':self.state.active_workspace,
                    'task_id':self.state.pending_task.task_id, 'artifact_id':code_ref.artifact_id if code_ref else '',
                    'hash':file_hash(held), 'verification':verification_rows[-1]['verification_result'],
                    'expected_destination_hash':self._artifact_bindings.path_snapshots.get(str(final_path),'ABSENT'),
                    'previous_ref':action.resolved_artifacts[0] if action.task_spec and
                        action.task_spec.goal == 'modify_code' and action.resolved_artifacts else None,
                    'task_goal':self._executing_plan.goal if self._executing_plan else '',
                    'created_at':time.time()}
                self.state.recent_artifact_state.focus_unavailable = True
            else:
                temp_path.unlink(missing_ok=True)
            self.state.last_verification = {
                "status": "VERIFY_FAILED",
                "tests": verification_rows,
                "final_artifact_committed": False,
            }
            self._remember_results(results)
            return tuple(results)

        if self._artifact_bindings is not None:
            for binding, ref in list(self._artifact_bindings.bindings.items()):
                if ref.path == str(temp_path):
                    self._artifact_bindings.transition(binding, ArtifactState.VERIFIED)

        commit_effect = PermissionEffect.OVERWRITE if final_path.exists() else PermissionEffect.CREATE
        commit = self._execute_capability(
            "filesystem.atomic_replace",
            {
                "source": str(temp_path),
                "destination": str(final_path),
                "commit_effect": commit_effect.value,
            },
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        results.append(commit)
        committed = commit.status == ExecutionStatus.SUCCESS and final_path.is_file()
        if committed:
            try:
                committed_source = final_path.read_text(encoding="utf-8")
            except OSError:
                committed_source = ""
            committed = committed_source == artifact.code
        if not committed:
            temp_path.unlink(missing_ok=True)
            results.append(self._synthetic_failure(
                turn_id,
                "filesystem.atomic_replace",
                "code_atomic_commit_verification_failed",
                status=ExecutionStatus.VERIFY_FAILED,
            ))
        else:
            plan = GeneralTaskPlan.from_dict(dict(action.arguments.get("general_task_plan", {})))
            if "archive.compress_zip" in plan.required_capabilities:
                archive_path = self._lowered_archive_path(action, final_path)
                compressed = self._execute_capability(
                    "archive.compress_zip",
                    {
                        "sources": [str(final_path)],
                        "destination": str(archive_path),
                        "overwrite": archive_path.exists(),
                    },
                    turn_id=turn_id,
                    confirmed_effects=confirmed_effects,
                )
                results.append(compressed)
                committed = compressed.status == ExecutionStatus.SUCCESS and archive_path.is_file()
        self.state.last_verification = {
            "status": "VERIFIED" if committed else "VERIFY_FAILED",
            "tests": verification_rows,
            "test_count": len(verification_rows),
            "final_artifact_committed": committed,
            "path": str(final_path),
        }
        self.audit.record(
            "general_code_verification",
            turn_id=turn_id,
            task_id=action.task_spec.task_id if action.task_spec else "",
            **self.state.last_verification,
        )
        self._remember_results(results)
        return tuple(results)

    def _execute_general_code_modification(self, action, turn_id, confirmed_effects):
        if self.code_modifier is None:
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                "code_modifier_unavailable",
            )
            self._remember_results((failure,))
            return (failure,)
        path = Path(str(action.arguments.get("path", ""))).resolve(strict=False)
        read = self._execute_capability(
            "developer.read_code",
            {"path": str(path)},
            turn_id=turn_id,
            confirmed_effects=confirmed_effects,
        )
        if read.status != ExecutionStatus.SUCCESS:
            self._remember_results((read,))
            return (read,)
        existing = str(read.output.get("content", read.output.get("code", "")))
        if not existing.strip():
            failure = self._synthetic_failure(
                turn_id,
                "developer.read_code",
                "code_modification_source_empty",
                status=ExecutionStatus.VERIFY_FAILED,
            )
            self._remember_results((read, failure))
            return (read, failure)
        try:
            specification = CodeModificationSpec.from_dict(
                dict(action.arguments.get("code_modification_spec", {}))
            ).with_existing_code(existing)
            artifact = self.code_modifier(specification)
            if isinstance(artifact, dict):
                artifact = GeneratedCodeArtifact.from_dict(artifact)
            tests = specification.new_tests or specification.tests_to_preserve or artifact.tests
            generation_spec = CodeGenerationSpec(
                language=specification.language,
                goal=specification.requested_change,
                functional_requirements=(
                    *specification.preserve_requirements,
                    *specification.new_requirements,
                ),
                inputs=(),
                outputs=(),
                constraints=("preserve_existing_artifact",),
                dependencies=(),
                target_filename=path.name,
                tests=tests,
                execution_required=specification.execution_required,
                operation_descriptor=specification.operation_descriptor,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                f"code_modification_failed:{type(exc).__name__}",
            )
            self._remember_results((read, failure))
            return (read, failure)
        if artifact.code == existing:
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                "code_modification_no_change",
                status=ExecutionStatus.VERIFY_FAILED,
            )
            self._remember_results((read, failure))
            return (read, failure)
        lowered = replace(
            action,
            operation="general_generate_code",
            arguments={
                **action.arguments,
                "oracle_authority": "MODEL_GENERATED",
                "code_generation_spec": generation_spec.to_dict(),
                "generated_code_artifact": {
                    "language": artifact.language,
                    "filename": artifact.filename,
                    "code": artifact.code,
                    "tests": [asdict(test) for test in artifact.tests],
                },
            },
        )
        generated = self._execute_general_code(lowered, turn_id, confirmed_effects)
        results = (read, *generated)
        if (
            self.state.last_verification.get("status") == "VERIFIED"
            and path.is_file()
            and path.read_text(encoding="utf-8") == existing
        ):
            failure = self._synthetic_failure(
                turn_id,
                "developer.write_code",
                "code_modification_atomic_replace_unchanged",
                status=ExecutionStatus.VERIFY_FAILED,
            )
            results = (*results, failure)
            self.state.last_verification["status"] = "VERIFY_FAILED"
        self._remember_results(results)
        return tuple(results)

    def _execute_general_composition(self, action, turn_id, confirmed_effects):
        plan = self._executing_plan
        if plan is None:
            raise ValueError("executable_contract_required")
        outputs = {}
        results = []
        for step in plan.steps:
            self._dataflow_step = step
            try:
                args = _resolve_references(step.args, outputs)
                args, refs = self._artifact_bindings.resolve_arguments(replace(step, args=args))
            except (KeyError, ValueError) as exc:
                failure = self._synthetic_failure(turn_id, step.capability_id,
                    str(exc) if isinstance(exc, DataflowError) else f"invalid_step_reference:{type(exc).__name__}",
                    status=ExecutionStatus.INVALID_INPUT)
                results.append(failure)
                self.last_step_results.append(StepResult(step.step_id, failure.status.value,
                    error=failure.error, values={"request_id": failure.request_id}))
                break
            step_results = self._execute_flow_step(replace(step, args=args), turn_id, confirmed_effects)
            results.extend(step_results)
            if any(r.status != ExecutionStatus.SUCCESS for r in step_results):
                break
            outputs[step.step_id] = step_results[-1].output
        self._dataflow_step = None
        succeeded = len(outputs) == len(plan.steps) and all(r.status == ExecutionStatus.SUCCESS for r in results)
        self.state.last_verification = {
            "status": "VERIFIED" if succeeded else "VERIFY_FAILED",
            "step_count": len(results),
            "expected_steps": len(plan.steps),
            "all_steps_succeeded": succeeded,
        }
        self._remember_results(results)
        return tuple(results)

    def _execute_capability(self, capability, arguments, *, turn_id, confirmed_effects, transaction_io=False,
                            runtime_maintenance=False):
        plan = self._executing_plan
        step = self._dataflow_step or (next((s for s in plan.steps if s.capability_id == capability), None) if plan else None)
        if step is None and plan and capability.startswith("document.create"):
            step = next((s for s in plan.steps if s.capability_id == "document.edit_document"), None)
        consumed = []
        if self._artifact_bindings is not None and plan is not None:
            try:
                if step and step.capability_id == capability and not transaction_io:
                    arguments, consumed = self._artifact_bindings.resolve_arguments(replace(step, args=dict(arguments)))
                if capability == "filesystem.atomic_replace":
                    for binding, ref in self._artifact_bindings.bindings.items():
                        if ref.producer_step_id == "INPUT":
                            self._artifact_bindings.require(binding, verified=True, allow_unverified=(
                                plan.legacy_action.capabilities == ('archive.compress_zip',) or
                                plan.legacy_action.operation in {'general_modify_code','retain_unverified'}))
                    arguments = {**arguments, "expected_destination_hash":
                        self._artifact_bindings.snapshot(arguments["destination"])}
                if capability == "archive.compress_zip" and not self._archive_transaction:
                    return self._execute_bound_archive(arguments, step, consumed,
                        turn_id=turn_id, confirmed_effects=confirmed_effects)
            except DataflowError as exc:
                result = self._synthetic_failure(turn_id, capability, str(exc), status=ExecutionStatus.INVALID_INPUT)
                self.last_step_results.append(StepResult(step.step_id if step else "transaction",
                    result.status.value, error=str(exc), values={"request_id": result.request_id}))
                return result
        self._executor_called = True
        tools = self.registry.find_by_capability(capability)
        if not tools:
            return self._synthetic_failure(turn_id, capability, "missing_capability")
        self.audit.record(
            "conversation_executor_dispatch",
            turn_id=turn_id,
            tool_id=tools[0].manifest.tool_id,
            capability=capability,
            executor_called=True,
        )
        stage = "B2_EXECUTION" if capability in {"developer.run_python", "developer.run_cpp", "developer.test"} else "COMMIT" if capability == "filesystem.atomic_replace" else "EXECUTION_DISPATCH"
        result = self._pipeline_trace.invoke(stage, self.executor.execute,
            ToolRequest(
                request_id=f"conversation:{turn_id}:{uuid4().hex[:10]}",
                turn_id=turn_id,
                tool_id=tools[0].manifest.tool_id,
                capability=capability,
                arguments=dict(arguments),
                authorized_scopes=self.state.authorized_scopes,
                confirmed_effects=confirmed_effects,
                reflection_approved=True,
                explicit_user_request=True,
            )
        )
        if runtime_maintenance:
            result = replace(result, side_effects=tuple(replace(e, scope="RUNTIME_MAINTENANCE",
                user_visible=False, task_semantic=False, committed=False) for e in result.side_effects))
        effect_operations = {"filesystem.create_text": "CREATE", "filesystem.copy": "CREATE",
            "filesystem.move": "MOVE", "filesystem.rename": "RENAME", "filesystem.delete": "DELETE",
            "filesystem.edit_text": "EDIT"}
        if result.status == ExecutionStatus.SUCCESS and capability in effect_operations:
            typed = VerificationDispatcher.verify("FILESYSTEM_CHANGE", operation=effect_operations[capability],
                target=arguments.get("destination") or arguments.get("path"), source=arguments.get("source"),
                before_hash=self._artifact_bindings.path_snapshots.get(str(Path(arguments["path"]).resolve(strict=False)))
                    if capability == "filesystem.edit_text" and self._artifact_bindings else None,
                require_change=arguments.get("old_text") != arguments.get("new_text"))
            self._remember_typed_verification(typed)
            result = replace(result, output={**result.output, "verification_result": typed.to_dict()},
                status=result.status if typed.status == VerificationStatus.PASS else ExecutionStatus.VERIFY_FAILED)
        match = next((s for s in plan.steps if s.capability_id == capability), None) if plan else None
        step_id = step.step_id if step and step.capability_id == capability else match.step_id if match else "legacy_expansion"
        if capability == "filesystem.atomic_replace" and match is None and self._artifact_bindings is not None:
            producer = next((a for a in self._artifact_bindings.bindings.values()
                if a.path == str(Path(arguments["source"]).resolve(strict=False))), None)
            if producer:
                step_id = producer.producer_step_id
        produced = []
        if result.status == ExecutionStatus.SUCCESS and capability == "filesystem.delete":
            canonical = str(Path(arguments["path"]).resolve(strict=False))
            previous = self._operational_artifacts.get(canonical)
            if previous:
                self._operational_artifacts[canonical] = replace(previous, state=ArtifactState.DELETED)
        if result.status == ExecutionStatus.SUCCESS and self._artifact_bindings is not None and plan is not None:
            ledger = self._artifact_bindings
            if capability == "filesystem.atomic_replace":
                for binding, ref in list(ledger.bindings.items()):
                    if ref.path == str(Path(arguments["source"]).resolve(strict=False)):
                        previous = next((p for p in ledger.bindings.values()
                            if p.producer_step_id == "INPUT" and p.path == str(Path(arguments["destination"]).resolve(strict=False))), None)
                        if previous:
                            ref = replace(ref, artifact_id=previous.artifact_id, version=previous.version,
                                metadata={**ref.metadata, "previous_version": previous.version, "previous_hash": previous.hash})
                            ledger.bindings[binding] = ref
                        committed = ledger.transition(binding, ArtifactState.COMMITTED,
                            path=arguments["destination"], new_version=arguments["expected_destination_hash"] != "ABSENT")
                        if previous:
                            for input_binding, item in list(ledger.bindings.items()):
                                if item.artifact_id == previous.artifact_id and item.producer_step_id == "INPUT":
                                    ledger.bindings[input_binding] = committed
                        produced.append(committed.artifact_id)
            elif step and step.produces_artifacts and (step.capability_id == capability or
                    capability.startswith("document.create") and step.capability_id == "document.edit_document"):
                output_path = result.output.get("path") or result.output.get("destination")
                if output_path and Path(output_path).is_file():
                    for binding in step.produces_artifacts:
                        if binding not in ledger.bindings:
                            existing = next((a for a in ledger.bindings.values() if a.path == str(Path(output_path).resolve(strict=False))), None)
                            if existing is not None and capability in {"filesystem.read", "document.read_document", "developer.read_code"}:
                                ledger.bindings[binding] = existing
                                produced.append(existing.artifact_id)
                                continue
                            ref = ledger.produce(binding, output_path, step_id,
                                state=ArtifactState.COMMITTED if capability in {"filesystem.read", "document.read_document", "developer.read_code"}
                                    else ArtifactState.TEMPORARY,
                                metadata={"consumed_artifacts": [r.artifact_id for r in consumed],
                                    'task_goal':plan.goal[:2000]})
                            produced.append(ref.artifact_id)
                elif step.produces_artifacts:
                    result = replace(result, status=ExecutionStatus.INVALID_INPUT,
                        error="DATAFLOW_PRODUCER_NO_ARTIFACT")
        self.last_step_results.append(StepResult(
            step_id=step_id, status=result.status.value,
            values={"request_id": result.request_id, "capability": capability},
            error=result.error,
            verification={"status": result.output.get("status", "")},
            artifact_ids=tuple(produced), produced_artifacts=tuple(produced),
            consumed_artifacts=tuple(r.artifact_id for r in consumed),
            observed_effects=tuple(asdict(e) for e in result.side_effects),
        ))
        self.audit.record("executable_step_result", turn_id=turn_id,
            task_id=plan.task_id if plan else "", step_id=step_id,
            capability=capability, status=result.status.value,
            error_code=result.error, executor_called=True)
        if plan is not None:
            self.audit.record("artifact_dataflow", turn_id=turn_id, task_id=plan.task_id,
                step_id=step_id, consumed_artifact_ids=[r.artifact_id for r in consumed],
                produced_artifact_ids=produced,
                artifact_versions={r.artifact_id: r.version for r in consumed},
                artifact_states={r.artifact_id: r.state.value for r in consumed},
                binding_source="LEGACY_ADAPTER" if step and step.step_type == "LEGACY_OPERATION_ADAPTER" else "EXECUTABLE_PLAN",
                commit_status=result.status.value if capability == "filesystem.atomic_replace" else "NOT_COMMITTED")
        return result

    def _execute_bound_archive(self, arguments, step, consumed, *, turn_id, confirmed_effects):
        if self._retained_draft and any(Path(self._retained_draft['source']).is_relative_to(Path(source))
                for source in self._executing_plan.typed_inputs.get('explicit_directory_sources', ())):
            raise DataflowError('QUALITY_APPROVAL_REQUIRED_FOR_DIRECTORY')
        final = Path(arguments["destination"]).resolve(strict=False)
        self._artifact_bindings.snapshot(final)
        temporary = final.with_name(f".anydes-archive-{uuid4().hex[:12]}.zip")
        results = []
        self._archive_transaction = True
        typed = None
        try:
            expected = {}
            for raw in arguments["sources"]:
                source = Path(raw)
                if source.is_file():
                    members = [(source.name, source)]
                elif source.is_dir() and not consumed and str(source.resolve(strict=False)) in (
                        self._executing_plan.typed_inputs.get("explicit_directory_sources", ())):
                    # A directory is allowed only when explicitly requested, not as fallback.
                    members = [(f"{source.name}/{p.relative_to(source).as_posix()}", p)
                               for p in source.rglob("*") if p.is_file()]
                else:
                    raise DataflowError("DATAFLOW_ARCHIVE_SOURCE_INVALID")
                for name, path in members:
                    if name in expected:
                        raise DataflowError("DATAFLOW_ARCHIVE_MEMBER_COLLISION")
                    decision = self.boundary.classify(path, operation="READ",
                        authorized_scopes=self.state.authorized_scopes)
                    if not decision.allowed:
                        raise DataflowError("DATAFLOW_ARCHIVE_SOURCE_DENIED")
                    expected[name] = file_hash(path)
            created = self._execute_capability("archive.compress_zip",
                {**arguments, "destination": str(temporary), "overwrite": False},
                turn_id=turn_id, confirmed_effects=confirmed_effects)
            results.append(created)
            if created.status != ExecutionStatus.SUCCESS:
                return created
            typed = VerificationDispatcher.verify("ARCHIVE", path=temporary, expected_members=expected,
                artifact_id=next((r.artifact_id for r in self._artifact_bindings.bindings.values()
                    if r.path == str(temporary)), ""))
            if typed.status != VerificationStatus.PASS:
                raise DataflowError("DATAFLOW_ARCHIVE_MEMBERS_MISMATCH")
            self._remember_typed_verification(typed)
            with zipfile.ZipFile(temporary) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise DataflowError("DATAFLOW_ARCHIVE_MEMBERS_MISMATCH")
                for name in names:
                    digest = hashlib.sha256()
                    with archive.open(name) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected[name]:
                        raise DataflowError("STALE_ARTIFACT:archive_member")
            for ref in consumed:
                if file_hash(ref.path) != ref.hash:
                    raise DataflowError("STALE_ARTIFACT:" + ref.artifact_id)
            if step:
                for binding in step.produces_artifacts:
                    if binding in self._artifact_bindings.bindings:
                        self._artifact_bindings.transition(binding, ArtifactState.VERIFIED)
            commit = self._execute_capability("filesystem.atomic_replace", {
                "source": str(temporary), "destination": str(final),
                "commit_effect": "OVERWRITE" if arguments.get("overwrite") else "CREATE"},
                turn_id=turn_id, confirmed_effects=confirmed_effects)
            results.append(commit)
            self.state.last_verification = {"status": "VERIFIED" if commit.status == ExecutionStatus.SUCCESS else "VERIFY_FAILED",
                "verification_result": typed.to_dict(), "archive_exact_members": commit.status == ExecutionStatus.SUCCESS,
                "member_count": len(expected), "atomic_commit": commit.status == ExecutionStatus.SUCCESS}
            return replace(commit, capability="archive.compress_zip", output={
                **commit.output, "path": str(final), "members": list(expected),
                "verification": dict(self.state.last_verification)})
        except (DataflowError, OSError, zipfile.BadZipFile) as exc:
            failed = self._synthetic_failure(turn_id, "archive.compress_zip", str(exc),
                status=ExecutionStatus.VERIFY_FAILED, output={"verification_result": typed.to_dict()} if typed else {})
            self.last_step_results.append(StepResult(step.step_id if step else "archive_transaction",
                "VERIFY_FAILED", error=str(exc), values={"request_id": failed.request_id}))
            return failed
        finally:
            self._archive_transaction = False
            if temporary.exists():
                self._discard_staged(temporary, turn_id, confirmed_effects, results)

    def _request_template(self, turn_id, confirmed_effects):
        """Deprecated request template; runtime composition no longer uses ToolComposer."""
        return ToolRequest(
            request_id=f"conversation:{turn_id}",
            turn_id=turn_id,
            tool_id="official.filesystem",
            capability="filesystem.inspect",
            authorized_scopes=self.state.authorized_scopes,
            confirmed_effects=confirmed_effects,
            reflection_approved=True,
            explicit_user_request=True,
        )

    def _synthetic_failure(
        self,
        turn_id,
        capability,
        error,
        *,
        status=ExecutionStatus.FAILED,
        output=None,
    ):
        tools = self.registry.find_by_capability(capability)
        return ToolExecutionResult(
            request_id=f"conversation:{turn_id}:synthetic",
            turn_id=turn_id,
            tool_id=tools[0].manifest.tool_id if tools else "",
            capability=capability,
            status=status,
            output=dict(output or {}),
            error=error,
        )

    def _required_effects(self, action: PlannedToolAction) -> tuple[PermissionEffect, ...]:
        effects: list[PermissionEffect] = []
        for capability, arguments in self._effective_steps(action):
            tools = self.registry.find_by_capability(capability)
            if tools:
                effects.append(tools[0].effect_for(capability, arguments))
        return tuple(dict.fromkeys(effects))

    def _confirmation_effects(self, action: PlannedToolAction) -> tuple[PermissionEffect, ...]:
        effects = self._required_effects(action)
        if PermissionEffect.EXECUTE_CODE in effects:
            return tuple(effect for effect in effects if effect != PermissionEffect.CREATE)
        return effects

    def _effective_steps(self, action):
        if action.operation == 'retain_unverified':
            return (('filesystem.atomic_replace', {'source':self._retained_draft['source'],
                'destination':action.target_path, 'commit_effect':'OVERWRITE' if Path(action.target_path).exists() else 'CREATE'}),)
        if action.operation == "document_conversion":
            specification = DocumentConversionSpec.from_dict(
                dict(action.arguments.get("conversion_spec", {}))
            )
            commit_effect = (
                PermissionEffect.OVERWRITE if Path(specification.target_path).exists()
                else PermissionEffect.CREATE
            )
            return (
                ("document.read_document", {"path": specification.source_path}),
                (action.structured_action_plan.capability, {
                    "path": specification.target_path,
                    "overwrite": Path(specification.target_path).exists(),
                }),
                ("filesystem.atomic_replace", {
                    "source": specification.target_path,
                    "destination": specification.target_path,
                    "commit_effect": commit_effect.value,
                }),
            )
        if action.operation == "text_files_to_pdf":
            destination = action.arguments.get("destination", "")
            return (
                ("filesystem.search", {"root": action.arguments.get("root", "")}),
                ("filesystem.read", {"path": action.arguments.get("root", "")}),
                ("document.create_pdf", {"path": destination, "overwrite": Path(destination).exists()}),
            )
        if action.operation == "append_document":
            return (
                ("document.read_document", {"path": action.arguments.get("path", "")}),
                ("document.edit_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "structural_document_edit":
            return (
                ("document.read_document", {"path": action.arguments.get("path", "")}),
                ("document.edit_document", {"path": action.arguments.get("path", "")}),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "rename_existing":
            return (("filesystem.rename", action.arguments),)
        if action.operation == "copy_existing":
            return (("filesystem.copy", action.arguments),)
        if action.operation == "verified_create_document":
            return (
                (action.capabilities[0], action.arguments),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "generated_create_document":
            return (
                (action.capabilities[0], action.arguments),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "general_generated_document":
            return (
                ("document.read_document", {"path": action.arguments.get("source_path", "")}),
                (action.structured_action_plan.capability, action.arguments),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "generated_document_edit":
            return (
                ("document.read_document", {"path": action.arguments.get("path", "")}),
                ("document.edit_document", {"path": action.arguments.get("path", "")}),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "delete_collection_request":
            return (("filesystem.search", {"root": action.arguments.get("root", "")}),)
        if action.operation == "delete_collection":
            plan = DeletePlan.from_dict(dict(action.arguments.get("delete_plan", {})))
            return tuple(
                ("filesystem.delete", {"path": target.absolute_canonical_path})
                for target in plan.targets
            )
        if action.operation == "structured_edit_text":
            return (
                ("document.read_document", {"path": action.arguments.get("path", "")}),
                ("document.edit_document", {"path": action.arguments.get("path", "")}),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "document_transform":
            return (
                ("document.read_document", {"path": action.arguments.get("path", "")}),
                ("document.edit_document", {"path": action.arguments.get("path", "")}),
                ("document.read_document", {"path": action.arguments.get("path", "")}),
            )
        if action.operation == "run_inline_python":
            return (
                ("developer.write_code", {"path": str(Path(self.state.active_workspace) / "codigo.py")}),
                ("developer.run_python", {"path": str(Path(self.state.active_workspace) / "codigo.py")}),
            )
        if action.operation == "general_generate_code":
            specification = CodeGenerationSpec.from_dict(
                dict(action.arguments.get("code_generation_spec", {}))
            )
            final_path = str(action.arguments.get("path", ""))
            preferred_run = (
                "developer.run_python"
                if specification.language == "python"
                else "developer.run_cpp"
            )
            run_capability = preferred_run if preferred_run in action.capabilities else "developer.test"
            steps: list[tuple[str, dict[str, Any]]] = [
                ("developer.write_code", {"path": final_path}),
            ]
            if specification.execution_required:
                steps.append((run_capability, {"path": final_path}))
            steps.append(("filesystem.atomic_replace", {
                "source": final_path,
                "destination": final_path,
                "commit_effect": (
                    PermissionEffect.OVERWRITE.value
                    if Path(final_path).exists()
                    else PermissionEffect.CREATE.value
                ),
            }))
            if "archive.compress_zip" in action.capabilities:
                steps.append(("archive.compress_zip", {
                    "sources": [final_path],
                    "destination": str(self._lowered_archive_path(action, final_path)),
                }))
            return tuple(steps)
        if action.operation == "general_modify_code":
            specification = CodeModificationSpec.from_dict(
                dict(action.arguments.get("code_modification_spec", {}))
            )
            final_path = str(action.arguments.get("path", ""))
            preferred_run = (
                "developer.run_python"
                if specification.language == "python"
                else "developer.run_cpp"
            )
            run_capability = preferred_run if preferred_run in action.capabilities else "developer.test"
            steps: list[tuple[str, dict[str, Any]]] = [
                ("developer.read_code", {"path": final_path}),
                ("developer.write_code", {"path": final_path}),
            ]
            if specification.execution_required:
                steps.append((run_capability, {"path": final_path}))
            steps.append(("filesystem.atomic_replace", {
                "source": final_path,
                "destination": final_path,
                "commit_effect": PermissionEffect.OVERWRITE.value,
            }))
            return tuple(steps)
        if action.operation == "general_tool_composition":
            general = GeneralTaskPlan.from_dict(
                dict(action.arguments.get("general_task_plan", {}))
            )
            return tuple((step.capability, dict(step.arguments)) for step in general.steps)
        return tuple((capability, action.arguments) for capability in action.capabilities)

    def _scope_requirements(self, action: PlannedToolAction) -> tuple[str, tuple[str, ...]]:
        scopes: list[str] = []
        for path, operation in self._paths_for_action(action):
            if not path:
                continue
            decision = self.boundary.classify(
                path,
                operation=operation,
                authorized_scopes=self.state.authorized_scopes,
                developer_only=any(item.startswith("developer.") for item in action.capabilities),
            )
            if decision.category == "PROTECTED_CORE":
                return decision.reason, ()
            if not decision.allowed and decision.reason == "DENIED_OUTSIDE_AUTHORIZED_SCOPE":
                candidate = Path(path).resolve(strict=False)
                scope = candidate if candidate.exists() and candidate.is_dir() and operation == "READ" else candidate.parent
                scope_decision = self.boundary.classify(
                    candidate,
                    operation=operation,
                    authorized_scopes=(scope,),
                    developer_only=any(item.startswith("developer.") for item in action.capabilities),
                )
                if not scope_decision.allowed:
                    return scope_decision.reason, ()
                scopes.append(str(scope))
            elif not decision.allowed:
                return decision.reason, ()
        return "", tuple(dict.fromkeys(scopes))

    @staticmethod
    def _lowered_archive_path(action, final_path):
        raw = action.arguments.get("general_task_plan", {})
        if not raw.get("artifact_requirements", {}).get("semantic_lowered"):
            return Path(final_path).with_suffix(".zip")
        destinations = [row.get("arguments", {}).get("destination") for row in raw.get("steps", [])
            if row.get("capability") == "archive.compress_zip"]
        if len(destinations) != 1 or not destinations[0]:
            raise DataflowError("DATAFLOW_ARCHIVE_DESTINATION_AMBIGUOUS")
        return Path(destinations[0])

    def _paths_for_action(self, action):
        args = action.arguments
        if action.operation == "document_conversion":
            return ((args.get("source", ""), "READ"), (args.get("destination", ""), "WRITE"))
        if action.operation == "text_files_to_pdf":
            return ((args.get("root", ""), "READ"), (args.get("destination", ""), "WRITE"))
        if action.operation in {"append_document", "structured_edit_text", "document_transform", "structural_document_edit"}:
            return ((args.get("path", ""), "WRITE"),)
        if action.operation == "rename_existing":
            source = str(args.get("path", ""))
            destination = str(Path(source).with_name(str(args.get("new_name", "")))) if source and args.get("new_name") else ""
            return ((source, "WRITE"), (destination, "WRITE"))
        if action.operation == "general_generated_document":
            return (
                (args.get("source_path", ""), "READ"),
                (args.get("path", ""), "WRITE"),
            )
        if action.operation in {"verified_create_document", "generated_create_document"}:
            return ((args.get("path", ""), "WRITE"),)
        if action.operation == "generated_document_edit":
            return ((args.get("path", ""), "WRITE"),)
        if action.operation == "delete_collection_request":
            return ((args.get("root", ""), "READ"),)
        if action.operation == "delete_collection":
            plan = DeletePlan.from_dict(dict(args.get("delete_plan", {})))
            return tuple(
                (target.absolute_canonical_path, "WRITE") for target in plan.targets
            )
        if action.operation == "run_inline_python":
            return ()
        if action.operation == "general_generate_code":
            final_path = str(args.get("path", ""))
            paths: list[tuple[str, str]] = [(final_path, "WRITE")]
            if "archive.compress_zip" in action.capabilities:
                paths.append((str(self._lowered_archive_path(action, final_path)), "WRITE"))
            return tuple(paths)
        if action.operation == "general_modify_code":
            return ((str(args.get("path", "")), "WRITE"),)
        if action.operation == "general_tool_composition":
            general = GeneralTaskPlan.from_dict(
                dict(args.get("general_task_plan", {}))
            )
            paths: list[tuple[str, str]] = []
            for step in general.steps:
                for key in ("path", "source", "destination", "root"):
                    value = step.arguments.get(key)
                    if not isinstance(value, str) or not value or value.startswith("$steps."):
                        continue
                    operation = "READ" if key in {"source", "root"} else "WRITE"
                    paths.append((value, operation))
            return tuple(paths)
        capability = action.capabilities[0] if action.capabilities else ""
        if capability == "archive.compress_zip":
            return tuple((item, "READ") for item in args.get("sources", [])) + ((args.get("destination", ""), "WRITE"),)
        if capability == "filesystem.move":
            return ((args.get("source", ""), "WRITE"), (args.get("destination", ""), "WRITE"))
        if capability == "filesystem.copy":
            return ((args.get("source", ""), "READ"), (args.get("destination", ""), "WRITE"))
        operation = "READ" if capability in {
            "filesystem.inspect", "filesystem.search", "filesystem.read",
            "document.read_document", "developer.read_code", "developer.run_python",
            "developer.run_cpp", "developer.inspect_error", "archive.inspect",
        } else "WRITE"
        key = "root" if capability == "filesystem.search" else "path"
        return ((args.get(key, ""), operation),)

    def _remember_results(self, results):
        for result in results:
            summary = {
                "tool_id": result.tool_id,
                "capability": result.capability,
                "status": result.status.value,
                "error": result.error,
                "duration_ms": round(result.duration_ms, 3),
            }
            self.state.remember_result(summary)
            if result.capability == "filesystem.delete" and result.status == ExecutionStatus.SUCCESS:
                deleted = str(result.output.get("path", ""))
                if deleted:
                    canonical = str(Path(deleted).resolve(strict=False))
                    self.state.recent_files = [
                        item for item in self.state.recent_files
                        if str(Path(item).resolve(strict=False)).casefold() != canonical.casefold()
                    ]
                    if (
                        self.state.active_document
                        and str(Path(self.state.active_document).resolve(strict=False)).casefold()
                        == canonical.casefold()
                    ):
                        self.state.active_document = ""
                        self.state.active_document_ref = None
                    self.state.recent_artifact_state.remove(canonical)
                continue
            if result.capability == "filesystem.rename" and result.status == ExecutionStatus.SUCCESS:
                source = str(result.output.get("source", ""))
                destination = str(result.output.get("destination", ""))
                if source:
                    self.state.recent_artifact_state.relate(source, renamed_to=destination)
                    self.state.recent_artifact_state.remove(source)
                    self.state.recent_files = [
                        item for item in self.state.recent_files
                        if str(Path(item).resolve(strict=False)).casefold()
                        != str(Path(source).resolve(strict=False)).casefold()
                    ]
                if destination:
                    self.state.remember_file(destination)
                    self.state.recent_artifact_state.relate(destination, renamed_from=source)
                continue
            if result.capability == "filesystem.copy" and result.status == ExecutionStatus.SUCCESS:
                source = str(result.output.get("source", ""))
                destination = str(result.output.get("destination", ""))
                if destination and ".anydes-" not in Path(destination).name:
                    self.state.remember_file(destination)
                    self.state.recent_artifact_state.relate(destination, copied_from=source)
                continue
            for key in ("path", "destination"):
                value = result.output.get(key)
                if isinstance(value, str) and value and ".anydes-" not in Path(value).name:
                    self.state.remember_file(value)
            if result.output.get("stderr") or result.status != ExecutionStatus.SUCCESS:
                self.state.last_error = {
                    "stderr": str(result.output.get("stderr") or result.stderr or result.error),
                    "error": result.error,
                    "capability": result.capability,
                }

    def _response_from_results(self, action, results, resolution, *, turn_id: str):
        if action.operation == 'retain_unverified' and results and all(r.status == ExecutionStatus.SUCCESS for r in results):
            self._record_attempt_outcome('RETAINED_UNVERIFIED', turn_id=turn_id)
            self.state.pending_task.committed_effects = True
            self._finish_task('COMPLETED', clear_action=True)
            failure_ref = next((r.metadata.get('failure_ref') for r in self._artifact_bindings.artifacts.values()
                if r.state == ArtifactState.COMMITTED_UNVERIFIED), '')
            self.state.last_tool_failure = next((f for f in self.state.failure_history if f.failure_id == failure_ref), None)
            self.state.failure_topic_active = False
            return ConversationToolResponse(True, 'El borrador quedo guardado sin confirmar su correccion.',
                resolution.kind, resolution.tool_ids, executed=True,
                debug={'response_source':'USER_ACCEPTED_UNVERIFIED'})
        if (
            action.operation == "delete_collection"
            and self.state.last_verification.get("status") == "VERIFIED_PARTIAL"
        ):
            gate = gate_action_response(
                action,
                tuple(results),
                self.state.last_verification,
            )
            if gate.allowed:
                deleted = len(self.state.last_verification.get("deleted", ()))
                failed = self.state.last_verification.get("failed", ())
                skipped = self.state.last_verification.get("skipped", ())
                details: list[str] = [f"Eliminé {deleted} archivo(s)."]
                if failed:
                    details.append(
                        f"No pude eliminar {len(failed)} archivo(s) porque cambiaron o estaban en uso."
                    )
                if skipped:
                    details.append(
                        f"Omití {len(skipped)} archivo(s) temporales o de bloqueo de aplicaciones."
                    )
                self.state.last_failed_action = None
                self.state.retry_count = 0
                self._finish_task("COMPLETED_PARTIAL", clear_action=True)
                return ConversationToolResponse(
                    True,
                    " ".join(details),
                    resolution.kind,
                    resolution.tool_ids,
                    executed=bool(deleted),
                    debug={
                        "response_source": "VERIFIED_BATCH_ACTION_RESULT",
                        "verification_status": "VERIFIED_PARTIAL",
                    },
                )
        failed = next(
            (
                item for item in results
                if item.status != ExecutionStatus.SUCCESS
                or (
                    item.capability.startswith(("developer.run_", "developer.test"))
                    and int(item.output.get("returncode", 0) or 0) != 0
                )
            ),
            None,
        )
        if failed is not None:
            stage, error_code, safe_message, retryable = _failure_details(action, failed)
            self._record_failure(
                action,
                turn_id=turn_id,
                stage=stage,
                error_code=error_code,
                safe_user_message=safe_message,
                retryable=retryable,
                failed_result=failed,
            )
            safe_message = self.state.last_tool_failure.safe_user_message
            self._finish_task("FAILED", clear_action=False)
            if action.operation.startswith("general_"):
                self.audit.record(
                    "general_task_outcome",
                    turn_id=turn_id,
                    task_id=action.task_spec.task_id if action.task_spec else "",
                    executor_called=bool(results),
                    verification_passed=False,
                    failure_reason=failed.error,
                )
            return ConversationToolResponse(
                True,
                safe_message,
                resolution.kind,
                resolution.tool_ids,
                executed=False,
                debug={
                    "error": failed.error,
                    "failed_capability": failed.capability,
                    "retry_count": self.state.retry_count,
                    "verification_status": self.state.last_verification.get("status", ""),
                    "response_source": "ACTION_RESULT_FAILURE",
                },
            )
        gate = gate_action_response(
            action,
            tuple(results),
            self.state.last_verification,
        )
        self.audit.record(
            "conversation_response_gate",
            response_type=gate.response_type.value,
            allowed=gate.allowed,
            reason=gate.reason,
            expected_side_effect=gate.expected_side_effect,
            expected_path=gate.expected_path,
            linked_request_id=gate.linked_request_id,
        )
        if not gate.allowed:
            self._record_failure(
                action,
                turn_id=turn_id,
                stage="VERIFY",
                error_code="ACTION_RESULT_LINKAGE_FAILED",
                safe_user_message=(
                    "no pude verificar que la acción se ejecutara realmente"
                ),
                retryable=False,
            )
            self._finish_task("VERIFY_FAILED", clear_action=False)
            return ConversationToolResponse(
                True,
                "No pude verificar que la acción se ejecutara realmente, así que no voy a afirmar que se completó.",
                resolution.kind,
                resolution.tool_ids,
                executed=False,
                debug={
                    "response_source": "ACTION_RESULT_GATE_DENIED",
                    "response_gate_reason": gate.reason,
                    "verification_status": self.state.last_verification.get("status", ""),
                },
            )
        self._record_attempt_outcome("SUCCESS", turn_id=turn_id)
        self.state.last_failed_action = None
        self.state.last_tool_failure = None
        self.state.retry_count = 0
        self._finish_task("COMPLETED", clear_action=True)
        if action.operation.startswith("general_"):
            self.audit.record(
                "general_task_outcome",
                turn_id=turn_id,
                task_id=action.task_spec.task_id if action.task_spec else "",
                executor_called=bool(results),
                verification_passed=True,
                step_count=len(results),
            )
        final = results[-1]
        return ConversationToolResponse(
            True,
            _friendly_success(action, final),
            resolution.kind,
            resolution.tool_ids,
            executed=True,
            debug={
                "steps": len(results),
                "verification_status": self.state.last_verification.get("status", ""),
                "response_source": "VERIFIED_ACTION_RESULT",
                "linked_request_id": gate.linked_request_id,
            },
        )

    def _remember_task_plan(self, action: PlannedToolAction) -> None:
        task = self.state.pending_task
        task.plan = None
        task.task_id = action.task_spec.task_id if action.task_spec else f"task:{uuid4().hex}"
        task.plan_revision = 1
        task.attempt_number = 1
        task.status = "PLANNED"
        task.created_at = task.updated_at = time.time()
        try:
            task.plan = self.plan_adapter.compile(action, source=self._plan_source,
                task_id=task.task_id)
        except ValueError:
            pass  # Missing/forbidden capabilities are explained by CapabilityResolver.
        self.state.pending_user_task = action.task_spec
        self.state.pending_structured_plan = action.structured_action_plan
        self.state.pending_task_action = action
        self.state.replan_count = 0
        self.audit.record(
            "conversation_user_task_planned",
            task_id=task.task_id,
            plan_source=self._plan_source,
            plan_revision=task.plan_revision,
            status=task.status,
        )

    def _record_failure(
        self,
        action: PlannedToolAction,
        *,
        turn_id: str,
        stage: str,
        error_code: str,
        safe_user_message: str,
        retryable: bool,
        failed_result: ToolExecutionResult | None = None,
    ) -> None:
        from copy import deepcopy
        pending = self.state.pending_task
        task_id = pending.task_id
        capability = failed_result.capability if failed_result else (
            action.capabilities[0] if action.capabilities else "")
        artifact_type = artifact_type_for(action.operation, capability)
        step = next((s for s in self.last_step_results
                     if failed_result and s.values.get("request_id") == failed_result.request_id), None)
        if step is None and pending.plan:
            step = next((s for s in pending.plan.steps if s.capability_id == capability), None)
        evidence = (failed_result.output.get("verification_evidence", {}) if failed_result else {})
        if not evidence and failed_result:
            linked = next((v for v in self.state.verification_evidence.values()
                           if v.execution_request_id == failed_result.request_id), None)
            evidence = asdict(linked) if linked else {}
        verification_ref = str(evidence.get("verification_id", ""))
        typed = (failed_result.output.get("verification_result", {}) if failed_result else {}) or evidence.get("verification_result", {})
        if typed:
            verification_ref = typed.get("verification_id", verification_ref)
            artifact_type = typed.get("artifact_type", artifact_type)
            error_code = typed.get("error_code") or error_code
            safe_user_message = typed.get("safe_message") or safe_user_message
        if not typed and evidence.get("cause_code") and verification_ref in self.state.verification_evidence:
            error_code = evidence["cause_code"]
            check = self.state.verification_evidence[verification_ref].failed_check
            safe_user_message = (
                f"El programa se ejecutó, pero no superó la prueba {check}."
                if error_code == "TEST_ASSERTION_FAILED"
                else f"El programa terminó con un error durante la prueba {check}.")
        original_ref = plan_reference(pending.plan)
        if original_ref and original_ref not in self.state.failure_plans:
            self.state.failure_plans[original_ref] = deepcopy(pending.plan)
            while len(self.state.failure_plans) > 32:
                self.state.failure_plans.pop(next(iter(self.state.failure_plans)))
        failure = LastToolFailure.create(
            task_id=task_id,
            plan_id=(
                action.structured_action_plan.plan_id
                if action.structured_action_plan is not None
                else ""
            ),
            stage=stage,
            error_code=error_code,
            safe_user_message=safe_user_message,
            retryable=retryable,
            technical_message=(failed_result.error or error_code) if failed_result else error_code,
            plan_revision=pending.plan_revision, attempt_number=pending.attempt_number,
            step_id=step.step_id if step else "", operation_type=action.operation or action.intent,
            artifact_type=artifact_type, capability_id=capability,
            original_plan_ref=original_ref, verification_ref=verification_ref,
        )
        self.state.last_tool_failure = failure
        self.state.last_failed_action = action
        self.state.pending_task.failed_plan = self.state.failure_plans.get(original_ref, pending.plan)
        self.state.pending_task.committed_effects = any(
            r.status == ExecutionStatus.SUCCESS and r.capability in {
                "filesystem.atomic_replace", "filesystem.delete", "filesystem.move", "filesystem.rename"
            } and any(e.committed and e.task_semantic and e.user_visible for e in r.side_effects)
            for r in self.last_results
        ) or bool(self._artifact_bindings and self._artifact_bindings.task_id == pending.task_id and any(
            ref.state == ArtifactState.COMMITTED and ref.producer_step_id != "INPUT"
            for ref in self._artifact_bindings.bindings.values()))
        self.state.pending_task.status = "FAILED"
        self.state.pending_task.updated_at = time.time()
        self.audit.record(
            "conversation_tool_failure_state",
            turn_id=turn_id,
            task_id=task_id,
            stage=stage,
            error_code=error_code,
            safe_user_message=safe_user_message,
            retryable=retryable,
            failure_record=asdict(failure),
        )
        self._record_attempt_outcome("FAILED", turn_id=turn_id, failure=failure)

    def _record_attempt_outcome(self, status, *, turn_id, failure=None):
        pending = self.state.pending_task
        linked_failure = failure or self.state.last_tool_failure
        outcome = AttemptOutcome(pending.task_id, pending.plan_revision,
            pending.attempt_number, status,
            failure.original_plan_ref if failure else plan_reference(pending.plan),
            linked_failure.failure_id if linked_failure and linked_failure.task_id == pending.task_id else "")
        self.state.attempt_history.append(outcome)
        del self.state.attempt_history[:-64]
        self.audit.record("conversation_attempt_outcome", turn_id=turn_id, **asdict(outcome))

    def _finish_task(self, status: str, *, clear_action: bool) -> None:
        self.state.pending_task.status = status
        self.state.pending_task.updated_at = time.time()
        if self.state.pending_user_task is not None:
            self.state.pending_user_task = replace(
                self.state.pending_user_task,
                status=status,
            )
        if self.state.pending_structured_plan is not None:
            self.state.pending_structured_plan = replace(
                self.state.pending_structured_plan,
                status=status,
            )
        if clear_action:
            self.state.pending_task_action = None


def _missing_required_detail(action: PlannedToolAction) -> str:
    if action.intent == "invalid_semantic_creation":
        return "No pude preparar de forma segura el contenido solicitado. No creé ningún archivo."
    if action.intent == "invalid_document_transform":
        return "No pude preparar de forma segura esa edición. No hice ningún cambio."
    if action.intent == "invalid_document_conversion":
        return "No pude resolver de forma segura el documento de origen o el formato de conversión. No creé ningún archivo."
    if action.intent == "copy_missing_target":
        return "No pude identificar de forma inequívoca el archivo que quieres duplicar. ¿Cuál archivo es?"
    if action.intent == "needs_patch_details":
        return "Puedo corregirlo, pero necesito que me indiques qué texto debo reemplazar y por cuál."
    path_required = action.intent in {
        "read_document", "delete", "append_document", "structured_edit_text",
        "document_transform", "structural_document_edit", "generate_document_edit",
        "rename", "rename_missing_name", "run_python_file", "patch_code",
        "copy_artifact", "copy_missing_target",
    }
    if path_required and not action.target_path:
        return "Necesito la ruta exacta del archivo para hacerlo."
    if action.intent == "append_document" and not str(action.arguments.get("addition", "")).strip():
        return "¿Qué texto quieres que agregue al documento?"
    return ""


def _create_capability_for_path(path: str) -> str:
    suffix = Path(path).suffix.casefold()
    capabilities = {
        ".txt": "document.create_txt",
        ".md": "document.create_markdown",
        ".docx": "document.create_docx",
        ".pdf": "document.create_pdf",
    }
    if suffix not in capabilities:
        raise ValueError(f"unsupported_document_format:{suffix}")
    return capabilities[suffix]


def _missing_capability_text(capabilities: tuple[str, ...]) -> str:
    if any(item.startswith("media.video") for item in capabilities):
        return "No tengo todavía una herramienta de edición de video."
    return "Todavía no tengo una herramienta aprobada para hacer eso."


def _confirmation_text(effect: PermissionEffect, action: PlannedToolAction) -> str:
    path = action.target_path or str(action.arguments.get("path", ""))
    name = Path(path).name if path else "ese elemento"
    if effect == PermissionEffect.DELETE:
        if action.operation == "delete_collection":
            plan = DeletePlan.from_dict(dict(action.arguments.get("delete_plan", {})))
            files = "\n".join(f"- {item.display_name}" for item in plan.targets)
            return (
                f"Encontré {len(plan.targets)} archivos en el workspace:\n\n{files}\n\n"
                f"¿Quieres que elimine los {len(plan.targets)} archivos?"
            )
        return f"Voy a eliminar {name}. Esta acción no se puede deshacer. ¿Quieres que continúe?"
    if effect == PermissionEffect.OVERWRITE:
        return f"Voy a sobrescribir {name}. ¿Quieres que continúe?"
    if effect == PermissionEffect.MOVE:
        if action.intent == "rename":
            destination = str(action.arguments.get("new_name", "")).strip()
            return f"Voy a cambiar el nombre de {name} a {destination}. ¿Quieres que continúe?"
        return f"Voy a mover {name}. ¿Quieres que continúe?"
    if effect == PermissionEffect.EXECUTE_CODE:
        return "Voy a ejecutar ese código dentro del sandbox aislado. ¿Quieres que continúe?"
    if effect == PermissionEffect.CREATE:
        parent = str(Path(path).parent) if path else "el workspace autorizado"
        return (
            f"Crearé {name} en {_display_workspace(parent)}. "
            "¿Quieres que continúe?"
        )
    return "Esta acción necesita tu confirmación. ¿Quieres que continúe?"


def _friendly_success(action: PlannedToolAction, result: ToolExecutionResult) -> str:
    output = result.output
    path = str(output.get("path") or output.get("destination") or "")
    if action.intent == "read_document":
        content = str(output.get("content", "")).strip()
        preview = content[:900] + ("…" if len(content) > 900 else "")
        return f"Leí {Path(path).name}. Su contenido es:\n\n{preview}" if preview else f"Leí {Path(path).name}; está vacío."
    if action.intent == "show_full_path":
        return f"La ubicación completa es `{path}`."
    if action.intent in {"run_inline_python", "run_python_file"}:
        stdout = str(output.get("stdout", "")).strip()
        return f"El código terminó correctamente. Salida:\n\n{stdout}" if stdout else "El código terminó correctamente y no produjo salida."
    if action.intent == "generate_code":
        return (
            f"Listo, generé y verifiqué {Path(action.target_path).name} en "
            f"{_display_workspace(str(Path(action.target_path).parent))}."
        )
    if action.intent == "modify_code":
        return f"Listo, actualicé y verifiqué {Path(action.target_path).name}."
    if action.intent == "inspect_error":
        return f"El error parece ser {output.get('category', 'un error de ejecución')}: {output.get('summary', '')}"
    if action.intent == "delete":
        return f"Listo, eliminé {Path(path).name}."
    if action.intent == "delete_collection":
        count = len(action.arguments.get("delete_plan", {}).get("targets", ()))
        return f"Listo, eliminé y verifiqué {count} archivos del workspace."
    if action.intent in {"append_document", "structural_document_edit"}:
        return f"Listo, actualicé {Path(path).name}."
    if action.intent in {"structured_edit_text", "document_transform"}:
        changes = int(
            result.output.get("applied_changes", result.output.get("actual_changes", 0)) or 0
        )
        detail = f" ({changes} cambios verificados)" if changes else ""
        return f"Listo, actualicé y verifiqué {Path(path).name}{detail}."
    if action.intent == "generate_document_edit":
        return f"Listo, generé el contenido y actualicé {Path(path).name} con verificación física."
    if action.intent == "compress_zip":
        return _friendly_created(path, noun="ZIP")
    if action.intent in {"convert_document", "text_files_to_pdf"}:
        return _friendly_created(path, noun="PDF")
    if action.intent == "rename":
        return f"Listo, cambié el nombre a {Path(path).name} en {_display_workspace(str(Path(path).parent))}."
    if action.intent == "copy_artifact":
        return f"Listo, dupliqué {Path(path).name} en {_display_workspace(str(Path(path).parent))}."
    if action.intent == "patch_code":
        return f"Listo, corregí {Path(path).name}."
    return _friendly_created(path) if path else "Listo, la acción terminó correctamente."




def _friendly_failure(result: ToolExecutionResult) -> str:
    error = result.error
    if result.status == ExecutionStatus.VERIFY_FAILED:
        return (
            "La operación no superó la verificación del archivo, así que no voy a afirmar "
            "que el cambio quedó aplicado."
        )
    if "DENIED_CORE_PROTECTED" in error:
        return "No puedo acceder a esa ruta porque pertenece al núcleo protegido de Anydes."
    if "DENIED_OUTSIDE_AUTHORIZED_SCOPE" in error:
        return "No pude hacerlo porque esa carpeta no está autorizada."
    if "FileExistsError" in error:
        return "No pude crear el archivo porque ya existe y no tengo autorización para sobrescribirlo."
    if "FileNotFoundError" in error:
        return "No encontré el archivo o la carpeta indicada."
    if error == "no_text_files_found":
        return "No encontré archivos TXT en esa carpeta."
    if result.capability.startswith("developer.run"):
        return "El código terminó con un error. Puedo revisar el detalle en el siguiente paso."
    return "No pude completar la acción de forma segura."


def _failure_details(
    action: PlannedToolAction,
    result: ToolExecutionResult,
) -> tuple[str, str, str, bool]:
    error = str(result.error or "").strip()
    if error == "semantic_prose_generator_unavailable":
        return (
            "CONTENT_GENERATION",
            "CONTENT_GENERATOR_UNAVAILABLE",
            "el generador de contenido no estaba disponible",
            True,
        )
    if error == "self_context_unavailable":
        return (
            "SELF_CONTEXT",
            "SELF_CONTEXT_UNAVAILABLE",
            "el contexto de identidad permitido no estaba disponible",
            True,
        )
    if error.startswith("document_render_failed:"):
        return (
            "RENDER",
            "RENDER_FAILED",
            "el renderer no pudo construir el documento temporal",
            True,
        )
    if error.startswith("content_generation_failed:"):
        return (
            "CONTENT_GENERATION",
            "CONTENT_GENERATION_FAILED",
            "el cerebro no devolvió contenido documental válido",
            True,
        )
    if error.startswith("semantic_provider_interrupted:"):
        return (
            "CONTENT_GENERATION",
            "CONTENT_PROVIDER_INTERRUPTED",
            "el proveedor se interrumpió antes de devolver contenido",
            True,
        )
    if error == "semantic_prose_response_schema_invalid":
        return (
            "CONTENT_GENERATION",
            "CONTENT_RESPONSE_INVALID",
            "el cerebro devolvió una respuesta con formato de control inválido",
            True,
        )
    if error == "generated_content_empty":
        return "CONTENT_VALIDATION", "CONTENT_EMPTY", "el contenido generado quedó vacío", True
    if error == "document_conversion_source_empty":
        return (
            "VERIFY",
            "DOCUMENT_CONVERSION_SOURCE_EMPTY",
            "el documento de origen no contenía texto extraíble, por eso no inicié la conversión",
            True,
        )
    if error == "general_document_source_empty":
        return (
            "VERIFY",
            "DOCUMENT_SOURCE_EMPTY",
            "el documento intermedio no contenía texto verificable",
            True,
        )
    if error.startswith("generated_content_"):
        return (
            "CONTENT_VALIDATION",
            "VALIDATION_FAILED",
            _content_validation_failure_message(error),
            True,
        )
    if result.status == ExecutionStatus.VERIFY_FAILED:
        return (
            "VERIFY",
            "VERIFY_FAILED",
            "el resultado físico no superó la verificación requerida",
            True,
        )
    if "DENIED_CORE_PROTECTED" in error:
        return "EXECUTION", "DENIED_CORE_PROTECTED", _friendly_failure(result), False
    if "DENIED_OUTSIDE_AUTHORIZED_SCOPE" in error:
        return "EXECUTION", "DENIED_OUTSIDE_AUTHORIZED_SCOPE", _friendly_failure(result), False
    if "FileExistsError" in error:
        return "EXECUTION", "TARGET_ALREADY_EXISTS", _friendly_failure(result), False
    if "FileNotFoundError" in error:
        return "EXECUTION", "TARGET_NOT_FOUND", _friendly_failure(result), True
    return (
        "EXECUTION",
        "TOOL_EXECUTION_FAILED",
        _friendly_failure(result),
        True,
    )


def _content_validation_failure_message(error: str) -> str:
    return {
        "generated_content_contains_tool_markup": (
            "el contenido generado contenía texto de control no permitido"
        ),
        "generated_content_contains_placeholder": (
            "el contenido generado contenía marcadores incompletos"
        ),
        "generated_content_off_topic": (
            "el contenido generado no quedó suficientemente relacionado con el tema pedido"
        ),
        "generated_content_length_out_of_tolerance": (
            "la longitud generada quedó fuera del margen solicitado"
        ),
    }.get(error, "el contenido generado no superó la validación local")


def _semantic_generation_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message.startswith("semantic_provider_interrupted:"):
        return message
    if message == "semantic_prose_response_schema_invalid":
        return message
    return f"content_generation_failed:{type(exc).__name__}"


def _is_retry_request(text: str) -> bool:
    normalized = _normalize_confirmation(text)
    return normalized in {
        "reintenta", "reintentalo", "reinténtalo", "intenta otra vez",
        "intentalo de nuevo", "inténtalo de nuevo", "prueba otra vez",
        "hazlo tu entonces", "entonces hazlo tu", "hazlo entonces",
    }


def _is_task_resume_request(text: str) -> bool:
    normalized = _normalize_confirmation(text)
    direct = {
        "hazlo", "crealo", "generalo", "preparalo", "intenta hacerlo",
        "intentalo", "entonces crealo", "entonces hazlo",
    }
    if normalized in direct:
        return True
    has_action = bool(re.search(r"\b(?:haz|crea|genera|prepara|continua|intenta)\w*\b", normalized))
    has_referent = bool(re.search(r"\b(?:documento|archivo|tarea|pedido|eso|lo)\b", normalized))
    return has_action and has_referent


def _next_code_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("no_available_code_name")


def _normalize_confirmation(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    ascii_like = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", ascii_like)).strip()


def _filename_from_followup(text: str, target_path: str) -> str:
    value = re.sub(
        r"^\s*(?:ponle|llamalo|llamala|que\s+se\s+llame|con\s+nombre)\s+",
        "",
        str(text),
        flags=re.IGNORECASE,
    ).strip(" \t\r\n.,;:\"'")
    supplied = Path(value)
    safe = sanitize_windows_stem(supplied.stem if supplied.suffix else value)
    if not safe:
        return ""
    suffix = supplied.suffix.casefold() or Path(target_path).suffix.casefold()
    if suffix not in {".txt", ".md", ".docx", ".pdf", ".py", ".cpp", ".zip"}:
        return ""
    return f"{safe}{suffix}"


def _display_workspace(path: str) -> str:
    name = Path(path).resolve(strict=False).name
    return name or "el workspace autorizado"


def _friendly_created(path: str, *, noun: str = "archivo") -> str:
    artifact = Path(path)
    return f"Listo, creé {artifact.name} en {_display_workspace(str(artifact.parent))}."


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: str) -> str:
    return _display_workspace(path)
