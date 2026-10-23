from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Protocol

from .contracts import PlannedToolAction
from .general_reasoning import GeneralTaskPlan
from .verification_types import VerificationRequirement
from .contract_canonicalization import ContractCanonicalizer
from .discourse_context import ConversationPurpose, DiscourseEntity
from .operation_intent import ArtifactOperation, RequestedEffects, operation_errors


class InteractionCategory(str, Enum):
    CONVERSATION = "CONVERSATION"
    QUESTION = "QUESTION"
    ACTION_REQUEST = "ACTION_REQUEST"
    ACTION_FOLLOWUP = "ACTION_FOLLOWUP"
    FAILURE_QUERY = "FAILURE_QUERY"
    CAPABILITY_QUERY = "CAPABILITY_QUERY"


class RequestedOutcome(str, Enum):
    NONE = "NONE"
    CODE_ARTIFACT = "CODE_ARTIFACT"
    DOCUMENT_ARTIFACT = "DOCUMENT_ARTIFACT"
    ARCHIVE_ARTIFACT = "ARCHIVE_ARTIFACT"
    FILESYSTEM_EFFECT = "FILESYSTEM_EFFECT"
    TOOL_RESULT = "TOOL_RESULT"


@dataclass(frozen=True)
class InteractionDecision:
    category: InteractionCategory
    confidence: float
    primary_goal: str = ""
    requested_outcome: RequestedOutcome = RequestedOutcome.NONE
    needs_content_generation: bool = False
    references_existing_artifact: bool = False
    reason: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "InteractionDecision":
        try:
            category = InteractionCategory(str(value.get("category", "CONVERSATION")).upper())
            outcome = RequestedOutcome(str(value.get("requested_outcome", "NONE")).upper())
        except ValueError as exc:
            raise ValueError("interaction_classification_invalid") from exc
        return cls(
            category=category,
            confidence=max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
            primary_goal=str(value.get("primary_goal", "")).strip(),
            requested_outcome=outcome,
            needs_content_generation=bool(value.get("needs_content_generation", False)),
            references_existing_artifact=bool(
                value.get("references_existing_artifact", False)
            ),
            reason=str(value.get("reason", "")).strip()[:240],
        )


class InteractionKind(str, Enum):
    CONVERSATION = "CONVERSATION"
    CONTENT_RESPONSE = "CONTENT_RESPONSE"
    TOOL_ACTION = "TOOL_ACTION"
    CLARIFICATION = "CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"


class TaskControl(str, Enum):
    NONE = "NONE"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    RETRY = "RETRY"
    EXPLAIN_FAILURE = "EXPLAIN_FAILURE"
    CONTINUE_EXPLANATION = "CONTINUE_EXPLANATION"
    REMEDIATION_QUERY = "REMEDIATION_QUERY"
    RETAIN_UNVERIFIED = "RETAIN_UNVERIFIED"


# Read-only controls concerning the active failure, never execution authority.
EXPLANATION_CONTROLS = {TaskControl.EXPLAIN_FAILURE, TaskControl.CONTINUE_EXPLANATION,
    TaskControl.REMEDIATION_QUERY}


class TurnRelation(str, Enum):
    UNSPECIFIED = 'UNSPECIFIED'  # Deprecated prepared/legacy contracts only.
    NEW_TOPIC = 'NEW_TOPIC'
    TOPIC_CONTINUATION = 'TOPIC_CONTINUATION'
    TOPIC_CLOSURE = 'TOPIC_CLOSURE'
    OFFER_ACCEPTANCE = 'OFFER_ACCEPTANCE'
    NEW_ACTION = 'NEW_ACTION'
    FAILURE_CAUSE = 'FAILURE_CAUSE'
    FAILURE_DETAIL = 'FAILURE_DETAIL'
    FAILURE_REMEDIATION = 'FAILURE_REMEDIATION'
    NEW_CORRECTIVE_ACTION = 'NEW_CORRECTIVE_ACTION'
    PENDING_CONFIRMATION = 'PENDING_CONFIRMATION'
    PENDING_CANCEL = 'PENDING_CANCEL'
    FAILED_TASK_RETRY = 'FAILED_TASK_RETRY'
    ARTIFACT_FOLLOWUP = 'ARTIFACT_FOLLOWUP'
    QUALITY_ACCEPTANCE = 'QUALITY_ACCEPTANCE'
    AMBIGUOUS = 'AMBIGUOUS'


FAILURE_RELATIONS = {TurnRelation.FAILURE_CAUSE, TurnRelation.FAILURE_DETAIL,
    TurnRelation.FAILURE_REMEDIATION}


class ExplanationRelation(str, Enum):
    UNSPECIFIED = 'UNSPECIFIED'
    CAUSE = 'ASK_FAILURE_CAUSE'
    ELABORATION = 'ELABORATE_PREVIOUS_EXPLANATION'
    REMEDIATION = 'ASK_REMEDIATION'
    NEW_ACTION = 'NEW_ACTION'
    TOPIC_CHANGE = 'TOPIC_CHANGE'
    AMBIGUOUS = 'AMBIGUOUS'


class PendingActionRelation(str, Enum):
    UNSPECIFIED = 'UNSPECIFIED'
    ACCEPT = 'ACCEPT_PENDING_ACTION'
    REJECT = 'REJECT_PENDING_ACTION'
    NEW_ACTION = 'NEW_ACTION'
    AMBIGUOUS = 'AMBIGUOUS'
    UNRELATED = 'UNRELATED'


@dataclass(frozen=True)
class ResolutionContext:
    pending_confirmation: dict
    active_task: dict
    active_failure_topic: dict
    conversation_artifact_focus: tuple[dict, ...]
    recent_action: dict
    previous_assistant_act: str
    current_turn_relation: str = 'UNRESOLVED'
    discourse: dict | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class InteractionResolution:
    kind: InteractionKind
    output_kind: RequestedOutcome = RequestedOutcome.NONE
    requested_effects: tuple[str, ...] = ()
    artifact_intent: bool = False
    task_control: TaskControl = TaskControl.NONE
    references: tuple[str, ...] = ()
    confidence: float = 0.0
    ambiguities: tuple[str, ...] = ()
    source: str = "SEMANTIC_CLASSIFIER"
    primary_goal: str = ""
    needs_content_generation: bool = False
    capability_query: bool = False
    requested_operations: tuple[str, ...] = ()
    requires_execution: bool = False
    verification_requirement: VerificationRequirement = VerificationRequirement.NONE
    failure_ref: str = ""
    turn_relation: TurnRelation = TurnRelation.UNSPECIFIED
    conversation_purpose: ConversationPurpose = ConversationPurpose.GENERAL
    subjects: tuple[DiscourseEntity, ...] = ()
    operation_intent: ArtifactOperation = ArtifactOperation.NONE
    effect_intent: RequestedEffects = RequestedEffects()
    explanation_relation: ExplanationRelation = ExplanationRelation.UNSPECIFIED
    pending_action_relation: PendingActionRelation = PendingActionRelation.UNSPECIFIED
    identity_attribute: str = 'UNSPECIFIED'

    @classmethod
    def from_dict(cls, value):
        effects = value.get('effect_intent', {})
        if (not isinstance(effects, dict) or set(effects)-set(RequestedEffects.__dataclass_fields__)
                or any(type(v) is not bool for v in effects.values())):
            raise ValueError('requested_effects_schema_invalid')
        if 'subjects' in value and (not isinstance(value['subjects'], list) or len(value['subjects']) > 4):
            raise ValueError('discourse_subjects_invalid')
        for key in ("requested_effects", "references", "ambiguities", "requested_operations"):
            if key in value and (not isinstance(value[key], list) or
                                 any(not isinstance(item, str) for item in value[key])):
                raise ValueError("interaction_array_invalid:" + key)
        for key in ("artifact_intent", "needs_content_generation", "capability_query", "requires_execution"):
            if key in value and type(value[key]) is not bool:
                raise ValueError("interaction_boolean_invalid:" + key)
        return cls(
            kind=InteractionKind(value["kind"]),
            output_kind=RequestedOutcome(value.get("output_kind", "NONE")),
            requested_effects=tuple(value.get("requested_effects", ())),
            artifact_intent=value.get("artifact_intent", False),
            task_control=TaskControl(value.get("task_control", "NONE")),
            references=tuple(value.get("references", ())),
            confidence=ContractCanonicalizer.interaction(value)["confidence"],
            ambiguities=tuple(value.get("ambiguities", ())),
            primary_goal=str(value.get("primary_goal", ""))[:500],
            needs_content_generation=value.get("needs_content_generation", False),
            capability_query=value.get("capability_query", False),
            requested_operations=tuple(value.get("requested_operations", ())),
            requires_execution=value.get("requires_execution", False),
            verification_requirement=VerificationRequirement(value.get("verification_requirement", "NONE")),
            turn_relation=TurnRelation(value.get('turn_relation', 'UNSPECIFIED')),
            conversation_purpose=ConversationPurpose(value.get('conversation_purpose', 'GENERAL')),
            subjects=tuple(DiscourseEntity.parse(e) for e in value.get('subjects', ())),
            operation_intent=ArtifactOperation(value.get('operation_intent', 'NO_ARTIFACT_ACTION')),
            effect_intent=RequestedEffects(**effects),
            explanation_relation=ExplanationRelation(value.get('explanation_relation', 'UNSPECIFIED')),
            pending_action_relation=PendingActionRelation(value.get('pending_action_relation','UNSPECIFIED')),
            identity_attribute=cls._identity_attribute(value),
        )

    @staticmethod
    def _identity_attribute(value):
        from .discourse_context import IdentityAttribute
        return IdentityAttribute(value.get('identity_attribute', 'UNSPECIFIED')).value

    # Temporary read-only view for the existing goal validator and observation harness.
    @property
    def category(self):
        if self.task_control in EXPLANATION_CONTROLS:
            return InteractionCategory.FAILURE_QUERY
        if self.capability_query:
            return InteractionCategory.CAPABILITY_QUERY
        return {
            InteractionKind.CONVERSATION: InteractionCategory.CONVERSATION,
            InteractionKind.CONTENT_RESPONSE: InteractionCategory.QUESTION,
            InteractionKind.TOOL_ACTION: InteractionCategory.ACTION_REQUEST,
        }.get(self.kind, InteractionCategory.CONVERSATION)

    @property
    def requested_outcome(self):
        return self.output_kind

    @property
    def references_existing_artifact(self):
        return bool(self.references)


def adapt_legacy_decision(value: InteractionDecision) -> InteractionResolution:
    """Deprecated classifier output adapter, not a second routing decision."""
    kind = {
        InteractionCategory.CONVERSATION: InteractionKind.CONVERSATION,
        InteractionCategory.QUESTION: InteractionKind.CONTENT_RESPONSE,
        InteractionCategory.ACTION_REQUEST: InteractionKind.TOOL_ACTION,
        InteractionCategory.ACTION_FOLLOWUP: InteractionKind.TOOL_ACTION,
        InteractionCategory.FAILURE_QUERY: InteractionKind.CONTENT_RESPONSE,
        InteractionCategory.CAPABILITY_QUERY: InteractionKind.CONTENT_RESPONSE,
    }[value.category]
    return InteractionResolution(
        kind=kind, output_kind=value.requested_outcome,
        artifact_intent=value.requested_outcome != RequestedOutcome.NONE,
        task_control=(TaskControl.EXPLAIN_FAILURE if value.category == InteractionCategory.FAILURE_QUERY
                      else TaskControl.NONE),
        confidence=value.confidence, primary_goal=value.primary_goal,
        needs_content_generation=value.needs_content_generation,
        capability_query=value.category == InteractionCategory.CAPABILITY_QUERY,
        source="LEGACY_CLASSIFIER_ADAPTER",
    )


class InteractionResolver:
    """The only active turn decision. It never receives a precomputed tool plan."""
    def __init__(self, classifier=None):
        self.classifier = classifier

    @staticmethod
    def contradictions(value, state):
        errors = list(operation_errors(value))
        context = state.get('resolution_context') or {}
        pending = context.get('pending_confirmation') or {}
        relation = value.pending_action_relation
        if (pending.get('available') and pending.get('requested_action') and
                context.get('previous_assistant_act') == 'CONFIRMATION_REQUEST' and
                value.kind == InteractionKind.TOOL_ACTION and value.task_control == TaskControl.NONE
                and relation == PendingActionRelation.UNSPECIFIED):
            errors.append('pending_action_relation_required')
        if value.task_control == TaskControl.CONFIRM and (value.requested_effects or
                value.operation_intent != ArtifactOperation.NONE or value.requested_operations or
                value.references or value.requires_execution or value.needs_content_generation):
            errors.append('confirmation_cannot_extend_pending_action')
        if relation in {PendingActionRelation.ACCEPT, PendingActionRelation.REJECT}:
            if not state.get('pending_action') or not pending.get('available', True):
                errors.append('pending_relation_without_pending_action')
            expected_pending = TaskControl.CONFIRM if relation == PendingActionRelation.ACCEPT else TaskControl.CANCEL
            if value.task_control != expected_pending:
                errors.append('pending_acceptance_control_conflict')
        if (relation == PendingActionRelation.NEW_ACTION and
                value.task_control in {TaskControl.CONFIRM, TaskControl.CANCEL}):
            errors.append('new_action_cannot_authorize_pending')
        following_explanation = (state.get('failure_available') and
            context.get('previous_assistant_act') == 'FAILURE_EXPLANATION')
        expected_control = {
            ExplanationRelation.CAUSE: TaskControl.EXPLAIN_FAILURE,
            ExplanationRelation.ELABORATION: TaskControl.CONTINUE_EXPLANATION,
            ExplanationRelation.REMEDIATION: TaskControl.REMEDIATION_QUERY,
        }.get(value.explanation_relation)
        if expected_control and value.task_control != expected_control:
            errors.append('explanation_semantic_relation_conflict')
        if following_explanation and value.kind == InteractionKind.CLARIFICATION:
            if (value.explanation_relation == ExplanationRelation.UNSPECIFIED and
                    value.requested_operations != ('EXPLAIN',)):
                errors.append('explanation_followup_relation_required')
            elif value.explanation_relation == ExplanationRelation.AMBIGUOUS and not value.ambiguities:
                errors.append('explanation_ambiguity_basis_required')
        if not 0 <= value.confidence <= 1:
            errors.append("confidence_invalid")
        operational = any(op not in {"GENERATE_CONTENT", "EXPLAIN", "NONE"} for op in value.requested_operations)
        side_effect_need = (value.requested_effects or value.artifact_intent or value.requires_execution
            or operational or (value.output_kind != RequestedOutcome.NONE and not value.capability_query)
            or value.verification_requirement in {VerificationRequirement.EXECUTION,VerificationRequirement.TEST_SUITE,
                VerificationRequirement.CONVERSION_PRESERVATION})
        if value.kind in {InteractionKind.CONTENT_RESPONSE, InteractionKind.CONVERSATION} and side_effect_need:
            errors.append("non_action_has_effect_or_artifact")
        if value.kind == InteractionKind.TOOL_ACTION and not (
            side_effect_need
            or value.task_control in {TaskControl.CONFIRM, TaskControl.RETRY, TaskControl.CANCEL, TaskControl.RETAIN_UNVERIFIED}
        ):
            errors.append("action_requires_tool_need")
        if value.task_control == TaskControl.RETRY and not state["retry_available"]:
            errors.append("retry_without_failed_plan")
        if (value.task_control == TaskControl.CONFIRM or
                value.task_control == TaskControl.CANCEL and value.turn_relation == TurnRelation.UNSPECIFIED) and not state["pending_action"]:
            errors.append("control_without_pending_action")
        if value.task_control in EXPLANATION_CONTROLS and not state["failure_available"]:
            errors.append("explain_without_failure")
        expected = {
            TurnRelation.FAILURE_CAUSE: TaskControl.EXPLAIN_FAILURE,
            TurnRelation.FAILURE_DETAIL: TaskControl.CONTINUE_EXPLANATION,
            TurnRelation.FAILURE_REMEDIATION: TaskControl.REMEDIATION_QUERY,
            TurnRelation.PENDING_CONFIRMATION: TaskControl.CONFIRM,
            TurnRelation.PENDING_CANCEL: TaskControl.CANCEL,
            TurnRelation.FAILED_TASK_RETRY: TaskControl.RETRY,
            TurnRelation.QUALITY_ACCEPTANCE: TaskControl.RETAIN_UNVERIFIED,
        }.get(value.turn_relation)
        if expected and value.task_control != expected:
            errors.append('relation_control_conflict')
        if value.turn_relation in FAILURE_RELATIONS and not state['failure_available']:
            errors.append('failure_relation_without_topic')
        if value.task_control in EXPLANATION_CONTROLS:
            if value.kind != InteractionKind.CONTENT_RESPONSE or side_effect_need:
                errors.append('explanation_requires_readonly_content')
            if value.turn_relation not in FAILURE_RELATIONS | {TurnRelation.UNSPECIFIED}:
                errors.append('explanation_relation_conflict')
        if (state.get('failure_available') and value.requested_operations == ('EXPLAIN',)
                and value.task_control == TaskControl.NONE
                and value.turn_relation == TurnRelation.UNSPECIFIED):
            errors.append('explanation_relation_required')
        if value.turn_relation in {TurnRelation.NEW_TOPIC, TurnRelation.TOPIC_CLOSURE} and value.task_control != TaskControl.NONE:
            errors.append('new_topic_has_task_control')
        if value.turn_relation in {TurnRelation.NEW_ACTION, TurnRelation.NEW_CORRECTIVE_ACTION, TurnRelation.ARTIFACT_FOLLOWUP} and value.task_control != TaskControl.NONE:
            errors.append('new_action_has_task_control')
        if value.turn_relation in {TurnRelation.NEW_ACTION, TurnRelation.NEW_CORRECTIVE_ACTION, TurnRelation.ARTIFACT_FOLLOWUP} and value.kind != InteractionKind.TOOL_ACTION:
            errors.append('action_relation_requires_tool_action')
        if value.turn_relation == TurnRelation.AMBIGUOUS and value.kind != InteractionKind.CLARIFICATION:
            errors.append('ambiguous_relation_requires_clarification')
        if value.turn_relation == TurnRelation.OFFER_ACCEPTANCE:
            offer = (state.get('resolution_context', {}).get('discourse') or {}).get('pending_offer')
            if not offer:
                errors.append('acceptance_without_supported_offer')
            elif (value.kind != InteractionKind.TOOL_ACTION or value.task_control != TaskControl.NONE
                    or value.requested_operations != (offer['operation_kind'],)):
                errors.append('offer_acceptance_operation_mismatch')
        if value.task_control == TaskControl.RETAIN_UNVERIFIED and not state.get('retention_available'):
            errors.append('retention_without_draft')
        if value.task_control in {TaskControl.CONFIRM, TaskControl.RETRY, TaskControl.RETAIN_UNVERIFIED} and value.kind != InteractionKind.TOOL_ACTION:
            errors.append("execution_control_requires_action")
        active_failure = state.get('active_failure', {})
        if (value.kind == InteractionKind.CLARIFICATION and value.requested_operations == ('EXPLAIN',)
                and active_failure.get('relevant') and active_failure.get('explained')
                and value.turn_relation == TurnRelation.UNSPECIFIED):
            errors.append('explanation_topic_available_not_unresolved')
        if (value.requested_operations == ('CREATE_DOCUMENT',)
                and value.output_kind == RequestedOutcome.CODE_ARTIFACT):
            errors.append('operation_output_type_incompatible')
        if value.capability_query and value.kind == InteractionKind.TOOL_ACTION:
            errors.append("capability_query_conflicts_with_action")
        if "TEST_CODE" in value.requested_operations and value.verification_requirement != VerificationRequirement.TEST_SUITE:
            errors.append("test_intent_requires_test_suite")
        if value.conversation_purpose == ConversationPurpose.PERSONAL_IDENTITY:
            if value.identity_attribute == 'UNSPECIFIED':
                errors.append('identity_attribute_required')
            if (value.kind != InteractionKind.CONTENT_RESPONSE or value.task_control != TaskControl.NONE
                    or side_effect_need or value.references):
                errors.append('identity_requires_readonly_question')
        return tuple(errors)

    def resolve(self, text, *, state):
        request = {"user_request": text[:12000], "state": state}
        explanation_root = ""
        explanation_control = None
        identity_branch = False
        self.validation_history = []
        if self.classifier is not None:
            for attempt in range(2):
                try:
                    value = self.classifier(request)
                    if isinstance(value, InteractionDecision):
                        value = adapt_legacy_decision(value)
                    if not isinstance(value, InteractionResolution):
                        raise ValueError("interaction_contract_invalid")
                    if value.conversation_purpose == ConversationPurpose.PERSONAL_IDENTITY:
                        identity_branch = True
                    # Only an exact duplicate of the declared read-only control is
                    # representational. Real operations/effects remain contradictions.
                    if value.task_control in EXPLANATION_CONTROLS:
                        value = replace(value, requested_operations=tuple(
                            'EXPLAIN' if op == value.task_control.value else op for op in value.requested_operations),
                            failure_ref=state.get('active_failure', {}).get('failure_id', ''))
                        explanation_root = value.failure_ref
                        if explanation_control is None:
                            explanation_control = value.task_control
                    errors = self.contradictions(value, state)
                    if identity_branch and value.conversation_purpose != ConversationPurpose.PERSONAL_IDENTITY:
                        errors += ('identity_repair_changed_branch',)
                    if value.turn_relation in FAILURE_RELATIONS:
                        explanation_root = state.get('active_failure', {}).get('failure_id', '')
                    if explanation_root and value.task_control not in EXPLANATION_CONTROLS:
                        errors += ('explanation_repair_changed_control',)
                    elif explanation_root and explanation_control is not None and value.task_control != explanation_control:
                        errors += ('explanation_repair_changed_control',)
                    self.last_validation = {"attempt": attempt+1, "errors": errors,
                        "kind": value.kind.value, "requested_effects": value.requested_effects,
                        "artifact_intent": value.artifact_intent, "requested_operations": value.requested_operations,
                        "requires_execution": value.requires_execution,
                        "task_control":value.task_control.value, "turn_relation":value.turn_relation.value,
                        "explanation_relation":value.explanation_relation.value,
                        "pending_action_relation":value.pending_action_relation.value,
                        "verification_requirement": value.verification_requirement.value}
                    if identity_branch:
                        self.last_validation['identity_attribute'] = value.identity_attribute
                    self.validation_history.append(self.last_validation.copy())
                    threshold = .95 if state.get("destructive_confirmation") and value.task_control == TaskControl.CONFIRM else .65
                    if not errors and value.confidence >= threshold:
                        return value
                    if 'explanation_topic_available_not_unresolved' in errors:
                        explanation_root = state.get('active_failure', {}).get('failure_id', '')
                    request = {**request, "repair_errors": errors or ("confidence_insufficient",),
                        "allowed_task_controls":[c.value for c in TaskControl],
                        "preserve_explanation": {"failure_ref":explanation_root,
                            "task_control":{TurnRelation.FAILURE_DETAIL:'CONTINUE_EXPLANATION',
                                TurnRelation.FAILURE_REMEDIATION:'REMEDIATION_QUERY'}.get(
                                    value.turn_relation,'EXPLAIN_FAILURE')}
                            if explanation_root else None}
                    if any(e.startswith(('pending_', 'new_action_cannot_authorize_pending',
                            'confirmation_cannot_extend_pending_action')) for e in errors):
                        request['pending_confirmation_contract'] = {
                            'pending_action':(state.get('resolution_context') or {}).get('pending_confirmation',{}),
                            'previous_assistant_act':(state.get('resolution_context') or {}).get('previous_assistant_act'),
                            'allowed_controls':['CONFIRM','CANCEL','NONE'],
                            'constraint':'Acceptance resumes only the pending task/plan/effects; a genuinely different action is NEW_ACTION.',
                            'fields_in_conflict':self.last_validation}
                    if any(error in errors for error in (
                            'explanation_semantic_relation_conflict',
                            'explanation_followup_relation_required',
                            'explanation_ambiguity_basis_required')):
                        context = state.get('resolution_context') or {}
                        request.update(continuation_context={
                            'previous_assistant_act':context.get('previous_assistant_act'),
                            'active_failure_topic':context.get('active_failure_topic', {}),
                            'failure_ref':state.get('active_failure', {}).get('failure_id', ''),
                            'allowed_discourse_relations':[r.value for r in ExplanationRelation
                                if r != ExplanationRelation.UNSPECIFIED],
                            'clarification_requires':'A genuine unresolved referent or user goal, not a request for explanation depth'},
                            fields_in_conflict=self.last_validation)
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as exc:
                    reason = str(exc)[:180] if isinstance(exc, ValueError) and str(exc).startswith(('interaction_', 'operation_', 'requested_', 'read_', 'create_')) else 'classification_unavailable'
                    fields = getattr(exc, 'resolution_fields', {})
                    if (fields.get('intent') == 'PERSONAL_IDENTITY'
                            and fields.get('speech_act') == 'INFORMATION_REQUEST'):
                        identity_branch = True
                    if (fields.get('intent') in {c.value for c in EXPLANATION_CONTROLS}
                            and fields.get('speech_act') == 'INFORMATION_REQUEST'
                            and state.get('failure_available')):
                        explanation_root = state.get('active_failure', {}).get('failure_id', '')
                        explanation_control = TaskControl(fields['intent'])
                    self.last_validation = {'attempt':attempt+1, 'errors':(reason,),
                        'conflicting_fields':fields}
                    self.validation_history.append(self.last_validation.copy())
                    request = {**request, "repair_errors": (reason,)}
                    if getattr(exc, 'branch_contract', None):
                        request['branch_contract'] = exc.branch_contract
                    if getattr(exc, 'pre_request_contract', None):
                        request['pre_request_contract'] = exc.pre_request_contract
                if identity_branch:
                    from .discourse_context import IdentityAttribute
                    request['identity_attribute_contract'] = {
                        'branch':'IDENTITY', 'intent':'PERSONAL_IDENTITY',
                        'speech_act':'INFORMATION_REQUEST',
                        'current_identity_attribute':self.last_validation.get('identity_attribute',
                            self.last_validation.get('conflicting_fields',{}).get('identity_attribute','UNSPECIFIED')),
                        'allowed_identity_attributes':[a.value for a in IdentityAttribute
                            if a != IdentityAttribute.UNSPECIFIED],
                        'constraint':'Preserve the identity question and context; complete only the requested attribute, never its value or operational effects.'}
                if explanation_root and explanation_control is not None:
                    request.update(preserve_explanation={
                        'failure_ref':explanation_root,'task_control':explanation_control.value},
                        allowed_explanation_contract={
                            'intent':explanation_control.value,'speech_act':'INFORMATION_REQUEST',
                            'kind':'CONTENT_RESPONSE','failure_ref':explanation_root,
                            'requested_effects':[], 'requires_execution':False,
                            'verification_requirement':'NONE',
                            'model_fields':['intent','speech_act','primary_goal','confidence']},
                        fields_in_conflict=self.last_validation,
                        allowed_task_controls=[explanation_control.value])
        if identity_branch:
            return InteractionResolution(InteractionKind.CLARIFICATION,
                ambiguities=('identity_attribute_unresolved',),
                source='IDENTITY_ATTRIBUTE_UNRESOLVED', confidence=1.0)
        return InteractionResolution(
            InteractionKind.CLARIFICATION, ambiguities=("interaction_unresolved",),
            source="LOCAL_FAIL_CLOSED", confidence=1.0,
        )


class InteractionClassifier(Protocol):
    def __call__(self, request: dict) -> InteractionDecision:
        ...


class InteractionRouter:
    """Deprecated pre-C API. The staging controller never instantiates this router."""

    def __init__(self, classifier: InteractionClassifier | None = None) -> None:
        self.classifier = classifier

    def route(
        self,
        text: str,
        *,
        deterministic_action: PlannedToolAction | None,
        capability_candidate: bool,
        failure_candidate: bool,
        has_last_failure: bool,
        has_active_artifact: bool,
        has_pending_action: bool,
        unresolved_action_candidate: bool = False,
    ) -> InteractionDecision:
        if self.classifier is not None:
            try:
                decision = self.classifier({
                    "user_request": str(text)[:12_000],
                    "state": {
                        "has_last_failure": has_last_failure,
                        "has_active_artifact": has_active_artifact,
                        "has_pending_action": has_pending_action,
                        "deterministic_candidate": deterministic_action is not None,
                        "capability_candidate": capability_candidate,
                        "failure_candidate": failure_candidate,
                    },
                })
                if decision.confidence >= 0.65:
                    return decision
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        if failure_candidate and has_last_failure:
            return InteractionDecision(
                InteractionCategory.FAILURE_QUERY,
                1.0,
                primary_goal="explain_recent_tool_failure",
            )
        if capability_candidate and deterministic_action is None:
            return InteractionDecision(
                InteractionCategory.CAPABILITY_QUERY,
                1.0,
                primary_goal="describe_current_capabilities",
            )
        if deterministic_action is not None:
            return InteractionDecision(
                InteractionCategory.ACTION_REQUEST,
                0.9,
                primary_goal=deterministic_action.intent,
                requested_outcome=_outcome_for_action(deterministic_action),
                references_existing_artifact=bool(deterministic_action.target_path),
            )
        if unresolved_action_candidate:
            return InteractionDecision(
                InteractionCategory.ACTION_REQUEST,
                0.7,
                primary_goal="resolve_requested_side_effect",
            )
        return InteractionDecision(InteractionCategory.CONVERSATION, 0.5)


class GoalConsistencyValidator:
    """Ensure the dominant user goal survives planning and lowering."""

    @staticmethod
    def validate_plan(
        decision: InteractionDecision,
        plan: GeneralTaskPlan,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        outcome = decision.requested_outcome
        intent = getattr(decision, 'operation_intent', ArtifactOperation.NONE)
        if intent == ArtifactOperation.CREATE and (plan.task_type == 'MODIFY_CODE' or
                any(s.capability in {'document.edit_document', 'filesystem.edit_text',
                    'filesystem.delete', 'filesystem.rename', 'filesystem.move'} for s in plan.steps)):
            errors.append('create_new_plan_contains_existing_target_mutation')
        if intent == ArtifactOperation.READ and any(s.capability not in {
                'filesystem.read', 'document.read_document', 'developer.read_code'} for s in plan.steps):
            errors.append('read_existing_plan_contains_non_read_operation')
        if outcome == RequestedOutcome.CODE_ARTIFACT and plan.task_type not in {
            "GENERATE_CODE", "MODIFY_CODE", "TOOL_COMPOSITION",
        }:
            errors.append("primary_goal_requires_code_artifact")
        if outcome == RequestedOutcome.DOCUMENT_ARTIFACT and plan.task_type not in {
            "GENERATE_DOCUMENT", "TOOL_COMPOSITION",
        }:
            errors.append("primary_goal_requires_document_artifact")
        if outcome == RequestedOutcome.ARCHIVE_ARTIFACT and plan.task_type not in {
            "TOOL_COMPOSITION", "GENERATE_CODE",
        }:
            errors.append("primary_goal_requires_archive_artifact")
        if decision.needs_content_generation and not plan.needs_content_generation:
            errors.append("primary_goal_requires_content_generation")
        return tuple(errors)

    @staticmethod
    def validate_action(
        decision: InteractionDecision,
        action: PlannedToolAction,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        outcome = decision.requested_outcome
        intent = getattr(decision, 'operation_intent', ArtifactOperation.NONE)
        if intent == ArtifactOperation.CREATE and action.operation in {
                'general_modify_code', 'patch_code', 'generated_document_edit', 'structural_document_edit'}:
            errors.append('create_new_action_contains_existing_target_mutation')
        if getattr(decision,"verification_requirement",VerificationRequirement.NONE) == VerificationRequirement.TEST_SUITE:
            spec = action.arguments.get("code_generation_spec", {})
            if not spec.get("tests") or not spec.get("execution_required"):
                errors.append("deterministic_action_lost_verification_requirement")
        if outcome == RequestedOutcome.CODE_ARTIFACT and action.operation not in {
            "general_generate_code", "general_modify_code", "run_inline_python",
            "patch_code",
        }:
            errors.append("deterministic_action_lost_code_goal")
        if outcome == RequestedOutcome.DOCUMENT_ARTIFACT and action.operation not in {
            "verified_create_document", "generated_create_document",
            "general_generated_document", "document_conversion",
            "generated_document_edit", "structural_document_edit",
        }:
            errors.append("deterministic_action_lost_document_goal")
        if (
            decision.needs_content_generation
            and action.operation == "verified_create_document"
        ):
            errors.append("semantic_content_requires_generation_pipeline")
        return tuple(errors)


def _outcome_for_action(action: PlannedToolAction) -> RequestedOutcome:
    if any(capability.startswith("developer.") for capability in action.capabilities):
        return RequestedOutcome.CODE_ARTIFACT
    if any(capability.startswith("document.") for capability in action.capabilities):
        return RequestedOutcome.DOCUMENT_ARTIFACT
    if any(capability.startswith("archive.") for capability in action.capabilities):
        return RequestedOutcome.ARCHIVE_ARTIFACT
    if any(capability.startswith("filesystem.") for capability in action.capabilities):
        return RequestedOutcome.FILESYSTEM_EFFECT
    return RequestedOutcome.TOOL_RESULT
