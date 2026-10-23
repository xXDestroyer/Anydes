"""Local operation contract. Context selects targets, never the requested effect."""
from dataclasses import dataclass, asdict
from enum import Enum


class ArtifactOperation(str, Enum):
    NONE = 'NO_ARTIFACT_ACTION'
    CREATE = 'CREATE_NEW'
    EDIT = 'EDIT_EXISTING'
    READ = 'READ_EXISTING'
    REPLACE = 'REPLACE_EXISTING'
    EXECUTE = 'EXECUTE_EXISTING'
    DELETE = 'DELETE_EXISTING'
    CONVERT = 'CONVERT_EXISTING'
    ARCHIVE = 'ARCHIVE_EXISTING'
    MOVE = 'MOVE_EXISTING'
    RENAME = 'RENAME_EXISTING'


@dataclass(frozen=True)
class RequestedEffects:
    create: bool = False
    mutate: bool = False
    read: bool = False
    execute: bool = False
    delete: bool = False
    replace: bool = False
    version: bool = False

    def to_dict(self):
        return asdict(self)


EFFECTS = {
    ArtifactOperation.NONE: RequestedEffects(),
    ArtifactOperation.CREATE: RequestedEffects(create=True),
    ArtifactOperation.EDIT: RequestedEffects(mutate=True, read=True, version=True),
    ArtifactOperation.READ: RequestedEffects(read=True),
    ArtifactOperation.REPLACE: RequestedEffects(mutate=True, replace=True, version=True),
    ArtifactOperation.EXECUTE: RequestedEffects(read=True, execute=True),
    ArtifactOperation.DELETE: RequestedEffects(delete=True),
    ArtifactOperation.CONVERT: RequestedEffects(create=True, read=True),
    ArtifactOperation.ARCHIVE: RequestedEffects(create=True, read=True),
    ArtifactOperation.MOVE: RequestedEffects(mutate=True),
    ArtifactOperation.RENAME: RequestedEffects(mutate=True),
}

OPERATIONS = {
    ArtifactOperation.CREATE: {'GENERATE_CODE', 'CREATE_DOCUMENT', 'GENERATE_CONTENT', 'TEST_CODE', 'EXECUTE_CODE'},
    ArtifactOperation.EDIT: {'EDIT_ARTIFACT', 'TEST_CODE', 'EXECUTE_CODE'},
    ArtifactOperation.READ: {'READ_ARTIFACT', 'READ_DOCUMENT'},
    ArtifactOperation.REPLACE: {'EDIT_ARTIFACT'},
    ArtifactOperation.EXECUTE: {'TEST_CODE', 'EXECUTE_CODE'},
    ArtifactOperation.DELETE: {'DELETE_ARTIFACT'},
    ArtifactOperation.CONVERT: {'CONVERT_DOCUMENT'},
    ArtifactOperation.ARCHIVE: {'CREATE_ARCHIVE'},
    ArtifactOperation.MOVE: {'MOVE_ARTIFACT'},
    ArtifactOperation.RENAME: {'RENAME_ARTIFACT'},
}


def operation_errors(decision):
    intent = decision.operation_intent
    if intent == ArtifactOperation.NONE:
        return ()  # Prepared pre-contract decisions retain their existing validators.
    errors = []
    if set(decision.requested_operations) - OPERATIONS[intent]:
        errors.append('operation_intent_conflict:' + intent.value)
    if intent == ArtifactOperation.CREATE and decision.references:
        errors.append('create_new_cannot_require_existing_target')
    if intent == ArtifactOperation.READ and (decision.requires_execution or
            decision.needs_content_generation or decision.effect_intent.mutate or
            decision.effect_intent.replace or decision.effect_intent.create):
        errors.append('read_existing_must_be_non_mutating')
    expected = EFFECTS[intent]
    # Execution may accompany creation or editing, but cannot authorize mutation.
    actual = decision.effect_intent
    for name, required in expected.to_dict().items():
        if name == 'read' and intent == ArtifactOperation.REPLACE:
            continue
        if name == 'execute' and intent in {ArtifactOperation.CREATE, ArtifactOperation.EDIT}:
            continue
        if getattr(actual, name) != required:
            errors.append('requested_effect_conflict:' + name)
    return tuple(errors)


def resolution_payload(value):
    """Project one semantic wire decision; no second classifier or text rules."""
    required = {'turn_relation', 'operation_intent', 'conversation_purpose',
                'output_kind', 'requested_operations', 'confidence', 'primary_goal'}
    if not required <= value.keys():
        raise ValueError('interaction_missing_fields:' + ','.join(sorted(required-value.keys())))
    intent = ArtifactOperation(value['operation_intent'])
    relation = value['turn_relation']
    controls = {'FAILURE_CAUSE':'EXPLAIN_FAILURE', 'FAILURE_DETAIL':'CONTINUE_EXPLANATION',
        'FAILURE_REMEDIATION':'REMEDIATION_QUERY', 'PENDING_CANCEL':'CANCEL',
        'PENDING_CONFIRMATION':'CONFIRM', 'FAILED_TASK_RETRY':'RETRY',
        'QUALITY_ACCEPTANCE':'RETAIN_UNVERIFIED'}
    control = controls.get(relation, 'NONE')
    if control != 'NONE' and intent != ArtifactOperation.NONE:
        raise ValueError('interaction_control_operation_conflict')
    if 'task_control' in value and value['task_control'] != control:
        raise ValueError('interaction_relation_control_conflict')
    readonly = control in {'EXPLAIN_FAILURE', 'CONTINUE_EXPLANATION', 'REMEDIATION_QUERY', 'CANCEL'}
    if readonly and (value['requested_operations'] not in ([], ['EXPLAIN']) or
                     value['output_kind'] != 'NONE' or value.get('requires_execution', False)):
        raise ValueError('interaction_readonly_control_conflict')
    result = dict(value)
    result['task_control'] = control
    result['kind'] = ('CLARIFICATION' if relation == 'AMBIGUOUS' else
        'CONTENT_RESPONSE' if readonly else
        'TOOL_ACTION' if intent != ArtifactOperation.NONE or control != 'NONE' or relation == 'OFFER_ACCEPTANCE' else
        value.get('kind', 'CONVERSATION'))
    effects = EFFECTS[intent].to_dict()
    effects['execute'] = bool(value.get('requires_execution', effects['execute']))
    result['effect_intent'] = effects
    result['requested_effects'] = [k for k, v in effects.items() if v]
    result['artifact_intent'] = intent != ArtifactOperation.NONE
    if readonly:
        result['verification_requirement'] = 'NONE'
    return result


class ResolutionSchemaError(ValueError):
    """Safe field-level rejection metadata; never embeds a model value."""
    def __init__(self, field, constraint):
        self.field, self.constraint = field, constraint
        super().__init__('interaction_schema:' + field + ':' + constraint)


def semantic_resolution_payload(value, state, *, record=None):
    from .resolution_contract import canonicalize_branch_fields, branch_contract
    try:
        return _semantic_resolution_payload(canonicalize_branch_fields(value, record, state), state)
    except ResolutionSchemaError as error:
        # Preserve only bounded protocol enums, never rejected free text.
        from .interaction_router import TaskControl, TurnRelation, ExplanationRelation
        from .verification_types import VerificationRequirement
        fields = {}
        from .discourse_context import IdentityAttribute
        allowed = {'intent':{c.value for c in TaskControl} | {'PERSONAL_IDENTITY'},
            'conversation_purpose':{'GENERAL','PRE_REQUEST'},
            'identity_attribute':{a.value for a in IdentityAttribute},
            'speech_act':{'ACTION_REQUEST','INFORMATION_REQUEST','SOCIAL','ACKNOWLEDGEMENT'},
            'turn_relation':{r.value for r in TurnRelation},
            'verification_requirement':{r.value for r in VerificationRequirement},
            'explanation_relation':{r.value for r in ExplanationRelation}}
        if isinstance(value, dict):
            for name, choices in allowed.items():
                if isinstance(value.get(name),str) and value[name] in choices:
                    fields[name] = value[name]
            if type(value.get('execute')) is bool:
                fields['execute'] = value['execute']
        error.resolution_fields = fields
        error.branch_contract = branch_contract(value,state)
        if error.field == 'conversation_purpose':
            error.pre_request_contract = {
                'declared_intent':value.get('intent'),
                'declared_purpose':value.get('conversation_purpose') if isinstance(value.get('conversation_purpose'),str)
                    and value['conversation_purpose'] in {'GENERAL','PRE_REQUEST'} else 'INVALID',
                'allowed_purposes':['GENERAL','PRE_REQUEST'],
                'constraint':'PRE_REQUEST prepares a future request without a concrete objective; GENERAL is actual casual chat; CLARIFICATION requires an attempted concrete request with genuinely missing information. Reevaluate the original turn in context; never infer operational authority.'}
        raise


def _semantic_resolution_payload(value, state):
    """Single semantic discriminator -> existing runtime contract, no text parsing."""
    from .interaction_router import TaskControl, TurnRelation, ExplanationRelation, PendingActionRelation
    from .discourse_context import ConversationPurpose
    from .verification_types import VerificationRequirement
    required = {'intent', 'speech_act', 'primary_goal', 'confidence'}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ResolutionSchemaError('response', 'missing_required_fields')
    allowed = required | {'artifact_type', 'references', 'execute', 'verification_requirement',
        'turn_relation', 'subjects', 'ambiguities', 'explanation_relation', 'execution_intent', 'pending_action_relation',
        'identity_attribute','conversation_purpose'}
    if set(value)-allowed:
        raise ResolutionSchemaError('response', 'unexpected_fields')
    intent = value['intent']
    operations = {a.value:a for a in ArtifactOperation if a != ArtifactOperation.NONE}
    conversational = {'CONVERSATION','CONTENT_RESPONSE','PERSONAL_MEMORY_RECALL',
        'SELF_PREFERENCE','SELF_RELATIONAL','PERSONAL_IDENTITY','ENTITY_QUERY','TOPIC_CLOSURE','CAPABILITY_QUERY',
        'OFFER_ACCEPTANCE','CLARIFICATION','UNSUPPORTED','PRE_REQUEST'}
    controls = {c.value for c in TaskControl if c != TaskControl.NONE}
    if not isinstance(intent, str) or intent not in set(operations) | conversational | controls:
        raise ResolutionSchemaError('intent','unsupported_value')
    purpose = value.get('conversation_purpose')
    if purpose is not None:
        if not isinstance(purpose, str) or purpose not in {'GENERAL','PRE_REQUEST'}:
            raise ResolutionSchemaError('conversation_purpose','unsupported_value')
        if purpose == 'PRE_REQUEST' and intent != 'PRE_REQUEST':
            raise ResolutionSchemaError('conversation_purpose','pre_request_intent_required')
        if intent == 'PRE_REQUEST' and purpose != 'PRE_REQUEST':
            raise ResolutionSchemaError('conversation_purpose','pre_request_purpose_required')
    act = value['speech_act']
    from .discourse_context import IdentityAttribute
    try:
        identity_attribute = IdentityAttribute(value.get('identity_attribute', 'UNSPECIFIED'))
    except (ValueError, TypeError):
        raise ResolutionSchemaError('identity_attribute', 'unsupported_value') from None
    if identity_attribute != IdentityAttribute.UNSPECIFIED and intent != 'PERSONAL_IDENTITY':
        raise ResolutionSchemaError('identity_attribute', 'identity_domain_required')
    if intent == 'PERSONAL_IDENTITY':
        if identity_attribute == IdentityAttribute.UNSPECIFIED:
            raise ResolutionSchemaError('identity_attribute', 'required_for_identity')
        if act != 'INFORMATION_REQUEST':
            raise ResolutionSchemaError('speech_act', 'identity_information_request_required')
    try:
        explanation_relation = ExplanationRelation(value.get('explanation_relation', 'UNSPECIFIED'))
    except (ValueError, TypeError):
        raise ResolutionSchemaError('explanation_relation', 'unsupported_value') from None
    if not isinstance(act,str) or act not in {'ACTION_REQUEST','INFORMATION_REQUEST','SOCIAL','ACKNOWLEDGEMENT'}:
        raise ResolutionSchemaError('speech_act','unsupported_value')
    if intent == 'PRE_REQUEST' and act not in {'SOCIAL','ACKNOWLEDGEMENT'}:
        raise ResolutionSchemaError('speech_act','pre_request_is_non_operational')
    operation = operations.get(intent, ArtifactOperation.NONE)
    try:
        pending_relation = PendingActionRelation(value.get('pending_action_relation','UNSPECIFIED'))
    except (ValueError, TypeError):
        raise ResolutionSchemaError('pending_action_relation','unsupported_value') from None
    control = intent if intent in controls else 'NONE'
    informational = {'EXPLAIN_FAILURE','CONTINUE_EXPLANATION','REMEDIATION_QUERY'}
    action = operation != ArtifactOperation.NONE or control in {'CONFIRM','CANCEL','RETRY','RETAIN_UNVERIFIED'} or intent == 'OFFER_ACCEPTANCE'
    valid_action_act = act == 'ACTION_REQUEST' or (
        operation == ArtifactOperation.READ and act == 'INFORMATION_REQUEST') or (
        intent in {'CONFIRM','CANCEL','OFFER_ACCEPTANCE','RETAIN_UNVERIFIED'} and act == 'ACKNOWLEDGEMENT')
    if (action and not valid_action_act) or (intent in informational and act != 'INFORMATION_REQUEST'):
        raise ResolutionSchemaError('intent','speech_act_conflict')
    context = state.get('resolution_context') or {}
    focus = context.get('conversation_artifact_focus') or ()
    artifact_type = value.get('artifact_type')
    if artifact_type is None and operation != ArtifactOperation.CREATE and len(focus) == 1:
        artifact_type = focus[0].get('artifact_type')
    if artifact_type is not None and (not isinstance(artifact_type,str) or artifact_type not in {'CODE','DOCUMENT','TEXT','ARCHIVE'}):
        raise ResolutionSchemaError('artifact_type','unsupported_value')
    if operation == ArtifactOperation.CREATE and artifact_type not in {'CODE','DOCUMENT','TEXT'}:
        raise ResolutionSchemaError('artifact_type','create_type_required')
    for key in ('references','ambiguities'):
        if key in value and (not isinstance(value[key],list) or any(not isinstance(x,str) for x in value[key])):
            raise ResolutionSchemaError(key,'string_array_required')
    if 'execute' in value and type(value['execute']) is not bool:
        raise ResolutionSchemaError('execute','boolean_required')
    execution_intent = value.get('execution_intent', 'UNSPECIFIED')
    if execution_intent not in {'UNSPECIFIED', 'NONE', 'RUN', 'TEST'}:
        raise ResolutionSchemaError('execution_intent', 'unsupported_value')
    execution_requested = execution_intent in {'RUN', 'TEST'}
    verification = value.get('verification_requirement',
        'TEST_SUITE' if execution_intent == 'TEST' else
        'EXECUTION' if operation == ArtifactOperation.EXECUTE or value.get('execute') is True or execution_requested else
        'BASIC' if operation in {ArtifactOperation.CREATE,ArtifactOperation.EDIT,ArtifactOperation.REPLACE} else 'NONE')
    if not isinstance(verification,str) or verification not in {v.value for v in VerificationRequirement}:
        raise ResolutionSchemaError('verification_requirement','unsupported_value')
    execute = operation == ArtifactOperation.EXECUTE or execution_requested or value.get('execute', verification in {'EXECUTION','TEST_SUITE'})
    if execution_requested and (value.get('execute') is False or verification not in {'EXECUTION','TEST_SUITE'}):
        raise ResolutionSchemaError('execution_intent', 'composite_execution_dropped')
    if execute and verification not in {'EXECUTION','TEST_SUITE'}:
        raise ResolutionSchemaError('verification_requirement','execution_verifier_required')
    if operation == ArtifactOperation.READ and (execute or verification != 'NONE'):
        raise ResolutionSchemaError('verification_requirement','read_is_non_operational')
    if control in informational | {'CANCEL'} and (execute or verification != 'NONE'):
        raise ResolutionSchemaError('execute','readonly_control_has_effect')
    if intent == 'PRE_REQUEST' and (execute or verification != 'NONE' or value.get('artifact_type') is not None
            or value.get('references')):
        raise ResolutionSchemaError('intent','pre_request_has_operational_metadata')
    if verification == 'TEST_SUITE' and not execute:
        raise ResolutionSchemaError('execute','test_suite_requires_execution')
    relations = {'EXPLAIN_FAILURE':'FAILURE_CAUSE','CONTINUE_EXPLANATION':'FAILURE_DETAIL',
        'REMEDIATION_QUERY':'FAILURE_REMEDIATION','CANCEL':'PENDING_CANCEL','CONFIRM':'PENDING_CONFIRMATION',
        'RETRY':'FAILED_TASK_RETRY','RETAIN_UNVERIFIED':'QUALITY_ACCEPTANCE',
        'TOPIC_CLOSURE':'TOPIC_CLOSURE','OFFER_ACCEPTANCE':'OFFER_ACCEPTANCE','CLARIFICATION':'AMBIGUOUS'}
    relation = value.get('turn_relation', relations.get(intent,
        'NEW_ACTION' if operation == ArtifactOperation.CREATE else
        'ARTIFACT_FOLLOWUP' if operation != ArtifactOperation.NONE else 'NEW_TOPIC'))
    if not isinstance(relation,str) or relation not in {r.value for r in TurnRelation} - {'UNSPECIFIED'}:
        raise ResolutionSchemaError('turn_relation','unsupported_value')
    if intent in relations and relation != relations[intent]:
        raise ResolutionSchemaError('turn_relation','intent_relation_conflict')
    if operation != ArtifactOperation.NONE and relation not in {'NEW_ACTION','ARTIFACT_FOLLOWUP','NEW_CORRECTIVE_ACTION'}:
        raise ResolutionSchemaError('turn_relation','current_operation_not_failure_question')
    canonical = {
        ArtifactOperation.CREATE:'GENERATE_CODE' if artifact_type == 'CODE' else 'CREATE_DOCUMENT',
        ArtifactOperation.EDIT:'EDIT_ARTIFACT', ArtifactOperation.REPLACE:'EDIT_ARTIFACT',
        ArtifactOperation.READ:'READ_ARTIFACT', ArtifactOperation.EXECUTE:'TEST_CODE',
        ArtifactOperation.DELETE:'DELETE_ARTIFACT', ArtifactOperation.ARCHIVE:'CREATE_ARCHIVE',
        ArtifactOperation.CONVERT:'CONVERT_DOCUMENT', ArtifactOperation.MOVE:'MOVE_ARTIFACT',
        ArtifactOperation.RENAME:'RENAME_ARTIFACT'}
    requested = [canonical[operation]] if operation != ArtifactOperation.NONE else ['EXPLAIN'] if control in informational else []
    if execute and operation in {ArtifactOperation.CREATE,ArtifactOperation.EDIT,ArtifactOperation.REPLACE}:
        requested.append('TEST_CODE' if verification == 'TEST_SUITE' else 'EXECUTE_CODE')
    # Executing is not always an independent test suite.
    if operation == ArtifactOperation.EXECUTE and verification != 'TEST_SUITE':
        requested = ['EXECUTE_CODE']
    purpose = {'PERSONAL_MEMORY_RECALL':'PERSONAL_MEMORY','SELF_PREFERENCE':'SELF_PREFERENCE',
        'SELF_RELATIONAL':'SELF_RELATIONAL','PERSONAL_IDENTITY':'PERSONAL_IDENTITY',
        'ENTITY_QUERY':'ENTITY_QUERY','TOPIC_CLOSURE':'ACKNOWLEDGEMENT',
        'PRE_REQUEST':'PRE_REQUEST'}.get(intent,'GENERAL')
    recall = (context.get('discourse') or {}).get('memory_recall') or {}
    if intent in {'CONVERSATION','CONTENT_RESPONSE'} and relation == 'TOPIC_CONTINUATION' and recall.get('active'):
        purpose = 'PERSONAL_MEMORY'
    output = ({'CODE':'CODE_ARTIFACT','DOCUMENT':'DOCUMENT_ARTIFACT','TEXT':'DOCUMENT_ARTIFACT'}.get(artifact_type,'TOOL_RESULT')
        if operation in {ArtifactOperation.CREATE,ArtifactOperation.EDIT,ArtifactOperation.REPLACE} else
        'ARCHIVE_ARTIFACT' if operation == ArtifactOperation.ARCHIVE else
        'DOCUMENT_ARTIFACT' if operation == ArtifactOperation.CONVERT else
        'TOOL_RESULT' if operation != ArtifactOperation.NONE else 'NONE')
    effects = EFFECTS[operation].to_dict()
    effects['execute'] = execute
    if intent == 'OFFER_ACCEPTANCE':
        offer = (context.get('discourse') or {}).get('pending_offer') or {}
        requested = [offer['operation_kind']] if offer else []
    return {'kind': 'CLARIFICATION' if intent == 'CLARIFICATION' else 'UNSUPPORTED' if intent == 'UNSUPPORTED' else
            'TOOL_ACTION' if action and control not in informational | {'CANCEL'} else
            'CONTENT_RESPONSE' if act == 'INFORMATION_REQUEST' or control == 'CANCEL' else 'CONVERSATION',
        'output_kind':output, 'operation_intent':operation.value, 'task_control':control,
        'effect_intent':effects, 'requested_effects':[k for k,v in effects.items() if v],
        'artifact_intent':operation != ArtifactOperation.NONE,
        'requested_operations':requested, 'requires_execution':execute,
        'verification_requirement':verification, 'turn_relation':relation,
        'explanation_relation':explanation_relation.value,
        'pending_action_relation':pending_relation.value,
        'identity_attribute':identity_attribute.value,
        'conversation_purpose':purpose, 'primary_goal':value['primary_goal'],
        'confidence':value['confidence'], 'references':value.get('references',[]),
        'ambiguities':value.get('ambiguities',[]), 'subjects':value.get('subjects',[]),
        'needs_content_generation':operation in {ArtifactOperation.CREATE,ArtifactOperation.EDIT,ArtifactOperation.REPLACE},
        'capability_query':intent == 'CAPABILITY_QUERY'}
