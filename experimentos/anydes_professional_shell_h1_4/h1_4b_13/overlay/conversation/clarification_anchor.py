"""Task-scoped clarification is a state of an encargo, not an orphan response."""
import hashlib
import json

INSTRUCTIONS='''
CLARIFICATION ANCHOR CONTRACT (instructions only):
Distinguish missing information in an actual creation/action request from a genuinely
conversational clarification. For CLARIFICATION explicitly declare clarification_scope:
TASK or CONVERSATION. A valid task_proposal already implies TASK.
TASK with no existing PendingTask MUST include task_proposal: relation START/SWITCH,
operation CREATE_NEW/EDIT_EXISTING, artifact_type, known requirements, requirement_items
with REQUIRED unresolved definitions/questions and clarification_target {key,question}.
primary_goal remains the user's operational goal, not the act of asking a question.
Do not invent known values, identifiers, permissions, artifacts or planner data.
The host persists the incomplete proposal and assigns its ID BEFORE asking the question.
TASK with existing PendingTask can refer to that existing anchor; an answer resolving
its clarification is CONTINUE_TASK with the existing task_update, never a new proposal.
CONVERSATION means no operational task was requested: do not use it to discard a
creation request just because its language, requirements or referent are incomplete.
If an earlier JSON was malformed, reconstruct from the ORIGINAL user request and
trusted context, never from arbitrary unparsed fields. Preserve the request's task
class even if a material requirement remains missing. Return a public instance only.
'''


def validate(value,state,*,model_boundary=False,task_seed=None):
    from .operation_intent import ResolutionSchemaError
    from .classifier_output import pending_task
    if not isinstance(value,dict):
        return
    scope=value.get('clarification_scope')
    if 'clarification_scope' in value and (value.get('intent')!='CLARIFICATION' or scope not in {'TASK','CONVERSATION'}):
        raise ResolutionSchemaError('clarification_scope','clarification_scope_invalid')
    if value.get('intent')!='CLARIFICATION':
        return
    proposal=value.get('task_proposal')
    if scope is None and proposal is None and 'explanation_relation' in value:
        return  # Existing failure-reference clarification keeps its own strict contract.
    context=(state or {}).get('resolution_context') or {}
    if (scope is None and proposal is None and value.get('pending_action_relation')=='AMBIGUOUS'
            and (context.get('pending_confirmation') or {}).get('available')):
        return  # Clarifying an existing confirmation is not a new operational task.
    task=bool(proposal) or bool(pending_task(state))
    operational=(scope=='TASK' or proposal is not None or value.get('speech_act')=='ACTION_REQUEST'
        or bool(task_seed and task_seed.get('preserve_goal_class') in {'NEW_ARTIFACT_TASK','EDIT_ARTIFACT_TASK'}))
    if scope=='CONVERSATION' and operational:
        raise ResolutionSchemaError('clarification_scope','operational_request_cannot_be_conversational')
    if operational and not task:
        raise ResolutionSchemaError('task_proposal','OPERATIONAL_CLARIFICATION_WITHOUT_TASK_ANCHOR')
    if model_boundary and not task and scope is None:
        raise ResolutionSchemaError('clarification_scope','CLARIFICATION_SCOPE_REQUIRED')


def parse_evidence(error,text,outcome,*,repair_available):
    """Observe parser result only. Never alter extraction or retain its doc/message."""
    encoded=str(text or '').encode('utf-8')
    reason=getattr(outcome,'done_reason',None)
    truncated='YES' if reason in {'length','max_tokens'} else 'NO' if reason in {'stop','eos','completed'} else 'UNKNOWN'
    structure=isinstance(error,ValueError) and not isinstance(error,json.JSONDecodeError)
    return {'reason_code':'MODEL_JSON_TRUNCATED' if truncated=='YES' else
            'MODEL_JSON_INVALID_STRUCTURE' if structure else 'MODEL_JSON_PARSE_FAILURE',
        'parse_failure_category':'NON_OBJECT_JSON' if structure else 'INVALID_JSON_SYNTAX',
        'parser_position':getattr(error,'pos',None),'parser_line':getattr(error,'lineno',None),
        'parser_column':getattr(error,'colno',None),'position_basis':'PARSER_CANDIDATE',
        'truncated_status':truncated,'response_byte_length':len(encoded),
        'response_sha256':hashlib.sha256(encoded).hexdigest(),
        'obvious_outer_object_detected':str(text or '').strip().startswith('{'),
        'repair_invoked':repair_available,'raw_response_retained':False}


def reconstruction_contract(state):
    from .classifier_output import pending_task
    return {'original_semantic_branch_candidate':'UNKNOWN_AFTER_PARSE_FAILURE',
        'pending_task_exists':bool(pending_task(state)),
        'safe_retained_classifier_hints':[],
        'context_authority':'ORIGINAL_USER_REQUEST_AND_TRUSTED_STATE_ONLY',
        'operational_clarification_required_fields':['intent','speech_act','primary_goal',
            'confidence','clarification_scope','task_proposal'],
        'conditional_rule':'If the original user requests creation/action, preserve that semantic task. If it needs clarification and there is no PendingTask, emit TASK scope with a valid ID-free task_proposal and unresolved definitions/target. Pure conversational clarification must explicitly declare CONVERSATION. Do not invent a task for a non-operational turn.'}
