"""Narrow internal/public boundary for a declared task continuation only."""
import json

INTERNAL_FIELDS={'output':'INTERNAL_OUTPUT_FIELD_LEAK',
    'resolution_context':'INTERNAL_RESOLUTION_CONTEXT_LEAK'}


def project(value, state, record=None):
    from .classifier_output import pending_task
    from .operation_intent import ResolutionSchemaError
    if not isinstance(value,dict) or value.get('intent')!='CONTINUE_TASK':
        return value
    # A discriminator must be valid before discarding known non-authority metadata.
    if value.get('speech_act') not in {'ACTION_REQUEST','INFORMATION_REQUEST','ACKNOWLEDGEMENT','SOCIAL'}:
        return value
    proposal=value.get('task_proposal')
    if isinstance(proposal,dict) and 'task_id' in proposal:
        raise ResolutionSchemaError('task_proposal.task_id','TASK_PROPOSAL_ID_NOT_ALLOWED')
    if 'task_proposal' in value:
        raise ResolutionSchemaError('task_proposal','CONTINUATION_PUBLIC_CONTRACT_VIOLATION')
    if not pending_task(state):
        return value  # No projection can create the missing pending-task authority.
    removed=sorted(set(value)&INTERNAL_FIELDS.keys())
    if record is not None and removed:
        record('continuation_public_projection',branch='TASK_CONTINUATION',
            internal_field_names=removed,reason_codes=[INTERNAL_FIELDS[k] for k in removed],
            pending_task_exists=True,values_logged=False)
    return {key:item for key,item in value.items() if key not in removed}


def instructions(request, task):
    """No output/publication-shaped instruction keys or examples in the producer input."""
    from .resolution_contract import branch_contract
    contract=branch_contract({'intent':'CONTINUE_TASK','speech_act':'ACKNOWLEDGEMENT'},request.get('state'))
    rules={'pending_task_exists':True,'existing_task_id':task['task_id'],
        'allowed_public_fields':contract['allowed_fields'],
        'required_continuation_fields':['intent','speech_act','primary_goal','confidence','task_update'],
        'forbidden_fields':['output','resolution_context','branch_contract','task_request_contract',
            'task_proposal','task_id'],
        'allowed_update_fields':['task_id','relation','requirements','clarification_target',
            'requirement_items','planning_readiness','operation','artifact_type'],
        'allowed_update_relations':['CONTINUE','REVISE']}
    if request.get('task_continuation_contract'):
        rules['repair_target_intent']='CONTINUE_TASK'
    return ('\nPENDING TASK CLASSIFICATION - instructions, not response fields.\n'
        'Determine the semantic relation of the current turn to the pending task. '
        'An answer to its clarification or requirement delta is CONTINUE_TASK, not CREATE_NEW. '
        'A genuinely unrelated request may retain its own branch; pending context is not authority to change intent. '
        'For CONTINUE_TASK emit one minimal PUBLIC instance: intent, speech_act, primary_goal, confidence, '
        'task_update. The delta references the exact existing task_id inside task_update only. '
        'Use the exact unresolved concept keys; never invent an answer or overwrite other requirements. '
        'Never echo input context, publication text, schemas, helper objects or instruction keys. '
        'Do not use task_proposal for continuation. '
        +json.dumps(rules,ensure_ascii=False)+
        '\nWhen repairing CONTINUE_TASK, preserve that branch and exact existing ID. '
        'Correct only the reported fields; no new task, confirmation, response text or authority.\n')


def enforce_repair(value, request):
    from .operation_intent import ResolutionSchemaError
    if request.get('task_continuation_contract') and (
            not isinstance(value,dict) or value.get('intent')!='CONTINUE_TASK'):
        raise ResolutionSchemaError('intent','CONTINUATION_PUBLIC_CONTRACT_VIOLATION')
