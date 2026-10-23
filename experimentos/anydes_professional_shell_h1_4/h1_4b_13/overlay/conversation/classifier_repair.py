"""Preserve recoverable task semantics during the existing single schema repair."""
from .resolution_contract import branch_contract


def task_seed(value,state):
    if not isinstance(value,dict):
        return None
    intent=value.get('intent');act=value.get('speech_act')
    if not isinstance(intent,str) or not isinstance(act,str):
        return None
    update=value.get('task_proposal') or value.get('task_update')
    task_clarification=(intent=='CLARIFICATION' and (value.get('clarification_scope')=='TASK' or
        isinstance(update,dict) and isinstance(update.get('relation'),str) and update['relation'] in {'START','SWITCH'}))
    if not (intent in {'CREATE_NEW','EDIT_EXISTING'} and act=='ACTION_REQUEST'
            or task_clarification and act in {'INFORMATION_REQUEST','ACTION_REQUEST'}):
        return None
    goal=value.get('primary_goal')
    if not isinstance(goal,str) or not goal.strip() or len(goal)>2000:
        return None
    contract=branch_contract(value,state)
    seed={'intent':intent,'speech_act':act,'primary_goal':goal,
        'declared_branch':contract['declared_branch'],
        'required_fields':sorted({'intent','speech_act','primary_goal','confidence'}|set(contract['required_fields'])),
        'allowed_fields':contract['allowed_fields'],'forbidden_fields':contract['forbidden_fields'],
        'preserve_goal_class':'NEW_ARTIFACT_TASK' if intent=='CREATE_NEW' or task_clarification else 'EDIT_ARTIFACT_TASK',
        'instruction':'Correct only the exact structural errors, preserving intent, speech_act, primary_goal and known operational semantics. Do not turn a task clarification into CREATE_NEW without a new user answer. Never return ACKNOWLEDGEMENT, IDENTITY or EXPLANATION for this repair. clarification_target belongs only inside task_proposal for a new task or task_update for a continuation, never at the root. Keep the original user context; never invent permissions, execution or facts.'}
    if intent in {'CREATE_NEW','EDIT_EXISTING'}:
        seed['operation']=intent
    if isinstance(update,dict):
        if isinstance(update.get('operation'),str) and update['operation'] in {'CREATE_NEW','EDIT_EXISTING'}:
            seed['operation']=update['operation']
        if isinstance(update.get('relation'),str) and update['relation'] in {'START','SWITCH'}:
            seed['task_relation']=update['relation']
        if isinstance(update.get('artifact_type'),str) and update['artifact_type'] in {'CODE','DOCUMENT','TEXT'}:
            seed['artifact_type']=update['artifact_type']
    for field,choices in {'artifact_type':{'CODE','DOCUMENT','TEXT'},
            'execution_intent':{'NONE','RUN','TEST'},
            'verification_requirement':{'NONE','BASIC','EXECUTION','TEST_SUITE'}}.items():
        if isinstance(value.get(field),str) and value[field] in choices:
            seed[field]=value[field]
    if type(value.get('execute')) is bool and not (value['execute'] is False and value.get('execution_intent') in {'RUN','TEST'}):
        seed['execute']=value['execute']
    if (value.get('execution_intent') in {'RUN','TEST'}
            and seed.get('verification_requirement') in {'NONE','BASIC'}):
        seed.pop('verification_requirement')
    seed['schema']={'type':'object','required':seed['required_fields'],
        'additionalProperties':False,'properties':{key:{} for key in seed['allowed_fields']}}
    for field in ('intent','speech_act','primary_goal'):
        seed['schema']['properties'][field]={'const':seed[field]}
    if intent in {'CREATE_NEW','EDIT_EXISTING'}:
        seed['schema']['properties']['intent']={'enum':[intent,'CLARIFICATION']}
        seed['schema']['properties']['speech_act']={'enum':['ACTION_REQUEST','INFORMATION_REQUEST']}
        seed['instruction']+=' A repair may choose CLARIFICATION only with an ID-free task_proposal preserving the exact goal, operation, artifact and known requirements, with real unresolved requirements and target. It may not become a generic conversational clarification.'
    return seed


def enforce(value,seed):
    from .operation_intent import ResolutionSchemaError
    if not seed:
        return
    anchored_clarification=(seed.get('intent') in {'CREATE_NEW','EDIT_EXISTING'}
        and value.get('intent')=='CLARIFICATION' and isinstance(value.get('task_proposal'),dict)
        and value['task_proposal'].get('operation')==seed['intent'])
    for field in ('intent','speech_act','primary_goal'):
        if anchored_clarification and (field=='intent' or field=='speech_act' and value.get(field) in {'ACTION_REQUEST','INFORMATION_REQUEST'}):
            continue
        if value.get(field)!=seed[field]:
            raise ResolutionSchemaError(field,'task_request_repair_changed_semantics')
    update=value.get('task_proposal') or value.get('task_update') or {}
    if not isinstance(update,dict):
        raise ResolutionSchemaError('task_update','task_request_repair_changed_semantics')
    operation=value.get('intent') if value.get('intent') in {'CREATE_NEW','EDIT_EXISTING'} else update.get('operation')
    if seed.get('operation') and operation!=seed['operation']:
        raise ResolutionSchemaError('task_update.operation','task_request_repair_changed_semantics')
    if seed.get('task_relation') and update.get('relation')!=seed['task_relation']:
        raise ResolutionSchemaError('task_update.relation','task_request_repair_changed_semantics')
    for field in ('execute','execution_intent','verification_requirement','artifact_type'):
        actual=value.get(field,update.get(field) if field=='artifact_type' else None)
        if field in seed and actual is not None and actual!=seed[field]:
            raise ResolutionSchemaError(field,'task_request_repair_changed_semantics')
    effective_execute=(value.get('execute') is True or value.get('execution_intent') in {'RUN','TEST'}
        or value.get('verification_requirement') in {'EXECUTION','TEST_SUITE'})
    if 'execute' in seed and effective_execute!=seed['execute']:
        raise ResolutionSchemaError('execute','task_request_repair_changed_semantics')
    effective_verification=value.get('verification_requirement',
        'TEST_SUITE' if value.get('execution_intent')=='TEST' else 'EXECUTION' if effective_execute else 'BASIC')
    if 'verification_requirement' in seed and effective_verification!=seed['verification_requirement']:
        raise ResolutionSchemaError('verification_requirement','task_request_repair_changed_semantics')
    if seed.get('execution_intent') in {'RUN','TEST'} and not effective_execute:
        raise ResolutionSchemaError('execution_intent','task_request_repair_changed_semantics')
    if 'artifact_type' in seed and value.get('artifact_type',update.get('artifact_type'))!=seed['artifact_type']:
        raise ResolutionSchemaError('artifact_type','task_request_repair_changed_semantics')
    if seed.get('execute') is True and not (value.get('execute') is True or value.get('execution_intent') in {'RUN','TEST'}):
        raise ResolutionSchemaError('execute','task_request_repair_changed_semantics')
