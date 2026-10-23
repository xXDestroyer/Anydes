"""Classifier shape evidence: names, typed enums and hashes, never prose."""
import hashlib
import re

COMMON=frozenset({'intent','speech_act','primary_goal','confidence'})
EXTRA_INTENTS=frozenset({'CONVERSATION','CONTENT_RESPONSE','PERSONAL_MEMORY_RECALL',
    'PERSONAL_IDENTITY','SELF_PREFERENCE','SELF_RELATIONAL','ENTITY_QUERY','PRE_REQUEST',
    'ARTIFACT_METADATA_QUERY','CONTINUE_TASK','TOPIC_CLOSURE','OFFER_ACCEPTANCE',
    'CLARIFICATION','UNSUPPORTED','CAPABILITY_QUERY'})


def safe_name(value):
    return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',value) else (
        'FIELD_SHA256_'+hashlib.sha256(str(value).encode()).hexdigest()[:16])


def known_intents():
    from .operation_intent import ArtifactOperation
    from .interaction_router import TaskControl
    return EXTRA_INTENTS | {v.value for v in ArtifactOperation} | {v.value for v in TaskControl}


def shape(value,state=None,error=None):
    from .resolution_contract import branch_contract, canonicalize_branch_fields
    from .interaction_router import TurnRelation
    contract=branch_contract(value,state)
    raw=value if isinstance(value,dict) else {}
    projected=canonicalize_branch_fields(raw,state=state)
    from .classifier_output import pending_task
    from .continuation_public import INTERNAL_FIELDS
    internal=set(raw)&INTERNAL_FIELDS.keys()
    if (raw.get('intent')=='CONTINUE_TASK' and contract['declared_branch']=='TASK_CONTINUATION'
            and pending_task(state)):
        projected={key:item for key,item in projected.items() if key not in internal}
    enums={'intent':known_intents(),'speech_act':{'ACTION_REQUEST','INFORMATION_REQUEST','SOCIAL','ACKNOWLEDGEMENT'},
        'turn_relation':{v.value for v in TurnRelation},'conversation_purpose':{'GENERAL','PRE_REQUEST'},
        'clarification_scope':{'TASK','CONVERSATION'}}
    result={key:(raw[key] if isinstance(raw.get(key),str) and raw[key] in choices else
        'INVALID' if key in raw else 'ABSENT') for key,choices in enums.items()}
    fields=set(projected)
    allowed=set(contract['allowed_fields'])
    required=COMMON | set(contract['required_fields'])
    unexpected=fields-allowed
    unexpected.update(getattr(error,'unexpected_fields',[]))
    conflict=getattr(error,'field','')
    constraint=getattr(error,'constraint','')
    conflicts=({'intent','turn_relation'} if constraint=='intent_relation_conflict' else
        unexpected if conflict=='response' and constraint=='unexpected_fields' else {conflict} if conflict else set())
    context=(state or {}).get('resolution_context') or {}
    active=context.get('active_task') or {}
    failure=context.get('active_failure_topic') or {}
    result.update(classifier_branch=contract['declared_branch'] or 'UNRESOLVED',
        received_field_names=sorted(safe_name(k) for k in raw),
        unexpected_field_names=sorted(safe_name(k) for k in unexpected),
        unexpected_field_paths=['$.'+safe_name(k) for k in sorted(unexpected)],
        missing_required_field_names=sorted(required-fields),
        conflicting_field_names=sorted({safe_name(k.split('.')[0]) for k in conflicts}),
        conflicting_field_paths=['$.'+'.'.join(safe_name(part) for part in k.split('.')) for k in sorted(conflicts)],
        expected_allowed_fields=sorted(allowed),
        canonicalized_field_names=sorted(safe_name(k) for k in set(raw)-fields),
        received_types={safe_name(k):type(v).__name__ for k,v in raw.items()},
        reason_code='INTERACTION_SCHEMA_'+constraint.upper() if constraint else 'CLASSIFIER_SHAPE',
        task_id=active.get('task_id') or None,failure_id=failure.get('failure_id') or None)
    from .classifier_output import pending_task, valid_id
    pending=pending_task(state)
    update_id=(raw.get('task_update') or {}).get('task_id') if isinstance(raw.get('task_update'),dict) else None
    result.update(pending_task_exists=bool(pending),
        expected_task_id_hash=hashlib.sha256(pending['task_id'].encode()).hexdigest() if pending else None,
        received_task_id_format='ABSENT' if update_id is None else 'VALID' if valid_id(update_id) else 'INVALID',
        field_path='$.'+getattr(error,'field','') if error else None)
    result['task_id_match_status']=('MATCH' if pending and update_id==pending.get('task_id') else
        'MISMATCH' if pending and update_id is not None else 'ABSENT')
    result['internal_field_names']=sorted(internal)
    result['internal_reason_codes']=[INTERNAL_FIELDS[key] for key in sorted(internal)]
    if constraint in {'TASK_PROPOSAL_ID_NOT_ALLOWED','CONTINUATION_PUBLIC_CONTRACT_VIOLATION',
            'OPERATIONAL_CLARIFICATION_WITHOUT_TASK_ANCHOR','CLARIFICATION_SCOPE_REQUIRED'}:
        result['reason_code']=constraint
    update=raw.get('task_update')
    if isinstance(update,dict):
        result['task_update_field_names']=sorted(safe_name(k) for k in update)
        result['task_update_discriminator']={key:update.get(key,'ABSENT') if update.get(key,'ABSENT') in choices else 'INVALID'
            for key,choices in {'relation':{'START','CONTINUE','REVISE','SWITCH','ABSENT'},
                'operation':{'CREATE_NEW','EDIT_EXISTING','ABSENT'},'artifact_type':{'CODE','DOCUMENT','TEXT','ABSENT'}}.items()
            if isinstance(update.get(key,'ABSENT'),str)}
    return result
