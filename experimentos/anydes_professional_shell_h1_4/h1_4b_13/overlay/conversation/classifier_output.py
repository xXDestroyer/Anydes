"""Canonical classifier ingress. Semantic proposals carry no new task ID authority."""
from dataclasses import dataclass, replace
import re


def pending_task(state):
    active=((state or {}).get('resolution_context') or {}).get('active_task') or {}
    task=active.get('semantic_contract') or {}
    if (not task or task.get('suspended') or active.get('status') in
            {'IDLE','COMPLETED','COMPLETED_PARTIAL','CANCELLED','SUPERSEDED','EXPIRED'}):
        return {}
    return task


def valid_id(value):
    return isinstance(value,str) and re.fullmatch(r'task:[A-Za-z0-9_-]{1,123}',value) is not None


@dataclass(frozen=True)
class SemanticBranch:
    """Host-only helpers are never model-facing output containers."""
    public_fields: dict
    branch_contract: dict
    task_request_contract: dict | None = None


def build_public(proposal, state=None, record=None):
    from .operation_intent import ResolutionSchemaError
    from .resolution_contract import branch_contract, canonicalize_branch_fields
    value=proposal.public_fields if isinstance(proposal,SemanticBranch) else proposal
    if not isinstance(value,dict):
        raise ResolutionSchemaError('response','missing_required_fields')
    from .continuation_public import project
    value=project(value,state,record)
    from .clarification_anchor import validate
    validate(value,state)
    wrappers=set(value)&{'branch_contract','task_request_contract'}
    if wrappers:
        error=ResolutionSchemaError('response','INTERNAL_WRAPPER_LEAKED_TO_PUBLIC_SCHEMA')
        error.unexpected_fields=sorted(wrappers)
        raise error
    contract=branch_contract(value,state)
    if not contract['declared_branch']:
        return dict(value)  # Existing strict discriminator validation supplies the exact error.
    unknown=set(value)-set(contract['allowed_fields'])-set(contract['inapplicable_fields'])
    if unknown:
        raise ResolutionSchemaError('response','unexpected_fields')
    result=canonicalize_branch_fields(value,record=record,state=state)
    update=result.get('task_update')
    task=pending_task(state)
    if 'task_update' in result:
        if not task:
            raise ResolutionSchemaError('task_update','TASK_UPDATE_WITHOUT_PENDING_TASK')
        if not isinstance(update,dict):
            raise ResolutionSchemaError('task_update','task_update_schema')
        if not valid_id(update.get('task_id')):
            raise ResolutionSchemaError('task_update.task_id','TASK_UPDATE_TASK_ID_INVALID_FORMAT')
        if update['task_id']!=task['task_id']:
            raise ResolutionSchemaError('task_update.task_id','TASK_UPDATE_TASK_ID_MISMATCH')
        if update.get('relation') not in {'CONTINUE','REVISE'}:
            raise ResolutionSchemaError('task_update.relation','continuation_relation_required')
    if (result.get('intent') in {'CREATE_NEW','EDIT_EXISTING'} and update is None
            and 'task_proposal' not in result and result.get('artifact_type') in {'CODE','DOCUMENT','TEXT'}):
        result={**result,'task_proposal':{'relation':'START','operation':result['intent'],
            'artifact_type':result['artifact_type'],'requirements':[]}}
    new=result.get('task_proposal')
    if 'task_proposal' in result:
        if update is not None:
            raise ResolutionSchemaError('task_proposal','mutually_exclusive_task_semantics')
        if not isinstance(new,dict):
            raise ResolutionSchemaError('task_proposal','task_proposal_schema')
        if 'task_id' in new:
            raise ResolutionSchemaError('task_proposal.task_id','RUNTIME_OWNS_NEW_TASK_ID')
        if new.get('relation') not in {'START','SWITCH'}:
            raise ResolutionSchemaError('task_proposal.relation','new_task_relation_required')
        if result['intent'] not in {'CREATE_NEW','EDIT_EXISTING','CLARIFICATION'}:
            raise ResolutionSchemaError('task_proposal','operational_task_domain_required')
        from .task_continuity import parse_update
        try:
            parse_update(new)
        except ValueError as error:
            raise ResolutionSchemaError('task_proposal',str(error)) from error
        if new.get('operation') not in {'CREATE_NEW','EDIT_EXISTING'}:
            raise ResolutionSchemaError('task_proposal.operation','required_operation')
        if new.get('artifact_type') not in {'CODE','DOCUMENT','TEXT'}:
            raise ResolutionSchemaError('task_proposal.artifact_type','required_artifact_type')
        if result['intent']!='CLARIFICATION' and new['operation']!=result['intent']:
            raise ResolutionSchemaError('task_proposal.operation','operation_conflict')
        if result.get('artifact_type',new['artifact_type'])!=new['artifact_type']:
            raise ResolutionSchemaError('task_proposal.artifact_type','artifact_type_conflict')
        if result['intent']!='CLARIFICATION':
            result={**result,'artifact_type':new['artifact_type']}
    # Projection copies only the selected public variant; meaningful unknowns were rejected.
    validated=SemanticBranch(result,contract)
    return {key:validated.public_fields[key] for key in validated.branch_contract['allowed_fields']
        if key in validated.public_fields}


def accept_task_semantics(controller, decision, turn_id):
    """Ingress adapter only; reuse the accepted runtime ID allocator and merge unchanged."""
    from .task_continuity import accept_update
    if decision.task_proposal is not None:
        accept_update(controller,replace(decision,task_update=decision.task_proposal),turn_id)
        controller.audit.record('new_task_id_assigned',
            task_id=controller.state.pending_task.task_id,authority='RUNTIME',
            model_task_id=False,public_task_update=False)
    else:
        accept_update(controller,decision,turn_id)
