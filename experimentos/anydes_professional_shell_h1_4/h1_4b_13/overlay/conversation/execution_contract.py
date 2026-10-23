"""User action authority and generated-code verification are separate contracts."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from enum import Enum


class ExecutionPurpose(str, Enum):
    VERIFICATION = 'VERIFICATION'
    ACTION = 'ACTION'


def code_execution_contract(spec, user_requested):
    """Host-owned intent; execution_required retains its verifier meaning."""
    return {'user_execution_requested':bool(user_requested),
        'verification_execution_required':spec.get('execution_required',False),
        'execution_purpose':ExecutionPurpose.ACTION.value if user_requested else ExecutionPurpose.VERIFICATION.value}

RUNNERS=frozenset({'developer.run_python','developer.run_cpp','developer.test'})


def diagnostic(record, *, task_id, payload, received, origin, reason, operations, value_origin):
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    record('execution_contract_validation',task_id=task_id,plan_id=None,
        proposal_id='sha256:'+digest,stage='PRE_COMPILE',
        field_path='inputs.code_spec.execution_required',received_value=received,
        received_type=type(received).__name__,value_origin=value_origin,
        expected_capability_family=sorted(RUNNERS),proposed_operations=operations,
        reason_code=reason,proposal_origin=origin)


def prepare_semantic_execution(proposal, constraint, record, *, origin):
    """Bind internal verification only to this proposal's generated source.

    A purpose label is not permission: runner scopes, B2 and confirmation remain
    enforced by the existing action compiler/permission bridge.
    """
    from .plan_lowering import PlanLoweringError
    payload=deepcopy(proposal.payload)
    required=constraint['requires_execution']
    kinds=[op['operation_kind'] for op in payload['operations']]
    verification_outputs=set()
    for op in payload['operations']:
        if op['operation_kind']!='GENERATE_CODE':
            continue
        spec=op['specification']
        value=spec.get('execution_required',required)
        value_origin='PLANNER_EXPLICIT' if 'execution_required' in spec else 'HOST_SEMANTIC_CONTRACT'
        reason='CONSISTENT'
        if type(value) is not bool:
            reason='EXECUTION_REQUIRED_TYPE_INVALID'
        elif required and not value:
            reason='REQUESTED_EXECUTION_MISSING'
        elif value and not any(run['operation_kind']=='TEST_CODE' and
                any(out['name'] in run['inputs'] for out in op['outputs']) for run in payload['operations']):
            reason='EXECUTION_REQUIRED_WITHOUT_EXECUTION_CAPABILITY'
        diagnostic(record,task_id=proposal.task_id,payload=payload,received=value,origin=origin,
            reason=reason,operations=kinds,value_origin=value_origin)
        if reason!='CONSISTENT':
            raise PlanLoweringError(reason+':inputs.code_spec.execution_required')
        spec.setdefault('execution_required',required)
        spec.setdefault('verification_requirement',
            ('TEST_SUITE' if spec.get('tests') or spec.get('test_cases') else 'EXECUTION')
            if value else constraint['verification_requirement'])
        if value:
            verification_outputs.update(out['name'] for out in op['outputs'] if out['artifact_type']=='CODE')
    for run in payload['operations']:
        if run['operation_kind']!='TEST_CODE':
            continue
        internal=bool(run['inputs']) and set(run['inputs'])<=verification_outputs and not run['outputs']
        purpose=ExecutionPurpose.ACTION if required else ExecutionPurpose.VERIFICATION
        if not required and not internal:
            raise PlanLoweringError('EXECUTION_NOT_REQUESTED:TEST_CODE')
        supplied=run['specification'].get('execution_purpose')
        if supplied is not None and supplied!=purpose.value:
            raise PlanLoweringError('EXECUTION_PURPOSE_CONFLICT:TEST_CODE')
        run['specification']['execution_purpose']=purpose.value
        record('execution_purpose_bound',task_id=proposal.task_id,proposal_origin=origin,
            execution_purpose=purpose.value,user_execution_requested=required,
            verification_execution_required=internal,exact_generated_input=internal)
    return replace(proposal,payload=payload)


def plan_execution_errors(plan, decision, record, *, origin):
    if plan.task_type!='GENERATE_CODE':
        return ()
    raw=plan.inputs.get('code_spec',{})
    value=raw.get('execution_required',True)
    # GeneralTaskPlan compatibility: never silently rewrite an explicit proposal.
    kinds=[s.purpose for s in plan.steps]
    has_runner=any(s.capability in RUNNERS for s in plan.steps)
    reason='CONSISTENT'
    if type(value) is not bool:
        reason='EXECUTION_REQUIRED_TYPE_INVALID'
    elif value and not has_runner:
        reason='EXECUTION_REQUIRED_WITHOUT_EXECUTION_CAPABILITY'
    elif decision.requires_execution and not value:
        reason='REQUESTED_EXECUTION_MISSING'
    elif not decision.requires_execution and has_runner:
        generated={name for step in plan.steps if step.capability=='developer.write_code'
            for name in step.produces_artifacts}
        if not value or any(not step.consumes_artifacts or not set(step.consumes_artifacts)<=generated
                or step.produces_artifacts for step in plan.steps if step.capability in RUNNERS):
            reason='EXECUTION_NOT_REQUESTED'
    if reason!='CONSISTENT':
        diagnostic(record,task_id=plan.task_id,payload=plan.to_dict(),received=value,origin=origin,
            reason=reason,operations=kinds,
            value_origin='PLANNER_EXPLICIT' if 'execution_required' in raw else 'LEGACY_DEFAULT_TRUE')
        return (reason+':inputs.code_spec.execution_required',)
    return ()
