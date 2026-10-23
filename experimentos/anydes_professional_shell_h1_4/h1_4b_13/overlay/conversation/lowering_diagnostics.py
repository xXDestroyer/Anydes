"""Field-level diagnostics contain types and hashes, never rejected free text."""
import hashlib
import json


def record_rejection(controller, plan, *, path, reason, expected, received):
    payload=json.dumps(plan.to_dict(),sort_keys=True,default=str).encode()
    controller.audit.record('proposal_conversion_rejected',task_id=plan.task_id,
        proposal_id='sha256:'+hashlib.sha256(payload).hexdigest(),
        stage='PLAN_TO_ACTION',field_path=path,reason_code=reason,
        expected_contract=expected,received_type=type(received).__name__,
        received_value_class='ABSENT' if received is None else 'PRESENT_REDACTED',
        source='REPAIR' if getattr(controller,'_proposal_repair_origin',False) else 'ORIGINAL')


def spec_rejection(controller,plan,error):
    raw=plan.inputs.get('code_spec')
    path='inputs.code_spec'
    if isinstance(raw,dict):
        from .verification_types import VerificationRequirement
        if 'verification_requirement' in raw and raw['verification_requirement'] not in {v.value for v in VerificationRequirement}:
            path+='.'+'verification_requirement'
        elif 'operation_descriptor' in str(error):
            path+='.'+'operation_descriptor'
        elif 'source_name' in str(error) or 'filename' in str(error):
            path+='.'+'target_filename'
        elif 'test' in str(error).casefold():
            path+='.'+('test_cases' if 'test_cases' in raw else 'tests')
    record_rejection(controller,plan,path=path,reason='CODE_SPEC_REJECTED',
        expected='CodeGenerationSpec strict constructor',
        received=raw.get(path.rsplit('.',1)[-1]) if isinstance(raw,dict) and path!='inputs.code_spec' else raw)
