"""One short UI validation; two explicit failure-data fixtures, no tool execution."""
import os
import sys
from anydes_tools.tests.validate_identity_attribute import runner


runner.OUT=runner.ROOT/'anydes_tools/logs/natural_failure_voice/replay'
runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/outcome_voice.py',
    'anydes_tools/conversation/natural_response.py',
    'anydes_tools/conversation/reviewer_transport.py',
    'anydes_tools/conversation/operation_intent.py',
    'anydes_tools/conversation/resolution_contract.py',
    'anydes_tools/conversation/discourse_context.py',
    'anydes_tools/tests/test_outcome_voice.py',
    'anydes_tools/tests/validate_outcome_voice.py']))


def install_failure_data_fixtures():
    if os.environ.get('ANYDES_RUNTIME_KIND')!='CANDIDATE':
        raise RuntimeError('failure_data_fixture_requires_candidate')
    from anydes_tools.conversation.controller import ToolConversationController
    from anydes_tools.conversation.failure_state import FailureRecord
    original=ToolConversationController._handle_resolved_turn
    counts={}
    def handle(controller,text,*,turn_id):
        ordinal=counts.get(id(controller),0)+1
        counts[id(controller)]=ordinal
        if ordinal in {3,4}:
            record=FailureRecord.create(task_id='acceptance_fixture_'+str(ordinal),stage='EXECUTION',
                error_code='TOOL_EXECUTION_FAILED' if ordinal==3 else 'UNKNOWN',
                technical_message='ValueError:python_import_not_allowed' if ordinal==3 else '',
                safe_user_message='',retryable=False)
            controller.state.pending_task.task_id=record.task_id
            controller.state.last_tool_failure=record
            controller.state.previous_assistant_act='ACTION_FAILED'
            controller.audit.record('acceptance_failure_data_fixture',ordinal=ordinal,
                cause_detail_available=ordinal==3,failure_id=record.failure_id,
                tools_executed=False,classifier_mocked=False,model_response_injected=False)
        return original(controller,text,turn_id=turn_id)
    ToolConversationController._handle_resolved_turn=handle


if __name__=='__main__':
    if sys.argv[1]=='gui':
        from anydes_tools.tests import observe_candidate_g2
        original_install=observe_candidate_g2.install
        def install(output):
            original_install(output)
            install_failure_data_fixtures()
        observe_candidate_g2.install=install
    raise SystemExit(runner.main() or 0)
