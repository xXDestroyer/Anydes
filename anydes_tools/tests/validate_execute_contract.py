"""Frozen candidate; one UI execution session, no unrelated acceptance replay."""
from anydes_tools.tests.validate_planner_resilience import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/execute_contract'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/operation_intent.py',
    'anydes_tools/conversation/resolution_contract.py',
    'anydes_tools/conversation/semantic_proposal.py',
    'anydes_tools/conversation/general_reasoning.py',
    'anydes_tools/conversation/semantic_runtime.py',
    'anydes_tools/conversation/plan_lowering.py',
    'anydes_tools/conversation/controller.py',
    'anydes_tools/tests/test_execute_contract.py',
    'anydes_tools/tests/validate_execute_contract.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
