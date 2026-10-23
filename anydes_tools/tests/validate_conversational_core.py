"""Isolated candidate for conversational-domain acceptance only."""
from anydes_tools.tests.validate_pending_confirmation import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/conversational_core/replay'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/conversational_bridge.py',
    'anydes_tools/conversation/discourse_context.py',
    'anydes_tools/conversation/controller.py',
    'anydes_tools/conversation/interaction_router.py',
    'anydes_tools/conversation/resolution_contract.py',
    'anydes_tools/conversation/operation_intent.py',
    'anydes_tools/conversation/semantic_runtime.py',
    'anydes_tools/tests/validate_conversational_core.py',
    'anydes_tools/tests/test_conversational_core.py',
    'anydes_tools/tests/test_operation_grounding.py',
    'anydes_tools/tests/test_failure_topic_preservation.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
