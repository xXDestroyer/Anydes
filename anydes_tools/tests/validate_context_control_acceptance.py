"""Frozen normal human bootstrap; isolated data, no semantic/executor doubles."""
from anydes_tools.tests.validate_human_acceptance_round1 import runner

runner.OUT = runner.ROOT/'anydes_tools/logs/context_control_acceptance'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_failure_topic_preservation.py',
    'anydes_tools/tests/test_context_control_resolution.py',
    'anydes_tools/tests/validate_context_control_acceptance.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
