"""Normal UI bootstrap with isolated acceptance evidence; no cognition fixtures."""
from anydes_tools.tests.validate_human_acceptance_round1 import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/failure_topic_acceptance'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_failure_topic_acceptance.py',
    'anydes_tools/tests/test_failure_topic_preservation.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
