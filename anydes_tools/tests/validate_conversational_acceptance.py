"""One frozen normal-bootstrap UI session. No cognition or verifier fixtures."""
from anydes_tools.tests.validate_human_acceptance_round1 import runner

runner.OUT=runner.ROOT/'anydes_tools/logs/conversational_acceptance'
runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_conversational_discourse.py',
    'anydes_tools/tests/validate_conversational_acceptance.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
