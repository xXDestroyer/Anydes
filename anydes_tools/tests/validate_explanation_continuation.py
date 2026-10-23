"""One frozen normal-bootstrap UI session, explanations only."""
from anydes_tools.tests.validate_two_major_fix import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/explanation_continuation_final'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_explanation_continuation.py',
    'anydes_tools/tests/test_explanation_continuation.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
