"""Freeze/check and normal human UI bootstrap; no substituted cognition."""
from anydes_tools.tests.validate_intent_acceptance import runner

runner.OUT = runner.ROOT/'anydes_tools/logs/operation_grounding'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_operation_grounding.py',
    'anydes_tools/tests/test_operation_grounding.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
