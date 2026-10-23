"""Frozen candidate, normal human UI bootstrap and protected-file checks."""
from anydes_tools.tests.validate_conversational_acceptance import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/intent_acceptance'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    *[str(p.relative_to(runner.ROOT)).replace('\\', '/') for p in
      (runner.ROOT/'anydes_tools/conversation').glob('*.py')],
    'anydes_tools/tests/validate_intent_acceptance.py']))
runner.SOURCES.append('anydes_tools/tests/test_intent_acceptance.py')

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
