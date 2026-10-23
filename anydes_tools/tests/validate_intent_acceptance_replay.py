"""Definitive post-regression freeze, single normal UI run, no runtime patches."""
from anydes_tools.tests.validate_intent_acceptance import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/intent_acceptance_replay'
runner.SOURCES.append('anydes_tools/tests/validate_intent_acceptance_replay.py')

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
