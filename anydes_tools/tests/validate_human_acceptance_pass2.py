"""Validation only: normal human bootstrap, isolated data and frozen source graph."""
from anydes_tools.tests.validate_human_acceptance_round1 import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/human_acceptance_pass2'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_human_acceptance_pass2.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
