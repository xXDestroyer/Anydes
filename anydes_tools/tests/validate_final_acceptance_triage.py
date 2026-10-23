"""One frozen UI run for execution, literal source and read-only memory. No fixes."""
from anydes_tools.tests.validate_explanation_continuation import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/final_acceptance_triage'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_final_acceptance_triage.py',
    'anydes_tools/tests/test_final_acceptance_triage.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
