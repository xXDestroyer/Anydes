"""One frozen candidate UI session for pending confirmation/execution continuity."""
from anydes_tools.tests.validate_testspec_contract import runner

runner.OUT=runner.ROOT/'anydes_tools/logs/pending_confirmation_final'
runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_pending_confirmation_contract.py',
    'anydes_tools/tests/validate_pending_confirmation.py']))

if __name__=='__main__':
    raise SystemExit(runner.main() or 0)
