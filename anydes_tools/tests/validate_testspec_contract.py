"""Frozen candidate; one TestSpec/execution UI session with production guard."""
from anydes_tools.tests.validate_classifier_branches import runner

runner.OUT = runner.ROOT/'anydes_tools/logs/testspec_contract'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/verification_types.py',
    'anydes_tools/conversation/code_generation.py',
    'anydes_tools/tests/test_testspec_contract.py',
    'anydes_tools/tests/validate_testspec_contract.py']))

if __name__=='__main__':
    raise SystemExit(runner.main() or 0)
