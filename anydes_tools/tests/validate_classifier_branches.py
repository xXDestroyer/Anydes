"""One frozen classifier/execution UI session, isolated from production."""
from anydes_tools.tests.validate_execute_contract import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/classifier_branches'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/interaction_router.py',
    'anydes_tools/tests/test_classifier_branches.py',
    'anydes_tools/tests/validate_classifier_branches.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
