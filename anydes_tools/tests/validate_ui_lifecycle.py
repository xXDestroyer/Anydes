"""One normal-bootstrap UI lifecycle smoke; no conversation replay."""
from anydes_tools.tests.validate_operation_grounding import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/ui_lifecycle_fix'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/ui_lifecycle.py', 'anydes_tools/observer_filesystem.py',
    'anydes_tools/tests/test_ui_lifecycle.py', 'anydes_tools/tests/ui_lifecycle_fixture.py',
    'anydes_tools/tests/validate_ui_lifecycle.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
