"""Frozen final conversational replay, using the unchanged candidate bootstrap."""
from anydes_tools.tests.validate_ui_lifecycle import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/final_stabilization'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_final_stabilization.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
