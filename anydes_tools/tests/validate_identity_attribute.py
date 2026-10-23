"""Freeze and one isolated UI validation for the identity attribute contract."""
import sys
from anydes_tools.tests.validate_conversational_core import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/identity_attribute_final/replay'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_identity_attribute.py',
    'anydes_tools/tests/test_identity_attribute.py']))

if __name__ == '__main__':
    if sys.argv[1] == 'freeze-before':
        runner.OUT = runner.OUT.parent / 'before'
        sys.argv[1] = 'freeze'
    raise SystemExit(runner.main() or 0)
