"""One short normal-bootstrap UI validation; no execute/source/memory replay."""
from anydes_tools.tests.validate_final_stabilization import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/two_major_fix'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/validate_two_major_fix.py','anydes_tools/tests/test_two_major_fix.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
