"""One three-turn classification UI run, no cognition doubles or tool requests."""
from anydes_tools.tests.validate_local_invariant import runner

runner.OUT=runner.ROOT/'anydes_tools/logs/pre_request_phase2b/ui'
runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_pre_request_semantics.py',
    'anydes_tools/tests/validate_pre_request.py']))

if __name__=='__main__':
    raise SystemExit(runner.main() or 0)
