"""One normal-bootstrap UI execution. All other accepted areas remain frozen."""
from anydes_tools.tests.validate_final_acceptance_triage import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/planner_resilience'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/planner_transport.py',
    'anydes_tools/tests/test_planner_resilience.py',
    'anydes_tools/tests/validate_planner_resilience.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
