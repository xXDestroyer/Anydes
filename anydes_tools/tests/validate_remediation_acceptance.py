"""Single normal human runtime replay with isolated evidence and frozen sources."""
from anydes_tools.tests.validate_context_control_acceptance import runner

runner.OUT=runner.ROOT/'anydes_tools/logs/remediation_acceptance'
runner.SOURCES=list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/tests/test_failure_remediation_intent.py',
    'anydes_tools/tests/validate_remediation_acceptance.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
