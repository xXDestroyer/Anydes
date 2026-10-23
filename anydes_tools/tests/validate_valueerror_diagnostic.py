"""One diagnostic conversation only; normal candidate, no failure-data fixtures."""
from anydes_tools.tests.validate_identity_attribute import runner

runner.OUT = runner.ROOT / 'anydes_tools/logs/valueerror_phase1/replay'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/diagnostics.py',
    'anydes_tools/conversation/natural_response.py',
    'anydes_tools/conversation/reviewer_transport.py',
    'anydes_tools/conversation/outcome_voice.py',
    'anydes_tools/tests/test_conversation_diagnostics.py',
    'anydes_tools/tests/validate_valueerror_diagnostic.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
