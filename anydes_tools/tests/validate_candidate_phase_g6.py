"""G6 normal human bootstrap, isolated observation and immutable run snapshot."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.validate_candidate_phase_g5 import runner
runner.OUT = ROOT / 'anydes_tools/logs/phase_g6'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/contracts.py',
    'anydes_tools/tests/test_recent_artifact_index_g6.py',
    'anydes_tools/tests/collect_candidate_g6.py',
    'anydes_tools/tests/validate_candidate_phase_g6.py']))
if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
