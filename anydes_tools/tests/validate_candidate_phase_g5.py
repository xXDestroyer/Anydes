"""G5 normal human bootstrap, frozen code and isolated opt-in observation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.validate_candidate_phase_g4 import runner
runner.OUT = ROOT / 'anydes_tools/logs/phase_g5'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/pipeline_trace.py',
    'anydes_tools/conversation/publication.py',
    'anydes_tools/conversation/runtime_patch.py',
    'anydes_tools/conversation/oracle_registry.py',
    'anydes_tools/tests/test_pipeline_terminality_g5.py',
    'anydes_tools/tests/validate_candidate_phase_g5.py']))
if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
