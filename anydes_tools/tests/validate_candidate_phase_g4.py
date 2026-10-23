"""G4 frozen human bootstrap; same observer, independent evidence root."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from anydes_tools.tests.validate_candidate_phase_g3 import runner
runner.OUT = ROOT / 'anydes_tools/logs/phase_g4'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/conversation/verification_types.py',
    'anydes_tools/conversation/code_generation.py',
    'anydes_tools/conversation/content_generation.py',
    'anydes_tools/tests/test_pipeline_consistency_g4.py',
    'anydes_tools/tests/validate_candidate_phase_g4.py']))
if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
