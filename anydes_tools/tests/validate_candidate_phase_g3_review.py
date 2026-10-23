"""Separate frozen validation after the first G3 battery; no evidence overwritten."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.validate_candidate_phase_g3 import runner
runner.OUT = ROOT / 'anydes_tools/logs/phase_g3_review'
runner.SOURCES.append('anydes_tools/tests/validate_candidate_phase_g3_review.py')
if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
