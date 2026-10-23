"""Frozen human runtime; normal bootstrap, opt-in observation, no cognition doubles."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests.validate_candidate_phase_g6 import runner

runner.OUT = ROOT / 'anydes_tools/logs/human_acceptance_round1'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    *[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'anydes_tools/conversation').glob('*.py')],
    'anydes_tools/registry/semantic_operations.json',
    'anydes_tools/tests/test_human_acceptance_round1.py',
    'anydes_tools/tests/validate_human_acceptance_round1.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
