"""G3 uses the unchanged human bootstrap, G2 observer and separate evidence root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anydes_tools.tests import validate_candidate_phase_g2 as runner

runner.OUT = ROOT / 'anydes_tools/logs/phase_g3'
runner.SOURCES = list(dict.fromkeys([*runner.SOURCES,
    'anydes_tools/contracts.py',
    'anydes_tools/conversation/artifact_state.py',
    'anydes_tools/conversation/controller.py',
    'anydes_tools/conversation/plan_completeness.py',
    'anydes_tools/conversation/task_adherence.py',
    'anydes_tools/conversation/semantic_runtime.py',
    'anydes_tools/tests/test_candidate_blockers_g3.py',
    'anydes_tools/tests/collect_candidate_g3.py',
    'anydes_tools/tests/validate_candidate_phase_g3.py']))

if __name__ == '__main__':
    raise SystemExit(runner.main() or 0)
