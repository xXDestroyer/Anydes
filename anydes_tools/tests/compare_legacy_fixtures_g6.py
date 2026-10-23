"""Offline compatibility comparison using frozen G5 modules, never human runtime."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class FrozenSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'anydes_tools.conversation.' + name for name in
                ('artifact_state', 'planner', 'executable_plan', 'controller')}:
            source = ROOT / 'anydes_tools/logs/phase_g5/snapshot' / (fullname.replace('.', '/') + '.py')
            return importlib.util.spec_from_file_location(fullname, source)
        return None


if __name__ == '__main__':
    if sys.argv[1] == 'frozen':
        sys.meta_path.insert(0, FrozenSources())
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'anydes_tools.tests.test_voice_artifact_workflow_v001_5',
        'anydes_tools.tests.test_action_experience_memory_v001_5'])
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    print(json.dumps({'mode':sys.argv[1], 'tests':result.testsRun,
        'failures':[test.id() for test, error in result.failures],
        'errors':[test.id() for test, error in result.errors]}))
