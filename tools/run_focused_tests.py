"""Run selected existing offline tests using only the exported staging overlay."""
from pathlib import Path
import importlib
import json
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / 'experimentos/anydes_professional_shell_h1_4/h1_4b_13'
sys.dont_write_bytecode = True
sys.path[:0] = [str(PHASE), str(ROOT), str(ROOT / 'programa_principal')]


def main():
    # No live model requests, tools, device initialization or UI in this selection.
    from bootstrap import install
    install()
    import anydes_tools.tests
    anydes_tools.tests.__path__.insert(0, str(PHASE / 'tests'))
    names = ['test_classifier_wire', 'test_code_output', 'test_technical_literals']
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('anydes_tools.tests.' + name))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    outside = []
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('anydes', 'bootstrap', 'sandbox_overlay')):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(ROOT):
                outside.append(name)
    report = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), outside_release_modules=outside,
                  live_model_calls=0, ui_opened=False)
    (ROOT / 'audit/focused_tests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return not result.wasSuccessful() or bool(outside)


if __name__ == '__main__':
    sys.exit(main())
