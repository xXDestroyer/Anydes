import tempfile
import unittest
from pathlib import Path
from anydes_tools.runtime_environment import RuntimeEnvironment

class EnvironmentTests(unittest.TestCase):
    def test_explicit_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for kind in ('STAGING', 'CANDIDATE', 'TEST'):
                env = RuntimeEnvironment(kind, root, root / kind)
                self.assertEqual(env.path('telemetry/ui'), root / kind / 'telemetry/ui')
                with self.assertRaises(PermissionError):
                    env.assert_write(root / 'programa_principal/datos/log.json')
                with self.assertRaises(ValueError):
                    env.path('../escape')

    def test_production_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            env = RuntimeEnvironment('PRODUCTION', root, root / 'programa_principal/datos')
            env.assert_write(env.path('telemetry/ui'))

    def test_reject_bad_roots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            with self.assertRaises(ValueError):
                RuntimeEnvironment('CANDIDATE', root, root / 'programa_principal/datos')
            with self.assertRaises(ValueError):
                RuntimeEnvironment('TEST', root, Path('relative'))

    def test_nested_destinations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            env = RuntimeEnvironment('TEST', root, root / 'test')
            for component in ('ui', 'error', 'crash', 'voice', 'parakeet', 'tools'):
                self.assertTrue(env.path('telemetry/' + component).is_relative_to(env.data_root))


    def test_real_audit_hook_denies_write(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'programa_principal').mkdir()
            code = """import sys
from pathlib import Path
from anydes_tools.runtime_environment import RuntimeEnvironment, activate
r=Path(sys.argv[1]); activate(RuntimeEnvironment('TEST',r,r/'test'))
try:
 (r/'programa_principal/forbidden.txt').write_text('denied')
except PermissionError:
 assert not (r/'programa_principal/forbidden.txt').exists()
 assert (r/'test/environment_violations.jsonl').is_file()
else:
 raise AssertionError('write escaped guard')
(r/'test/allowed.txt').write_text('ok')
"""
            result = subprocess.run([sys.executable,'-B','-c',code,str(root)],
                capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'test/allowed.txt').read_text(),'ok')

if __name__ == '__main__':
    unittest.main()

