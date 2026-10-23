import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from assets import asset_path, shell_document, ShellAssetServer, LAB
from policy import RendererKind, status_label, can_switch, load_selection, save_selection


class PolicyTests(unittest.TestCase):
    def test_two_renderers_only(self):
        self.assertEqual(len(RendererKind), 2)
        with self.assertRaises(ValueError):
            RendererKind('ANOTHER_CORE')

    def test_canonical_states(self):
        for state in ('IDLE', 'THINKING', 'LISTENING', 'SPEAKING', 'WORKING'):
            self.assertTrue(status_label(state))

    def test_work_modifier_only_when_working(self):
        self.assertEqual(status_label('WORKING', 'PROGRAMMING'), 'Programando...')
        self.assertEqual(status_label('IDLE', 'PROGRAMMING'), 'En reposo')

    def test_disconnect_neutral(self):
        self.assertEqual(status_label('SPEAKING', connected=False), 'Presencia en pausa')

    def test_switch_idle_only(self):
        self.assertTrue(can_switch('IDLE', True))
        for state in ('SPEAKING', 'THINKING', 'WORKING', 'LISTENING'):
            self.assertFalse(can_switch(state, True))
        self.assertFalse(can_switch('IDLE', False))
        self.assertFalse(can_switch('IDLE', True, True))

    def test_preference_not_memory(self):
        with tempfile.TemporaryDirectory() as root:
            first = Path(root)/'20261020T010000'; first.mkdir()
            save_selection(first, RendererKind.AVATAR_2_5D, True)
            self.assertEqual(load_selection(root), RendererKind.AVATAR_2_5D)
            second = Path(root)/'20261020T020000'; second.mkdir()
            save_selection(second, RendererKind.HOLOGRAM, False)
            self.assertIsNone(load_selection(root))
            self.assertEqual(sorted(p.name for p in first.iterdir()), ['ui_selection.json'])

    def test_bad_preference_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            session = Path(root)/'x'; session.mkdir()
            (session/'ui_selection.json').write_text('{')
            self.assertIsNone(load_selection(root))


class AssetTests(unittest.TestCase):
    def test_allowlist(self):
        self.assertEqual(asset_path('/app.js'), LAB/'app.js')
        self.assertIsNotNone(asset_path('/node_modules/three/build/three.module.js'))

    def test_traversal_and_private_blocked(self):
        for path in ('/../.env', '/%2e%2e/.env', '/h1_3/sessions/private.json', '/server.mjs', '/node_modules/three/build/../../../../.env', '/node_modules/three/build/../../../h1_3/run_staging.py'):
            self.assertIsNone(asset_path(path))

    def test_embeds_approved_runtime_not_new_geometry(self):
        page = shell_document().decode()
        self.assertIn('src="app.js"', page)
        self.assertIn('importmap', page)
        self.assertIn('display:none!important', page)

    def test_server_read_only_and_local(self):
        server = ShellAssetServer()
        try:
            self.assertEqual(server.server.server_address[0], '127.0.0.1')
            with urlopen(server.url) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("object-src 'none'", response.headers['Content-Security-Policy'])
            base = server.url.split('/shell')[0]
            for path in ('/h1_3/run_staging.py', '/../.env'):
                with self.assertRaises(HTTPError):
                    urlopen(base+path)
            with self.assertRaises(HTTPError):
                urlopen(Request(server.url, data=b'action', method='POST'))
        finally:
            server.close(); server.close()
        self.assertFalse(server.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
