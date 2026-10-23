"""Controlled observer races and subprocess Qt shutdown; not human replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from anydes_tools.observer_filesystem import ObserverBoundary, read_metadata_bytes, scan_sizes

ROOT = Path(__file__).resolve().parents[2]


class ObserverFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.record = lambda event, **details: self.events.append((event, details))

    def test_rename_between_enumeration_and_stat(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'anydes_tools/sandbox') as folder:
            root = Path(folder)
            source = root / '.sample.json.tmp'
            source.write_text('{}')
            original = os.scandir

            class RenameOnClose:
                def __enter__(self):
                    self.scan = original(root)
                    # Windows DirEntry may cache enumeration metadata. Force an
                    # uncached metadata lookup to exercise the exception path.
                    class UncachedEntry:
                        def __init__(self, entry):
                            self.path = entry.path

                        def stat(self, *, follow_symlinks):
                            return Path(self.path).stat(follow_symlinks=follow_symlinks)
                    return [UncachedEntry(entry) for entry in self.scan]

                def __exit__(self, *args):
                    self.scan.close()
                    source.replace(root / 'sample.json')

            with patch('anydes_tools.observer_filesystem.os.scandir', return_value=RenameOnClose()):
                result = scan_sizes(root, self.record)
            self.assertEqual(result['skipped_items'], 1)
            self.assertTrue(any(event == 'observer_transient_race' for event, _ in self.events))
            self.assertEqual(scan_sizes(root, self.record)['data_files'], 1)

    def test_read_disappearing_is_benign(self):
        with patch.object(Path, 'read_bytes', side_effect=FileNotFoundError()):
            self.assertIsNone(read_metadata_bytes('fixture.json', self.record))
        self.assertEqual(self.events[-1][0], 'observer_transient_race')

    def test_replaced_directory_is_benign(self):
        with patch('anydes_tools.observer_filesystem.os.scandir', side_effect=NotADirectoryError()):
            self.assertEqual(scan_sizes('fixture', self.record)['skipped_items'], 1)

    def test_permission_is_real_error_and_disables_only_observer(self):
        boundary = ObserverBoundary(self.record)
        with patch('anydes_tools.observer_filesystem.os.scandir', side_effect=PermissionError()):
            boundary.call('PERIODIC', lambda: scan_sizes('fixture', self.record))
        self.assertTrue(boundary.disabled)
        self.assertEqual(self.events[-1][1]['error_type'], 'PermissionError')
        self.assertFalse(any(e == 'observer_transient_race' for e, _ in self.events))
        self.assertIsNone(boundary.call('PERIODIC', lambda: self.fail('disabled callback ran')))

    def test_programming_error_is_not_classified_as_transient(self):
        boundary = ObserverBoundary(self.record)
        boundary.call('START', lambda: 1 / 0)
        self.assertEqual(self.events[-1][0], 'observer_unexpected_error')
        self.assertEqual(self.events[-1][1]['error_type'], 'ZeroDivisionError')

    def test_exit_is_not_swallowed(self):
        with self.assertRaises(SystemExit):
            ObserverBoundary(self.record).call('START', lambda: sys.exit(2))

    def test_concurrent_atomic_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'anydes_tools/sandbox') as folder:
            root = Path(folder)
            errors = []

            def writer():
                try:
                    for i in range(300):
                        source = root / f'.sample{i}.json.tmp'
                        source.write_text('{}')
                        target = root / 'sample.json'
                        source.replace(target)
                        target.unlink()
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=writer)
            thread.start()
            scans = 0
            while thread.is_alive() or scans < 30:
                scan_sizes(root, self.record)
                scans += 1
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertGreaterEqual(scans, 30)


class QtSubprocessTests(unittest.TestCase):
    def run_case(self, case):
        result = subprocess.run([sys.executable, '-B', '-m',
            'anydes_tools.tests.ui_lifecycle_fixture', case], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Destroyed while thread is still running', result.stderr)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(data['running'], 0)
        self.assertTrue(data['stopped_before_owner_close'])
        self.assertEqual(data['started'], data['finished'])
        if case == 'callback':
            # Qt/Python may retain a handled callback traceback until process
            # exit. The native worker must already be finished, not running.
            self.assertLessEqual(data['destroyed'], data['started'])
        else:
            self.assertEqual(data['started'], data['destroyed'])
        return data

    def test_exception_after_voice_worker_start(self):
        self.assertTrue(self.run_case('exception')['outer_exception_received_after_stop'])

    def test_normal_close(self):
        self.assertTrue(self.run_case('normal')['normal_return'])

    def test_qt_callback_exception_uses_coordinated_shutdown(self):
        self.assertTrue(self.run_case('callback')['normal_return'])

    def test_bounded_timeout_does_not_destroy_running_worker(self):
        self.assertGreater(self.run_case('timeout')['pending'], 0)


if __name__ == '__main__':
    unittest.main()
