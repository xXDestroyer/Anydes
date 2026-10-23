from pathlib import Path
from types import SimpleNamespace
import errno
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from anydes_tools import windows_sandbox as sandbox


class UsageRace(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def limits(self,**changes):
        return sandbox.Limits(**changes)
    def scan(self,**limits):
        return sandbox._usage((self.root,),self.limits(**limits))
    def test_existing_files_still_counted(self):
        (self.root/'one').write_bytes(b'1234')
        self.assertEqual(self.scan(output_bytes=4),'')
        self.assertEqual(self.scan(output_bytes=3),'FILESYSTEM_LIMIT')
    def test_vanished_entry_ignored_remaining_entry_counted(self):
        vanished=self.root/'vanished.o'
        vanished.write_bytes(b'123456789')
        (self.root/'remaining').write_bytes(b'1234')
        walk=os.walk
        def disappear(*args,**kwargs):
            for root,dirs,files in walk(*args,**kwargs):
                vanished.unlink()
                yield root,dirs,files
        with patch.object(os,'walk',disappear):
            self.assertEqual(self.scan(output_files=1,output_bytes=4),'')
        self.assertEqual(self.scan(output_bytes=3),'FILESYSTEM_LIMIT')
    def test_only_missing_entry_is_skipped(self):
        (self.root/'remaining').write_bytes(b'1234')
        with patch.object(os,'walk',return_value=iter([(str(self.root),[],['gone','remaining'])])):
            self.assertEqual(self.scan(output_bytes=3),'FILESYSTEM_LIMIT')
    def test_permission_error_propagates(self):
        with patch.object(os,'walk',return_value=iter([(str(self.root),[],['entry'])])), \
                patch.object(Path,'lstat',side_effect=PermissionError(errno.EACCES,'denied')):
            with self.assertRaises(PermissionError): self.scan()
    def test_unexpected_oserror_propagates(self):
        with patch.object(os,'walk',return_value=iter([(str(self.root),[],['entry'])])), \
                patch.object(Path,'lstat',side_effect=OSError(errno.EIO,'io error')):
            with self.assertRaises(OSError): self.scan()
    def test_reparse_still_rejected(self):
        with patch.object(os,'walk',return_value=iter([(str(self.root),[],['entry'])])), \
                patch.object(Path,'lstat',return_value=SimpleNamespace(st_file_attributes=0x400)):
            self.assertEqual(self.scan(),'REPARSE_OUTPUT')
    def test_hardlink_still_rejected(self):
        first=self.root/'first'
        first.write_bytes(b'123')
        os.link(first,self.root/'second')
        self.assertEqual(self.scan(),'REPARSE_OUTPUT')
    def test_file_quota_still_enforced(self):
        (self.root/'one').write_bytes(b'1')
        (self.root/'two').write_bytes(b'1')
        self.assertEqual(self.scan(output_files=1),'FILESYSTEM_LIMIT')
    def test_worker_uses_exact_staging_copy(self):
        from anydes_tools.executor import _worker_environment
        root=Path(__file__).resolve().parents[4]
        home=self.root/'home'
        temp=self.root/'temp'
        home.mkdir()
        temp.mkdir()
        env=_worker_environment(root,home,temp)
        value=subprocess.run([sys.executable,'-c',
            'import anydes_tools.windows_sandbox as s; print(s.__file__)'],
            cwd=root,env=env,capture_output=True,text=True,timeout=10,check=True)
        self.assertEqual(Path(value.stdout.strip()).resolve(),Path(sandbox.__file__).resolve())
        self.assertEqual(env['ANYDES_STAGING_B2_SOURCE_SHA256'],
            hashlib.sha256(Path(sandbox.__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main()
