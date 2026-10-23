"""Real Windows integration tests; raw probes deliberately bypass AST policy.

No model doubles. These are not UI/voice or full cognitive E2E tests.
Run with the Windows account that owns an initialized user profile.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from anydes_tools.windows_sandbox import execute_private, Limits, SandboxUnavailable, publish_output, cleanup_worker_profiles
from anydes_tools.context import ToolRuntimeContext
from anydes_tools.path_policy import FilesystemBoundary
from anydes_tools.registry import build_official_registry
from anydes_tools.executor import ToolExecutor
from anydes_tools.permissions import PermissionBridge
from anydes_tools.audit import ToolAuditLog
from anydes_tools.contracts import ToolRequest, PermissionEffect, ExecutionStatus

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"
EVIDENCE = []


@unittest.skipUnless(os.name == "nt", "Windows OS boundary required")
class WindowsB2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=TOOLS / "workspaces", prefix="b2-fixtures-")
        cls.base = Path(cls.temp.name)
        cls.protected = cls.base / ".env"
        cls.protected.write_text("FAKE_B2_CANARY_NOT_A_SECRET")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        out = TOOLS / "logs" / "b2_test_results.json"
        out.write_text(json.dumps(EVIDENCE, indent=2), encoding="utf-8")

    def run_code(self, code, **kwargs):
        result = execute_private(TOOLS, code, **kwargs)
        self.assertEqual(result["cleanup_status"], "CLEAN")
        self.assertEqual(result["residual_processes"], 0)
        self.assertTrue(result["lpac"])
        self.assertEqual(result["network_capabilities"], [])
        EVIDENCE.append({"test": self.id().rsplit(".",1)[-1], **{k:v for k,v in result.items() if k not in {"stdout", "stderr", "outputs"}},
                         "stdout_bytes": len(result["stdout"].encode()), "stderr_bytes": len(result["stderr"].encode()),
                         "stdout_sha256": hashlib.sha256(result["stdout"].encode()).hexdigest()})
        return result

    def test_python_positive(self):
        r = self.run_code("import math,json\nn=float(input('Numero: '))\nprint(json.dumps({'result': math.sqrt(n)}))", stdin="81\n")
        self.assertEqual(r["returncode"], 0)
        self.assertIn('"result": 9.0', r["stdout"])

    def test_celsius_original_no_oracle_change(self):
        code = "c=float(input('Celsius: '))\nprint(c*9/5+32)"
        r = self.run_code(code, stdin="0\n")
        self.assertEqual(r["returncode"], 0)
        self.assertIn("32.0", r["stdout"])
        bad = self.run_code(code, stdin="abc\n")
        self.assertNotEqual(bad["returncode"], 0)
        self.assertIn("ValueError", bad["stderr"])

    def test_allowed_file_input_output_and_host_publication(self):
        r = self.run_code("from pathlib import Path\nPath('result.txt').write_text(Path('input.txt').read_text().upper())", inputs={"input.txt": b"hello"})
        self.assertEqual(r["outputs"], {"result.txt": b"HELLO"})
        target = self.base / "published.txt"
        self.assertFalse(target.exists())
        context = ToolRuntimeContext(ROOT, TOOLS, FilesystemBoundary(ROOT, TOOLS), (), self.base, self.base)
        publish_output(context, r["outputs"]["result.txt"], target, expected_sha256=hashlib.sha256(b"HELLO").hexdigest())
        self.assertEqual(target.read_bytes(), b"HELLO")
        with self.assertRaises(FileExistsError):
            publish_output(context, b"HELLO", target, expected_sha256=hashlib.sha256(b"HELLO").hexdigest())

    def test_filesystem_escape_os_not_ast(self):
        paths = [str(self.protected), str(ROOT / "anydes_tools/executor.py"), str(ROOT / "programa_principal"), str(self.base)]
        code = "import os,json\npaths=" + repr(paths) + "\nout=[]\nfor p in paths:\n try:\n  os.listdir(p) if os.path.isdir(p) else open(p).read()\n  out.append('ESCAPED')\n except PermissionError:\n  out.append('DENIED')\nprint(json.dumps(out))"
        r = self.run_code(code)
        self.assertEqual(json.loads(r["stdout"]), ["DENIED"]*4)
        destination = self.base / "escaped.txt"
        r = self.run_code("try:\n open("+repr(str(destination))+",'w').write('bad')\n print('ESCAPED')\nexcept PermissionError:\n print('DENIED')")
        self.assertEqual(r["stdout"].strip(), "DENIED")
        self.assertFalse(destination.exists())

    def test_traversal_private_parent_not_visible(self):
        r = self.run_code("try:\n open('../host_logs/stdout','w').write('escape')\n print('ESCAPED')\nexcept PermissionError:\n print('DENIED')")
        self.assertEqual(r["stdout"].strip(), "DENIED")

    def test_lpac_excludes_all_application_packages(self):
        path = self.base / "all-app-packages.txt"
        path.write_text("public_app_fixture")
        exe = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/icacls.exe")
        grant = subprocess.run([exe,str(path),"/grant","*S-1-15-2-1:R"],capture_output=True)
        self.assertEqual(grant.returncode,0)
        r = self.run_code("try:\n print(open("+repr(str(path))+").read())\nexcept PermissionError:\n print('LPAC_DENIED')")
        self.assertEqual(r["stdout"].strip(),"LPAC_DENIED")

    def test_network_socket_http_dns(self):
        code = """import socket,urllib.request,json
results={}
for name,fn in [('socket',lambda: socket.create_connection(('1.1.1.1',443),timeout=1)),('http',lambda: urllib.request.urlopen('http://example.com',timeout=1)),('dns',lambda: socket.getaddrinfo('example.com',443))]:
 try:
  fn();results[name]='ALLOWED'
 except OSError as e:
  results[name]={'type':type(e).__name__,'errno':e.errno,'winerror':getattr(e,'winerror',None)}
print(json.dumps(results))
"""
        r = self.run_code(code)
        data = json.loads(r["stdout"])
        EVIDENCE[-1]["network_probe"] = data
        self.assertNotEqual(data["socket"], "ALLOWED")
        self.assertNotEqual(data["http"], "ALLOWED")
        self.assertNotEqual(data["dns"], "ALLOWED")

    def test_environment_secrets_not_inherited(self):
        with patch.dict(os.environ, {"GROQ_API_KEY":"B2_FAKE_NOT_REAL", "B2_FAKE_TOKEN":"not_real"}):
            r = self.run_code("import os,json\nprint(json.dumps({'key': 'GROQ_API_KEY' in os.environ, 'token':'B2_FAKE_TOKEN' in os.environ, 'home_ok':os.path.isdir(os.environ['USERPROFILE']), 'temp_ok':os.path.isdir(os.environ['TEMP'])}))")
        self.assertEqual(json.loads(r["stdout"]), {"key":False,"token":False,"home_ok":True,"temp_ok":True})

    def test_child_killed_after_parent_exit(self):
        r = self.run_code("import subprocess,sys\np=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(30)'])\nprint(p.pid)")
        pid = int(r["stdout"].strip())
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]; k.OpenProcess.restype=ctypes.c_void_p
        h = k.OpenProcess(0x1000,False,pid)
        if h:
            code = ctypes.c_uint32(); k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
            k.GetExitCodeProcess(h,ctypes.byref(code)); k.CloseHandle.argtypes=[ctypes.c_void_p]; k.CloseHandle(h)
            self.assertNotEqual(code.value,259)

    def test_process_count_limit(self):
        r = self.run_code("import subprocess,sys,json\np=[]\ntry:\n for i in range(10): p.append(subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(30)']))\nexcept OSError:\n print(json.dumps({'children':len(p),'denied':True}))")
        data = json.loads(r["stdout"])
        self.assertTrue(data["denied"])
        self.assertLessEqual(data["children"],3)

    def test_child_inherits_filesystem_restriction(self):
        child = "try:\n open("+repr(str(self.protected))+").read()\n print('ESCAPED')\nexcept PermissionError:\n print('DENIED')"
        r = self.run_code("import subprocess,sys\nr=subprocess.run([sys.executable,'-I','-c',"+repr(child)+"],capture_output=True,text=True)\nprint(r.stdout.strip())")
        self.assertEqual(r["stdout"].strip(),"DENIED")

    def test_timeout_whole_tree(self):
        r = self.run_code("import subprocess,sys\nsubprocess.Popen([sys.executable,'-I','-c','while True: pass'])\nwhile True: pass", limits=Limits(timeout=.3))
        self.assertTrue(r["timeout"])

    def test_breakaway_denied(self):
        r = self.run_code("import subprocess,sys\ntry:\n subprocess.Popen([sys.executable,'-c','print(1)'],creationflags=0x01000000)\n print('ESCAPED')\nexcept OSError:\n print('DENIED')")
        self.assertEqual(r["stdout"].strip(), "DENIED")

    def test_memory_pressure_bounded(self):
        r = self.run_code("x=[]\nwhile True: x.append(bytearray(8*1024*1024))", limits=Limits(memory=64*1024*1024,timeout=3))
        self.assertNotEqual(r["returncode"],0)
        self.assertIn("MEMORY_LIMIT",r["resource_limit_events"])
        self.assertIn("MemoryError",r["stderr"])
        # Peak aggregate includes Windows auxiliary process startup/teardown.
        self.assertLessEqual(r["peak_job_memory_bytes"], 72*1024*1024)

    def test_stdout_flood_bounded(self):
        r = self.run_code("while True: print('x'*10000,flush=True)", limits=Limits(stdio_bytes=32768,timeout=2))
        self.assertIn("STDIO_LIMIT",r["resource_limit_events"])
        self.assertLessEqual(len(r["stdout"]),12000)

    def test_many_files_bounded(self):
        r = self.run_code("from pathlib import Path\nfor i in range(2000): Path(str(i)).write_text('x')", limits=Limits(output_files=32,timeout=2))
        self.assertIn("FILESYSTEM_LIMIT",r["resource_limit_events"])
        self.assertNotIn("outputs",r)

    def test_junction_escape(self):
        # Host creates only an isolated fixture, then OS enforces the target ACL.
        code = "import os\ntry:\n os.symlink("+repr(str(self.base))+",'link',target_is_directory=True)\n print(open('link/.env').read())\nexcept OSError:\n print('DENIED')"
        r = self.run_code(code)
        self.assertEqual(r["stdout"].strip(),"DENIED")

    def test_cpp_positive(self):
        r = self.run_code('#include <iostream>\nint main(){int n;std::cin>>n;std::cout<<n*n;}', language="cpp",stdin="7\n")
        self.assertEqual(r["returncode"],0)
        self.assertEqual(r["stdout"],"49")
        self.assertEqual(r["compile"]["returncode"],0)

    def test_backend_disabled_never_legacy(self):
        for backend in ("UNAVAILABLE", "LEGACY_SANDBOX"):
            with patch.dict(os.environ, {"ANYDES_SANDBOX_BACKEND":backend}):
                with self.assertRaises(SandboxUnavailable):
                    execute_private(TOOLS,"print('never')")

    def test_junction_parent_rejected_before_launch(self):
        fake_tools = self.base / "fake-tools"
        (fake_tools / "sandbox").mkdir(parents=True)
        destination = self.base / "junction-target"
        destination.mkdir()
        junction = fake_tools / "sandbox/executions_b2"
        completed = subprocess.run(["cmd","/c","mklink","/J",str(junction),str(destination)],capture_output=True)
        if completed.returncode:
            self.skipTest("fixture junction privilege unavailable")
        try:
            with self.assertRaises(SandboxUnavailable):
                execute_private(fake_tools,"print('never')")
            self.assertEqual(list(destination.iterdir()),[])
        finally:
            junction.rmdir()

    def test_executor_disabled_backend_denied(self):
        registry = build_official_registry(TOOLS)
        executor = ToolExecutor(ROOT,TOOLS,registry,PermissionBridge(),ToolAuditLog(TOOLS / "logs/b2_executor_test.jsonl"))
        p = self.base / "disabled.py"; p.write_text("print('never')")
        req = ToolRequest("b2-disable","b2","official.developer","developer.run_python",{"path":str(p)},(),(PermissionEffect.EXECUTE_CODE,),True,True)
        with patch.dict(os.environ,{"ANYDES_SANDBOX_BACKEND":"LEGACY_SANDBOX"}):
            r = executor.execute(req)
        self.assertEqual(r.status,ExecutionStatus.DENIED)
        self.assertIn("DENIED_STRONG_SANDBOX_UNAVAILABLE",r.error)

    def test_host_crash_job_kill_and_parent_recovery(self):
        from anydes_tools.executor import _prepare_worker_runtime, _cleanup_worker_runtime, _worker_environment
        import sys
        home,temp = _prepare_worker_runtime(TOOLS,"b2-host-crash")
        code = "from pathlib import Path; from anydes_tools.windows_sandbox import execute_private,Limits; execute_private(Path('anydes_tools'),'while True: pass',limits=Limits(timeout=20),runtime_temp=Path("+repr(str(temp))+"))"
        host = subprocess.Popen([sys.executable,"-B","-c",code],cwd=ROOT,env=_worker_environment(ROOT,home,temp),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        execution = None
        try:
            deadline = time.monotonic()+15
            while time.monotonic()<deadline:
                journals = list((temp / "b2_profiles").glob("*"))
                if journals:
                    execution = TOOLS / "sandbox/executions_b2" / journals[0].name
                    if (execution / "host_logs/process.json").exists():
                        break
                time.sleep(.02)
            self.assertIsNotNone(execution)
            self.assertTrue((execution / "host_logs/process.json").exists())
            pid = json.loads((execution / "host_logs/process.json").read_text())["pid"]
            host.kill(); host.wait(timeout=5)
            k = ctypes.WinDLL("kernel32",use_last_error=True)
            k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]; k.OpenProcess.restype=ctypes.c_void_p
            handle = k.OpenProcess(0x100000,False,pid)
            if handle:
                k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
                self.assertEqual(k.WaitForSingleObject(handle,5000),0)
                k.CloseHandle.argtypes=[ctypes.c_void_p]; k.CloseHandle(handle)
            cleanup_worker_profiles(temp)
            self.assertFalse(execution.exists())
        finally:
            if host.poll() is None:
                host.kill(); host.wait(timeout=5)
            _cleanup_worker_runtime(home,temp)

    def test_rollback_files_restore_in_fixture(self):
        backup = ROOT / "archivo_versiones/2026-09-30_pre_b2"
        restore = self.base / "rollback-restore"
        restore.mkdir()
        expected = {"developer_tool.py":"81b476dc3a72f844025be7f583c6dd3ca82a7519ce23d21ecad9819757979a41",
                    "executor.py":"681744924ec4a4996fc2a1223189f1b3a19597bee73072bb41f3b53330f2c60e"}
        for name,digest in expected.items():
            content=(backup/name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(),digest)
            (restore/name).write_bytes(content)
            self.assertEqual(hashlib.sha256((restore/name).read_bytes()).hexdigest(),digest)

    def test_real_tool_executor_python_cpp(self):
        registry = build_official_registry(TOOLS)
        executor = ToolExecutor(ROOT,TOOLS,registry,PermissionBridge(),ToolAuditLog(TOOLS / "logs/b2_executor_test.jsonl"))
        for lang, source, capability in (("py","print(42)","developer.run_python"),("cpp",'#include <iostream>\nint main(){std::cout<<42;}',"developer.run_cpp")):
            path = self.base / ("positive."+lang); path.write_text(source)
            req = ToolRequest("b2-"+lang,"b2", "official.developer",capability,{"path":str(path)},(),(PermissionEffect.EXECUTE_CODE,),True,True)
            r = executor.execute(req)
            self.assertEqual(r.status,ExecutionStatus.SUCCESS,r.error)
            self.assertEqual(r.output["returncode"],0,r.output)
            self.assertEqual(r.output["stdout"].strip(),"42")
            self.assertFalse(path.with_suffix(".tool.exe").exists())


if __name__ == "__main__":
    unittest.main()
