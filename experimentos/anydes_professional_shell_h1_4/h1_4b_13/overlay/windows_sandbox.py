"""Trusted Windows host. Never fall back to an unrestricted subprocess.

Untrusted code gets an LPAC token with registryRead only, a private copy of
its runtime and a Job Object before its initial thread is resumed.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from .errors import PathDeniedError

STRONG = "STRONG_WINDOWS_SANDBOX"


@dataclass(frozen=True)
class Limits:
    timeout: float = 8.0
    memory: int = 256 * 1024 * 1024
    processes: int = 4
    cpu_percent: int = 25
    stdio_bytes: int = 1024 * 1024
    output_bytes: int = 20 * 1024 * 1024
    output_files: int = 256

    def __post_init__(self):
        if not (0 < self.timeout <= 30 and 64*1024*1024 <= self.memory <= 512*1024*1024
                and 1 <= self.processes <= 4 and 1 <= self.cpu_percent <= 50
                and 1 <= self.stdio_bytes <= 1024*1024 and 1 <= self.output_bytes <= 20*1024*1024
                and 1 <= self.output_files <= 256):
            raise ValueError("invalid_sandbox_limits")


class SandboxUnavailable(PathDeniedError):
    pass


def _deny(stage: str, error: int = 0):
    raise SandboxUnavailable(f"DENIED_STRONG_SANDBOX_UNAVAILABLE:{stage}:{error}")


def _safe_tree(path: Path, *, trusted_source=False):
    for parent in (path, *path.parents):
        if parent.exists() and (parent.lstat().st_file_attributes & 0x400):
            _deny("reparse")
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            info = (Path(root) / name).lstat()
            if info.st_file_attributes & 0x400 or (not trusted_source and info.st_nlink > 1 and not (Path(root) / name).is_dir()):
                _deny("unsafe_runtime_entry")


def _acl(path: Path, sid: str, permission: str):
    exe = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/icacls.exe")
    for args in ([str(path), "/grant", f"*{sid}:(OI)(CI){permission}", "/T", "/Q"],
                 [str(path), "/setintegritylevel", "(OI)(CI)L", "/T", "/Q"] if permission == "M" else None):
        if args is None:
            continue
        result = subprocess.run([exe, *args], capture_output=True, timeout=30)
        if result.returncode:
            _deny("acl", result.returncode)


def _python_runtime(destination: Path):
    base = Path(sys.base_prefix).resolve()
    _safe_tree(base / "DLLs", trusted_source=True)
    _safe_tree(base / "Lib", trusted_source=True)
    destination.mkdir()
    for name in ("python.exe", "python3.dll", f"python{sys.version_info.major}{sys.version_info.minor}.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        p = base / name
        if p.exists():
            if p.lstat().st_file_attributes & 0x400:
                _deny("runtime_reparse")
            shutil.copy2(p, destination / name)
    # Standard library only: never copy site-packages, user hooks or project code.
    shutil.copytree(base / "Lib", destination / "Lib", ignore=shutil.ignore_patterns(
        "site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip", "venv"))
    shutil.copytree(base / "DLLs", destination / "DLLs")
    _safe_tree(destination)
    return destination / "python.exe"


def _cpp_runtime(destination: Path):
    compiler = shutil.which("g++")
    if not compiler:
        _deny("cpp_missing_compiler")
    base = Path(compiler).resolve().parent.parent
    versions = list((base / "lib/gcc/x86_64-w64-mingw32").glob("*"))
    if len(versions) != 1 or not (base / "include/c++" / versions[0].name).is_dir():
        _deny("cpp_unsupported_layout")
    for source in (base / "include", base / "lib", base / "bin"):
        _safe_tree(source, trusted_source=True)
    destination.mkdir()
    (destination / "bin").mkdir(); (destination / "lib").mkdir()
    for name in ("g++.exe", "as.exe", "ld.exe", "ld.bfd.exe"):
        shutil.copy2(base / "bin" / name, destination / "bin" / name)
    for p in (base / "bin").glob("*.dll"):
        shutil.copy2(p, destination / "bin" / p.name)
    for p in (base / "lib").iterdir():
        if p.is_file() and p.suffix in {".a", ".o"}:
            shutil.copy2(p, destination / "lib" / p.name)
    shutil.copytree(base / "include", destination / "include")
    version_dir = destination / "lib/gcc/x86_64-w64-mingw32" / versions[0].name
    version_dir.mkdir(parents=True)
    for p in versions[0].iterdir():
        if p.is_file() and (p.name in {"cc1plus.exe", "collect2.exe"} or p.suffix in {".a", ".o"}):
            shutil.copy2(p, version_dir / p.name)
    for folder in ("include", "include-fixed"):
        shutil.copytree(versions[0] / folder, version_dir / folder)
    _safe_tree(destination)
    return destination / "bin/g++.exe"


class _Win:
    def __init__(self):
        if os.name != "nt":
            _deny("windows_required")
        self.k = C.WinDLL("kernel32", use_last_error=True)
        self.u = C.WinDLL("userenv", use_last_error=True)
        self.a = C.WinDLL("advapi32", use_last_error=True)
        self.b = C.WinDLL("kernelbase", use_last_error=True)
        P = C.c_void_p
        S = C.c_size_t
        class SI(C.Structure):
            _fields_ = [("cb", W.DWORD), ("reserved", W.LPWSTR), ("desktop", W.LPWSTR), ("title", W.LPWSTR),
                        ("x", W.DWORD), ("y", W.DWORD), ("xs", W.DWORD), ("ys", W.DWORD),
                        ("xc", W.DWORD), ("yc", W.DWORD), ("fill", W.DWORD), ("flags", W.DWORD),
                        ("show", W.WORD), ("reserved2", W.WORD), ("reservedp", P),
                        ("stdin", W.HANDLE), ("stdout", W.HANDLE), ("stderr", W.HANDLE)]
        class SX(C.Structure):
            _fields_ = [("si", SI), ("attrs", P)]
        class PI(C.Structure):
            _fields_ = [("process", W.HANDLE), ("thread", W.HANDLE), ("pid", W.DWORD), ("tid", W.DWORD)]
        class CAP(C.Structure):
            _fields_ = [("sid", P), ("capabilities", P), ("count", W.DWORD), ("reserved", W.DWORD)]
        class BASIC(C.Structure):
            _fields_ = [("process_time", C.c_longlong), ("job_time", C.c_longlong), ("flags", W.DWORD),
                        ("min_ws", S), ("max_ws", S), ("active", W.DWORD), ("affinity", S),
                        ("priority", W.DWORD), ("scheduling", W.DWORD)]
        class IO(C.Structure):
            _fields_ = [(n, C.c_ulonglong) for n in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class EXT(C.Structure):
            _fields_ = [("basic", BASIC), ("io", IO), ("process_memory", S), ("job_memory", S), ("peak_process", S), ("peak_job", S)]
        class ACCOUNT(C.Structure):
            _fields_ = [(n, C.c_longlong) for n in ("user", "kernel", "period_user", "period_kernel")]+[(n,W.DWORD) for n in ("faults","total","active","terminated")]
        self.SX, self.PI, self.CAP, self.EXT, self.ACCOUNT = SX, PI, CAP, EXT, ACCOUNT
        definitions = {
            "CreateJobObjectW": ([P, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, P, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, P, W.DWORD, P], W.BOOL),
            "AssignProcessToJobObject": ([W.HANDLE, W.HANDLE], W.BOOL),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
            "CloseHandle": ([W.HANDLE], W.BOOL),
            "LocalFree": ([P], P),
            "InitializeProcThreadAttributeList": ([P, W.DWORD, W.DWORD, C.POINTER(S)], W.BOOL),
            "UpdateProcThreadAttribute": ([P, W.DWORD, S, P, S, P, P], W.BOOL),
            "DeleteProcThreadAttributeList": ([P], None),
            "CreateProcessW": ([W.LPCWSTR, W.LPWSTR, P, P, W.BOOL, W.DWORD, P, W.LPCWSTR, C.POINTER(SX), C.POINTER(PI)], W.BOOL),
            "ResumeThread": ([W.HANDLE], W.DWORD),
            "WaitForSingleObject": ([W.HANDLE, W.DWORD], W.DWORD),
            "GetExitCodeProcess": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
            "TerminateProcess": ([W.HANDLE, W.UINT], W.BOOL),
        }
        for name, (args, restype) in definitions.items():
            fn = getattr(self.k, name); fn.argtypes = args; fn.restype = restype
        self.u.CreateAppContainerProfile.argtypes = [W.LPCWSTR, W.LPCWSTR, W.LPCWSTR, P, W.DWORD, C.POINTER(P)]
        self.u.CreateAppContainerProfile.restype = C.c_long
        self.u.DeleteAppContainerProfile.argtypes = [W.LPCWSTR]
        self.u.DeleteAppContainerProfile.restype = C.c_long
        self.a.ConvertSidToStringSidW.argtypes = [P, C.POINTER(W.LPWSTR)]
        self.a.ConvertSidToStringSidW.restype = W.BOOL
        self.a.FreeSid.argtypes = [P]; self.a.FreeSid.restype = P
        self.b.DeriveCapabilitySidsFromName.argtypes = [W.LPCWSTR, C.POINTER(P), C.POINTER(W.DWORD), C.POINTER(P), C.POINTER(W.DWORD)]
        self.b.DeriveCapabilitySidsFromName.restype = W.BOOL
        self.a.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.HANDLE)]
        self.a.OpenProcessToken.restype = W.BOOL
        self.a.GetTokenInformation.argtypes = [W.HANDLE, C.c_int, P, W.DWORD, C.POINTER(W.DWORD)]
        self.a.GetTokenInformation.restype = W.BOOL

    def check(self, ok, stage):
        if not ok:
            _deny(stage, C.get_last_error())

    def token_is_container(self, process):
        token = W.HANDLE()
        self.check(self.a.OpenProcessToken(process, 8, C.byref(token)), "token_open")
        try:
            value, size = W.DWORD(), W.DWORD()
            self.check(self.a.GetTokenInformation(token, 29, C.byref(value), C.sizeof(value), C.byref(size)), "token_query")
            return bool(value.value)
        finally:
            self.k.CloseHandle(token)


def _usage(roots, limits):
    count = total = 0
    for base in roots:
        for root, dirs, files in os.walk(base, followlinks=False):
            for name in dirs + files:
                try:
                    info = (Path(root) / name).lstat()
                except FileNotFoundError:
                    continue
                if info.st_file_attributes & 0x400 or info.st_nlink > 1 and not (Path(root) / name).is_dir():
                    return "REPARSE_OUTPUT"
                count += 1
                total += info.st_size if (Path(root) / name).is_file() else 0
                if count > limits.output_files or total > limits.output_bytes:
                    return "FILESYSTEM_LIMIT"
    return ""


def _remove_private(directory):
    # After job termination, detach junctions/symlinks, never follow their targets.
    info = directory.lstat()
    if info.st_file_attributes & 0x400:
        if directory.is_symlink():
            directory.unlink()
        else:
            directory.rmdir()
        return
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            p = Path(root) / name
            info = p.lstat()
            if info.st_file_attributes & 0x400:
                if info.st_file_attributes & 0x10 and not p.is_symlink():
                    p.rmdir()
                else:
                    p.unlink()
                if name in dirs:
                    dirs.remove(name)
    shutil.rmtree(directory)


def cleanup_worker_profiles(runtime_temp: Path):
    """Trusted parent cleanup if the tool worker was terminated mid-request."""
    base = runtime_temp / "b2_profiles"
    if not base.exists():
        return
    try:
        w = _Win()
    except (OSError, AttributeError) as exc:
        raise SandboxUnavailable("DENIED_STRONG_SANDBOX_UNAVAILABLE:windows_api") from exc
    for directory in base.iterdir():
        if len(directory.name) != 32 or any(c not in "0123456789abcdef" for c in directory.name):
            _deny("cleanup_unknown_entry")
        hr = w.u.DeleteAppContainerProfile("anydes.b2." + directory.name)
        # ERROR_FILE_NOT_FOUND means the worker already removed its profile.
        if hr < 0 and (hr & 0xFFFFFFFF) not in {0x80070002, 0x80070003}:
            _deny("cleanup_profile", hr & 0xFFFFFFFF)
        execution = runtime_temp.parents[1] / "executions_b2" / directory.name
        if execution.exists():
            _remove_private(execution)
        directory.rmdir()


def _run(w: _Win, sid, command, cwd: Path, home: Path, logs: Path, limits: Limits, stdin: str):
    import msvcrt
    logs.mkdir(exist_ok=True)
    (logs / "stdin").write_text(stdin[:10000], encoding="utf-8")
    handles = []
    job = attrs = None
    pi = w.PI()
    assigned = False
    result = {"sandbox_backend": STRONG, "lpac": True, "capabilities": ["registryRead"], "network_capabilities": [], "resource_limit_events": [],
              "memory_limit_bytes": limits.memory, "process_limit": limits.processes, "cpu_hard_cap_percent": limits.cpu_percent}
    group_sids, cap_sids = C.c_void_p(), C.c_void_p()
    group_count, cap_count = W.DWORD(), W.DWORD()
    try:
        for name, mode in (("stdin", "rb"), ("stdout", "wb"), ("stderr", "wb")):
            handles.append(open(logs / name, mode))
        raw = [msvcrt.get_osfhandle(f.fileno()) for f in handles]
        for h in raw:
            os.set_handle_inheritable(h, True)
        job = w.k.CreateJobObjectW(None, None)
        w.check(job, "job_create")
        ext = w.EXT()
        ext.basic.flags = 0x2000 | 0x8 | 0x100 | 0x200  # kill-on-close, processes, process/job memory; no breakaway
        ext.basic.active = limits.processes
        ext.process_memory = ext.job_memory = limits.memory
        w.check(w.k.SetInformationJobObject(job, 9, C.byref(ext), C.sizeof(ext)), "job_limits")
        cpu = (W.DWORD * 2)(5, limits.cpu_percent * 100)  # ENABLE | HARD_CAP
        w.check(w.k.SetInformationJobObject(job, 15, cpu, C.sizeof(cpu)), "cpu_limit")
        size = C.c_size_t()
        w.k.InitializeProcThreadAttributeList(None, 3, 0, C.byref(size))
        attrs = C.create_string_buffer(size.value)
        w.check(w.k.InitializeProcThreadAttributeList(attrs, 3, 0, C.byref(size)), "attributes_init")
        w.check(w.b.DeriveCapabilitySidsFromName("registryRead", C.byref(group_sids), C.byref(group_count), C.byref(cap_sids), C.byref(cap_count)), "capability_derive")
        class SIDATTR(C.Structure):
            _fields_ = [("sid", C.c_void_p), ("attributes", W.DWORD)]
        cap_array = (SIDATTR * cap_count.value)(*[SIDATTR(s, 4) for s in C.cast(cap_sids, C.POINTER(C.c_void_p))[:cap_count.value]])
        caps = w.CAP(sid, C.cast(cap_array,C.c_void_p), cap_count.value, 0)
        optout = W.DWORD(1)
        inherited = (W.HANDLE * 3)(*raw)
        for key, value in ((0x20009, caps), (0x2000f, optout), (0x20002, inherited)):
            w.check(w.k.UpdateProcThreadAttribute(attrs, 0, key, C.byref(value), C.sizeof(value), None, None), "attributes_set")
        sx = w.SX(); sx.si.cb = C.sizeof(sx); sx.attrs = C.cast(attrs, C.c_void_p)
        sx.si.flags = 0x100; sx.si.stdin, sx.si.stdout, sx.si.stderr = raw
        env = {"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"), "WINDIR": os.environ.get("SystemRoot", r"C:\Windows"),
               "HOME": str(home), "USERPROFILE": str(home), "TEMP": str(home), "TMP": str(home),
               "APPDATA": str(home), "LOCALAPPDATA": str(home), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        env["PATH"] = str(Path(command[0]).parent)
        block = C.create_unicode_buffer("\0".join(f"{k}={v}" for k,v in sorted(env.items())) + "\0\0")
        cmd = C.create_unicode_buffer(subprocess.list2cmdline([str(c) for c in command]))
        w.check(w.k.CreateProcessW(str(command[0]), cmd, None, None, True,
            0x80000 | 0x400 | 0x4 | 0x08000000, block, str(cwd), C.byref(sx), C.byref(pi)), "process_create")
        for h in raw:
            os.set_handle_inheritable(h, False)
        w.check(w.k.AssignProcessToJobObject(job, pi.process), "job_assign")
        assigned = True
        if not w.token_is_container(pi.process):
            _deny("token_not_appcontainer")
        (logs / "process.json").write_text(json.dumps({"pid":pi.pid}), encoding="utf-8")
        if w.k.ResumeThread(pi.thread) == 0xFFFFFFFF:
            _deny("resume", C.get_last_error())
        result["pid"] = pi.pid
        started = time.monotonic()
        event = ""
        while w.k.WaitForSingleObject(pi.process, 25) == 0x102:
            if time.monotonic() - started > limits.timeout:
                event = "TIMEOUT"
            elif any((logs / n).stat().st_size > limits.stdio_bytes for n in ("stdout", "stderr")):
                event = "STDIO_LIMIT"
            else:
                event = _usage((cwd, home), limits)
            if event:
                w.check(w.k.TerminateJobObject(job, 0xB2), "job_terminate")
                break
        w.k.WaitForSingleObject(pi.process, 5000)
        code = W.DWORD()
        w.check(w.k.GetExitCodeProcess(pi.process, C.byref(code)), "exit_code")
        if code.value == 259:
            _deny("process_survived")
        # Also kill descendants when the initial process exits normally.
        w.check(w.k.TerminateJobObject(job, 0xB2), "job_cleanup")
        account = w.ACCOUNT()
        for _ in range(100):
            w.check(w.k.QueryInformationJobObject(job, 1, C.byref(account), C.sizeof(account), None), "job_query")
            if not account.active:
                break
            time.sleep(.01)
        if account.active:
            _deny("descendant_survived")
        measured = w.EXT()
        w.check(w.k.QueryInformationJobObject(job, 9, C.byref(measured), C.sizeof(measured), None), "memory_query")
        result.update(returncode=C.c_int32(code.value).value, process_count=account.total,
            residual_processes=account.active, duration_ms=round((time.monotonic()-started)*1000),
            peak_job_memory_bytes=measured.peak_job, resource_limit_events=[event] if event else [], timeout=event == "TIMEOUT")
    finally:
        if pi.process:
            if assigned:
                w.k.TerminateJobObject(job, 0xB2)
            else:
                w.k.TerminateProcess(pi.process, 0xB2)
            w.k.WaitForSingleObject(pi.process, 5000)
        for h in (pi.thread, pi.process, job):
            if h:
                w.k.CloseHandle(h)
        if attrs:
            w.k.DeleteProcThreadAttributeList(attrs)
        for array, count in ((group_sids, group_count), (cap_sids, cap_count)):
            if array:
                for item in C.cast(array, C.POINTER(C.c_void_p))[:count.value]:
                    w.k.LocalFree(item)
                w.k.LocalFree(array)
        for f in handles:
            os.set_handle_inheritable(msvcrt.get_osfhandle(f.fileno()), False)
            f.close()
    for name in ("stdout", "stderr"):
        with (logs / name).open("rb") as f:
            value = f.read(limits.stdio_bytes)
        result[name] = value.decode("utf-8", "replace")[-12000:].replace(str(cwd.parent), "<sandbox>").replace(cwd.parent.as_posix(), "<sandbox>")
        if (logs / name).stat().st_size > limits.stdio_bytes and "STDIO_LIMIT" not in result["resource_limit_events"]:
            result["resource_limit_events"].append("STDIO_LIMIT")
    final_usage = _usage((cwd, home), limits)
    if final_usage and final_usage not in result["resource_limit_events"]:
        result["resource_limit_events"].append(final_usage)
    if "MemoryError" in result["stderr"] and result["peak_job_memory_bytes"] >= limits.memory * .9:
        result["resource_limit_events"].append("MEMORY_LIMIT")
    return result


def execute_private(tools_root: Path, source: str, *, language="python", stdin="", limits=Limits(), inputs=None, runtime_temp=None):
    """Host API, not registered as a tool. Tests may probe raw code without AST."""
    if os.environ.get("ANYDES_SANDBOX_BACKEND", STRONG) != STRONG:
        _deny("backend_not_strong")
    try:
        w = _Win()
    except (OSError, AttributeError) as exc:
        raise SandboxUnavailable("DENIED_STRONG_SANDBOX_UNAVAILABLE:windows_api") from exc
    tools_root = tools_root.resolve()
    # Keep compiler paths below legacy MinGW MAX_PATH search limits. A trusted
    # journal in the worker temp area lets its parent reclaim interrupted runs.
    parent = tools_root / "sandbox" / "executions_b2"
    parent.mkdir(parents=True, exist_ok=True)
    for p in (parent, *parent.parents):
        if p.lstat().st_file_attributes & 0x400:
            _deny("reparse_parent")
    request = uuid4().hex
    directory = parent / request
    directory.mkdir()
    journal = None
    name = "anydes.b2." + request
    sid = C.c_void_p()
    profile = False
    result = {"sandbox_backend": "UNAVAILABLE", "request_id": request, "language": language,
              "timestamp":datetime.now(timezone.utc).isoformat(), "network_policy":"DENY"}
    try:
        if runtime_temp:
            journal = Path(runtime_temp) / "b2_profiles" / request
            journal.mkdir(parents=True)
        hr = w.u.CreateAppContainerProfile(name, name, "Anydes private execution", None, 0, C.byref(sid))
        if hr < 0:
            _deny("profile_create", hr & 0xFFFFFFFF)
        profile = True
        text = W.LPWSTR()
        w.check(w.a.ConvertSidToStringSidW(sid, C.byref(text)), "sid_string")
        try:
            sid_text = text.value
        finally:
            w.k.LocalFree(C.cast(text, C.c_void_p))
        work, home = directory / "work", directory / "home"
        work.mkdir(); home.mkdir()
        # Windows rewrites TEMP/LOCALAPPDATA under the package profile at launch.
        (home / "Packages" / name / "AC/Temp").mkdir(parents=True)
        for filename, data in (inputs or {}).items():
            if Path(filename).name != filename or filename in {".", ".."} or len(data) > limits.output_bytes:
                _deny("invalid_input")
            (work / filename).write_bytes(data)
        runtime = directory / "runtime"
        if language == "python":
            (work / "program.py").write_text(source, encoding="utf-8")
            exe = _python_runtime(runtime)
            command = [exe, "-I", "-B", work / "program.py"]
        elif language == "cpp":
            (work / "program.cpp").write_text(source, encoding="utf-8")
            exe = _cpp_runtime(runtime)
            command = [exe, "-std=c++20", "-O0", "-static", "-fno-use-linker-plugin", work / "program.cpp", "-o", work / "program.exe"]
        else:
            _deny("unsupported_language")
        _acl(runtime, sid_text, "RX")
        _acl(work, sid_text, "M"); _acl(home, sid_text, "M")
        result.update(_run(w, sid, command, work, home, directory / "host_logs", limits if language == "python" else Limits(timeout=15, memory=512*1024*1024), stdin))
        if language == "cpp":
            result["phase"] = "compile"
            if result["returncode"] == 0 and not result["resource_limit_events"]:
                result["compile"] = {k:result[k] for k in ("returncode", "process_count", "duration_ms")}
                result.update(_run(w, sid, [work / "program.exe"], work, home, directory / "run_logs", limits, stdin))
                result["phase"] = "execute"
        if result["returncode"] == 0 and not result["resource_limit_events"]:
            _safe_tree(work)
            outputs = {}
            for p in work.rglob("*"):
                if p.is_file() and p.name not in {"program.py", "program.cpp", "program.exe"} and p.name not in (inputs or {}):
                    outputs[str(p.relative_to(work))] = p.read_bytes()
            if len(outputs) > limits.output_files or sum(map(len, outputs.values())) > limits.output_bytes:
                _deny("output_limit")
            result["outputs"] = outputs
        return result
    except BaseException as exc:
        result["error_code"] = str(exc) if isinstance(exc, SandboxUnavailable) else type(exc).__name__
        raise
    finally:
        if sid:
            w.a.FreeSid(sid)
        cleanup = []
        if profile:
            hr = w.u.DeleteAppContainerProfile(name)
            if hr < 0:
                cleanup.append(f"profile:{hr & 0xFFFFFFFF}")
        try:
            _remove_private(directory)
        except (OSError, PathDeniedError):
            cleanup.append("private_directory")
        if journal and journal.exists() and not cleanup:
            journal.rmdir()
        result["cleanup_status"] = "CLEAN" if not cleanup else "FAILED:" + ",".join(cleanup)
        # Only structured metadata; never stdout/source/environment/artifact data.
        log = tools_root / "logs" / "sandbox_b2.jsonl"
        log.parent.mkdir(exist_ok=True)
        event = {k: v for k,v in result.items() if k not in {"stdout", "stderr", "outputs"}}
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if cleanup:
            _deny("cleanup_failed")


def publish_output(context, output: bytes, destination: Path, *, expected_sha256: str):
    """Trusted publication only after caller verification; no overwrite implied."""
    if hashlib.sha256(output).hexdigest() != expected_sha256:
        _deny("publication_hash")
    target = context.revalidate(destination, "WRITE")
    temp = target.with_name(".b2-publish-" + uuid4().hex)
    try:
        with context.revalidate(temp, "WRITE").open("xb") as handle:
            handle.write(output)
        if hashlib.sha256(context.revalidate(temp, "READ").read_bytes()).hexdigest() != expected_sha256:
            _deny("publication_readback")
        context.revalidate(target, "WRITE")
        temp.rename(target)  # Windows rename does not replace an existing target.
    finally:
        if temp.exists():
            temp.unlink()
    return {"path": str(target), "sha256": expected_sha256, "bytes": len(output)}
