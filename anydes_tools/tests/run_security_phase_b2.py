"""Persist reproducible B2/B1/A regression counts and integrity, no model calls."""
import hashlib
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULES = ["anydes_tools.tests." + n for n in (
    "test_windows_sandbox_b2", "test_security_boundary_b1", "test_architecture_v001",
    "test_runtime_observation_phase_a", "test_runtime_comparison_phase_a1")]


def main():
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    baseline=json.loads((ROOT / "ANYDES_SECURITY_BOUNDARY_B1_HASHES.json").read_text())
    production=[]
    for item in baseline["production"]:
        digest=hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()
        production.append({"path":item["path"],"before":item["after"].lower(),"after":digest,"unchanged":digest==item["after"].lower()})
    unchanged_staging=[]
    for item in baseline["files"]:
        if "conversation/" in item["path"].replace("\\","/") or item["path"]=="probar_anydes_tools_conversacion_v001_5.py":
            digest=hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()
            unchanged_staging.append({"path":item["path"],"unchanged":digest==item["after"].lower()})
    changed=[]
    for path,name in (("anydes_tools/executor.py","executor.py"),("anydes_tools/official/developer_tool.py","developer_tool.py")):
        backup=ROOT / "archivo_versiones/2026-09-30_pre_b2" / name
        changed.append({"path":path,"before":hashlib.sha256(backup.read_bytes()).hexdigest(),"after":hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),"backup":str(backup.relative_to(ROOT))})
    added=[]
    for path in ("anydes_tools/windows_sandbox.py","anydes_tools/tests/test_windows_sandbox_b2.py","anydes_tools/tests/run_security_phase_b2.py","anydes_tools/tests/fixtures/phase_b2_baseline.json"):
        added.append({"path":path,"sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest()})
    report={"test_type":"UNIT_AND_REAL_WINDOWS_INTEGRATION","model_calls":0,"tests":result.testsRun,
        "passed":result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),"skipped":len(result.skipped),
        "failures":len(result.failures),"errors":len(result.errors),"duration_seconds":round(time.monotonic()-started,2),
        "skip_reasons":[reason for _,reason in result.skipped],"production":production,"unchanged_cognitive_staging":unchanged_staging,
        "changed":changed,"added":added}
    (ROOT / "ANYDES_WINDOWS_SANDBOX_B2_HASHES.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"production","unchanged_cognitive_staging","changed","added"}}))
    return 0 if result.wasSuccessful() and all(p["unchanged"] for p in production+unchanged_staging) else 1


if __name__=="__main__":
    raise SystemExit(main())
