"""C unit/integration evidence; never labels brain doubles as full-runtime E2E."""
import ast
import hashlib
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / "archivo_versiones/2026-10-04_pre_c"
MODULES = ["anydes_tools.tests." + name for name in (
    "test_interaction_execution_phase_c",
    "test_interaction_router_code_dataflow_v001_5",
    "test_general_task_reasoner_v001_5",
    "test_security_boundary_b1", "test_architecture_v001", "test_windows_sandbox_b2",
    "test_runtime_observation_phase_a", "test_runtime_comparison_phase_a1",
)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    baseline = json.loads((ROOT / "ANYDES_SECURITY_BOUNDARY_B1_HASHES.json").read_text())
    production = [{"path": r["path"], "before": r["after"].lower(),
        "after": digest(ROOT / r["path"]), "unchanged": digest(ROOT / r["path"]) == r["after"].lower()}
        for r in baseline["production"]]
    changed = []
    for backup in BACKUP.rglob("*.py"):
        relative = backup.relative_to(BACKUP)
        ast.parse(backup.read_text(encoding="utf-8-sig"))
        changed.append({"path": relative.as_posix(), "before": digest(backup),
            "after": digest(ROOT / relative), "backup": str(backup.relative_to(ROOT)),
            "rollback_source_parseable": True})
    b2 = json.loads((ROOT / "ANYDES_WINDOWS_SANDBOX_B2_HASHES.json").read_text())
    sandbox = [{"path": row["path"], "unchanged": digest(ROOT / row["path"]) == row.get("after", row.get("sha256", "")).lower()}
        for row in b2["changed"] + b2["added"] if row["path"].endswith(".py")]
    added_paths = [
        "anydes_tools/conversation/executable_plan.py",
        "anydes_tools/tests/test_interaction_execution_phase_c.py",
        "anydes_tools/tests/run_interaction_execution_phase_c.py",
        "anydes_tools/tests/fixtures/phase_c_runtime.json",
    ]
    added = [{"path": path, "sha256": digest(ROOT / path)} for path in added_paths]
    report = {"test_type": "UNIT_AND_WINDOWS_INTEGRATION", "model_calls": 0,
        "brain_doubles": "Explicit queue/lambda/mocks in UNIT and cognitive INTEGRATION tests",
        "tests": result.testsRun,
        "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
        "skipped": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors),
        "skip_reasons": [reason for _, reason in result.skipped],
        "duration_seconds": round(time.monotonic() - started, 2),
        "production": production, "sandbox_b2": sandbox, "changed": changed, "added": added,
        "real_groq_replay": "BLOCKED_PENDING_EXTERNAL_PAYLOAD_APPROVAL",
        "rollback": "Original source snapshots parsed and hashed, live restoration not performed",
    }
    (ROOT / "ANYDES_INTERACTION_EXECUTION_C_HASHES.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"production", "sandbox_b2", "changed", "added"}}))
    return 0 if result.wasSuccessful() and all(r["unchanged"] for r in production + sandbox) else 1


if __name__ == "__main__":
    raise SystemExit(main())
