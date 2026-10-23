"""D.5 INTEGRATION: prepared semantic input and code, real executor/B2/ZIP.

No cognition claim and no Groq calls. The profiler only observes real host returns.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from uuid import uuid4
import zipfile

from anydes_tools.contracts import ExecutionStatus, PermissionEffect
from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.dataflow import ArtifactRef, ArtifactState
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer
from anydes_tools.conversation.semantic_proposal import SemanticProposal, parse_semantic
from anydes_tools.settings.store import SettingsStore

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"
FIXTURE = Path(__file__).with_name("fixtures") / "phase_d5_sum.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_paths(value):
    if isinstance(value, dict):
        return {k: Path(v).name if isinstance(v, str) and (k.endswith("path_ref") or k == "path")
                else safe_paths(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_paths(v) for v in value]
    return value


class DataflowClosureIntegration(unittest.TestCase):
    def setUp(self):
        self.workspace = TOOLS / "workspaces" / ("phase_d5_" + uuid4().hex[:12])
        self.workspace.mkdir()
        settings = SettingsStore(ROOT, TOOLS, settings_path=self.workspace / "settings.json")
        settings.set_temporary_workspace(self.workspace)
        self.controller = ToolConversationController(ROOT, TOOLS, settings=settings)

    def tearDown(self):
        self.controller.close()
        shutil.rmtree(self.workspace)

    def execute_fixture(self, expected):
        c = self.controller
        payload = {"goal": "materialize, test and archive exactly one sum program", "operations": [
            {"operation_id": "write", "operation_kind": "GENERATE_CODE", "inputs": [],
             "outputs": [{"name": "code_a", "artifact_type": "CODE", "format": "py", "filename": "sum.py"}],
             "specification": {"language": "python", "goal": "sum two integers",
                "functional_requirements": ["read two integers", "print their sum"],
                "tests": [{"test_id": "sum_check", "stdin": "2\n4\n", "expected_stdout_contains": [expected]}],
                "execution_required": True}},
            {"operation_id": "test", "operation_kind": "TEST_CODE", "inputs": ["code_a"], "outputs": [], "specification": {}},
            {"operation_id": "archive", "operation_kind": "CREATE_ARCHIVE", "inputs": ["code_a"],
             "outputs": [{"name": "zip_b", "artifact_type": "ARCHIVE", "format": "zip", "filename": "sum.zip"}],
             "specification": {}}], "requested_final_outputs": ["zip_b"]}
        proposal = SemanticProposal.from_dict(parse_semantic(json.dumps(payload)))
        requirements = ExecutablePlanLowerer(c.registry, self.workspace).lower_requirements(proposal)
        action = c._action_from_general_plan(requirements, "D5 prepared semantic integration")
        source = FIXTURE.read_text(encoding="utf-8")
        action.arguments["generated_code_artifact"] = {
            "language": "python", "filename": "sum.py", "code": source, "tests": []}
        plan = c.plan_adapter.compile(action, source="GENERAL_REASONER")
        (self.workspace / "unrelated.txt").write_text("not an archive member", encoding="utf-8")
        transitions = []
        def observe(frame, event, result):
            if event == "return" and frame.f_code.co_name in {"produce", "transition"} and isinstance(result, ArtifactRef):
                transitions.append(safe_paths(asdict(result)))
        old_profile = sys.getprofile()
        sandbox_log = TOOLS / "logs/sandbox_b2.jsonl"
        before = sandbox_log.stat().st_size if sandbox_log.exists() else 0
        try:
            sys.setprofile(observe)
            results = c._execute(plan, turn_id="D5-" + expected,
                confirmed_effects=(PermissionEffect.EXECUTE_CODE,))
        finally:
            sys.setprofile(old_profile)
        with sandbox_log.open("rb") as stream:
            stream.seek(before)
            sandbox = [json.loads(row) for row in stream.read().decode("utf-8").splitlines()]
        self.assertTrue(sandbox)
        self.assertTrue(all(row["sandbox_backend"] == "STRONG_WINDOWS_SANDBOX" for row in sandbox), sandbox)
        self.assertTrue(all(row["cleanup_status"] == "CLEAN" for row in sandbox), sandbox)
        summary = results[-1].output["task_execution"]
        evidence = {"test_type": "INTEGRATION", "prepared_inputs": ["semantic proposal", "code fixture"],
            "replaced_components": [], "groq_calls": 0, "task_id": plan.task_id,
            "code_fixture_hash": digest(source.encode()), "transitions": transitions,
            "task_execution": safe_paths(summary), "sandbox": sandbox,
            "capabilities_executed": [r.capability for r in results],
            "execution_statuses": [r.status.value for r in results],
            "final_outputs": sorted(p.name for p in self.workspace.iterdir() if p.suffix in {".py", ".zip"}),
            "temporary_files_remaining": [p.name for p in self.workspace.glob(".anydes-*")]}
        return source, results, summary, evidence

    def save(self, name, evidence):
        folder = TOOLS / "logs/phase_d5"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".json")).write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")

    def test_success_real_b2_exact_zip_identity(self):
        source, results, summary, evidence = self.execute_fixture("6")
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        ledger = self.controller._artifact_bindings
        code = ledger.require("code_a", verified=True)
        archive = ledger.require("zip_b", verified=True)
        self.assertEqual(code.state, ArtifactState.COMMITTED)
        self.assertEqual(archive.state, ArtifactState.COMMITTED)
        code_states = [r["state"] for r in evidence["transitions"] if r["artifact_id"] == code.artifact_id]
        self.assertEqual(code_states, ["TEMPORARY", "VERIFIED", "COMMITTED"])
        self.assertTrue(all(r["version"] == 1 for r in evidence["transitions"]))
        self.assertEqual(Path(code.path).read_text(encoding="utf-8"), source)
        self.assertEqual(code.hash, digest(Path(code.path).read_bytes()))
        with zipfile.ZipFile(archive.path) as zipped:
            self.assertEqual(zipped.namelist(), ["sum.py"])
            self.assertEqual(digest(zipped.read("sum.py")), code.hash)
            evidence["zip_exactness"] = {"expected": ["sum.py"], "actual": zipped.namelist(), "member_hash": code.hash}
        archived = next(r for r in self.controller.last_step_results if r.step_id == "archive")
        tested = next(r for r in self.controller.last_step_results if r.step_id == "test")
        self.assertEqual(archived.consumed_artifacts, (code.artifact_id,))
        self.assertEqual(tested.consumed_artifacts, (code.artifact_id,))
        self.assertEqual(archive.metadata["consumed_artifacts"], [code.artifact_id])
        self.assertFalse(summary["blocked_steps"])
        self.assertFalse(evidence["temporary_files_remaining"])
        evidence["artifacts"] = safe_paths([asdict(code), asdict(archive)])
        self.save("success", evidence)

    def test_failure_blocks_consumer_real_execution(self):
        # Deliberately incorrect local assertion, not a change to any real model test.
        source, results, summary, evidence = self.execute_fixture("7")
        self.assertTrue(any(r.status == ExecutionStatus.VERIFY_FAILED for r in results))
        self.assertNotIn("archive.compress_zip", evidence["capabilities_executed"])
        self.assertEqual(summary["completed_steps"], ["write"])
        self.assertEqual(summary["failed_step"], "test")
        self.assertEqual(summary["blocked_steps"], ["archive"])
        blocked = next(r for r in self.controller.last_step_results if r.step_id == "archive")
        ref = self.controller._artifact_bindings.bindings["code_a"]
        self.assertEqual(blocked.status, "BLOCKED_DEPENDENCY")
        self.assertEqual(blocked.blocked_by_step_id, "test")
        self.assertEqual(blocked.blocked_artifact_id, ref.artifact_id)
        self.assertEqual(blocked.values["required_state"], "VERIFIED")
        self.assertEqual(blocked.values["actual_state"], "FAILED")
        self.assertEqual(ref.state, ArtifactState.FAILED)
        self.assertFalse(Path(ref.path).exists())
        self.assertFalse(summary["committed_artifacts"])
        self.assertFalse(summary["temporary_artifacts"])
        self.assertFalse(summary["quarantined_artifacts"])
        self.assertEqual(summary["removed_artifact_ids"], [ref.artifact_id])
        self.assertFalse(evidence["final_outputs"])
        self.assertFalse(evidence["temporary_files_remaining"])
        self.save("failure", evidence)
