"""UNIT bindings; INTEGRATION real workers/B2, with explicitly prepared cognition."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from anydes_tools.contracts import PermissionEffect, ExecutionStatus
from anydes_tools.conversation.dataflow import (
    ArtifactBindings, ArtifactState, DataflowError, file_hash, validate_dataflow,
)
from anydes_tools.conversation.executable_plan import ExecutableStep
from anydes_tools.conversation.contracts import PlannedToolAction, UserTaskSpec, StructuredActionPlan
from anydes_tools.tests.test_interaction_execution_phase_c import PhaseC
from anydes_tools.tests.test_failure_record_phase_c import FailureRecordIntegration


class BindingUnit(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "a.txt"
        self.path.write_text("before", encoding="utf-8")
        self.ledger = ArtifactBindings("task")

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_no_fallback(self):
        with self.assertRaisesRegex(DataflowError, "DATAFLOW_MISSING_ARTIFACT"):
            self.ledger.require("missing")

    def test_verified_required(self):
        self.ledger.produce("a", self.path, "write")
        with self.assertRaisesRegex(DataflowError, "UNVERIFIED"):
            self.ledger.require("a", verified=True)

    def test_failed_never_consumed(self):
        self.ledger.produce("a", self.path, "write", state=ArtifactState.FAILED)
        with self.assertRaisesRegex(DataflowError, "FAILED_ARTIFACT"):
            self.ledger.require("a")

    def test_stale_hash(self):
        self.ledger.produce("a", self.path, "write", state=ArtifactState.COMMITTED)
        self.path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(DataflowError, "STALE_ARTIFACT"):
            self.ledger.require("a")

    def test_exact_multiple_selection(self):
        b = self.path.with_name("b.txt")
        b.write_text("other", encoding="utf-8")
        first = self.ledger.produce("a", self.path, "write", state=ArtifactState.COMMITTED)
        second = self.ledger.produce("b", b, "write2", state=ArtifactState.COMMITTED)
        step = ExecutableStep("zip", "CAPABILITY", "archive.compress_zip", {"sources": ["workspace"]},
            consumes_artifacts=("b", "a"), requires_verified_inputs=True)
        args, refs = self.ledger.resolve_arguments(step)
        self.assertEqual(args["sources"], [second.path, first.path])

    def test_type_checked(self):
        self.ledger.produce("a", self.path, "write", state=ArtifactState.COMMITTED)
        with self.assertRaisesRegex(DataflowError, "TYPE_MISMATCH"):
            self.ledger.require("a", artifact_type="CODE")

    def test_version_identity_preserved(self):
        ref = self.ledger.produce("a", self.path, "write")
        changed = self.ledger.transition("a", ArtifactState.COMMITTED, new_version=True)
        self.assertEqual(ref.artifact_id, changed.artifact_id)
        self.assertEqual(changed.version, 2)


class DataflowIntegration(unittest.TestCase):
    setUp = PhaseC.setUp
    tearDown = PhaseC.tearDown
    controller = PhaseC.controller
    prepared_action = FailureRecordIntegration.prepared_action

    def code_action(self):
        base = self.prepared_action()
        return replace(base, capabilities=(*base.capabilities, "archive.compress_zip"), arguments={
            **base.arguments, "general_task_plan": {"required_capabilities": [*base.capabilities, "archive.compress_zip"],
                "desired_outputs": [{"binding": "code", "artifact_type": "CODE"}, {"binding": "zip", "format": "zip"}],
                "steps": [
                    {"step_id": "write", "capability": "developer.write_code", "produces_artifacts": ["code"]},
                    {"step_id": "test", "capability": "developer.run_python", "consumes_artifacts": ["code"], "depends_on": ["write"]},
                    {"step_id": "zip", "capability": "archive.compress_zip", "consumes_artifacts": ["code"], "depends_on": ["test"], "produces_artifacts": ["zip"]}]}})

    def execute(self, c, action):
        c._remember_task_plan(action)
        plan = c.state.pending_task.plan
        return c._execute(plan, turn_id="d", confirmed_effects=(PermissionEffect.EXECUTE_CODE, PermissionEffect.OVERWRITE))

    def test_code_test_zip_exact(self):
        (self.workspace / "unrelated.txt").write_text("never archive", encoding="utf-8")
        c = self.controller()
        results = self.execute(c, self.code_action())
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        with zipfile.ZipFile(self.workspace / "sum.zip") as archive:
            self.assertEqual(archive.namelist(), ["sum.py"])
            self.assertEqual(archive.read("sum.py"), (self.workspace / "sum.py").read_bytes())
        write = next(r for r in c.last_step_results if r.step_id == "write")
        test = next(r for r in c.last_step_results if r.step_id == "test")
        zipped = next(r for r in c.last_step_results if r.step_id == "zip")
        self.assertEqual(write.produced_artifacts, test.consumed_artifacts)
        self.assertEqual(write.produced_artifacts, zipped.consumed_artifacts)

    def test_failure_blocks_zip(self):
        action = self.code_action()
        arguments = dict(action.arguments)
        arguments["generated_code_artifact"] = {**arguments["generated_code_artifact"], "code": "print('wrong')\n"}
        c = self.controller()
        results = self.execute(c, replace(action, arguments=arguments))
        self.assertTrue(any(r.status == ExecutionStatus.VERIFY_FAILED for r in results))
        self.assertFalse((self.workspace / "sum.zip").exists())
        blocked = next(r for r in c.last_step_results if r.step_id == "zip")
        self.assertEqual(blocked.status, "BLOCKED_DEPENDENCY")
        self.assertTrue(blocked.blocked_by_step_id)

    def test_adapter_bindings_preserved(self):
        c = self.controller()
        plan = c.plan_adapter.compile(self.code_action(), source="LEGACY_ADAPTER")
        self.assertEqual(plan.steps[0].produces_artifacts, ("code",))
        self.assertEqual(plan.steps[2].consumes_artifacts, ("code",))

    def test_incomplete_legacy_archive_is_not_executable(self):
        from anydes_tools.conversation.interaction_router import InteractionResolution, InteractionKind
        c = self.controller()
        c.last_resolution = InteractionResolution(InteractionKind.TOOL_ACTION,
            requested_effects=("developer.write_code", "developer.run_python", "archive.compress_zip"), confidence=1)
        archive_only = PlannedToolAction("compress_zip", ("archive.compress_zip",), arguments={
            "sources": [str(self.workspace)], "destination": str(self.workspace / "wrong.zip")})
        with self.assertRaisesRegex(ValueError, "DATAFLOW_LEGACY_INCOMPLETE_EFFECTS"):
            c.plan_adapter.compile(archive_only, source="LEGACY_ADAPTER")
        self.assertFalse((self.workspace / "wrong.zip").exists())

    def test_missing_model_arguments_fail_closed(self):
        c = self.controller()
        action = PlannedToolAction("composition", ("document.create_docx",), "general_tool_composition",
            arguments={"general_task_plan": {"steps": [{"step_id": "bad", "capability": "document.create_docx",
                "arguments": {"content": "Nonempty but no destination"}}]}}, source_text="Create a document")
        with self.assertRaisesRegex(ValueError, "PLAN_INCOMPLETE"):
            c.plan_adapter.compile(action, source="GENERAL_REASONER")
        self.assertFalse(list(self.workspace.glob("*.docx")))

    def test_invalid_order_denied_before_execution(self):
        c = self.controller()
        plan = c.plan_adapter.compile(self.code_action(), source="LEGACY_ADAPTER")
        with self.assertRaises(DataflowError):
            validate_dataflow(replace(plan, steps=tuple(reversed(plan.steps))))

    def test_atomic_stale_preserves_original(self):
        c = self.controller()
        final = self.workspace / "final.txt"
        temp = self.workspace / "temp.txt"
        final.write_text("old", encoding="utf-8")
        temp.write_text("candidate", encoding="utf-8")
        c._artifact_bindings = ArtifactBindings("test")
        c._artifact_bindings.snapshot(final)
        plan = c.plan_adapter.compile(self.prepared_action(), source="LEGACY_ADAPTER")
        c._executing_plan = plan
        final.write_text("concurrent", encoding="utf-8")
        result = c._execute_capability("filesystem.atomic_replace", {
            "source": str(temp), "destination": str(final), "commit_effect": "OVERWRITE"},
            turn_id="stale", confirmed_effects=(PermissionEffect.OVERWRITE,))
        c._executing_plan = None
        self.assertNotEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertIn("STALE_ARTIFACT", result.error)
        self.assertEqual(final.read_text(), "concurrent")

    def test_word_pdf_exact_binding(self):
        c = self.controller()
        word = self.workspace / "source.docx"
        pdf = self.workspace / "source.pdf"
        content = "This is the exact document source. " * 20
        action = PlannedToolAction("composition", ("document.create_docx", "document.create_pdf"),
            "general_tool_composition", arguments={"general_task_plan": {
                "desired_outputs": [{"binding": "word", "format": "docx"}, {"binding": "pdf", "format": "pdf"}], "steps": [
                {"step_id": "word", "capability": "document.create_docx", "arguments": {"path": str(word), "content": content}, "produces_artifacts": ["word"]},
                {"step_id": "pdf", "capability": "document.create_pdf", "arguments": {"path": str(pdf)}, "consumes_artifacts": ["word"], "produces_artifacts": ["pdf"], "depends_on": ["word"]}]}},
            source_text="Create a Word document and convert that document to PDF")
        results = self.execute(c, action)
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        a = c._artifact_bindings.require("word", verified=True)
        b = c._artifact_bindings.require("pdf", verified=True)
        self.assertEqual(b.metadata["converted_from"], a.artifact_id)
        self.assertTrue(pdf.is_file())

    def test_partial_success_records_committed_and_blocked(self):
        c = self.controller()
        path = self.workspace / "retained.txt"
        action = PlannedToolAction("composition", ("document.create_txt", "filesystem.read", "archive.compress_zip"),
            "general_tool_composition", arguments={"general_task_plan": {
                "desired_outputs": [{"binding": "text", "format": "txt"}, {"binding": "zip", "format": "zip"}], "steps": [
                {"step_id": "write", "capability": "document.create_txt", "arguments": {"path": str(path), "content": "retained content"}, "produces_artifacts": ["text"]},
                {"step_id": "fail", "capability": "filesystem.read", "arguments": {"path": str(self.workspace / "absent.txt")}},
                {"step_id": "blocked", "capability": "archive.compress_zip", "arguments": {"destination": str(self.workspace / "no.zip")}, "consumes_artifacts": ["text"], "produces_artifacts": ["zip"], "depends_on": ["fail"]}]}},
            source_text="Create an artifact and then process it")
        self.execute(c, action)
        self.assertTrue(path.exists())
        self.assertTrue(c.last_dataflow_summary["committed_artifacts"])
        self.assertEqual(c.last_dataflow_summary["failed_step"], "fail")
        self.assertIn("blocked", c.last_dataflow_summary["blocked_steps"])

    def test_content_value_written_without_regeneration(self):
        c = self.controller()
        calls = []
        content = " ".join(["Anydes", "proyecto", "personal"] * 50)
        def generate(spec):
            calls.append(spec)
            return content
        c.semantic_prose_generator = generate
        path = self.workspace / "project.docx"
        action = PlannedToolAction("generate_document", ("document.create_docx",), "generated_create_document",
            arguments={"path": str(path), "content_generation_spec": {"content_kind": "expository_text",
                "topic": "Anydes", "purpose": "document", "target_length_words": 150,
                "language": "es", "tone": "natural", "structure": "paragraphs"}},
            task_spec=UserTaskSpec(task_id="content-task", goal="generate_document", target_type="document",
                output_format="docx", filename=path.name, destination=str(self.workspace)),
            source_text="Write 150 words about Anydes and save them in Word")
        action = replace(action, structured_action_plan=StructuredActionPlan(
            "content-plan", "EXACT_TOOL", "official.document", "document.create_docx",
            action.arguments, str(self.workspace), "content-task"))
        results = self.execute(c, action)
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        self.assertEqual(len(calls), 1)
        self.assertEqual(c._artifact_bindings.values["content.main"], content)
        self.assertTrue(any(r.step_id == "content_generation" and r.typed_values for r in c.last_step_results))

    def test_plan_source_stale_before_execution(self):
        c = self.controller()
        path = self.workspace / "source.py"
        path.write_text("print('old')\n", encoding="utf-8")
        action = replace(self.prepared_action(), operation="general_modify_code", arguments={
            "path": str(path), "code_modification_spec": {"language": "python", "requested_change": "new output"}})
        plan = c.plan_adapter.compile(action, source="LEGACY_ADAPTER")
        path.write_text("print('concurrent')\n", encoding="utf-8")
        with self.assertRaisesRegex(DataflowError, "STALE_ARTIFACT"):
            c.plan_adapter.revalidate(plan)

    def test_edit_zip_consumes_new_version(self):
        from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
        c = self.controller()
        path = self.workspace / "sum.py"
        path.write_text("print('before')\n", encoding="utf-8")
        before = file_hash(path)
        c.code_modifier = lambda spec: GeneratedCodeArtifact.from_dict({"language": "python",
            "filename": "sum.py", "code": "print('after')\n", "tests": []})
        # Explicitly reviewed local oracle for this prepared integration source.
        c.code_oracle_provider = lambda spec, test: replace(test, expected="after", oracle_source="LOCAL_REVIEW")
        action = self.code_action()
        action = replace(action, operation="general_modify_code",
            capabilities=("developer.read_code", *action.capabilities), arguments={**action.arguments,
                "code_modification_spec": {"language": "python", "requested_change": "change output",
                    "new_requirements": ["print after"], "execution_required": True,
                    "new_tests": [{"test_id": "new", "expected_stdout_contains": ["after"]}]},
                "general_task_plan": {"required_capabilities": ["developer.read_code", *action.capabilities],
                    "desired_outputs": [{"binding": "code", "artifact_type": "CODE"}, {"binding": "zip", "format": "zip"}],
                    "steps": [
                        {"step_id": "read", "capability": "developer.read_code"},
                        *action.arguments["general_task_plan"]["steps"]]}})
        results = self.execute(c, action)
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        ref = c._artifact_bindings.require("code", verified=True)
        self.assertEqual(ref.version, 2)
        self.assertEqual(ref.metadata["previous_hash"], before)
        with zipfile.ZipFile(self.workspace / "sum.zip") as archive:
            self.assertEqual(archive.read("sum.py"), path.read_bytes())
        self.assertEqual(path.read_text(encoding="utf-8"), "print('after')\n")

    def test_docx_edit_updates_operational_version(self):
        c = self.controller()
        path = self.workspace / "edited.docx"
        created = c._execute_capability("document.create_docx", {"path": str(path), "content": "Original paragraph"},
            turn_id="fixture", confirmed_effects=())
        self.assertEqual(created.status, ExecutionStatus.SUCCESS)
        before = file_hash(path)
        action = PlannedToolAction("edit", ("document.read_document", "document.edit_document"),
            "structural_document_edit", arguments={"path": str(path), "edit_spec": {
                "target_path": str(path), "operation": "APPEND_PARAGRAPH", "payload": "Final paragraph"}},
            source_text="Append the final paragraph to this document")
        results = self.execute(c, action)
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        ref = c._operational_artifacts[str(path.resolve())]
        self.assertEqual(ref.version, 2)
        self.assertEqual(ref.hash, file_hash(path))
        self.assertEqual(ref.metadata["previous_hash"], before)


del PhaseC, FailureRecordIntegration

if __name__ == "__main__":
    unittest.main()
