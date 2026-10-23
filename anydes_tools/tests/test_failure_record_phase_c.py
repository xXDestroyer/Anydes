"""UNIT metadata/rendering; INTEGRATION uses real workers/B2 with explicit cognition doubles."""
from dataclasses import asdict, fields
import json
import unittest
from unittest.mock import patch

from anydes_tools.contracts import ExecutionStatus, ToolExecutionResult
from anydes_tools.conversation.code_generation import CodeTestCase
from anydes_tools.conversation.contracts import PlannedToolAction
from anydes_tools.conversation.failure_state import (
    FailureRecord, LastToolFailure, FailureResponseRenderer, VerificationEvidence,
    artifact_type_for, plan_reference, text_summary,
)
from anydes_tools.conversation.interaction_router import InteractionKind as K, InteractionResolution as R, TaskControl as T
from anydes_tools.tests.test_interaction_execution_phase_c import PhaseC


class FailureRecordUnit(unittest.TestCase):
    def record(self, kind, **extra):
        return FailureRecord.create(task_id="T", stage="VERIFY", error_code="CHECK_FAILED",
            safe_user_message="La comprobacion registrada fallo.", retryable=True,
            artifact_type=kind, **extra)

    def test_one_record_with_legacy_alias_and_complete_schema(self):
        self.assertIs(LastToolFailure, FailureRecord)
        required = {"failure_id", "task_id", "plan_revision", "attempt_number", "step_id",
            "stage", "operation_type", "artifact_type", "capability_id", "error_code",
            "technical_message", "safe_user_message", "retryable", "original_plan_ref",
            "verification_ref", "timestamp"}
        self.assertTrue(required.issubset({f.name for f in fields(FailureRecord)}))
        self.assertNotEqual(self.record("CODE").failure_id, self.record("CODE").failure_id)

    def test_renderer_code_not_document(self):
        rendered = FailureResponseRenderer.render(self.record("CODE"))
        self.assertIn("programa", rendered)
        self.assertNotIn("documento", rendered)

    def test_renderer_document(self):
        self.assertIn("documento", FailureResponseRenderer.render(self.record("DOCUMENT")))

    def test_renderer_archive(self):
        self.assertIn("ZIP", FailureResponseRenderer.render(self.record("ARCHIVE")))

    def test_renderer_conversion_and_filesystem(self):
        self.assertIn("conversión", FailureResponseRenderer.render(self.record("CONVERSION")))
        self.assertIn("cambio en el archivo", FailureResponseRenderer.render(self.record("FILESYSTEM")))

    def test_unknown_cause_not_invented(self):
        record = FailureRecord.create(task_id="T", stage="VERIFY", error_code="UNKNOWN",
            safe_user_message="", retryable=False, artifact_type="CODE")
        response = FailureResponseRenderer.render(record)
        self.assertIn("no tengo una causa", response)
        self.assertNotIn("test", response)

    def test_secrets_and_large_outputs_not_stored(self):
        secret = "gsk_private123456"
        record = self.record("CODE", technical_message=f"api_key={secret} " + "x" * 2000)
        self.assertNotIn(secret, record.technical_message)
        self.assertLessEqual(len(record.technical_message), 500)
        run = ToolExecutionResult("r", "t", "official.developer", "developer.run_python",
            ExecutionStatus.SUCCESS, output={"returncode": 0, "stdout": secret * 10000, "stderr": secret})
        evidence = VerificationEvidence.code_test(CodeTestCase("test_003", secret, (secret,)), run, False)
        self.assertNotIn(secret, json.dumps(asdict(evidence)))
        self.assertEqual(evidence.failed_check, "test_003")
        self.assertEqual(evidence.cause_code, "TEST_ASSERTION_FAILED")
        self.assertEqual(evidence.actual_summary, text_summary(secret * 10000))

    def test_artifact_type_uses_operation_not_error_text(self):
        self.assertEqual(artifact_type_for("general_generate_code", "filesystem.atomic_replace"), "CODE")
        self.assertEqual(artifact_type_for("document_conversion", "document.create_pdf"), "CONVERSION")
        self.assertEqual(artifact_type_for("", "archive.compress_zip"), "ARCHIVE")


class FailureRecordIntegration(unittest.TestCase):
    setUp = PhaseC.setUp
    tearDown = PhaseC.tearDown
    controller = PhaseC.controller

    def prepared_action(self):
        return PlannedToolAction("general_generate_code",
            ("developer.write_code", "developer.run_python"), "general_generate_code",
            arguments={"path": str(self.workspace / "sum.py"),
                "code_generation_spec": {"language": "python", "target_filename": "sum.py",
                    "goal": "sum two numbers", "execution_required": True,
                    "tests": [{"test_id": "test_003", "stdin": "2\n4\n", "expected_stdout_contains": ["6"]}]},
                "generated_code_artifact": {"language": "python", "filename": "sum.py",
                    "code": "a = int(input())\nb = int(input())\nprint(a + b)\n", "tests": []}},
            source_text="Create and test sum.py in the authorized workspace")

    def test_real_b2_propagation_explanation_retry_and_history(self):
        c = self.controller(R(K.CONTENT_RESPONSE, task_control=T.EXPLAIN_FAILURE, confidence=1),
            R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1),
            R(K.TOOL_ACTION, task_control=T.CONFIRM, confidence=1))
        action = self.prepared_action()
        c._remember_task_plan(action)
        prepared = c._resolve_and_prepare(action, turn_id="start")
        self.assertIsNotNone(c.state.pending_tool_action, prepared.text)
        from anydes_tools.conversation.code_generation import verify_code_test
        with patch("anydes_tools.conversation.controller.verify_code_test",
                   side_effect=lambda stdout, test: False if verify_code_test(stdout, test) else False):
            first = c._handle_pending("confirm", "first", control=T.CONFIRM)
        self.assertFalse(first.executed)
        self.assertFalse((self.workspace / "sum.py").exists())
        failure = c.state.last_tool_failure
        self.assertEqual(failure.artifact_type, "CODE")
        self.assertEqual(failure.error_code, "TEST_ASSERTION_FAILED")
        self.assertEqual(failure.technical_message, "code_test_failed:test_003")
        self.assertEqual(failure.attempt_number, 1)
        self.assertTrue(failure.step_id)
        self.assertEqual(failure.plan_revision, 1)
        self.assertEqual(failure.original_plan_ref, plan_reference(c.state.pending_task.failed_plan))
        evidence = c.state.verification_evidence[failure.verification_ref]
        self.assertEqual(evidence.failed_check, "test_003")
        self.assertEqual(evidence.exit_code, 0)
        step = next(s for s in c.last_step_results if s.error == failure.technical_message)
        self.assertEqual(step.verification["verification_evidence"]["verification_id"], failure.verification_ref)
        with patch.object(c.executor, "execute", side_effect=AssertionError("explanation executed tool")):
            explanation = c.handle("Explain the previous failure", turn_id="why")
        self.assertFalse(c._executor_called)
        self.assertIn("test_003", explanation.text)
        self.assertNotIn("documento", explanation.text)
        c.handle("Retry the failed task", turn_id="retry")
        self.assertEqual(c.state.pending_task.attempt_number, 2)
        self.assertEqual(c.state.pending_task.task_id, failure.task_id)
        self.assertEqual(c.state.retry_count, 1)
        final = c.handle("Confirm execution", turn_id="confirm")
        self.assertTrue(final.executed, final.debug)
        self.assertTrue((self.workspace / "sum.py").is_file())
        self.assertIsNone(c.state.last_tool_failure)
        self.assertEqual(c.state.failure_history, [failure])
        self.assertEqual([(a.attempt_number, a.status) for a in c.state.attempt_history], [(1, "FAILED"), (2, "SUCCESS")])
        self.assertEqual(c.state.attempt_history[-1].failure_id, failure.failure_id)
        self.assertEqual(c.state.attempt_history[-1].original_plan_ref, failure.original_plan_ref)

    def test_plan_snapshot_not_last_plan_and_mutation_detected(self):
        c = self.controller()
        action = self.prepared_action()
        c._remember_task_plan(action)
        c._record_failure(action, turn_id="f", stage="VERIFY", error_code="TEST_FAILURE",
            safe_user_message="test fixture", retryable=True)
        failure = c.state.last_tool_failure
        snapshot = c.state.failure_plans[failure.original_plan_ref]
        c.state.pending_task.plan = None
        self.assertEqual(snapshot.task_id, failure.task_id)
        self.assertEqual(plan_reference(snapshot), failure.original_plan_ref)
        snapshot.legacy_action.arguments["path"] = "changed"
        response = c._handle_retry("retry", "r", force=True)
        self.assertFalse(response.executed)
        self.assertEqual(c.state.retry_count, 0)

    def test_nonzero_execution_keeps_verification_link(self):
        c = self.controller()
        action = self.prepared_action()
        c._remember_task_plan(action)
        run = ToolExecutionResult("execution_failed", "t", "official.developer", "developer.run_python",
            ExecutionStatus.SUCCESS, output={"returncode": 1, "stderr": "bounded fixture error"})
        evidence = VerificationEvidence.code_test(CodeTestCase("test_error"), run, False)
        c.state.verification_evidence[evidence.verification_id] = evidence
        c._record_failure(action, turn_id="t", stage="EXECUTION", error_code="TOOL_EXECUTION_FAILED",
            safe_user_message="execution failed", retryable=True, failed_result=run)
        failure = c.state.last_tool_failure
        self.assertEqual(failure.error_code, "CODE_EXECUTION_FAILED")
        self.assertEqual(failure.verification_ref, evidence.verification_id)
        self.assertEqual(failure.technical_message, "CODE_EXECUTION_FAILED")


del PhaseC

if __name__ == "__main__":
    unittest.main()
