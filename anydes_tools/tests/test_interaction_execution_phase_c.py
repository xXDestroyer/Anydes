"""UNIT routing doubles are explicit; INTEGRATION uses real registry/workers/B2.

These tests are not full-runtime/Groq generalization evidence.
"""
from dataclasses import replace
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from anydes_tools.conversation.controller import ToolConversationController
from anydes_tools.conversation.contracts import PlannedToolAction
from anydes_tools.conversation.executable_plan import ExecutableTaskPlan, BrainPlanProposal
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.conversation.interaction_router import (
    InteractionKind as K, InteractionResolution as R, InteractionResolver,
    TaskControl as T, RequestedOutcome as O,
)
from anydes_tools.settings.store import SettingsStore
from anydes_tools.contracts import ToolExecutionResult, ExecutionStatus

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "anydes_tools"


def action(**kw):
    return R(K.TOOL_ACTION, kw.pop("output_kind", O.DOCUMENT_ARTIFACT), artifact_intent=True, confidence=1, **kw)


class ExplicitClassifierDouble:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        return self.results.pop(0)


class PhaseC(unittest.TestCase):
    def setUp(self):
        self.workspace = TOOLS / "workspaces" / f"phase_c_{uuid4().hex[:12]}"
        self.workspace.mkdir(parents=True)
        self.settings = SettingsStore(ROOT, TOOLS, settings_path=self.workspace / "settings.json")
        self.settings.set_temporary_workspace(self.workspace)
        self.controllers = []

    def tearDown(self):
        for c in self.controllers:
            c.close()
        shutil.rmtree(self.workspace)

    def controller(self, *decisions, **kwargs):
        # Prepared cognition tests explicitly supply the new relevance assessment double.
        kwargs.setdefault("task_adherence_evaluator", lambda evidence: {
            "status": "PASS", "goal_aligned": True, "subject_aligned": True, "reason_code": "ALIGNED"})
        c = ToolConversationController(ROOT, TOOLS, settings=self.settings,
            interaction_classifier=ExplicitClassifierDouble(*decisions), **kwargs)
        self.controllers.append(c)
        return c

    def test_unit_social_holdout_never_enters_planner(self):
        texts = ["Buenas, hoy terminé agotado", "Me apetece conversar un rato", "¿Cómo va tu día?"]
        c = self.controller(*(R(K.CONVERSATION, confidence=1) for _ in texts))
        with patch("anydes_tools.conversation.controller.plan_from_text", side_effect=AssertionError("planner called")):
            for i, text in enumerate(texts):
                self.assertIsNone(c.handle(text, turn_id=str(i)))
                self.assertFalse(c._executor_called)
                self.assertFalse(c._reasoner_called)

    def test_unit_content_holdout_has_no_tool_validation(self):
        texts = ["Redacta una reflexión sobre nuestro proyecto", "Explícame un concepto nuevo", "Resume lo que recuerdas de Anydes"]
        c = self.controller(*(R(K.CONTENT_RESPONSE, confidence=1) for _ in texts))
        with patch("anydes_tools.conversation.controller.plan_from_text", side_effect=AssertionError("planner called")):
            for i, text in enumerate(texts):
                self.assertIsNone(c.handle(text, turn_id=str(i)))

    def test_unit_classifier_sees_no_precomputed_plan(self):
        c = self.controller(R(K.CONVERSATION, confidence=1))
        c.handle("Nueva charla", turn_id="one")
        req = c.interaction_resolver.classifier.calls[0]
        self.assertNotIn("deterministic_candidate", req["state"])

    def test_unit_contradiction_repaired_once(self):
        invalid = R(K.CONTENT_RESPONSE, requested_effects=("CREATE_FILE",), confidence=1)
        good = action()
        model = ExplicitClassifierDouble(invalid, good)
        result = InteractionResolver(model).resolve("solicitud", state=self.context())
        self.assertEqual(result, good)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("repair_errors", model.calls[-1])

    def test_unit_persistent_contradiction_clarifies(self):
        bad = R(K.CONTENT_RESPONSE, task_control=T.RETRY, confidence=1)
        model = ExplicitClassifierDouble(bad, bad)
        result = InteractionResolver(model).resolve("solicitud", state=self.context())
        self.assertEqual(result.kind, K.CLARIFICATION)
        self.assertEqual(len(model.calls), 2)

    def test_unit_semantic_control_not_phrase_list(self):
        model = ExplicitClassifierDouble(R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1))
        state = {**self.context(), "retry_available": True}
        result = InteractionResolver(model).resolve("Démosle otra oportunidad al intento anterior", state=state)
        self.assertEqual(result.task_control, T.RETRY)

    def test_unit_delete_requires_high_confidence(self):
        ambiguous = R(K.TOOL_ACTION, task_control=T.CONFIRM, confidence=.9)
        result = InteractionResolver(ExplicitClassifierDouble(ambiguous, ambiguous)).resolve(
            "transcripción dudosa", state={**self.context(), "pending_action": True, "destructive_confirmation": True})
        self.assertEqual(result.kind, K.CLARIFICATION)

    def test_unit_no_classifier_fail_closed(self):
        self.assertEqual(InteractionResolver().resolve("crea", state=self.context()).kind, K.CLARIFICATION)

    def test_unit_invalid_control_without_state(self):
        for control in (T.CONFIRM, T.CANCEL, T.RETRY, T.EXPLAIN_FAILURE):
            bad = R(K.TOOL_ACTION, task_control=control, confidence=1)
            with self.subTest(control=control):
                result = InteractionResolver(ExplicitClassifierDouble(bad, bad)).resolve("mensaje", state=self.context())
                self.assertEqual(result.kind, K.CLARIFICATION)

    def test_integration_create_has_single_plan_and_real_step_results(self):
        c = self.controller(action())
        response = c.handle("crea un txt que diga hola", turn_id="create")
        self.assertTrue(response.executed, response.debug)
        self.assertIsInstance(c.state.pending_task.plan, ExecutableTaskPlan)
        self.assertEqual(c.state.pending_task.plan.plan_source, "FAST_PATH")
        self.assertIsNone(c.state.pending_tool_action)
        self.assertTrue(c.last_step_results)
        self.assertTrue(any(p.suffix == ".txt" for p in self.workspace.iterdir()))

    def test_unit_executor_rejects_old_contract(self):
        c = self.controller()
        with self.assertRaises(ValueError):
            c._execute(PlannedToolAction("create"), turn_id="old", confirmed_effects=())

    def test_integration_pending_cancel_consumed_once(self):
        target = self.workspace / "keep.txt"
        target.write_text("keep", encoding="utf-8")
        c = self.controller(action(output_kind=O.FILESYSTEM_EFFECT), R(K.TOOL_ACTION, task_control=T.CANCEL, confidence=1))
        c.handle("borra keep.txt", turn_id="delete")
        self.assertIsNotNone(c.state.pending_tool_action)
        task = c.state.pending_task.task_id
        c.handle("Prefiero dejarlo como estaba", turn_id="cancel")
        self.assertTrue(target.exists())
        self.assertIsNone(c.state.pending_tool_action)
        self.assertEqual(c.state.pending_task.task_id, task)
        self.assertEqual(c.state.pending_task.status, "CANCELLED")

    def test_integration_pending_confirm_keeps_task_id(self):
        target = self.workspace / "gone.txt"
        target.write_text("gone", encoding="utf-8")
        c = self.controller(action(output_kind=O.FILESYSTEM_EFFECT), R(K.TOOL_ACTION, task_control=T.CONFIRM, confidence=1))
        c.handle("borra gone.txt", turn_id="delete")
        before = c.state.pending_task.task_id
        self.assertEqual(c.state.pending_tool_action.task_id, before)
        response = c.handle("Autorizo esa eliminación", turn_id="confirm")
        self.assertTrue(response.executed, response.debug)
        self.assertFalse(target.exists())
        self.assertEqual(c.state.pending_task.task_id, before)

    def test_integration_disabled_tool_between_plan_and_confirm_denied(self):
        target = self.workspace / "safe.txt"
        target.write_text("safe", encoding="utf-8")
        c = self.controller(action(output_kind=O.FILESYSTEM_EFFECT), R(K.TOOL_ACTION, task_control=T.CONFIRM, confidence=1))
        c.handle("borra safe.txt", turn_id="delete")
        self.assertIsNotNone(c.state.pending_tool_action)
        c.registry.disable("official.filesystem")
        response = c.handle("Autorizo eliminarlo", turn_id="confirm")
        self.assertFalse(response.executed)
        self.assertTrue(target.exists())

    def test_unit_legacy_accessors_share_one_owner(self):
        c = self.controller()
        c.state.retry_count = 1
        self.assertEqual(c.state.pending_task.retry_count, 1)
        c.state.pending_task.retry_count = 0
        self.assertEqual(c.state.retry_count, 0)

    def test_unit_legacy_plan_preserves_complete_artifact_declarations(self):
        c = self.controller()
        legacy = PlannedToolAction("composition", ("filesystem.read", "archive.compress_zip"),
            operation="general_tool_composition", arguments={"general_task_plan": {"steps": [
                {"step_id": "read", "capability": "filesystem.read", "produces_artifacts": ["A"]},
                {"step_id": "zip", "capability": "archive.compress_zip", "consumes_artifacts": ["A"],
                 "produces_artifacts": ["ZIP"], "arguments": {"destination": str(self.workspace / "fixture.zip")}},
            ], "desired_outputs": [{"binding": "ZIP", "artifact_type": "ARCHIVE", "format": "zip"}]}})
        compiled = c.plan_adapter.compile(legacy, source="LEGACY_ADAPTER")
        self.assertEqual(compiled.steps[0].produces_artifacts, ("A",))
        self.assertEqual(compiled.steps[1].consumes_artifacts, ("A",))
        self.assertEqual(compiled.steps[1].risk_class, c.registry.find_by_capability("archive.compress_zip")[0].manifest.capability_contracts["archive.compress_zip"]["risk_level"])

    def test_unit_step_budget(self):
        c = self.controller()
        legacy = PlannedToolAction("composition", ("filesystem.read",), arguments={"general_task_plan": {
            "steps": [{"step_id": str(i), "capability": "filesystem.read"} for i in range(9)]}})
        with self.assertRaises(ValueError):
            c.plan_adapter.compile(legacy, source="GENERAL_REASONER")

    def test_unit_failure_explain_and_retry_bind_same_plan(self):
        c = self.controller(action(),
            R(K.CONTENT_RESPONSE, task_control=T.EXPLAIN_FAILURE, confidence=1),
            R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1),
            R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1))
        failure = ToolExecutionResult("forced", "first", "official.document",
            "document.create_txt", ExecutionStatus.FAILED, error="fixture_generation_failure")
        with patch.object(c, "_execute_legacy_operation", return_value=(failure,)) as execution:
            c.handle("crea un txt que diga hola", turn_id="first")
            original = c.state.pending_task.failed_plan
            self.assertIsNotNone(original)
            with patch("anydes_tools.conversation.controller.plan_from_text", side_effect=AssertionError("replanned")):
                explanation = c.handle("Explícame la causa registrada", turn_id="why")
                self.assertEqual(explanation.debug["error_code"], c.state.last_tool_failure.error_code)
                c.handle("Dale otra oportunidad a la tarea", turn_id="retry")
                self.assertEqual(c.state.pending_task.plan.task_id, original.task_id)
                c.handle("Otra oportunidad", turn_id="bounded")
            self.assertEqual(execution.call_count, 2)
            self.assertEqual(c.state.retry_count, 1)

    def test_unit_retry_does_not_regrant_execution_permission(self):
        c = self.controller(R(K.TOOL_ACTION, O.CODE_ARTIFACT, artifact_intent=True, confidence=1),
            R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1))
        legacy = PlannedToolAction("run_python_file", ("developer.run_python",),
            arguments={"path": str(self.workspace / "fixture.py")},
            target_path=str(self.workspace / "fixture.py"), source_text="run")
        (self.workspace / "fixture.py").write_text("print(1)", encoding="utf-8")
        c._remember_task_plan(legacy)
        c._record_failure(legacy, turn_id="failure", stage="EXECUTION", error_code="FIXTURE_FAILURE",
            safe_user_message="fallo de fixture", retryable=True)
        c.interaction_resolver.classifier = ExplicitClassifierDouble(
            R(K.TOOL_ACTION, task_control=T.RETRY, confidence=1))
        response = c.handle("Repite el intento anterior", turn_id="retry")
        self.assertFalse(response.executed)
        self.assertIsNotNone(c.state.pending_tool_action)
        self.assertEqual(c.state.pending_task.task_id, c.state.pending_task.failed_plan.task_id)

    def test_unit_plan_detects_mutated_arguments(self):
        c = self.controller()
        legacy = PlannedToolAction("read", ("filesystem.read",),
            arguments={"path": str(self.workspace / "a.txt")})
        plan = c.plan_adapter.compile(legacy, source="FAST_PATH")
        plan.steps[0].args["path"] = str(self.workspace / "b.txt")
        with self.assertRaises(ValueError):
            c.plan_adapter.revalidate(plan)

    def test_unit_primitive_schema_validated_before_execution(self):
        c = self.controller()
        bad = PlannedToolAction("read", ("filesystem.read",), arguments={"path": 123})
        with self.assertRaises(ValueError):
            c.plan_adapter.compile(bad, source="FAST_PATH")

    def test_unit_capability_is_locally_enabled_not_model_granted(self):
        c = self.controller()
        c.registry.disable_capability("filesystem.read")
        bad = PlannedToolAction("read", ("filesystem.read",), arguments={"path": "a.txt"})
        with self.assertRaises(ValueError):
            c.plan_adapter.compile(bad, source="GENERAL_REASONER")

    def test_unit_unknown_operation_denied(self):
        c = self.controller()
        bad = PlannedToolAction("unknown", ("filesystem.read",), operation="brain_granted_operation")
        with self.assertRaises(ValueError):
            c.plan_adapter.compile(bad, source="GENERAL_REASONER")

    def test_integration_general_proposal_compiles_to_one_executable_plan(self):
        source = self.workspace / "source.txt"
        target = self.workspace / "result.txt"
        # D requires a typed producer, not a read result masquerading as a file.
        proposal = GeneralTaskPlan.from_dict({
            "task_id": "C-proposal", "goal": "read and create", "task_type": "TOOL_COMPOSITION",
            "inputs": {}, "desired_outputs": [{"binding": "B", "artifact_type": "DOCUMENT", "format": "txt"}], "constraints": [],
            "required_capabilities": ["document.create_txt"],
            "steps": [
                {"step_id": "read", "capability": "document.create_txt", "purpose": "produce fixture",
                 "arguments": {"path": str(source), "content": "contenido de fixture"}, "produces_artifacts": ["A"]},
                {"step_id": "write", "capability": "document.create_txt", "purpose": "write fixture",
                 "arguments": {"path": str(target)}, "produces_artifacts": ["B"],
                 "depends_on": ["read"], "consumes_artifacts": ["A"]},
            ],
            "artifact_requirements": {}, "needs_content_generation": False,
            "needs_reasoning": True, "needs_tools": True, "risk": "low",
            "ambiguities": [], "success_criteria": ["fixture copied"],
        })
        c = self.controller(action(output_kind=O.TOOL_RESULT),
            general_task_reasoner=lambda request: BrainPlanProposal(proposal))
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None), \
                patch.object(c, "_execute", wraps=c._execute) as execution:
            response = c.handle("objetivo holdout de fixture", turn_id="proposal")
        self.assertTrue(response.executed, response.debug)
        self.assertEqual(execution.call_count, 1)
        self.assertIsInstance(execution.call_args.args[0], ExecutableTaskPlan)
        self.assertEqual(c.state.pending_task.plan.plan_source, "GENERAL_REASONER")
        self.assertEqual(c.state.pending_task.plan.steps[1].consumes_artifacts, ("A",))
        self.assertEqual(target.read_text(encoding="utf-8"), "contenido de fixture")
        self.assertEqual(len(c.last_step_results), 2)

    def test_unit_capability_query_uses_registry_without_planning(self):
        c = self.controller(R(K.CONTENT_RESPONSE, capability_query=True, confidence=1))
        with patch("anydes_tools.conversation.controller.plan_from_text", side_effect=AssertionError("planned")):
            response = c.handle("¿Qué herramientas tienes?", turn_id="capability")
        self.assertIsNotNone(response)
        self.assertFalse(response.executed)
        self.assertFalse(c._executor_called)

    def test_unit_phase_a_observer_knows_new_boundary(self):
        from anydes_tools.tests.runtime_comparison import checkpoints
        from anydes_tools.tests.runtime_observation import safe
        self.assertIn("INTERACTION_RESOLUTION", checkpoints(
            "anydes_tools.conversation.interaction_router", "resolve", "InteractionResolver", {}))
        redacted = safe("private source", "source_request")
        self.assertIsInstance(redacted, dict)
        self.assertNotIn("private source", str(redacted))

    def test_unit_reasoner_failure_does_not_link_stale_failed_plan(self):
        def unavailable(request):
            raise ValueError("fixture invalid provider JSON")
        c = self.controller(action(output_kind=O.CODE_ARTIFACT), general_task_reasoner=unavailable)
        old = PlannedToolAction("read", ("filesystem.read",), arguments={"path": str(self.workspace / "old.txt")})
        c._remember_task_plan(old)
        c._record_failure(old, turn_id="old", stage="EXECUTION", error_code="OLD_FAILURE",
            safe_user_message="old fixture failure", retryable=True)
        old_id = c.state.pending_task.task_id
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            c.handle("un nuevo objetivo de código", turn_id="new")
        self.assertNotEqual(c.state.pending_task.task_id, old_id)
        self.assertIsNone(c.state.pending_task.failed_plan)
        self.assertIsNone(c.state.last_failed_action)
        self.assertEqual(c.state.last_tool_failure.error_code, "GENERAL_REASONER_FAILED:ValueError")

    def test_integration_save_content_uses_prior_response_context(self):
        captured = []
        proposal = GeneralTaskPlan.from_dict({
            "task_id": "save-content", "goal": "save previous explanation", "task_type": "GENERATE_DOCUMENT",
            "inputs": {"content_spec": {"content_kind": "expository_text", "topic": "continuidad",
                "target_length_words": 100, "source_mode": "CONVERSATION_CONTEXT"}},
            "desired_outputs": [{"binding": "word", "artifact_type": "DOCUMENT", "format": "docx"}], "constraints": [],
            "required_capabilities": ["document.create_docx"],
            "steps": [{"step_id": "write", "capability": "document.create_docx", "purpose": "save",
                "produces_artifacts": ["word"], "consumes_values": ["content.main"]}],
            "artifact_requirements": {"format": "docx", "filename": "Continuidad.docx"},
            "needs_content_generation": True, "needs_reasoning": True, "needs_tools": True,
            "risk": "low", "ambiguities": [], "success_criteria": ["saved previous response"],
        })
        def generator(spec):
            captured.append(spec.source_context)
            return " ".join(["continuidad contexto proyecto permitido"] * 25)
        c = self.controller(action(references=("previous_content",), needs_content_generation=True),
            general_task_reasoner=lambda request: BrainPlanProposal(proposal),
            semantic_prose_generator=generator)
        c.recent_content_response = "La explicación anterior trata sobre continuidad del proyecto."
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            response = c.handle("guarda eso en Word", turn_id="save")
        self.assertTrue(response.executed, response.debug)
        self.assertTrue((self.workspace / "Continuidad.docx").is_file())
        self.assertIn(c.recent_content_response, captured[0])
        self.assertIn("DIRECT_CONVERSATION", captured[0])

    @staticmethod
    def context():
        return {"pending_action": False, "retry_available": False, "failure_available": False}


if __name__ == "__main__":
    unittest.main()
