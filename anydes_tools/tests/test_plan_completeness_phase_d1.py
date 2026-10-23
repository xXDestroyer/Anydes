"""UNIT: prepared proposal holdouts, not LLM intent/generalization evidence.

INTEGRATION reuses real local D transactions with explicitly prepared cognition.
"""
from dataclasses import replace
import unittest

from anydes_tools.conversation.contracts import PlannedToolAction
from anydes_tools.conversation.executable_plan import ExecutableStep, ExecutableTaskPlan
from anydes_tools.conversation.plan_completeness import PlanCompletenessValidator, PlanIncomplete
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration
from anydes_tools.tests.test_general_task_reasoner_v001_5 import QueueReasoner
from anydes_tools.tests.test_interaction_execution_phase_c import action
from unittest.mock import patch


class CompletenessUnit(unittest.TestCase):
    def plan(self, steps, outputs, **kw):
        return ExecutableTaskPlan("C.1", "holdout", 1, "prepared semantic goal",
            {"brain_proposal": True}, tuple(steps), tuple(outputs), (), {"max_steps": 8},
            "prepared holdout request", "GENERAL_REASONER",
            PlannedToolAction("composition", tuple(s.capability_id for s in steps if s.capability_id),
                "general_tool_composition"), requested_final_outputs=tuple(kw.get("required", outputs)))

    def writer(self, name="a", cap="document.create_txt", **kw):
        return ExecutableStep("write_" + name, "CAPABILITY", cap,
            {"path": "test.py" if cap == "developer.write_code" else "test.txt", "content": "body"},
            produces_artifacts=(name,), **kw)

    def archive(self, members=("a",)):
        return ExecutableStep("zip", "CAPABILITY", "archive.compress_zip", {"destination": "test.zip"},
            consumes_artifacts=members, produces_artifacts=("z",), requires_verified_inputs=True)

    def test_holdout_script_and_archive(self):
        p = self.plan([self.writer(cap="developer.write_code"), self.archive()], [{"binding": "z", "format": "zip"}])
        bindings = PlanCompletenessValidator().validate(p)
        self.assertEqual(bindings[0].consumer_step_ids, ("zip",))
        self.assertEqual(bindings[0].artifact_type, "CODE")
        self.assertEqual(bindings[0].required_state, "VERIFIED")

    def test_holdout_word_and_conversion(self):
        convert = ExecutableStep("convert", "CONVERSION", "document.create_pdf", {"path": "test.pdf"},
            consumes_artifacts=("a",), produces_artifacts=("pdf",), requires_verified_inputs=True)
        p = self.plan([self.writer(cap="document.create_docx"), convert], [{"binding": "pdf", "format": "pdf"}])
        self.assertEqual(PlanCompletenessValidator().validate(p)[0].consumer_step_ids, ("convert",))

    def test_holdout_two_outputs_only_one_archived(self):
        p = self.plan([self.writer("a"), self.writer("b"), self.archive(("b",))], [{"binding": "z"}])
        bindings = PlanCompletenessValidator().validate(p)
        self.assertEqual(bindings[0].consumer_step_ids, ())
        self.assertEqual(bindings[1].consumer_step_ids, ("zip",))

    def test_holdout_edited_version_archived(self):
        p = self.plan([self.writer("edited", "document.edit_document"), self.archive(("edited",))], [{"binding": "z"}])
        self.assertEqual(PlanCompletenessValidator().validate(p)[0].logical_name, "edited")

    def test_holdout_code_test_dependency(self):
        run = ExecutableStep("test", "CAPABILITY", "developer.run_python", {}, consumes_artifacts=("a",))
        arc = replace(self.archive(), depends_on=("test",))
        p = self.plan([self.writer(cap="developer.write_code"), run, arc], [{"binding": "z"}])
        self.assertEqual(PlanCompletenessValidator().validate(p)[0].consumer_step_ids, ("test", "zip"))

    def test_holdout_shared_content_two_outputs(self):
        generation = ExecutableStep("generate", "CONTENT_GENERATION", "", {}, produces_values=("body",))
        word = replace(self.writer("word", "document.create_docx"), consumes_values=("body",))
        text = replace(self.writer("text"), consumes_values=("body",))
        p = self.plan([generation, word, text], [{"binding": "word"}, {"binding": "text"}])
        self.assertEqual(len(PlanCompletenessValidator().validate(p)), 2)

    def test_empty_archive_denied(self):
        p = self.plan([self.archive(())], [{"binding": "z"}])
        with self.assertRaisesRegex(PlanIncomplete, "archive_members_binding_required"):
            PlanCompletenessValidator().validate(p)

    def test_missing_final_output_denied(self):
        p = self.plan([self.writer()], [{"binding": "a"}], required=[{"format": "pdf"}])
        with self.assertRaisesRegex(PlanIncomplete, "requested_final_output_missing"):
            PlanCompletenessValidator().validate(p)

    def test_missing_producer_output_denied(self):
        p = self.plan([replace(self.writer(), produces_artifacts=())], [])
        with self.assertRaisesRegex(PlanIncomplete, "producer_output_required"):
            PlanCompletenessValidator().validate(p)

    def test_conversion_without_source_denied(self):
        step = replace(self.writer("pdf", "document.create_pdf"), step_type="CONVERSION")
        with self.assertRaisesRegex(PlanIncomplete, "conversion_source_binding_required"):
            PlanCompletenessValidator().validate(self.plan([step], [{"binding": "pdf"}]))

    def test_unknown_or_future_consumer_denied(self):
        p = self.plan([self.archive(), self.writer()], [{"binding": "z"}])
        with self.assertRaisesRegex(PlanIncomplete, "DATAFLOW_MISSING_PRODUCER"):
            PlanCompletenessValidator().validate(p)

    def test_conversion_code_type_denied(self):
        pdf = replace(self.writer("pdf", "document.create_pdf"), consumes_artifacts=("a",))
        p = self.plan([self.writer(cap="developer.write_code"), pdf], [{"binding": "pdf"}])
        with self.assertRaisesRegex(PlanIncomplete, "conversion_type_mismatch"):
            PlanCompletenessValidator().validate(p)

    def test_code_test_without_binding_denied(self):
        run = ExecutableStep("test", "CAPABILITY", "developer.run_python", {"path": "test.py"})
        p = self.plan([self.writer(cap="developer.write_code"), run], [{"binding": "a"}])
        with self.assertRaisesRegex(PlanIncomplete, "code_execution_binding_required"):
            PlanCompletenessValidator().validate(p)

    def test_parse_preserves_values_and_deliverables(self):
        p = GeneralTaskPlan.from_dict({"steps": [{"step_id": "a", "produces_values": ["body"],
            "consumes_values": ["previous"]}], "requested_final_outputs": [{"binding": "pdf"}]})
        self.assertEqual(p.steps[0].consumes_values, ("previous",))
        self.assertEqual(p.requested_final_outputs[0]["binding"], "pdf")


class CompletenessIntegration(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller
    prepared_action = DataflowIntegration.prepared_action
    code_action = DataflowIntegration.code_action
    execute = DataflowIntegration.execute

    def test_rejected_before_executor(self):
        c = self.controller()
        bad = self.code_action()
        raw = {**bad.arguments["general_task_plan"], "desired_outputs": []}
        with self.assertRaisesRegex(PlanIncomplete, "final_output_declaration_required"):
            c.plan_adapter.compile(replace(bad, arguments={**bad.arguments, "general_task_plan": raw}), source="GENERAL_REASONER")
        self.assertFalse(getattr(c, "_executor_called", False))
        self.assertFalse(list(self.workspace.glob("*.zip")))

    def test_validated_bindings_cannot_change(self):
        c = self.controller()
        plan = c.plan_adapter.compile(self.code_action(), source="GENERAL_REASONER")
        changed = replace(plan.steps[-1], consumes_artifacts=("missing",))
        with self.assertRaises(ValueError):
            c.plan_adapter.revalidate(replace(plan, steps=(*plan.steps[:-1], changed)))

    def proposal(self, complete):
        return GeneralTaskPlan.from_dict({
            "task_id": "repair-fixture", "goal": "create the requested text artifact",
            "task_type": "TOOL_COMPOSITION", "needs_tools": True,
            "required_capabilities": ["document.create_txt"],
            "steps": [{"step_id": "write", "capability": "document.create_txt", "purpose": "write text",
                "arguments": {"path": str(self.workspace / "repaired.txt"), "content": "Verified fixture content"},
                "produces_artifacts": ["text"] if complete else []}],
            "desired_outputs": [{"binding": "text", "format": "txt"}] if complete else [],
            "requested_final_outputs": [{"binding": "text", "format": "txt"}],
        })

    def test_one_repair_complete_plan_then_execution(self):
        reasoner = QueueReasoner(self.proposal(False), self.proposal(True))
        c = self.controller(action(), general_task_reasoner=reasoner)
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            response = c.handle("holdout request requiring a text artifact", turn_id="repair")
        self.assertTrue(response.executed, response.debug)
        self.assertEqual(reasoner.repairs, 1)
        self.assertIn("required_artifact_bindings", reasoner.requests[0])
        self.assertTrue((self.workspace / "repaired.txt").is_file())

    def test_persistent_incomplete_plan_repair_bounded_no_execution(self):
        reasoner = QueueReasoner(self.proposal(False), self.proposal(False))
        c = self.controller(action(), general_task_reasoner=reasoner)
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            response = c.handle("holdout request requiring a text artifact", turn_id="repair")
        self.assertFalse(response.executed)
        self.assertEqual(reasoner.repairs, 1)
        self.assertFalse(getattr(c, "_executor_called", False))
        self.assertFalse((self.workspace / "repaired.txt").exists())


del DataflowIntegration
