"""Prepared inputs only; real lowering, B2, dispatch, selector, refs, commits and failures."""
from dataclasses import replace
import json
import unittest
from anydes_tools.contracts import ExecutionStatus, PermissionEffect
from anydes_tools.conversation.operation_semantics import OperationSemantics, canonical_descriptor
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer
from anydes_tools.conversation.verification_types import TestSpec, VerificationStatus as S
from anydes_tools.conversation.typed_verification import VerificationDispatcher
from anydes_tools.conversation.oracle_registry import OracleSelector
from anydes_tools.conversation.interaction_router import TaskControl
from anydes_tools.conversation.content_generation import ContentGenerationSpec, ContentValidator, PostActionVerifier
from anydes_tools.tests.test_dataflow_closure_phase_d5 import DataflowClosureIntegration


UNITS = {"semantic_kind": "UNIT_CONVERSION", "input_type": "NUMBER", "output_type": "NUMBER",
    "parameters": {"source_unit": "C", "target_unit": "F"}}
COUNT = {"semantic_kind": "CHARACTER_COUNT", "input_type": "TEXT", "output_type": "NUMBER",
    "parameters": {"alphabet": "aeiou", "casefold": True, "strip_accents": False}}


class SemanticContracts(unittest.TestCase):
    def verify(self, descriptor, stdin, actual):
        test = TestSpec("reference", stdin=stdin, expected_hint=-999)
        test = OperationSemantics.from_dict(descriptor).bind(test)
        return VerificationDispatcher.verify("CODE", test=test, stdout=actual, exit_code=0, oracle_selector=OracleSelector())

    def test_unit_reference_passes(self):
        for value, expected in (("0", 32), ("100", 212), ("-40", -40)):
            result = self.verify(UNITS, value, str(expected))
            self.assertEqual(result.status, S.PASS)
            self.assertEqual(result.evidence["oracle_id"], "units.affine")
            self.assertEqual(result.evidence["oracle_version"], 1)

    def test_wrong_output_fails(self):
        self.assertEqual(self.verify(UNITS, "0", "132").status, S.FAIL)

    def test_membership_rules(self):
        for text, expected in (("Python es genial", 5), ("AEIOU aeiou", 10), ("ÁéÍóÚ", 0)):
            result = self.verify(COUNT, text, str(expected))
            self.assertEqual(result.status, S.PASS)
            self.assertEqual(result.evidence["oracle_id"], "text.character_count")
        accented = {**COUNT, "parameters": {**COUNT["parameters"], "strip_accents": True}}
        self.assertEqual(self.verify(accented, "ÁéÍóÚ", "5").status, S.PASS)

    def test_invalid_input_inconclusive(self):
        self.assertEqual(self.verify(UNITS, "abc", "0").status, S.INCONCLUSIVE)

    def test_descriptor_not_prompt_derived(self):
        with self.assertRaises(ValueError):
            canonical_descriptor({"goal": "convert temperature please"})
        for bad in (False, [], "unit conversion", {**UNITS, "oracle_id": "invented"}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                canonical_descriptor(bad)

    def test_error_behavior_not_functional(self):
        test = TestSpec("error", stdin="abc", invalid_input_behavior="NONZERO_EXIT")
        self.assertIs(OperationSemantics.from_dict(UNITS).bind(test), test)

    def test_missing_descriptor_inconclusive(self):
        result = VerificationDispatcher.verify("CODE", test=TestSpec("missing", stdin="0", expected=32),
            stdout="32", exit_code=0, oracle_selector=OracleSelector(), trusted_oracle=False)
        self.assertEqual(result.status, S.INCONCLUSIVE)

    def test_113_words_rejected(self):
        spec = ContentGenerationSpec.from_dict({"topic": "Project", "target_length_words": 180})
        result = ContentValidator.validate(" ".join(["Project"] * 113), spec)
        self.assertEqual((result.minimum_words, result.maximum_words), (144, 216))
        self.assertEqual(result.status, "INVALID")


class RuntimePropagation(unittest.TestCase):
    setUp = DataflowClosureIntegration.setUp
    tearDown = DataflowClosureIntegration.tearDown

    def execute(self, descriptor, code, inputs):
        c = self.controller
        tests = [{"test_id": "t" + str(i), "stdin": value, "expected_hint": -999}
            for i, value in enumerate(inputs)]
        payload = {"goal": "perform declared deterministic transformation and verify output", "operations": [
            {"operation_id": "write", "operation_kind": "GENERATE_CODE", "inputs": [],
                "outputs": [{"name": "code", "artifact_type": "CODE", "format": "py", "filename": "known.py"}],
                "specification": {"language": "python", "goal": "declared transformation", "tests": tests,
                    "operation_descriptor": descriptor, "execution_required": True, "verification_requirement": "TEST_SUITE"}},
            {"operation_id": "test", "operation_kind": "TEST_CODE", "inputs": ["code"], "outputs": [], "specification": {}}],
            "requested_final_outputs": ["code"]}
        requirements = ExecutablePlanLowerer(c.registry, self.workspace).lower_requirements(SemanticProposal.from_dict(payload))
        self.assertEqual(requirements.inputs["code_spec"]["operation_descriptor"], descriptor)
        action = c._action_from_general_plan(requirements, "prepared deterministic runtime fixture")
        action.arguments["oracle_authority"] = "MODEL_GENERATED"
        action.arguments["generated_code_artifact"] = {"language": "python", "filename": "known.py", "code": code, "tests": []}
        plan = c.plan_adapter.compile(action, source="GENERAL_REASONER")
        self.assertEqual(plan.typed_inputs["operation_descriptor"], descriptor)
        self.assertEqual(next(s.operation_descriptor for s in plan.steps if s.capability_id == "developer.run_python"), descriptor)
        c._plan_source = "GENERAL_REASONER"
        c._remember_task_plan(action)
        c._resolve_and_prepare(action, turn_id="E4-prepare")
        c._handle_pending("fixture confirmation", "E4-run", control=TaskControl.CONFIRM)
        results = c.last_results
        verifications = [e.verification_result for e in c.state.verification_evidence.values()]
        print(json.dumps({"test_type": "INTEGRATION_PREPARED_INPUTS_REAL_EXECUTION", "task_id": plan.task_id,
            "descriptor": descriptor, "results": [v for v in verifications],
            "committed": (self.workspace / "known.py").exists()}, ensure_ascii=True))
        self.assertFalse(list(self.workspace.glob(".anydes-code-*")))
        return results, verifications

    def test_runtime_unit_pass_complete(self):
        results, evidence = self.execute(UNITS, "x = float(input())\nprint(x * 9 / 5 + 32)\n", ["0", "100", "-40"])
        self.assertTrue((self.workspace / "known.py").exists())
        self.assertEqual(len(evidence), 3)
        self.assertTrue(all(v["status"] == "PASS" for v in evidence))
        self.assertEqual(results[-1].status, ExecutionStatus.SUCCESS)

    def test_runtime_count_pass_complete(self):
        results, evidence = self.execute(COUNT,
            "text = input().casefold()\nprint(sum(c in 'aeiou' for c in text))\n", ["Python es genial", "AEIOU", "ÁéÍóÚ"])
        self.assertTrue((self.workspace / "known.py").exists())
        self.assertTrue(all(v["status"] == "PASS" for v in evidence))

    def test_runtime_wrong_code_fail_no_commit(self):
        results, evidence = self.execute(UNITS, "x = input()\nprint(999)\n", ["0"])
        self.assertFalse((self.workspace / "known.py").exists())
        self.assertEqual(evidence[0]["status"], "FAIL")
        self.assertEqual(evidence[0]["evidence"]["oracle_source"], "DETERMINISTIC_REFERENCE")
        failure = self.controller.state.last_tool_failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.artifact_type, "CODE")
        self.assertEqual(failure.verification_ref, evidence[0]["verification_id"])

    def test_runtime_113_words_structural_pass_requirement_fail(self):
        body = " ".join(["Project"] * 113)
        spec = ContentGenerationSpec.from_dict({"topic": "Project", "target_length_words": 180})
        final = self.workspace / "Insufficient.docx"
        results = self.controller._stage_and_commit_document(final_path=str(final),
            capability="document.create_docx", content=body, title="", turn_id="E4-length",
            confirmed_effects=(), require_nonempty=True,
            verify=lambda actual: PostActionVerifier.generated_creation(actual, body, spec, generated_body=body))
        self.assertTrue(any(r.status == ExecutionStatus.VERIFY_FAILED for r in results))
        v = self.controller.state.last_verification
        self.assertEqual(v["word_count"], 113)
        self.assertTrue(v["verification_result"]["checks"]["readable"])
        self.assertTrue(v["verification_result"]["checks"]["nonempty"])
        self.assertFalse(v["verification_result"]["checks"]["semantic_constraints"])
        self.assertEqual(v["verification_result"]["status"], "FAIL")
        self.assertFalse(final.exists())
        self.assertFalse(list(self.workspace.glob("*.anydes-*")))


del DataflowClosureIntegration
