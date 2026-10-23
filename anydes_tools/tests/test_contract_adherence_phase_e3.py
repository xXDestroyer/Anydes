"""UNIT: explicit classifier/adherence doubles; INTEGRATION: real staging executor."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from anydes_tools.conversation.contract_canonicalization import ConfidenceValue, ContractCanonicalizer
from anydes_tools.conversation.interaction_router import InteractionResolution, InteractionResolver, InteractionKind
from anydes_tools.conversation.content_generation import ContentGenerationSpec
from anydes_tools.conversation.task_adherence import evaluate_adherence, aggregate_adherence
from anydes_tools.conversation.verification_types import VerificationResult, VerificationStatus as S
from anydes_tools.contracts import ExecutionStatus
from anydes_tools.tests.test_interaction_execution_phase_c import PhaseC
from anydes_tools.tests.test_word_pdf_phase_d3 import WordPdfIntegration


STATE = {"pending_action": False, "failure_available": False, "retry_available": False}
TOOL = {"kind": "TOOL_ACTION", "output_kind": "CODE_ARTIFACT", "artifact_intent": True,
    "requested_operations": ["GENERATE_CODE", "TEST_CODE"], "requires_execution": True,
    "verification_requirement": "TEST_SUITE", "confidence": "HIGH"}


class CanonicalizationTests(unittest.TestCase):
    def test_numeric(self):
        for value in (0, .65, 1):
            self.assertEqual(ConfidenceValue.canonical(value), float(value))

    def test_labels(self):
        for label, number in ConfidenceValue.LABELS.items():
            self.assertEqual(ConfidenceValue.canonical(label), number)
            self.assertEqual(ConfidenceValue.canonical(label.lower()), number)
        self.assertLess(ConfidenceValue.canonical("HIGH"), .95)

    def test_invalid(self):
        for value in (True, None, "very high probably", "0.9", " HIGH ", -1, 2, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ConfidenceValue.canonical(value)

    def test_normalization_no_repair(self):
        calls = []
        def classifier(request):
            calls.append(request)
            return InteractionResolution.from_dict(TOOL)
        result = InteractionResolver(classifier).resolve("prepared action", state=STATE)
        self.assertEqual(result.kind, InteractionKind.TOOL_ACTION)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("repair_errors", calls[0])
        self.assertEqual(TOOL["confidence"], "HIGH")

    def test_semantic_contradiction_repairs_once(self):
        calls = []
        def classifier(request):
            calls.append(request)
            return InteractionResolution.from_dict({**TOOL, "kind": "CONTENT_RESPONSE"})
        self.assertEqual(InteractionResolver(classifier).resolve("fixture", state=STATE).kind, InteractionKind.CLARIFICATION)
        self.assertEqual(len(calls), 2)
        self.assertIn("non_action_has_effect_or_artifact", calls[1]["repair_errors"])

    def test_genuine_content(self):
        result = InteractionResolution.from_dict({"kind": "CONTENT_RESPONSE", "confidence": "HIGH"})
        self.assertFalse(InteractionResolver.contradictions(result, STATE))


class AdherenceTests(unittest.TestCase):
    def evaluate(self, status, aligned, reason):
        spec = ContentGenerationSpec.from_dict({"topic": "Project X", "source_context": "Project X is a software project"})
        return evaluate_adherence(lambda evidence: {"status": status, "goal_aligned": aligned,
            "subject_aligned": aligned, "reason_code": reason}, goal="describe our project",
            specification=spec, content="A geographic village unrelated to the requested software project.")

    def test_wrong_subject(self):
        a = self.evaluate("FAIL", False, "SUBJECT_MISMATCH")
        result = aggregate_adherence(VerificationResult("DOCUMENT", "DOCUMENT", S.PASS), a)
        self.assertEqual(result.status, S.FAIL)
        self.assertEqual(result.error_code, "TASK_ADHERENCE_FAIL")
        self.assertEqual(result.evidence["task_adherence"]["reason_code"], "SUBJECT_MISMATCH")

    def test_aligned(self):
        a = self.evaluate("PASS", True, "ALIGNED")
        self.assertEqual(aggregate_adherence(VerificationResult("DOCUMENT", "DOCUMENT", S.PASS), a).status, S.PASS)

    def test_inconclusive(self):
        a = self.evaluate("INCONCLUSIVE", False, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(aggregate_adherence(VerificationResult("DOCUMENT", "DOCUMENT", S.PASS), a).status, S.INCONCLUSIVE)

    def test_missing_evaluator(self):
        a = evaluate_adherence(None, goal="subject", specification=ContentGenerationSpec.from_dict({"topic": "subject"}), content="text")
        self.assertEqual(a.status, S.INCONCLUSIVE)

    def test_contradictory_evaluator_fail_safe(self):
        self.assertEqual(self.evaluate("PASS", False, "SUBJECT_MISMATCH").status, S.INCONCLUSIVE)


class ReachabilityIntegration(unittest.TestCase):
    setUp = PhaseC.setUp
    tearDown = PhaseC.tearDown
    controller = PhaseC.controller
    def test_high_reaches_planner(self):
        c = self.controller()
        c.interaction_resolver.classifier = lambda request: InteractionResolution.from_dict(TOOL)
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None) as planner:
            c.handle("prepared normal action", turn_id="high")
        self.assertTrue(planner.called)

    def test_valid_docx_wrong_subject_no_commit(self):
        c = self.controller()
        adherence = AdherenceTests().evaluate("FAIL", False, "SUBJECT_MISMATCH")
        path = self.workspace / "Mismatch.docx"
        results = c._stage_and_commit_document(final_path=str(path), capability="document.create_docx",
            content="A valid document about an unrelated place.", title="Project X", turn_id="mismatch",
            confirmed_effects=(), verify=lambda actual: {"status": "VERIFIED", "task_adherence": adherence})
        self.assertTrue(any(r.status == ExecutionStatus.VERIFY_FAILED for r in results))
        self.assertFalse(path.exists())
        self.assertFalse(list(self.workspace.glob("*.anydes-*")))
        typed = c.state.last_verification["verification_result"]
        self.assertTrue(typed["checks"]["readable"])
        self.assertEqual(typed["error_code"], "TASK_ADHERENCE_FAIL")

    def test_adherence_inconclusive_no_commit(self):
        c = self.controller()
        adherence = AdherenceTests().evaluate("INCONCLUSIVE", False, "INSUFFICIENT_EVIDENCE")
        path = self.workspace / "Uncertain.docx"
        c._stage_and_commit_document(final_path=str(path), capability="document.create_docx",
            content="A physically readable document.", title="", turn_id="uncertain", confirmed_effects=(),
            verify=lambda actual: {"status": "VERIFIED", "task_adherence": adherence})
        self.assertFalse(path.exists())
        self.assertEqual(c.state.last_verification["verification_result"]["status"], "INCONCLUSIVE")


del PhaseC, WordPdfIntegration
