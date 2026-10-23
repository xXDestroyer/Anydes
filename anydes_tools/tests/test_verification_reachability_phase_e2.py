"""UNIT contract fixtures; INTEGRATION uses real controller and B2, prepared cognition."""
from dataclasses import replace
import unittest
from anydes_tools.conversation.interaction_router import InteractionResolution, InteractionResolver, InteractionKind, RequestedOutcome
from anydes_tools.conversation.verification_types import VerificationRequirement as R, TestSpec, VerificationStatus as S
from anydes_tools.conversation.typed_verification import VerificationDispatcher
from anydes_tools.conversation.oracle_registry import OracleSelector
from anydes_tools.tests.test_dataflow_closure_phase_d5 import DataflowClosureIntegration
from anydes_tools.conversation.contracts import PlannedToolAction


STATE={"retry_available":False,"failure_available":False,"pending_action":False}


class ResolutionContracts(unittest.TestCase):
    def test_nonaction_operational_evidence_rejected(self):
        base=InteractionResolution(InteractionKind.CONTENT_RESPONSE,confidence=1)
        for value in (replace(base,requested_effects=("CREATE",)),replace(base,artifact_intent=True),
                replace(base,requires_execution=True),replace(base,requested_operations=("CREATE_ARCHIVE",)),
                replace(base,output_kind=RequestedOutcome.CODE_ARTIFACT)):
            self.assertIn("non_action_has_effect_or_artifact",InteractionResolver.contradictions(value,STATE))

    def test_actual_content_response_accepted(self):
        value=InteractionResolution(InteractionKind.CONTENT_RESPONSE,confidence=1,
            requested_operations=("GENERATE_CONTENT",))
        self.assertFalse(InteractionResolver.contradictions(value,STATE))

    def test_exactly_one_resolution_repair(self):
        calls=[]
        invalid=InteractionResolution(InteractionKind.CONTENT_RESPONSE,requires_execution=True,confidence=1)
        def classifier(request):
            calls.append(request)
            return invalid
        result=InteractionResolver(classifier).resolve("isolated semantic fixture",state=STATE)
        self.assertEqual(result.kind,InteractionKind.CLARIFICATION)
        self.assertEqual(len(calls),2)
        self.assertIn("non_action_has_effect_or_artifact",calls[1]["repair_errors"])

    def test_repair_preserves_tool_action(self):
        invalid=InteractionResolution(InteractionKind.CONTENT_RESPONSE,requested_operations=("TEST_CODE",),confidence=1)
        valid=replace(invalid,kind=InteractionKind.TOOL_ACTION,requires_execution=True,
            verification_requirement=R.TEST_SUITE)
        values=iter((invalid,valid))
        self.assertEqual(InteractionResolver(lambda r:next(values)).resolve("fixture",state=STATE),valid)

    def test_execution_verifier_does_not_select_oracle(self):
        class UnexpectedOracle:
            def resolve(self,test):
                raise AssertionError("non-oracle verification called OracleSelector")
        result=VerificationDispatcher.verify("CODE",execution_only=True,execution_succeeded=True,exit_code=0,
            oracle_selector=UnexpectedOracle())
        self.assertEqual(result.status,S.PASS)
        self.assertEqual(result.evidence["coverage"],"EXECUTION_ONLY")

    def test_functional_test_reaches_registered_selector(self):
        test=TestSpec("typed",stdin="100",expected_hint=-5,
            operation_semantics={"kind":"TEMPERATURE_CONVERSION","source_unit":"C","target_unit":"F"},
            oracle_requirement={"input_type":"NUMBER","output_type":"NUMBER"})
        result=VerificationDispatcher.verify("CODE",test=test,stdout="212",exit_code=0,oracle_selector=OracleSelector())
        self.assertEqual(result.status,S.PASS)
        self.assertEqual(result.evidence["oracle_id"],"units.affine")


class CompletenessIntegration(unittest.TestCase):
    setUp=DataflowClosureIntegration.setUp
    tearDown=DataflowClosureIntegration.tearDown

    def action(self, requirement="BASIC", tests=(), execute=False):
        return PlannedToolAction("generate_code",("developer.write_code","developer.run_python") if execute else ("developer.write_code",),
            "general_generate_code",arguments={"path":str(self.workspace/"example.py"),
                "code_generation_spec":{"language":"python","goal":"emit a value","target_filename":"example.py",
                    "tests":list(tests),"verification_requirement":requirement,"execution_required":execute},
                "generated_code_artifact":{"language":"python","filename":"example.py","code":"print(7)\n","tests":[]}},
            source_text="prepared contract fixture")

    def test_required_suite_missing_rejected(self):
        with self.assertRaisesRegex(ValueError,"PLAN_INCOMPLETE"):
            self.controller.plan_adapter.compile(self.action("TEST_SUITE",execute=True),source="LEGACY_ADAPTER")

    def test_required_verifier_missing_rejected(self):
        action=self.action("TEST_SUITE",tests=[{"test_id":"required","expected":7}],execute=True)
        action=replace(action,capabilities=("developer.write_code",))
        with self.assertRaisesRegex(ValueError,"required_code_verifier_missing"):
            self.controller.plan_adapter.compile(action,source="LEGACY_ADAPTER")

    def test_create_only_does_not_invent_suite(self):
        action=self.action()
        plan=self.controller.plan_adapter.compile(action,source="LEGACY_ADAPTER")
        self.assertFalse(any(s.capability_id.startswith("developer.run") for s in plan.steps))
        self.assertFalse(plan.legacy_action.arguments["code_generation_spec"]["tests"])
        results=self.controller._execute(plan,turn_id="E2-basic",confirmed_effects=())
        self.assertTrue(all(r.status.value=="SUCCESS" for r in results),[r.error for r in results])
        self.assertFalse(any(e.verification_result.get("evidence",{}).get("oracle_id")
            for e in self.controller.state.verification_evidence.values()))


del DataflowClosureIntegration
