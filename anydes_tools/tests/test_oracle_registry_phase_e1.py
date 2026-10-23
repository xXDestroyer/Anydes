"""Local contract tests and real B2 execution. No Groq or simulated executors."""
from dataclasses import replace
import unittest
from anydes_tools.conversation.oracle_registry import OracleRegistry, OracleSelector, DEFAULT_ORACLES
from anydes_tools.conversation.verification_types import TestSpec, VerificationStatus as S
from anydes_tools.conversation.typed_verification import VerificationDispatcher
from anydes_tools.tests.test_dataflow_closure_phase_d5 import DataflowClosureIntegration


def numerical_test(value="0"):
    return TestSpec.from_dict({"test_id":"reference", "input":value, "expected_hint":-999,
        "operation_semantics":{"kind":"TEMPERATURE_CONVERSION","source_unit":"C","target_unit":"F"},
        "oracle_requirement":{"input_type":"NUMBER","output_type":"NUMBER"}, "comparison_mode":"NUMERIC"})


class OracleContracts(unittest.TestCase):
    def verify(self, test, output, selector=None):
        return VerificationDispatcher.verify("CODE",test=test,stdout=output,exit_code=0,
            oracle_selector=selector or OracleSelector())

    def test_affine_units_reference(self):
        for value,expected in (("0",32),("100",212),("-40",-40)):
            with self.subTest(value=value):
                result=self.verify(numerical_test(value),str(expected))
                self.assertEqual(result.status,S.PASS)
                self.assertEqual(result.evidence["oracle_source"],"DETERMINISTIC_REFERENCE")
                self.assertEqual(result.evidence["oracle_id"],"units.affine")
                self.assertEqual(result.evidence["oracle_version"],1)

    def test_invalid_input_is_not_zero(self):
        result=self.verify(numerical_test("abc"),"0")
        self.assertEqual(result.status,S.INCONCLUSIVE)
        self.assertEqual(result.error_code,"INPUT_OR_SEMANTICS_UNVERIFIED")

    def test_character_count_normalization(self):
        test=TestSpec("count",stdin="Python es genial",expected=6,
            operation_semantics={"kind":"COUNT_VOWELS","casefold":True,"include_accented":False},
            oracle_requirement={"input_type":"TEXT","output_type":"NUMBER"})
        self.assertEqual(self.verify(test,"5").status,S.PASS)
        self.assertEqual(self.verify(test,"6").status,S.FAIL)
        self.assertEqual(self.verify(replace(test,stdin="ÁRBOL y pingüino",
            operation_semantics={"kind":"COUNT_VOWELS","casefold":True,"include_accented":True}),"6").status,S.PASS)
        self.assertEqual(self.verify(replace(test,stdin="ÁRBOL y pingüino"),"4").status,S.PASS)

    def test_unknown_semantics(self):
        self.assertEqual(self.verify(replace(numerical_test(),operation_semantics={"kind":"UNREGISTERED"}),"32").status,S.INCONCLUSIVE)

    def test_ambiguous_registry(self):
        definition=DEFAULT_ORACLES[0]
        selector=OracleSelector(OracleRegistry([definition,definition]))
        self.assertEqual(self.verify(numerical_test(),"32",selector).error_code,"ORACLE_AMBIGUOUS")

    def test_disabled_and_type_mismatch(self):
        selector=OracleSelector(OracleRegistry([{**DEFAULT_ORACLES[0],"enabled":False}]))
        self.assertEqual(self.verify(numerical_test(),"32",selector).status,S.INCONCLUSIVE)
        self.assertEqual(self.verify(replace(numerical_test(),oracle_requirement={"input_type":"JSON","output_type":"NUMBER"}),"32").status,S.INCONCLUSIVE)

    def test_version_supersession(self):
        selector=OracleSelector(OracleRegistry([DEFAULT_ORACLES[0],{**DEFAULT_ORACLES[0],"version":2}]))
        self.assertEqual(self.verify(numerical_test(),"32",selector).evidence["oracle_version"],2)

    def test_no_substring_numeric(self):
        self.assertEqual(self.verify(numerical_test(),"132").status,S.FAIL)
        self.assertEqual(self.verify(numerical_test(),"32 or 132").status,S.FAIL)

    def test_tolerance(self):
        test=replace(numerical_test(),comparison_mode="FLOAT_TOLERANCE",tolerance=.01)
        result=self.verify(test,"32.001")
        self.assertEqual(result.status,S.PASS)
        self.assertEqual(result.evidence["tolerance"],.01)

    def test_schema_rejects_oracle_authority(self):
        with self.assertRaises(ValueError):
            TestSpec.from_dict({"test_id":"attack","oracle_id":"units.affine"})


class RuntimeOracleIntegration(unittest.TestCase):
    setUp=DataflowClosureIntegration.setUp
    tearDown=DataflowClosureIntegration.tearDown

    def execute(self, code):
        from anydes_tools.conversation.contracts import PlannedToolAction
        from anydes_tools.conversation.interaction_router import TaskControl
        action=PlannedToolAction("generate_code",("developer.write_code","developer.run_python"),"general_generate_code",
            arguments={"path":str(self.workspace/"reference.py"),"oracle_authority":"MODEL_GENERATED",
                "code_generation_spec":{"language":"python","target_filename":"reference.py","goal":"unit conversion",
                    "tests":[numerical_test(). __dict__],"execution_required":True},
                "generated_code_artifact":{"language":"python","filename":"reference.py","code":code,"tests":[]}},
            source_text="local test of runtime reference selection")
        c=self.controller
        self.assertIsNone(c.code_oracle_provider)
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id="E1-prepare")
        return c._handle_pending("confirm","E1-run",control=TaskControl.CONFIRM)

    def test_runtime_selects_without_test_only_provider(self):
        response=self.execute("x = float(input())\nprint(x * 9 / 5 + 32)\n")
        self.assertTrue(response.executed)
        self.assertTrue((self.workspace/"reference.py").is_file())
        results=[e.verification_result for e in self.controller.state.verification_evidence.values()]
        self.assertEqual(results[0]["evidence"]["oracle_id"],"units.affine")

    def test_failure_record_retains_registry_evidence(self):
        self.execute("input()\nprint(999)\n")
        c=self.controller
        failure=c.state.last_tool_failure
        self.assertEqual(failure.error_code,"TEST_ASSERTION_FAILED")
        evidence=c.state.verification_evidence[failure.verification_ref].verification_result
        self.assertEqual(evidence["evidence"]["oracle_source"],"DETERMINISTIC_REFERENCE")
        self.assertEqual(evidence["evidence"]["oracle_version"],1)
        self.assertEqual(evidence["failed_check"],"reference")
        self.assertFalse((self.workspace/"reference.py").exists())


del DataflowClosureIntegration
