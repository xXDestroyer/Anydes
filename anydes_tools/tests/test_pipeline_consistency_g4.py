"""Prepared-input contracts and real executor regressions, not human E2E."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from anydes_tools.conversation.verification_types import TestSpec, build_runtime_tests, VerificationResult, VerificationStatus as S
from anydes_tools.conversation.code_generation import CodeGenerationSpec
from anydes_tools.conversation.content_generation import ContentGenerationSpec, ContentValidator
from anydes_tools.conversation.typed_verification import VerificationDispatcher
from anydes_tools.conversation.oracle_registry import OracleSelector
from anydes_tools.conversation.task_adherence import evaluate_adherence, aggregate_adherence
from anydes_tools.tests.test_semantics_propagation_phase_e4 import RuntimePropagation

DESCRIPTOR = {'semantic_kind':'UNIT_CONVERSION','input_type':'NUMBER','output_type':'NUMBER',
    'parameters':{'source_unit':'C','target_unit':'F'}}


class BuilderContract(unittest.TestCase):
    def test_declared_descriptor_all_plan_sources(self):
        spec = CodeGenerationSpec.from_dict({'operation_descriptor':DESCRIPTOR,'tests':[{'test_id':'t','stdin':'0','expected_hint':999}]})
        for source in ('FAST_PATH','GENERAL_REASONER','LEGACY_ADAPTER','RETRY','FOLLOW_UP'):
            with self.subTest(source=source):
                step = SimpleNamespace(verifier_type='CODE',operation_descriptor=DESCRIPTOR)
                plan = SimpleNamespace(source=source,typed_inputs={'operation_descriptor':DESCRIPTOR},steps=(step,))
                test = build_runtime_tests(spec.tests,specification=spec,plan=plan)[0]
                self.assertEqual(test.operation_semantics['source_unit'],'C')
                self.assertEqual(test.comparison_mode,'NUMERIC')
                result = VerificationDispatcher.verify('CODE',test=test,stdout='32.0',exit_code=0,oracle_selector=OracleSelector())
                self.assertEqual(result.status,S.PASS)
                self.assertEqual(result.evidence['oracle_source'],'DETERMINISTIC_REFERENCE')

    def test_step_descriptor_is_not_lost_when_plan_root_empty(self):
        plan = SimpleNamespace(typed_inputs={},steps=(SimpleNamespace(verifier_type='CODE',operation_descriptor=DESCRIPTOR),))
        self.assertTrue(build_runtime_tests([TestSpec('t')],plan=plan)[0].operation_semantics)

    def test_conflict_fail_closed(self):
        spec = SimpleNamespace(operation_descriptor=DESCRIPTOR)
        conflicting = {**DESCRIPTOR,'parameters':{'source_unit':'F','target_unit':'C'}}
        plan = SimpleNamespace(typed_inputs={'operation_descriptor':conflicting},steps=())
        with self.assertRaisesRegex(ValueError,'descriptor_conflict'):
            build_runtime_tests([TestSpec('t')],specification=spec,plan=plan)

    def test_true_absence_inconclusive(self):
        test = build_runtime_tests([TestSpec('missing',stdin='0',expected_hint=32)])[0]
        result = VerificationDispatcher.verify('CODE',test=test,stdout='32',exit_code=0)
        self.assertEqual(result.status,S.INCONCLUSIVE)

    def test_invalid_input_not_assigned_functional_oracle(self):
        test = TestSpec('error',stdin='abc',invalid_input_behavior='NONZERO_EXIT')
        result = build_runtime_tests([test],specification=SimpleNamespace(operation_descriptor=DESCRIPTOR))[0]
        self.assertFalse(result.operation_semantics)


class ValidationResponsibilities(unittest.TestCase):
    def test_multilingual_topic_not_lexical_precheck(self):
        spec = ContentGenerationSpec.from_dict({'topic':'healthy habits','target_length_words':10})
        result = ContentValidator.validate('Dormir bien ayuda a cuidar nuestra salud y mantener energia.',spec)
        self.assertEqual(result.status,'VALID')
        self.assertIsNone(result.topic_grounded)

    def test_quantitative_short_circuit_remains(self):
        spec = ContentGenerationSpec.from_dict({'topic':'anything','target_length_words':180})
        result = ContentValidator.validate(' '.join(['word']*50),spec)
        self.assertEqual(result.reason,'generated_content_length_out_of_tolerance')

    def test_out_of_goal_rejected_by_adherence_not_precheck(self):
        spec = ContentGenerationSpec.from_dict({'topic':'software project','target_length_words':10})
        content = 'The village has mountains forests rivers farms and historic houses.'
        self.assertEqual(ContentValidator.validate(content,spec).status,'VALID')
        adherence = evaluate_adherence(lambda e:{'status':'FAIL','goal_aligned':False,'subject_aligned':False,
            'reason_code':'SUBJECT_MISMATCH','evidence_basis':'Requested software; produced geography.'},
            goal='describe a software project',specification=spec,content=content)
        combined = aggregate_adherence(VerificationResult('DOCUMENT','DOCUMENT',S.PASS),adherence)
        self.assertEqual(combined.status,S.FAIL)
        self.assertEqual(combined.failed_check,'task_adherence')


class RuntimeBuilderRegression(RuntimePropagation):
    """Inherited E4 prepared-input tests use real B2, dispatcher, oracle and commit."""


del RuntimePropagation
