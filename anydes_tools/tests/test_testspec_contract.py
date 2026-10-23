"""Canonical TestSpec producer, strict parser and real B2/oracle controlled E2E."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from anydes_tools.tests import test_planner_resilience as prior
from anydes_tools.tests.test_operation_grounding import resolved
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner
from anydes_tools.conversation.verification_types import (
    TestSpec, TestSpecSchemaError, build_runtime_tests, test_spec_diagnostics)
from anydes_tools.conversation.typed_verification import VerificationDispatcher
from anydes_tools.conversation.oracle_registry import OracleSelector

DESCRIPTOR = {'semantic_kind':'UNIT_CONVERSION','input_type':'NUMBER','output_type':'NUMBER',
    'parameters':{'source_unit':'C','target_unit':'F'}}


def proposal(language='python'):
    ext = 'py' if language=='python' else 'cpp'
    return {'goal':'create and test a standalone numeric transformation', 'operations':[{
        'operation_id':'source','operation_kind':'GENERATE_CODE','inputs':[],
        'outputs':[{'name':'created','artifact_type':'CODE','format':ext,'filename':'canonical.'+ext}],
        'specification':{'language':language,'goal':'convert Celsius to Fahrenheit',
            'functional_requirements':['read one temperature and print the converted value'],
            'verification_requirement':'TEST_SUITE','execution_required':True,
            'operation_descriptor':DESCRIPTOR,
            'test_cases':[{'stdin':'0\n','expected_hint':'999',
                'comparison_mode':'misspelled internal enum','oracle_id':'not-authorized',
                'schema_version':'wrong internal version'}]}},
        {'operation_id':'test','operation_kind':'TEST_CODE','inputs':['created'],
            'outputs':[],'specification':{'language':language,'stdin':'0\n'}}],
        'requested_final_outputs':['created']}


class CanonicalTests(unittest.TestCase):
    def test_strict_reproduction_identifies_exact_field_without_plaintext_input(self):
        with self.assertRaises(TestSpecSchemaError) as raised:
            TestSpec.from_dict({'test_id':'zero','stdin':'secret-test-input', 'comparison_mode':'numeric'})
        detail = raised.exception.diagnostic
        self.assertEqual(detail['errors'][0]['field_path'],['comparison_mode'])
        self.assertEqual(detail['errors'][0]['constraint'],'enum')
        self.assertNotIn('secret-test-input',json.dumps(detail))

    def test_canonical_builder_owns_internal_fields_and_keeps_hint_untrusted(self):
        case = proposal()['operations'][0]['specification']['test_cases']
        test = build_runtime_tests(case,semantic_cases=True,operation_descriptor=DESCRIPTOR)[0]
        self.assertEqual(test.comparison_mode,'NUMERIC')
        self.assertEqual(test.oracle_source,'MODEL_GENERATED')
        self.assertIsNone(test.expected)
        self.assertEqual(test.expected_hint,'999')
        self.assertEqual(TestSpec.from_dict(asdict(test)),test)
        result = VerificationDispatcher.verify('CODE',test=test,stdout='32',exit_code=0,
            trusted_oracle=False,oracle_selector=OracleSelector())
        self.assertEqual(result.status.value,'PASS')
        self.assertEqual(result.evidence['oracle_source'],'DETERMINISTIC_REFERENCE')

    def test_missing_input_and_contradiction_fail_closed(self):
        for case in ({'case_id':'missing'}, {'stdin':'one','input':'two'}, {'stdin':42}):
            with self.assertRaisesRegex(ValueError,'semantic_test_invalid:'):
                build_runtime_tests([case],semantic_cases=True)

    def test_generic_python_and_cpp_input_cases_have_valid_specs(self):
        for language in ('python','cpp'):
            for stdin in ('5\n','a text string\n','[1,2,3]\n','a=1;b=2\n'):
                test = build_runtime_tests([{'stdin':stdin}],semantic_cases=True)[0]
                self.assertEqual(TestSpec.from_dict(asdict(test)).stdin,stdin)

    def test_unavailable_semantics_remain_inconclusive_not_model_expected_pass(self):
        test = build_runtime_tests([{'stdin':'abc','expected_hint':'answer','oracle_id':'force-pass'}],
            semantic_cases=True)[0]
        result = VerificationDispatcher.verify('CODE',test=test,stdout='answer',exit_code=0,
            trusted_oracle=False,oracle_selector=OracleSelector())
        self.assertEqual(result.status.value,'INCONCLUSIVE')

    def test_diagnostics_preserve_source_and_repair_origin(self):
        events=[]
        with test_spec_diagnostics(lambda event,**details:events.append((event,details)),
                source_step='source',proposal_origin='REPAIR'):
            with self.assertRaises(TestSpecSchemaError):
                TestSpec.from_dict({'test_id':'case','comparison_mode':'numeric'})
        self.assertEqual(events[0][1]['source_step'],'source')
        self.assertEqual(events[0][1]['proposal_origin'],'REPAIR')


class ControlledE2E(unittest.TestCase):
    setUp = prior.prior.Acceptance.setUp
    tearDown = prior.prior.Acceptance.tearDown
    controller = prior.prior.Acceptance.controller

    def run_flow(self, language, *, execution_step_cases=False):
        payload=proposal(language)
        if execution_step_cases:
            payload['operations'][0]['specification'].pop('test_cases')
        assistant=prior.Assistant([json.dumps(payload)],[])
        ext='py' if language=='python' else 'cpp'
        code=('x=float(input())\nprint(x*9/5+32)\n' if language=='python' else
            '#include <iostream>\nint main(){double x;std::cin>>x;std::cout<<x*9/5+32;return 0;}\n')
        c=self.controller(resolved('CREATE_NEW',artifact_type='CODE',execution_intent='TEST'),
            resolved('CONFIRM'),resolved('EXECUTE_EXISTING'),resolved('CONFIRM'),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':language,'filename':'canonical.'+ext,'code':code,'tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute, \
                patch.object(c.oracle_selector,'select',wraps=c.oracle_selector.select) as select:
            c.handle('Make a standalone conversion program and test input zero',turn_id='create')
            self.assertEqual(execute.call_count,0)
            self.assertIsNotNone(c.state.pending_tool_action,c.state.last_tool_failure)
            task=c.state.pending_task.task_id
            approved=c.state.pending_tool_action.action.arguments['code_generation_spec']['tests']
            self.assertEqual(approved[0]['comparison_mode'],'NUMERIC')
            c.handle('Proceed',turn_id='confirm')
            self.assertEqual(c.state.pending_task.task_id,task)
            cap='developer.run_python' if language=='python' else 'developer.run_cpp'
            runs=[call for call in execute.call_args_list if call.args[0].capability==cap]
            self.assertEqual(len(runs),1)
            self.assertEqual(select.call_count,1)
            self.assertEqual(select.call_args.args[0].stdin,approved[0]['stdin'])
            self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')
            ref=c.state.recent_artifact_state.ref_by_path(str(self.workspace/('canonical.'+ext)))
            self.assertIsNotNone(ref,c.state.last_tool_failure)
            value={'goal':'execute the same source','operations':[{
                'operation_id':'run','operation_kind':'TEST_CODE','inputs':['selected_artifact'],
                'outputs':[],'specification':{'language':language,'stdin':'10\n'}}],
                'requested_final_outputs':[]}
            assistant.online.values.append(json.dumps(value))
            c.handle('Run that program with ten now',turn_id='existing')
            self.assertIsNotNone(c.state.pending_tool_action,c.state.last_tool_failure)
            pending=c.state.pending_tool_action
            self.assertEqual(pending.action.resolved_artifact_bindings['selected_artifact'].artifact_id,ref.artifact_id)
            next_task=c.state.pending_task.task_id
            c.handle('Proceed with this execution',turn_id='confirm-existing')
            self.assertEqual(c.state.pending_task.task_id,next_task)
            runs=[call for call in execute.call_args_list if call.args[0].capability==cap]
            self.assertEqual(len(runs),2)
            self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')

    def test_python_real_runtime_b2_and_oracle(self):
        self.run_flow('python')

    def test_test_operation_input_without_duplicate_producer_case_reaches_b2(self):
        self.run_flow('python',execution_step_cases=True)

    @unittest.skipUnless(shutil.which('g++') or shutil.which('clang++'),'C++ compiler unavailable')
    def test_cpp_real_runtime_b2_and_oracle(self):
        self.run_flow('cpp')


if __name__=='__main__':
    unittest.main()
