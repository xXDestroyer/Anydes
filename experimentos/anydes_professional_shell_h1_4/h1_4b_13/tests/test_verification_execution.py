"""Real permission bridge/B2 with prepared semantic proposals, no UI injection."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from anydes_tools.tests import test_execute_contract as prior
from anydes_tools.tests.test_clarification_completeness import creation, constraint
from anydes_tools.conversation.execution_contract import prepare_semantic_execution
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.plan_lowering import PlanLoweringError, ExecutablePlanLowerer
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner


def proposal():
    value=creation(True)
    value['operations'][0]['specification'].update(execution_required=True,
        verification_requirement='EXECUTION')
    return value


class Purpose(unittest.TestCase):
    def prepare(self,value,user=False):
        return prepare_semantic_execution(SemanticProposal.from_dict(value),constraint(user),
            lambda *a,**k:None,origin='TEST')

    def test_internal_verification_does_not_change_user_intent(self):
        value=proposal();before=deepcopy(value)
        result=self.prepare(value)
        self.assertEqual(value,before)
        self.assertEqual(result.payload['operations'][1]['specification']['execution_purpose'],'VERIFICATION')

    def test_requested_action_is_distinct(self):
        result=self.prepare(proposal(),True)
        self.assertEqual(result.payload['operations'][1]['specification']['execution_purpose'],'ACTION')

    def test_label_cannot_grant_unrequested_action(self):
        value=proposal();value['operations'][1]['specification']['execution_purpose']='ACTION'
        with self.assertRaisesRegex(PlanLoweringError,'PURPOSE_CONFLICT'):
            self.prepare(value)

    def test_existing_execution_cannot_call_itself_verification(self):
        value=proposal();value['operations']=value['operations'][1:]
        value['operations'][0]['inputs']=['selected_artifact']
        value['requested_final_outputs']=[]
        with self.assertRaisesRegex(PlanLoweringError,'EXECUTION_NOT_REQUESTED'):
            self.prepare(value)

    def test_verification_requires_declared_compatible_capability(self):
        value=proposal();value['operations'].pop()
        with self.assertRaisesRegex(PlanLoweringError,'WITHOUT_EXECUTION_CAPABILITY'):
            self.prepare(value)


class Integration(unittest.TestCase):
    setUp=prior.ExecutionIntegration.setUp
    tearDown=prior.ExecutionIntegration.tearDown
    controller=prior.ExecutionIntegration.controller

    def run_source(self,source='print(42)\n',user=False,semantic=None):
        assistant=prior.provider.Assistant([json.dumps(semantic or proposal())],[])
        c=self.controller(prior.resolved('CREATE_NEW',artifact_type='CODE',execution_intent='RUN' if user else 'NONE'),
            prior.resolved('CONFIRM'),general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':'python','filename':'sample.py','code':source,'tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            c.handle('Prepare the requested source.',turn_id='create')
            self.assertIsNotNone(c.state.pending_tool_action,(c.state.last_tool_failure,c.last_runtime_outcome))
            self.assertEqual(execute.call_count,0,'verification still requires normal execution confirmation')
            action=c.state.pending_tool_action.action
            contract=action.arguments['code_execution_contract']
            self.assertEqual(contract['user_execution_requested'],user)
            self.assertTrue(contract['verification_execution_required'])
            rows=action.arguments['general_task_plan']['steps']
            self.assertEqual(next(r['purpose'] for r in rows if r['capability']=='developer.run_python'),
                'EXECUTE_AS_USER_ACTION' if user else 'EXECUTE_FOR_VERIFICATION')
            c.handle('Proceed with that plan.',turn_id='approve')
            runs=[call.args[0] for call in execute.call_args_list if call.args[0].capability=='developer.run_python']
            self.assertEqual(len(runs),1)
            self.assertIn('.anydes-code-',runs[0].arguments['path'])
            self.assertTrue(runs[0].confirmed_effects)
        return c

    def test_create_only_real_internal_b2_verifies_and_commits(self):
        c=self.run_source()
        self.assertTrue((self.workspace/'sample.py').exists(),c.state.last_tool_failure)
        self.assertFalse(c.state.pending_task.semantic_contract.execution_required)
        self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')

    def test_requested_action_still_uses_normal_authorization(self):
        c=self.run_source(user=True)
        self.assertTrue((self.workspace/'sample.py').exists(),c.state.last_tool_failure)

    def test_failed_verification_does_not_commit_or_claim_success(self):
        c=self.run_source('raise RuntimeError("controlled failure")\n')
        self.assertFalse((self.workspace/'sample.py').exists())
        self.assertTrue(c.state.last_tool_failure)
        self.assertNotEqual(c.last_runtime_outcome.status.value,'SUCCESS')
        self.assertFalse(any(row.get('status')=='PASS' for row in c.last_runtime_outcome.verification_summary))

    def test_timeout_verification_does_not_commit(self):
        c=self.run_source('while True: pass\n')
        self.assertFalse((self.workspace/'sample.py').exists())
        self.assertTrue(c.state.last_tool_failure)

    def test_inconclusive_verifier_does_not_commit(self):
        from anydes_tools.conversation.verification_types import VerificationResult, VerificationStatus
        inconclusive=VerificationResult('CODE','CODE',VerificationStatus.INCONCLUSIVE,
            error_code='CONTROLLED_INCONCLUSIVE')
        with patch('anydes_tools.conversation.controller.VerificationDispatcher.verify',return_value=inconclusive):
            c=self.run_source()
        self.assertFalse((self.workspace/'sample.py').exists())
        self.assertNotEqual(c.last_runtime_outcome.status.value,'SUCCESS')

    def test_letter_program_create_only_internal_verification(self):
        value=proposal()
        spec=value['operations'][0]['specification']
        spec.update(goal='Count ASCII letters from an input sentence',
            functional_requirements=['Read a sentence and count only ASCII letters'],
            verification_requirement='TEST_SUITE',test_cases=[{'stdin':'Hola mundo!\n'}],
            operation_descriptor={'semantic_kind':'CHARACTER_COUNT','input_type':'TEXT','output_type':'NUMBER',
                'parameters':{'alphabet':'abcdefghijklmnopqrstuvwxyz','casefold':True,'strip_accents':False}})
        c=self.run_source('text=input()\nprint(sum(c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text))\n',
            semantic=value)
        self.assertTrue((self.workspace/'sample.py').exists(),c.state.last_tool_failure)
        self.assertFalse(c.state.pending_task.semantic_contract.execution_required)
        self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')

    def test_lowered_purposes_have_same_exact_generated_binding(self):
        c=self.controller(prior.resolved('CREATE_NEW',artifact_type='CODE'))
        for user,purpose in [(False,'EXECUTE_FOR_VERIFICATION'),(True,'EXECUTE_AS_USER_ACTION')]:
            prepared=prepare_semantic_execution(SemanticProposal.from_dict(proposal()),constraint(user),
                lambda *a,**k:None,origin='TEST')
            plan=ExecutablePlanLowerer(c.registry,self.workspace).lower_requirements(prepared)
            self.assertEqual(plan.steps[1].purpose,purpose)
            self.assertEqual(plan.steps[1].consumes_artifacts,plan.steps[0].produces_artifacts)
