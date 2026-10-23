"""Material requirements and execution authority: no domain-specific text rules."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from anydes_tools.tests import test_manual_flow as prior
from anydes_tools.tests import test_execute_contract as execution
from anydes_tools.conversation.task_continuity import TaskContract
from anydes_tools.conversation.execution_contract import prepare_semantic_execution,plan_execution_errors
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer,PlanLoweringError
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.conversation.publication import OutcomeStatus


def start():
    return {**prior.update(target=('language','Which language?')),
        'requirement_items':[
            {'key':'language','necessity':'REQUIRED','question':'Which language?'},
            {'key':'selection_policy','necessity':'REQUIRED','question':'Which elements should be selected?'}],
        'planning_readiness':{'goal_resolvable':True,'artifact_specified':True,'material_contradictions':[]}}


def creation(execute=False):
    result={'goal':'Create a simple source','operations':[{
        'operation_id':'make','operation_kind':'GENERATE_CODE','inputs':[],
        'outputs':[{'name':'source','artifact_type':'CODE','format':'py','filename':'sample.py'}],
        'specification':{'language':'python','goal':'print 42','functional_requirements':['print 42']}}],
        'requested_final_outputs':['source']}
    if execute:
        result['operations'].append({'operation_id':'run','operation_kind':'TEST_CODE',
            'inputs':['source'],'outputs':[],'specification':{'language':'python'}})
    return result


def constraint(execute=False):
    return {'requires_execution':execute,'verification_requirement':'EXECUTION' if execute else 'BASIC'}


class Requirements(unittest.TestCase):
    def task(self,update=None):
        update=update or start()
        decision=prior.wire('CLARIFICATION',task_update=update)
        task=TaskContract.start(decision,update); task.merge(update,'start')
        return task

    def test_first_answer_does_not_clear_other_material_requirement(self):
        task=self.task();task.merge(prior.update('CONTINUE',[('language','Python')]),'answer1')
        self.assertFalse(task.planning_gate()['ready'])
        self.assertEqual(task.planning_gate()['next_target'],'selection_policy')
        self.assertEqual(task.snapshot()['required_resolved'],['language'])

    def test_last_required_answer_opens_gate_same_task_and_provenance(self):
        task=self.task();ident=task.task_id
        task.merge(prior.update('CONTINUE',[('language','Python')]),'a')
        task.merge(prior.update('CONTINUE',[('selection_policy','letters only')]),'b')
        self.assertTrue(task.planning_gate()['ready'])
        self.assertEqual(task.task_id,ident)
        self.assertEqual([r['turn_id'] for r in task.history],['a','b'])

    def test_optional_does_not_require_question(self):
        update=prior.update()
        update['requirement_items']=[{'key':'theme','necessity':'OPTIONAL'}]
        task=self.task(update)
        self.assertTrue(task.planning_gate()['ready'])
        self.assertEqual(task.snapshot()['optional'],['theme'])

    def test_defaultable_has_explicit_provenance(self):
        update=prior.update()
        update['requirement_items']=[{'key':'layout','necessity':'DEFAULTABLE','default_value':'plain paragraphs'}]
        task=self.task(update)
        self.assertTrue(task.planning_gate()['ready'])
        self.assertEqual(task.requirements['layout']['source'],'SEMANTIC_DEFAULT')
        self.assertEqual(task.requirements['layout']['turn_id'],'start')

    def test_required_cannot_silently_default(self):
        update=start();update['requirement_items'][0]['default_value']='Python'
        with self.assertRaisesRegex(ValueError,'required_cannot_be_defaulted'):
            self.task(update)

    def test_missing_required_question_rejected_before_merge(self):
        from anydes_tools.conversation.task_continuity import validate_continuity
        update=start();update['requirement_items'][1].pop('question')
        with self.assertRaisesRegex(ValueError,'required_unresolved_question_missing'):
            validate_continuity(update,{})

    def test_revision_of_materiality_is_explicit(self):
        task=self.task()
        revision=prior.update('REVISE',[('language','Python')])
        revision['requirement_items']=[{'key':'selection_policy','necessity':'DEFAULTABLE',
            'default_value':'use the established document order'}]
        task.merge(revision,'revision')
        self.assertTrue(task.planning_gate()['ready'])
        self.assertEqual(task.requirements['selection_policy']['source'],'SEMANTIC_DEFAULT')

    def test_incomplete_goal_artifact_or_contradiction_blocks(self):
        for field,value in [('artifact_specified',False),
                ('material_contradictions',['conflicting_scope'])]:
            update=prior.update();update['planning_readiness']={
                'goal_resolvable':True,'artifact_specified':True,'material_contradictions':[],field:value}
            self.assertFalse(self.task(update).planning_gate()['ready'])


class SemanticExecution(unittest.TestCase):
    def prepare(self,value,execute=False):
        self.events=[]
        return prepare_semantic_execution(SemanticProposal.from_dict(value),constraint(execute),
            lambda e,**d:self.events.append((e,d)),origin='ORIGINAL')

    def test_create_only_omission_uses_semantic_false_not_legacy_true(self):
        value=creation();before=deepcopy(value)
        result=self.prepare(value)
        self.assertIs(result.payload['operations'][0]['specification']['execution_required'],False)
        self.assertEqual(value,before)
        self.assertEqual(self.events[-1][1]['value_origin'],'HOST_SEMANTIC_CONTRACT')

    def test_true_without_run_rejected_and_diagnosed(self):
        value=creation();value['operations'][0]['specification']['execution_required']=True
        with self.assertRaisesRegex(PlanLoweringError,'WITHOUT_EXECUTION_CAPABILITY'):
            self.prepare(value,True)
        event=self.events[-1][1]
        self.assertIs(event['received_value'],True)
        self.assertEqual(event['value_origin'],'PLANNER_EXPLICIT')
        self.assertEqual(event['proposed_operations'],['GENERATE_CODE'])
        self.assertIn('developer.run_python',event['expected_capability_family'])

    def test_create_only_can_verify_only_exact_generated_source(self):
        value=creation(True);value['operations'][0]['specification']['execution_required']=True
        result=self.prepare(value)
        self.assertEqual(result.payload['operations'][1]['specification']['execution_purpose'],'VERIFICATION')
        value['operations'][1]['inputs']=['existing_source']
        with self.assertRaises(PlanLoweringError):
            self.prepare(value)

    def test_composite_has_true_and_explicit_generated_binding(self):
        result=self.prepare(creation(True),True)
        self.assertTrue(result.payload['operations'][0]['specification']['execution_required'])
        self.assertEqual(result.payload['operations'][1]['inputs'],['source'])

    def test_false_with_create_only_is_valid(self):
        value=creation();value['operations'][0]['specification']['execution_required']=False
        self.assertFalse(self.prepare(value).payload['operations'][0]['specification']['execution_required'])

    def test_true_nonboolean_not_coerced(self):
        value=creation();value['operations'][0]['specification']['execution_required']='false'
        with self.assertRaisesRegex(PlanLoweringError,'TYPE_INVALID'):
            self.prepare(value)

    def test_legacy_lowered_plan_default_diagnostic(self):
        plan=GeneralTaskPlan.from_dict({'task_id':'t','task_type':'GENERATE_CODE','goal':'g',
            'inputs':{'code_spec':{'goal':'g'}},'required_capabilities':['developer.write_code']})
        events=[]
        errors=plan_execution_errors(plan,prior.wire('CREATE_NEW',artifact_type='CODE'),
            lambda e,**d:events.append(d),origin='ORIGINAL')
        self.assertTrue(errors)
        self.assertEqual(events[0]['value_origin'],'LEGACY_DEFAULT_TRUE')


class GateIntegration(unittest.TestCase):
    setUp=prior.TaskIntegration.setUp
    tearDown=prior.TaskIntegration.tearDown
    controller=prior.TaskIntegration.controller

    def test_two_required_items_create_then_location_without_added_target(self):
        requests=[]
        payload=creation()
        def reason(request):
            requests.append(request)
            from anydes_tools.conversation.executable_plan import BrainPlanProposal
            return BrainPlanProposal(SemanticProposal.from_dict(payload))
        updates=[('CLARIFICATION',start()),('CONTINUE_TASK',prior.update('CONTINUE',[('language','Python')])),
            ('CONTINUE_TASK',prior.update('CONTINUE',[('selection_policy','only letters')]))]
        c=self.controller(general_task_reasoner=reason,code_generator=lambda spec:{
            'language':'python','filename':'sample.py','code':'print(42)\n','tests':[]})
        def classify(request):
            intent,update=updates.pop(0)
            return prior.wire(intent,request['state'],task_update=update)
        c.interaction_resolver.classifier=classify
        c.handle('Prepare a source with unspecified language and selection rules.',turn_id='start')
        task=c.state.pending_task.task_id
        c.handle('Choose Python.',turn_id='language')
        self.assertEqual(c.last_runtime_outcome.status,OutcomeStatus.CLARIFICATION)
        self.assertEqual(c.state.pending_task.task_id,task)
        self.assertEqual(len(requests),0)
        self.assertFalse(c._executor_called)
        c.handle('Select letters only.',turn_id='policy')
        self.assertEqual(len(requests),1)
        self.assertEqual(c.state.pending_task.task_id,task)
        self.assertTrue((self.workspace/'sample.py').exists(),c.state.last_tool_failure)
        self.assertFalse(c.state.pending_task.semantic_contract.execution_required)
        self.assertEqual(set(c.state.pending_task.semantic_contract.requirements),{'language','selection_policy'})
        c.interaction_resolver.classifier=lambda r:prior.wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION')
        c.handle('Where is the result?',turn_id='location')
        self.assertEqual(c._metadata_evidence['value'],str(self.workspace/'sample.py'))
        self.assertFalse(c._executor_called or c._reasoner_called)


class ExecutionRepairIntegration(unittest.TestCase):
    setUp=execution.ExecutionIntegration.setUp
    tearDown=execution.ExecutionIntegration.tearDown
    controller=execution.ExecutionIntegration.controller

    def test_missing_runner_repair_same_task_then_real_b2_once(self):
        bad=creation();bad['operations'][0]['specification']['execution_required']=True
        good=creation(True)
        assistant=execution.provider.Assistant([json.dumps(bad),json.dumps(good)],[])
        c=self.controller(execution.resolved('CREATE_NEW',artifact_type='CODE',execution_intent='RUN'),
            execution.resolved('CONFIRM'),general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':'python','filename':'sample.py','code':'print(42)\n','tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as run:
            c.handle('Make a source and run it.',turn_id='create')
            self.assertIsNotNone(c.state.pending_tool_action,c.state.last_tool_failure)
            task=c.state.pending_task.task_id
            repair=json.loads(assistant.online.calls[1][0][1]['content'])
            self.assertEqual(repair['context']['execution_contract_repair']['task_id'],task)
            self.assertIn('WITHOUT_EXECUTION_CAPABILITY',str(repair['repair']))
            self.assertEqual(len(assistant.online.calls),2)
            c.handle('Proceed with that task.',turn_id='approve')
            self.assertEqual(c.state.pending_task.task_id,task)
            self.assertEqual(len([x for x in run.call_args_list if x.args[0].capability=='developer.run_python']),1)
            self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')

