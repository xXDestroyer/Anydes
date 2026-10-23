"""Execution contract tests: prepared cognition, real lowering/permission/B2."""
import json
import unittest
from unittest.mock import patch
from anydes_tools.tests import test_planner_resilience as provider
from anydes_tools.tests.test_operation_grounding import resolved, semantic
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, operation_errors, ResolutionSchemaError
from anydes_tools.conversation.semantic_proposal import SemanticProposal, operation_mismatches, operation_constraint
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner


def execution():
    value = provider.proposal()
    value['requested_final_outputs'] = []
    return value


def content():
    return {'goal':'execution limitation note','operations':[{
        'operation_id':'note','operation_kind':'GENERATE_CONTENT','inputs':[],
        'outputs':[{'name':'note','artifact_type':'CONTENT','format':'text'}],
        'specification':{'topic':'execution limitation'}}], 'requested_final_outputs':['note']}


class Projection(unittest.TestCase):
    def test_composite_run_and_tests_preserve_effects(self):
        for mode, verification in [('RUN','EXECUTION'),('TEST','TEST_SUITE')]:
            value = semantic('CREATE_NEW',artifact_type='CODE',execution_intent=mode)
            wire = semantic_resolution_payload(value,{})
            self.assertTrue(wire['requires_execution'])
            self.assertEqual(wire['verification_requirement'],verification)
            decision = resolved('CREATE_NEW',artifact_type='CODE',execution_intent=mode)
            self.assertFalse(operation_errors(decision))
            self.assertTrue(decision.effect_intent.create and decision.effect_intent.execute)

    def test_composite_contradiction_cannot_drop_execution(self):
        with self.assertRaisesRegex(ResolutionSchemaError,'composite_execution_dropped'):
            semantic_resolution_payload(semantic('CREATE_NEW',artifact_type='CODE',
                execution_intent='TEST',execute=False,verification_requirement='BASIC'),{})

    def test_create_only_and_existing_execution_compatible(self):
        self.assertFalse(resolved('CREATE_NEW',artifact_type='CODE',execution_intent='NONE').requires_execution)
        self.assertFalse(operation_errors(resolved('EXECUTE_EXISTING')))


class ExecutionIntegration(unittest.TestCase):
    setUp = provider.prior.Acceptance.setUp
    tearDown = provider.prior.Acceptance.tearDown
    controller = provider.prior.Acceptance.controller
    source = provider.prior.Acceptance.source

    def prepared(self, responses):
        assistant = provider.Assistant([json.dumps(v) for v in responses],[])
        c = self.controller(resolved('EXECUTE_EXISTING'),resolved('CONFIRM'),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant))
        ref = self.source(c,'exact.py')
        return c, ref, assistant

    def test_existing_external_binding_requires_host_authorization(self):
        c, ref, _ = self.prepared([execution()])
        plan = ExecutablePlanLowerer(c.registry,self.workspace,
            external_artifacts={'selected_artifact':ref}).lower_requirements(SemanticProposal.from_dict(execution()))
        missing = c.general_plan_validator.validate(plan,authorized_scopes=c.state.authorized_scopes)
        self.assertTrue(any('artifact_dependency_not_preceding' in e for e in missing.errors))
        valid = c.general_plan_validator.validate(plan,authorized_scopes=c.state.authorized_scopes,
            external_artifacts=('selected_artifact',))
        self.assertTrue(valid.valid,valid.errors)
        self.assertFalse(plan.desired_outputs or plan.requested_final_outputs)

    def run_once(self, responses):
        c, ref, assistant = self.prepared(responses)
        original = (self.workspace/'exact.py').read_bytes()
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            answer = c.handle('Run the selected program',turn_id='request')
            self.assertIsNotNone(c.state.pending_tool_action,(answer,c.state.last_tool_failure))
            self.assertEqual(execute.call_count,0)
            task = c.state.pending_task.task_id
            action = c.state.pending_tool_action.action
            self.assertEqual(action.resolved_artifact_bindings['selected_artifact'].artifact_id,ref.artifact_id)
            c.handle('Yes, proceed',turn_id='approve')
            self.assertEqual(c.state.pending_task.task_id,task)
            runs = [call for call in execute.call_args_list if call.args[0].capability=='developer.run_python']
            self.assertEqual(len(runs),1)
            self.assertEqual((self.workspace/'exact.py').read_bytes(),original)
            self.assertTrue(c._executor_called)
            self.assertTrue(c.last_runtime_outcome.verification_summary,c.last_runtime_outcome)
        return assistant

    def test_existing_execution_confirmation_b2_once_and_verification(self):
        self.run_once([execution()])

    def test_content_mismatch_repaired_once_same_operation_and_task(self):
        assistant = self.run_once([content(),execution()])
        self.assertEqual(len(assistant.online.calls),2)
        repair = json.loads(assistant.online.calls[1][0][1]['content'])
        self.assertEqual(repair['context']['operation_constraint']['requested_operation'],'EXECUTE_EXISTING')
        self.assertIn('SEMANTIC_OPERATION_MISMATCH',str(repair['repair']))
        self.assertEqual({r['operation_kind'] for r in repair['context']['available_operations']},{'TEST_CODE'})

    def test_repeated_content_fails_closed_no_permission_or_execution(self):
        c, _, assistant = self.prepared([content(),content()])
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            c.handle('Execute the current program',turn_id='rejected')
        self.assertEqual(execute.call_count,0)
        self.assertIsNone(c.state.pending_tool_action)
        self.assertEqual(len(assistant.online.calls),2)
        self.assertTrue(c.state.last_tool_failure.error_code.startswith('SEMANTIC_OPERATION_MISMATCH'))
        self.assertEqual(c.state.last_tool_failure.safe_user_message,'No pude preparar la ejecucion del programa.')

    def test_creation_requires_generated_binding_not_existing_focus(self):
        d = resolved('CREATE_NEW',artifact_type='CODE',execution_intent='RUN')
        constraint = operation_constraint({'operation_intent':d.operation_intent.value,
            'requires_execution':True,'effect_intent':d.effect_intent.to_dict(),
            'verification_requirement':d.verification_requirement.value})
        errors = operation_mismatches(SemanticProposal.from_dict(execution()),constraint)
        self.assertIn('SEMANTIC_OPERATION_MISMATCH:execute_exact_generated_code',errors)

    def test_create_and_execute_composite_confirmation_then_b2(self):
        value = {'goal':'create source and execute it','operations':[{
            'operation_id':'create','operation_kind':'GENERATE_CODE','inputs':[],
            'outputs':[{'name':'created','artifact_type':'CODE','format':'py','filename':'created.py'}],
            'specification':{'language':'python','goal':'print 42',
                'functional_requirements':['print 42'], 'verification_requirement':'EXECUTION',
                'execution_required':True,'tests':[]}},
            {'operation_id':'run','operation_kind':'TEST_CODE','inputs':['created'],
                'outputs':[],'specification':{'language':'python'}}],
            'requested_final_outputs':['created']}
        assistant = provider.Assistant([json.dumps(value)],[])
        c = self.controller(resolved('CREATE_NEW',artifact_type='CODE',execution_intent='RUN'),
            resolved('CONFIRM'), general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':'python','filename':'created.py',
                'code':'print(42)\n','tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            c.handle('Make a source that prints 42 and run it',turn_id='composite')
            self.assertIsNotNone(c.state.pending_tool_action,c.state.last_tool_failure)
            self.assertEqual(execute.call_count,0)
            task = c.state.pending_task.task_id
            plan = c.state.pending_task.plan
            runs = [s for s in plan.steps if s.capability_id=='developer.run_python']
            self.assertEqual(len(runs),1)
            self.assertTrue(runs[0].consumes_artifacts)
            c.handle('Go ahead',turn_id='confirmation')
            self.assertEqual(c.state.pending_task.task_id,task)
            calls = [a for a in execute.call_args_list if a.args[0].capability=='developer.run_python']
            self.assertEqual(len(calls),1)
            self.assertEqual((self.workspace/'created.py').read_text(),'print(42)\n')
            self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')


if __name__ == '__main__':
    unittest.main()
