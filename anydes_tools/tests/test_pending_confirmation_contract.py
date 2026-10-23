"""Prepared semantic outputs; real existing confirmation boundary and B2."""
import copy
import json
import unittest
from unittest.mock import patch
from anydes_tools.tests import test_planner_resilience as prior
from anydes_tools.tests.test_testspec_contract import proposal
from anydes_tools.tests.test_operation_grounding import resolved
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner
from anydes_tools.conversation.interaction_router import InteractionResolver, TaskControl, PendingActionRelation


class PendingRelations(unittest.TestCase):
    state={'failure_available':False,'retry_available':False,'pending_action':True,
        'resolution_context':{'previous_assistant_act':'CONFIRMATION_REQUEST',
            'pending_confirmation':{'available':True,'task_id':'task:one','plan_id':'plan:one',
                'requested_action':{'operation':'general_generate_code'}}}}

    def test_contextual_affirmatives_keep_confirm_without_phrase_rules(self):
        for phrase in ('Yes, carry on with the run','Go ahead with that execution',
                       'Please proceed as proposed','That is fine, do it'):
            d=resolved('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION')
            self.assertEqual(InteractionResolver(lambda _:d).resolve(phrase,state=self.state).task_control,TaskControl.CONFIRM)

    def test_acceptance_operation_conflict_repairs_once(self):
        calls=[]
        bad=resolved('EXECUTE_EXISTING',state=self.state,pending_action_relation='ACCEPT_PENDING_ACTION')
        good=resolved('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION')
        def classifier(request):
            calls.append(request)
            return bad if len(calls)==1 else good
        result=InteractionResolver(classifier).resolve('Continue with the proposed execution',state=self.state)
        self.assertEqual(result.task_control,TaskControl.CONFIRM)
        self.assertEqual(len(calls),2)
        self.assertIn('pending_acceptance_control_conflict',calls[1]['repair_errors'])
        self.assertEqual(calls[1]['pending_confirmation_contract']['pending_action']['plan_id'],'plan:one')

    def test_undeclared_pending_relation_requires_one_repair_not_auto_confirm(self):
        calls=[]
        bad=resolved('EXECUTE_EXISTING')
        def classifier(request):
            calls.append(request)
            return bad
        result=InteractionResolver(classifier).resolve('A follow-up with no discourse metadata',state=self.state)
        self.assertEqual(result.kind.value,'CLARIFICATION')
        self.assertEqual(len(calls),2)
        self.assertIn('pending_action_relation_required',calls[1]['repair_errors'])

    def test_explicit_different_artifact_remains_new_action(self):
        decision=resolved('EXECUTE_EXISTING',state=self.state,references=['other.py'],pending_action_relation='NEW_ACTION')
        result=InteractionResolver(lambda _:decision).resolve('Run the other source instead',state=self.state)
        self.assertEqual(result.task_control,TaskControl.NONE)
        self.assertEqual(result.references,('other.py',))

    def test_rejection_and_real_ambiguity_remain_non_operational(self):
        for intent,relation,expected in [('CANCEL','REJECT_PENDING_ACTION','CANCEL'),
                ('CLARIFICATION','AMBIGUOUS','NONE')]:
            decision=resolved(intent,state=self.state,pending_action_relation=relation)
            result=InteractionResolver(lambda _:decision).resolve('Prepared semantic reply',state=self.state)
            self.assertEqual(result.task_control.value,expected)
            self.assertFalse(result.requested_effects)

    def test_confirm_cannot_expand_authority(self):
        decision=resolved('CONFIRM',execute=True,verification_requirement='EXECUTION',
            pending_action_relation='ACCEPT_PENDING_ACTION')
        self.assertIn('confirmation_cannot_extend_pending_action',InteractionResolver.contradictions(decision,self.state))


class PendingE2E(unittest.TestCase):
    setUp=prior.prior.Acceptance.setUp
    tearDown=prior.prior.Acceptance.tearDown
    controller=prior.prior.Acceptance.controller

    def test_semantic_affirmative_resumes_exact_plan_testspec_and_b2_once(self):
        assistant=prior.Assistant([json.dumps(proposal())],[])
        c=self.controller(resolved('CREATE_NEW',artifact_type='CODE',execution_intent='TEST'),
            resolved('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION'),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':'python','filename':'canonical.py',
                'code':'v=float(input())\nprint(v*9/5+32)\n','tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute, \
                patch.object(c.oracle_selector,'select',wraps=c.oracle_selector.select) as oracle:
            c.handle('Create the transformation source and test zero',turn_id='create')
            pending=c.state.pending_tool_action
            self.assertIsNotNone(pending,c.state.last_tool_failure)
            task=c.state.pending_task.task_id
            plan=c.state.pending_task.plan
            approved=copy.deepcopy(pending.action.arguments['code_generation_spec']['tests'])
            bindings=copy.deepcopy(pending.action.resolved_artifact_bindings)
            context=c._context_snapshot().pending_confirmation
            self.assertEqual(context['plan_id'],pending.plan_id)
            self.assertEqual(context['task_id'],task)
            self.assertIn('EXECUTE_CODE',context['permission_effects'])
            self.assertTrue(context['requested_action'])
            self.assertEqual(execute.call_count,0)
            c.handle('Yes, go ahead with that execution',turn_id='confirm')
            self.assertEqual(c.state.pending_task.task_id,task)
            self.assertEqual(c.state.pending_task.plan.legacy_action.structured_action_plan.plan_id,
                plan.legacy_action.structured_action_plan.plan_id)
            self.assertEqual(c.state.pending_task.plan.legacy_action.arguments['code_generation_spec']['tests'],approved)
            self.assertEqual(c.state.pending_task.plan.legacy_action.resolved_artifact_bindings,bindings)
            self.assertIsNone(c.state.pending_tool_action)
            self.assertEqual(len(assistant.online.calls),1)
            runs=[a for a in execute.call_args_list if a.args[0].capability=='developer.run_python']
            self.assertEqual(len(runs),1)
            self.assertEqual(oracle.call_count,1)
            self.assertEqual(c.last_runtime_outcome.verification_summary[0]['status'],'PASS')

    def test_pending_rejection_executes_nothing(self):
        assistant=prior.Assistant([json.dumps(proposal())],[])
        c=self.controller(resolved('CREATE_NEW',artifact_type='CODE',execution_intent='TEST'),
            resolved('CANCEL',pending_action_relation='REJECT_PENDING_ACTION'),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant))
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            c.handle('Create and test a program',turn_id='request')
            self.assertIsNotNone(c.state.pending_tool_action)
            c.handle('No, leave it undone',turn_id='reject')
            self.assertEqual(execute.call_count,0)
            self.assertIsNone(c.state.pending_tool_action)


if __name__=='__main__':
    unittest.main()
