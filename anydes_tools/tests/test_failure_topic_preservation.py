"""Prepared discourse contracts; real controller/publication, no operational plans for explanations."""
from unittest.mock import patch
import unittest
from anydes_tools.tests import test_interaction_execution_phase_c as phase_c
from anydes_tools.conversation.failure_state import FailureRecord, FailureResponseRenderer
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T, InteractionResolver)
from anydes_tools.conversation.publication import FinalResponseComposer


class FailureTopic(unittest.TestCase):
    setUp = phase_c.PhaseC.setUp
    tearDown = phase_c.PhaseC.tearDown
    controller = phase_c.PhaseC.controller

    def root(self, c, task='original'):
        record = FailureRecord.create(task_id=task, stage='PLAN_VALIDATION',
            error_code='PLAN_INCOMPLETE:requested_execution_missing',
            safe_user_message='No pude preparar la tarea.', retryable=False,
            validation_detail={'failed_constraint':'PLAN_INCOMPLETE',
                'errors':['PLAN_INCOMPLETE:requested_execution_missing',
                    'PLAN_INCOMPLETE:requested_test_suite_missing']})
        c.state.pending_task.task_id = task
        c.state.last_tool_failure = record
        return record

    def explain(self):
        return R(K.CONTENT_RESPONSE, task_control=T.EXPLAIN_FAILURE, confidence=1,
            requested_operations=('EXPLAIN_FAILURE',))

    def test_three_depths_same_root_no_planner_or_executor(self):
        c = self.controller(*(self.explain() for _ in range(3)))
        root = self.root(c)
        texts = []
        for i, text in enumerate(('why did it fail', 'explain the implication', 'is it repairable')):
            c.handle(text, turn_id=str(i))
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertFalse(c._executor_called)
            self.assertFalse(c._reasoner_called)
            self.assertEqual(c.state.active_failure_topic.explanation_depth,i+1)
            final = FinalResponseComposer.compose(c.last_runtime_outcome)
            self.assertEqual(final.visible_text,final.spoken_text)
            texts.append(final.visible_text)
        self.assertIn('ejecutar el programa',texts[0])
        self.assertIn('no significa',texts[1])
        self.assertIn('No hay un reintento automatico',texts[2])

    def test_secondary_renderer_error_preserves_topic_and_task(self):
        c = self.controller(self.explain(),self.explain())
        root = self.root(c)
        with patch.object(FailureResponseRenderer,'render',side_effect=ValueError('private detail')):
            c.handle('explain that failure',turn_id='renderer-error')
        secondary = c.state.latest_runtime_error
        self.assertNotEqual(secondary.failure_id,root.failure_id)
        self.assertEqual(secondary.caused_during_explanation_of,root.failure_id)
        self.assertEqual(c.state.pending_task.failure_ref,root)
        self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
        c.handle('clarify the same reason',turn_id='follow-up')
        self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
        self.assertIn(secondary,c.state.failure_history)
        self.assertIn(root,c.state.failure_history)

    def test_exhausted_resolution_repair_preserves_topic(self):
        c = self.controller(R(K.CLARIFICATION,source='LOCAL_FAIL_CLOSED',confidence=1),self.explain())
        root = self.root(c)
        c.handle('explanation question',turn_id='bad-resolution')
        self.assertEqual(c.last_runtime_outcome.status.value,'CLARIFICATION')
        self.assertEqual(c.last_runtime_outcome.failure_ref,'')
        self.assertEqual(c.state.failure_topic_record(),root)
        c.handle('more detail',turn_id='follow-up')
        self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)

    def test_exact_control_duplicate_normalized_no_repair(self):
        calls=[]
        resolver=InteractionResolver(lambda request: calls.append(request) or self.explain())
        value=resolver.resolve('explain',state={'failure_available':True,'pending_action':False,
            'retry_available':False,'active_failure':{'failure_id':'failure:root'}})
        self.assertEqual(value.requested_operations,('EXPLAIN',))
        self.assertEqual(value.failure_ref,'failure:root')
        self.assertEqual(len(calls),1)

    def test_repair_cannot_replace_explanation_with_action(self):
        values=iter([R(K.CLARIFICATION,requested_operations=('EXPLAIN',),confidence=1),
            R(K.TOOL_ACTION,requested_effects=('CREATE',),confidence=1)])
        calls=[]
        resolver=InteractionResolver(lambda req:calls.append(req) or next(values))
        value=resolver.resolve('discourse',state={'failure_available':True,'pending_action':False,
            'retry_available':False,'active_failure':{'failure_id':'root','relevant':True,'explained':True}})
        self.assertEqual(value.source,'LOCAL_FAIL_CLOSED')
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1]['preserve_explanation']['failure_ref'],'root')

    def test_real_effect_is_not_normalized_away(self):
        value=R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,
            requested_operations=('DELETE_ARTIFACT',),confidence=1)
        self.assertIn('non_action_has_effect_or_artifact',InteractionResolver.contradictions(value,
            {'failure_available':True,'retry_available':False,'pending_action':False}))

    def test_specific_detail_precedes_fallback(self):
        c=self.controller()
        root=self.root(c)
        text=FailureResponseRenderer.render(root,depth=1)
        self.assertIn('pruebas necesarias',text)
        self.assertNotIn('No tengo mas detalle',text)

    def test_no_detail_uses_honest_fallback(self):
        root=FailureRecord.create(task_id='x',stage='PLAN_VALIDATION',error_code='UNKNOWN',
            safe_user_message='',retryable=False)
        self.assertIn('No tengo mas detalle',FailureResponseRenderer.render(root,depth=2))

    def test_unrelated_conversation_exits_topic(self):
        from anydes_tools.conversation.interaction_router import TurnRelation
        c=self.controller(R(K.CONVERSATION,confidence=1,turn_relation=TurnRelation.NEW_TOPIC))
        self.root(c)
        c.handle('tell me about astronomy',turn_id='unrelated')
        self.assertIsNone(c.state.failure_topic_record())
        self.assertEqual(c.state.active_failure_topic.topic_status,'DISMISSED')

    def test_new_task_failure_supersedes_topic_preserves_history(self):
        c=self.controller()
        first=self.root(c)
        second=self.root(c,'new-user-task')
        self.assertEqual(c.state.failure_topic_record(),second)
        self.assertIn(first,c.state.failure_history)

    def test_no_record_no_invented_explanation(self):
        c=self.controller(R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,confidence=1))
        c.handle('explain',turn_id='none')
        self.assertIsNone(c.state.active_failure_topic)
        self.assertFalse(c._executor_called)


if __name__ == '__main__':
    unittest.main()
