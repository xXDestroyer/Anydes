"""Prepared semantic contracts; real controller and publication, no operational planner for queries."""
from dataclasses import replace
import unittest
from anydes_tools.tests import test_failure_topic_preservation as prior
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T,
    TurnRelation as D, InteractionResolver)
from anydes_tools.conversation.failure_state import FailureRecord, RemediationAssessment
from anydes_tools.conversation.publication import FinalResponseComposer


class RemediationIntent(unittest.TestCase):
    setUp=prior.FailureTopic.setUp
    tearDown=prior.FailureTopic.tearDown
    controller=prior.FailureTopic.controller
    root=prior.FailureTopic.root

    def query(self):
        return R(K.CONTENT_RESPONSE, task_control=T.REMEDIATION_QUERY,
            turn_relation=D.FAILURE_REMEDIATION, requested_operations=('EXPLAIN',),confidence=.85)

    def state(self):
        return {'failure_available':True,'pending_action':False,'retry_available':True,
            'active_failure':{'failure_id':'root','relevant':True,'explained':True}}

    def test_remediation_paraphrases_readonly_same_root(self):
        texts=('Could this be fixed?', 'What options are available?', 'Is there a way to address this?')
        c=self.controller(*(self.query() for _ in texts))
        root=self.root(c)
        for i,text in enumerate(texts):
            response=c.handle(text,turn_id=str(i))
            self.assertEqual(c.last_resolution.task_control,T.REMEDIATION_QUERY)
            self.assertEqual(c.last_resolution.failure_ref,root.failure_id)
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertEqual(response.debug['remediation_assessment']['failure_id'],root.failure_id)
            self.assertFalse(c._reasoner_called)
            self.assertFalse(c._executor_called)
            self.assertEqual(c.state.failure_topic_record(),root)
            final=FinalResponseComposer.compose(c.last_runtime_outcome)
            self.assertEqual(final.visible_text,final.spoken_text)

    def test_retry_corrective_and_queries_are_distinct(self):
        cases=[('Try the same task again',R(K.TOOL_ACTION,task_control=T.RETRY,
            turn_relation=D.FAILED_TASK_RETRY,confidence=.85)),
            ('Fix it',R(K.TOOL_ACTION,requested_effects=('EDIT',),
                turn_relation=D.NEW_CORRECTIVE_ACTION,confidence=.85)),
            ('Why did that fail',R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,
                turn_relation=D.FAILURE_CAUSE,confidence=.85)),
            ('Explain what that means',R(K.CONTENT_RESPONSE,task_control=T.CONTINUE_EXPLANATION,
                turn_relation=D.FAILURE_DETAIL,confidence=.85))]
        for text,value in cases:
            resolver=InteractionResolver(lambda request:value)
            final=resolver.resolve(text,state=self.state())
            self.assertEqual(final.task_control,value.task_control)
            self.assertEqual(final.turn_relation,value.turn_relation)
            self.assertNotEqual(final.task_control,T.REMEDIATION_QUERY)

    def test_remediation_effects_rejected_not_normalized_away(self):
        for field in ({'requested_effects':('EDIT',)}, {'requires_execution':True},
                {'kind':K.TOOL_ACTION}, {'artifact_intent':True}):
            value=replace(self.query(),**field)
            resolver=InteractionResolver(lambda request:value)
            self.assertEqual(resolver.resolve('options',state=self.state()).source,'LOCAL_FAIL_CLOSED')
            self.assertEqual(len(resolver.validation_history),2)

    def test_one_repair_preserves_query_target(self):
        values=iter([replace(self.query(),task_control=T.NONE),self.query()])
        calls=[]
        resolver=InteractionResolver(lambda req:calls.append(req) or next(values))
        result=resolver.resolve('What could help?',state=self.state())
        self.assertEqual(result.task_control,T.REMEDIATION_QUERY)
        self.assertEqual(calls[1]['preserve_explanation'],
            {'failure_ref':'root','task_control':'REMEDIATION_QUERY'})
        self.assertEqual(len(calls),2)

    def test_no_failure_does_not_invent_remediation(self):
        c=self.controller(self.query(),self.query())
        c.handle('Would there be a solution?',turn_id='no-root')
        self.assertIsNone(c.state.active_failure_topic)
        self.assertFalse(c._executor_called)

    def test_unrelated_turn_exits_context(self):
        c=self.controller(self.query(),R(K.CONTENT_RESPONSE,turn_relation=D.NEW_TOPIC,confidence=.85))
        self.root(c)
        c.handle('Are there options?',turn_id='query')
        c.handle('Explain photosynthesis',turn_id='unrelated')
        self.assertIsNone(c.state.failure_topic_record())

    def test_assessment_unknown_oracle_is_not_fix_promise(self):
        failure=FailureRecord.create(task_id='t',stage='VERIFY',error_code='ORACLE_UNVERIFIED',
            safe_user_message='',retryable=True)
        result=RemediationAssessment.assess(failure,retry_available=True)
        self.assertEqual(result.remediation_possible,'UNKNOWN')
        self.assertEqual(result.corrective_action_available,'UNKNOWN')
        self.assertTrue(result.retry_possible)
        self.assertIn('referencia independiente',result.safe_reason)
        self.assertNotIn('funciona correctamente',result.safe_reason)

    def test_assessment_retry_budget_and_known_constraint(self):
        c=self.controller()
        root=self.root(c)
        result=RemediationAssessment.assess(root,retry_available=True)
        self.assertEqual(result.remediation_possible,'POSSIBLE')
        self.assertFalse(result.retry_possible)
        self.assertIn('validacion',result.safe_reason)
        uncertain=replace(root,error_code='UNRECOGNIZED',validation_detail={})
        self.assertEqual(RemediationAssessment.assess(uncertain).remediation_possible,'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
