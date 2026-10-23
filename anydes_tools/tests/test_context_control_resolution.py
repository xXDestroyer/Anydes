"""Prepared semantic decisions, real resolution/controller/publication. Not human E2E."""
from dataclasses import replace
import unittest
from anydes_tools.tests import test_failure_topic_preservation as prior
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T,
    TurnRelation as D, InteractionResolver)
from anydes_tools.conversation.publication import FinalResponseComposer


class ContextControl(unittest.TestCase):
    setUp=prior.FailureTopic.setUp
    tearDown=prior.FailureTopic.tearDown
    controller=prior.FailureTopic.controller
    root=prior.FailureTopic.root

    def decision(self, relation, control=T.NONE, kind=K.CONTENT_RESPONSE):
        return R(kind,task_control=control,turn_relation=relation,
            requested_operations=('EXPLAIN',),confidence=.85)

    def state(self):
        return {'failure_available':True,'pending_action':True,'retry_available':True,
            'active_failure':{'failure_id':'root','relevant':True,'explained':False}}

    def test_causal_relation_without_control_requires_one_repair(self):
        for text in ('How did this go wrong?', 'What prevented completion?', 'Can you explain the cause?'):
            with self.subTest(text=text):
                values=iter([self.decision(D.FAILURE_CAUSE),self.decision(D.FAILURE_CAUSE,T.EXPLAIN_FAILURE)])
                calls=[]
                resolver=InteractionResolver(lambda req:calls.append(req) or next(values))
                result=resolver.resolve(text,state=self.state())
                self.assertEqual(result.task_control,T.EXPLAIN_FAILURE)
                self.assertEqual(result.failure_ref,'root')
                self.assertEqual(len(calls),2)
                self.assertIn('relation_control_conflict',calls[1]['repair_errors'])

    def test_original_missing_relation_not_accepted_as_context_exit(self):
        values=iter([R(K.CONTENT_RESPONSE,requested_operations=('EXPLAIN',),confidence=.85),
            self.decision(D.FAILURE_CAUSE,T.EXPLAIN_FAILURE)])
        resolver=InteractionResolver(lambda req:next(values))
        result=resolver.resolve('What happened there?',state=self.state())
        self.assertEqual(result.task_control,T.EXPLAIN_FAILURE)
        self.assertEqual(len(resolver.validation_history),2)

    def test_exhausted_semantic_repair_preserves_root(self):
        c=self.controller(self.decision(D.FAILURE_CAUSE),self.decision(D.FAILURE_CAUSE))
        root=self.root(c)
        c.handle('Explain what stopped it',turn_id='rejected')
        self.assertEqual(c.state.failure_topic_record(),root)
        self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
        self.assertEqual(len(c.interaction_resolver.validation_history),2)
        self.assertFalse(c._executor_called)

    def test_detail_and_remediation_same_reference_and_previous_act(self):
        c=self.controller(self.decision(D.FAILURE_CAUSE,T.EXPLAIN_FAILURE),
            self.decision(D.FAILURE_DETAIL,T.CONTINUE_EXPLANATION),
            self.decision(D.FAILURE_REMEDIATION,T.REMEDIATION_QUERY))
        root=self.root(c)
        c.state.previous_assistant_act='ACTION_FAILED'
        for i,text in enumerate(('What stopped it?', 'Could you unpack that?', 'Is there a way to address it?')):
            c.handle(text,turn_id=str(i))
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertEqual(c.state.active_failure_topic.explanation_depth,min(i+1,2))
            self.assertFalse(c._executor_called)
            self.assertFalse(c._reasoner_called)
            self.assertEqual(c.state.previous_assistant_act,
                'FAILURE_REMEDIATION_RESPONSE' if i == 2 else 'FAILURE_EXPLANATION')
            final=FinalResponseComposer.compose(c.last_runtime_outcome)
            self.assertEqual(final.visible_text,final.spoken_text)
        calls=c.interaction_resolver.classifier.calls
        self.assertEqual(calls[0]['state']['resolution_context']['previous_assistant_act'],'ACTION_FAILED')
        self.assertEqual(calls[1]['state']['resolution_context']['previous_assistant_act'],'FAILURE_EXPLANATION')

    def test_unrelated_explanation_clears_only_after_outcome(self):
        c=self.controller()
        root=self.root(c)
        observations=[]
        def classify(request):
            observations.append(c.state.failure_topic_record())
            self.assertEqual(request['state']['resolution_context']['active_failure_topic']['failure_id'],root.failure_id)
            return self.decision(D.NEW_TOPIC)
        c.interaction_resolver.classifier=classify
        c.handle('Explain what a star is',turn_id='unrelated')
        self.assertEqual(observations,[root])
        self.assertIsNone(c.state.failure_topic_record())
        self.assertEqual(c.state.active_failure_topic.topic_status,'DISMISSED')

    def test_ambiguous_relation_keeps_topic(self):
        c=self.controller(self.decision(D.AMBIGUOUS,kind=K.CLARIFICATION))
        root=self.root(c)
        c.handle('Could you clarify that part?',turn_id='ambiguous')
        self.assertEqual(c.state.failure_topic_record(),root)
        self.assertFalse(c._executor_called)

    def test_no_failure_does_not_invent_reference(self):
        state={**self.state(),'failure_available':False,'active_failure':{}}
        resolver=InteractionResolver(lambda req:self.decision(D.FAILURE_CAUSE,T.EXPLAIN_FAILURE))
        result=resolver.resolve('What caused that?',state=state)
        self.assertEqual(result.kind,K.CLARIFICATION)
        self.assertEqual(result.failure_ref,'')

    def test_multiple_contexts_semantic_relation_selects_not_fixed_priority(self):
        choices=[self.decision(D.FAILURE_DETAIL,T.CONTINUE_EXPLANATION),
            R(K.TOOL_ACTION,task_control=T.CONFIRM,turn_relation=D.PENDING_CONFIRMATION,confidence=.85),
            R(K.TOOL_ACTION,requested_effects=('EDIT',),turn_relation=D.ARTIFACT_FOLLOWUP,confidence=.85),
            R(K.TOOL_ACTION,requested_effects=('CREATE',),turn_relation=D.NEW_ACTION,confidence=.85)]
        for value in choices:
            resolver=InteractionResolver(lambda req:value)
            result=resolver.resolve('Prepared discourse decision',state=self.state())
            self.assertEqual(result,value if value.task_control not in {T.EXPLAIN_FAILURE,T.CONTINUE_EXPLANATION}
                else replace(value,failure_ref='root'))
            self.assertEqual(len(resolver.validation_history),1)

    def test_affirmative_relation_needs_real_pending_confirmation(self):
        value=R(K.TOOL_ACTION,task_control=T.CONFIRM,turn_relation=D.PENDING_CONFIRMATION,confidence=.85)
        for text in ('Go ahead with that', 'Yes, please proceed', 'You have my approval'):
            resolver=InteractionResolver(lambda req:value)
            self.assertEqual(resolver.resolve(text,state=self.state()).task_control,T.CONFIRM)
        resolver=InteractionResolver(lambda req:value)
        self.assertEqual(resolver.resolve('approval',state={**self.state(),'pending_action':False}).kind,K.CLARIFICATION)

    def test_snapshot_generic_artifact_focus_and_no_content_or_paths(self):
        from anydes_tools.tests.test_human_acceptance_round1 import Acceptance
        c=self.controller()
        first=Acceptance.source(self,c,'first.py')
        latest=Acceptance.source(self,c,'latest.py')
        snapshot=c._context_snapshot().to_dict()
        self.assertEqual(snapshot['conversation_artifact_focus'][0]['artifact_id'],latest.artifact_id)
        self.assertNotEqual(first.artifact_id,latest.artifact_id)
        self.assertNotIn(str(self.workspace),str(snapshot))

    def test_relation_enum_is_strict(self):
        with self.assertRaises(ValueError):
            R.from_dict({'kind':'CONTENT_RESPONSE','turn_relation':'probably explanation'})


if __name__ == '__main__':
    unittest.main()
