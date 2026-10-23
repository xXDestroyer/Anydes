"""Prepared discourse decisions; real contract/controller/publication, not human E2E."""
from copy import deepcopy
import unittest

from anydes_tools.tests import test_failure_topic_preservation as prior
from anydes_tools.tests.test_operation_grounding import resolved, semantic
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, ResolutionSchemaError
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionResolver, InteractionKind as K,
    TaskControl as T, ExplanationRelation as E)
from anydes_tools.conversation.publication import FinalResponseComposer


def context():
    return {'failure_available':True, 'pending_action':False, 'retry_available':True,
        'active_failure':{'failure_id':'root', 'relevant':True, 'explained':True},
        'resolution_context':{'active_failure_topic':{'failure_id':'root',
            'root_failure_id':'root', 'explanation_depth':1},
            'previous_assistant_act':'FAILURE_EXPLANATION'}}


class ContinuationContract(unittest.TestCase):
    def test_historical_ambiguity_without_relation_requires_one_repair(self):
        calls = []

        def classify(request):
            calls.append(deepcopy(request))
            return resolved('CLARIFICATION') if len(calls)==1 else resolved(
                'CONTINUE_EXPLANATION', explanation_relation=E.ELABORATION.value)

        resolver = InteractionResolver(classify)
        result = resolver.resolve('Could you unpack the implications of that?', state=context())
        self.assertEqual(result.task_control,T.CONTINUE_EXPLANATION)
        self.assertEqual(result.failure_ref,'root')
        self.assertEqual(len(calls),2)
        self.assertIn('explanation_followup_relation_required',calls[1]['repair_errors'])
        self.assertEqual(calls[1]['continuation_context']['previous_assistant_act'],'FAILURE_EXPLANATION')
        self.assertEqual(calls[1]['continuation_context']['failure_ref'],'root')
        self.assertIn('NEW_ACTION',calls[1]['continuation_context']['allowed_discourse_relations'])

    def test_semantic_elaboration_cannot_be_clarification(self):
        value = resolved('CLARIFICATION',explanation_relation=E.ELABORATION.value)
        self.assertIn('explanation_semantic_relation_conflict',
            InteractionResolver.contradictions(value,context()))

    def test_typed_relation_does_not_parse_user_phrases(self):
        for text in ('What does that imply for the request?', 'Unpack the limitation a little.',
                     'Could you give the practical consequence?'):
            with self.subTest(text=text):
                value = resolved('CONTINUE_EXPLANATION',explanation_relation=E.ELABORATION.value)
                result = InteractionResolver(lambda _:value).resolve(text,state=context())
                self.assertEqual(result.task_control,T.CONTINUE_EXPLANATION)
                self.assertEqual(result.failure_ref,'root')
                self.assertFalse(result.requires_execution or result.requested_effects)

    def test_genuine_ambiguity_with_basis_is_not_coerced(self):
        value = resolved('CLARIFICATION',explanation_relation=E.AMBIGUOUS.value,
            ambiguities=['Two failures were explicitly contrasted; requested one is unresolved'])
        resolver = InteractionResolver(lambda _:value)
        result = resolver.resolve('Which of those?',state=context())
        self.assertEqual(result.kind,K.CLARIFICATION)
        self.assertEqual(len(resolver.validation_history),1)

    def test_ambiguity_needs_a_basis_after_explanation(self):
        value = resolved('CLARIFICATION',explanation_relation=E.AMBIGUOUS.value)
        self.assertIn('explanation_ambiguity_basis_required',
            InteractionResolver.contradictions(value,context()))

    def test_repair_budget_not_extended(self):
        calls = []
        resolver = InteractionResolver(lambda request:calls.append(request) or resolved('CLARIFICATION'))
        result = resolver.resolve('Contextual depth question',state=context())
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertEqual(len(calls),2)

    def test_explicit_operation_and_retry_remain_independent(self):
        for intent in ('EXECUTE_EXISTING','RETRY'):
            with self.subTest(intent=intent):
                resolver = InteractionResolver(lambda _:resolved(intent))
                result = resolver.resolve('A prepared explicit action',state=context())
                self.assertEqual(result.kind,K.TOOL_ACTION)
                self.assertNotIn(result.task_control,{T.CONTINUE_EXPLANATION,T.REMEDIATION_QUERY})
                self.assertEqual(len(resolver.validation_history),1)

    def test_no_failure_vague_turn_does_not_invent_reference(self):
        state = {**context(),'failure_available':False,'active_failure':{},'resolution_context':{}}
        result = InteractionResolver(lambda _:resolved('CLARIFICATION')).resolve('And that?',state=state)
        self.assertEqual(result.kind,K.CLARIFICATION)
        self.assertEqual(result.failure_ref,'')

    def test_relation_schema_is_strict(self):
        with self.assertRaises(ResolutionSchemaError):
            semantic_resolution_payload(semantic('CONTINUE_EXPLANATION',explanation_relation='free text'),context())


class ContinuationIntegration(unittest.TestCase):
    setUp = prior.FailureTopic.setUp
    tearDown = prior.FailureTopic.tearDown
    controller = prior.FailureTopic.controller
    root = prior.FailureTopic.root

    def test_cause_depth_remediation_exit_same_root_no_tools(self):
        c = self.controller(
            resolved('EXPLAIN_FAILURE',explanation_relation=E.CAUSE.value),
            resolved('CONTINUE_EXPLANATION',explanation_relation=E.ELABORATION.value),
            resolved('REMEDIATION_QUERY',explanation_relation=E.REMEDIATION.value),
            resolved('CONTENT_RESPONSE',explanation_relation=E.TOPIC_CHANGE.value))
        root = self.root(c)
        texts = []
        for i,text in enumerate(('What prevented completion?', 'What does that mean practically?',
                                 'Is repair possible with what is available?')):
            c.handle(text,turn_id=str(i))
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertEqual(c.state.active_failure_topic.root_failure_id,root.failure_id)
            self.assertEqual(c.state.active_failure_topic.explanation_depth,min(i+1,2))
            self.assertFalse(c._executor_called or c._reasoner_called)
            final = FinalResponseComposer.compose(c.last_runtime_outcome)
            self.assertEqual(final.visible_text,final.spoken_text)
            texts.append(final.visible_text)
        self.assertNotEqual(texts[0],texts[1])
        self.assertIn('ejecutar el programa',texts[1])
        self.assertIn('antes de ejecutar cambios',texts[1])
        c.handle('Describe a rainbow',turn_id='new-topic')
        self.assertIsNone(c.state.failure_topic_record())

    def test_secondary_renderer_failure_does_not_replace_root(self):
        from unittest.mock import patch
        from anydes_tools.conversation.failure_state import FailureResponseRenderer
        c = self.controller(resolved('EXPLAIN_FAILURE'),resolved('CONTINUE_EXPLANATION'))
        root = self.root(c)
        with patch.object(FailureResponseRenderer,'render',side_effect=ValueError('private')):
            c.handle('Cause',turn_id='first')
        c.handle('More context',turn_id='depth')
        self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
        self.assertFalse(c._executor_called or c._reasoner_called)


if __name__ == '__main__':
    unittest.main()
