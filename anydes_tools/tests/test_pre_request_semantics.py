"""Prepared semantic branches with varied utterances; UI supplies live evidence."""
import unittest
from unittest.mock import patch

from anydes_tools.conversation.discourse_context import ConversationPurpose as P
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolution as R, InteractionKind as K, InteractionResolver, TaskControl as T
from anydes_tools.tests import test_conversational_core as core


def payload(intent, **fields):
    return dict(intent=intent, speech_act='SOCIAL', primary_goal='Prepared discourse purpose',
                confidence=.95, **fields)


STATE = dict(failure_available=False, pending_action=False, retry_available=False,
             resolution_context={'previous_assistant_act':'CONVERSATIONAL_RESPONSE',
                'discourse':{'subjects':[], 'topic':'casual conversation'}})


class PreRequestSemantics(unittest.TestCase):
    def test_paraphrased_openings_are_nonoperational_prepared_semantics(self):
        for turn in ('Me vendria bien hablar de una cosa contigo.',
                     'Tengo una consulta en mente; puedo contartela?',
                     'Hay un asunto para el que me gustaria contar contigo.'):
            with self.subTest(turn=turn):
                requests = []
                def classifier(request):
                    requests.append(request)
                    return R.from_dict(semantic_resolution_payload(payload('PRE_REQUEST',
                        conversation_purpose='PRE_REQUEST'), request['state']))
                decision = InteractionResolver(classifier).resolve(turn, state=STATE)
                self.assertEqual(decision.kind, K.CONVERSATION)
                self.assertEqual(decision.conversation_purpose, P.PRE_REQUEST)
                self.assertFalse(decision.requested_effects)
                self.assertFalse(decision.requires_execution)
                self.assertEqual(decision.task_control, T.NONE)
                self.assertEqual(requests[0]['user_request'], turn)

    def test_casual_chat_missing_parameter_action_and_failure_remain_distinct(self):
        cases = [
            ('Hoy estuvo agradable la tarde.', payload('CONVERSATION', conversation_purpose='GENERAL'), K.CONVERSATION, P.GENERAL),
            ('Compara esos dos, pero no te he dicho cuales.', payload('CLARIFICATION',
                ambiguities=['comparison targets absent']), K.CLARIFICATION, P.GENERAL),
            ('Prepara un texto sobre volcanes.', dict(intent='CREATE_NEW',speech_act='ACTION_REQUEST',
                primary_goal='Create a text about volcanoes', artifact_type='TEXT',confidence=.95), K.TOOL_ACTION, P.GENERAL),
            ('Que impidio que se completara?', dict(intent='EXPLAIN_FAILURE',speech_act='INFORMATION_REQUEST',
                primary_goal='Explain recorded cause',confidence=.95), K.CONTENT_RESPONSE, P.GENERAL),
        ]
        for turn, raw, kind, purpose in cases:
            with self.subTest(turn=turn):
                state = {**STATE, 'failure_available':True, 'active_failure':{'failure_id':'root'}}
                resolution = InteractionResolver(lambda req:R.from_dict(
                    semantic_resolution_payload(raw, req['state']))).resolve(turn,state=state)
                self.assertEqual(resolution.kind, kind)
                self.assertEqual(resolution.conversation_purpose, purpose)
                if raw['intent'] == 'EXPLAIN_FAILURE':
                    self.assertEqual(resolution.task_control,T.EXPLAIN_FAILURE)

    def test_conflicting_general_intent_and_pre_request_purpose_get_one_semantic_repair(self):
        raw = [payload('CONVERSATION',conversation_purpose='PRE_REQUEST'),
               payload('PRE_REQUEST',conversation_purpose='PRE_REQUEST')]
        requests = []
        def classifier(request):
            requests.append(request)
            return R.from_dict(semantic_resolution_payload(raw[len(requests)-1],request['state']))
        resolver = InteractionResolver(classifier)
        result = resolver.resolve('Quiero reservar un momento para una consulta.',state=STATE)
        self.assertEqual(result.conversation_purpose,P.PRE_REQUEST)
        self.assertEqual(len(requests),2)
        self.assertEqual(requests[0]['user_request'],requests[1]['user_request'])
        self.assertEqual(requests[0]['state'],requests[1]['state'])
        self.assertIn('pre_request_contract',requests[1])
        self.assertIn('pre_request_intent_required',str(requests[1]['repair_errors']))

    def test_repeated_conflict_fails_closed_after_two_calls_not_a_loop(self):
        requests=[]
        def classifier(request):
            requests.append(request)
            return R.from_dict(semantic_resolution_payload(payload('CONVERSATION',
                conversation_purpose='PRE_REQUEST'),request['state']))
        result=InteractionResolver(classifier).resolve('An opening to a future request',state=STATE)
        self.assertEqual(len(requests),2)
        self.assertEqual(result.kind,K.CLARIFICATION)
        self.assertFalse(result.requested_effects)

    def test_invalid_purpose_and_pre_request_effects_stay_strict(self):
        for extra in ({'conversation_purpose':[]},{'conversation_purpose':'other'},
                      {'conversation_purpose':'GENERAL'},{'execute':True},
                      {'artifact_type':'CODE'},{'references':['existing.py']}):
            with self.subTest(extra=extra), self.assertRaises(ResolutionSchemaError):
                semantic_resolution_payload(payload('PRE_REQUEST',**extra),STATE)

    def test_old_artifact_focus_does_not_create_an_operation_in_a_pre_request(self):
        state={**STATE,'resolution_context':{**STATE['resolution_context'],
            'conversation_artifact_focus':[{'artifact_id':'existing','artifact_type':'CODE'}]}}
        result=R.from_dict(semantic_resolution_payload(payload('PRE_REQUEST',
            conversation_purpose='PRE_REQUEST'),state))
        self.assertEqual(result.conversation_purpose,P.PRE_REQUEST)
        self.assertFalse(result.requested_effects)
        self.assertFalse(result.references)

    def test_controller_returns_to_existing_natural_conversation_without_planner_or_tools(self):
        case=core.DomainController()
        case.setUp()
        try:
            decision=R.from_dict(semantic_resolution_payload(payload('PRE_REQUEST',
                conversation_purpose='PRE_REQUEST'),STATE))
            controller=case.controller(decision)
            result=controller.handle('An opening without a concrete task',turn_id='pre')
            self.assertIsNone(result)
            self.assertFalse(controller._executor_called)
            self.assertFalse(controller._reasoner_called)
        finally:
            case.tearDown()
        for text in ('Cuentame, te escucho.', 'Claro, que tienes en mente?'):
            with patch.object(core, 'resolved', return_value=decision):
                outcome,validation,events=core.DomainContracts().adapter('PRE_REQUEST',
                    {'text':text,'offer':None})
            self.assertEqual(outcome.text,text)
            self.assertTrue(validation.valid,validation.issues)


if __name__=='__main__':
    unittest.main()
