"""Prepared domain decisions; real controller/context/validator, not human evidence."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'programa_principal'))

from anydes_tools.tests import test_context_control_resolution as prior
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T,
    TurnRelation as D, InteractionResolver)
from anydes_tools.conversation.operation_intent import semantic_resolution_payload
from anydes_tools.conversation.discourse_context import ConversationPurpose as P, DiscourseContext
from anydes_tools.conversation.conversational_bridge import resolved_conversation, identity_evidence


def resolved(intent, **fields):
    if intent == 'PERSONAL_IDENTITY':
        fields.setdefault('identity_attribute','DISPLAY_NAME')
    return R.from_dict(semantic_resolution_payload(dict(intent=intent,
        speech_act='INFORMATION_REQUEST',primary_goal='Current semantic question',confidence=.97,**fields),{}))


class DomainController(unittest.TestCase):
    setUp = prior.ContextControl.setUp
    tearDown = prior.ContextControl.tearDown
    controller = prior.ContextControl.controller
    root = prior.ContextControl.root

    def test_explicit_conversational_domain_exits_failure_even_when_chat_continues(self):
        for intent in ('SELF_PREFERENCE','SELF_RELATIONAL','PERSONAL_IDENTITY','PERSONAL_MEMORY_RECALL','ENTITY_QUERY'):
            with self.subTest(intent=intent):
                c=self.controller(resolved(intent,turn_relation='TOPIC_CONTINUATION'))
                root=self.root(c)
                c.handle('A different personal question',turn_id=intent)
                self.assertIsNone(c.state.failure_topic_record())
                self.assertIn(root.failure_id, [f.failure_id for f in c.state.failure_history])
                self.assertFalse(c._executor_called)
                self.assertFalse(c._reasoner_called)

    def test_new_question_resolution_error_does_not_become_old_failure_explanation(self):
        c=self.controller()
        old=self.root(c)
        c.interaction_resolver.classifier=lambda req: (_ for _ in ()).throw(ValueError('bad_shape'))
        result=c.handle('A new personal question the classifier could not resolve',turn_id='new')
        self.assertNotEqual(result.debug.get('response_source'),'EXPLANATION_SECONDARY_FAILURE')
        self.assertNotEqual(c.state.previous_assistant_act,'FAILURE_EXPLANATION_UNAVAILABLE')
        self.assertFalse(c._executor_called)

    def test_actual_failure_cause_and_remediation_stay_grounded(self):
        c=self.controller(resolved('EXPLAIN_FAILURE'),resolved('REMEDIATION_QUERY'))
        root=self.root(c)
        for turn in ('Cause of that failure?', 'Could it be remedied?'):
            c.handle(turn,turn_id=turn)
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertFalse(c._executor_called)
            self.assertFalse(c._reasoner_called)

    def test_unrelated_content_exits_history_survives(self):
        c=self.controller(resolved('CONTENT_RESPONSE',turn_relation='NEW_TOPIC'))
        root=self.root(c)
        c.handle('How are rainbows formed?',turn_id='external')
        self.assertIsNone(c.state.failure_topic_record())
        self.assertTrue(c.state.failure_history)


class DomainContracts(unittest.TestCase):
    def test_subjective_domains_have_no_effects_or_authority(self):
        for intent,purpose in [('SELF_PREFERENCE',P.SELF_PREFERENCE),('SELF_RELATIONAL',P.SELF_RELATIONAL),
                               ('PERSONAL_IDENTITY',P.PERSONAL_IDENTITY)]:
            decision=resolved(intent)
            self.assertEqual(decision.conversation_purpose,purpose)
            self.assertFalse(decision.requested_effects)
            self.assertFalse(decision.requires_execution)
            self.assertEqual(decision.task_control,T.NONE)
            self.assertFalse(decision.requested_operations)

    def test_identity_attribute_is_typed_not_a_value_supplied_by_model(self):
        self.assertEqual(resolved('PERSONAL_IDENTITY',identity_attribute='DISPLAY_NAME').identity_attribute,'DISPLAY_NAME')
        for intent,attribute in [('PERSONAL_IDENTITY','arbitrary'),('SELF_PREFERENCE','DISPLAY_NAME')]:
            with self.assertRaises(ValueError):
                resolved(intent,identity_attribute=attribute)

    def test_authorized_identity_grounded_without_broad_memory_or_hardcoded_value(self):
        for name in ('Test Person One','Test Person Two'):
            fact,status=identity_evidence(NS(config=NS(user_name=name)),'DISPLAY_NAME')
            self.assertEqual(status,'GROUNDED')
            self.assertEqual(fact['value'],name)
            self.assertEqual(fact['source'],'AUTHORIZED_USER_CONTEXT')
        for assistant,attribute in [(NS(config=NS()),'DISPLAY_NAME'),
                (NS(config=NS(user_name='private'),speaker_allows_memory_context=False),'DISPLAY_NAME'),
                (NS(config=NS(user_name='known')),'UNSPECIFIED')]:
            self.assertIsNone(identity_evidence(assistant,attribute)[0])

    def adapter(self,intent,reply,**fields):
        from anydes.realtime.transport import StreamOutcome
        from anydes.realtime.context import RealtimeContext
        from anydes.realtime.validation import RealtimeOutputValidator
        context=RealtimeContext(system_prompt='',turn_prompt='',user_text='Current question',
            recent_turn_count=0,memory_fact_count=0,self_preference_count=0,
            recent_evidence_match=False,asks_self_preference=True,asks_user_preference=False,
            user_states_preference=True,user_correction=False,approximate_input_tokens=0)
        events=[]
        def stream(messages,**kwargs):
            from anydes_tools.tests.test_natural_response import review_payload
            value = review_payload() if messages and messages[-1]['role']=='user' else reply
            return StreamOutcome(json.dumps(value),0,0,0,0,0,0,0,0,0,'test')
        service=NS(router=NS(classify=lambda *a:None),local_routes=NS(resolve=lambda *a:None),
            session=NS(resolve_reference=lambda *a:None),context_builder=NS(build=lambda *a,**k:context),
            transport=NS(stream=stream),validator=RealtimeOutputValidator(),
            phase2_enabled=True)
        assistant=NS(config=NS(user_name=fields.pop('user_name','Example User')),
            _realtime_service=service,audit=NS(record=lambda event,**kw:events.append((event,kw))))
        controller=NS(last_resolution=resolved(intent,**fields),discourse=DiscourseContext(),
            project_root=Path(__file__).resolve().parents[2],registry=NS(list_capabilities=lambda **kw:()))
        with resolved_conversation(assistant,controller):
            built=service.context_builder.build('Current question')
            outcome=service.transport.stream([])
            validation=service.validator.validate(outcome.text,built)
        return outcome,validation,events

    def test_tentative_preference_survives_real_validator_without_fake_persisted_self(self):
        outcome,validation,events=self.adapter('SELF_PREFERENCE',
            {'text':'Creo que elegiria el verde. Me atrae su serenidad.','offer':None})
        self.assertTrue(validation.valid,validation.issues)
        self.assertEqual(outcome.text,'Creo que elegiria el verde. Me atrae su serenidad.')
        self.assertNotIn('self_preference_unverified',validation.issues)
        self.assertFalse(events[-1][1]['persistent_preference_written'])

    def test_relational_assessment_is_framed_as_impression_not_historical_fact(self):
        outcome,validation,events=self.adapter('SELF_RELATIONAL',
            {'text':'Destaca tu curiosidad en nuestras preguntas; es mi impresion de esta charla.','offer':None})
        self.assertTrue(validation.valid,validation.issues)
        self.assertIn('esta charla',outcome.text)
        self.assertIn('curiosidad',outcome.text)
        self.assertFalse(events[-1][1]['tools_dispatched'])

    def test_identity_response_is_exact_host_evidence_not_model_prose(self):
        outcome,validation,events=self.adapter('PERSONAL_IDENTITY',{'text':'Invented name'},
            identity_attribute='DISPLAY_NAME',user_name='Authorized Example')
        self.assertIn('Authorized Example',outcome.text)
        self.assertNotIn('Invented',outcome.text)
        self.assertEqual(next(detail for event,detail in events if event=='conversation_identity_grounding')['status'],'GROUNDED')
        self.assertNotIn('Authorized Example',str(events))

    def test_subjective_domain_does_not_bypass_action_claim_validation(self):
        with self.assertRaisesRegex(ValueError,'natural_response_local_invariant_rejected'):
            self.adapter('SELF_RELATIONAL',
                {'text':'Ya lo ejecute; he ejecutado el programa.','offer':None})


if __name__=='__main__':
    unittest.main()
