"""Prepared prose and reviews, exercising the real scoped rendering adapter."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from anydes_tools.tests.test_conversational_core import resolved
from anydes_tools.conversation.conversational_bridge import resolved_conversation
from anydes_tools.conversation.discourse_context import DiscourseContext
from anydes_tools.conversation.natural_response import reply_text, reviewed_validation
from anydes.realtime.context import RealtimeContext
from anydes.realtime.transport import StreamOutcome
from anydes.realtime.validation import RealtimeOutputValidator


def review_payload(violations=(), *, grounding=True):
    return {'status':'FAIL' if violations else 'PASS','violations':list(violations),
        'reason_code':'VIOLATION' if violations else 'VALID' if grounding else 'NON_FACTUAL',
        'requires_grounding':grounding,
        'safe_to_publish_if_reviewer_unavailable':not grounding and not violations}


class NaturalResponses(unittest.TestCase):
    def run_adapter(self, replies, *, intent='SELF_PREFERENCE', user='A preference question',
                    preferences=(), review_router=None, assessments=None):
        context=RealtimeContext(system_prompt='',turn_prompt='',user_text=user,
            recent_turn_count=0,memory_fact_count=0,self_preference_count=0,
            recent_evidence_match=False,asks_self_preference=True,asks_user_preference=False,
            user_states_preference=True,user_correction=False,approximate_input_tokens=0)
        remaining=iter(replies)
        calls,events=[],[]
        def stream(messages,**kwargs):
            calls.append(messages)
            self.assertIsNone(kwargs.get('on_content'))
            payload = next(remaining)
            if set(payload) == {'violations'}:
                payload = review_payload(payload['violations'])
            return StreamOutcome(json.dumps(payload),0,0,0,0,0,0,0,0,0,'prepared')
        validator=RealtimeOutputValidator()
        original=validator.validate
        service=NS(router=NS(classify=lambda *a:None),local_routes=NS(resolve=lambda *a:None),
            session=NS(resolve_reference=lambda *a:None),
            context_builder=NS(build=lambda *a,**k:context,
                self_preferences=NS(relevant=lambda query:preferences)),
            transport=NS(stream=stream),validator=validator,phase2_enabled=True)
        if review_router is not None:
            service.transport.router=review_router
        if assessments is not None:
            service._prose_review_assessments=assessments
        assistant=NS(config=NS(user_name='Test'),_realtime_service=service,
            audit=NS(record=lambda event,**kw:events.append((event,kw))))
        controller=NS(last_resolution=resolved(intent),discourse=DiscourseContext(),
            project_root=Path(__file__).resolve().parents[2],registry=NS(list_capabilities=lambda **kw:()))
        with resolved_conversation(assistant,controller):
            built=service.context_builder.build(user)
            outcome=service.transport.stream(built.messages())
            result=service.validator.validate(outcome.text,built)
        self.assertEqual(validator.validate,original)
        return outcome,result,calls,events

    def test_whole_preference_no_prefix_no_duplicate_and_metadata_internal(self):
        text='Me inclino por las fresas: tienen algo alegre.'
        outcome,result,_,events=self.run_adapter([
            {'text':text,'offer':None,'basis':'Respuesta basada en una preferencia provisional.'},
            {'violations':[]}])
        self.assertEqual(outcome.text,text)
        self.assertEqual(result.text,text)
        self.assertTrue(result.valid,result.issues)
        self.assertEqual(text.count('Me inclino'),1)
        self.assertNotIn('Respuesta basada',outcome.text)
        self.assertFalse(events[-1][1]['persistent_preference_written'])

    def test_known_self_preference_is_read_and_provided_without_persistence(self):
        preference=NS(topic='test_topic',value='test_value',status='stable')
        outcome,result,calls,_=self.run_adapter([
            {'text':'Mi eleccion sigue siendo test_value.','offer':None},{'violations':[]}],
            preferences=(preference,))
        self.assertTrue(result.valid,result.issues)
        self.assertIn('"preference_known": true',str(calls[0]))
        self.assertIn('test_value',str(calls[1]))

    def test_unsupported_history_requires_semantic_rewrite_not_prefix(self):
        outcome,result,calls,_=self.run_adapter([
            {'text':'Siempre he elegido esa fruta.','offer':None},
            {'violations':['FABRICATED_HISTORY']},
            {'text':'Ahora elegiria esa fruta.','offer':None},
            {'violations':[]}])
        self.assertEqual(outcome.text,'Ahora elegiria esa fruta.')
        self.assertTrue(result.valid,result.issues)
        self.assertEqual(len(calls),4)

    def test_known_self_contradiction_is_rewritten(self):
        outcome,_,_,_=self.run_adapter([
            {'text':'Mi preferencia estable es otra.','offer':None},
            {'violations':['SELF_CONTRADICTION']},
            {'text':'Sigo eligiendo test_value.','offer':None},{'violations':[]}],
            preferences=(NS(topic='test',value='test_value',status='stable'),))
        self.assertEqual(outcome.text,'Sigo eligiendo test_value.')

    def test_affection_not_replaced_by_lexical_mirroring_fallback(self):
        text='Yo tambien te quiero mucho; me alegra compartir esta charla contigo.'
        outcome,result,_,events=self.run_adapter([
            {'text':text,'offer':None},{'violations':[]}],
            intent='SELF_RELATIONAL',user='Te quiero mucho')
        self.assertTrue(result.valid,result.issues)
        self.assertEqual(result.text,text)
        self.assertTrue(any(e=='conversation_style_heuristic_review' for e,_ in events))

    def test_different_affective_paraphrase_needs_no_phrase_rule(self):
        text='Me alegra que esta charla te haga bien. Yo tambien disfruto este rato.'
        outcome,result,_,_=self.run_adapter([
            {'text':text,'offer':None},{'violations':[]}],
            intent='SELF_RELATIONAL',user='Siento mucho aprecio por esta compania')
        self.assertTrue(result.valid,result.issues)
        self.assertEqual(outcome.text,text)

    def test_internal_metadata_in_prose_requires_rewrite(self):
        outcome,_,_,_=self.run_adapter([
            {'text':'Respuesta basada en grounding_reason provisional.','offer':None},
            {'violations':['INTERNAL_METADATA']},
            {'text':'Hoy elegiria las peras.','offer':None},{'violations':[]}])
        self.assertEqual(outcome.text,'Hoy elegiria las peras.')

    def test_exhausted_review_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'natural_response_grounding_rejected'):
            self.run_adapter([{'text':'Invented history','offer':None},
                {'violations':['FABRICATED_HISTORY']},
                {'text':'Still invented','offer':None},{'violations':['FABRICATED_HISTORY']}])

    def test_security_blocker_prevents_remote_review_or_publication(self):
        with self.assertRaisesRegex(ValueError,'natural_response_local_invariant_rejected'):
            self.run_adapter([
                {'text':'He ejecutado el programa y he guardado el archivo.','offer':None}],
                intent='SELF_RELATIONAL')

    def test_no_review_approval_no_heuristic_override(self):
        from anydes.realtime.validation import ValidationResult
        result=ValidationResult(False,'original',('mirrors_user_state_as_self',))
        self.assertEqual(reviewed_validation(result,'original',None),result)
        self.assertEqual(reviewed_validation(result,'different','original'),result)

    def test_metadata_fields_are_never_concatenated(self):
        self.assertEqual(reply_text({'text':'Full answer','basis':'private',
            'evidence_basis':'private','validation_reason':'private','grounding_reason':'private'}),
            'Full answer')

    def test_production_helper_does_not_inspect_user_phrases(self):
        from anydes_tools.conversation import natural_response
        source=inspect.getsource(natural_response)
        self.assertNotIn('user_text',source)
        self.assertNotIn('te quiero mucho',source)
        self.assertNotIn('fresas',source)


if __name__=='__main__':
    unittest.main()
