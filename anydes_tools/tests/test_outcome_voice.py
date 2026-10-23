"""Prepared prose/reviews, real outcome projection, guards and publication hook."""
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace as NS
import unittest

from anydes_tools.tests.test_conversational_core import resolved
from anydes_tools.tests.test_natural_response import review_payload
from anydes_tools.conversation.operation_intent import semantic_resolution_payload
from anydes_tools.conversation.interaction_router import InteractionResolution,InteractionKind
from anydes_tools.conversation.discourse_context import DiscourseContext,ConversationPurpose
from anydes_tools.conversation.failure_state import FailureRecord
from anydes_tools.conversation.publication import RuntimeOutcome,OutcomeStatus as S,FinalResponseComposer
from anydes_tools.conversation.outcome_voice import (
    ResponseIntent,request_for,naturalize_final,generate_outcome_voice)
from anydes.realtime.validation import RealtimeOutputValidator
from anydes.realtime.transport import StreamOutcome


class OutcomeVoiceTests(unittest.TestCase):
    def controller(self, *, failure=None,goal='Revisar el contenido indicado',ambiguities=('material ausente',)):
        return NS(last_resolution=NS(primary_goal=goal,ambiguities=ambiguities),
            discourse=DiscourseContext(),state=NS(failure_history=[failure] if failure else [],
                active_failure_topic=NS(explanation_depth=2)))

    def assistant(self,replies):
        queue=iter(replies)
        calls,events=[],[]
        def stream(messages,**kwargs):
            calls.append(messages)
            value=next(queue)
            if isinstance(value,Exception):
                raise value
            return StreamOutcome(json.dumps(value),0,0,0,0,0,0,0,0,0,'prepared')
        a=NS(personality=NS(prompt_text='SELF fixture: warm, concise, attentive.'),
            _realtime_service=NS(transport=NS(stream=stream),validator=RealtimeOutputValidator()),
            audit=NS(record=lambda e,**k:events.append((e,k))))
        return a,calls,events

    def test_pre_request_is_typed_nonoperational_not_a_clarification(self):
        value=semantic_resolution_payload(dict(intent='PRE_REQUEST',speech_act='SOCIAL',
            primary_goal='Preparar una peticion posterior',confidence=.95),{})
        r=InteractionResolution.from_dict(value)
        self.assertEqual(r.kind,InteractionKind.CONVERSATION)
        self.assertEqual(r.conversation_purpose,ConversationPurpose.PRE_REQUEST)
        self.assertFalse(r.requested_effects)
        self.assertFalse(r.requires_execution)
        self.assertFalse(r.requested_operations)
        for extra in ({'execute':True},{'artifact_type':'CODE'},{'references':['x.py']}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                semantic_resolution_payload({**dict(intent='PRE_REQUEST',speech_act='SOCIAL',
                    primary_goal='Wait for request',confidence=.95),**extra},{})

    def test_clarification_uses_partial_understanding_not_generic_host_question(self):
        outcome=RuntimeOutcome(status=S.CLARIFICATION,user_goal='Revisar algo mostrado',
            clarification='Necesito que aclares que quieres hacer antes de continuar.')
        c=self.controller()
        req=request_for(outcome,c,'Podrias revisar eso?')
        self.assertEqual(req.response_intent,ResponseIntent.ASK_NATURALLY)
        self.assertEqual(req.understood_goal,'Revisar el contenido indicado')
        self.assertEqual(req.missing_information,('material ausente',))
        text='Quieres revisar un texto o una imagen? Puedes compartirlo aqui.'
        a,calls,_=self.assistant([{'text':text},review_payload()])
        f=FinalResponseComposer.compose(outcome)
        result=naturalize_final(a,c,outcome,f,req.user_turn)
        self.assertEqual(result.visible_text,text)
        self.assertEqual(result.spoken_text,text)
        self.assertIs(result.decision,f.decision)
        self.assertIn('material ausente',str(calls[0]))
        self.assertEqual(outcome.status,S.CLARIFICATION)

    def test_same_outcome_allows_different_natural_wording(self):
        outcome=RuntimeOutcome(status=S.CLARIFICATION)
        c=self.controller()
        variants=('Que parte del contenido te gustaria revisar?',
                  'Puedes compartir el material y contarme que te preocupa de el?')
        for text in variants:
            a,_,_=self.assistant([{'text':text},review_payload()])
            final=naturalize_final(a,c,outcome,FinalResponseComposer.compose(outcome),'Revisa eso')
            self.assertEqual(final.visible_text,text)

    def test_specific_failure_record_is_preserved_and_used_as_evidence(self):
        failure=FailureRecord.create(task_id='t',stage='EXECUTION',error_code='TOOL_EXECUTION_FAILED',
            safe_user_message='',technical_message='ValueError:python_import_not_allowed',retryable=False)
        outcome=RuntimeOutcome(status=S.FAILED,task_id='t',failure_ref=failure.failure_id)
        c=self.controller(failure=failure)
        req=request_for(outcome,c,'Que lo impidio?')
        self.assertIn('importacion',req.facts['recorded_failure'])
        self.assertTrue(req.facts['cause_detail_present'])
        text='El programa pide una libreria que no esta permitida para esa tarea.'
        a,calls,_=self.assistant([{'text':text},review_payload()])
        result=naturalize_final(a,c,outcome,FinalResponseComposer.compose(outcome),req.user_turn)
        self.assertEqual(result.visible_text,text)
        self.assertEqual(outcome.failure_ref,failure.failure_id)
        self.assertNotIn('python_import_not_allowed',str(calls))

    def test_missing_failure_detail_is_explicit_evidence_not_an_invented_cause(self):
        failure=FailureRecord.create(task_id='t',stage='EXECUTION',error_code='UNKNOWN',
            safe_user_message='',retryable=False)
        outcome=RuntimeOutcome(status=S.FAILED,task_id='t',failure_ref=failure.failure_id)
        c=self.controller(failure=failure)
        request=request_for(outcome,c,'Que sucedio?')
        self.assertFalse(request.facts['cause_detail_present'])
        text='No tengo el detalle de lo que paso, asi que prefiero no adivinar la causa.'
        a,_,_=self.assistant([{'text':text},review_payload()])
        result=naturalize_final(a,c,outcome,FinalResponseComposer.compose(outcome),request.user_turn)
        self.assertEqual(result.visible_text,text)

    def test_unsupported_uses_conversational_alternatives_without_granting_tools(self):
        outcome=RuntimeOutcome(status=S.UNSUPPORTED,user_goal='Enviar un mensaje por otra aplicacion')
        req=request_for(outcome,self.controller(),'Puedes enviarlo por mi?')
        self.assertEqual(req.response_intent,ResponseIntent.EXPLAIN_LIMITATION)
        self.assertEqual(req.facts['conversational_alternatives'],
                         ['RESPOND_IN_CHAT','DISCUSS_USER_PROVIDED_INFORMATION'])
        self.assertFalse(req.facts['success_claim_allowed'])
        text='No puedo enviarlo desde aqui, pero podemos preparar juntos el texto.'
        a,_,_=self.assistant([{'text':text},review_payload()])
        f=FinalResponseComposer.compose(outcome)
        result=naturalize_final(a,self.controller(),outcome,f,req.user_turn)
        self.assertIs(result.decision,f.decision)
        self.assertEqual(result.visible_text,text)

    def test_no_success_confirmation_or_code_publication_behavior_changes(self):
        for status in (S.SUCCESS,S.PENDING_CONFIRMATION,S.PARTIAL_SUCCESS,S.INCONCLUSIVE,S.CONVERSATIONAL):
            outcome=RuntimeOutcome(status=status)
            f=FinalResponseComposer.compose(outcome,conversational_text='Natural conversation')
            a,_,_=self.assistant([])
            self.assertIs(naturalize_final(a,self.controller(),outcome,f,'Current turn'),f)

    def test_fabricated_effect_never_published_even_if_model_calls_it_a_response(self):
        a,_,events=self.assistant([{'text':'Ya lo guarde.'},{'text':'He guardado el archivo.'}])
        outcome=RuntimeOutcome(status=S.FAILED)
        result=naturalize_final(a,self.controller(),outcome,FinalResponseComposer.compose(outcome),'Save it')
        self.assertEqual(result.visible_text,'')
        self.assertEqual(result.decision.status,S.FAILED)
        self.assertTrue(any(e=='natural_outcome_unavailable' for e,_ in events))

    def test_runtime_metacommentary_is_rewritten_after_semantic_rejection(self):
        text='Que parte te gustaria que revisemos primero?'
        a,calls,_=self.assistant([{'text':'El planner necesita otro requisito.'},
            review_payload(['INTERNAL_METADATA']),{'text':text},review_payload()])
        outcome=RuntimeOutcome(status=S.CLARIFICATION)
        result=naturalize_final(a,self.controller(),outcome,FinalResponseComposer.compose(outcome),'Revisa esto')
        self.assertEqual(result.visible_text,text)
        self.assertEqual(len(calls),4)

    def test_explicit_technical_question_does_not_require_lexical_input_router(self):
        outcome=RuntimeOutcome(status=S.FAILED)
        text='En esta explicacion tecnica, schema describe la estructura esperada de los datos.'
        a,_,_=self.assistant([{'text':text},review_payload()])
        result=naturalize_final(a,self.controller(),outcome,FinalResponseComposer.compose(outcome),
            'Quiero una explicacion tecnica del formato de datos')
        self.assertEqual(result.visible_text,text)

    def test_unavailable_model_abstains_instead_of_publishing_internal_fallback(self):
        a,_,_=self.assistant([OSError('unavailable')])
        outcome=RuntimeOutcome(status=S.CLARIFICATION,clarification='INTERNAL OUTCOME')
        f=naturalize_final(a,self.controller(),outcome,FinalResponseComposer.compose(outcome),'Current turn')
        self.assertEqual(f.visible_text,'')
        self.assertNotIn('INTERNAL',f.spoken_text)

    def test_no_user_phrase_table_in_outcome_selection(self):
        from anydes_tools.conversation import outcome_voice
        source=inspect.getsource(outcome_voice)
        for phrase in ('quiero que me ayudes con algo','te queria pedir un favor','ahorita necesito una cosa'):
            self.assertNotIn(phrase,source)
        self.assertNotIn('if user_turn',source)


if __name__=='__main__':
    unittest.main()
