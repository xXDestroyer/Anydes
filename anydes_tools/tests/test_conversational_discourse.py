"""Unit/contract tests. Prepared decisions are NOT human runtime evidence."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'programa_principal'))
from anydes_tools.conversation.discourse_context import DiscourseContext, DiscourseEntity, ConversationPurpose as P
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T, TurnRelation as D,
    InteractionResolver)
from anydes_tools.conversation.conversational_bridge import resolved_conversation, memory_evidence
from anydes_tools.conversation.discourse_context import supported_offer
from anydes_tools.conversation.failure_state import FailureRecord, FailureResponseRenderer
from anydes_tools.tests import test_context_control_resolution as prior

TEMP_ROOT=Path(__file__).resolve().parents[1]/'logs'


class DiscourseTests(unittest.TestCase):
    def test_entity_continues_and_switches_without_entity_specific_rules(self):
        state=DiscourseContext()
        for name in ('Orion', 'Mira'):
            state.resolve(R(K.CONVERSATION,turn_relation=D.NEW_TOPIC,
                subjects=(DiscourseEntity(name,'CHARACTER'),)))
            state.resolve(R(K.CONTENT_RESPONSE,turn_relation=D.TOPIC_CONTINUATION,
                conversation_purpose=P.ENTITY_QUERY))
            self.assertEqual(state.subjects[0].name,name)
        state.resolve(R(K.CONVERSATION,turn_relation=D.TOPIC_CLOSURE))
        self.assertFalse(state.subjects)

    def test_bounded_history_not_in_audit_snapshot(self):
        state=DiscourseContext()
        for i in range(20):
            state.record('private question',str(i))
        self.assertEqual(len(state.turns),12)
        self.assertNotIn('recent_turns',state.snapshot())
        self.assertEqual(len(state.snapshot(include_turns=True)['recent_turns']),6)

    def test_entity_schema_strict(self):
        for value in ({'name':'A'}, {'name':'A','semantic_type':[]}, 'A'):
            with self.assertRaises(ValueError):
                DiscourseEntity.parse(value)

    def test_no_causal_phrase_router(self):
        state={'failure_available':True,'pending_action':False,'retry_available':False}
        decision=R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,
            turn_relation=D.FAILURE_CAUSE,confidence=.85)
        for text in ('What prevented completion?', 'Could you explain the cause?'):
            result=InteractionResolver(lambda req:decision).resolve(text,state=state)
            self.assertEqual(result.task_control,T.EXPLAIN_FAILURE)

    def test_safe_execution_error_explains_recorded_policy_not_a_guessed_cause(self):
        record=FailureRecord.create(task_id='t',stage='EXECUTION',
            error_code='TOOL_EXECUTION_FAILED',technical_message='ValueError:python_import_not_allowed',
            safe_user_message='',retryable=True)
        self.assertIn('importacion',FailureResponseRenderer.render(record,depth=1))
        self.assertIn('antes de ejecutarlo',FailureResponseRenderer.render(record,depth=2))
        unknown=FailureRecord.create(task_id='t',stage='EXECUTION',error_code='UNKNOWN',
            technical_message='C:/private/secret password',safe_user_message='',retryable=False)
        result=FailureResponseRenderer.render(unknown,depth=2)
        self.assertNotIn('password',result)
        self.assertIn('no tengo',result)

    def test_real_memory_apis_read_only_and_secret_filter(self):
        from anydes.memoria.local_memory import JsonMemoryStore
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root=Path(directory)
            data=root/'candidate'/'data'
            store=JsonMemoryStore(root/'programa_principal'/'datos'/'memoria'/'permanente.json')
            store.set('hobby','cultivo plantas')
            store.set('api_key','not-to-be-shown')
            before=store.path.read_bytes()
            assistant=NS(config=NS(data_dir=data),speaker_allows_memory_context=True)
            facts,meta=memory_evidence(assistant,root)
            self.assertEqual([x['value'] for x in facts],['cultivo plantas'])
            self.assertEqual(store.path.read_bytes(),before)
            self.assertEqual(meta['selected'],1)
            self.assertNotIn('cultivo',str(meta))
            assistant.speaker_allows_memory_context=False
            self.assertEqual(memory_evidence(assistant,root)[0],[])

    def test_empty_memory_does_not_fabricate(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root=Path(directory)
            self.assertEqual(memory_evidence(NS(config=NS(data_dir=root/'data')),root)[0],[])

    def test_adapter_disables_duplicate_classifier_and_restores_on_exception(self):
        @dataclass
        class Context:
            turn_prompt:str='current request'
            user_states_preference:bool=False
            memory_fact_count:int=0
        audit=NS(record=lambda *a,**k:None)
        service=NS(router=NS(classify=lambda *a,**k:'OLD'),
            local_routes=NS(resolve=lambda *a,**k:'OLD'),
            session=NS(resolve_reference=lambda *a,**k:'OLD'),
            context_builder=NS(build=lambda *a,**k:Context()), phase2_enabled=True)
        service.transport=NS(stream=lambda *a,**k:None)
        assistant=NS(_realtime_service=service,audit=audit)
        discourse=DiscourseContext(subjects=(DiscourseEntity('Orion','CHARACTER'),))
        controller=NS(last_resolution=R(K.CONTENT_RESPONSE,turn_relation=D.TOPIC_CONTINUATION),
                      discourse=discourse,project_root=Path(__file__).resolve().parents[2],
                      registry=NS(list_capabilities=lambda **k:()))
        with self.assertRaises(RuntimeError):
            with resolved_conversation(assistant,controller):
                self.assertEqual(service.router.classify('anything').primary,'question')
                self.assertFalse(service.local_routes.resolve().handled)
                self.assertIsNone(service.session.resolve_reference())
                self.assertFalse(service.phase2_enabled)
                self.assertIn('Orion',service.context_builder.build().turn_prompt)
                raise RuntimeError('test')
        self.assertEqual(service.router.classify(),'OLD')
        self.assertTrue(service.phase2_enabled)

    def test_offer_is_grounded_not_execution_authority(self):
        contracts=[{'operation_kind':'CREATE','capability':'create','requires':['read']}]
        registry=NS(list_capabilities=lambda **k:('create','read'))
        value={'request':'Create a short note','operation_kind':'CREATE'}
        offer=supported_offer(value,registry,contracts)
        self.assertTrue(offer['offer_id'])
        self.assertNotIn('permissions',offer)
        self.assertIsNone(supported_offer({**value,'operation_kind':'INVENTED'},registry,contracts))
        registry.list_capabilities=lambda **k:('create',)
        self.assertIsNone(supported_offer(value,registry,contracts))

    def test_reply_adapter_honors_real_transport_callback_signature(self):
        import inspect
        from anydes.realtime.transport import OllamaRealtimeTransport
        @dataclass
        class Reply:
            text:str
        calls=[]
        def stream(messages, *, on_content=None, max_output_tokens=None):
            inspect.signature(OllamaRealtimeTransport.stream).bind(None,messages,
                on_content=on_content,max_output_tokens=max_output_tokens)
            calls.append(messages)
            if messages[-1]['role']=='user' and 'candidate' in messages[-1]['content']:
                import json
                from anydes_tools.tests.test_natural_response import review_payload
                return Reply(json.dumps(review_payload()))
            return Reply('{"text":"Hola, me alegra verte.","offer":null}')
        service=NS(router=NS(classify=lambda *a,**k:None),
            local_routes=NS(resolve=lambda *a,**k:None),session=NS(resolve_reference=lambda *a,**k:None),
            context_builder=NS(build=lambda *a,**k:None),transport=NS(stream=stream),phase2_enabled=True)
        assistant=NS(_realtime_service=service,audit=NS(record=lambda *a,**k:None))
        controller=NS(last_resolution=R(K.CONVERSATION,turn_relation=D.NEW_TOPIC),
            discourse=DiscourseContext(),project_root=Path(__file__).resolve().parents[2],
            registry=NS(list_capabilities=lambda **k:()))
        with resolved_conversation(assistant,controller):
            result=service.transport.stream([{'role':'user','content':'Good to see you'}],
                on_content=lambda _:self.fail('Internal JSON must not be streamed'),max_output_tokens=128)
            self.assertEqual(result.text,'Hola, me alegra verte.')
            result=service.transport.stream([{'role':'user','content':'A short greeting'}],max_output_tokens=None)
            self.assertEqual(result.text,'Hola, me alegra verte.')
        self.assertEqual(len(calls),4)

    def test_offer_acceptance_requires_matching_supported_operation(self):
        offer={'offer_id':'o','request':'create note','operation_kind':'CREATE_DOCUMENT'}
        state={'failure_available':False,'pending_action':False,'retry_available':False,
               'resolution_context':{'discourse':{'pending_offer':offer}}}
        decision=R(K.TOOL_ACTION,turn_relation=D.OFFER_ACCEPTANCE,confidence=.85,
                   requested_operations=('CREATE_DOCUMENT',),requested_effects=('filesystem.create',))
        self.assertFalse(InteractionResolver.contradictions(decision,state))
        state['resolution_context']['discourse']['pending_offer']=None
        self.assertIn('acceptance_without_supported_offer',InteractionResolver.contradictions(decision,state))


class ControllerDiscourse(unittest.TestCase):
    setUp=prior.ContextControl.setUp
    tearDown=prior.ContextControl.tearDown
    controller=prior.ContextControl.controller
    root=prior.ContextControl.root

    def test_acknowledgement_closes_failure_after_resolution(self):
        c=self.controller(R(K.CONVERSATION,turn_relation=D.TOPIC_CLOSURE,confidence=.85,
            conversation_purpose=P.ACKNOWLEDGEMENT))
        self.root(c)
        c.handle('Understood, let us leave it there',turn_id='close')
        self.assertIsNone(c.state.failure_topic_record())
        self.assertFalse(c._executor_called)

    def test_explicit_operation_not_forced_to_explanation(self):
        decision=R(K.TOOL_ACTION,turn_relation=D.NEW_ACTION,confidence=.85,
            requested_effects=('filesystem.read',),requested_operations=('READ_ARTIFACT',),
            requires_execution=True)
        resolver=InteractionResolver(lambda req:decision)
        result=resolver.resolve('Please inspect the recent artifact now',state={
            'failure_available':True,'pending_action':False,'retry_available':False})
        self.assertEqual(result.kind,K.TOOL_ACTION)
        self.assertEqual(result.task_control,T.NONE)
        self.assertEqual(len(resolver.validation_history),1)


if __name__ == '__main__':
    unittest.main()
