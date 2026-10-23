import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from anydes_tools.conversation.classifier_wire import extract,ClassifierWireError
from anydes_tools.conversation.semantic_runtime import AssistantInteractionClassifier
from anydes_tools.conversation.interaction_router import InteractionResolver
from anydes_tools.tests import test_continue_task_contract as prior


def social(**extra):
    return dict(intent='CONVERSATION',speech_act='SOCIAL',primary_goal='Talk casually',
        confidence=.98,conversation_purpose='GENERAL',**extra)
def action(**extra):
    return dict(intent='CREATE_NEW',speech_act='ACTION_REQUEST',primary_goal='Create a small program',
        confidence=.98,artifact_type='CODE',execute=False,verification_requirement='BASIC',**extra)
class Audit:
    def __init__(self): self.events=[]
    def record(self,event,**values): self.events.append((event,values))
class Owner:
    def __init__(self): self.audit=Audit()
def model(*responses):
    owner=Owner()
    classifier=AssistantInteractionClassifier(owner)
    classifier.complete=Mock(side_effect=[(owner,NS(text=v if isinstance(v,str) else json.dumps(v))) for v in responses])
    return owner,classifier
def context():
    return {'pending_action':False,'failure_available':False,'retry_available':False,
        'resolution_context':{'active_task':{},'pending_confirmation':{'available':False}}}


class ClassifierWire(unittest.TestCase):
    def test_pure(self):
        owner,c=model(social())
        self.assertEqual(InteractionResolver(c).resolve('A quiet afternoon.',state=context()).semantic_intent,'CONVERSATION')
    def test_fence(self):
        owner,c=model('```json\n'+json.dumps(social())+'\n```')
        self.assertEqual(InteractionResolver(c).resolve('A quiet afternoon.',state=context()).semantic_intent,'CONVERSATION')
    def test_prose(self):
        owner,c=model('Result:\n'+json.dumps(social())+'\nEnd.')
        self.assertEqual(InteractionResolver(c).resolve('A quiet afternoon.',state=context()).semantic_intent,'CONVERSATION')
    def test_multiple_fenced_objects_rejected(self):
        with self.assertRaisesRegex(ClassifierWireError,'AMBIGUOUS'):
            extract('```json\n'+json.dumps(social())+'\n```\n```json\n'+json.dumps(social())+'\n```')
    def test_non_json_one_repair_valid(self):
        owner,c=model('not JSON',social())
        result=InteractionResolver(c).resolve('A quiet afternoon.',state=context())
        self.assertEqual(result.semantic_intent,'CONVERSATION')
        self.assertEqual(c.complete.call_count,2)
    def test_both_malformed_fail_closed(self):
        owner,c=model('not JSON','still not JSON',social())
        resolver=InteractionResolver(c)
        result=resolver.resolve('A quiet afternoon.',state=context())
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertEqual(c.complete.call_count,2)
        self.assertEqual(resolver.validation_history[-1]['errors'],('classification_unavailable',))
    def test_extra_fields_rejected_on_both_paths(self):
        owner,c=model(social(invented=True),social(invented=True))
        self.assertEqual(InteractionResolver(c).resolve('A quiet afternoon.',state=context()).source,'LOCAL_FAIL_CLOSED')
    def continuation(self):
        task=prior.contract()
        value=prior.payload(task_update={'task_id':task.task_id,'relation':'CONTINUE',
            'requirements':[{'key':'input_kind','value':'integers'}]})
        return task,value,prior.state(task)
    def test_valid_continuation_exact_id(self):
        task,value,state=self.continuation()
        owner,c=model(value)
        result=InteractionResolver(c).resolve('Integers.',state=state)
        self.assertEqual(result.task_update['task_id'],task.task_id)
        self.assertEqual(result.semantic_intent,'CONTINUE_TASK')
    def test_invalid_update_id_rejected(self):
        task,value,state=self.continuation()
        value['task_update']['task_id']='task:forged'
        owner,c=model(value,value)
        result=InteractionResolver(c).resolve('Integers.',state=state)
        self.assertNotEqual(result.semantic_intent,'CONTINUE_TASK')
        self.assertEqual(c.complete.call_count,2)
    def test_new_task_model_id_forbidden(self):
        value=action(task_proposal={'relation':'START','operation':'CREATE_NEW','task_id':'task:forged'})
        owner,c=model(value,value)
        self.assertEqual(InteractionResolver(c).resolve('Create a source.',state=context()).source,'LOCAL_FAIL_CLOSED')
    def test_repair_cannot_add_execution(self):
        owner,c=model(action(invented=True),{**action(), 'execute':True,'execution_intent':'RUN'})
        result=InteractionResolver(c).resolve('Create a source only.',state=context())
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertFalse(result.requires_execution)
    def test_repair_cannot_change_goal(self):
        changed={**action(),'primary_goal':'Delete a different file'}
        owner,c=model(action(invented=True),changed)
        result=InteractionResolver(c).resolve('Create a source.',state=context())
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
    def test_pending_task_does_not_force_continuation(self):
        task,value,state=self.continuation()
        owner,c=model(social())
        result=InteractionResolver(c).resolve('This sunset is beautiful.',state=state)
        self.assertEqual(result.semantic_intent,'CONVERSATION')
        self.assertIsNone(result.task_update)
    def test_same_boundary_for_repair_wrappers(self):
        owner,c=model('bad','Some prose\n```json\n'+json.dumps(social())+'\n```\nEnd')
        self.assertEqual(InteractionResolver(c).resolve('A quiet afternoon.',state=context()).semantic_intent,'CONVERSATION')
    def test_no_payload_in_diagnostics(self):
        private='PRIVATE_DO_NOT_RETAIN'
        owner,c=model(private,{**social(),'primary_goal':private})
        InteractionResolver(c).resolve('A quiet afternoon.',state=context())
        self.assertNotIn(private,json.dumps(owner.audit.events))
        attempts=[d for e,d in owner.audit.events if e=='classifier_wire_attempt']
        self.assertEqual([d['origin'] for d in attempts],['ORIGINAL','REPAIR'])
        self.assertEqual(attempts[-1]['schema_status'],'VALID')
    def test_array_not_promoted_to_object(self):
        with self.assertRaisesRegex(ValueError,'semantic_json_object_required'):
            extract(json.dumps([social()]))
    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ClassifierWireError,'DUPLICATE'):
            extract('{"intent":"CONVERSATION","intent":"CREATE_NEW"}')
    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ClassifierWireError,'NONFINITE'):
            extract('{"confidence":NaN}')


class BackendAttribution(unittest.TestCase):
    def test_existing_429_fallback_role_preserved(self):
        from anydes.hybrid.router import HybridBrainRouter,HybridRouterConfig,ProviderFailure,_stream_outcome
        class Primary:
            model='openai/gpt-oss-120b'
            def stream(self,*args,**kwargs): raise ProviderFailure('rate_limit',status=429,retry_after=12)
        class Local:
            model='qwen3:8b'
            def is_ready(self): return True
            def preload(self): return 0.0
            def unload(self): pass
            def stream(self,*args,**kwargs): return _stream_outcome(text=json.dumps(social()),done_reason='stop')
        router=HybridBrainRouter(HybridRouterConfig(auto_housekeeping=False),online=Primary(),local=Local())
        owner=Owner()
        owner._realtime_service=NS(transport=NS(router=router,stream=router.stream))
        try:
            result=InteractionResolver(AssistantInteractionClassifier(owner)).resolve('A quiet afternoon.',state=context())
            self.assertEqual(result.semantic_intent,'CONVERSATION')
            event=[d for e,d in owner.audit.events if e=='classifier_wire_attempt'][0]
            self.assertEqual(event['providers'][0]['http_status'],429)
            self.assertEqual(event['providers'][-1]['role'],'FALLBACK')
            self.assertEqual(event['providers'][-1]['model'],'qwen3:8b')
        finally:
            router.close()


if __name__=='__main__': unittest.main()
