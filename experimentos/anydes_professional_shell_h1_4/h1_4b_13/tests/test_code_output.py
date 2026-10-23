import json
from types import SimpleNamespace as NS
import unittest

from anydes_tools.conversation.code_generation import CodeGenerationSpec
from anydes_tools.conversation.semantic_runtime import AssistantCodeGenerator, _parse_json_object
from anydes_tools.conversation.code_output import CodeOutputFailure, decode, structure


def specification():
    return CodeGenerationSpec.from_dict(dict(language='python', goal='Count letters',
        functional_requirements=['Count only alphabetic characters'], inputs=['sentence'],
        outputs=['count'], constraints=[], dependencies=[], target_filename='letter_counter.py',
        tests=[], execution_required=False, verification_requirement='BASIC'))


def payload(**changes):
    value=dict(language='python', filename='letter_counter.py', code='print(1)\n', tests=[])
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


class Audit:
    def __init__(self):
        self.events=[]
    def record(self, event, **values):
        self.events.append(dict(event=event, **values))


class Owner:
    def __init__(self, responses):
        self.audit=Audit()
        self.calls=[]
        self.responses=list(responses)
        self._realtime_service=NS(transport=NS(stream=self.stream))
    def stream(self, messages, **kwargs):
        self.calls.append(messages)
        value=self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if not isinstance(value,str) else NS(text=value, done_reason='stop', connection_state='completed')


class CodeOutputTests(unittest.TestCase):
    def run_generation(self, responses):
        owner=Owner(responses)
        artifact=AssistantCodeGenerator(owner)(specification())
        return owner,artifact
    def test_pure(self):
        self.assertEqual(self.run_generation([payload()])[1].filename,'letter_counter.py')
    def test_fenced(self):
        self.run_generation(['```json\n'+payload()+'\n```'])
    def test_prose(self):
        self.run_generation(['Here is the result:\n'+payload()+'\nDone.'])
    def test_multiple(self):
        with self.assertRaisesRegex(CodeOutputFailure,'AMBIGUOUS'):
            decode(payload()+payload(),specification(),_parse_json_object)
    def test_non_json_one_repair(self):
        owner,_=self.run_generation(['Not JSON',payload()])
        self.assertEqual(len(owner.calls),2)
    def test_invalid_escape_repair(self):
        owner,_=self.run_generation(['{"code":"\\q"}',payload()])
        self.assertEqual(owner.audit.events[0]['parse_family'],'INVALID_JSON_ESCAPING')
    def test_truncated_no_repair(self):
        owner=Owner([NS(text='{"code":',done_reason='length')])
        with self.assertRaisesRegex(CodeOutputFailure,'TRUNCATED'):
            AssistantCodeGenerator(owner)(specification())
        self.assertEqual(len(owner.calls),1)
    def test_wrong_schema_repair(self):
        self.run_generation(['{"code":3}',payload()])
    def test_exact_source_preservation(self):
        source='p = r"C:\\Users\\DemoUser\\file.py"\nname = "árbol"\nprint(f"{name}\\t\\n")\n'
        self.assertEqual(self.run_generation([payload(code=source)])[1].code,source)
    def test_filename(self):
        self.assertEqual(self.run_generation([payload()])[1].filename,specification().target_filename)
    def test_language_cannot_change(self):
        with self.assertRaisesRegex(CodeOutputFailure,'REPAIR_FAILED'):
            self.run_generation(['bad',payload(language='cpp')])
    def test_filename_cannot_change(self):
        with self.assertRaisesRegex(CodeOutputFailure,'REPAIR_FAILED'):
            self.run_generation(['bad',payload(filename='other.py')])
    def test_no_execution_field(self):
        with self.assertRaisesRegex(CodeOutputFailure,'REPAIR_FAILED'):
            self.run_generation(['bad',payload(execute=True)])
    def test_one_repair_maximum(self):
        owner=Owner(['bad','bad',payload()])
        with self.assertRaisesRegex(CodeOutputFailure,'REPAIR_FAILED'):
            AssistantCodeGenerator(owner)(specification())
        self.assertEqual(len(owner.calls),2)
    def test_same_validation_on_repair(self):
        with self.assertRaisesRegex(CodeOutputFailure,'REPAIR_FAILED'):
            self.run_generation(['bad',payload(code='def broken(')])
    def test_transport_exception_no_repair(self):
        owner=Owner([RuntimeError('unavailable'),payload()])
        with self.assertRaises(RuntimeError):
            AssistantCodeGenerator(owner)(specification())
        self.assertEqual(len(owner.calls),1)
    def test_transport_outcome_no_repair(self):
        owner=Owner([NS(text='',done_reason='error',error_type='rate_limit')])
        with self.assertRaisesRegex(CodeOutputFailure,'TRANSPORT_FAILURE'):
            AssistantCodeGenerator(owner)(specification())
    def test_safe_diagnostics(self):
        owner,_=self.run_generation(['private invalid response',payload()])
        logs=json.dumps(owner.audit.events)
        self.assertNotIn('private invalid response',logs)
        self.assertNotIn('print(1)',logs)
        self.assertIn('response_sha256',logs)
    def test_nested_objects_not_ambiguous(self):
        self.assertEqual(structure('{"x":{"y":"{escaped}"}}')['candidate_objects'],1)
    def test_schema_field_paths(self):
        with self.assertRaises(CodeOutputFailure) as ctx:
            decode('{"code":4}',specification(),_parse_json_object)
        self.assertIn('language',ctx.exception.details['missing_fields'])
        self.assertEqual(ctx.exception.details['invalid_types'],['code'])
    def test_no_artifact_side_effect(self):
        # Generation has no filesystem/tool interface; failure returns no artifact.
        owner=Owner(['bad','bad'])
        with self.assertRaises(CodeOutputFailure):
            AssistantCodeGenerator(owner)(specification())
        self.assertFalse(hasattr(owner,'tools'))
    def test_host_tests_authority(self):
        _,artifact=self.run_generation([payload(tests=[{'untrusted':'ignored'}])])
        self.assertEqual(artifact.tests,specification().tests)


class FallbackTests(unittest.TestCase):
    def exercise(self, responses):
        from anydes.hybrid.router import HybridBrainRouter, HybridRouterConfig, ProviderFailure, _stream_outcome
        class Primary:
            model='openai/gpt-oss-120b'
            def stream(self,*args,**kwargs):
                raise ProviderFailure('rate_limit',status=429,retry_after=12)
        class Local:
            model='qwen3:8b'
            calls=0
            def is_ready(self): return True
            def preload(self): return 0.0
            def unload(self): pass
            def stream(self,*args,**kwargs):
                self.calls+=1
                return _stream_outcome(text=responses.pop(0),done_reason='stop')
        local=Local()
        router=HybridBrainRouter(HybridRouterConfig(auto_housekeeping=False),online=Primary(),local=local)
        owner=Owner([])
        owner._realtime_service=NS(transport=NS(router=router,stream=router.stream))
        try:
            try:
                artifact=AssistantCodeGenerator(owner)(specification())
            except CodeOutputFailure:
                artifact=None
            return owner,artifact,local.calls
        finally:
            router.close()
    def test_429_fallback_valid(self):
        owner,artifact,calls=self.exercise([payload()])
        self.assertIsNotNone(artifact)
        self.assertEqual(calls,1)
        info=owner.audit.events[0]['providers']
        self.assertEqual(info[0]['http_status'],429)
        self.assertEqual(info[-1]['backend'],'FALLBACK_GENERATION')
        self.assertEqual(info[-1]['model'],'qwen3:8b')
    def test_429_fallback_malformed_once(self):
        owner,artifact,calls=self.exercise(['bad','bad'])
        self.assertIsNone(artifact)
        self.assertEqual(calls,2)
        self.assertEqual(owner.audit.events[-1]['providers'][-1]['model'],'qwen3:8b')


if __name__=='__main__':
    unittest.main()
