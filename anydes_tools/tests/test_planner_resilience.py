"""Controlled provider failures, same real semantic parser; no live rate-limit induction."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from anydes_tools.tests import test_human_acceptance_round1 as prior
from anydes_tools.tests.test_operation_grounding import resolved
from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner
from anydes_tools.conversation.plan_protocol import PlanProtocolError
from anydes_tools.conversation.interaction_router import TaskControl as T
from anydes.hybrid.router import ProviderFailure, OnlineContextSanitizer, _stream_outcome


def proposal():
    return {'goal':'execute the bound source once','operations':[{
        'operation_id':'run','operation_kind':'TEST_CODE','inputs':['selected_artifact'],
        'outputs':[],'specification':{'language':'python','stdin':''}}],
        'requested_final_outputs':['selected_artifact']}


class Audit:
    def __init__(self): self.events = []
    def record(self, event, **detail): self.events.append((event,detail))


class Online:
    model = 'openai/gpt-oss-120b'
    def __init__(self, values): self.values, self.calls = list(values), []
    def stream(self, messages, **options):
        self.calls.append((messages,options))
        value = self.values.pop(0)
        if isinstance(value,Exception): raise value
        return _stream_outcome(text=value,done_reason='stop')


class LocalTransport:
    responses = []
    calls = []
    def __init__(self, **config): self.config = config
    def stream(self, messages, **options):
        self.calls.append((messages,options,self.config))
        value = self.responses.pop(0)
        if isinstance(value,Exception): raise value
        return _stream_outcome(text=value,done_reason='stop')


class Assistant:
    def __init__(self, online, local):
        self.audit = Audit()
        LocalTransport.responses, LocalTransport.calls = list(local), []
        self.online = Online(online)
        router = SimpleNamespace(online=self.online, sanitizer=OnlineContextSanitizer(),
            config=SimpleNamespace(local_keep_alive='10m',local_num_ctx=768),
            local=SimpleNamespace(transport=LocalTransport(),model='qwen3:8b',base_url='http://127.0.0.1:11434'))
        self._realtime_service = SimpleNamespace(transport=SimpleNamespace(router=router))


def request():
    return {'user_request':'Run the selected program',
        'artifact_focus':{'logical_name':'selected_artifact','version':2,'hash':'known-hash'},
        'semantic_operations':[]}


class ProviderPolicy(unittest.TestCase):
    def test_primary_success_no_fallback(self):
        assistant = Assistant([json.dumps(proposal())],[])
        result = AssistantGeneralTaskReasoner(assistant)(request()).proposal
        self.assertEqual(result.payload,proposal())
        self.assertEqual(len(assistant.online.calls),1)
        self.assertEqual(LocalTransport.calls,[])

    def test_429_fallback_same_contract_and_context(self):
        assistant = Assistant([ProviderFailure('rate_limit',status=429,retry_after=12)], [json.dumps(proposal())])
        req = request()
        result = AssistantGeneralTaskReasoner(assistant)(req).proposal
        self.assertEqual(result.payload,proposal())
        self.assertEqual(len(assistant.online.calls),1)
        self.assertEqual(len(LocalTransport.calls),1)
        self.assertEqual(assistant.online.calls[0][0],LocalTransport.calls[0][0])
        self.assertEqual(json.loads(LocalTransport.calls[0][0][1]['content'])['context']['artifact_focus'],req['artifact_focus'])
        self.assertEqual(LocalTransport.calls[0][2]['num_ctx'],8192)
        self.assertFalse(req['plan_repair_budget']['used'])

    def test_malformed_fallback_one_repair_retains_original_context(self):
        assistant = Assistant([ProviderFailure('rate_limit',status=429)], ['not JSON',json.dumps(proposal())])
        req = request()
        AssistantGeneralTaskReasoner(assistant)(req)
        self.assertEqual(len(assistant.online.calls),1)
        self.assertEqual(len(LocalTransport.calls),2)
        self.assertTrue(req['plan_repair_budget']['used'])
        repaired = json.loads(LocalTransport.calls[1][0][1]['content'])
        self.assertEqual(repaired['context']['user_request'],req['user_request'])
        self.assertEqual(repaired['context']['artifact_focus'],req['artifact_focus'])
        self.assertIn('schema',repaired)

    def test_bad_repair_fails_closed_without_third_call(self):
        assistant = Assistant([ProviderFailure('rate_limit',status=429)],['bad','{}'])
        with self.assertRaises(PlanProtocolError):
            AssistantGeneralTaskReasoner(assistant)(request())
        self.assertEqual(len(assistant.online.calls),1)
        self.assertEqual(len(LocalTransport.calls),2)

    def test_all_providers_unavailable(self):
        assistant = Assistant([ProviderFailure('rate_limit',status=429)], [OSError('offline')])
        with self.assertRaisesRegex(PlanProtocolError,'PLANNER_PROVIDERS_UNAVAILABLE'):
            AssistantGeneralTaskReasoner(assistant)(request())
        self.assertEqual(len(LocalTransport.calls),1)


class FallbackExecutionIntegration(unittest.TestCase):
    setUp = prior.Acceptance.setUp
    tearDown = prior.Acceptance.tearDown
    controller = prior.Acceptance.controller
    source = prior.Acceptance.source

    def test_all_unavailable_has_safe_response_and_no_executor(self):
        from anydes_tools.conversation.publication import FinalResponseComposer
        assistant = Assistant([ProviderFailure('rate_limit',status=429)], [OSError('offline')])
        c = self.controller(resolved('EXECUTE_EXISTING'),
            general_task_reasoner=AssistantGeneralTaskReasoner(assistant))
        self.source(c,'selected.py')
        c.handle('Execute the selected program',turn_id='unavailable')
        self.assertFalse(c._executor_called)
        self.assertIsNone(c.state.pending_tool_action)
        final = FinalResponseComposer.compose(c.last_runtime_outcome)
        self.assertIn('No pude preparar la ejecucion en este momento',final.visible_text)
        self.assertNotIn('PLANNER_PROVIDERS_UNAVAILABLE',final.visible_text)

    def test_fallback_does_not_bypass_permission_or_duplicate_execution(self):
        value = {'goal':'create and execute once','operations':[{
            'operation_id':'code','operation_kind':'GENERATE_CODE','inputs':[],
            'outputs':[{'name':'main','artifact_type':'CODE','format':'py','filename':'once.py'}],
            'specification':{'language':'python','goal':'convert Celsius to Fahrenheit',
                'functional_requirements':['read Celsius and print Fahrenheit'],
                'operation_descriptor':{'semantic_kind':'UNIT_CONVERSION','input_type':'NUMBER',
                    'output_type':'NUMBER','parameters':{'source_unit':'C','target_unit':'F'}},
                'tests':[{'test_id':'zero','stdin':'0\n','expected_hint':'32','comparison_mode':'NUMERIC'}],
                'verification_requirement':'TEST_SUITE','execution_required':True}},
            {'operation_id':'run','operation_kind':'TEST_CODE','inputs':['main'],'outputs':[],
             'specification':{'language':'python','stdin':''}}],'requested_final_outputs':['main']}
        assistant = Assistant([ProviderFailure('rate_limit',status=429)], [json.dumps(value),json.dumps(value)])
        c = self.controller(resolved('CREATE_NEW',artifact_type='CODE',execute=True,verification_requirement='TEST_SUITE'),
            resolved('CONFIRM'), general_task_reasoner=AssistantGeneralTaskReasoner(assistant),
            code_generator=lambda spec:{'language':'python','filename':'once.py',
                'code':'value=float(input())\nprint(value*9/5+32)\n','tests':[]})
        with patch.object(c.executor,'execute',wraps=c.executor.execute) as execute:
            c.handle('Create a minimal source and execute it',turn_id='request')
            self.assertFalse(c._executor_called)
            self.assertEqual(execute.call_count,0)
            self.assertIsNotNone(c.state.pending_tool_action,repr(c.state.last_tool_failure))
            task = c.state.pending_task.task_id
            c.handle('Proceed with the execution',turn_id='approval')
            self.assertEqual(c.state.pending_task.task_id,task)
            runs = [call for call in execute.call_args_list if call.args[0].capability=='developer.run_python']
            self.assertEqual(len(runs),1)
            self.assertTrue(c._executor_called)
            self.assertEqual((self.workspace/'once.py').read_text(),'value=float(input())\nprint(value*9/5+32)\n')


if __name__ == '__main__':
    unittest.main()
