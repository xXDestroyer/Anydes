"""An incomplete task exists before its clarification is published."""
import inspect
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from anydes_tools.conversation.operation_intent import semantic_resolution_payload,ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolution as R,InteractionResolver
from anydes_tools.conversation.classifier_output import accept_task_semantics
from anydes_tools.conversation.clarification_anchor import parse_evidence
from anydes_tools.tests import test_classifier_request_contract as fixtures
from anydes_tools.tests import test_manual_flow as prior


def task_request():
    return fixtures.clarification(clarification_scope='TASK',task_proposal={
        'relation':'START','operation':'CREATE_NEW','artifact_type':'CODE',
        'requirements':[{'key':'purpose','value':'format supplied values'}],
        'requirement_items':[{'key':'language','necessity':'REQUIRED','question':'Which language?'},
            {'key':'layout','necessity':'REQUIRED','question':'Which result layout?'}],
        'clarification_target':{'key':'language','question':'Which language?'}})


def host():
    return NS(state=NS(pending_task=NS(semantic_contract=None)),audit=NS(record=lambda *a,**k:None))


class Anchor(unittest.TestCase):
    def test_incomplete_task_allocates_once_before_waiting(self):
        decision=fixtures.resolve_wire(task_request())
        c=host();accept_task_semantics(c,decision,'new')
        task=c.state.pending_task.semantic_contract;task_id=task.task_id
        self.assertTrue(task_id.startswith('task:'))
        self.assertEqual(c.state.pending_task.status,'AWAITING_CLARIFICATION')
        self.assertEqual(set(task.unresolved_questions),{'language','layout'})
        self.assertEqual(task.goal,decision.primary_goal)
        from anydes_tools.tests.test_continue_task_contract import state
        for key,value in [('language','Python'),('layout','one per line')]:
            d=prior.wire('CONTINUE_TASK',state(task),task_update={'task_id':task_id,'relation':'CONTINUE',
                'requirements':[{'key':key,'value':value}]})
            accept_task_semantics(c,d,key)
            self.assertEqual(c.state.pending_task.task_id,task_id)
            self.assertIn(key,task.resolved_targets)
            if key=='language':
                self.assertFalse(task.planning_gate()['ready'])
        self.assertTrue(task.planning_gate()['ready'])
        self.assertEqual(set(task.requirements),{'purpose','language','layout'})

    def test_one_unresolved_task_is_persistable(self):
        raw=task_request();raw['task_proposal']['requirement_items']=raw['task_proposal']['requirement_items'][:1]
        c=host();accept_task_semantics(c,fixtures.resolve_wire(raw),'new')
        self.assertEqual(list(c.state.pending_task.semantic_contract.unresolved_questions),['language'])

    def test_complete_task_ready_without_artificial_clarifications(self):
        c=host();accept_task_semantics(c,fixtures.resolve_wire(fixtures.action()),'new')
        self.assertTrue(c.state.pending_task.semantic_contract.planning_gate()['ready'])

    def test_orphan_operational_clarification_rejected(self):
        raw=task_request();raw.pop('task_proposal')
        with self.assertRaisesRegex(ResolutionSchemaError,'OPERATIONAL_CLARIFICATION_WITHOUT_TASK_ANCHOR'):
            semantic_resolution_payload(raw,fixtures.STATE)

    def test_conversational_clarification_allowed_without_task(self):
        raw={'intent':'CLARIFICATION','speech_act':'INFORMATION_REQUEST','primary_goal':'Clarify a conversational referent',
            'confidence':.9,'clarification_scope':'CONVERSATION'}
        self.assertIsNone(fixtures.resolve_wire(raw).task_proposal)

    def test_operational_scope_cannot_be_erased_by_conversation_label(self):
        raw=task_request();raw['clarification_scope']='CONVERSATION'
        with self.assertRaisesRegex(ResolutionSchemaError,'operational_request_cannot_be_conversational'):
            semantic_resolution_payload(raw,fixtures.STATE)

    def test_proposal_cannot_contain_id_authority_or_host_fields(self):
        for key,value in [('task_id','task:invented'),('task_update',{}),('execute',True),('plan_id','plan:x'),('artifact_ref','artifact:x')]:
            raw=task_request();raw['task_proposal'][key]=value
            with self.assertRaises(ResolutionSchemaError):
                fixtures.resolve_wire(raw)

    def test_parse_repair_reconstructs_anchor_from_original_request(self):
        owner,model=fixtures.classifier(task_request())
        private='PRIVATE_BROKEN_RESPONSE'
        model.complete=Mock(side_effect=[(owner,NS(text='{'+private)),(owner,NS(text=json.dumps(task_request())))])
        result=InteractionResolver(model).resolve('Prepare a source, with missing language and layout.',state=fixtures.STATE)
        self.assertEqual(result.semantic_intent,'CLARIFICATION')
        self.assertIsNotNone(result.task_proposal)
        request=json.loads(model.complete.call_args_list[1].args[0][1]['content'])
        contract=request['classifier_reconstruction_contract']
        self.assertFalse(contract['pending_task_exists'])
        self.assertIn('task_proposal',contract['operational_clarification_required_fields'])
        self.assertEqual(contract['safe_retained_classifier_hints'],[])
        diagnostics=[d for e,d in owner.audit.events if e=='classifier_json_parse_failure']
        self.assertEqual(diagnostics[0]['reason_code'],'MODEL_JSON_PARSE_FAILURE')
        self.assertIsInstance(diagnostics[0]['parser_position'],int)
        self.assertTrue(diagnostics[0]['repair_invoked'])
        self.assertNotIn(private,json.dumps(owner.audit.events))
        self.assertNotIn(private,json.dumps(request))
        self.assertEqual(model.complete.call_count,2)

    def test_unscoped_generic_repair_after_parse_error_fails_closed(self):
        owner,model=fixtures.classifier(task_request())
        model.complete=Mock(side_effect=[(owner,NS(text='invalid json')),
            (owner,NS(text=json.dumps({'intent':'CLARIFICATION','speech_act':'INFORMATION_REQUEST','primary_goal':'Clarify','confidence':.9})))])
        result=InteractionResolver(model).resolve('Prepare a new artifact.',state=fixtures.STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertEqual(model.complete.call_count,2)
        self.assertFalse(result.requested_effects)

    def test_known_task_can_repair_to_anchored_clarification(self):
        owner,model=fixtures.classifier(fixtures.action(unknown=True),task_request())
        result=InteractionResolver(model).resolve('Prepare source with a material choice.',state=fixtures.STATE)
        self.assertEqual(result.semantic_intent,'CLARIFICATION')
        self.assertIsNotNone(result.task_proposal)
        self.assertEqual(model.complete.call_count,2)

    def test_declared_task_scope_repair_fills_missing_proposal(self):
        bad=task_request();bad.pop('task_proposal')
        owner,model=fixtures.classifier(bad,task_request())
        result=InteractionResolver(model).resolve('Prepare a source with missing material choices.',state=fixtures.STATE)
        self.assertIsNotNone(result.task_proposal)
        self.assertEqual(result.semantic_intent,'CLARIFICATION')
        request=json.loads(model.complete.call_args_list[1].args[0][1]['content'])
        self.assertEqual(request['fields_in_conflict']['protocol_diagnostics']['reason_code'],
            'OPERATIONAL_CLARIFICATION_WITHOUT_TASK_ANCHOR')
        self.assertEqual(model.complete.call_count,2)

    def test_parse_diagnostics_truncation_is_not_guessed(self):
        try:
            json.loads('{')
        except json.JSONDecodeError as error:
            for reason,status,code in [('length','YES','MODEL_JSON_TRUNCATED'),('stop','NO','MODEL_JSON_PARSE_FAILURE'),(None,'UNKNOWN','MODEL_JSON_PARSE_FAILURE')]:
                result=parse_evidence(error,'{',NS(done_reason=reason),repair_available=True)
                self.assertEqual(result['truncated_status'],status)
                self.assertEqual(result['reason_code'],code)
                self.assertEqual(result['response_byte_length'],1)
                self.assertTrue(result['obvious_outer_object_detected'])

    def test_parser_behavior_is_unchanged(self):
        from pathlib import Path
        import ast
        from anydes_tools.conversation.semantic_runtime import _parse_json_object
        old=Path(__file__).resolve().parents[2]/'h1_4b_6/overlay/conversation/semantic_runtime.py'
        previous=next(n for n in ast.parse(old.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='_parse_json_object')
        current=ast.parse(inspect.getsource(_parse_json_object)).body[0]
        self.assertEqual(ast.dump(previous),ast.dump(current))


class Ordering(unittest.TestCase):
    setUp=prior.TaskIntegration.setUp
    tearDown=prior.TaskIntegration.tearDown
    controller=prior.TaskIntegration.controller

    def test_task_exists_before_controller_returns_clarification(self):
        c=self.controller();c.interaction_resolver.classifier=lambda request:fixtures.resolve_wire(task_request(),request['state'])
        c.handle('Prepare an artifact with material questions.',turn_id='before-publication')
        self.assertEqual(c.state.pending_task.status,'AWAITING_CLARIFICATION')
        self.assertIsNotNone(c.state.pending_task.semantic_contract)
        self.assertTrue(c.state.pending_task.task_id.startswith('task:'))
        self.assertFalse(c._reasoner_called or c._executor_called)


if __name__=='__main__':
    unittest.main()
