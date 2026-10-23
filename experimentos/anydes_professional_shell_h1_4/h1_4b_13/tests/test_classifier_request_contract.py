"""Discourse relation is not task intent; schema repair cannot switch variants."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from anydes_tools.conversation.operation_intent import semantic_resolution_payload,ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolution as R,InteractionResolver
from anydes_tools.conversation.semantic_runtime import AssistantInteractionClassifier
from anydes_tools.conversation.resolution_contract import branch_contract
from anydes_tools.tests import test_manual_flow as prior
from anydes_tools.tests import test_classifier_branches as fixtures

STATE={'pending_action':False,'failure_available':False,'retry_available':False,
    'resolution_context':{'active_task':{'task_id':'','semantic_contract':None},
        'active_failure_topic':{},'pending_confirmation':{'available':False}}}


def action(**extra):
    return {'intent':'CREATE_NEW','speech_act':'ACTION_REQUEST','primary_goal':'Create a small source',
        'confidence':.97,'artifact_type':'CODE','turn_relation':'NEW_TOPIC',**extra}


def clarification(**extra):
    return {'intent':'CLARIFICATION','speech_act':'INFORMATION_REQUEST','primary_goal':'Create a small source',
        'confidence':.97,'turn_relation':'NEW_TOPIC',
        'task_proposal':prior.update(target=('language','Which language?')),**extra}


def resolve_wire(value,state=None):
    return R.from_dict(semantic_resolution_payload(value,state or STATE))


def classifier(*values):
    owner=fixtures.prior.Assistant([],[])
    model=AssistantInteractionClassifier(owner)
    model.complete=Mock(side_effect=[(owner,NS(text=json.dumps(value))) for value in values])
    return owner,model


class Branches(unittest.TestCase):
    def test_frozen_legacy_rule_reproduces_new_topic_conflict(self):
        import hashlib
        import importlib.util
        from pathlib import Path
        import sys
        here=Path(__file__).resolve().parents[1]
        old=here.parent/'h1_4b_3/overlay/conversation/operation_intent.py'
        name='anydes_tools.conversation._legacy_classifier_rule_probe'
        spec=importlib.util.spec_from_file_location(name,old)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
        evidence=[]
        try:
            for raw in (clarification(),action()):
                if 'task_proposal' in raw:
                    raw['task_update']=raw.pop('task_proposal')
                with self.assertRaises(ValueError) as failure:
                    module.semantic_resolution_payload(raw,STATE)
                evidence.append({'intent':raw['intent'],'speech_act':raw['speech_act'],
                    'turn_relation':raw['turn_relation'],'rejection':str(failure.exception)})
        finally:
            sys.modules.pop(name,None)
        self.assertIn('intent_relation_conflict',evidence[0]['rejection'])
        self.assertIn('current_operation_not_failure_question',evidence[1]['rejection'])
        (here/'evidence').mkdir(exist_ok=True)
        (here/'evidence/legacy_rule_reproduction.json').write_text(json.dumps({
            'legacy_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'cases':evidence,
            'provider_calls':0,'tools':0},indent=2),encoding='utf-8')

    def test_new_topic_create_is_action_not_information(self):
        result=resolve_wire(action())
        self.assertEqual(branch_contract(action())['declared_branch'],'ARTIFACT_ACTION')
        self.assertEqual(result.kind.value,'TOOL_ACTION')
        self.assertEqual(result.turn_relation.value,'NEW_TOPIC')
        self.assertEqual(result.operation_intent.value,'CREATE_NEW')
        self.assertFalse(InteractionResolver.contradictions(result,STATE))

    def test_new_topic_incomplete_task_is_one_clarification_variant(self):
        result=resolve_wire(clarification())
        self.assertEqual(branch_contract(clarification())['declared_branch'],'TASK_REQUEST_CLARIFICATION')
        self.assertEqual(result.kind.value,'CLARIFICATION')
        self.assertEqual(result.turn_relation.value,'NEW_TOPIC')
        self.assertFalse(result.requested_effects)
        self.assertFalse(InteractionResolver.contradictions(result,STATE))

    def test_new_topic_factual_question_keeps_information_branch(self):
        value={'intent':'CONTENT_RESPONSE','speech_act':'INFORMATION_REQUEST',
            'primary_goal':'Explain a scientific concept','confidence':.97,'turn_relation':'NEW_TOPIC'}
        self.assertEqual(branch_contract(value)['declared_branch'],'INFORMATION')
        result=resolve_wire(value)
        self.assertEqual(result.kind.value,'CONTENT_RESPONSE')
        self.assertFalse(result.requested_effects)

    def test_acknowledgement_does_not_allow_task_or_identity_fields(self):
        value={'intent':'CONVERSATION','speech_act':'ACKNOWLEDGEMENT','primary_goal':'Acknowledge',
            'confidence':.97,'conversation_purpose':'GENERAL'}
        self.assertEqual(branch_contract(value)['declared_branch'],'ACKNOWLEDGEMENT')
        self.assertEqual(resolve_wire(value).kind.value,'CONVERSATION')
        for extra in ({'identity_attribute':'DISPLAY_NAME'},{'task_update':prior.update()}):
            with self.assertRaises(ResolutionSchemaError):
                resolve_wire({**value,**extra})

    def test_identity_field_is_exclusive_and_path_is_exact(self):
        raw=action(identity_attribute='DISPLAY_NAME')
        with self.assertRaises(ResolutionSchemaError) as rejected:
            resolve_wire(raw)
        d=rejected.exception.protocol_diagnostics
        self.assertEqual(d['unexpected_field_names'],['identity_attribute'])
        self.assertEqual(d['unexpected_field_paths'],['$.identity_attribute'])
        self.assertNotIn('identity_attribute',d['expected_allowed_fields'])
        identity={'intent':'PERSONAL_IDENTITY','speech_act':'INFORMATION_REQUEST',
            'primary_goal':'Request identity','confidence':.97,'identity_attribute':'DISPLAY_NAME'}
        self.assertEqual(resolve_wire(identity).identity_attribute,'DISPLAY_NAME')

    def test_foreign_explanation_uses_only_existing_non_authority_canonicalization(self):
        result=resolve_wire(action(explanation_relation='unsupported'))
        self.assertEqual(result.operation_intent.value,'CREATE_NEW')
        self.assertEqual(result.explanation_relation.value,'UNSPECIFIED')
        with self.assertRaises(ResolutionSchemaError):
            resolve_wire({'intent':'EXPLAIN_FAILURE','speech_act':'INFORMATION_REQUEST',
                'primary_goal':'Explain','confidence':.97,'explanation_relation':'unsupported'})

    def test_unknown_root_field_is_not_moved_silently(self):
        raw=clarification(clarification_target={'key':'language','question':'Which language?'})
        with self.assertRaises(ResolutionSchemaError) as rejected:
            resolve_wire(raw)
        self.assertEqual(rejected.exception.protocol_diagnostics['unexpected_field_paths'],['$.clarification_target'])
        self.assertEqual(rejected.exception.resolution_fields['intent'],'CLARIFICATION')

    def test_same_typed_artifact_is_projected_without_invention(self):
        raw=action(task_proposal=prior.update());raw.pop('artifact_type')
        before=deepcopy(raw)
        self.assertEqual(resolve_wire(raw).output_kind.value,'CODE_ARTIFACT')
        self.assertEqual(raw,before)
        raw['task_proposal'].pop('artifact_type')
        with self.assertRaisesRegex(ResolutionSchemaError,'required_artifact_type'):
            resolve_wire(raw)

    def test_conflicting_typed_artifact_is_still_rejected(self):
        raw=action(task_proposal=prior.update(artifact_type='DOCUMENT'))
        with self.assertRaisesRegex(ResolutionSchemaError,'artifact_type_conflict'):
            resolve_wire(raw)

    def test_actual_action_cannot_claim_information_or_ack_act(self):
        for act in ('INFORMATION_REQUEST','ACKNOWLEDGEMENT'):
            with self.assertRaisesRegex(ResolutionSchemaError,'speech_act_conflict'):
                resolve_wire(action(speech_act=act))


class Repair(unittest.TestCase):
    def test_repair_of_typed_task_contradiction_cannot_become_ack(self):
        owner,model=classifier(action(references=['old.py']),
            {'intent':'CONVERSATION','speech_act':'ACKNOWLEDGEMENT','primary_goal':'Create a small source','confidence':.97})
        result=InteractionResolver(model).resolve('Prepare a new source.',state=STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        instructions=model.complete.call_args_list[1].args[0][0]['content']
        self.assertIn('"intent": "CREATE_NEW"',instructions)

    def test_repair_cannot_gain_execution_via_missing_bool(self):
        original=action(unknown='invalid',execute=False,verification_requirement='BASIC')
        changed=action(execution_intent='RUN',verification_requirement='EXECUTION')
        owner,model=classifier(original,changed)
        result=InteractionResolver(model).resolve('Prepare source only.',state=STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertFalse(result.requires_execution)

    def test_nonstring_discriminator_has_safe_shape_diagnostics(self):
        bad={'intent':{'bad':'value'},'speech_act':'ACTION_REQUEST','primary_goal':'private goal','confidence':.9}
        owner,model=classifier(bad,bad)
        result=InteractionResolver(model).resolve('A request.',state=STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        events=[d for e,d in owner.audit.events if e=='classifier_protocol_shape']
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['intent'],'INVALID')
        self.assertNotIn('private goal',json.dumps(owner.audit.events))

    def test_observed_misnested_shape_repairs_without_changing_task_clarification(self):
        owner,model=classifier(clarification(clarification_target={'key':'language','question':'Which language?'}),
            clarification())
        resolver=InteractionResolver(model)
        result=resolver.resolve('Prepare a source with a material language choice.',state=STATE)
        self.assertEqual(result.semantic_intent,'CLARIFICATION')
        self.assertEqual(result.kind.value,'CLARIFICATION')
        self.assertEqual(model.complete.call_count,2)
        request=json.loads(model.complete.call_args_list[1].args[0][1]['content'])
        instructions=model.complete.call_args_list[1].args[0][0]['content']
        self.assertIn('TASK_REQUEST_CLARIFICATION',instructions)
        self.assertIn('"additionalProperties": false',instructions)
        self.assertNotIn('task_request_contract',request)
        self.assertEqual(request['fields_in_conflict']['protocol_diagnostics']['unexpected_field_names'],['clarification_target'])

    def test_repair_cannot_swap_task_for_ack_identity_or_explanation(self):
        alternatives=[{'intent':'CONVERSATION','speech_act':'ACKNOWLEDGEMENT','conversation_purpose':'GENERAL'},
            {'intent':'PERSONAL_IDENTITY','speech_act':'INFORMATION_REQUEST','identity_attribute':'DISPLAY_NAME'},
            {'intent':'EXPLAIN_FAILURE','speech_act':'INFORMATION_REQUEST'}]
        for alternate in alternatives:
            owner,model=classifier(action(unknown_marker='not used'),
                {'primary_goal':'Create a small source','confidence':.97,**alternate})
            result=InteractionResolver(model).resolve('Prepare a new source.',state=STATE)
            self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
            self.assertEqual(model.complete.call_count,2)
            self.assertFalse(result.requested_effects)
            detail=[d for e,d in owner.audit.events if e=='classifier_protocol_shape'][-1]
            self.assertEqual(detail['reason_code'],'INTERACTION_SCHEMA_TASK_REQUEST_REPAIR_CHANGED_SEMANTICS')

    def test_repair_cannot_convert_unanswered_clarification_to_creation(self):
        owner,model=classifier(clarification(clarification_target={'key':'language','question':'Which language?'}),action())
        result=InteractionResolver(model).resolve('Prepare a program.',state=STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertEqual(model.complete.call_count,2)

    def test_diagnostics_store_names_enums_types_hashes_not_prose(self):
        private='PRIVATE_PAYLOAD_DO_NOT_LOG'
        bad=action(private_notes=private,primary_goal=private)
        owner,model=classifier(bad,bad)
        InteractionResolver(model).resolve('A harmless new task.',state=STATE)
        events=[d for e,d in owner.audit.events if e=='classifier_protocol_shape']
        self.assertEqual(events[0]['unexpected_field_names'],['private_notes'])
        self.assertEqual(events[0]['classifier_branch'],'ARTIFACT_ACTION')
        self.assertEqual(events[0]['intent'],'CREATE_NEW')
        self.assertNotIn(private,json.dumps(owner.audit.events))
        self.assertIn('response_summary',str(owner.audit.events))

    def test_missing_required_fields_are_observable(self):
        bad=action();bad.pop('primary_goal')
        owner,model=classifier(bad,bad)
        InteractionResolver(model).resolve('A task.',state=STATE)
        detail=[d for e,d in owner.audit.events if e=='classifier_protocol_shape'][0]
        self.assertIn('primary_goal',detail['missing_required_field_names'])


class ControllerBoundary(unittest.TestCase):
    setUp=prior.TaskIntegration.setUp
    tearDown=prior.TaskIntegration.tearDown
    controller=prior.TaskIntegration.controller

    def test_new_topic_task_clarification_creates_pending_contract_without_tools(self):
        owner,model=classifier(clarification())
        c=self.controller();c.interaction_resolver.classifier=model
        c.handle('Create a simple source with a material choice.',turn_id='new-topic')
        self.assertIsNotNone(c.state.pending_task.semantic_contract)
        self.assertIn('language',c.state.pending_task.semantic_contract.unresolved_questions)
        self.assertFalse(c._executor_called or c._reasoner_called)

    def test_terminal_classifier_failure_preserves_nonempty_publication_boundary(self):
        from anydes_tools.conversation.outcome_voice import naturalize_final
        from anydes_tools.conversation.publication import FinalResponseComposer
        from anydes_tools.conversation.publication_boundary import require_visible
        owner,model=classifier(action(unknown='invalid'),
            {'intent':'CONVERSATION','speech_act':'ACKNOWLEDGEMENT','primary_goal':'Create a small source','confidence':.97})
        c=self.controller();c.interaction_resolver.classifier=model
        c.handle('Create a source.',turn_id='failure')
        self.assertFalse(c._executor_called or c._reasoner_called)
        writer,_,_=prior.voice_fixture.OutcomeVoiceTests().assistant([
            {'text':'No consegui entender bien esa peticion; podemos retomarla.'},prior.review_payload()])
        outcome=c.last_runtime_outcome
        final=naturalize_final(writer,c,outcome,FinalResponseComposer.compose(outcome),'Create a source.')
        self.assertTrue(require_visible(final).visible_text.strip())
        self.assertEqual(final.visible_text,final.spoken_text)
