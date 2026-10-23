"""Prepared semantic completions; actual identity schema, repair, provider and controller."""
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from anydes_tools.tests import test_conversational_core as prior
from anydes_tools.conversation.semantic_runtime import AssistantInteractionClassifier
from anydes_tools.conversation.operation_intent import semantic_resolution_payload
from anydes_tools.conversation.interaction_router import InteractionResolver, InteractionResolution as R
from anydes_tools.conversation.discourse_context import ConversationPurpose as P
from anydes_tools.conversation.resolution_contract import branch_contract


def identity(**fields):
    return dict(intent='PERSONAL_IDENTITY',speech_act='INFORMATION_REQUEST',
        primary_goal='Identify the requested contextual user identity attribute',confidence=.97,**fields)


class Assistant:
    def __init__(self):
        self.config=NS(user_name='Authorized Test User')
        self.audit=NS(record=Mock())


def classifier(*payloads):
    owner=Assistant()
    adapter=AssistantInteractionClassifier(owner)
    adapter.complete=Mock(side_effect=[(owner,NS(text=json.dumps(payload))) for payload in payloads])
    return owner,adapter


STATE={'failure_available':False,'pending_action':False,'retry_available':False,
       'resolution_context':{'previous_assistant_act':'CONTENT_RESPONSE'}}


class IdentityContract(unittest.TestCase):
    def test_explicit_attribute_accepted_with_no_effects(self):
        for question in ('Which name is attached to my profile?', 'What do you know me as?'):
            owner,model=classifier(identity(identity_attribute='DISPLAY_NAME'))
            result=InteractionResolver(model).resolve(question,state=STATE)
            self.assertEqual(result.conversation_purpose,P.PERSONAL_IDENTITY)
            self.assertEqual(result.identity_attribute,'DISPLAY_NAME')
            self.assertFalse(result.requested_effects or result.requires_execution)
            self.assertEqual(model.complete.call_count,1)

    def test_missing_and_unspecified_attribute_require_schema_repair(self):
        for missing in (identity(),identity(identity_attribute='UNSPECIFIED')):
            owner,model=classifier(missing,identity(identity_attribute='DISPLAY_NAME'))
            result=InteractionResolver(model).resolve('Which name identifies me here?',state=STATE)
            self.assertEqual(result.identity_attribute,'DISPLAY_NAME')
            self.assertEqual(model.complete.call_count,2)
            request=json.loads(model.complete.call_args_list[1].args[0][1]['content'])
            contract=request['identity_attribute_contract']
            self.assertEqual(contract['intent'],'PERSONAL_IDENTITY')
            self.assertEqual(contract['current_identity_attribute'],'UNSPECIFIED')
            self.assertEqual(contract['allowed_identity_attributes'],['DISPLAY_NAME','UNSUPPORTED'])
            self.assertEqual(request['user_request'],'Which name identifies me here?')
            self.assertEqual(request['state'],STATE)
            self.assertEqual(request['branch_contract']['required_fields'],['identity_attribute'])

    def test_repeated_unspecified_is_bounded_and_safe(self):
        owner,model=classifier(identity(),identity(identity_attribute='UNSPECIFIED'))
        result=InteractionResolver(model).resolve('A query with unresolved attribute',state=STATE)
        self.assertEqual(model.complete.call_count,2)
        self.assertEqual(result.source,'IDENTITY_ATTRIBUTE_UNRESOLVED')
        self.assertFalse(result.requested_effects or result.requires_execution)

    def test_repair_cannot_switch_to_another_domain_or_action(self):
        for payload in (
            dict(intent='PERSONAL_MEMORY_RECALL',speech_act='INFORMATION_REQUEST',primary_goal='Recall',confidence=.97),
            dict(intent='SELF_PREFERENCE',speech_act='INFORMATION_REQUEST',primary_goal='Choose',confidence=.97),
            dict(intent='CREATE_NEW',speech_act='ACTION_REQUEST',artifact_type='CODE',primary_goal='Create',confidence=.97)):
            owner,model=classifier(identity(),payload)
            result=InteractionResolver(model).resolve('Resolve my identity attribute',state=STATE)
            self.assertEqual(result.source,'IDENTITY_ATTRIBUTE_UNRESOLVED')
            self.assertEqual(model.complete.call_count,2)
            self.assertFalse(result.requested_effects)

    def test_identity_cannot_acquire_execution_effects(self):
        payload=identity(identity_attribute='DISPLAY_NAME',execute=True,verification_requirement='EXECUTION')
        owner,model=classifier(payload,payload)
        result=InteractionResolver(model).resolve('Contextual identity question',state=STATE)
        self.assertEqual(result.source,'IDENTITY_ATTRIBUTE_UNRESOLVED')
        self.assertFalse(result.requires_execution)

    def test_unsupported_attribute_is_not_mapped_to_display_name(self):
        owner,model=classifier(identity(identity_attribute='UNSUPPORTED'))
        result=InteractionResolver(model).resolve('A different identity attribute',state=STATE)
        self.assertEqual(result.identity_attribute,'UNSUPPORTED')
        self.assertEqual(branch_contract(identity())['declared_branch'],'IDENTITY')

    def test_broad_memory_and_self_keep_their_domains(self):
        for intent,purpose in [('PERSONAL_MEMORY_RECALL',P.PERSONAL_MEMORY),('SELF_PREFERENCE',P.SELF_PREFERENCE)]:
            owner,model=classifier(dict(intent=intent,speech_act='INFORMATION_REQUEST',
                primary_goal='Independent conversational question',confidence=.97))
            result=InteractionResolver(model).resolve('A different conversational domain',state=STATE)
            self.assertEqual(result.conversation_purpose,purpose)
            self.assertEqual(model.complete.call_count,1)


class IdentityController(unittest.TestCase):
    setUp=prior.DomainController.setUp
    tearDown=prior.DomainController.tearDown
    controller=prior.DomainController.controller
    root=prior.DomainController.root

    def test_active_failure_does_not_hijack_repaired_identity(self):
        owner,model=classifier(identity(identity_attribute='UNSPECIFIED'),identity(identity_attribute='DISPLAY_NAME'))
        c=self.controller()
        historical=self.root(c)
        c.interaction_resolver.classifier=model
        response=c.handle('Which name do you associate with me?',turn_id='identity')
        self.assertIsNone(response)
        self.assertEqual(c.last_resolution.identity_attribute,'DISPLAY_NAME')
        self.assertIsNone(c.state.failure_topic_record())
        self.assertIn(historical,c.state.failure_history)
        self.assertFalse(c._executor_called or c._reasoner_called)

    def test_unresolved_attribute_gets_identity_clarification_not_missing_fact(self):
        owner,model=classifier(identity(),identity(identity_attribute='UNSPECIFIED'))
        c=self.controller()
        c.interaction_resolver.classifier=model
        response=c.handle('An unresolved identity query',turn_id='missing-attribute')
        self.assertEqual(response.debug['response_source'],'IDENTITY_ATTRIBUTE_CLARIFICATION')
        self.assertIn('determinar',response.text)
        self.assertNotIn('respaldado',response.text)
        self.assertEqual(c.last_runtime_outcome.status.value,'CLARIFICATION')
        self.assertFalse(c._executor_called or c._reasoner_called)


class IdentityGrounding(unittest.TestCase):
    adapter=prior.DomainContracts.adapter

    def test_evidence_exists_host_value_wins_over_model_value(self):
        result,validation,events=self.adapter('PERSONAL_IDENTITY',{'text':'Invented User'},
            identity_attribute='DISPLAY_NAME',user_name='Context Example')
        self.assertIn('Context Example',result.text)
        self.assertNotIn('Invented',result.text)
        event=next(d for e,d in events if e=='conversation_identity_grounding')
        self.assertEqual(event['status'],'GROUNDED')
        self.assertTrue(event['evidence_id'])
        self.assertNotIn('Context Example',str(events))

    def test_no_evidence_is_honest_and_does_not_call_broad_memory(self):
        result,validation,events=self.adapter('PERSONAL_IDENTITY',{'text':'Invented User'},
            identity_attribute='DISPLAY_NAME',user_name='')
        self.assertIn('No tengo ese dato',result.text)
        self.assertNotIn('Invented',result.text)
        event=next(d for e,d in events if e=='conversation_identity_grounding')
        self.assertEqual(event['status'],'NO_DATA')
        self.assertFalse(event['evidence_id'])
        self.assertFalse(any(e=='conversation_memory_retrieval' for e,d in events))


if __name__=='__main__':
    unittest.main()
