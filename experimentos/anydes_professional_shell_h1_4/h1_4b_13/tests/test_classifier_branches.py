"""Discriminated projection; only inapplicable, non-authority metadata is disposable."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, ResolutionSchemaError
from anydes_tools.conversation.resolution_contract import canonicalize_branch_fields
from anydes_tools.conversation.interaction_router import InteractionResolver
from anydes_tools.conversation.semantic_runtime import AssistantInteractionClassifier
from anydes_tools.tests import test_planner_resilience as prior
from anydes_tools.tests.test_operation_grounding import semantic


def action(**extra):
    return semantic('CREATE_NEW',artifact_type='CODE',execution_intent='TEST',
        execute=True,verification_requirement='TEST_SUITE',turn_relation='NEW_ACTION',**extra)


class BranchContracts(unittest.TestCase):
    def test_action_discards_valid_and_invalid_foreign_metadata(self):
        for relation in ('unsupported', 'ASK_FAILURE_CAUSE', None, {'not':'an enum'}):
            raw = action(explanation_relation=relation)
            expected = {k:v for k,v in raw.items() if k!='explanation_relation'}
            self.assertEqual(canonicalize_branch_fields(raw),expected)
            self.assertIn('explanation_relation',raw)  # no mutation of model evidence
            result = semantic_resolution_payload(raw,{})
            self.assertEqual(result['operation_intent'],'CREATE_NEW')
            self.assertTrue(result['requires_execution'])
            self.assertEqual(result['verification_requirement'],'TEST_SUITE')
            self.assertEqual(result['explanation_relation'],'UNSPECIFIED')

    def test_relevant_explanation_metadata_is_strict(self):
        for intent in ('EXPLAIN_FAILURE','CONTINUE_EXPLANATION','REMEDIATION_QUERY','CLARIFICATION'):
            raw = semantic(intent,explanation_relation='unsupported')
            with self.assertRaises(ResolutionSchemaError) as raised:
                semantic_resolution_payload(raw,{})
            self.assertEqual(raised.exception.field,'explanation_relation')

    def test_invalid_discriminator_is_not_salvaged(self):
        for extra in ({'intent':'NOT_AN_OPERATION'}, {'speech_act':'UNKNOWN'},
                      {'intent':'CREATE_NEW','speech_act':'SOCIAL'}):
            raw = {**action(explanation_relation='unsupported'),**extra}
            self.assertEqual(canonicalize_branch_fields(raw),raw)
            with self.assertRaises(ResolutionSchemaError):
                semantic_resolution_payload(raw,{})

    def test_authority_fields_are_never_discarded(self):
        for extra in ({'execute':False}, {'verification_requirement':'BASIC'},
                      {'references':[{}]}, {'operation':'INVALID'},
                      {'requested_effects':['delete']}, {'permissions':['all']},
                      {'task_id':'invented'}, {'capability':'developer.run_python'}):
            raw = {**action(explanation_relation='unsupported'),**extra}
            canonical = canonicalize_branch_fields(raw)
            for key,value in extra.items():
                self.assertEqual(canonical[key],value)
            with self.assertRaises(ResolutionSchemaError):
                semantic_resolution_payload(raw,{})

    def test_other_branches_do_not_inherit_explanation_metadata(self):
        for intent in ('CONVERSATION','PERSONAL_MEMORY_RECALL','CONFIRM','CANCEL','RETRY'):
            raw = semantic(intent,explanation_relation='unsupported')
            result = semantic_resolution_payload(raw,{})
            self.assertEqual(result['explanation_relation'],'UNSPECIFIED')
            if intent in {'CONFIRM','CANCEL','RETRY'}:
                self.assertEqual(result['task_control'],intent)
            else:
                self.assertEqual(result['task_control'],'NONE')

    def test_real_error_repair_receives_branch_contract_and_original_context(self):
        assistant = prior.Assistant([],[])
        classifier = AssistantInteractionClassifier(assistant)
        invalid = {**action(),'execute':False}
        classifier.complete = Mock(side_effect=[(assistant,SimpleNamespace(text=json.dumps(invalid))),
            (assistant,SimpleNamespace(text=json.dumps(action())))])
        resolver = InteractionResolver(classifier)
        state = {'failure_available':False,'pending_action':False,'retry_available':False}
        decision = resolver.resolve('Prepare and test a small new source',state=state)
        self.assertTrue(decision.requires_execution)
        self.assertEqual(classifier.complete.call_count,2)
        repair = json.loads(classifier.complete.call_args_list[1].args[0][1]['content'])
        instructions=classifier.complete.call_args_list[1].args[0][0]['content']
        self.assertIn('ARTIFACT_ACTION',instructions)
        self.assertIn('inapplicable_fields',instructions)
        self.assertIn('strict_authority_fields',instructions)
        self.assertNotIn('branch_contract',repair)
        self.assertEqual(repair['state'],state)


class RealCaseIntegration(unittest.TestCase):
    setUp = prior.prior.Acceptance.setUp
    tearDown = prior.prior.Acceptance.tearDown
    controller = prior.prior.Acceptance.controller

    def test_observed_shape_survives_real_adapter_and_calls_reasoner(self):
        assistant = prior.Assistant([],[])
        classifier = AssistantInteractionClassifier(assistant)
        classifier.complete = Mock(return_value=(assistant,SimpleNamespace(
            text=json.dumps(action(explanation_relation='invalid unrelated enum')))))
        reasoner = Mock(side_effect=RuntimeError('test boundary reached'))
        c = self.controller(general_task_reasoner=reasoner)
        c.interaction_resolver.classifier = classifier
        c.handle('Create a simple source and check it on two inputs',turn_id='observed-shape')
        reasoner.assert_called_once()
        self.assertTrue(c._reasoner_called)
        self.assertFalse(c._executor_called)
        request = reasoner.call_args.args[0]
        self.assertTrue(request['interaction_resolution']['requires_execution'])
        self.assertEqual(request['interaction_resolution']['verification_requirement'],'TEST_SUITE')
        self.assertEqual(classifier.complete.call_count,1)
        events = [d for e,d in assistant.audit.events if e=='interaction_branch_canonicalization']
        self.assertEqual(events[0]['removed_fields'],['explanation_relation'])


if __name__ == '__main__':
    unittest.main()
