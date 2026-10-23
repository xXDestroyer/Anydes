"""Known internal metadata is separate from task semantics, IDs and public schema."""
import json
from copy import deepcopy
import unittest

from anydes_tools.conversation.classifier_output import build_public,accept_task_semantics
from anydes_tools.conversation.operation_intent import semantic_resolution_payload,ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolver
from anydes_tools.tests import test_classifier_request_contract as fixtures
from anydes_tools.tests import test_continue_task_contract as prior


class ContinuationPublic(unittest.TestCase):
    def setUp(self):
        self.task=prior.contract()
        self.state=prior.state(self.task)
        self.raw=prior.payload(task_update={'task_id':self.task.task_id,'relation':'CONTINUE',
            'requirements':[{'key':'input_kind','value':'integers'}]})

    def test_valid_public_instance_has_only_allowed_fields(self):
        self.assertEqual(build_public(self.raw,self.state),self.raw)
        self.assertEqual(prior.resolve(self.raw,self.state).task_update['task_id'],self.task.task_id)

    def test_output_is_internal_not_response_publication(self):
        raw={**self.raw,'output':'PRIVATE_NOT_A_RESPONSE'}
        self.assertEqual(build_public(raw,self.state),self.raw)
        self.assertNotIn('output',semantic_resolution_payload(raw,self.state))

    def test_model_context_is_internal_never_authority(self):
        raw={**self.raw,'resolution_context':{'active_task':{'task_id':'task:forged'},'private':'do not log'}}
        result=build_public(raw,self.state)
        self.assertEqual(result,self.raw)
        self.assertEqual(result['task_update']['task_id'],self.task.task_id)

    def test_internal_projection_leaves_semantics_and_input_unchanged(self):
        raw={**self.raw,'output':{'text':'not public'},'resolution_context':{'private':True}}
        before=deepcopy(raw)
        self.assertEqual(build_public(raw,self.state),self.raw)
        self.assertEqual(raw,before)

    def test_task_proposal_id_is_forbidden_not_stripped(self):
        with self.assertRaises(ResolutionSchemaError) as rejected:
            semantic_resolution_payload({**self.raw,'task_proposal':{'task_id':self.task.task_id}},self.state)
        self.assertEqual(rejected.exception.constraint,'TASK_PROPOSAL_ID_NOT_ALLOWED')
        self.assertEqual(rejected.exception.protocol_diagnostics['field_path'],'$.task_proposal.task_id')

    def test_proposal_without_id_is_still_semantic_contradiction(self):
        with self.assertRaisesRegex(ResolutionSchemaError,'CONTINUATION_PUBLIC_CONTRACT_VIOLATION'):
            semantic_resolution_payload({**self.raw,'task_proposal':{'relation':'START'}},self.state)

    def test_incompatible_relation_still_fails(self):
        for relation in ('START','SWITCH','invalid'):
            with self.assertRaises(ResolutionSchemaError):
                build_public({**self.raw,'output':'ignored','task_update':{
                    **self.raw['task_update'],'relation':relation}},self.state)

    def test_mismatched_id_not_hidden_by_internal_context(self):
        raw={**self.raw,'task_update':{**self.raw['task_update'],'task_id':'task:forged'},
            'resolution_context':{'task_id':'task:forged'}}
        with self.assertRaisesRegex(ResolutionSchemaError,'TASK_UPDATE_TASK_ID_MISMATCH'):
            semantic_resolution_payload(raw,self.state)

    def test_conflicting_requirement_update_still_rejected(self):
        self.task.merge(self.raw['task_update'],'answer')
        self.state=prior.state(self.task)
        raw={**self.raw,'output':'ignored','task_update':{**self.raw['task_update'],
            'requirements':[{'key':'input_kind','value':'strings'}]}}
        with self.assertRaisesRegex(ResolutionSchemaError,'requires_revision'):
            semantic_resolution_payload(raw,self.state)

    def test_unknown_field_and_raw_wrapper_not_permissively_stripped(self):
        for key in ('unrecognized_field','branch_contract','task_request_contract'):
            with self.assertRaises(ResolutionSchemaError) as rejected:
                semantic_resolution_payload({**self.raw,key:True},self.state)
            self.assertIn('$.'+key,rejected.exception.protocol_diagnostics['unexpected_field_paths'])

    def test_projection_not_applied_to_wrong_or_invalid_branch(self):
        for raw in ({**fixtures.action(),'output':'not accepted'},
                {**self.raw,'speech_act':'invalid','output':'not accepted'}):
            with self.assertRaises(ResolutionSchemaError):
                semantic_resolution_payload(raw,self.state)

    def test_repair_stays_continuation_exact_id(self):
        bad={**self.raw,'unknown':True}
        owner,model=fixtures.classifier(bad,{**self.raw,'output':'do not publish','resolution_context':{}})
        result=InteractionResolver(model).resolve('Use integers for that input.',state=self.state)
        self.assertEqual(result.semantic_intent,'CONTINUE_TASK')
        self.assertEqual(result.task_update['task_id'],self.task.task_id)
        self.assertIsNone(result.task_proposal)
        self.assertEqual(model.complete.call_count,2)
        system=model.complete.call_args_list[1].args[0][0]['content']
        self.assertIn('"repair_target_intent": "CONTINUE_TASK"',system)
        self.assertIn(self.task.task_id,system)
        self.assertNotIn('"output": "One public',system)

    def test_repair_cannot_change_to_new_task_with_id(self):
        bad={**self.raw,'unknown':True}
        changed=fixtures.action(task_proposal={'task_id':self.task.task_id,'relation':'START'})
        owner,model=fixtures.classifier(bad,changed)
        resolver=InteractionResolver(model)
        result=resolver.resolve('Use integers.',state=self.state)
        self.assertEqual(result.source,'TASK_UPDATE_UNRESOLVED')
        self.assertEqual(model.complete.call_count,2)
        self.assertFalse(result.requested_effects)
        events=[d for e,d in owner.audit.events if e=='classifier_protocol_shape']
        self.assertEqual(events[-1]['reason_code'],'CONTINUATION_PUBLIC_CONTRACT_VIOLATION')

    def test_diagnostics_do_not_retain_private_metadata(self):
        private='PRIVATE_METADATA_NOT_TO_LOG'
        owner,model=fixtures.classifier({**self.raw,'output':private,'resolution_context':{'private':private}})
        result=InteractionResolver(model).resolve('The chosen input is integers.',state=self.state)
        self.assertEqual(result.semantic_intent,'CONTINUE_TASK')
        encoded=json.dumps(owner.audit.events)
        self.assertNotIn(private,encoded)
        diagnostic=[d for e,d in owner.audit.events if e=='classifier_protocol_shape'][0]
        self.assertEqual(diagnostic['task_id_match_status'],'MATCH')
        self.assertEqual(diagnostic['internal_reason_codes'],[
            'INTERNAL_OUTPUT_FIELD_LEAK','INTERNAL_RESOLUTION_CONTEXT_LEAK'])
        self.assertEqual(diagnostic['proposal_origin'],'ORIGINAL')
        public=[d for e,d in owner.audit.events if e=='continuation_public_contract'][0]
        self.assertNotIn('output',public['public_field_names'])
        self.assertNotIn('resolution_context',public['public_field_names'])

    def test_terminal_failure_still_nonempty(self):
        # Reuse the accepted real publication recovery boundary without changing it.
        fixture=fixtures.ControllerBoundary('test_terminal_classifier_failure_preserves_nonempty_publication_boundary')
        fixture.setUp()
        try:
            fixture.test_terminal_classifier_failure_preserves_nonempty_publication_boundary()
        finally:
            fixture.tearDown()


if __name__=='__main__':
    unittest.main()
