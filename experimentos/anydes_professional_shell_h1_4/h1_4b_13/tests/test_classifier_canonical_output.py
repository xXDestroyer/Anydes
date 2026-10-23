"""ID authority and canonical public instances, independent of user wording."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import unittest

from anydes_tools.conversation.classifier_output import build_public,SemanticBranch,accept_task_semantics
from anydes_tools.conversation.operation_intent import semantic_resolution_payload,ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolution as R,InteractionResolver
from anydes_tools.tests import test_classifier_request_contract as prior
from anydes_tools.tests import test_continue_task_contract as continuation


class CanonicalOutput(unittest.TestCase):
    def test_new_proposal_has_no_update_and_runtime_owns_id(self):
        decision=prior.resolve_wire(prior.clarification())
        self.assertIsNone(decision.task_update)
        self.assertNotIn('task_id',decision.task_proposal)
        controller=NS(state=NS(pending_task=NS(semantic_contract=None)),audit=NS(record=lambda *a,**k:None))
        accept_task_semantics(controller,decision,'turn:new')
        task=controller.state.pending_task.semantic_contract
        self.assertTrue(task.task_id.startswith('task:'))
        self.assertEqual(len(task.task_id),37)
        self.assertIn('language',task.unresolved_questions)

    def test_new_minimal_action_is_canonical_proposal(self):
        decision=prior.resolve_wire(prior.action())
        self.assertIsNone(decision.task_update)
        self.assertEqual(decision.task_proposal['operation'],'CREATE_NEW')

    def test_task_update_without_pending_even_if_well_formed_rejected(self):
        raw=prior.action(task_update={'relation':'START','task_id':'task:invented'})
        with self.assertRaises(ResolutionSchemaError) as rejected:
            prior.resolve_wire(raw)
        self.assertEqual(rejected.exception.constraint,'TASK_UPDATE_WITHOUT_PENDING_TASK')
        self.assertFalse(rejected.exception.protocol_diagnostics['pending_task_exists'])

    def test_new_proposal_cannot_supply_any_task_id(self):
        for bad in (None,'','task:invented'):
            raw=prior.clarification()
            raw['task_proposal']['task_id']=bad
            with self.assertRaisesRegex(ResolutionSchemaError,'RUNTIME_OWNS_NEW_TASK_ID'):
                prior.resolve_wire(raw)

    def test_continuation_exact_id_and_only_delta(self):
        task=continuation.contract();state=continuation.state(task)
        raw=continuation.payload(task_update={'task_id':task.task_id,'relation':'CONTINUE',
            'requirements':[{'key':'input_kind','value':'integers'}]})
        d=R.from_dict(semantic_resolution_payload(raw,state))
        self.assertEqual(d.task_update['task_id'],task.task_id)
        self.assertIsNone(d.task_proposal)
        task.merge(d.task_update,'answer')
        self.assertIn('input_kind',task.resolved_targets)

    def test_fabricated_stale_and_invalid_ids_are_not_rebound(self):
        task=continuation.contract();state=continuation.state(task)
        for bad,code in [('task:other','TASK_UPDATE_TASK_ID_MISMATCH'),
                ('arbitrary','TASK_UPDATE_TASK_ID_INVALID_FORMAT'),('', 'TASK_UPDATE_TASK_ID_INVALID_FORMAT'),
                (None,'TASK_UPDATE_TASK_ID_INVALID_FORMAT')]:
            raw=continuation.payload(task_update={'task_id':bad,'relation':'CONTINUE','requirements':[]})
            with self.assertRaises(ResolutionSchemaError) as rejected:
                semantic_resolution_payload(raw,state)
            self.assertEqual(rejected.exception.constraint,code)
            detail=rejected.exception.protocol_diagnostics
            self.assertEqual(detail['field_path'],'$.task_update.task_id')
            self.assertTrue(detail['expected_task_id_hash'])

    def test_missing_continuation_id_rejected(self):
        task=continuation.contract()
        with self.assertRaisesRegex(ResolutionSchemaError,'TASK_UPDATE_TASK_ID_INVALID_FORMAT'):
            semantic_resolution_payload(continuation.payload(task_update={'relation':'CONTINUE'}),continuation.state(task))

    def test_start_update_rejected_even_with_existing_id(self):
        task=continuation.contract()
        with self.assertRaisesRegex(ResolutionSchemaError,'continuation_relation_required'):
            semantic_resolution_payload(continuation.payload(task_update={'relation':'START','task_id':task.task_id}),continuation.state(task))

    def test_new_and_update_are_mutually_exclusive(self):
        task=continuation.contract();raw=prior.clarification(task_update={
            'task_id':task.task_id,'relation':'CONTINUE','requirements':[]})
        with self.assertRaisesRegex(ResolutionSchemaError,'mutually_exclusive_task_semantics'):
            semantic_resolution_payload(raw,continuation.state(task))

    def test_internal_representation_projects_only_public_fields(self):
        raw=prior.clarification()
        internal=SemanticBranch(raw,{'declared_branch':'TASK_REQUEST_CLARIFICATION'},
            {'intent':'CLARIFICATION','private_helper':True})
        result=build_public(internal,prior.STATE)
        self.assertEqual(result,raw)
        self.assertNotIn('branch_contract',result)
        self.assertNotIn('task_request_contract',result)
        self.assertEqual(prior.resolve_wire(result).semantic_intent,'CLARIFICATION')

    def test_raw_wrappers_rejected_with_exact_paths(self):
        for name in ('branch_contract','task_request_contract'):
            with self.assertRaises(ResolutionSchemaError) as rejected:
                prior.resolve_wire({**prior.action(),name:{'intent':'CREATE_NEW'}})
            self.assertEqual(rejected.exception.constraint,'INTERNAL_WRAPPER_LEAKED_TO_PUBLIC_SCHEMA')
            self.assertIn('$.'+name,rejected.exception.protocol_diagnostics['unexpected_field_paths'])

    def test_no_silent_discard_of_material_contradiction(self):
        raw=prior.action(task_proposal={**prior.prior.update(),'operation':'EDIT_EXISTING'})
        with self.assertRaisesRegex(ResolutionSchemaError,'operation_conflict'):
            prior.resolve_wire(raw)
        raw=prior.action(execute=False,execution_intent='TEST')
        with self.assertRaises(ResolutionSchemaError):
            prior.resolve_wire(raw)

    def test_new_repair_receives_instructions_not_wrapper_output_example(self):
        bad=prior.clarification();bad['unexpected']=True
        owner,model=prior.classifier(bad,prior.clarification())
        result=InteractionResolver(model).resolve('Prepare a new artifact with an unresolved parameter.',state=prior.STATE)
        self.assertEqual(result.semantic_intent,'CLARIFICATION')
        self.assertIsNone(result.task_update)
        system=model.complete.call_args_list[1].args[0][0]['content']
        request=json.loads(model.complete.call_args_list[1].args[0][1]['content'])
        self.assertIn('"pending_task_exists": false',system)
        self.assertNotIn('task_request_contract',request)
        self.assertNotIn('branch_contract',request)
        self.assertEqual(model.complete.call_count,2)

    def test_continuation_repair_preserves_injected_exact_id(self):
        task=continuation.contract();state=continuation.state(task)
        fixed=continuation.payload(task_update={'task_id':task.task_id,'relation':'CONTINUE',
            'requirements':[{'key':'input_kind','value':'integers'}]})
        owner,model=prior.classifier(continuation.payload(),fixed)
        result=InteractionResolver(model).resolve('Use integers for the requested input.',state=state)
        self.assertEqual(result.semantic_intent,'CONTINUE_TASK')
        self.assertEqual(result.task_update['task_id'],task.task_id)
        system=model.complete.call_args_list[1].args[0][0]['content']
        self.assertIn(task.task_id,system)
        self.assertEqual(model.complete.call_count,2)

    def test_repair_cannot_manufacture_new_task_update(self):
        bad=prior.action(unknown=True)
        hybrid=prior.action(task_update={'relation':'START','task_id':'task:invented'})
        owner,model=prior.classifier(bad,hybrid)
        result=InteractionResolver(model).resolve('Prepare a new source.',state=prior.STATE)
        self.assertEqual(result.source,'LOCAL_FAIL_CLOSED')
        self.assertFalse(result.requested_effects)

    def test_input_not_mutated(self):
        raw=prior.clarification();before=deepcopy(raw)
        build_public(raw,prior.STATE)
        self.assertEqual(before,raw)


if __name__=='__main__':
    unittest.main()
