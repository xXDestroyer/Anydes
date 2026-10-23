"""Prepared cognition; real lowering, validation, edit commit and publication."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import unittest

from anydes_tools.tests import test_human_acceptance_round1 as prior
from anydes_tools.tests.test_operation_grounding import semantic, resolved
from anydes_tools.tests import test_operation_grounding as operation_tests
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, ResolutionSchemaError, operation_errors
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionResolver, GoalConsistencyValidator,
    TaskControl as T, InteractionKind as K)
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.executable_plan import BrainPlanProposal
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer
from anydes_tools.conversation.failure_state import FailureRecord
from anydes_tools.conversation.publication import FinalResponseComposer


def edit_payload(change='print two instead'):
    return {'goal':'modify the bound program without executing it', 'operations':[{
        'operation_id':'edit', 'operation_kind':'EDIT_ARTIFACT',
        'inputs':['selected_artifact'],
        'outputs':[{'name':'revised','artifact_type':'CODE','format':'py'}],
        'specification':{'requested_change':change, 'new_requirements':[change],
                         'execution_required':False}}], 'requested_final_outputs':['revised']}


class EditLowering(unittest.TestCase):
    setUp = prior.Acceptance.setUp
    tearDown = prior.Acceptance.tearDown
    controller = prior.Acceptance.controller
    source = prior.Acceptance.source

    def lower(self, c, ref, payload):
        return ExecutablePlanLowerer(c.registry,self.workspace,
            external_artifacts={'selected_artifact':ref}).lower_requirements(SemanticProposal.from_dict(payload))

    def test_modification_spec_is_the_only_generation_contract(self):
        c = self.controller()
        ref = self.source(c,'current.py')
        plan = self.lower(c,ref,edit_payload())
        self.assertTrue(plan.needs_content_generation)
        self.assertEqual(plan.task_type,'MODIFY_CODE')
        self.assertEqual(set(plan.inputs),{'code_modification_spec'})
        self.assertEqual(plan.inputs['code_modification_spec']['target_artifact'],'current.py')
        self.assertEqual(GoalConsistencyValidator.validate_plan(
            resolved('EDIT_EXISTING',artifact_type='CODE'),plan),())
        self.assertEqual(plan.steps[0].arguments['path'],ref.path)

    def test_complete_semantic_edit_pipeline_keeps_lineage(self):
        proposal = BrainPlanProposal(SemanticProposal.from_dict(edit_payload()))
        c = self.controller(resolved('EDIT_EXISTING',artifact_type='CODE'),
            general_task_reasoner=lambda request:proposal,
            code_modifier=lambda spec:{'language':'python','filename':spec.target_artifact,
                                       'code':'print(2)\n','tests':[]})
        ref = self.source(c,'current.py')
        old_bytes = (self.workspace/'current.py').read_bytes()
        old_hash = ref.hash
        response = c.handle('Revise the program in focus',turn_id='edit-semantic')
        if c.state.pending_tool_action is not None:
            response = c._handle_pending('approved','edit-confirm',control=T.CONFIRM)
        updated = c.state.recent_artifact_state.ref_by_path(ref.path)
        self.assertEqual(updated.artifact_id,ref.artifact_id)
        self.assertEqual((ref.version,updated.version),(1,2))
        self.assertEqual(ref.hash,old_hash)
        self.assertEqual(hashlib.sha256(old_bytes).hexdigest(),old_hash)
        self.assertNotEqual(updated.hash,old_hash)
        self.assertEqual((self.workspace/'current.py').read_text(),'print(2)\n')

    def test_missing_modification_meaning_remains_rejected(self):
        c = self.controller()
        ref = self.source(c,'current.py')
        plan = self.lower(c,ref,edit_payload(''))
        validation = c.general_plan_validator.validate(plan,authorized_scopes=c.state.authorized_scopes)
        self.assertFalse(validation.valid)
        self.assertIn('code_modification_change_required',validation.errors)


class ExplanationContract(unittest.TestCase):
    state = operation_tests.OperationContract.state

    def test_other_operation_effect_contracts_unchanged(self):
        for intent in ('CREATE_NEW','READ_EXISTING','REPLACE_EXISTING'):
            with self.subTest(intent=intent):
                decision = resolved(intent,artifact_type='CODE')
                self.assertEqual(operation_errors(decision),())
                self.assertEqual(decision.operation_intent.value,intent)

    def test_minimal_control_is_readonly_and_same_root(self):
        for intent in ('EXPLAIN_FAILURE','CONTINUE_EXPLANATION','REMEDIATION_QUERY'):
            with self.subTest(intent=intent):
                value = InteractionResolver(lambda _:resolved(intent)).resolve('contextual question',state=self.state)
                self.assertEqual(value.kind,K.CONTENT_RESPONSE)
                self.assertEqual(value.task_control.value,intent)
                self.assertEqual(value.failure_ref,'root')
                self.assertEqual(value.requested_effects,())
                self.assertFalse(value.requires_execution or value.artifact_intent)
                self.assertFalse(any(value.effect_intent.to_dict().values()))

    def test_both_historical_constraints_keep_safe_fields(self):
        for extra, field in (({'verification_requirement':'BASIC'},'execute'),
                             ({'turn_relation':'TOPIC_CONTINUATION'},'turn_relation')):
            with self.subTest(field=field),self.assertRaises(ResolutionSchemaError) as caught:
                semantic_resolution_payload(semantic('EXPLAIN_FAILURE',**extra),self.state)
            error = caught.exception
            self.assertEqual(error.field,field)
            self.assertEqual(error.resolution_fields['intent'],'EXPLAIN_FAILURE')
            self.assertNotIn('primary_goal',error.resolution_fields)

    def test_single_schema_repair_gets_same_purpose_contract(self):
        calls = []

        def classifier(request):
            calls.append(deepcopy(request))
            raw = semantic('EXPLAIN_FAILURE',verification_requirement='BASIC') if len(calls)==1 else semantic('EXPLAIN_FAILURE')
            return R.from_dict(semantic_resolution_payload(raw,request['state']))

        result = InteractionResolver(classifier).resolve('What prevented the requested change?',state=self.state)
        self.assertEqual(result.task_control,T.EXPLAIN_FAILURE)
        self.assertEqual(result.failure_ref,'root')
        self.assertEqual(len(calls),2)
        repair = calls[1]
        self.assertEqual(repair['preserve_explanation'],{'failure_ref':'root','task_control':'EXPLAIN_FAILURE'})
        self.assertEqual(repair['allowed_task_controls'],['EXPLAIN_FAILURE'])
        self.assertEqual(repair['allowed_explanation_contract']['requested_effects'],[])
        self.assertIn('verification_requirement',repair['fields_in_conflict']['conflicting_fields'])

    def test_repair_cannot_introduce_new_action(self):
        calls = []

        def classifier(request):
            calls.append(request)
            return (resolved('EXPLAIN_FAILURE',verification_requirement='BASIC') if len(calls)==1
                    else resolved('EDIT_EXISTING',artifact_type='CODE'))

        value = InteractionResolver(classifier).resolve('Causal question',state=self.state)
        self.assertEqual(value.kind,K.CLARIFICATION)
        self.assertEqual(len(calls),2)

    def test_explicit_new_action_is_not_pinned_to_old_failure(self):
        value = InteractionResolver(lambda _:resolved('EXECUTE_EXISTING')).resolve('Run the saved program',state=self.state)
        self.assertEqual(value.operation_intent.value,'EXECUTE_EXISTING')
        self.assertEqual(value.task_control,T.NONE)


class ExplanationRendererIntegration(unittest.TestCase):
    setUp = prior.Acceptance.setUp
    tearDown = prior.Acceptance.tearDown
    controller = prior.Acceptance.controller

    def test_specific_constraint_and_depth_without_tools(self):
        c = self.controller(resolved('EXPLAIN_FAILURE'),resolved('CONTINUE_EXPLANATION'))
        root = FailureRecord.create(task_id='known-failure',stage='PLAN_VALIDATION',
            error_code='primary_goal_requires_content_generation',safe_user_message='',retryable=False,
            validation_detail={'failed_constraint':'primary_goal_requires_content_generation'})
        c.state.pending_task.task_id = root.task_id
        c.state.last_tool_failure = root
        texts = []
        for i,text in enumerate(('What went wrong with that change?','What does that imply in practice?')):
            c.handle(text,turn_id=str(i))
            self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
            self.assertFalse(c._executor_called or c._reasoner_called)
            final = FinalResponseComposer.compose(c.last_runtime_outcome)
            self.assertEqual(final.visible_text,final.spoken_text)
            self.assertIn('contenido nuevo o modificado',final.visible_text)
            self.assertNotIn(root.error_code,final.visible_text)
            texts.append(final.visible_text)
        self.assertNotEqual(texts[0],texts[1])


if __name__ == '__main__':
    unittest.main()
