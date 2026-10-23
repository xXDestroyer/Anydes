"""Derived task sufficiency, independent of old or repeated model booleans."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from anydes_tools.tests import test_clarification_completeness as prepared
from anydes_tools.tests import test_manual_flow as prior
from anydes_tools.conversation.task_continuity import TaskContract


def readiness(**changes):
    return {'goal_resolvable':False,'artifact_specified':True,'material_contradictions':[],**changes}


def task():
    update=prepared.start();update['planning_readiness']=readiness()
    decision=prior.wire('CLARIFICATION',task_update=update)
    result=TaskContract.start(decision,update);result.merge(update,'start')
    return result


def finish(result):
    result.merge(prior.update('CONTINUE',[('language','Python')]),'answer1')
    result.merge(prior.update('CONTINUE',[('selection_policy','only letters')]),'answer2')
    return result


class Derivation(unittest.TestCase):
    def test_two_required_unresolved_are_not_resolvable(self):
        result=task()
        self.assertFalse(result.goal_resolvable)
        self.assertEqual(len(result.unresolved_questions),2)

    def test_first_answer_keeps_false_and_remaining_target(self):
        result=task();result.merge(prior.update('CONTINUE',[('language','Python')]),'first')
        self.assertFalse(result.goal_resolvable)
        self.assertEqual(result.planning_gate()['next_target'],'selection_policy')

    def test_second_answer_recomputes_true_without_readiness_delta(self):
        result=finish(task())
        self.assertTrue(result.goal_resolvable)
        self.assertTrue(result.planning_gate()['ready'])
        self.assertEqual(result.snapshot()['planning_readiness']['goal_resolvable'],True)

    def test_false_model_hint_cannot_override_complete_task(self):
        result=finish(task())
        result.merge({**prior.update('CONTINUE'),'planning_readiness':readiness()},'model-hint')
        self.assertTrue(result.goal_resolvable)
        self.assertTrue(result.planning_gate()['ready'])

    def test_resolved_requirement_provenance_stays_and_cannot_reopen(self):
        result=finish(task());before=deepcopy(result.history)
        result.planning_gate();result.snapshot();result.planning_gate()
        self.assertFalse(result.unresolved_questions)
        self.assertEqual(result.history,before)
        self.assertEqual(result.requirements['language']['turn_id'],'answer1')
        self.assertEqual(result.snapshot()['requirement_states']['language'],'RESOLVED')
        with self.assertRaisesRegex(ValueError,'already_resolved'):
            result.merge(prior.update('CONTINUE',target=('language','Which language?')),'repeat')

    def test_optional_unresolved_does_not_block(self):
        result=finish(task())
        result.merge({**prior.update('CONTINUE'),
            'requirement_items':[{'key':'theme','necessity':'OPTIONAL'}]},'optional')
        self.assertTrue(result.goal_resolvable)

    def test_default_recomputes_and_preserves_provenance(self):
        result=task()
        update=prior.update('REVISE',[('language','Python')])
        update['requirement_items']=[{'key':'selection_policy','necessity':'DEFAULTABLE',
            'default_value':'use the existing selection order'}]
        result.merge(update,'default')
        self.assertTrue(result.goal_resolvable)
        self.assertEqual(result.requirements['selection_policy']['source'],'SEMANTIC_DEFAULT')
        self.assertEqual(result.requirements['selection_policy']['turn_id'],'default')

    def test_new_contradiction_false_then_resolution_true(self):
        result=finish(task())
        result.merge({**prior.update('REVISE'),
            'planning_readiness':readiness(material_contradictions=['selection_scope'])},'conflict')
        self.assertFalse(result.goal_resolvable)
        result.merge({**prior.update('REVISE'),
            'planning_readiness':readiness()},'resolved')
        self.assertTrue(result.goal_resolvable)

    def test_stale_false_is_caught_recomputed_and_audited(self):
        result=finish(task());result.goal_resolvable=False;events=[]
        gate=result.planning_gate(record=lambda e,**d:events.append((e,d)))
        self.assertTrue(gate['ready'])
        violation=[d for e,d in events if e=='pending_task_validation_error']
        self.assertEqual(violation[0]['reason_code'],'STALE_GOAL_RESOLVABILITY_STATE')
        self.assertTrue(violation[0]['recomputed'])
        self.assertEqual(violation[0]['resolved_requirements'],['language','selection_policy'])

    def test_false_has_exact_structural_reason_without_requirement_values(self):
        result=task();events=[]
        result.planning_gate(record=lambda e,**d:events.append((e,d)))
        detail=events[-1][1]
        self.assertFalse(detail['goal_resolvable'])
        self.assertEqual(detail['reason_code'],'REQUIRED_REQUIREMENT_UNRESOLVED')
        self.assertIn('artifact_intent_state',detail)
        self.assertNotIn('Python',str(detail))
        self.assertFalse(detail['values_logged'])

    def test_blank_goal_artifact_and_execution_blocks_not_relaxed(self):
        result=finish(task());result.goal=' '
        self.assertIn('GOAL_NOT_SPECIFIED',result.planning_gate()['reason_codes'])
        self.assertFalse(result.goal_resolvable)
        result.goal='Create a useful source';result.artifact_specified=False
        self.assertFalse(result.planning_gate()['ready'])
        self.assertFalse(result.goal_resolvable)
        result.artifact_specified=True;result.execution_required=True
        self.assertIn('EXECUTION_VERIFICATION_INCONSISTENT',result.planning_gate()['reason_codes'])
        self.assertTrue(result.goal_resolvable)

    def test_superseded_requirement_does_not_reopen_other_targets(self):
        result=finish(task())
        result.merge(prior.update('REVISE',[('selection_policy','all alphabetic codepoints')]),'revise')
        self.assertTrue(result.goal_resolvable)
        self.assertEqual(result.requirements['language']['turn_id'],'answer1')
        self.assertTrue(next(r for r in result.history if r['key']=='selection_policy' and r['turn_id']=='answer2')['superseded'])


class ControllerPath(unittest.TestCase):
    setUp=prior.TaskIntegration.setUp
    tearDown=prior.TaskIntegration.tearDown
    controller=prior.TaskIntegration.controller

    def test_genuine_unresolved_task_still_has_nonempty_natural_publication(self):
        from anydes_tools.conversation.publication import RuntimeOutcome,OutcomeStatus,FinalResponseComposer
        from anydes_tools.conversation.outcome_voice import naturalize_final
        from anydes_tools.conversation.publication_boundary import require_visible
        c=self.controller();c.state.pending_task.semantic_contract=task()
        pending=c.state.pending_task.semantic_contract
        c.last_resolution=prior.wire('CLARIFICATION',{'resolution_context':{'active_task':{
            'task_id':pending.task_id,'semantic_contract':pending.snapshot()}}},ambiguities=['Which input type?'])
        outcome=RuntimeOutcome(status=OutcomeStatus.CLARIFICATION,clarification='Which input type?')
        assistant,_,_=prior.voice_fixture.OutcomeVoiceTests().assistant([
            {'text':'Para ese programa, que lenguaje prefieres?'},prior.review_payload()])
        final=naturalize_final(assistant,c,outcome,FinalResponseComposer.compose(outcome),'Prepare a source')
        self.assertTrue(require_visible(final).visible_text.strip())
        self.assertEqual(final.visible_text,final.spoken_text)

    def test_old_false_model_assessment_then_creation_and_exact_location(self):
        from anydes_tools.conversation.executable_plan import BrainPlanProposal
        from anydes_tools.conversation.semantic_proposal import SemanticProposal
        start=prepared.start();start['planning_readiness']=readiness()
        updates=[('CLARIFICATION',start),
            ('CONTINUE_TASK',prior.update('CONTINUE',[('language','Python')])),
            ('CONTINUE_TASK',{**prior.update('CONTINUE',[('selection_policy','all letters')]),
                'planning_readiness':readiness()})]
        calls=[]
        def reason(request):
            calls.append(request)
            return BrainPlanProposal(SemanticProposal.from_dict(prepared.creation()))
        c=self.controller(general_task_reasoner=reason,code_generator=lambda spec:{
            'language':'python','filename':'sample.py','code':'print(42)\n','tests':[]})
        def classify(request):
            intent,update=updates.pop(0)
            return prior.wire(intent,request['state'],task_update=update)
        c.interaction_resolver.classifier=classify
        c.handle('Prepare a small program with two material choices.',turn_id='start')
        ident=c.state.pending_task.task_id
        c.handle('Use Python.',turn_id='answer1')
        self.assertFalse(c.state.pending_task.semantic_contract.goal_resolvable)
        self.assertFalse(calls)
        c.handle('Count all letters.',turn_id='answer2')
        self.assertEqual(c.state.pending_task.task_id,ident)
        self.assertTrue(c.state.pending_task.semantic_contract.goal_resolvable)
        self.assertEqual(len(calls),1)
        self.assertTrue((self.workspace/'sample.py').is_file(),c.state.last_tool_failure)
        c.interaction_resolver.classifier=lambda request:prior.wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION')
        c.handle('Where is that result?',turn_id='location')
        self.assertEqual(c._metadata_evidence['value'],str(self.workspace/'sample.py'))
        self.assertFalse(c._executor_called or c._reasoner_called)

    def test_ready_task_cannot_keep_clarification_projection(self):
        from anydes_tools.conversation.executable_plan import BrainPlanProposal
        from anydes_tools.conversation.semantic_proposal import SemanticProposal
        update=prior.update(rows=[('audience','beginners')])
        update['planning_readiness']=readiness()
        c=self.controller(general_task_reasoner=lambda request:BrainPlanProposal(
            SemanticProposal.from_dict(prepared.creation())),code_generator=lambda spec:{
            'language':'python','filename':'sample.py','code':'print(42)\n','tests':[]})
        c.interaction_resolver.classifier=lambda request:prior.wire('CLARIFICATION',task_update=update)
        c.handle('Prepare the source for beginners.',turn_id='ready')
        self.assertEqual(c.last_resolution.source,'PENDING_TASK_READY')
        self.assertTrue((self.workspace/'sample.py').exists(),c.state.last_tool_failure)

