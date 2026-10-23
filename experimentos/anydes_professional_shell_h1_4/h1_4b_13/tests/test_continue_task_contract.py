"""One semantic delta/repair; no phrase classification or permission shortcuts."""
from dataclasses import replace
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from anydes_tools.tests import test_manual_flow as prior
from anydes_tools.conversation.task_continuity import TaskContract
from anydes_tools.conversation.operation_intent import semantic_resolution_payload,ResolutionSchemaError
from anydes_tools.conversation.interaction_router import InteractionResolution as R,InteractionResolver,TaskControl
from anydes_tools.conversation.publication import RuntimeOutcome,FinalResponseComposer,OutcomeStatus


def payload(intent='CONTINUE_TASK', **extra):
    return {'intent':intent,'speech_act':'ACKNOWLEDGEMENT','primary_goal':'Create a small program',
        'confidence':.98,**extra}


def contract():
    decision=prior.wire('CLARIFICATION',task_update=prior.update(target=('input_kind','What input type?')))
    result=TaskContract.start(decision,decision.task_proposal)
    result.merge(decision.task_proposal,'first')
    return result


def state(task):
    return {'pending_action':False,'failure_available':False,'retry_available':False,
        'resolution_context':{'previous_assistant_act':'CLARIFICATION_REQUEST',
            'active_task':{'task_id':task.task_id,'semantic_contract':task.snapshot()},
            'pending_confirmation':{'available':False,'task_id':task.task_id}}}


def resolve(raw,context):
    update=raw.get('task_update')
    task=(context.get('resolution_context',{}).get('active_task') or {}).get('semantic_contract') or {}
    if update and task and 'task_id' not in update:
        raw={**raw,'task_update':{**update,'task_id':task['task_id']}}
    return R.from_dict(semantic_resolution_payload(raw,context))


class ContinuationContract(unittest.TestCase):
    def test_contract_fits_existing_provider_context_limit(self):
        from anydes_tools.conversation.resolution_contract import PROMPT
        self.assertLessEqual(len(PROMPT),16000)

    def test_answer_binds_same_task_and_resolves_target(self):
        task=contract();context=state(task)
        decision=resolve(payload(task_update=prior.update('CONTINUE',[('input_kind','whole numbers')])),context)
        self.assertEqual(decision.semantic_intent,'CONTINUE_TASK')
        self.assertEqual(decision.task_update['task_id'],task.task_id)
        task.merge(decision.task_update,'answer')
        self.assertFalse(task.unresolved_questions)
        self.assertIn('input_kind',task.resolved_targets)

    def test_second_answer_keeps_first_requirement_and_provenance(self):
        task=contract()
        task.merge(prior.update('CONTINUE',[('input_kind','integers')],('output_style','Which layout?')),'second')
        task.merge(prior.update('CONTINUE',[('output_style','one per line')]),'third')
        self.assertEqual(task.requirements['input_kind']['value'],'integers')
        self.assertEqual(task.requirements['output_style']['value'],'one per line')
        self.assertEqual([r['turn_id'] for r in task.history],['second','third'])

    def test_missing_update_one_repair_preserves_branch_and_fills_delta(self):
        task=contract();context=state(task);calls=[]
        queue=[payload(),payload(task_update=prior.update('CONTINUE',[('input_kind','whole numbers')]))]
        def classifier(request):
            calls.append(request)
            return resolve(queue.pop(0),request['state'])
        result=InteractionResolver(classifier).resolve('Use whole numbers for that.',state=context)
        self.assertEqual(result.semantic_intent,'CONTINUE_TASK')
        self.assertEqual(len(calls),2)
        repair=calls[1]['task_continuation_contract']
        self.assertEqual(repair['task_id'],task.task_id)
        self.assertIn('input_kind',repair['unresolved_questions'])
        self.assertIn('task_update',repair['required_fields'])
        self.assertEqual(calls[1]['allowed_task_controls'],['NONE'])

    def test_repair_cannot_invent_confirmation(self):
        task=contract();context=state(task)
        queue=[payload(),payload('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION')]
        resolver=InteractionResolver(lambda request:resolve(queue.pop(0),request['state']))
        result=resolver.resolve('My answer to the pending question.',state=context)
        self.assertEqual(result.source,'TASK_UPDATE_UNRESOLVED')
        self.assertEqual(len(resolver.validation_history),2)
        self.assertIn('task_continuation_repair_changed_branch',resolver.validation_history[-1]['errors'])
        self.assertEqual(task.task_id,context['resolution_context']['active_task']['task_id'])

    def test_real_pending_confirmation_remains_valid(self):
        context=state(contract());active=context['resolution_context']['active_task']
        active['plan_id']='plan:current'
        context['pending_action']=True
        context['resolution_context']['pending_confirmation']={'available':True,
            'task_id':active['task_id'],'plan_id':'plan:current','confirmation_expected':True}
        resolver=InteractionResolver(lambda request:resolve(payload('CONFIRM',
            pending_action_relation='ACCEPT_PENDING_ACTION'),request['state']))
        result=resolver.resolve('Proceed with the approved scope.',state=context)
        self.assertEqual(result.task_control,TaskControl.CONFIRM)
        self.assertEqual(len(resolver.validation_history),1)

    def test_confirmation_task_plan_and_expected_guards(self):
        c=state(contract());c['pending_action']=True
        c['resolution_context']['active_task']['plan_id']='plan:current'
        decision=resolve(payload('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION'),c)
        for changed,reason in [({'task_id':'different'},'confirmation_task_id_mismatch'),
                ({'plan_id':'different'},'confirmation_plan_id_mismatch'),
                ({'confirmation_expected':False},'confirmation_not_expected')]:
            pending={'available':True,'task_id':c['resolution_context']['active_task']['task_id'],
                'plan_id':'plan:current','confirmation_expected':True,**changed}
            s={**c,'resolution_context':{**c['resolution_context'],'pending_confirmation':pending}}
            self.assertIn(reason,InteractionResolver.contradictions(decision,s))

    def test_wrong_task_id_rejected_not_rebound(self):
        task=contract()
        change={**prior.update('CONTINUE',[('input_kind','whole numbers')]),'task_id':'task:other'}
        with self.assertRaisesRegex(ResolutionSchemaError,'TASK_UPDATE_TASK_ID_MISMATCH'):
            resolve(payload(task_update=change),state(task))

    def test_completed_task_is_not_a_pending_resume(self):
        task=contract();task.merge(prior.update('CONTINUE',[('input_kind','integers')]),'answer')
        context=state(task);context['resolution_context']['active_task']['status']='COMPLETED'
        with self.assertRaisesRegex(ResolutionSchemaError,'requires_pending_task'):
            resolve(payload(turn_relation='TOPIC_CONTINUATION'),context)

    def test_answer_must_address_existing_target(self):
        task=contract()
        for update in (prior.update('CONTINUE'),prior.update('CONTINUE',[('unrelated','a value')])):
            with self.assertRaisesRegex(ResolutionSchemaError,'pending_clarification_answer_required'):
                resolve(payload(task_update=update),state(task))

    def test_noop_resume_allowed_only_without_open_question(self):
        task=contract();task.merge(prior.update('CONTINUE',[('input_kind','integers')]),'answer')
        result=resolve(payload(turn_relation='TOPIC_CONTINUATION'),state(task))
        self.assertEqual(result.task_update['requirements'],[])
        self.assertEqual(result.task_update['task_id'],task.task_id)
        with self.assertRaises(ResolutionSchemaError):
            resolve(payload(turn_relation='TOPIC_CONTINUATION'),state(contract()))

    def test_revision_supersedes_only_affected_field(self):
        task=contract();task.merge(prior.update('CONTINUE',[('input_kind','integers'),('style','short')]),'answer')
        task.merge(prior.update('REVISE',[('style','detailed')]),'revision')
        self.assertEqual(task.requirements['input_kind']['value'],'integers')
        self.assertTrue(next(r for r in task.history if r['key']=='style')['superseded'])


class ControllerContinuity(unittest.TestCase):
    setUp=prior.TaskIntegration.setUp
    tearDown=prior.TaskIntegration.tearDown
    controller=prior.TaskIntegration.controller

    def test_exhausted_repair_keeps_pending_state_and_natural_nonempty_reply(self):
        task=contract();c=self.controller()
        c.state.pending_task.semantic_contract=task
        c.state.pending_task.task_id=task.task_id
        c.state.pending_task.status='AWAITING_CLARIFICATION'
        c.state.previous_assistant_act='CLARIFICATION_REQUEST'
        queue=[payload(),payload('CONFIRM',pending_action_relation='ACCEPT_PENDING_ACTION')]
        c.interaction_resolver.classifier=lambda request:resolve(queue.pop(0),request['state'])
        response=c.handle('Select whole numbers for the question.',turn_id='failed-repair')
        self.assertEqual(c.last_runtime_outcome.status,OutcomeStatus.CLARIFICATION)
        self.assertEqual(c.state.pending_task.task_id,task.task_id)
        self.assertIs(c.state.pending_task.semantic_contract,task)
        self.assertIn('input_kind',task.unresolved_questions)
        self.assertFalse(c._executor_called or c._reasoner_called)
        from anydes_tools.conversation.outcome_voice import naturalize_final
        from anydes_tools.conversation.publication_boundary import require_visible
        writer,_,_=prior.voice_fixture.OutcomeVoiceTests().assistant([
            {'text':'Para ese programa, que tipo de entrada prefieres?'},prior.review_payload()])
        final=naturalize_final(writer,c,c.last_runtime_outcome,
            FinalResponseComposer.compose(c.last_runtime_outcome),'Select a type')
        self.assertTrue(require_visible(final).visible_text.strip())
        self.assertEqual(final.visible_text,final.spoken_text)

    def test_unrelated_task_starts_new_contract_without_merging(self):
        old=contract();c=self.controller()
        c.state.pending_task.semantic_contract=old;c.state.pending_task.task_id=old.task_id
        request=prior.update('SWITCH',target=('audience','For whom?'),artifact_type='DOCUMENT')
        c.interaction_resolver.classifier=lambda r:prior.wire('CLARIFICATION',task_update=request)
        c.handle('Instead prepare a document for another audience.',turn_id='new-task')
        fresh=c.state.pending_task.semantic_contract
        self.assertNotEqual(fresh.task_id,old.task_id)
        self.assertEqual(fresh.artifact_type,'DOCUMENT')
        self.assertNotIn('input_kind',fresh.unresolved_questions)
