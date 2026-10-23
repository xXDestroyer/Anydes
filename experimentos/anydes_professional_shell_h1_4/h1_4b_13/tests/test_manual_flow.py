"""Prepared semantic decisions; real contracts, review and publication boundaries."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from anydes_tools.conversation.publication import RuntimeOutcome, OutcomeStatus, FinalResponseComposer
from anydes_tools.conversation.publication_boundary import PublicationFailure, require_visible, recover_publication
from anydes_tools.conversation.operation_intent import semantic_resolution_payload
from anydes_tools.conversation.interaction_router import InteractionResolution, InteractionResolver
from anydes_tools.conversation.task_continuity import TaskContract, accept_update
from anydes_tools.conversation.artifact_metadata import metadata_evidence
from anydes_tools.conversation.natural_response import classify_local_response_failure
from anydes_tools.tests import test_outcome_voice as voice_fixture
from anydes_tools.tests.test_natural_response import review_payload
from anydes_tools.tests import test_human_acceptance_round1 as action_fixture


def wire(intent, state=None, **fields):
    # Prepared fixtures now use the new-task proposal rather than a START update.
    delta=fields.get('task_update')
    if delta and delta.get('relation') in {'START','SWITCH'}:
        fields['task_proposal']=fields.pop('task_update')
    elif delta and state:
        task=(state.get('resolution_context',{}).get('active_task') or {}).get('semantic_contract') or {}
        if task and 'task_id' not in delta:
            fields['task_update']={**delta,'task_id':task['task_id']}
    return InteractionResolution.from_dict(semantic_resolution_payload(dict(intent=intent,
        speech_act='INFORMATION_REQUEST' if intent=='ARTIFACT_METADATA_QUERY' else
        'SOCIAL' if intent=='CONVERSATION' else 'ACTION_REQUEST',
        primary_goal='Prepare a useful artifact',confidence=.98,**fields),state or {}))


def update(relation='START', rows=(), target=None, artifact_type='CODE'):
    result={'relation':relation,'operation':'CREATE_NEW','artifact_type':artifact_type,
        'requirements':[{'key':key,'value':value} for key,value in rows]}
    if target:
        result['clarification_target']={'key':target[0],'question':target[1]}
    return result


class PublicationTests(unittest.TestCase):
    def test_null_empty_whitespace_and_empty_voice_rejected(self):
        final=FinalResponseComposer.compose(RuntimeOutcome(),conversational_text='A valid reply.')
        for bad in (None,'','  \n\t'):
            with self.subTest(bad=bad),self.assertRaises(PublicationFailure):
                require_visible(replace(final,visible_text=bad))
        with self.assertRaises(PublicationFailure):
            require_visible(replace(final,spoken_text=''))

    def test_recovery_uses_generated_reviewed_same_payload(self):
        fixture=voice_fixture.OutcomeVoiceTests()
        text='No consegui completar mi respuesta; podemos retomarla si te parece.'
        assistant,calls,events=fixture.assistant([{'text':text},review_payload()])
        outcome=RuntimeOutcome()
        final=FinalResponseComposer.compose(outcome,conversational_text='unapproved')
        result=recover_publication(assistant,None,outcome,final,'A question',PublicationFailure())
        self.assertEqual(result.visible_text,text)
        self.assertEqual(result.spoken_text,text)
        self.assertEqual(len(calls),2)
        self.assertTrue(any(e=='publication_recovered' for e,_ in events))

    def test_recovery_exhaustion_stays_typed_and_never_publishes(self):
        fixture=voice_fixture.OutcomeVoiceTests()
        assistant,_,events=fixture.assistant([OSError('offline')])
        outcome=RuntimeOutcome()
        with self.assertRaisesRegex(PublicationFailure,'PUBLICATION_RECOVERY_UNAVAILABLE'):
            recover_publication(assistant,None,outcome,
                FinalResponseComposer.compose(outcome),'x',PublicationFailure())
        self.assertFalse(any(e=='publication_recovered' for e,_ in events))

    def test_empty_naturalization_is_recovered_before_payload_sync(self):
        from anydes_tools.conversation.runtime_patch import _compose_final_observed
        fixture=voice_fixture.OutcomeVoiceTests()
        assistant,_,_=fixture.assistant([])
        assistant.last_result=None
        with patch('anydes_tools.conversation.outcome_voice.naturalize_final',
                side_effect=lambda a,c,o,f,u:replace(f,visible_text=' ',spoken_text=' ')), \
             patch('anydes_tools.conversation.outcome_voice.generate_outcome_voice',return_value='Podemos retomar la respuesta.'), \
             patch('anydes_tools.conversation.runtime_patch._sync_final_payload') as sync:
            result=_compose_final_observed(assistant,'candidate','question',None,outcome_override=RuntimeOutcome())
        self.assertEqual(result,'Podemos retomar la respuesta.')
        sync.assert_called_once_with(assistant,result)
        self.assertEqual(assistant.last_final_response.spoken_text,result)
        from anydes.continuity_v2.math_presentation import present_math
        presentation=present_math(result)
        self.assertFalse(presentation.fallback_used)
        self.assertTrue(presentation.chat_text.strip())

    def test_review_and_grounding_never_transport(self):
        for typ,layer in [('GroundingRejected','grounding'),('GroundingInconclusive','grounding'),('ReviewerUnavailable','review')]:
            code,actual=classify_local_response_failure(lambda *a,**k:('transport_exception','transport'),
                None,None,stream_error_type=typ)
            self.assertEqual(actual,layer)
            self.assertNotIn('transport',code)


class MetadataTests(unittest.TestCase):
    setUp=action_fixture.Acceptance.setUp
    tearDown=action_fixture.Acceptance.tearDown
    controller=action_fixture.Acceptance.controller
    source=action_fixture.Acceptance.source

    def test_location_is_exact_read_only_and_controller_does_not_execute(self):
        decision=wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION')
        c=self.controller(decision)
        ref=self.source(c,'one.py')
        response=c.handle('Where did the recent result end up?',turn_id='location')
        self.assertEqual(response.debug['response_source'],'ARTIFACT_METADATA_QUERY')
        self.assertEqual(c._metadata_evidence['value'],ref.path)
        self.assertEqual(c._metadata_evidence['evidence_id'],ref.artifact_id)
        self.assertFalse(c._executor_called or c._reasoner_called)
        self.assertFalse(decision.requested_effects or decision.requires_execution)

    def test_latest_version_same_identity(self):
        c=self.controller()
        ref=self.source(c,'version.py')
        newer=replace(ref,version=2,metadata={**ref.metadata,'previous_hash':ref.hash})
        c.state.recent_artifact_state.register(newer)
        evidence=metadata_evidence(c,wire('ARTIFACT_METADATA_QUERY',metadata_field='VERSION'))
        self.assertEqual(evidence['value'],2)
        self.assertEqual(evidence['evidence_id'],ref.artifact_id)

    def test_two_artifacts_explicit_reference_beats_focus(self):
        c=self.controller()
        first=self.source(c,'first.py')
        self.source(c,'second.py')
        decision=wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION',references=[first.artifact_id])
        self.assertEqual(metadata_evidence(c,decision)['value'],first.path)

    def test_ambiguous_reference_does_not_guess(self):
        c=self.controller()
        first=self.source(c,'first.py'); second=self.source(c,'second.py')
        c.state.recent_artifact_state.focus_artifact_ids=(first.artifact_id,second.artifact_id)
        self.assertEqual(metadata_evidence(c,wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION'))['metadata_status'],'AMBIGUOUS')

    def test_no_artifact_honest_and_no_execution_authority(self):
        c=self.controller()
        self.assertEqual(metadata_evidence(c,wire('ARTIFACT_METADATA_QUERY',metadata_field='NAME'))['metadata_status'],'NOT_FOUND')
        with self.assertRaises(ValueError):
            wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION',execute=True)
        with self.assertRaises(ValueError):
            wire('ARTIFACT_METADATA_QUERY',metadata_field='secret_field')

    def test_grounded_location_wording_uses_authorized_evidence(self):
        from anydes_tools.conversation.outcome_voice import request_for,generate_outcome_voice
        c=self.controller(); ref=self.source(c,'answer.py')
        c.last_resolution=wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION')
        c._metadata_evidence=metadata_evidence(c,c.last_resolution)
        req=request_for(RuntimeOutcome(),c,'Where is it?')
        text='El archivo esta en `'+ref.path+'`'
        a,_,_=voice_fixture.OutcomeVoiceTests().assistant([{'text':text},review_payload()])
        self.assertEqual(generate_outcome_voice(a,req),text)


class TaskTests(unittest.TestCase):
    def contract(self,kind='CODE'):
        d=wire('CLARIFICATION',task_update=update(target=('input','Which input?'),artifact_type=kind))
        t=TaskContract.start(d,d.task_proposal);t.merge(d.task_proposal,'t1')
        return t

    def test_two_answers_accumulate_general_code_requirements(self):
        t=self.contract()
        t.merge(update('CONTINUE',[('input','whole numbers')],('format','Which output format?')),'t2')
        t.merge(update('CONTINUE',[('format','one result per line')]),'t3')
        self.assertEqual(set(t.requirements),{'input','format'})
        self.assertFalse(t.unresolved_questions)
        self.assertEqual([r['turn_id'] for r in t.history],['t2','t3'])

    def test_document_preferences_accumulate(self):
        t=self.contract('DOCUMENT')
        t.merge(update('CONTINUE',[('input','provided notes')],('audience','Who reads it?'),artifact_type='DOCUMENT'),'t2')
        t.merge(update('CONTINUE',[('audience','new students')],artifact_type='DOCUMENT'),'t3')
        self.assertEqual(t.artifact_type,'DOCUMENT')
        self.assertEqual(len(t.requirements),2)

    def test_revision_supersedes_only_one_requirement(self):
        t=self.contract()
        t.merge(update('CONTINUE',[('input','integers'),('style','brief')]),'t2')
        t.merge(update('REVISE',[('style','detailed')]),'t3')
        self.assertEqual(t.requirements['input']['value'],'integers')
        self.assertTrue(next(r for r in t.history if r['key']=='style' and r['turn_id']=='t2')['superseded'])
        self.assertFalse(t.requirements['style']['superseded'])

    def test_contradiction_requires_semantic_revision(self):
        t=self.contract();t.merge(update('CONTINUE',[('input','integers')]),'t2')
        with self.assertRaisesRegex(ValueError,'conflict'):
            t.merge(update('CONTINUE',[('input','text')]),'t3')
        self.assertEqual(t.requirements['input']['value'],'integers')

    def test_resolved_question_cannot_be_asked_again(self):
        t=self.contract();t.merge(update('CONTINUE',[('input','integers')]),'t2')
        with self.assertRaisesRegex(ValueError,'already_resolved'):
            t.merge(update('CONTINUE',target=('input','What input?')),'t3')

    def test_new_topic_suspends_no_merge(self):
        t=self.contract()
        c=NS(state=NS(pending_task=NS(semantic_contract=t)),audit=NS(record=lambda *a,**k:None))
        accept_update(c,wire('CONVERSATION'),'new')
        self.assertTrue(t.suspended)
        self.assertFalse(t.requirements)

    def test_continuation_preserves_operation_and_cannot_add_execution(self):
        t=self.contract()
        state={'resolution_context':{'active_task':{'semantic_contract':t.snapshot()}}}
        d=wire('CONTINUE_TASK',state,task_update=update('CONTINUE',[('input','integers')]))
        self.assertEqual(d.operation_intent.value,'CREATE_NEW')
        self.assertFalse(d.requires_execution)
        with self.assertRaisesRegex(ValueError,'cannot_change_authority'):
            wire('CONTINUE_TASK',state,task_update=update('CONTINUE',[('input','integers')]),execute=True)

    def test_missing_pending_contract_never_invents_continuation(self):
        with self.assertRaises(ValueError):
            wire('CONTINUE_TASK',task_update=update('CONTINUE',[('input','integers')]))

    def test_retry_does_not_erase_valid_requirements(self):
        t=self.contract();t.merge(update('CONTINUE',[('input','integers')]),'t2')
        c=NS(state=NS(pending_task=NS(semantic_contract=t)),audit=NS(record=lambda *a,**k:None))
        decision=NS(task_update=None,turn_relation=NS(value='FAILED_TASK_RETRY'))
        accept_update(c,decision,'retry')
        self.assertEqual(t.requirements['input']['value'],'integers')
        self.assertFalse(t.suspended)


class TaskIntegration(unittest.TestCase):
    setUp=action_fixture.Acceptance.setUp
    tearDown=action_fixture.Acceptance.tearDown
    controller=action_fixture.Acceptance.controller

    def test_two_clarifications_same_task_reach_real_creation(self):
        from anydes_tools.conversation.semantic_proposal import SemanticProposal
        from anydes_tools.conversation.executable_plan import BrainPlanProposal
        starts=update(target=('input','What input should be accepted?'))
        updates=[('CLARIFICATION',starts),('CONTINUE_TASK',update('CONTINUE',
            [('input','accept whole numbers')],('presentation','How should results be shown?'))),
            ('CONTINUE_TASK',update('CONTINUE',[('presentation','one result per line')]))]
        requests=[]; generated=[]
        payload={'goal':'Create a whole number formatter','operations':[{
            'operation_id':'make','operation_kind':'GENERATE_CODE','inputs':[],
            'outputs':[{'name':'program','artifact_type':'CODE','format':'py','filename':'formatter.py'}],
            'specification':{'language':'python','goal':'Print whole numbers on individual lines',
                'functional_requirements':['read a whole number and print it'],
                'execution_required':False,'verification_requirement':'BASIC','tests':[]}}],
            'requested_final_outputs':['program']}
        def planner(request):
            requests.append(request)
            return BrainPlanProposal(SemanticProposal.from_dict(payload))
        def generator(spec):
            generated.append(spec)
            return {'language':'python','filename':'formatter.py','code':'value = int(input())\nprint(value)\n','tests':[]}
        c=self.controller(general_task_reasoner=planner,code_generator=generator)
        def classify(request):
            intent,change=updates.pop(0)
            return wire(intent,request['state'],task_update=change)
        c.interaction_resolver.classifier=classify
        c.handle('Make a small formatter, first ask me what input to use.',turn_id='first')
        task=c.state.pending_task.task_id
        self.assertTrue(c.state.pending_task.semantic_contract.unresolved_questions)
        c.handle('Use whole numbers.',turn_id='second')
        self.assertEqual(c.state.pending_task.task_id,task)
        self.assertFalse(c._executor_called)
        response=c.handle('Put one result on each line.',turn_id='third')
        self.assertEqual(c.state.pending_task.task_id,task)
        self.assertEqual(set(c.state.pending_task.semantic_contract.requirements),{'input','presentation'})
        self.assertEqual(requests[-1]['pending_task_contract']['task_id'],task)
        self.assertTrue((self.workspace/'formatter.py').is_file(),(response,c.state.last_tool_failure))
        self.assertTrue(generated)
        self.assertIn('accept whole numbers',generated[-1].functional_requirements)
        self.assertIn('one result per line',generated[-1].functional_requirements)
        ref=c.state.recent_artifact_state.ref_by_path(str(self.workspace/'formatter.py'))
        self.assertEqual(ref.task_id,task)


class PlannerDiagnostics(unittest.TestCase):
    def test_missing_runner_diagnostic_does_not_change_rejection(self):
        from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
        from anydes_tools.conversation.controller import ToolConversationController
        events=[]
        c=NS(registry=NS(find_by_capability=lambda c:True),
            state=NS(active_workspace='.'), audit=NS(record=lambda e,**k:events.append((e,k))))
        plan=GeneralTaskPlan.from_dict({'task_id':'known','task_type':'GENERATE_CODE','goal':'x',
            'inputs':{'code_spec':{'goal':'x','execution_required':True}},
            'required_capabilities':['developer.write_code']})
        result=ToolConversationController._action_from_general_plan(c,plan,'x')
        self.assertIsNone(result)
        detail=events[-1][1]
        self.assertEqual(detail['field_path'],'inputs.code_spec.execution_required')
        self.assertEqual(detail['reason_code'],'EXECUTION_CAPABILITY_MISSING')
        self.assertEqual(detail['task_id'],'known')
        self.assertIn('proposal_id',detail)
        self.assertNotIn('x',str(detail.values()).replace('execution',''))
