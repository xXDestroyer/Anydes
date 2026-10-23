"""Prepared semantic decisions / real local contracts, Qt document, workers and B2.
These unit/integration tests are not human-runtime evidence.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'programa_principal'))
from anydes_tools.conversation.operation_intent import semantic_resolution_payload, ResolutionSchemaError
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionResolver, TaskControl as T,
    InteractionKind as K, TurnRelation as D)
from anydes_tools.conversation.publication import RuntimeOutcome, OutcomeStatus, FinalResponseComposer
from anydes_tools.conversation.typed_content import (
    ContentSegment, ContentKind, approve_literal_publication, literal_segments, code_html)
from anydes_tools.conversation.memory_grounding import select_grounded_memory, candidate_batches, MemoryResultState
from anydes_tools.conversation.discourse_context import DiscourseContext, ConversationPurpose
from anydes_tools.conversation.failure_state import FailureRecord, FailureResponseRenderer
from anydes_tools.tests import test_human_acceptance_round1 as prior
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer


def semantic(intent, **kw):
    act = 'INFORMATION_REQUEST' if intent in {'EXPLAIN_FAILURE','CONTINUE_EXPLANATION','REMEDIATION_QUERY',
        'PERSONAL_MEMORY_RECALL','CONTENT_RESPONSE','READ_EXISTING'} else 'ACTION_REQUEST'
    if intent == 'READ_EXISTING':
        act = 'ACTION_REQUEST'
    return {'intent':intent,'speech_act':act,'primary_goal':'requested goal','confidence':.9, **kw}


def resolved(intent, state=None, **kw):
    return R.from_dict(semantic_resolution_payload(semantic(intent,**kw),state or {}))


class OperationContract(unittest.TestCase):
    state={'failure_available':True,'pending_action':False,'retry_available':True,
        'active_failure':{'failure_id':'root'},'resolution_context':{
            'active_failure_topic':{'failure_id':'root'},
            'conversation_artifact_focus':[{'artifact_type':'CODE','display_name':'source.py'}]}}

    def test_edit_read_mutation_version_is_valid(self):
        d=resolved('EDIT_EXISTING',state=self.state)
        self.assertEqual(d.requested_operations,('EDIT_ARTIFACT',))
        self.assertTrue(d.effect_intent.read and d.effect_intent.mutate and d.effect_intent.version)
        self.assertFalse(d.effect_intent.replace)
        self.assertFalse(InteractionResolver.contradictions(d,self.state))

    def test_edit_then_tests_keeps_intent_and_effects(self):
        d=resolved('EDIT_EXISTING',state=self.state,execute=True,verification_requirement='TEST_SUITE')
        self.assertEqual(d.requested_operations,('EDIT_ARTIFACT','TEST_CODE'))
        self.assertTrue(d.effect_intent.version and d.effect_intent.execute)
        self.assertFalse(InteractionResolver.contradictions(d,self.state))

    def test_execute_beats_stale_failure_with_no_phrase_logic(self):
        d=resolved('EXECUTE_EXISTING',state=self.state)
        for text in ('Run the saved program now','Please execute the revised source','I want to see its output'):
            r=InteractionResolver(lambda _:d).resolve(text,state=self.state)
            self.assertEqual(r.operation_intent.value,'EXECUTE_EXISTING')
            self.assertEqual(r.task_control,T.NONE)
            self.assertTrue(r.requires_execution)

    def test_remediation_does_not_execute(self):
        d=resolved('REMEDIATION_QUERY',state=self.state)
        self.assertEqual(d.task_control,T.REMEDIATION_QUERY)
        self.assertFalse(d.requires_execution or d.requested_effects)

    def test_retry_is_not_remediation(self):
        self.assertEqual(resolved('RETRY',state=self.state).task_control,T.RETRY)
        self.assertEqual(resolved('EDIT_EXISTING',state=self.state).task_control,T.NONE)

    def test_current_action_and_remediation_relation_conflict_repair_exactly_once(self):
        calls=[]
        def classifier(request):
            calls.append(request)
            if len(calls)==1:
                return resolved('REMEDIATION_QUERY',speech_act='ACTION_REQUEST',state=self.state)
            return resolved('EXECUTE_EXISTING',state=self.state)
        result=InteractionResolver(classifier).resolve('Run it now',state=self.state)
        self.assertEqual(result.operation_intent.value,'EXECUTE_EXISTING')
        self.assertEqual(len(calls),2)
        self.assertIn('interaction_schema:intent:speech_act_conflict',calls[1]['repair_errors'])

    def test_no_redundant_model_operation_vocabulary(self):
        with self.assertRaisesRegex(ResolutionSchemaError,'unexpected_fields'):
            resolved('EDIT_EXISTING',requested_operations=['EDIT'])

    def test_operation_cannot_have_failure_relation(self):
        with self.assertRaisesRegex(ResolutionSchemaError,'current_operation_not_failure_question'):
            resolved('EXECUTE_EXISTING',turn_relation='FAILURE_REMEDIATION')

    def test_read_stays_non_mutating(self):
        d=resolved('READ_EXISTING')
        self.assertTrue(d.effect_intent.read)
        self.assertFalse(d.effect_intent.mutate or d.effect_intent.version or d.effect_intent.replace)
        with self.assertRaisesRegex(ResolutionSchemaError,'non_operational'):
            resolved('READ_EXISTING',execute=True)

    def test_requested_execution_requires_its_verifier(self):
        with self.assertRaisesRegex(ResolutionSchemaError,'execution_verifier_required'):
            resolved('EDIT_EXISTING',execute=True,verification_requirement='BASIC')
        self.assertEqual(resolved('EDIT_EXISTING',execute=True).verification_requirement.value,'EXECUTION')

    def test_confirmation_and_read_accept_correct_communicative_acts(self):
        self.assertEqual(resolved('CONFIRM',speech_act='ACKNOWLEDGEMENT').task_control,T.CONFIRM)
        self.assertEqual(resolved('READ_EXISTING',speech_act='INFORMATION_REQUEST').operation_intent.value,'READ_EXISTING')

    def test_precise_schema_rejection(self):
        for value in (semantic('EDIT_EXISTING',references={}),semantic('EDIT_EXISTING',execute='yes')):
            with self.assertRaises(ResolutionSchemaError) as error:
                semantic_resolution_payload(value,{})
            self.assertIn(error.exception.field,{'references','execute'})

    def test_memory_followup_inherits_typed_context_not_phrase(self):
        context={'resolution_context':{'discourse':{'memory_recall':{'active':True,'subject':'USER'}}}}
        d=resolved('CONTENT_RESPONSE',state=context,turn_relation='TOPIC_CONTINUATION')
        self.assertEqual(d.conversation_purpose,ConversationPurpose.PERSONAL_MEMORY)
        self.assertEqual(resolved('CONTENT_RESPONSE',turn_relation='TOPIC_CONTINUATION').conversation_purpose,
                         ConversationPurpose.GENERAL)

    def test_memory_topic_closes(self):
        state=DiscourseContext()
        state.resolve(resolved('PERSONAL_MEMORY_RECALL'))
        self.assertTrue(state.memory_recall.active)
        state.resolve(resolved('CONTENT_RESPONSE',turn_relation='NEW_TOPIC'))
        self.assertFalse(state.memory_recall.active)

    def test_specific_failure_translation_and_honest_fallback(self):
        f=FailureRecord.create(task_id='t',stage='INTERACTION_RESOLUTION',error_code='INTERACTION_CLARIFICATION',
            safe_user_message='',retryable=False,validation_detail={
                'failed_constraint':'operation_intent_conflict:EDIT_EXISTING', 'repair_attempted':True})
        text=FailureResponseRenderer.render(f,depth=2)
        self.assertIn('no coincidieron',text)
        self.assertIn('antes de ejecutar',text)
        self.assertNotIn('EDIT_EXISTING',text)
        unknown=replace(f,validation_detail={})
        self.assertIn('No tengo registrado el detalle exacto',FailureResponseRenderer.render(unknown,depth=1))


class LiteralCode(unittest.TestCase):
    SOURCES=('def first_10_items(x_2):\n    return x_2 != 12 and x_2 >= 3\n',
        '#include <iostream>\nint main() {\n    int n_10 = 12;\n    std::cout << (n_10 >= 3);\n}\n')

    def final(self, source):
        outcome=RuntimeOutcome(status=OutcomeStatus.SUCCESS,safe_evidence={
            'information_only':True,'read_kind':'CODE','read_source_text':source,
            'read_name':'source.py','read_language':'python'})
        return FinalResponseComposer.compose(outcome)

    def test_source_survives_publication_and_qt_html_roundtrip(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QTextDocument
        app=QApplication.instance() or QApplication([])
        for source in self.SOURCES:
            final=self.final(source)
            segment=next(s for s in final.content_segments if s.kind==ContentKind.CODE)
            self.assertEqual(segment.text,source)
            approve_literal_publication(final)
            document=QTextDocument()
            document.setHtml(code_html(literal_segments(final.visible_text)))
            rendered=document.toPlainText()
            self.assertIn(source,rendered)
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hashlib.sha256(segment.text.encode()).hexdigest())

    def test_approved_code_bypasses_actual_math_adapter(self):
        from anydes_tools.conversation.typed_content import install_literal_presentation
        install_literal_presentation()
        from anydes.continuity_v2 import math_runtime
        for source in self.SOURCES:
            final=self.final(source)
            approve_literal_publication(final)
            result=math_runtime.present_math(final.visible_text)
            self.assertEqual(result.chat_text,final.visible_text)
            self.assertEqual(result.speech_text,final.visible_text)
            self.assertEqual(result.math_regions,0)

    def test_code_channel_does_not_bypass_secret_or_protocol_guard(self):
        for text in ('api_key = gsk_abcdefghijklmnop\n','{"tool_id":"internal"}'):
            final=self.final(text)
            self.assertFalse(final.content_segments)
            self.assertNotIn(text,final.visible_text)


class MemoryGrounding(unittest.TestCase):
    @dataclass
    class Outcome:
        text:str

    facts=[{'evidence_id':'a','source':'PERMANENT_MEMORY','value':'Estudio botanica.'},
           {'evidence_id':'b','source':'USER_CONVERSATION_V2','value':'Prefiero trabajar por la manana.'}]

    def stream(self,payload):
        return lambda *a,**k:self.Outcome(json.dumps(payload))

    def test_good_fact_is_exactly_grounded(self):
        outcome,selected,state=select_grounded_memory(self.facts,{},self.stream({'facts':[
            {'candidate':0,'quote':'Estudio botanica.','fact_kind':'STUDIES'}]}),record=lambda *a,**k:None)
        self.assertEqual(state,MemoryResultState.GROUNDED)
        self.assertEqual(selected.evidence_id,'a')
        self.assertIn(self.facts[0]['value'],outcome.text)

    def test_followup_excludes_used_fact(self):
        outcome,selected,state=select_grounded_memory(self.facts,{},self.stream({'facts':[
            {'candidate':0,'quote':'Prefiero trabajar por la manana.','fact_kind':'PREFERENCE'}]}),
            record=lambda *a,**k:None,used_ids=['a'])
        self.assertEqual(selected.evidence_id,'b')
        self.assertEqual(state,MemoryResultState.GROUNDED)

    def test_model_cannot_substitute_another_fact(self):
        outcome,selected,state=select_grounded_memory(self.facts,{},self.stream({'facts':[
            {'candidate':0,'quote':'Soy astronauta.','fact_kind':'PERSONAL_FACT'}]}),record=lambda *a,**k:None)
        self.assertIsNone(selected)
        self.assertEqual(state,MemoryResultState.UNUSABLE)
        self.assertNotIn('astronauta',outcome.text)

    def test_empty_filtered_and_metadata_have_distinct_states(self):
        for meta,expected in [({},MemoryResultState.EMPTY),({'sensitive_filtered':1},MemoryResultState.FILTERED),
                ({'metadata_excluded':2},MemoryResultState.LOW_QUALITY),({'reason':'NO_ACCESS'},MemoryResultState.UNUSABLE)]:
            out,selected,state=select_grounded_memory([],meta,lambda *a,**k:self.fail('no LLM needed'),record=lambda *a,**k:None)
            self.assertEqual(state,expected)
            self.assertIsNone(selected)

    def test_semantic_rejection_is_low_quality_not_no_data(self):
        out,selected,state=select_grounded_memory(self.facts,{},self.stream({'facts':[]}),record=lambda *a,**k:None)
        self.assertEqual(state,MemoryResultState.LOW_QUALITY)
        self.assertIsNone(selected)

    def test_candidates_and_batches_bounded_without_repeating_records(self):
        facts=[{'evidence_id':str(i),'source':'USER_CONVERSATION_V2','value':str(i)+'x'*300} for i in range(500)]
        batches=list(candidate_batches(facts))
        ids=[f['evidence_id'] for batch in batches for _,f in batch]
        self.assertLessEqual(len(batches),4)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertIn('499',ids)
        self.assertIn('0',ids)


class RealBindingIntegration(unittest.TestCase):
    setUp=prior.Acceptance.setUp
    tearDown=prior.Acceptance.tearDown
    controller=prior.Acceptance.controller
    source=prior.Acceptance.source

    def test_resolution_rejection_records_specific_constraint_and_explains_without_tools(self):
        bad=replace(resolved('EDIT_EXISTING',artifact_type='CODE'),requested_operations=('EDIT',))
        c=self.controller(bad,bad,resolved('EXPLAIN_FAILURE'),resolved('CONTINUE_EXPLANATION'))
        self.source(c,'selected.py')
        c.handle('Revise the saved source',turn_id='bad-interpretation')
        failure=c.state.failure_topic_record()
        self.assertEqual(failure.stage,'INTERACTION_RESOLUTION')
        self.assertIn('operation_intent_conflict:EDIT_EXISTING',failure.validation_detail['errors'])
        self.assertEqual(c.last_runtime_outcome.failure_ref,failure.failure_id)
        for i,text in enumerate(('What prevented this?','Could you explain that more clearly?')):
            response=c.handle(text,turn_id='explain'+str(i))
            self.assertEqual(c.last_runtime_outcome.failure_ref,failure.failure_id)
            self.assertIn('no coincidieron',response.text)
            self.assertNotIn('EDIT_EXISTING',response.text)
            self.assertFalse(c._executor_called or c._reasoner_called)

    def test_edit_lineage_and_execute_exact_new_version_b2(self):
        # Same real edit/atomic commit path as the established acceptance fixture.
        prior.Acceptance.test_real_edit_preserves_lineage_and_version(self)
        c=self.controllers[-1]
        ref=c.state.recent_artifact_state.resolve_ref('current.py',require_unique=True)
        self.assertEqual(ref.version,2)
        payload={'goal':'execute current saved source','operations':[{'operation_id':'run',
            'operation_kind':'TEST_CODE','inputs':['selected_artifact'],'outputs':[],
            'specification':{'stdin':'','timeout':8,'language':'python'}}],
            'requested_final_outputs':[]}
        plan=ExecutablePlanLowerer(c.registry,self.workspace,
            external_artifacts={'selected_artifact':ref}).lower_requirements(SemanticProposal.from_dict(payload))
        c._turn_reference=ref
        action=c._action_from_general_plan(plan,'execute the revised program')
        self.assertEqual(action.resolved_artifact_bindings['selected_artifact'],ref)
        c._remember_task_plan(action)
        response=c._resolve_and_prepare(action,turn_id='run-prepared')
        self.assertIsNotNone(c.state.pending_tool_action)
        c._handle_pending('approved','run-confirm',control=T.CONFIRM)
        runs=[r for r in c.last_results if r.capability in {'developer.run_python','developer.test'}]
        self.assertTrue(runs)
        self.assertEqual(runs[-1].output.get('stdout','').strip(),'2')
        self.assertEqual(c.state.recent_artifact_state.ref_by_path(ref.path).version,2)
