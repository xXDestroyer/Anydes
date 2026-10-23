"""Prepared semantics/inputs; real bindings, executor and B2. Not human UI E2E."""
from dataclasses import replace
import unittest
from anydes_tools.tests import test_interaction_execution_phase_c as phase_c
from anydes_tools.tests import test_failure_record_phase_c as failures
from anydes_tools.conversation.dataflow import ArtifactBindings, ArtifactState
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer
from anydes_tools.conversation.interaction_router import InteractionResolution as R, InteractionKind as K, TaskControl as T, TurnRelation
from anydes_tools.conversation.failure_state import FailureRecord
from anydes_tools.conversation.publication import FinalResponseComposer
from anydes_tools.conversation.presentation import normalize_chat_preserving_paths
from anydes_tools.conversation.source_names import canonical_source_name
from anydes_tools.conversation.planner import plan_from_text
import zipfile


class Acceptance(unittest.TestCase):
    setUp = phase_c.PhaseC.setUp
    tearDown = phase_c.PhaseC.tearDown
    controller = phase_c.PhaseC.controller

    def source(self, c, name):
        path = self.workspace/name
        path.write_text('print(1)\n')
        ref = ArtifactBindings('producer').produce('code', path, 'write', state=ArtifactState.COMMITTED, artifact_type='CODE')
        c.state.recent_artifact_state.register(ref)
        return ref

    def test_focus_newest_typed_artifact_and_ambiguity(self):
        c = self.controller()
        a = self.source(c,'a.py')
        b = self.source(c,'b.py')
        index = c.state.recent_artifact_state
        self.assertEqual(index.resolve_ref('modifica el programa', artifact_type='CODE', require_unique=True), b)
        self.assertEqual(index.last_resolution['status'],'RESOLVED')
        self.assertEqual(index.resolve_ref('a.py', require_unique=True), a)
        index.focus_artifact_ids = (a.artifact_id,b.artifact_id)
        self.assertIsNone(index.resolve_ref('modifica el programa', artifact_type='CODE', require_unique=True))
        self.assertEqual(index.last_resolution['status'],'AMBIGUOUS')
        index.remove(b.path)
        self.assertIsNone(index.resolve_ref('b.py', require_unique=True))
        self.assertEqual(index.last_resolution['status'],'NOT_FOUND')

    def inconclusive_draft(self,c):
        action = failures.FailureRecordIntegration.prepared_action(self)
        action = replace(action,arguments={**action.arguments,'oracle_authority':'MODEL_GENERATED',
            'code_generation_spec':{**action.arguments['code_generation_spec'],
                'tests':[{'test_id':'inspect','stdin':'2\n4\n','expected_hint':'6'}]}})
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id='prepare')
        c._handle_pending('approved','run',control=T.CONFIRM)
        self.assertIsNotNone(c._retained_draft)
        return c._retained_draft

    def test_retention_destination_change_blocks_and_close_discards_private_source(self):
        c = self.controller()
        draft = self.inconclusive_draft(c)
        from pathlib import Path
        from anydes_tools.conversation.dataflow import DataflowError
        Path(draft['destination']).write_text('new external version')
        with self.assertRaisesRegex(DataflowError,'STALE_ARTIFACT'):
            c._retain_unverified('keep','save for review')
        c.close()
        self.assertFalse(Path(draft['source']).exists())
        self.assertEqual(Path(draft['destination']).read_text(),'new external version')

    def test_retention_disabled_capability_cannot_be_overridden(self):
        c = self.controller()
        draft = self.inconclusive_draft(c)
        c.registry.disable_capability('developer.write_code')
        c._retain_unverified('keep','save for inspection')
        from pathlib import Path
        self.assertFalse(Path(draft['destination']).exists())
        self.assertTrue(Path(draft['source']).exists())

    def test_failed_reference_can_only_be_retained_as_failed_draft(self):
        c = self.controller(R(K.TOOL_ACTION,task_control=T.RETAIN_UNVERIFIED,confidence=1))
        from anydes_tools.tests.test_semantics_propagation_phase_e4 import UNITS
        action = failures.FailureRecordIntegration.prepared_action(self)
        action = replace(action,arguments={**action.arguments,'oracle_authority':'MODEL_GENERATED',
            'code_generation_spec':{**action.arguments['code_generation_spec'],
                'operation_descriptor':UNITS,
                'tests':[{'test_id':'reference','stdin':'0','expected_hint':'999'}]},
            'generated_code_artifact':{'language':'python','filename':'sum.py','code':'print(999)\n','tests':[]}})
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id='prepare')
        c._handle_pending('approved','run',control=T.CONFIRM)
        self.assertEqual(c._retained_draft['verification']['status'],'FAIL')
        c.handle('keep source only for inspection',turn_id='keep')
        ref = c.state.recent_artifact_state.ref_by_path(str(self.workspace/'sum.py'))
        self.assertEqual(ref.state,ArtifactState.COMMITTED_UNVERIFIED)
        self.assertEqual(ref.metadata['verification_result']['status'],'FAIL')
        final = FinalResponseComposer.compose(c.last_runtime_outcome)
        self.assertNotIn('VERIFICATION_PASSED',final.decision.allowed_claims)

    def test_semantic_edit_consumes_exact_host_reference(self):
        c = self.controller()
        ref = self.source(c,'actual.py')
        payload = {'goal':'modify the current program', 'operations':[{
            'operation_id':'edit', 'operation_kind':'EDIT_ARTIFACT', 'inputs':['selected_artifact'],
            'outputs':[{'name':'edited','artifact_type':'CODE','format':'py','filename':'wrong_modified.py'}],
            'specification':{'requested_change':'print two instead', 'new_requirements':['print two'],
                'execution_required':False}}], 'requested_final_outputs':['edited']}
        plan = ExecutablePlanLowerer(c.registry,self.workspace,external_artifacts={'selected_artifact':ref}).lower_requirements(SemanticProposal.from_dict(payload))
        c._turn_reference = ref
        action = c._action_from_general_plan(plan, 'modify current program')
        self.assertEqual(action.target_path,ref.path)
        executable = c.plan_adapter.compile(action,source='GENERAL_REASONER')
        self.assertEqual(executable.typed_inputs['artifact_bindings']['selected_artifact'].artifact_id,ref.artifact_id)

    def test_real_inconclusive_retention_preserves_evidence(self):
        c = self.controller(R(K.TOOL_ACTION,task_control=T.RETAIN_UNVERIFIED,confidence=1))
        action = failures.FailureRecordIntegration.prepared_action(self)
        action = replace(action,arguments={**action.arguments,'oracle_authority':'MODEL_GENERATED',
            'code_generation_spec':{**action.arguments['code_generation_spec'],
                'tests':[{'test_id':'inspect','stdin':'2\n4\n','expected_hint':'6'}]}})
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id='prepare')
        c._handle_pending('approved','run',control=T.CONFIRM)
        self.assertIsNotNone(c._retained_draft, [(r.capability,r.status.value,r.error) for r in c.last_results])
        evidence = c._retained_draft['verification']
        self.assertEqual(evidence['status'],'INCONCLUSIVE')
        self.assertFalse((self.workspace/'sum.py').exists())
        answer = c.handle('keep the draft for inspection',turn_id='keep')
        self.assertTrue((self.workspace/'sum.py').is_file(),answer)
        ref = c.state.recent_artifact_state.ref_by_path(str(self.workspace/'sum.py'))
        self.assertEqual(ref.state,ArtifactState.COMMITTED_UNVERIFIED)
        self.assertEqual(ref.metadata['verification_result'],evidence)
        final = FinalResponseComposer.compose(c.last_runtime_outcome)
        self.assertIn('sin verificar',final.visible_text)
        self.assertNotIn('VERIFICATION_PASSED',final.decision.allowed_claims)
        archive = plan_from_text('Comprime sum.py en un ZIP llamado review.zip.',c.state,
            requested_operations=('CREATE_ARCHIVE',))
        plan = c.plan_adapter.compile(archive,source='FAST_PATH')
        c._execute(plan,turn_id='archive-unverified',confirmed_effects=())
        with zipfile.ZipFile(archive.arguments['destination']) as actual:
            self.assertEqual(actual.namelist(),['sum.py'])
            self.assertEqual(actual.read('sum.py'),(self.workspace/'sum.py').read_bytes())

    def test_failure_topic_explanation_depth_without_execution(self):
        c = self.controller(R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,confidence=1),
            R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,confidence=1),
            R(K.CONVERSATION,confidence=1,turn_relation=TurnRelation.NEW_TOPIC))
        record = FailureRecord.create(task_id='failed',stage='PLAN_VALIDATION',error_code='PLAN_LOWERING_FAILED:final_output',
            safe_user_message='No pude preparar una secuencia valida.',retryable=False,
            validation_detail={'failed_constraint':'final_output'})
        c.state.pending_task.task_id = 'failed'
        c.state.last_tool_failure = record
        for text in ('why did that happen','what does that mean'):
            c.handle(text,turn_id=text)
            self.assertFalse(c._executor_called)
            self.assertEqual(c.last_runtime_outcome.failure_ref,record.failure_id)
        rendered = FinalResponseComposer.compose(c.last_runtime_outcome).visible_text
        self.assertIn('no significa',rendered)
        c.handle('tell me about music',turn_id='new-topic')
        self.assertFalse(c.state.failure_topic_active)

    def test_names_and_math_rendering(self):
        self.assertEqual(canonical_source_name('Calculadora Promedio','py'),'calculadora_promedio.py')
        self.assertEqual(canonical_source_name('first 10 primes','py'),'first_10_primes.py')
        self.assertEqual(canonical_source_name('first\u2081\u2080\u209aprimes.py','py'),'first10pprimes.py')
        self.assertEqual(canonical_source_name('User_Name.py','py'),'User_Name.py')
        with self.assertRaises(ValueError): canonical_source_name('../escape.py','py')
        self.assertEqual(normalize_chat_preserving_paths('word_stats.py',lambda x:x.replace('_','BAD')),'word_stats.py')

    def test_real_edit_preserves_lineage_and_version(self):
        c = self.controller(code_modifier=lambda spec:{'language':'python','filename':spec.target_artifact,
            'code':'print(2)\n','tests':[]})
        old = self.source(c,'older.py')
        ref = self.source(c,'current.py')
        payload = {'goal':'edit current code','operations':[{'operation_id':'edit','operation_kind':'EDIT_ARTIFACT',
            'inputs':['selected_artifact'],'outputs':[{'name':'edited','artifact_type':'CODE','format':'py'}],
            'specification':{'requested_change':'print two instead','new_requirements':['print two'],
                'execution_required':False}}],'requested_final_outputs':['edited']}
        requirements = ExecutablePlanLowerer(c.registry,self.workspace,
            external_artifacts={'selected_artifact':ref}).lower_requirements(SemanticProposal.from_dict(payload))
        c._turn_reference = ref
        action = c._action_from_general_plan(requirements,'change the current code')
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id='edit')
        c._handle_pending('approved','edit-confirm',control=T.CONFIRM)
        updated = c.state.recent_artifact_state.ref_by_path(ref.path)
        self.assertEqual(updated.artifact_id,ref.artifact_id)
        self.assertEqual(updated.version,2)
        self.assertNotEqual(updated.hash,ref.hash)
        self.assertEqual((self.workspace/'older.py').read_text(),'print(1)\n')
        self.assertEqual((self.workspace/'current.py').read_text(),'print(2)\n')

    def test_retention_without_draft_and_low_confidence_control_closed(self):
        c = self.controller(R(K.TOOL_ACTION,task_control=T.RETAIN_UNVERIFIED,confidence=1),
            R(K.TOOL_ACTION,task_control=T.RETAIN_UNVERIFIED,confidence=1))
        result = c.handle('keep it',turn_id='none')
        self.assertFalse(c._executor_called)
        self.assertFalse(result.executed)

    def test_source_collision_not_silently_overwritten(self):
        c = self.controller()
        ref = self.source(c,'sum.py')
        action = failures.FailureRecordIntegration.prepared_action(self)
        c._remember_task_plan(action)
        result = c._resolve_and_prepare(action,turn_id='collision')
        self.assertFalse(result.executed)
        self.assertIsNotNone(c.state.pending_tool_action)
        self.assertEqual((self.workspace/'sum.py').read_text(),'print(1)\n')

    def test_unknown_validation_cause_and_secrets_not_invented(self):
        record = FailureRecord.create(task_id='t',stage='PLAN_VALIDATION',error_code='UNKNOWN',
            safe_user_message='',retryable=False,validation_detail={
                'errors':['api_key=gsk_private123456'],'failed_constraint':'unknown'})
        from anydes_tools.conversation.failure_state import FailureResponseRenderer
        self.assertNotIn('gsk_private',str(record.validation_detail))
        self.assertIn('No tengo mas detalle',FailureResponseRenderer.render(record,depth=1))

    def test_general_code_contract_holdout(self):
        c = self.controller()
        for domain in ('numbers','strings','lists','statistics','parsing','small algorithms','allowed local files'):
            with self.subTest(domain=domain):
                payload={'goal':'create standalone code for '+domain,'operations':[{
                    'operation_id':'generate','operation_kind':'GENERATE_CODE','inputs':[],
                    'outputs':[{'name':'main','artifact_type':'CODE','format':'py'}],
                    'specification':{'language':'python','goal':domain,'functional_requirements':[domain],
                        'execution_required':False}}],'requested_final_outputs':['main']}
                plan = ExecutablePlanLowerer(c.registry,self.workspace).lower_requirements(SemanticProposal.from_dict(payload))
                self.assertEqual(plan.required_capabilities,('developer.write_code',))
                self.assertEqual(plan.desired_outputs[0]['binding'],'main')

    def test_explanation_contradiction_uses_one_semantic_repair(self):
        c = self.controller(R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,confidence=1),
            R(K.CLARIFICATION,requested_operations=('EXPLAIN',),confidence=1),
            R(K.CONTENT_RESPONSE,task_control=T.EXPLAIN_FAILURE,confidence=1))
        record = FailureRecord.create(task_id='failed',stage='PLAN_VALIDATION',error_code='BAD_PLAN',
            safe_user_message='No pude preparar la accion.',retryable=False)
        c.state.pending_task.task_id = record.task_id
        c.state.pending_task.status = 'FAILED'
        c.state.last_tool_failure = record
        c.handle('explain the failure',turn_id='first')
        c.handle('explain that meaning further',turn_id='second')
        self.assertEqual(c.state.failure_explanation_depth,2)
        self.assertFalse(c._executor_called)
        calls = c.interaction_resolver.classifier.calls
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[-1]['repair_errors'],('explanation_relation_required','explanation_topic_available_not_unresolved'))
        self.assertEqual(calls[-1]['state']['active_failure']['failure_id'],record.failure_id)

    def test_document_operation_cannot_materialize_source_code(self):
        from anydes_tools.conversation.interaction_router import RequestedOutcome, InteractionResolver
        invalid = R(K.TOOL_ACTION,output_kind=RequestedOutcome.CODE_ARTIFACT,
            artifact_intent=True,requested_operations=('CREATE_DOCUMENT',),confidence=1)
        state = {'retry_available':False,'pending_action':False,'failure_available':False}
        self.assertIn('operation_output_type_incompatible',InteractionResolver.contradictions(invalid,state))
        valid = replace(invalid,output_kind=RequestedOutcome.DOCUMENT_ARTIFACT)
        self.assertNotIn('operation_output_type_incompatible',InteractionResolver.contradictions(valid,state))

    def test_retention_context_identifies_exact_private_source(self):
        c = self.controller(R(K.TOOL_ACTION,task_control=T.RETAIN_UNVERIFIED,confidence=1))
        draft = self.inconclusive_draft(c)
        c.handle('retain the source for my inspection',turn_id='keep')
        state = c.interaction_resolver.classifier.calls[0]['state']
        self.assertEqual(state['unpublished_draft']['task_id'],draft['task_id'])
        self.assertEqual(state['unpublished_draft']['display_name'],'sum.py')
        self.assertEqual(state['unpublished_draft']['verification_status'],'INCONCLUSIVE')
        self.assertNotIn('source',state['unpublished_draft'])

    def test_adherence_reused_only_for_identical_contract_and_content(self):
        from anydes_tools.conversation.content_generation import ContentGenerationSpec
        from types import SimpleNamespace
        calls = []
        c = self.controller(task_adherence_evaluator=lambda evidence: calls.append(evidence) or {
            'status':'PASS','goal_aligned':True,'subject_aligned':True,'reason_code':'ALIGNED'})
        c._executing_plan = SimpleNamespace(task_id='one')
        spec = ContentGenerationSpec.from_dict({'topic':'a project'})
        first = c._content_adherence('describe a project',spec,'some project text')
        second = c._content_adherence('describe a project',spec,'some project text')
        self.assertIs(first,second)
        self.assertEqual(len(calls),1)
        c._content_adherence('describe a project',spec,'different project text')
        self.assertEqual(len(calls),2)
        c._executing_plan = SimpleNamespace(task_id='two')
        c._content_adherence('describe a project',spec,'some project text')
        self.assertEqual(len(calls),3)

    def test_real_docx_commit_rechecks_structure_without_second_semantic_judgment(self):
        calls = []
        def evaluator(evidence):
            calls.append(evidence)
            return {'status':'PASS','goal_aligned':True,'subject_aligned':True,'reason_code':'ALIGNED'} if len(calls)==1 else {
                'status':'INCONCLUSIVE','goal_aligned':False,'subject_aligned':True,'reason_code':'INSUFFICIENT_EVIDENCE'}
        c = self.controller(semantic_prose_generator=lambda spec:'A small project helps people learn through careful practice and reflection.',
            task_adherence_evaluator=evaluator)
        payload = {'goal':'describe a small project in Word','operations':[
            {'operation_id':'content','operation_kind':'GENERATE_CONTENT','inputs':[],
                'outputs':[{'name':'body','artifact_type':'CONTENT','format':'text'}],
                'specification':{'topic':'a small project','target_length_words':12,'source_mode':'GENERATIVE'}},
            {'operation_id':'word','operation_kind':'CREATE_DOCUMENT','inputs':['body'],
                'outputs':[{'name':'document','artifact_type':'DOCUMENT','format':'docx','filename':'project.docx'}],
                'specification':{}}],'requested_final_outputs':['document']}
        requirements = ExecutablePlanLowerer(c.registry,self.workspace).lower_requirements(SemanticProposal.from_dict(payload))
        action = c._action_from_general_plan(requirements,'create a short Word about a small project')
        c._remember_task_plan(action)
        result = c._resolve_and_prepare(action,turn_id='word')
        self.assertTrue((self.workspace/'project.docx').exists(),result)
        self.assertEqual(c.state.last_verification['status'],'VERIFIED')
        self.assertEqual(len(calls),1)
        records = [e.verification_result for e in c.state.verification_evidence.values() if e.verification_result.get('verifier_type')=='DOCUMENT']
        self.assertGreaterEqual(len(records),2)
        self.assertTrue(all(r['status']=='PASS' for r in records))

