"""Prepared cognition, real registry/executor; separate from human UI validation."""
from dataclasses import replace
import unittest
import zipfile

from anydes_tools.contracts import ExecutionStatus, FileEffect, PermissionEffect, ToolExecutionResult
from anydes_tools.conversation.planner import plan_from_text
from anydes_tools.conversation.contracts import PlannedToolAction
from anydes_tools.conversation.interaction_router import InteractionKind as K, InteractionResolution as R, TaskControl as T
from anydes_tools.conversation.dataflow import ArtifactBindings, ArtifactState
from anydes_tools.conversation.task_adherence import evaluate_adherence
from anydes_tools.conversation.content_generation import ContentGenerationSpec
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration


class CandidateBlockersG3(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller
    prepared_action = DataflowIntegration.prepared_action

    def source(self, c, suffix):
        path = self.workspace / ('selected' + suffix)
        path.write_bytes(b'known artifact fixture')
        ledger = ArtifactBindings('producer')
        ref = ledger.produce('source', path, 'create', state=ArtifactState.COMMITTED)
        c._operational_artifacts[str(path.resolve())] = ref
        c.state.recent_artifact_state.remember(str(path))
        (self.workspace / 'unrelated.txt').write_text('must never archive')
        return ref

    def archive(self, suffix, request):
        c = self.controller()
        ref = self.source(c, suffix)
        action = plan_from_text(request, c.state, requested_operations=('CREATE_ARCHIVE',))
        plan = c.plan_adapter.compile(action, source='FAST_PATH')
        self.assertEqual(plan.steps[0].consumes_artifacts, ('input.0',))
        self.assertEqual(plan.typed_inputs['artifact_bindings']['input.0'].artifact_id, ref.artifact_id)
        results = c._execute(plan, turn_id='g3', confirmed_effects=())
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        with zipfile.ZipFile(action.arguments['destination']) as archive:
            self.assertEqual(archive.namelist(), [ref.display_name])
            self.assertEqual(archive.read(ref.display_name), b'known artifact fixture')

    def test_contextual_pdf_exact_binding(self):
        self.archive('.pdf', 'Comprimelo en un ZIP llamado selected.zip.')

    def test_requested_output_extension_not_duplicated(self):
        c = self.controller()
        self.source(c, '.pdf')
        action = plan_from_text('Comprime ese PDF en un ZIP llamado exact.zip.', c.state,
            requested_operations=('CREATE_ARCHIVE',))
        self.assertEqual(__import__('pathlib').Path(action.arguments['destination']).name.casefold(), 'exact.zip')

    def test_contextual_document_exact_binding(self):
        self.archive('.docx', 'Comprime ese archivo.')

    def test_ambiguous_artifacts_no_execution(self):
        c = self.controller(R(K.TOOL_ACTION, references=('previous artifact',), confidence=1,
            artifact_intent=True, requested_operations=('CREATE_ARCHIVE',)))
        self.source(c, '.pdf')
        other = self.workspace / 'other.pdf'
        other.write_bytes(b'other')
        c.state.recent_artifact_state.remember(str(other))
        response = c.handle('Comprime ese archivo.', turn_id='ambiguous')
        self.assertFalse(response.executed)
        self.assertFalse(c._executor_called)
        self.assertEqual(response.debug['response_source'], 'ARTIFACT_REFERENCE_CLARIFICATION')

    def test_missing_reference_never_workspace(self):
        c = self.controller()
        action = plan_from_text('Comprime ese archivo.', c.state, requested_operations=('CREATE_ARCHIVE',))
        self.assertEqual(action.intent, 'needs_artifact_selection')

    def test_directory_requires_explicit_selection(self):
        c = self.controller()
        action = PlannedToolAction('compress_zip', ('archive.compress_zip',), arguments={
            'sources': [str(self.workspace)], 'destination': str(self.workspace / 'bad.zip')})
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_DIRECTORY_REQUIRED'):
            c.plan_adapter.compile(action, source='LEGACY_ADAPTER')

    def test_explicit_authorized_folder(self):
        c = self.controller()
        folder = self.workspace / 'folder'
        folder.mkdir()
        (folder / 'a.txt').write_text('exact')
        request = f'Comprime toda la carpeta "{folder}".'
        action = plan_from_text(request, c.state, requested_operations=('CREATE_ARCHIVE',))
        plan = c.plan_adapter.compile(action, source='FAST_PATH')
        self.assertEqual(plan.typed_inputs['explicit_directory_sources'], (str(folder.resolve()),))
        results = c._execute(plan, turn_id='folder', confirmed_effects=())
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        with zipfile.ZipFile(action.arguments['destination']) as archive:
            self.assertEqual(archive.namelist(), ['folder/a.txt'])

    def test_cleanup_is_maintenance_not_commit_and_retry_allowed(self):
        c = self.controller()
        action = self.prepared_action()
        c._remember_task_plan(action)
        temporary = self.workspace / 'temporary.txt'
        temporary.write_text('temporary')
        results = []
        c._discard_staged(temporary, 'cleanup', (), results)
        self.assertFalse(temporary.exists())
        self.assertTrue(results[0].side_effects)
        effect = results[0].side_effects[0]
        self.assertEqual(effect.scope, 'RUNTIME_MAINTENANCE')
        self.assertFalse(effect.task_semantic)
        c.last_results = tuple(results)
        c._record_failure(action, turn_id='failure', stage='VERIFY', error_code='TEST_FAILURE',
            safe_user_message='fixture failed', retryable=True)
        self.assertFalse(c.state.pending_task.committed_effects)
        task = c.state.pending_task.task_id
        c._handle_retry('retry', 'retry', force=True)
        self.assertEqual(c.state.pending_task.task_id, task)
        self.assertEqual(c.state.pending_task.attempt_number, 2)
        self.assertEqual(c.state.attempt_history[0].status, 'FAILED')

    def test_user_delete_and_atomic_commit_remain_conservative(self):
        for cap, effect in [('filesystem.delete', PermissionEffect.DELETE),
                            ('filesystem.atomic_replace', PermissionEffect.CREATE)]:
            c = self.controller()
            action = self.prepared_action()
            c._remember_task_plan(action)
            c.last_results = (ToolExecutionResult('r', 't', 'official.filesystem', cap,
                ExecutionStatus.SUCCESS, side_effects=(FileEffect(effect, str(self.workspace / 'user.txt')),)),)
            c._record_failure(action, turn_id='failure', stage='VERIFY', error_code='TEST_FAILURE',
                safe_user_message='fixture failed', retryable=True)
            self.assertTrue(c.state.pending_task.committed_effects)
            c._handle_retry('retry', 'retry', force=True)
            self.assertEqual(c.state.pending_task.attempt_number, 1)

    def test_adherence_diagnostic_bounded_and_cause_retained(self):
        spec = ContentGenerationSpec.from_dict({'topic':'reading', 'purpose':'document'})
        evidence = evaluate_adherence(lambda e: {'status':'FAIL', 'goal_aligned':False,
            'subject_aligned':True, 'reason_code':'GOAL_MISMATCH',
            'safe_reason':'Expected two paragraphs, observed one.', 'evidence_basis':'paragraph structure'},
            goal='Write two paragraphs', specification=spec, content='One paragraph. ' * 100)
        self.assertEqual(evidence.safe_reason, 'Expected two paragraphs, observed one.')
        self.assertEqual(evidence.diagnostic_evidence['paragraph_count'], 1)
        self.assertLessEqual(len(evidence.diagnostic_evidence['content_snippet']), 700)

    def test_committed_artifact_even_with_wrapped_result_blocks_retry(self):
        c = self.controller()
        action = self.prepared_action()
        c._remember_task_plan(action)
        path = self.workspace / 'published.txt'
        path.write_text('published')
        c._artifact_bindings = ArtifactBindings(c.state.pending_task.task_id)
        c._artifact_bindings.produce('published', path, 'publish', state=ArtifactState.COMMITTED)
        c._record_failure(action, turn_id='partial', stage='VERIFY', error_code='TEST_FAILURE',
            safe_user_message='later step failed', retryable=True)
        self.assertTrue(c.state.pending_task.committed_effects)

    def test_previous_task_ledger_is_not_current_commit(self):
        c = self.controller()
        action = self.prepared_action()
        c._remember_task_plan(action)
        path = self.workspace / 'previous.txt'
        path.write_text('previous')
        c._artifact_bindings = ArtifactBindings('previous-task')
        c._artifact_bindings.produce('previous', path, 'publish', state=ArtifactState.COMMITTED)
        c._record_failure(action, turn_id='new-failure', stage='VERIFY', error_code='TEST_FAILURE',
            safe_user_message='current task failed', retryable=True)
        self.assertFalse(c.state.pending_task.committed_effects)

    def test_adherence_scope_excludes_host_effects(self):
        captured = []
        def evaluator(evidence):
            captured.append(evidence)
            return {'status':'PASS', 'goal_aligned':True, 'subject_aligned':True, 'reason_code':'ALIGNED'}
        spec = ContentGenerationSpec.from_dict({'topic':'reading', 'purpose':'document'})
        evaluate_adherence(evaluator, goal='Write about reading and save as a document',
            specification=spec, content='Reading helps learning.')
        self.assertEqual(captured[0]['evaluation_scope'], 'CONTENT_SEMANTICS_ONLY')
        self.assertIn('COMMIT_PUBLICATION', captured[0]['host_verified_responsibilities'])


del DataflowIntegration
