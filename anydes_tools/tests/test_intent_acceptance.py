"""Typed decisions are unit inputs, real read workers are integration, NOT human E2E."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from anydes_tools.conversation.operation_intent import (
    ArtifactOperation as A, RequestedEffects, resolution_payload)
from anydes_tools.conversation.interaction_router import (
    InteractionResolution as R, InteractionKind as K, TaskControl as T,
    InteractionResolver, GoalConsistencyValidator, RequestedOutcome as O)
from anydes_tools.conversation.dataflow import ArtifactBindings, ArtifactState
from anydes_tools.conversation.publication import FinalResponseComposer
from anydes_tools.tests import test_interaction_execution_phase_c as phase_c

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'programa_principal'))


def wire(intent='NO_ARTIFACT_ACTION', **kw):
    return {'turn_relation':'NEW_TOPIC', 'operation_intent':intent,
        'conversation_purpose':'GENERAL', 'output_kind':'NONE', 'requested_operations':[],
        'primary_goal':'semantic goal', 'confidence':.9, **kw}


def decision(intent='NO_ARTIFACT_ACTION', **kw):
    return R.from_dict(resolution_payload(wire(intent, **kw)))


class IntentContract(unittest.TestCase):
    def test_create_is_not_edit_even_with_context(self):
        d = decision('CREATE_NEW', requested_operations=['EDIT_ARTIFACT'], output_kind='CODE_ARTIFACT')
        self.assertIn('operation_intent_conflict:CREATE_NEW', InteractionResolver.contradictions(d,
            {'failure_available':False,'pending_action':False,'retry_available':False}))

    def test_create_does_not_require_existing_target(self):
        d = decision('CREATE_NEW', requested_operations=['GENERATE_CODE'], references=['old.py'])
        self.assertTrue(InteractionResolver.contradictions(d,
            {'failure_available':False,'pending_action':False,'retry_available':False}))

    def test_read_only_effects(self):
        d = decision('READ_EXISTING', requested_operations=['READ_ARTIFACT'], references=['selected'])
        self.assertTrue(d.effect_intent.read)
        self.assertFalse(any(v for k,v in d.effect_intent.to_dict().items() if k != 'read'))

    def test_read_replacement_rejected(self):
        d = decision('READ_EXISTING', requested_operations=['EDIT_ARTIFACT'])
        self.assertIn('operation_intent_conflict:READ_EXISTING', InteractionResolver.contradictions(d,
            {'failure_available':False,'pending_action':False,'retry_available':False}))

    def test_read_generation_rejected(self):
        d = decision('READ_EXISTING', requested_operations=['READ_ARTIFACT'], needs_content_generation=True)
        self.assertIn('read_existing_must_be_non_mutating', InteractionResolver.contradictions(d,
            {'failure_available':False,'pending_action':False,'retry_available':False}))

    def test_control_projection_has_single_authority(self):
        for relation, control in [('FAILURE_CAUSE',T.EXPLAIN_FAILURE), ('FAILURE_DETAIL',T.CONTINUE_EXPLANATION),
                ('FAILURE_REMEDIATION',T.REMEDIATION_QUERY), ('PENDING_CANCEL',T.CANCEL)]:
            d = decision(turn_relation=relation)
            self.assertEqual(d.task_control,control)
            self.assertEqual(d.kind,K.CONTENT_RESPONSE)
            self.assertFalse(d.requested_effects)

    def test_explicit_conflicting_control_not_coerced(self):
        with self.assertRaisesRegex(ValueError,'relation_control_conflict'):
            decision(turn_relation='PENDING_CANCEL',task_control='CONFIRM')

    def test_control_cannot_hide_operation(self):
        with self.assertRaisesRegex(ValueError,'control_operation_conflict'):
            decision('REPLACE_EXISTING',turn_relation='FAILURE_CAUSE',requested_operations=['EDIT_ARTIFACT'])

    def test_schema_required_and_effects_strict(self):
        with self.assertRaises(ValueError):
            resolution_payload({})
        with self.assertRaises(ValueError):
            R.from_dict({'kind':'TOOL_ACTION','effect_intent':{'read':'true'}})

    def test_exact_semantic_error_repair_once(self):
        bad = decision('CREATE_NEW',requested_operations=['EDIT_ARTIFACT'])
        good = decision('CREATE_NEW',requested_operations=['GENERATE_CODE'])
        calls=[]
        values=iter([bad,good])
        r=InteractionResolver(lambda req:calls.append(req) or next(values))
        result=r.resolve('Another standalone program',state={
            'failure_available':False,'pending_action':False,'retry_available':False})
        self.assertEqual(result,good)
        self.assertEqual(len(calls),2)
        self.assertIn('operation_intent_conflict:CREATE_NEW',calls[1]['repair_errors'])


class ReadAndCancelIntegration(unittest.TestCase):
    setUp=phase_c.PhaseC.setUp
    tearDown=phase_c.PhaseC.tearDown
    controller=phase_c.PhaseC.controller

    def source(self,c):
        path=self.workspace/'example.py'
        path.write_text('print(17)\n')
        ref=ArtifactBindings('producer').produce('source',path,'write',state=ArtifactState.COMMITTED,artifact_type='CODE')
        c.state.recent_artifact_state.register(ref)
        return ref

    def test_read_exact_artifact_content_no_mutation_or_planner(self):
        c=self.controller(decision('READ_EXISTING',requested_operations=['READ_ARTIFACT'],
            references=['example.py'],output_kind='CODE_ARTIFACT'))
        ref=self.source(c)
        before=Path(ref.path).read_bytes()
        with patch('anydes_tools.conversation.controller.plan_from_text',side_effect=AssertionError('planner')):
            response=c.handle('Show the source you just saved',turn_id='read')
        self.assertEqual(response.text,'print(17)\n')
        self.assertEqual(c.last_runtime_outcome.status.value,'SUCCESS')
        self.assertFalse(c.last_runtime_outcome.committed_artifacts)
        self.assertFalse(c.state.pending_tool_action)
        self.assertEqual(Path(ref.path).read_bytes(),before)
        self.assertEqual(c.state.recent_artifact_state.resolve_ref('example.py'),ref)
        final=FinalResponseComposer.compose(c.last_runtime_outcome)
        self.assertIn('print(17)',final.visible_text)
        self.assertEqual(final.visible_text,final.spoken_text)

    def test_missing_read_target_clarifies_without_workspace_search(self):
        c=self.controller(decision('READ_EXISTING',requested_operations=['READ_ARTIFACT']))
        self.assertIn('archivo',c.handle('Let me see it',turn_id='missing').text)
        self.assertFalse(c._executor_called)

    def test_edit_without_artifact_requires_reference(self):
        c=self.controller(decision('EDIT_EXISTING',requested_operations=['EDIT_ARTIFACT'],
            output_kind='CODE_ARTIFACT',needs_content_generation=True))
        response=c.handle('Revise that program',turn_id='edit-missing')
        self.assertEqual(response.debug['response_source'],'ARTIFACT_REFERENCE_CLARIFICATION')
        self.assertFalse(c._executor_called)

    def test_create_does_not_consume_old_focus(self):
        c=self.controller(decision('CREATE_NEW',requested_operations=['GENERATE_CODE'],
            output_kind='CODE_ARTIFACT',needs_content_generation=True))
        ref=self.source(c)
        c.handle('Prepare another independent program',turn_id='new')
        self.assertIsNone(c._turn_reference)
        self.assertEqual(Path(ref.path).read_text(),'print(17)\n')

    def test_cancel_pending_confirmation_prevents_dispatch(self):
        from anydes_tools.tests.test_failure_record_phase_c import FailureRecordIntegration
        c=self.controller(decision(turn_relation='PENDING_CANCEL'))
        action=FailureRecordIntegration.prepared_action(self)
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id='prepare')
        self.assertIsNotNone(c.state.pending_tool_action)
        c.handle('Stop before running it',turn_id='cancel')
        self.assertIsNone(c.state.pending_tool_action)
        self.assertFalse(c._executor_called)

    def test_read_never_accepts_stale_bytes(self):
        c=self.controller(decision('READ_EXISTING',requested_operations=['READ_ARTIFACT'],references=['example.py']))
        ref=self.source(c)
        Path(ref.path).write_text('unexpected replacement')
        response=c.handle('Display the existing file',turn_id='stale')
        self.assertEqual(response.debug['response_source'],'ARTIFACT_REFERENCE_CLARIFICATION')
        self.assertFalse(c._executor_called)

    def test_disabled_read_does_not_dispatch(self):
        c=self.controller(decision('READ_EXISTING',requested_operations=['READ_ARTIFACT'],references=['example.py']))
        self.source(c)
        with patch.object(c.registry,'find_by_capability',return_value=[]):
            response=c.handle('Show source',turn_id='disabled')
        self.assertEqual(response.debug['response_source'],'READ_UNSUPPORTED')
        self.assertFalse(c._executor_called)

    def test_cancel_without_pending_is_natural(self):
        c=self.controller(decision(turn_relation='PENDING_CANCEL'))
        response=c.handle('Do not proceed',turn_id='cancel')
        self.assertIn('ninguna accion pendiente',response.text)
        self.assertFalse(c._executor_called)

    def test_cancel_clears_offer_and_retry_without_tools(self):
        c=self.controller(decision(turn_relation='PENDING_CANCEL'))
        c.discourse.pending_offer={'offer_id':'o','operation_kind':'CREATE_DOCUMENT','request':'create a note'}
        c.state.pending_task.status='FAILED'
        response=c.handle('Leave that action aside',turn_id='cancel')
        self.assertIn('no lo hago',response.text)
        self.assertIsNone(c.discourse.pending_offer)
        self.assertEqual(c.state.pending_task.status,'CANCELLED')
        self.assertFalse(c._executor_called)


class MemoryReadOnly(unittest.TestCase):
    def test_personal_fact_ranks_above_origin_metadata_and_followup_excludes_previous(self):
        from anydes.memoria.local_memory import JsonMemoryStore
        from anydes_tools.conversation.conversational_bridge import memory_evidence
        with tempfile.TemporaryDirectory(dir=ROOT/'anydes_tools/logs') as directory:
            root=Path(directory)
            store=JsonMemoryStore(root/'programa_principal/datos/memoria/permanente.json')
            for k,v in [('nacimiento_conversacion','2026-01-01'),('primera_palabra_usuario','hello'),
                        ('hobby','cultivo plantas'),('idioma','espanol'),('api_key','secret')]:
                store.set(k,v)
            before=store.path.read_bytes()
            assistant=NS(config=NS(data_dir=root/'candidate/data'))
            facts,meta=memory_evidence(assistant,root)
            self.assertEqual([f['key'] for f in facts],['hobby','idioma'])
            self.assertTrue(meta['read_only'])
            follow,_=memory_evidence(assistant,root,exclude_ids=[facts[0]['evidence_id']])
            self.assertEqual(len(follow),1)
            self.assertNotEqual(follow[0]['evidence_id'],facts[0]['evidence_id'])
            self.assertEqual(before,store.path.read_bytes())

    def test_encrypted_personal_memory_uses_configured_protector_read_only(self):
        from anydes.memoria.conversation_memory import ConversationMemory
        from anydes.memoria.data_protection import default_data_protector
        from anydes_tools.conversation.conversational_bridge import memory_evidence
        with tempfile.TemporaryDirectory(dir=ROOT/'anydes_tools/logs') as directory, patch.dict(
                'os.environ',{'ANYDES_MEMORY_PASSPHRASE':'fixture-passphrase-not-a-production-secret'}):
            root=Path(directory)
            base=root/'programa_principal/datos/memoria'
            m=ConversationMemory(base/'archivo_conversaciones.v2.jsonl',base/'categorias_conversaciones.v2.json',
                cognitive_path=base/'contexto_conversaciones.v2.jsonl',
                protector=default_data_protector(passphrase='fixture-passphrase-not-a-production-secret'))
            m.record_turn('Mi nombre es Test y estudio botanica.','Gracias por contarmelo.')
            before={p.name:p.read_bytes() for p in base.iterdir() if p.is_file()}
            facts,meta=memory_evidence(NS(config=NS(data_dir=root/'candidate/data')),root)
            self.assertEqual(meta['decryption_unavailable'],0)
            self.assertTrue(any(f['source']=='USER_CONVERSATION_V2' for f in facts))
            self.assertEqual(before,{p.name:p.read_bytes() for p in base.iterdir() if p.is_file()})

    def test_empty_and_no_access_are_distinguished(self):
        from anydes_tools.conversation.conversational_bridge import memory_evidence
        with tempfile.TemporaryDirectory(dir=ROOT/'anydes_tools/logs') as directory:
            root=Path(directory)
            assistant=NS(config=NS(data_dir=root/'data'))
            self.assertEqual(memory_evidence(assistant,root)[1]['reason'],'NO_DATA')
            assistant.speaker_allows_memory_context=False
            self.assertEqual(memory_evidence(assistant,root)[1]['reason'],'NO_ACCESS')
