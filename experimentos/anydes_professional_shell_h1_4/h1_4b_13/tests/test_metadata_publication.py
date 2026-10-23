"""Real read-only metadata selection and publication; prepared prose/reviews."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace as NS
import unittest

from anydes_tools.tests import test_manual_flow as prior
from anydes_tools.tests.test_outcome_voice import OutcomeVoiceTests
from anydes_tools.tests.test_natural_response import review_payload
from anydes_tools.conversation.outcome_voice import request_for,naturalize_final,generate_outcome_voice
from anydes_tools.conversation.publication import FinalResponseComposer,RuntimeOutcome
from anydes_tools.conversation.publication_boundary import require_visible,PublicationFailure
from anydes_tools.conversation.metadata_publication import MetadataPublicationError
from anydes_tools.conversation.reviewer_transport import ReviewerUnavailable

wire=prior.wire


class GroundedMetadata(unittest.TestCase):
    setUp=prior.MetadataTests.setUp
    tearDown=prior.MetadataTests.tearDown
    controller=prior.MetadataTests.controller
    source=prior.MetadataTests.source

    def ready(self,field='LOCATION'):
        c=self.controller(wire('ARTIFACT_METADATA_QUERY',metadata_field=field))
        ref=self.source(c,'letter_counter.py')
        c.handle('Where is the recent result?',turn_id='metadata')
        return c,ref

    def publish(self,c,text,review=None):
        a,calls,events=OutcomeVoiceTests().assistant([{'text':text},review or review_payload()])
        outcome=c.last_runtime_outcome
        final=naturalize_final(a,c,outcome,FinalResponseComposer.compose(outcome),'Where is it?')
        return require_visible(final),calls,events

    def test_location_exact_multiline_prose_approved_without_action_result(self):
        c,ref=self.ready()
        text='Esta es la ubicacion.\n`'+ref.path+'`.'
        final,calls,events=self.publish(c,text)
        self.assertEqual(final.visible_text,text)
        self.assertEqual(final.spoken_text,text)
        self.assertEqual(final.decision.artifact_ids,(ref.artifact_id,))
        self.assertEqual(final.decision.allowed_claims,('ARTIFACT_METADATA',))
        self.assertFalse(c.last_runtime_outcome.committed_artifacts)
        self.assertEqual(len(calls),2)
        self.assertTrue(any(e=='metadata_publication_bound' for e,_ in events))

    def test_direct_stored_location_is_metadata_not_new_action(self):
        c,ref=self.ready()
        text='Lo guarde en `'+ref.path+'`.'
        final,_,_=self.publish(c,text)
        self.assertEqual(final.visible_text,text)
        self.assertFalse(c._executor_called or c._reasoner_called)
        self.assertIsNone(c.state.pending_tool_action)

    def test_name_exact_not_sentence_split_at_extension(self):
        c,ref=self.ready('NAME')
        final,_,_=self.publish(c,'Se llama `'+ref.display_name+'`.')
        self.assertIn(ref.display_name,final.visible_text)

    def test_version_field_grounded(self):
        c,ref=self.ready('VERSION')
        final,_,_=self.publish(c,'Es la version '+str(ref.version)+'.')
        self.assertEqual(final.decision.artifact_ids,(ref.artifact_id,))

    def test_readonly_no_planning_execution_or_confirmation(self):
        c,ref=self.ready()
        self.publish(c,'Esta en `'+ref.path+'`.')
        self.assertFalse(c._reasoner_called or c._executor_called)
        self.assertIsNone(c.state.pending_tool_action)
        self.assertFalse(c.last_resolution.requires_execution)

    def test_changed_separator_rejected_exact_field_diagnostic(self):
        c,ref=self.ready()
        req=request_for(c.last_runtime_outcome,c,'Where is it?')
        altered=ref.path.replace('\\','/')
        if altered==ref.path:altered=ref.path.replace('/','\\')
        a,_,events=OutcomeVoiceTests().assistant([{'text':'Esta en `'+altered+'`.'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)
        errors=[d for e,d in events if e=='outcome_voice_rejection_diagnostic']
        self.assertEqual(errors[0]['reason_code'],'METADATA_VALUE_NOT_PRESERVED')
        self.assertEqual(errors[0]['field_path'],'response.text')
        self.assertNotIn(ref.path,str(errors))

    def test_extra_path_rejected_even_with_correct_value(self):
        c,ref=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'`'+ref.path+'` y `C:\\other\\wrong.py`.'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_path_prefix_is_not_exact_factual_value(self):
        c,ref=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'Esta en `'+ref.path+'.backup`.'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_version_prefix_is_not_exact_factual_value(self):
        c,_=self.ready('VERSION');req=request_for(c.last_runtime_outcome,c,'Which version?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'Es la version 10.'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_missing_path_natural_uncertainty_no_fabrication(self):
        c,ref=self.ready()
        item=c.state.recent_artifact_state.artifacts[0]
        item.runtime_ref=replace(ref,committed_path_ref='',temporary_path_ref='')
        c._metadata_evidence['value']=''
        text='No tengo registrada la ubicacion de ese archivo.'
        final,calls,_=self.publish(c,text)
        self.assertEqual(final.visible_text,text)
        self.assertIn('UNAVAILABLE',str(calls[0]))
        self.assertNotIn('ARTIFACT_METADATA',final.decision.allowed_claims)

    def test_missing_path_cannot_publish_invented_location(self):
        c,_=self.ready()
        c._metadata_evidence={'metadata_status':'NOT_FOUND','requested_field':'LOCATION'}
        req=request_for(c.last_runtime_outcome,c,'Where?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'Esta en C:\\invented\\file.py'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_mismatched_reference_rejected_before_wording(self):
        c,ref=self.ready()
        other=self.source(c,'another.py')
        c.last_resolution=wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION',references=[ref.artifact_id])
        c._metadata_evidence.update(evidence_id=other.artifact_id,value=other.path,version=other.version)
        with self.assertRaisesRegex(MetadataPublicationError,'BINDING_MISMATCH'):
            request_for(c.last_runtime_outcome,c,'Where?')

    def test_metadata_value_cannot_differ_from_same_ref(self):
        c,_=self.ready();c._metadata_evidence['value']=r'C:\wrong\file.py'
        with self.assertRaisesRegex(MetadataPublicationError,'BINDING_MISMATCH'):
            request_for(c.last_runtime_outcome,c,'Where?')

    def test_recent_artifact_binding_uses_current_focus(self):
        c,first=self.ready();latest=self.source(c,'latest.py')
        c.interaction_resolver.classifier=lambda request:wire('ARTIFACT_METADATA_QUERY',metadata_field='LOCATION')
        c.handle('Where is that result?',turn_id='latest')
        final,_,_=self.publish(c,'Esta en `'+latest.path+'`.')
        self.assertEqual(final.decision.artifact_ids,(latest.artifact_id,))
        self.assertNotIn(first.path,final.visible_text)

    def test_unsupported_execution_claim_rejected_locally(self):
        c,ref=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        a,calls,_=OutcomeVoiceTests().assistant([{'text':'Ya lo ejecute. Esta en `'+ref.path+'`.'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)
        self.assertEqual(len(calls),2,'no remote approval can override this local action blocker')

    def test_semantic_unsupported_claim_rejected_by_review(self):
        c,ref=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        text='Tiene resultados excelentes. Esta en `'+ref.path+'`.'
        a,_,_=OutcomeVoiceTests().assistant([{'text':text},review_payload(['UNSUPPORTED_FACT']),
            {'text':text},review_payload(['UNSUPPORTED_FACT'])])
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_no_empty_metadata_publication(self):
        c,_=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'   '}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)

    def test_chat_voice_same_approved_hash(self):
        c,ref=self.ready();final,_,_=self.publish(c,'El archivo esta en `'+ref.path+'`.')
        digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
        self.assertEqual(digest(final.visible_text),digest(final.spoken_text))

    def test_protocol_leak_still_rejected(self):
        c,ref=self.ready();req=request_for(c.last_runtime_outcome,c,'Where?')
        a,_,_=OutcomeVoiceTests().assistant([{'text':'basis=internal; `'+ref.path+'`'}]*2)
        with self.assertRaises(ReviewerUnavailable):generate_outcome_voice(a,req)
