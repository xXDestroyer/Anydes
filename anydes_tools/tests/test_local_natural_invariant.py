"""Exact legacy-style rejection, intact critical guards, no user phrase routing."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from anydes_tools.tests import test_conversational_core as core
from anydes_tools.conversation.natural_response import (
    validate_natural_prose, local_invariants, review_prose,
    LocalResponseInvariantError, classify_local_response_failure)
from anydes.realtime.context import RealtimeContext
from anydes.realtime.validation import RealtimeOutputValidator, ValidationResult
from anydes.realtime.service import _classify_realtime_failure


class LocalNaturalInvariantTests(unittest.TestCase):
    def setUp(self):
        self.validator = RealtimeOutputValidator()
        self.context = RealtimeContext(system_prompt='', turn_prompt='', user_text='A conversational opening',
            recent_turn_count=0, memory_fact_count=0, self_preference_count=0,
            recent_evidence_match=False, asks_self_preference=False, asks_user_preference=False,
            user_states_preference=False, user_correction=False, approximate_input_tokens=0)

    def test_exact_legacy_failure_and_property_based_natural_validation(self):
        text = 'Estoy aqui para ayudarte.'
        old = self.validator.validate(text, self.context)
        self.assertEqual(old.text, '')
        self.assertIn('generic_response_only', old.issues)
        self.assertIn('empty_output', old.issues)
        new = validate_natural_prose(self.validator.validate, text, self.context)
        self.assertTrue(new.valid, new.issues)
        self.assertEqual(new.text, text)
        self.assertTrue(local_invariants(text, new))
        # The original validator and its legacy non-conversational route remain intact.
        self.assertEqual(self.validator.validate(text, self.context), old)

    def test_legitimate_wording_varies_without_any_required_invitation(self):
        for text in ('Estoy aqui para ayudarte.', 'Cuentame, te escucho.',
                     'Puedes contarme lo que tienes en mente.', 'Que necesitas?'):
            with self.subTest(text=text):
                result = validate_natural_prose(self.validator.validate, text, self.context)
                self.assertTrue(local_invariants(text, result), result.issues)
                self.assertTrue(result.valid, result.issues)

    def test_actual_adapter_review_and_final_validation_preserve_invitation(self):
        text = 'Estoy aqui para ayudarte.'
        outcome, validation, events = core.DomainContracts().adapter('CONTENT_RESPONSE',
            {'text':text, 'offer':None})
        self.assertEqual(outcome.text, text)
        self.assertEqual(validation.text, text)
        self.assertTrue(validation.valid, validation.issues)
        self.assertTrue(any(e == 'conversation_prose_review' for e, _ in events))

    def test_metadata_protocol_and_real_empty_output_still_fail(self):
        for text in ('basis=private', '{"text":"not prose"}', '```json\n{}', ''):
            with self.subTest(text=text):
                result = validate_natural_prose(self.validator.validate, text, self.context)
                self.assertFalse(local_invariants(text, result))

    def test_style_scrubbing_cannot_hide_an_unsupported_tool_claim(self):
        for text in ('Ya lo ejecute, estoy aqui para ayudarte.',
                     'He enviado el mensaje. Cuentame mas.'):
            with self.subTest(text=text):
                result = validate_natural_prose(self.validator.validate, text, self.context)
                self.assertIn('unverified_action_claim', result.issues)
                self.assertFalse(local_invariants(text, result))

    def test_missing_factual_grounding_stays_blocking(self):
        from dataclasses import replace
        context = replace(self.context, asks_user_preference=True)
        result = validate_natural_prose(self.validator.validate,
            'Tu preferencia es el verde.', context)
        self.assertIn('user_preference_unverified', result.issues)
        self.assertFalse(local_invariants('Tu preferencia es el verde.', result))

    def test_local_rejection_is_validation_not_transport(self):
        result = classify_local_response_failure(_classify_realtime_failure,
            None, ValidationResult(False, '', ('empty_output',)),
            stream_error_type=LocalResponseInvariantError.__name__)
        self.assertEqual(result, ('natural_response_local_invariant_rejected', 'validation'))
        self.assertEqual(classify_local_response_failure(_classify_realtime_failure,
            None, ValidationResult(False, ''), stream_error_type='OSError'),
            ('transport_exception_OSError', 'transport'))

    def test_real_local_rejection_never_invokes_remote_reviewer(self):
        provider = Mock()
        with self.assertRaises(LocalResponseInvariantError):
            review_prose(provider, [], 'basis=not-visible', {}, record=lambda *a, **k:None)
        provider.assert_not_called()

    def test_local_error_reaches_failed_outcome_naturalization_not_conversation(self):
        from anydes_tools.conversation.runtime_patch import _compose_final
        from anydes_tools.conversation.publication import RuntimeOutcome, OutcomeStatus
        from anydes.realtime.service import RealtimeResult
        assistant = NS(audit=NS(record=lambda *a, **k:None),
            last_realtime_result=RealtimeResult('', NS(
                failure_code='natural_response_local_invariant_rejected')))
        observed = []
        def capture(a, c, outcome, final, text):
            observed.append(outcome)
            return final
        with patch('anydes_tools.conversation.publication.outcome_from_controller', return_value=RuntimeOutcome()), \
             patch('anydes_tools.conversation.outcome_voice.naturalize_final', side_effect=capture):
            _compose_final(assistant, 'old fallback', 'current turn', None)
        self.assertEqual(observed[0].status, OutcomeStatus.FAILED)
        self.assertEqual(observed[0].safe_evidence['failure_stage'], 'LOCAL_RESPONSE_VALIDATION')
        self.assertFalse(observed[0].committed_artifacts)


if __name__ == '__main__':
    unittest.main()
