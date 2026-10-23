"""Fault injection observes exceptions, never rewrites or recovers them."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from anydes_tools.conversation.diagnostics import ConversationDiagnostics
from anydes_tools.tests import test_conversational_core as core


ROOT = Path(__file__).resolve().parents[2]


class ConversationDiagnosticTests(unittest.TestCase):
    def diagnostic(self):
        events = []
        return ConversationDiagnostics(lambda e, **d: events.append((e, d)), ROOT, 'turn-17'), events

    def test_every_boundary_preserves_exception_and_records_private_trace(self):
        stages = ('SEMANTIC_CONTEXT_BUILD', 'SEMANTIC_ADAPTER', 'REALTIME_SERVICE',
            'GENERATION_STREAM_ASSEMBLY', 'GENERATION_ROUTER', 'GENERATION_PROVIDER_ONLINE',
            'GENERATION_PROVIDER_LOCAL', 'CONVERSATIONAL_JSON_PARSE',
            'LOCAL_RESPONSE_VALIDATION', 'CONVERSATIONAL_REVIEW',
            'REVIEWER_PROVIDER_PRIMARY', 'REVIEWER_PROVIDER_FALLBACK',
            'REVIEWER_JSON_PARSE', 'OFFER_CONTRACT_PARSE', 'PUBLICATION', 'REALTIME_UI_BRIDGE')
        for stage in stages:
            with self.subTest(stage=stage):
                diagnostic, events = self.diagnostic()
                error = ValueError('api_key=private-token; personal content; /private/location')
                def fault():
                    raise error
                with self.assertRaises(ValueError) as raised:
                    diagnostic.guard(stage, fault)()
                self.assertIs(raised.exception, error)
                event, data = events[0]
                self.assertEqual(event, 'conversation_boundary_exception')
                self.assertEqual(data['stage'], stage)
                self.assertEqual(data['exception_type'], 'ValueError')
                self.assertEqual(data['sanitized_message'], '[REDACTED_EXCEPTION_TEXT]')
                self.assertEqual(data['turn_id'], 'turn-17')
                self.assertEqual(data['callsite']['function'], 'fault')
                self.assertTrue(data['callsite']['line'])
                self.assertTrue(data['traceback'])
                self.assertNotIn('private-token', json.dumps(events))
                self.assertNotIn('/private/location', json.dumps(events))

    def test_known_static_code_and_json_coordinates_without_document(self):
        d, events = self.diagnostic()
        with self.assertRaises(ValueError), d.stage('CONVERSATIONAL_JSON_PARSE'):
            raise ValueError('conversational_reply_schema_invalid')
        self.assertEqual(events[0][1]['sanitized_message'], 'conversational_reply_schema_invalid')
        with self.assertRaises(ValueError), d.stage('CONVERSATIONAL_JSON_PARSE'):
            json.loads('private-document')
        data = events[-1][1]
        self.assertEqual(data['sanitized_message'], 'JSON_DECODE_ERROR')
        self.assertEqual(data['json_position'], 0)
        self.assertNotIn('private-document', json.dumps(events))

    def test_success_has_identical_value_no_exception_event(self):
        d, events = self.diagnostic()
        value = object()
        self.assertIs(d.guard('PUBLICATION', lambda: value)(), value)
        self.assertFalse(events)

    def test_nested_boundaries_share_correlation_and_source_callsite(self):
        d, events = self.diagnostic()
        with self.assertRaises(ValueError), d.stage('SEMANTIC_ADAPTER'):
            with d.stage('CONVERSATIONAL_JSON_PARSE'):
                raise ValueError('conversational_reply_schema_invalid')
        self.assertEqual(len({v['correlation_id'] for _, v in events}), 1)
        self.assertEqual(events[0][1]['callsite'], events[1][1]['callsite'])

    def test_observer_failure_does_not_replace_application_exception(self):
        d = ConversationDiagnostics(lambda *a, **k: (_ for _ in ()).throw(OSError()), ROOT)
        error = ValueError('original')
        with self.assertRaises(ValueError) as raised, d.stage('PUBLICATION'):
            raise error
        self.assertIs(raised.exception, error)

    def test_actual_adapter_parse_boundary_before_realtime_can_swallow_error(self):
        d, events = self.diagnostic()
        with patch('anydes_tools.conversation.conversational_bridge.current_or_new', return_value=d):
            with self.assertRaises(ValueError):
                core.DomainContracts().adapter('CONTENT_RESPONSE', {'unexpected': 'private-value'})
        self.assertIn('CONVERSATIONAL_JSON_PARSE', [v['stage'] for _, v in events])
        self.assertEqual(events[0][1]['sanitized_message'], 'conversational_reply_schema_invalid')
        self.assertNotIn('private-value', json.dumps(events))

    def test_local_gate_rejection_records_reason_without_provider_or_private_text(self):
        from anydes_tools.conversation.natural_response import review_prose
        from anydes.realtime.validation import ValidationResult
        cases = [
            ('A generic invitation', ValidationResult(False, '', ('generic_response_only', 'empty_output'))),
            ('A prohibited claim', ValidationResult(False, '', ('unverified_action_claim',))),
            ('{private-payload}', None),
        ]
        for text, validation in cases:
            with self.subTest(validation=validation):
                d, events = self.diagnostic()
                provider = unittest.mock.Mock()
                with self.assertRaisesRegex(ValueError, 'natural_response_local_invariant_rejected'):
                    with d.stage('CONVERSATIONAL_REVIEW'):
                        review_prose(provider, [], text, {}, record=d.record, validation=validation)
                provider.assert_not_called()
                detail = next(v for e, v in events if e == 'conversation_local_gate_rejected')
                self.assertEqual(detail['stage'], 'LOCAL_REVIEW_GATE')
                self.assertNotIn(text, json.dumps(events))
                self.assertTrue(detail['candidate_protocol_prefix'] or detail['validation_codes'])


if __name__ == '__main__':
    unittest.main()
