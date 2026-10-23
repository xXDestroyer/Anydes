"""UNIT/INTEGRATION; prepared intent, real registry and executor. No human E2E claim."""
import unittest
from anydes_tools.conversation.structured_text import ContentBuilder
from anydes_tools.conversation.planner import plan_from_text
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration


class CandidateBlockers(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller

    def test_literal_is_not_reinterpreted_as_instruction(self):
        value = 'El texto menciona numeros del 1 al 5 y dos titulos.'
        spec = ContentBuilder().build('Crea una nota con el texto: ' + value, value)
        self.assertEqual(spec.render(), value)
        self.assertEqual(spec.source_kind, 'literal')

    def test_empty_literal_is_not_a_complete_fast_path(self):
        c = self.controller()
        self.assertIsNone(plan_from_text('Crea una nota.docx.', c.state))

    def test_g1_word_literal_fixture_no_empty_artifact(self):
        c = self.controller()
        request = 'Crea ejemplo.docx con el titulo Prueba y el texto: Esta es una prueba segura.'
        action = plan_from_text(request, c.state)
        self.assertEqual(action.arguments['content'], 'Esta es una prueba segura.')
        plan = c.plan_adapter.compile(action, source='FAST_PATH')
        results = c._execute(plan, turn_id='g2-fixture', confirmed_effects=())
        self.assertTrue(all(r.status.value == 'SUCCESS' for r in results), [r.error for r in results])
        self.assertTrue((self.workspace / 'ejemplo.docx').is_file())

    def test_equivalent_effect_preserves_text_output(self):
        c = self.controller()
        c.plan_adapter.requested_effects_provider = lambda: ('filesystem.create_text',)
        action = plan_from_text('Crea ejemplo.txt con el texto: Prueba segura.', c.state)
        plan = c.plan_adapter.compile(action, source='FAST_PATH')
        self.assertEqual(plan.steps[0].capability_id, 'document.create_txt')
        self.assertIn({'artifact_type':'DOCUMENT', 'format':'txt'}, plan.requested_final_outputs)
        results = c._execute(plan, turn_id='g2-fixture', confirmed_effects=())
        self.assertTrue(all(r.status.value == 'SUCCESS' for r in results), [r.error for r in results])
        self.assertEqual((self.workspace / 'ejemplo.txt').read_text(), 'Prueba segura.')

    def test_wrong_format_cannot_cover_requested_effect(self):
        c = self.controller()
        c.plan_adapter.requested_effects_provider = lambda: ('filesystem.create_text',)
        action = plan_from_text('Crea ejemplo.docx con el texto: Prueba segura.', c.state)
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_EFFECTS'):
            c.plan_adapter.compile(action, source='FAST_PATH')

    def test_mutation_not_covered_by_creation(self):
        c = self.controller()
        c.plan_adapter.requested_effects_provider = lambda: ('filesystem.delete',)
        action = plan_from_text('Crea ejemplo.txt con el texto: Prueba segura.', c.state)
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_EFFECTS'):
            c.plan_adapter.compile(action, source='FAST_PATH')


del DataflowIntegration
