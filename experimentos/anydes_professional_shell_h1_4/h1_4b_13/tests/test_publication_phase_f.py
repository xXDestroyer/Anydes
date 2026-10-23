"""UNIT publication matrix; INTEGRATION prepared cognition with real B2/evidence."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

from anydes_tools.conversation.publication import (
    RuntimeOutcome, OutcomeStatus as S, FinalResponseComposer as Composer,
    outcome_from_controller, sanitize_public_text)
from anydes_tools.conversation.interaction_router import InteractionKind, TaskControl
from anydes_tools.tests.test_semantics_propagation_phase_e4 import RuntimePropagation, UNITS
from anydes_tools.tests import test_failure_record_phase_c as failure_fixture
from anydes_tools.tests import test_artifact_dataflow_phase_d as dataflow_fixture


ARTIFACT = {"artifact_id": "artifact:known", "task_id": "task:known", "state": "COMMITTED",
    "artifact_type": "CODE", "display_name": "contador.py"}


class PublicationMatrix(unittest.TestCase):
    def outcome(self, status, **kwargs):
        return RuntimeOutcome(status=status, task_id="task:known", **kwargs)

    def test_conversation(self):
        f = Composer.compose(RuntimeOutcome(), conversational_text="Estoy bien, gracias. ¿Y tú?")
        self.assertEqual(f.visible_text, "Estoy bien, gracias. ¿Y tú?")
        self.assertEqual(f.visible_text, f.spoken_text)

    def test_content(self):
        f = Composer.compose(RuntimeOutcome(interaction_kind="CONTENT_RESPONSE"), conversational_text="La energía se conserva.")
        self.assertEqual(f.visible_text, "La energía se conserva.")
        self.assertFalse(f.artifact_names)

    def test_content_introduction_not_a_tool_claim(self):
        f = Composer.compose(RuntimeOutcome(interaction_kind="CONTENT_RESPONSE"),
            conversational_text="Aquí tienes una explicación: la energía se conserva.")
        self.assertIn("la energía se conserva", f.visible_text)

    def test_unbacked_pronominal_action_claim(self):
        f = Composer.compose(RuntimeOutcome(), conversational_text="Ya lo guardé")
        self.assertIn("No tengo una acción verificada", f.visible_text)

    def test_success(self):
        f = Composer.compose(self.outcome(S.SUCCESS, committed_artifacts=(ARTIFACT,),
            verification_summary=({"status": "PASS"},)), conversational_text="Falló todo")
        self.assertIn("contador.py", f.visible_text)
        self.assertIn("Pasó", f.visible_text)
        self.assertNotIn("Falló", f.visible_text)

    def test_unbacked_success(self):
        for kwargs in ({}, {"committed_artifacts": (ARTIFACT,)}):
            f = Composer.compose(self.outcome(S.SUCCESS, **kwargs), conversational_text="Listo, ya lo guardé")
            self.assertEqual(f.decision.status, S.INCONCLUSIVE)
            self.assertNotIn("Listo", f.visible_text)

    def test_failed(self):
        f = Composer.compose(self.outcome(S.FAILED, safe_evidence={"failure_message": "El programa no superó la prueba 3."}),
            conversational_text="Listo, todo perfecto")
        self.assertIn("prueba 3", f.visible_text)
        self.assertNotIn("documento", f.visible_text)
        self.assertNotIn("Listo", f.visible_text)

    def test_inconclusive(self):
        f = Composer.compose(self.outcome(S.INCONCLUSIVE), conversational_text="El programa funciona")
        self.assertIn("evidencia suficiente", f.visible_text)
        self.assertNotIn("funciona", f.visible_text)

    def test_partial(self):
        f = Composer.compose(self.outcome(S.PARTIAL_SUCCESS, committed_artifacts=(ARTIFACT,),
            completed_steps=("create",), failed_step="conversion", blocked_steps=("archive",),
            safe_evidence={"failure_message": "La conversión no pudo verificarse."}))
        self.assertIn("contador.py", f.visible_text)
        self.assertIn("conversión", f.visible_text)
        self.assertIn("no se ejecutaron", f.visible_text)
        self.assertNotIn("Listo", f.visible_text)

    def test_pending(self):
        f = Composer.compose(self.outcome(S.PENDING_CONFIRMATION, pending_action={"effects": ["EXECUTE_CODE"]}))
        self.assertIn("aprobación", f.visible_text)
        self.assertIn("ejecutar el código", f.visible_text)
        self.assertNotIn("ejecuté", f.visible_text)

    def test_delete_confirmation_targets_preserved(self):
        f = Composer.compose(self.outcome(S.PENDING_CONFIRMATION,
            pending_action={"effects": ["DELETE"], "target_names": ["a.txt", "b.txt"]}))
        self.assertIn("a.txt, b.txt", f.visible_text)
        self.assertIn("no se puede deshacer", f.visible_text)

    def test_clarification(self):
        f = Composer.compose(self.outcome(S.CLARIFICATION, clarification="¿Cuál de los dos archivos quieres convertir?"))
        self.assertEqual(f.visible_text, "¿Cuál de los dos archivos quieres convertir?")

    def test_unsupported(self):
        self.assertIn("herramienta autorizada", Composer.compose(self.outcome(S.UNSUPPORTED)).visible_text)

    def test_latest_retry(self):
        first = self.outcome(S.FAILED, attempt_number=1)
        second = replace(first, status=S.SUCCESS, attempt_number=2,
            committed_artifacts=(ARTIFACT,), verification_summary=({"status": "PASS"},))
        self.assertIn("nuevo intento", Composer.compose(second).visible_text)

    def test_internal_json(self):
        for text in ('{"artifact_id":"artifact:private"}', '{"nested":{"tool_id":"official.developer"}}',
            '```json\n{"capability_id":"developer.test"}\n```', '<tool_call>{}</tool_call>',
            '<think>internal deliberation</think>', '<analysis>unclosed internal deliberation'):
            f = Composer.compose(RuntimeOutcome(), conversational_text=text)
            self.assertEqual(f.visible_text, "No puedo mostrar ese contenido interno.")

    def test_private_path(self):
        f = Composer.compose(RuntimeOutcome(), conversational_text=r"Está en C:\Users\DemoUser\workspace\archivo.txt")
        self.assertNotIn("Private", f.visible_text)
        self.assertIn("Carpeta de trabajo", f.visible_text)

    def test_secrets(self):
        clean, changed = sanitize_public_text("Bearer private-value api_key=private-value")
        self.assertTrue(changed)
        self.assertNotIn("private-value", clean)

    def test_friendly_name(self):
        f = Composer.compose(self.outcome(S.SUCCESS,
            committed_artifacts=({**ARTIFACT, "display_name": r"C:\Users\DemoUser\contador.py"},),
            verification_summary=({"status": "PASS"},)))
        self.assertEqual(f.artifact_names, ("contador.py",))

    def test_wrong_artifact_not_publishable(self):
        f = Composer.compose(self.outcome(S.SUCCESS,
            committed_artifacts=({**ARTIFACT, "task_id": "task:other"},),
            verification_summary=({"status": "PASS"},)))
        self.assertEqual(f.decision.status, S.INCONCLUSIVE)
        self.assertFalse(f.artifact_names)

    def test_temporary_artifact_not_publishable(self):
        f = Composer.compose(self.outcome(S.SUCCESS,
            committed_artifacts=({**ARTIFACT, "state": "TEMPORARY"},),
            verification_summary=({"status": "PASS"},)))
        self.assertEqual(f.decision.status, S.INCONCLUSIVE)

    def test_failed_verification_not_publishable(self):
        f = Composer.compose(self.outcome(S.SUCCESS, committed_artifacts=(ARTIFACT,),
            verification_summary=({"status": "FAIL"},)))
        self.assertEqual(f.decision.status, S.FAILED)

    def test_negative_verification_overrides_effect_flag(self):
        f = Composer.compose(self.outcome(S.SUCCESS, safe_evidence={"verified_effects": [{"effect": "CREATE"}]},
            verification_summary=({"status": "INCONCLUSIVE"},)))
        self.assertEqual(f.decision.status, S.INCONCLUSIVE)
        self.assertNotIn("Listo", f.visible_text)

    def test_history_does_not_authorize_new_effect_claim(self):
        f = Composer.compose(RuntimeOutcome(safe_evidence={"historical_action_evidence": True}),
            conversational_text="Ya guardé el archivo nuevo")
        self.assertNotIn("archivo nuevo", f.visible_text)

    def test_stream_buffer(self):
        from anydes_tools.conversation.runtime_patch import _guarded_callback
        emitted = []
        callback = _guarded_callback(None, emitted.append, "fixture", None)
        callback("Listo, ya lo creé")
        callback("untrusted partial")
        self.assertEqual(emitted, [])
        callback.finish("approved final")
        self.assertEqual(emitted, ["approved final"])

    def test_runtime_exception_uses_publication_not_raw_trace(self):
        from anydes_tools.conversation.runtime_patch import _publication_guard
        class AssistantDouble:
            audit = SimpleNamespace(record=lambda *a, **k: None)
        a = AssistantDouble()
        from anydes_tools.tests import test_outcome_voice as voice_fixture
        from anydes_tools.tests.test_natural_response import review_payload
        generated,_,_=voice_fixture.OutcomeVoiceTests().assistant([
            {'text':'No consegui completar mi respuesta; podemos retomarla.'},review_payload()])
        a._realtime_service=generated._realtime_service
        a.personality=generated.personality
        def bad(assistant, text, **kwargs):
            raise ValueError("private C:\\Users\\DemoUser\\secret.txt")
        old = list(sys.path)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "programa_principal"))
        try:
            chunks = []
            result = _publication_guard(bad)(a, "fixture", on_text=chunks.append)
            self.assertEqual(a.last_final_response.decision.status, S.FAILED)
            self.assertEqual(chunks, [result])
            self.assertEqual(a.last_intention.speech_intention.text, result)
            self.assertNotIn("Private", result)
        finally:
            sys.path[:] = old

    def test_superseded_turn_not_published(self):
        from anydes_tools.conversation.runtime_patch import _publication_guard
        old = list(sys.path)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "programa_principal"))
        try:
            from anydes.realtime.cancellation import ResponseSuperseded
            def cancelled(assistant, text):
                raise ResponseSuperseded()
            with self.assertRaises(ResponseSuperseded):
                _publication_guard(cancelled)(SimpleNamespace(), "fixture")
        finally:
            sys.path[:] = old


class PublicationRuntime(unittest.TestCase):
    setUp = RuntimePropagation.setUp
    tearDown = RuntimePropagation.tearDown
    execute = RuntimePropagation.execute

    def resolution(self, control=TaskControl.CONFIRM):
        self.controller.last_resolution = SimpleNamespace(kind=InteractionKind.TOOL_ACTION, task_control=control)

    def test_real_pass_message(self):
        self.execute(UNITS, "x=float(input())\nprint(x*9/5+32)\n", ["0"])
        self.resolution()
        outcome = outcome_from_controller(self.controller)
        self.assertEqual(outcome.status, S.SUCCESS)
        f = Composer.compose(outcome)
        self.assertIn("known.py", f.visible_text)
        self.assertIn("Pasó", f.visible_text)
        self.assertEqual(f.visible_text, f.spoken_text)

    def test_real_fail_explanation(self):
        self.execute(UNITS, "x=input()\nprint(999)\n", ["0"])
        self.resolution(TaskControl.EXPLAIN_FAILURE)
        before = len(self.controller.last_results)
        outcome = outcome_from_controller(self.controller)
        self.assertEqual(outcome.failure_ref, self.controller.state.last_tool_failure.failure_id)
        f = Composer.compose(outcome)
        self.assertIn("programa", f.visible_text)
        self.assertNotIn("documento", f.visible_text)
        self.assertEqual(len(self.controller.last_results), before)

    def test_previous_failure_not_cause_of_new_unresolved_request(self):
        self.execute(UNITS, "x=input()\nprint(999)\n", ["0"])
        self.resolution(TaskControl.NONE)
        c = self.controller
        prior = c.state.last_tool_failure.failure_id
        c.last_results = ()
        c.last_step_results = []
        outcome = outcome_from_controller(c, prior_failure_id=prior)
        self.assertEqual(outcome.failure_ref, "")
        self.assertEqual(outcome.task_id, "")
        self.assertNotIn("prueba", Composer.compose(outcome).visible_text)

    def test_protocol_resolution_failure_not_presented_as_user_ambiguity(self):
        c = self.controller
        c.last_resolution = SimpleNamespace(kind=InteractionKind.CLARIFICATION,
            task_control=TaskControl.NONE, source="LOCAL_FAIL_CLOSED")
        outcome = outcome_from_controller(c)
        self.assertEqual(outcome.status, S.FAILED)
        self.assertIn("interpretación válida", Composer.compose(outcome).visible_text)


class RetryPublicationRuntime(unittest.TestCase):
    setUp = failure_fixture.FailureRecordIntegration.setUp
    tearDown = failure_fixture.FailureRecordIntegration.tearDown
    prepared_action = failure_fixture.FailureRecordIntegration.prepared_action

    def controller(self, *decisions):
        c = failure_fixture.FailureRecordIntegration.controller(self, *decisions)
        original = c.handle
        self.messages = []
        def observed(text, *, turn_id):
            response = original(text, turn_id=turn_id)
            outcome = outcome_from_controller(c, response)
            self.messages.append((turn_id, outcome, Composer.compose(outcome, conversational_text=response.text)))
            return response
        c.handle = observed
        return c

    def test_failure_explain_retry_success_latest_publication(self):
        # Existing isolated C fixture: explicit classifier double and one verifier
        # failure injection; executor/B2/failure/retry/commit remain real.
        failure_fixture.FailureRecordIntegration.test_real_b2_propagation_explanation_retry_and_history(self)
        why = next(v for turn, _, v in self.messages if turn == "why")
        final_outcome, final = next((o, v) for turn, o, v in self.messages if turn == "confirm")
        self.assertIn("test_003", why.visible_text)
        self.assertNotIn("documento", why.visible_text)
        self.assertEqual(final_outcome.attempt_number, 2)
        self.assertEqual(final.decision.status, S.SUCCESS)
        self.assertIn("nuevo intento", final.visible_text)
        self.assertIn("sum.py", final.visible_text)
        self.assertEqual(final.visible_text, final.spoken_text)


class CompositionPublicationRuntime(unittest.TestCase):
    setUp = dataflow_fixture.DataflowIntegration.setUp
    tearDown = dataflow_fixture.DataflowIntegration.tearDown
    controller = dataflow_fixture.DataflowIntegration.controller
    prepared_action = dataflow_fixture.DataflowIntegration.prepared_action
    code_action = dataflow_fixture.DataflowIntegration.code_action

    def execute(self, c, action):
        from anydes_tools.contracts import CapabilityResolutionKind
        results = dataflow_fixture.DataflowIntegration.execute(self, c, action)
        c.last_results = tuple(results)
        c.last_resolution = SimpleNamespace(kind=InteractionKind.TOOL_ACTION, task_control=TaskControl.NONE)
        resolution = SimpleNamespace(kind=CapabilityResolutionKind.COMPOSABLE, tool_ids=())
        response = c._response_from_results(action, results, resolution, turn_id="F-composition")
        self.outcome = outcome_from_controller(c, response)
        self.final = Composer.compose(self.outcome)
        return results

    def test_real_partial_message(self):
        dataflow_fixture.DataflowIntegration.test_partial_success_records_committed_and_blocked(self)
        self.assertEqual(self.final.decision.status, S.PARTIAL_SUCCESS)
        self.assertIn("retained.txt", self.final.visible_text)
        self.assertIn("no se ejecutaron", self.final.visible_text)
        self.assertNotIn("Listo", self.final.visible_text)

    def test_real_word_pdf_message(self):
        dataflow_fixture.DataflowIntegration.test_word_pdf_exact_binding(self)
        self.assertEqual(self.final.decision.status, S.SUCCESS)
        self.assertIn("source.pdf", self.final.visible_text)
        self.assertNotIn(str(self.workspace), self.final.visible_text)

    def test_real_code_zip_message(self):
        dataflow_fixture.DataflowIntegration.test_code_test_zip_exact(self)
        self.assertEqual(self.final.decision.status, S.SUCCESS)
        self.assertIn("sum.zip", self.final.visible_text)


del RuntimePropagation
