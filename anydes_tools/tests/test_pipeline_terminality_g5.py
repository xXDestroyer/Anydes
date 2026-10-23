"""UNIT prepared cognition/faults; registry and permissions are real, not UI E2E."""
import json
from unittest.mock import patch
import unittest
from anydes_tools.tests import test_interaction_execution_phase_c as phase_c
from anydes_tools.tests.test_interaction_execution_phase_c import action
from anydes_tools.tests.test_semantic_lowering_phase_d4 import code_proposal
from anydes_tools.conversation.semantic_proposal import SemanticProposal
from anydes_tools.conversation.executable_plan import BrainPlanProposal
from anydes_tools.conversation.interaction_router import InteractionResolution, InteractionKind, RequestedOutcome
from anydes_tools.conversation.publication import OutcomeStatus
from anydes_tools.conversation.verification_types import VerificationRequirement
from anydes_tools.conversation.pipeline_trace import PipelineTrace, exception_metadata


class Terminality(unittest.TestCase):
    setUp = phase_c.PhaseC.setUp
    tearDown = phase_c.PhaseC.tearDown
    controller = phase_c.PhaseC.controller

    def tool(self):
        return action(output_kind=RequestedOutcome.CODE_ARTIFACT, requested_operations=('GENERATE_CODE', 'TEST_CODE'),
                      requires_execution=True, verification_requirement=VerificationRequirement.TEST_SUITE)

    def reasoner(self, *, repair_valid=False):
        invalid = code_proposal()
        invalid['operations'] = invalid['operations'][:2]
        invalid['requested_final_outputs'] = ['main']
        invalid['operations'][0]['specification']['tests'][0]['timeout'] = 100
        valid = code_proposal()
        valid['operations'] = valid['operations'][:2]
        valid['requested_final_outputs'] = ['main']
        class Prepared:
            repairs = 0
            def __call__(self, request):
                return BrainPlanProposal(SemanticProposal.from_dict(invalid))
            def repair(self, request, plan, errors):
                self.repairs += 1
                return BrainPlanProposal(SemanticProposal.from_dict(valid if repair_valid else invalid))
        return Prepared()

    def test_post_lowering_strict_subcontract_rejected_with_one_repair(self):
        reasoner = self.reasoner()
        c = self.controller(self.tool(), general_task_reasoner=reasoner)
        response = c.handle('Build the requested artifact and verify it.', turn_id='invalid')
        self.assertIsNotNone(response)
        self.assertEqual(reasoner.repairs, 1)
        self.assertEqual(c.last_runtime_outcome.status, OutcomeStatus.FAILED)
        self.assertEqual(c.state.last_tool_failure.stage, 'PLAN_VALIDATION')
        self.assertEqual(c.state.last_tool_failure.error_code, 'test_spec_schema_invalid')
        self.assertTrue(c.last_runtime_outcome.failure_ref)
        self.assertFalse(c._executor_called)

    def test_valid_repair_keeps_contract_and_reaches_pending_confirmation(self):
        reasoner = self.reasoner(repair_valid=True)
        c = self.controller(self.tool(), general_task_reasoner=reasoner)
        c.handle('Build the requested artifact and verify it.', turn_id='repair')
        self.assertEqual(reasoner.repairs, 1)
        self.assertEqual(c.last_runtime_outcome.status, OutcomeStatus.PENDING_CONFIRMATION)
        self.assertFalse(c._executor_called)
        self.assertTrue(c.state.pending_task.plan)

    def test_unexpected_exception_bound_to_task_and_safe_failure(self):
        reasoner = self.reasoner()
        c = self.controller(self.tool(), general_task_reasoner=reasoner)
        with patch.object(c, '_validate_complete_proposal_checked',
                          side_effect=RuntimeError('secret=not-for-publication')):
            response = c.handle('Build the requested artifact and verify it.', turn_id='unexpected')
        self.assertIsNotNone(response)
        self.assertEqual(c.state.last_tool_failure.stage, 'PLAN_VALIDATION')
        self.assertEqual(c.state.last_tool_failure.error_code, 'UNEXPECTED_EXCEPTION_RuntimeError')
        self.assertEqual(c.last_runtime_outcome.task_id, c.state.last_tool_failure.task_id)
        self.assertNotIn('not-for-publication', c.state.last_tool_failure.technical_message)
        self.assertFalse(c._executor_called)

    def test_tool_none_cannot_escape_boundary(self):
        c = self.controller()
        def drop(*args, **kwargs):
            c.last_resolution = self.tool()
            return None
        with patch.object(c, '_handle_resolved_turn', side_effect=drop):
            response = c.handle('Prepared tool intent', turn_id='drop')
        self.assertIsNotNone(response)
        self.assertEqual(c.last_runtime_outcome.status, OutcomeStatus.FAILED)
        self.assertEqual(c.state.last_tool_failure.error_code, 'TOOL_ACTION_WITHOUT_TERMINAL_RESPONSE')

    def test_conversation_delegation_has_typed_outcome(self):
        c = self.controller(InteractionResolution(InteractionKind.CONVERSATION, confidence=1))
        self.assertIsNone(c.handle('Hello', turn_id='conversation'))
        self.assertEqual(c.last_runtime_outcome.status, OutcomeStatus.CONVERSATIONAL)
        self.assertFalse(c._executor_called)

    def test_missing_artifact_is_structured_clarification_not_workspace(self):
        c = self.controller(action(requested_operations=('CREATE_ARCHIVE',), references=('previous artifact',)))
        c.handle('Comprime ese archivo.', turn_id='missing')
        self.assertEqual(c.last_runtime_outcome.status, OutcomeStatus.CLARIFICATION)
        self.assertFalse(c._executor_called)

    def test_trace_has_enter_exit_and_safe_exception_metadata(self):
        class Audit:
            events = []
            def record(self, event, **details):
                self.events.append((event, details))
        trace = PipelineTrace(Audit(), 'unit', lambda: ('task:unit', 1))
        with self.assertRaises(ValueError):
            trace.invoke('TESTSPEC_BUILD', lambda: (_ for _ in ()).throw(ValueError('test_spec_schema_invalid')))
        self.assertEqual(trace.failed_stage, 'TESTSPEC_BUILD')
        self.assertEqual(trace.failed_reason, 'test_spec_schema_invalid')
        self.assertEqual(Audit.events[-1][1]['status'], 'FAILED')
        self.assertNotIn('private-value', json.dumps(exception_metadata(RuntimeError('private-value'))))


if __name__ == '__main__':
    unittest.main()
