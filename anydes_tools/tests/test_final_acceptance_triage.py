"""Downstream label equivalence with prepared decisions; no classifier modification."""
import unittest

from anydes_tools.tests import test_failure_topic_preservation as prior
from anydes_tools.tests.test_operation_grounding import resolved
from anydes_tools.conversation.publication import FinalResponseComposer


class LabelTriage(unittest.TestCase):
    setUp = prior.FailureTopic.setUp
    tearDown = prior.FailureTopic.tearDown
    controller = prior.FailureTopic.controller
    root = prior.FailureTopic.root

    def test_explain_and_continue_have_same_downstream_effects(self):
        observed = []
        for depth_control in ('EXPLAIN_FAILURE','CONTINUE_EXPLANATION'):
            c = self.controller(resolved('EXPLAIN_FAILURE'),resolved(depth_control),
                resolved('REMEDIATION_QUERY'),resolved('CONTENT_RESPONSE'))
            root = self.root(c)
            sequence = []
            for i,text in enumerate(('Cause of failure','Elaborate on this failure','Remediation assessment')):
                c.handle(text,turn_id=str(i))
                final = FinalResponseComposer.compose(c.last_runtime_outcome)
                self.assertEqual(c.last_runtime_outcome.failure_ref,root.failure_id)
                self.assertEqual(c.state.active_failure_topic.root_failure_id,root.failure_id)
                self.assertFalse(c._executor_called or c._reasoner_called)
                self.assertFalse(c.last_resolution.requested_effects or c.last_resolution.requires_execution)
                self.assertIsNone(c.state.pending_tool_action)
                sequence.append((final.visible_text,c.state.failure_explanation_depth,
                    c.state.previous_assistant_act,c.state.pending_task.status,
                    c.state.pending_task.committed_effects))
            self.assertEqual([v[1] for v in sequence],[1,2,2])
            self.assertNotEqual(sequence[0][0],sequence[1][0])
            c.handle('A separate question about sunlight',turn_id='exit')
            self.assertIsNone(c.state.failure_topic_record())
            observed.append(sequence)
        self.assertEqual(observed[0],observed[1])


if __name__ == '__main__':
    unittest.main()
