"""UNIT contracts with explicit prepared brain outputs; not real-provider E2E."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from anydes_tools.tests.test_general_task_reasoner_v001_5 import QueueReasoner
from anydes_tools.tests.test_interaction_execution_phase_c import action
from dataclasses import replace
from anydes_tools.conversation.plan_protocol import parse_plan, PlanProtocolError, consume_repair
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.conversation.capability_grounding import CapabilityGroundingValidator, CapabilityAlias
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration


def payload():
    return {"goal": "write text", "task_type": "TOOL_COMPOSITION", "inputs": {},
        "steps": [{"step_id": "write", "capability": "document.create_txt", "purpose": "write",
            "arguments": {}, "depends_on": [], "produces_artifacts": ["text"], "consumes_artifacts": []}],
        "required_capabilities": ["document.create_txt"],
        "desired_outputs": [{"binding": "text", "artifact_type": "DOCUMENT", "format": "txt"}],
        "requested_final_outputs": [{"binding": "text", "artifact_type": "DOCUMENT", "format": "txt"}],
        "artifact_requirements": {}, "needs_tools": True, "needs_content_generation": False}


class ProtocolUnit(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_plan(json.dumps(payload())), payload())

    def test_no_free_extraction(self):
        for text in ("preface " + json.dumps(payload()), "```json\n" + json.dumps(payload()) + "\n```",
                json.dumps(payload()) + " trailing", "", '{"goal":"a","goal":"b"}', "NaN"):
            with self.subTest(text=text[:20]), self.assertRaises(PlanProtocolError):
                parse_plan(text)

    def test_nested_schema(self):
        bad = payload()
        bad["steps"][0]["consumes_artifacts"] = "text"
        with self.assertRaisesRegex(PlanProtocolError, "SCHEMA"):
            parse_plan(json.dumps(bad))

    def test_unknown_property(self):
        bad = {**payload(), "permission_granted": True}
        with self.assertRaises(PlanProtocolError):
            parse_plan(json.dumps(bad))

    def test_shared_budget(self):
        request = {}
        consume_repair(request, "PROTOCOL")
        with self.assertRaisesRegex(PlanProtocolError, "REPAIR_BUDGET_EXHAUSTED"):
            consume_repair(request, "CAPABILITY")

    def test_provider_independent(self):
        for model in ("GPT-OSS", "Qwen8B"):
            with self.subTest(model=model):
                self.assertEqual(parse_plan(json.dumps(payload())), payload())

    def bridge(self, outputs):
        from anydes_tools.conversation.semantic_runtime import LegacyAssistantGeneralTaskReasoner
        class ExplicitTransportDouble(LegacyAssistantGeneralTaskReasoner):
            def __init__(self):
                self.outputs = iter(outputs)
                self.calls = []
                self.events = []
            def complete(self, messages, **kwargs):
                self.calls.append(messages)
                audit = SimpleNamespace(record=lambda *a, **k: self.events.append((a, k)))
                return SimpleNamespace(audit=audit), SimpleNamespace(text=next(self.outputs))
        return ExplicitTransportDouble()

    def test_bad_json_repairs_once(self):
        bridge = self.bridge(['{"goal":', json.dumps(payload())])
        request = {}
        result = bridge(request)
        self.assertEqual(result.proposal.goal, "write text")
        self.assertEqual(len(bridge.calls), 2)
        self.assertTrue(request["plan_repair_budget"]["used"])
        self.assertNotIn("tool_registry", bridge.calls[1][1]["content"])

    def test_bad_repair_rejected(self):
        bridge = self.bridge(["not JSON", "still not JSON"])
        with self.assertRaises(PlanProtocolError):
            bridge({})
        self.assertEqual(len(bridge.calls), 2)

    def test_format_then_capability_repair_forbidden(self):
        bridge = self.bridge(["bad", json.dumps(payload())])
        request = {}
        plan = bridge(request).proposal
        with self.assertRaisesRegex(PlanProtocolError, "REPAIR_BUDGET_EXHAUSTED"):
            bridge.repair(request, plan, ("UNKNOWN_CAPABILITY",))
        self.assertEqual(len(bridge.calls), 2)


class GroundingUnit(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller

    def test_real_capability(self):
        c = self.controller()
        plan = GeneralTaskPlan.from_dict(payload())
        self.assertEqual(CapabilityGroundingValidator(c.registry).ground(plan), plan)

    def test_unknown_and_fuzzy_denied(self):
        c = self.controller()
        for cap in ("document.convert_to_pdf", "document.create_txtt"):
            bad = payload()
            bad["steps"][0]["capability"] = cap
            bad["required_capabilities"] = [cap]
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "UNKNOWN_CAPABILITY"):
                CapabilityGroundingValidator(c.registry).ground(GeneralTaskPlan.from_dict(bad))

    def test_disabled_denied(self):
        c = self.controller()
        c.registry.disable_capability("document.create_txt")
        with self.assertRaises(ValueError):
            CapabilityGroundingValidator(c.registry).ground(GeneralTaskPlan.from_dict(payload()))

    def test_explicit_alias_only(self):
        c = self.controller()
        bad = payload()
        bad["steps"][0]["capability"] = "fixture.text"
        bad["required_capabilities"] = ["fixture.text"]
        result = CapabilityGroundingValidator(c.registry,
            {"fixture.text": CapabilityAlias("document.create_txt")}).ground(GeneralTaskPlan.from_dict(bad))
        self.assertEqual(result.steps[0].capability, "document.create_txt")
        self.assertEqual(result.required_capabilities, ("document.create_txt",))

    def proposal(self, cap):
        value = payload()
        value["task_id"] = "d2-capability-fixture"
        value["steps"][0]["capability"] = cap
        value["steps"][0]["arguments"] = {"path": str(self.workspace / "grounded.txt"),
            "content": "Prepared contract fixture"}
        value["required_capabilities"] = [cap]
        return GeneralTaskPlan.from_dict(value)

    def test_unknown_capability_one_repair(self):
        reasoner = QueueReasoner(self.proposal("fixture.missing"), self.proposal("document.create_txt"))
        c = self.controller(action(), general_task_reasoner=reasoner)
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            response = c.handle("prepared artifact request", turn_id="grounding")
        self.assertTrue(response.executed, response.debug)
        self.assertTrue((self.workspace / "grounded.txt").is_file())

    def test_unknown_after_repair_never_executes(self):
        bad = self.proposal("fixture.missing")
        c = self.controller(action(), general_task_reasoner=QueueReasoner(bad, bad))
        with patch("anydes_tools.conversation.controller.plan_from_text", return_value=None):
            response = c.handle("prepared artifact request", turn_id="grounding")
        self.assertFalse(response.executed)
        self.assertFalse(c._executor_called)
        self.assertEqual(response.debug["repair_count"], 1)
        self.assertFalse((self.workspace / "grounded.txt").exists())


del DataflowIntegration
