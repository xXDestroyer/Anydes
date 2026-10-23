"""INTEGRATION: prepared content/proposal, real Registry, executor and transactions.

No claim of real brain coverage; the separate D.3 replay covers that boundary.
"""
import hashlib
import json
import unittest
import zipfile
from xml.etree import ElementTree

from anydes_tools.contracts import ExecutionStatus
from anydes_tools.conversation.general_reasoning import GeneralTaskPlan
from anydes_tools.conversation.plan_protocol import parse_plan
from anydes_tools.conversation.capability_grounding import CapabilityGroundingValidator
from anydes_tools.conversation.dataflow import validate_dataflow, file_hash
from anydes_tools.conversation.plan_completeness import PlanCompletenessValidator
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration


class WordPdfIntegration(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller

    def test_content_word_pdf_full_local_contract(self):
        c = self.controller()
        content = " ".join(["Anydes proyecto personal conversacion contexto"] * 30)
        calls = []
        def prepared_content(spec):
            calls.append(spec)
            return content
        c.semantic_prose_generator = prepared_content
        word = self.workspace / "Anydes.docx"
        pdf = self.workspace / "Anydes.pdf"
        output = {"binding": "PDF_C", "artifact_type": "DOCUMENT", "format": "pdf"}
        payload = {"task_id": "D3-local", "goal": "generate content, save Word and convert it to PDF",
            "task_type": "GENERATE_DOCUMENT", "inputs": {"content_spec": {
                "content_kind": "expository_text", "topic": "Anydes", "target_length_words": 150,
                "language": "es", "source_mode": "GENERATIVE"}},
            "steps": [
                {"step_id": "word", "capability": "document.create_docx", "purpose": "write generated body",
                 "arguments": {"path": str(word)}, "depends_on": [], "produces_artifacts": ["DOCX_B"],
                 "consumes_artifacts": [], "consumes_values": ["CONTENT_A"]},
                {"step_id": "pdf", "step_type": "CONVERSION", "capability": "document.create_pdf",
                 "purpose": "convert exact Word", "arguments": {"path": str(pdf)}, "depends_on": ["word"],
                 "produces_artifacts": ["PDF_C"], "consumes_artifacts": ["DOCX_B"]}],
            "required_capabilities": ["document.create_docx", "document.create_pdf"],
            "desired_outputs": [{"binding": "DOCX_B", "artifact_type": "DOCUMENT", "format": "docx"}, output],
            "requested_final_outputs": [output], "artifact_requirements": {"format": "docx", "filename": word.name},
            "needs_tools": True, "needs_content_generation": True}
        proposal = GeneralTaskPlan.from_dict(parse_plan(json.dumps(payload)))
        proposal = CapabilityGroundingValidator(c.registry).ground(proposal)
        self.assertTrue(c.general_plan_validator.validate(proposal,
            authorized_scopes=c.state.authorized_scopes).valid)
        action = c._action_from_general_plan(proposal, "Write 150 words about Anydes in Word and convert it to PDF")
        plan = c.plan_adapter.compile(action, source="GENERAL_REASONER", task_id=proposal.task_id)
        PlanCompletenessValidator().validate(plan)
        validate_dataflow(plan)
        generation = plan.steps[0]
        self.assertEqual(generation.produces_values, ("CONTENT_A",))
        self.assertEqual(plan.steps[1].consumes_values, ("CONTENT_A",))
        self.assertEqual(plan.steps[2].consumes_artifacts, ("DOCX_B",))
        results = c._execute(plan, turn_id="D3-local", confirmed_effects=())
        self.assertTrue(all(r.status == ExecutionStatus.SUCCESS for r in results), [r.error for r in results])
        self.assertEqual(len(calls), 1)
        self.assertEqual(c._artifact_bindings.values["CONTENT_A"], content)
        a = c._artifact_bindings.require("DOCX_B", verified=True)
        b = c._artifact_bindings.require("PDF_C", verified=True)
        self.assertEqual(a.hash, file_hash(word))
        self.assertEqual(b.hash, file_hash(pdf))
        self.assertEqual(b.metadata["converted_from"], a.artifact_id)
        step = next(s for s in c.last_step_results if s.step_id == "pdf")
        self.assertEqual(step.consumed_artifacts, (a.artifact_id,))
        with zipfile.ZipFile(word) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = ["".join(p.itertext()) for p in root.findall(".//w:p", ns)]
        self.assertEqual(paragraphs[0], "Anydes")
        # Paragraph boundaries are whitespace in text, not lost characters.
        body = " ".join(paragraphs[1:])
        self.assertEqual(body.split(), content.split())
        self.assertEqual(hashlib.sha256(c._artifact_bindings.values["CONTENT_A"].encode()).hexdigest(),
            hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(a.version, 1)
        self.assertEqual(b.version, 1)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        self.assertFalse(list(self.workspace.glob(".anydes-*")))
        print(json.dumps({"test_type": "INTEGRATION_PREPARED_COGNITION_REAL_EXECUTOR",
            "task_id": plan.task_id, "content_generation_calls": len(calls),
            "content_binding": "CONTENT_A", "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "docx": {"artifact_id": a.artifact_id, "version": a.version, "sha256": a.hash,
                "state": a.state.value},
            "pdf": {"artifact_id": b.artifact_id, "version": b.version, "sha256": b.hash,
                "state": b.state.value, "converted_from": b.metadata["converted_from"]},
            "source_hash_matches": a.hash == file_hash(word), "content_matches": body.split() == content.split(),
            "temporary_artifacts": len(list(self.workspace.glob(".anydes-*")))}))


del DataflowIntegration
