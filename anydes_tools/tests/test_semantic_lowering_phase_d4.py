"""UNIT/INTEGRATION with prepared semantic cognition, real workers and B2."""
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from anydes_tools.contracts import ExecutionStatus, PermissionEffect
from anydes_tools.conversation.semantic_proposal import SemanticProposal, parse_semantic
from anydes_tools.conversation.plan_lowering import ExecutablePlanLowerer, PlanLoweringError
from anydes_tools.conversation.plan_protocol import PlanProtocolError
from anydes_tools.conversation.code_generation import GeneratedCodeArtifact
from anydes_tools.conversation.executable_plan import BrainPlanProposal
from anydes_tools.tests.test_artifact_dataflow_phase_d import DataflowIntegration
from anydes_tools.tests.test_interaction_execution_phase_c import action

def output(name, kind, fmt):
    return {"name":name,"artifact_type":kind,"format":fmt}

def operation(oid, kind, inputs, outputs, spec=None):
    return {"operation_id":oid,"operation_kind":kind,"inputs":inputs,"outputs":outputs,"specification":spec or {}}

def document_proposal():
    return {"goal":"generate content, create document and convert that document", "operations":[
        operation("content","GENERATE_CONTENT",[],[output("body","CONTENT","text")],
            {"content_kind":"expository_text","topic":"Anydes","target_length_words":150,"language":"es","source_mode":"GENERATIVE"}),
        operation("word","CREATE_DOCUMENT",["body"],[output("word_file","DOCUMENT","docx")]),
        operation("pdf","CONVERT_DOCUMENT",["word_file"],[output("final_pdf","DOCUMENT","pdf")])],
        "requested_final_outputs":["final_pdf"]}

def code_proposal():
    return {"goal":"create code, test it and archive that code", "operations":[
        operation("code","GENERATE_CODE",[],[output("main","CODE","py")],
            {"language":"python","goal":"print a result","functional_requirements":["print ok"],
             "tests":[{"test_id":"positive","stdin":"","expected_stdout_contains":["ok"]}],"execution_required":True}),
        operation("test","TEST_CODE",["main"],[]),
        operation("zip","CREATE_ARCHIVE",["main"],[output("archive","ARCHIVE","zip")])],
        "requested_final_outputs":["archive"]}

class SemanticLoweringTests(unittest.TestCase):
    setUp = DataflowIntegration.setUp
    tearDown = DataflowIntegration.tearDown
    controller = DataflowIntegration.controller

    def compiler(self, c):
        return ExecutablePlanLowerer(c.registry,self.workspace)

    def parsed(self, value):
        return SemanticProposal.from_dict(parse_semantic(json.dumps(value)))

    def test_word_plan_without_ids(self):
        c = self.controller()
        p = self.compiler(c).lower(self.parsed(document_proposal()),c._action_from_general_plan,c.plan_adapter,"Create content in a document and convert it")
        self.assertEqual(p.steps[0].produces_values,("body",))
        self.assertEqual(p.steps[1].consumes_values,("body",))
        self.assertEqual(p.steps[2].consumes_artifacts,("word_file",))
        self.assertEqual(p.requested_final_outputs[0]["binding"],"final_pdf")

    def test_code_plan_without_ids(self):
        c = self.controller()
        p = self.compiler(c).lower(self.parsed(code_proposal()),c._action_from_general_plan,c.plan_adapter,"Create tested code and archive it")
        self.assertEqual(p.steps[-1].consumes_artifacts,("main",))
        self.assertEqual(p.steps[-1].capability_id,"archive.compress_zip")

    def test_operational_fields_rejected(self):
        for key in ("capability_id","tool_id","path","artifact_id","verifier_type"):
            value = document_proposal()
            value["operations"][1]["specification"][key] = "untrusted"
            with self.subTest(key=key),self.assertRaises(PlanProtocolError): self.parsed(value)

    def test_unknown_operation_rejected(self):
        value = document_proposal()
        value["operations"][1]["operation_kind"] = "MAGIC"
        with self.assertRaises(PlanProtocolError): self.parsed(value)

    def test_unsupported_operation_closed(self):
        value = {"goal":"edit", "operations":[operation("edit","EDIT_ARTIFACT",[],[output("a","DOCUMENT","txt")])],
            "requested_final_outputs":["a"]}
        with self.assertRaisesRegex(PlanLoweringError,"UNSUPPORTED_OPERATION"):
            self.compiler(self.controller()).lower_requirements(self.parsed(value))

    def test_disabled_never_selected(self):
        c = self.controller()
        c.registry.disable_capability("document.create_pdf")
        with self.assertRaisesRegex(PlanLoweringError,"UNSUPPORTED_OPERATION"):
            self.compiler(c).lower_requirements(self.parsed(document_proposal()))

    def test_ambiguous_capabilities(self):
        c = self.controller()
        lower = self.compiler(c)
        extra = deepcopy(lower.metadata[0])
        extra["capability"] = "filesystem.create_text"
        lower.metadata = [*lower.metadata,extra]
        with self.assertRaisesRegex(PlanLoweringError,"PLAN_AMBIGUOUS:capability_selection"):
            lower.lower_requirements(self.parsed(code_proposal()))

    def test_single_input_inferred(self):
        value = document_proposal()
        value["operations"][-1]["inputs"] = []
        result = self.compiler(self.controller()).lower_requirements(self.parsed(value))
        self.assertEqual(result.steps[-1].consumes_artifacts,("word_file",))

    def test_multiple_documents_never_guessed(self):
        value = document_proposal()
        value["operations"].insert(2,operation("second","CREATE_DOCUMENT",["body"],[output("other","DOCUMENT","docx")]))
        value["operations"][-1]["inputs"] = []
        with self.assertRaisesRegex(PlanLoweringError,"PLAN_AMBIGUOUS:input_binding"):
            self.compiler(self.controller()).lower_requirements(self.parsed(value))

    def test_missing_dependency(self):
        value = document_proposal()
        value["operations"][-1]["inputs"] = ["missing"]
        with self.assertRaisesRegex(PlanLoweringError,"dependency_not_preceding"):
            self.compiler(self.controller()).lower_requirements(self.parsed(value))

    def test_type_incompatible(self):
        value = code_proposal()
        value["operations"][-1] = operation("convert","CONVERT_DOCUMENT",["main"],[output("archive","DOCUMENT","pdf")])
        with self.assertRaisesRegex(PlanLoweringError,"input_type"):
            self.compiler(self.controller()).lower_requirements(self.parsed(value))

    def test_providers_same_contract(self):
        c = self.controller()
        for provider in ("GPT-OSS","Qwen8B"):
            with self.subTest(provider=provider):
                result = self.compiler(c).lower_requirements(self.parsed(document_proposal()))
                self.assertEqual(result.steps[-1].capability,"document.create_pdf")

    def test_real_document_execution(self):
        c = self.controller()
        calls = []
        def prepared_content(spec):
            calls.append(spec)
            return " ".join(["Anydes proyecto personal conversacion contexto"]*30)
        c.semantic_prose_generator = prepared_content
        p = self.compiler(c).lower(self.parsed(document_proposal()),c._action_from_general_plan,c.plan_adapter,"Write content and convert the document")
        results = c._execute(p,turn_id="D4-local",confirmed_effects=())
        self.assertTrue(all(r.status==ExecutionStatus.SUCCESS for r in results),[r.error for r in results])
        self.assertEqual(len(calls),1)
        a = c._artifact_bindings.require("word_file",verified=True)
        b = c._artifact_bindings.require("final_pdf",verified=True)
        self.assertEqual(b.metadata["converted_from"],a.artifact_id)

    def test_real_code_archive_execution(self):
        import zipfile
        c = self.controller()
        c.code_generator = lambda spec: GeneratedCodeArtifact.from_dict({"language":"python","filename":spec.target_filename,
            "code":"print('ok')\n","tests":[{"test_id":"positive","stdin":"","expected_stdout_contains":["ok"]}]})
        p = self.compiler(c).lower(self.parsed(code_proposal()),c._action_from_general_plan,c.plan_adapter,"Create tested code and archive it")
        # This integration fixture supplies a locally reviewed artifact, not a model oracle.
        p.legacy_action.arguments["generated_code_artifact"] = {"language":"python", "filename":"main.py",
            "code":"print('ok')\n", "tests":[]}
        results = c._execute(p,turn_id="D4-code",confirmed_effects=(PermissionEffect.EXECUTE_CODE,))
        self.assertTrue(all(r.status==ExecutionStatus.SUCCESS for r in results),[r.error for r in results])
        with zipfile.ZipFile(self.workspace/"archive.zip") as z:
            self.assertEqual(z.namelist(),["main.py"])

    def test_semantic_repair_bounded(self):
        bad = document_proposal()
        bad["operations"][-1]["inputs"] = ["missing"]
        class ExplicitSemanticDouble:
            def __init__(self): self.repairs=0
            def __call__(self, request): return BrainPlanProposal(SemanticProposal.from_dict(bad))
            def repair(self, request, plan, errors):
                self.repairs+=1
                return BrainPlanProposal(plan)
        reasoner = ExplicitSemanticDouble()
        c = self.controller(action(),general_task_reasoner=reasoner)
        with patch("anydes_tools.conversation.controller.plan_from_text",return_value=None):
            response=c.handle("prepared document action",turn_id="D4-repair")
        self.assertFalse(response.executed)
        self.assertFalse(c._executor_called)
        self.assertEqual(reasoner.repairs,1)

    def test_conversion_reader_disabled(self):
        c = self.controller()
        c.registry.disable_capability("document.read_document")
        with self.assertRaisesRegex(PlanLoweringError,"UNSUPPORTED_OPERATION"):
            self.compiler(c).lower_requirements(self.parsed(document_proposal()))

    def test_active_brain_never_receives_internal_ids(self):
        from types import SimpleNamespace
        from anydes_tools.conversation.semantic_runtime import AssistantGeneralTaskReasoner
        class ExplicitSemanticTransportDouble(AssistantGeneralTaskReasoner):
            def __init__(self): self.messages=[]
            def complete(self,messages,**kwargs):
                self.messages.append(messages)
                return SimpleNamespace(audit=SimpleNamespace(record=lambda *a,**k:None)), SimpleNamespace(text=json.dumps(document_proposal()))
        brain = ExplicitSemanticTransportDouble()
        request={"user_request":"create and convert a document", "semantic_operations":[],
            "tool_registry":{"capabilities":["document.create_docx"]},"authorized_workspace":"C:\\private"}
        first=brain(request).proposal
        repaired=brain.repair(request,first,("PLAN_AMBIGUOUS:input_binding",)).proposal
        self.assertEqual(repaired.task_id,first.task_id)
        self.assertNotIn("document.create_docx",json.dumps(brain.messages))
        self.assertNotIn("C:\\private",json.dumps(brain.messages))
        with self.assertRaisesRegex(PlanProtocolError,"REPAIR_BUDGET_EXHAUSTED"):
            brain.repair(request,repaired,("still invalid",))
        self.assertEqual(len(brain.messages),2)

del DataflowIntegration
