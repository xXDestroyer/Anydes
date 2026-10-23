"""UNIT typed checks plus INTEGRATION of real files and sandbox B2."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from anydes_tools.conversation.verification_types import TestSpec, VerificationStatus as S
from anydes_tools.conversation.typed_verification import VerificationDispatcher as V, ReferenceOracle
from anydes_tools.conversation.dataflow import ArtifactBindings, ArtifactState
from anydes_tools.conversation.grounding_evidence import verify_grounding
from anydes_tools.conversation.content_generation import ContentGenerationSpec, PostActionVerifier
from anydes_tools.tests.test_dataflow_closure_phase_d5 import DataflowClosureIntegration


class TypedUnit(unittest.TestCase):
    def check(self, test, output, trusted=True, exit_code=0):
        return V.verify("CODE", test=TestSpec.from_dict(test), stdout=output, exit_code=exit_code, trusted_oracle=trusted)

    def test_celsius_reference_not_lexical(self):
        reference = {"kind":"EXPRESSION", "expression":"x * 9 / 5 + 32", "variables":{"x":0}}
        for value, expected in ((0,32),(100,212),(-40,-40)):
            with self.subTest(value=value):
                r = self.check({"test_id":"affine", "stdin":str(value), "expected":-999,
                    "reference":reference, "comparison_mode":"NUMERIC"}, str(expected))
                self.assertEqual(r.status,S.PASS)
                self.assertEqual(r.evidence["oracle_source"],"INDEPENDENT_LOCAL_REFERENCE")
                self.assertTrue(r.evidence["generated_expectation_disagrees"])

    def test_count_reference_normalization(self):
        reference={"kind":"COUNT_CHARACTERS","alphabet":"aeiou","casefold":True}
        r=self.check({"test_id":"letters", "stdin":"Python es genial", "expected":6,
            "reference":reference,"comparison_mode":"NUMERIC"},"5\r\n")
        self.assertEqual(r.status,S.PASS)
        self.assertEqual(ReferenceOracle.evaluate(reference,"PYTHON es genial"),5)
        self.assertEqual(ReferenceOracle.evaluate(reference,"aÁA"),2)
        self.assertEqual(ReferenceOracle.evaluate({**reference,"strip_accents":True},"aÁA"),3)

    def test_invalid_input_contract(self):
        r=self.check({"test_id":"invalid", "input":"abc", "expected":"Entrada invalida",
            "expected_exit_code":0},"Entrada invalida\n")
        self.assertEqual(r.status,S.PASS)
        self.assertEqual(self.check({"test_id":"invalid", "expected":"Entrada invalida"},"32").status,S.FAIL)

    def test_substring_false_positive_rejected(self):
        self.assertEqual(self.check({"test_id":"exact","expected_stdout_contains":["6"]},"16").status,S.FAIL)
        self.assertEqual(self.check({"test_id":"exact","expected":"ok"},"not ok").status,S.FAIL)

    def test_tolerance_and_numeric_bounds(self):
        t={"test_id":"float","expected":1,"comparison_mode":"FLOAT_TOLERANCE","tolerance":.01}
        self.assertEqual(self.check(t,"1.005").status,S.PASS)
        for out in ("1.02","nan","inf","value=1.0"):
            self.assertEqual(self.check(t,out).status,S.FAIL)

    def test_untrusted_oracle_inconclusive_even_claiming_trust(self):
        r=self.check({"test_id":"model","expected":6,"oracle_source":"TRUSTED",
            "reference":{"kind":"EXPRESSION","expression":"6"}},"6",trusted=False)
        self.assertEqual(r.status,S.INCONCLUSIVE)
        self.assertEqual(r.error_code,"ORACLE_UNVERIFIED")
        self.assertEqual(r.legacy()["status"],"VERIFY_FAILED")

    def test_reference_no_code_or_calls(self):
        with self.assertRaises(ValueError):
            ReferenceOracle.evaluate({"kind":"EXPRESSION","expression":"__import__('os').system('x')"},"")

    def test_structured_types_and_properties(self):
        self.assertEqual(self.check({"test_id":"json","comparison_mode":"STRUCTURED","expected":{"a":1}},'{"a":true}').status,S.FAIL)
        self.assertEqual(self.check({"test_id":"property","comparison_mode":"PROPERTY","expected":"finite_number"},"2").status,S.PASS)

    def test_unknown_dispatch_inconclusive(self):
        self.assertEqual(V.verify("UNKNOWN").status,S.INCONCLUSIVE)

    def test_memory_grounding_does_not_claim_universal_truth(self):
        r=verify_grounding([], {"context":"Known project"},required=True)
        self.assertEqual(r.status,S.INCONCLUSIVE)
        self.assertEqual(verify_grounding([],{},required=False).status,S.SKIPPED)


class TypedFiles(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_docx_body_not_title(self):
        from anydes_tools.official.document_tool import _write_docx
        path=self.root/"body.docx"
        body=" ".join(["energia"]*180)
        title=" ".join(["titulo"]*80)
        _write_docx(path,body,title)
        r=V.verify("DOCUMENT",path=path,format_id="docx",body_text=body,title=title,
            target_words=180,expected_text=title+"\n"+body)
        self.assertEqual(r.status,S.PASS)
        self.assertEqual(r.evidence["body_word_count"],180)
        spec=ContentGenerationSpec.from_dict({"topic":"energia","target_length_words":180})
        self.assertEqual(PostActionVerifier.generated_creation(title+"\n"+body,title+"\n"+body,spec,generated_body=body)["status"],"VERIFIED")

    def test_pdf_empty_and_conversion(self):
        from anydes_tools.official.document_tool import _write_docx,_write_pdf
        source=self.root/"source.docx"; target=self.root/"target.pdf"
        _write_docx(source,"Exact source text","")
        _write_pdf(target,"","")
        self.assertEqual(V.verify("PDF",path=target).status,S.FAIL)
        _write_pdf(target,"Exact source text","")
        ledger=ArtifactBindings("task")
        a=ledger.produce("source",source,"source",state=ArtifactState.COMMITTED)
        b=ledger.produce("target",target,"convert")
        r=V.verify("CONVERSION",source_ref=a,target_ref=b,expected_source_id=a.artifact_id)
        self.assertEqual(r.status,S.PASS)
        self.assertEqual(r.verifier_type,"CONVERSION")
        self.assertEqual(V.verify("CONVERSION",source_ref=a,target_ref=b,expected_source_id="wrong").status,S.FAIL)

    def test_archive_exactness(self):
        path=self.root/"test.zip"
        with zipfile.ZipFile(path,"w") as z: z.writestr("only.py",b"print(1)")
        expected={"only.py":hashlib.sha256(b"print(1)").hexdigest()}
        self.assertEqual(V.verify("ARCHIVE",path=path,expected_members=expected).status,S.PASS)
        with zipfile.ZipFile(path,"a") as z: z.writestr("extra",b"x")
        self.assertEqual(V.verify("ARCHIVE",path=path,expected_members=expected).status,S.FAIL)

    def test_filesystem_real_effects(self):
        before=self.root/"old.txt"; after=self.root/"new.txt"
        before.write_text("old")
        self.assertEqual(V.verify("FILESYSTEM_CHANGE",operation="CREATE",target=before).status,S.PASS)
        old_hash=hashlib.sha256(before.read_bytes()).hexdigest()
        before.write_text("changed")
        self.assertEqual(V.verify("FILESYSTEM_CHANGE",operation="EDIT",target=before,before_hash=old_hash).status,S.PASS)
        before.rename(after)
        for op in ("RENAME","MOVE"):
            self.assertEqual(V.verify("FILESYSTEM_CHANGE",operation=op,target=after,source=before).status,S.PASS)
        after.unlink()
        self.assertEqual(V.verify("FILESYSTEM_CHANGE",operation="DELETE",target=after).status,S.PASS)


class TypedIntegration(unittest.TestCase):
    setUp=DataflowClosureIntegration.setUp
    tearDown=DataflowClosureIntegration.tearDown
    execute_fixture=DataflowClosureIntegration.execute_fixture

    def test_real_b2_independent_reference_and_invalid_input(self):
        from anydes_tools.conversation.contracts import PlannedToolAction
        from anydes_tools.contracts import PermissionEffect
        source="try:\n    value = float(input())\n    print(value * 9 / 5 + 32)\nexcept ValueError:\n    print('Entrada invalida')\n"
        reference={"kind":"EXPRESSION","expression":"x*9/5+32","variables":{"x":0}}
        tests=[{"test_id":"zero","stdin":"0\n","expected":-999,"comparison_mode":"NUMERIC","reference":reference},
            {"test_id":"hundred","stdin":"100\n","expected":-999,"comparison_mode":"NUMERIC","reference":reference},
            {"test_id":"invalid","stdin":"abc\n","expected":"Entrada invalida"}]
        action=PlannedToolAction("generate_code",("developer.write_code","developer.run_python"),"general_generate_code",
            arguments={"path":str(self.workspace/"affine.py"),
                "code_generation_spec":{"language":"python","target_filename":"affine.py",
                    "goal":"implement an affine numerical transform", "tests":tests,"execution_required":True},
                "generated_code_artifact":{"language":"python","filename":"affine.py","code":source,"tests":[]}},
            source_text="prepared reference integration")
        c=self.controller
        c._remember_task_plan(action)
        results=c._execute(c.state.pending_task.plan,turn_id="E-reference",confirmed_effects=(PermissionEffect.EXECUTE_CODE,))
        self.assertTrue(all(r.status.value=="SUCCESS" for r in results),[r.error for r in results])
        evidence=[e.verification_result for e in c.state.verification_evidence.values()]
        self.assertEqual(len(evidence),3)
        self.assertTrue(all(e["status"]=="PASS" for e in evidence))
        self.assertTrue(all(e["evidence"]["oracle_source"]=="INDEPENDENT_LOCAL_REFERENCE" for e in evidence[:2]))

    def test_failure_record_retains_typed_result(self):
        from anydes_tools.conversation.contracts import PlannedToolAction
        from anydes_tools.contracts import PermissionEffect
        action=PlannedToolAction("generate_code",("developer.write_code","developer.run_python"),"general_generate_code",
            arguments={"path":str(self.workspace/"fail.py"),
                "code_generation_spec":{"language":"python","target_filename":"fail.py","goal":"print a value",
                    "tests":[{"test_id":"typed_fail","expected":7}],"execution_required":True},
                "generated_code_artifact":{"language":"python","filename":"fail.py","code":"print(6)\n","tests":[]}},
            source_text="prepared evidence integration")
        c=self.controller
        c._remember_task_plan(action)
        c._resolve_and_prepare(action,turn_id="E-prepare")
        from anydes_tools.conversation.interaction_router import TaskControl
        response=c._handle_pending("confirm", "E-run",control=TaskControl.CONFIRM)
        self.assertFalse(response.executed)
        failure=c.state.last_tool_failure
        self.assertEqual(failure.error_code,"TEST_ASSERTION_FAILED")
        stored=c.state.verification_evidence[failure.verification_ref].verification_result
        self.assertEqual(stored["failed_check"],"typed_fail")
        self.assertEqual(stored["evidence"]["comparison_mode"],"EXACT")
        self.assertTrue(stored["artifact_id"])

    def test_real_b2_typed_evidence_and_dependency_block(self):
        _,results,summary,evidence=self.execute_fixture("7")
        failure=next(r for r in results if r.status.value=="VERIFY_FAILED")
        detail=failure.output["verification_evidence"]["verification_result"]
        self.assertEqual(detail["verifier_type"],"CODE")
        self.assertEqual(detail["status"],"FAIL")
        self.assertEqual(detail["failed_check"],"sum_check")
        self.assertEqual(detail["evidence"]["comparison_mode"],"EXACT")
        self.assertEqual(summary["blocked_steps"],["archive"])

del DataflowClosureIntegration
