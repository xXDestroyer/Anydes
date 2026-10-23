"""Run one named test gate once and keep its result; never retries failures."""
import json
from pathlib import Path
import sys
import unittest
import bootstrap

bootstrap.install()
import anydes_tools.tests as tests
tests.__path__.insert(0,str(bootstrap.HERE/'tests'))
gates={
    'classifier':['anydes_tools.tests.test_classifier_wire'],
    'interaction':['anydes_tools.tests.test_classifier_request_contract',
        'anydes_tools.tests.test_classifier_canonical_output','anydes_tools.tests.test_continuation_public',
        'anydes_tools.tests.test_clarification_anchor','anydes_tools.tests.test_continue_task_contract',
        'anydes_tools.tests.test_clarification_completeness','anydes_tools.tests.test_goal_resolvability'],
    'usage':['anydes_tools.tests.test_usage_race'],
    'cpp':['anydes_tools.tests.test_testspec_contract.ControlledE2E.test_cpp_real_runtime_b2_and_oracle'],
    'related':['anydes_tools.tests.test_windows_sandbox_b2','anydes_tools.tests.test_testspec_contract',
        'anydes_tools.tests.test_pipeline_consistency_g4'],
    'code_output':['anydes_tools.tests.test_code_output'],
}
name=sys.argv[1]
out=bootstrap.HERE/'evidence'
out.mkdir(exist_ok=True)
started=out/(name+'_started.json')
if started.exists(): raise SystemExit('Gate already attempted')
started.write_text(json.dumps(gates[name]),encoding='utf-8')
with (out/(name+'.txt')).open('w',encoding='utf-8') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(gates[name]))
data=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),passed=result.wasSuccessful())
(out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data))
raise SystemExit(0 if result.wasSuccessful() else 1)
