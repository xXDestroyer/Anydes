"""Prepared wording/review, real metadata publication and final math UI hook."""
import bootstrap
import json
import unittest
bootstrap.install()
import anydes_tools.tests as tests
tests.__path__.insert(0,str(bootstrap.HERE/'tests'))
if __name__=='__main__':
    names=['anydes_tools.tests.test_technical_literals']
    with (bootstrap.HERE/'evidence/literal_harness.txt').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(verbosity=2,stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    evidence={'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'ui_opened':False,'tool_calls':0}
    (bootstrap.HERE/'evidence/literal_harness.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence))
    raise SystemExit(0 if result.wasSuccessful() else 1)
