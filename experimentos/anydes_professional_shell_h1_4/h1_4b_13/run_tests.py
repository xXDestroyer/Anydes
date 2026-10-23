"""Run the accepted focal suite against the isolated candidate overlay."""
import re
from pathlib import Path
import sys
import unittest
import json
import bootstrap

bootstrap.install()
import anydes_tools.tests as test_package
test_package.__path__.insert(0,str(bootstrap.HERE/'tests'))

if __name__=='__main__':
    if '--focal' in sys.argv:
        names=['anydes_tools.tests.test_classifier_request_contract','anydes_tools.tests.test_classifier_branches','anydes_tools.tests.test_goal_resolvability','anydes_tools.tests.test_clarification_completeness','anydes_tools.tests.test_continue_task_contract','anydes_tools.tests.test_manual_flow','anydes_tools.tests.test_outcome_voice']
    else:
        previous=bootstrap.ROOT/'anydes_tools/logs/pre_request_phase2b/regression.txt'
        names=sorted(set(re.findall(r'\((anydes_tools\.tests\.test_[A-Za-z0-9_]+)\.',previous.read_text(encoding='utf-8-sig'))))
        names.append('anydes_tools.tests.test_manual_flow')
        names.append('anydes_tools.tests.test_continue_task_contract')
        names.append('anydes_tools.tests.test_clarification_completeness')
        names.append('anydes_tools.tests.test_goal_resolvability')
        names.append('anydes_tools.tests.test_classifier_request_contract')
    names.append('anydes_tools.tests.test_classifier_canonical_output')
    names.append('anydes_tools.tests.test_continuation_public')
    names.append('anydes_tools.tests.test_clarification_anchor')
    names.append('anydes_tools.tests.test_verification_execution')
    names.append('anydes_tools.tests.test_metadata_publication')
    names.append('anydes_tools.tests.test_technical_literals')
    names.append('anydes_tools.tests.test_code_output')
    names.append('anydes_tools.tests.test_usage_race')
    names.append('anydes_tools.tests.test_classifier_wire')
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    output=bootstrap.HERE/'evidence'
    output.mkdir(exist_ok=True)
    label='focal' if '--focal' in sys.argv else 'regression'
    with (output/(label+'.txt')).open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(verbosity=2,stream=log).run(suite)
    evidence={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'modules':names,
        'updated_contract_tests':['test_outcome_voice: no empty abstention; typed PublicationFailure',
            'test_publication_phase_f: injected runtime error now has prepared natural writer/review'],
        'production_modified':False}
    (output/('focal.json' if '--focal' in sys.argv else 'regression.json')).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='modules'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
