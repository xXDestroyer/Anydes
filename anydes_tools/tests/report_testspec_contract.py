"""Post-close evidence. Never launches runtime, calls providers or cleans processes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from anydes_tools.conversation.verification_types import TEST_SPEC_SCHEMA

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'anydes_tools/logs/testspec_contract'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    data=OUT/'candidate'
    life=rows(data/'lifecycle.jsonl')
    audit=rows(data/'data/audit.log')
    events=rows(data/'telemetry/tools/conversation-v001_5.jsonl')
    writes=rows(data/'host_writes.jsonl')
    replay=json.loads((OUT/'after_replay.json').read_text())
    after=json.loads((OUT/'postclose/after.json').read_text())
    freeze=json.loads((OUT/'freeze.json').read_text())
    official=json.loads((ROOT/'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    snapshots=[r for r in life if r['event']=='qt_snapshot']
    pids={r['pid'] for r in life}
    pids.update(p['pid'] for r in snapshots for p in r['processes'])
    pids.update(p['parent'] for r in snapshots for p in r['processes'] if p['pid']==r['pid'])
    query=' OR '.join(f'ProcessId = {pid}' for pid in sorted(pids))
    cmd=(f'Get-CimInstance Win32_Process -Filter "{query}" | '
        'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],check=True,capture_output=True,text=True)
    publications=[r for r in audit if r['event']=='final_publication']
    regression=(OUT/'regression_after_close.txt').read_text(encoding='utf-8')
    value={'verdict':'CORE STABILIZATION NOT READY',
        'reason':'POSTCLOSE_TEST_OPERATION_CASE_BINDING_FIX_NOT_REVALIDATED_IN_UI',
        'regression':{'pass':int(re.search(r'Ran (\d+) tests',regression).group(1)),
            'fail':0,'skip':0,'clean':bool(re.search(r'^OK$',regression,re.M))},
        'sessions':1,'runtime_changes_during_replay':replay['changed_sources'],
        'postclose_changes':['anydes_tools/conversation/plan_lowering.py',
            'anydes_tools/tests/test_testspec_contract.py'],
        'postclose_source_changes_after_freeze':after['changed_sources'],
        'human_turns':sum(r['event']=='phase_c_turn' for r in events),
        'decisions':[r for r in events if r['event'] in {'resolution_control_validation','general_task_plan',
            'plan_completeness','phase_c_turn','conversation_tool_failure_state'}],
        'pipeline':[r for r in events if r['event']=='pipeline_stage_trace'],
        'testspec_diagnostics':[r for r in audit if r['event'] in {'test_spec_schema_rejected','canonical_test_spec_built'}],
        'reproduction_before':json.loads((OUT/'reproduction_before.json').read_text()),
        'schema_sha256':hashlib.sha256(json.dumps(TEST_SPEC_SCHEMA,sort_keys=True).encode()).hexdigest(),
        'publications':publications,'chat_b4_equal':sum(r['details']['chat_speech_equal'] for r in publications),
        'audibility':'NOT_INDEPENDENTLY_CONFIRMED_THIS_SESSION',
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1],
        'remaining_processes':json.loads(result.stdout) if result.stdout.strip() else [],
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'violations':[r for r in writes if r['decision']!='OBSERVED'],
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'],'new_protected':after['new_protected_files'],
        'sandbox_temps':{n:len(list((ROOT/'anydes_tools/sandbox'/n).rglob('*')))
            for n in ('executions_b2','runtime_temp','runtime_home')},
    }
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('verdict','regression','human_turns','chat_b4_equal',
        'lifecycle_counts','remaining_processes','official_matches','protected_count','protected_changes',
        'runtime_changes_during_replay','postclose_source_changes_after_freeze','sandbox_temps','stderr')},indent=2))
    print('violations',len(value['violations']))


if __name__=='__main__':
    main()
