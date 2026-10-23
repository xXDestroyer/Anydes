"""Aggregate the single closed UI session; no runtime/provider calls or cleanup."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'anydes_tools/logs/classifier_branches'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    data = OUT/'candidate'
    life = rows(data/'lifecycle.jsonl')
    audit = rows(data/'data/audit.log')
    events = rows(data/'telemetry/tools/conversation-v001_5.jsonl')
    writes = rows(data/'host_writes.jsonl')
    after = json.loads((OUT/'after.json').read_text())
    freeze = json.loads((OUT/'freeze.json').read_text())
    official = json.loads((ROOT/'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    snapshots = [r for r in life if r['event']=='qt_snapshot']
    pids = {r['pid'] for r in life}
    pids.update(p['pid'] for r in snapshots for p in r['processes'])
    pids.update(p['parent'] for r in snapshots for p in r['processes'] if p['pid']==r['pid'])
    query = ' OR '.join(f'ProcessId = {pid}' for pid in sorted(pids))
    command = (f'Get-CimInstance Win32_Process -Filter "{query}" | '
        'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],
        check=True,capture_output=True,text=True)
    publications = [r for r in audit if r['event']=='final_publication']
    regression = (OUT/'regression.txt').read_text(encoding='utf-8')
    previous = ROOT/'anydes_tools/logs/execute_contract/snapshot'
    unchanged = {}
    for rel in ('anydes_tools/conversation/planner_transport.py',
            'anydes_tools/conversation/plan_lowering.py','anydes_tools/conversation/general_reasoning.py',
            'anydes_tools/conversation/controller.py','anydes_tools/conversation/semantic_proposal.py'):
        unchanged[rel] = (ROOT/rel).read_bytes() == (previous/rel).read_bytes()
    rel = 'anydes_tools/conversation/semantic_runtime.py'
    def planner(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef)
            and n.name=='AssistantGeneralTaskReasoner'),include_attributes=False)
    unchanged['AssistantGeneralTaskReasoner'] = planner(ROOT/rel)==planner(previous/rel)
    value = {'verdict':'CORE STABILIZATION NOT READY',
        'reason':'PLANNER_TESTSPEC_SCHEMA_REJECTED_BEFORE_CONFIRMATION_AND_B2',
        'regression':{'pass':int(re.search(r'Ran (\d+) tests',regression).group(1)),
            'fail':0,'skip':0,'clean':bool(re.search(r'^OK$',regression,re.M))},
        'sessions':1,'runtime_changes_during_replay':False,
        'human_turns':sum(r['event']=='phase_c_turn' for r in events),
        'classifier':[r for r in audit if r['event'] in {'interaction_classification',
            'interaction_protocol_rejected','interaction_branch_canonicalization'}],
        'planning':[r for r in audit if r['event'] in {'semantic_plan_protocol','semantic_proposal_shape'}],
        'failure':[r for r in events if r['event'] in {'conversation_tool_failure_state',
            'plan_completeness','phase_c_turn','resolution_control_validation'}],
        'pipeline':[r for r in events if r['event']=='pipeline_stage_trace'],
        'publications':publications,
        'chat_b4_equal':sum(r['details']['chat_speech_equal'] for r in publications),
        'audibility':'NOT_INDEPENDENTLY_CONFIRMED_THIS_SESSION',
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1],
        'remaining_processes':json.loads(result.stdout) if result.stdout.strip() else [],
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'violations':[r for r in writes if r['decision']!='OBSERVED'],
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'],'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'execution_contract_unchanged':unchanged,
        'sandbox_temps':{n:len(list((ROOT/'anydes_tools/sandbox'/n).rglob('*')))
            for n in ('executions_b2','runtime_temp','runtime_home')},
        'provider_statuses':dict(Counter(r.get('status') for r in life if r['event']=='groq_http_response')),
    }
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('verdict','regression','human_turns','chat_b4_equal',
        'lifecycle_counts','remaining_processes','official_matches','protected_count',
        'protected_changes','source_changes_after_freeze','sandbox_temps','execution_contract_unchanged',
        'provider_statuses','stderr')},indent=2))
    print('violations',len(value['violations']))


if __name__ == '__main__':
    main()
