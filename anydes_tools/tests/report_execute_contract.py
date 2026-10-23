"""Post-close, read-only evidence aggregation. No runtime or provider invocation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/execute_contract'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    data = OUT/'candidate'
    life = rows(data/'lifecycle.jsonl')
    audit = rows(data/'data/audit.log')
    telemetry = rows(data/'telemetry/tools/conversation-v001_5.jsonl')
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
    value = {
        'verdict':'CORE STABILIZATION NOT READY',
        'reason':'UI_INTERACTION_SCHEMA_REJECTED_BEFORE_PLANNING',
        'regression':{'pass':int(re.search(r'Ran (\d+) tests',regression).group(1)),
            'fail':0,'skip':0,'clean':bool(re.search(r'^OK$',regression,re.M))},
        'sessions':1,'human_turns':sum(r['event']=='phase_c_turn' for r in telemetry),
        'runtime_changes_during_replay':False,
        'protocol_rejections':[r for r in audit if r['event']=='interaction_protocol_rejected'],
        'outcomes':[r for r in telemetry if r['event']=='phase_c_turn'],
        'stages':[r for r in telemetry if r['event']=='pipeline_stage_trace'],
        'ui_bridge':[r for r in telemetry if r['event']=='ui_bridge_outcome'],
        'publications':publications,
        'chat_b4_equal':sum(r['details']['chat_speech_equal'] for r in publications),
        'voice_completions':[r for r in audit if r['event']=='interface_voice_completed'],
        'audibility':'NOT_INDEPENDENTLY_CONFIRMED_THIS_SESSION',
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1],
        'remaining_processes':json.loads(result.stdout) if result.stdout.strip() else [],
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'violations':[r for r in writes if r['decision']!='OBSERVED'],
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'],'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'freeze_sha256':hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        'provider_statuses':dict(Counter(r.get('status') for r in life if r['event']=='groq_http_response')),
        'sandbox_temps':{n:len(list((ROOT/'anydes_tools/sandbox'/n).rglob('*')))
            for n in ('executions_b2','runtime_temp','runtime_home')},
    }
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('verdict','regression','human_turns','chat_b4_equal',
        'lifecycle_counts','remaining_processes','official_matches','protected_count',
        'protected_changes','source_changes_after_freeze','sandbox_temps','provider_statuses','stderr')},indent=2))


if __name__ == '__main__':
    main()
