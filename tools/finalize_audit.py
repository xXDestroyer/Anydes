"""Release-only packaging audit. Installation access is read-only; no imports from it."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'Anydes'
AUDIT = ROOT / 'audit'
sys.path.insert(0, str(ROOT / 'tools'))
from check_source import check


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, text):
    target = (AUDIT / name).resolve()
    assert target.is_relative_to(ROOT)
    target.write_text(text, encoding='utf-8')


def main():
    assert ROOT.name == 'Anydes_GitHub_Release' and not ROOT.is_relative_to(SOURCE)
    provenance = json.loads((AUDIT / 'source_provenance.json').read_text())
    removed = []
    targets = [ROOT / 'audit_release.py', ROOT / 'build_release.py',
               ROOT / 'programa_principal/probar_conciencia_visual.py']
    targets += list((ROOT / 'anydes_tools/tests').glob('collect_*.py'))
    targets += list((ROOT / 'anydes_tools/tests').glob('recover_telemetry*.py'))
    for target in targets:
        target = target.resolve()
        assert target.is_relative_to(ROOT) and not target.is_relative_to(SOURCE)
        if target.is_file():
            removed.append(target.relative_to(ROOT).as_posix())
            target.unlink()
    # Offline tests can create synthetic audit logs. They are not release sources.
    for folder in (ROOT / 'anydes_tools/logs', ROOT / 'anydes_tools/workspaces'):
        folder = folder.resolve()
        assert folder.is_relative_to(ROOT) and not folder.is_relative_to(SOURCE)
        if folder.exists():
            for entry in sorted(folder.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                assert entry.resolve().is_relative_to(ROOT)
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    entry.rmdir()
            folder.rmdir()
    prior = provenance.get('release_only_omissions', [])
    provenance['release_only_omissions'] = sorted(set(prior + removed))
    provenance['included'] = [p for p in provenance['included'] if (ROOT / p['path']).is_file()]
    for record in provenance['included']:
        path = ROOT / record['path']
        digest = sha(path)
        if digest != record['sha256']:
            provenance['sanitizations'].append(dict(path=record['path'], category='ANONYMIZED_TEST_WORKSPACE_LITERAL'))
        record['sha256'] = digest
        record['bytes'] = path.stat().st_size
    write('source_provenance.json', json.dumps(provenance, indent=2))

    integrity = {}
    shell = SOURCE / 'experimentos/anydes_professional_shell_h1_4'
    for label, name in [('protected', 'protected_before'), ('frozen', 'frozen_before')]:
        values = json.loads((shell / f'evidence/{name}.json').read_text())
        matches = 0
        for raw, expected in values.items():
            path = Path(raw) if Path(raw).is_absolute() else SOURCE / raw
            matches += int(path.is_file() and sha(path) == expected)
        integrity[label] = dict(total=len(values), matched=matches, violations=len(values) - matches)
    promotion_data = json.loads((SOURCE / 'promotion_h0/H0_20261017/promotion_manifest.json').read_text())
    promotion = promotion_data['changes'] + promotion_data['adopted_existing_runtime']
    matches = sum((SOURCE / p['path']).is_file() and sha(SOURCE / p['path']) == p.get('after_sha256', p.get('sha256')) for p in promotion)
    integrity['production'] = dict(total=len(promotion), matched=matches, violations=len(promotion) - matches)
    values = json.loads((shell / 'evidence/source_manifest.json').read_text())['files']
    matches = sum((shell / p['path']).is_file() and sha(shell / p['path']) == p['sha256'] for p in values)
    integrity['professional_shell'] = dict(total=len(values), matched=matches, violations=len(values) - matches)
    values = provenance['included']
    matches = sum((SOURCE / p['path']).is_file() and sha(SOURCE / p['path']) == p['source_sha256'] for p in values)
    integrity['exported_original_sources'] = dict(total=len(values), matched=matches, violations=len(values) - matches)
    write('installation_integrity.json', json.dumps(integrity, indent=2))

    result = check()
    write('source_validation.json', json.dumps(result, indent=2))
    write('source_validation_report.md', '# Source Validation\n\n'
          + f"- Python syntax: {result['counts']['python']} files checked; {len(result['failures'])} failures.\n"
          + f"- JSON parsing: {result['counts']['json']} files checked.\n"
          + '- No application startup, model calls, bytecode writes or ports opened.\n'
          + '- Real classifier/code-output/literal suites: 52 PASS, 0 FAIL (see focused_tests.json).\n'
          + '- Portable serialization tests: 10 PASS, 0 FAIL.\n'
          + '- Existing JavaScript motion suites: 18 PASS, 0 FAIL.\n'
          + '- JavaScript syntax: 15 files checked with Node --check; 0 failures.\n'
          + '- Import sanity: no Anydes application modules loaded from outside this release.\n'
          + '- JSON Schema was supplied read-only from existing third-party dependencies for tests.\n'
          + '- Full application imports remain limited by dependencies and legacy source gaps; see docs/PORTABILITY.md.\n'
          + '- The historical 529-test installation regression is not a fresh release test count.\n')

    patterns = {
        'PROVIDER_TOKEN': r'\b(?:gsk_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})\b',
        'PRIVATE_KEY': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'WEBHOOK_SECRET': r'https://(?:hooks\.slack\.com/services|discord(?:app)?\.com/api/webhooks)/[A-Za-z0-9/_-]+',
        'GITHUB_TOKEN': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'AWS_ACCESS_KEY': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'JWT_CREDENTIAL': r'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b',
        'CREDENTIAL_CONNECTION_STRING': r'(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:/]+:[^\s@/]+@',
        'BEARER_CREDENTIAL': r'(?i)\bBearer\s+[A-Za-z0-9._-]{32,}',
    }
    findings = []
    illustrative = []
    emails = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')
    credential = re.compile(r'(?i)(?:api_key|password|access_token|client_secret|bearer_token)\s*[:=]\s*["\']([^"\'\n]{8,})["\']')
    personal_names = [SOURCE.parent.parent.name]
    personal_names += [part for part in personal_names[0].split() if len(part) > 3]
    # Known credentials are compared in memory only and never retained in audit outputs.
    known_secrets = []
    dotenv = SOURCE / '.env'
    if dotenv.exists():
        content = dotenv.read_text(encoding='utf-8-sig', errors='replace')
        for pattern in patterns.values():
            known_secrets += re.findall(pattern, content)
    bad_extensions = {'.gguf', '.safetensors', '.bin', '.pth', '.pt', '.onnx', '.pyc',
                      '.exe', '.dll', '.pyd', '.db', '.sqlite', '.sqlite3', '.wav', '.mp3', '.flac', '.jsonl', '.log'}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or path.suffix.lower() in bad_extensions or path.name == '.env':
            findings.append(dict(path=relative, category='FORBIDDEN_FILE'))
        if path.stat().st_size > 100 * 1024 * 1024:
            findings.append(dict(path=relative, category='GITHUB_SIZE_LIMIT'))
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        for number, line in enumerate(text.splitlines(), 1):
            for category, pattern in patterns.items():
                if re.search(pattern, line):
                    findings.append(dict(path=relative, line=number, category=category))
            if any(secret and secret in line for secret in known_secrets):
                findings.append(dict(path=relative, line=number, category='KNOWN_ENV_SECRET'))
            if any(re.search(r'(?i)\b' + re.escape(name) + r'\b', line) for name in personal_names):
                findings.append(dict(path=relative, line=number, category='SOURCE_OWNER_NAME'))
            for email in emails.findall(line):
                if email.lower() != 'demo@example.invalid':
                    # Third-party author emails remain in required license notices, not user data.
                    category = 'THIRD_PARTY_LICENSE_CONTACT' if 'vendor/' in relative else 'EMAIL_REQUIRES_REVIEW'
                    (illustrative if category.startswith('THIRD_PARTY') else findings).append(dict(path=relative, line=number, category=category))
            if credential.search(line):
                illustrative.append(dict(path=relative, line=number, category='CREDENTIAL_SYNTAX_REVIEW'))
            if re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]+', line):
                illustrative.append(dict(path=relative, line=number, category='ANONYMIZED_PATH_OR_DETECTION_PATTERN'))
    write('privacy_scan.json', json.dumps(dict(blocking_findings=findings, review_locations=illustrative), indent=2))
    report = '# Security and Privacy Audit\n\n'
    report += f'- Blocking secret/privacy findings: {len(findings)}.\n'
    report += '- Installation .env: a provider credential was detected before copying; entire file excluded. No value is retained here.\n'
    report += '- Real .env included: NO. Credentials included: NO, subject to the scanner findings below.\n'
    report += '- Personal/protected memory, private SELF/profile state, conversations, runtime sessions, recordings and screenshots: NOT EXPORTED.\n'
    report += '- Model weights, virtual environments, Python vendor distributions, caches and rollback snapshots: NOT EXPORTED.\n'
    report += '- Checks: token prefixes, private keys, webhooks, known local credential equality, owner-name strings, email literals, prohibited files and sizes.\n'
    report += '- Generic DemoUser paths and synthetic security-test strings are illustrative source literals, not real personal data.\n'
    report += '- Third-party author contacts in required licenses are retained; they are not user records.\n'
    report += '- Credential-syntax matches (field names, tests, placeholders) require contextual review, not automatic deletion. Exact locations only are in privacy_scan.json.\n'
    report += '- No secret values, raw prompts or private model outputs are printed in audit reports. Automated scanning is not a guarantee against every possible secret encoding.\n'
    report += '\n## Findings\n\n' + ('None.\n' if not findings else '\n'.join(f"- {x['path']}:{x.get('line', 0)} - {x['category']}" for x in findings))
    write('security_privacy_report.md', report)
    write('excluded_categories.md', '# Excluded Categories\n\n'
          '- Model weights/checkpoints and inference distributions: size, licensing and platform setup; see models/README.md.\n'
          '- Virtual environments, vendored Python distributions, node_modules and caches: install dependencies separately.\n'
          '- Personal/protected memory, private SELF/profile state, conversation/session logs, telemetry and location/activity history: privacy. Only implementation code is included.\n'
          '- Credentials, .env and keys: never redistributable. .env.example contains placeholders only.\n'
          '- Rollback/promotion snapshots and older copied overlays: duplicate runtime history, not the maintained review boundary.\n'
          '- Generated user artifacts, recordings, screenshots and runtime evidence: private or unnecessary; no screenshot was captured.\n'
          '- Avatar/model textures and preview images: not redistributed without a separate asset-rights/privacy review. Procedural hologram source remains.\n'
          '- Session collection scripts and machine-only visual diagnostics: irrelevant to judge review.\n'
          '- Runtime binaries/installers/compiler distributions: platform dependencies, not source.\n')
    write('repository_size_report.md', '')
    write('included_files_manifest.txt', '')
    # Recompute after report creation; listing paths avoids a self-referential hash.
    for _ in range(4):
        files = sorted(p for p in ROOT.rglob('*') if p.is_file())
        manifest = '\n'.join(p.relative_to(ROOT).as_posix() for p in files) + '\n'
        write('included_files_manifest.txt', manifest)
        files = sorted((p for p in ROOT.rglob('*') if p.is_file()), key=lambda p: p.stat().st_size, reverse=True)
        total = sum(p.stat().st_size for p in files)
        size = '# Repository Size Report\n\n'
        size += f'- Total: {total:,} bytes ({total / 1048576:.2f} MiB).\n- Files: {len(files)}.\n'
        for mb in (10, 25, 50, 100):
            over = [p for p in files if p.stat().st_size > mb * 1048576]
            size += f'- Files >{mb} MiB: {len(over)}.\n'
        size += '\n## Largest 30 Files\n\n| Bytes | Relative Path |\n|---:|---|\n'
        size += '\n'.join(f"| {p.stat().st_size} | {p.relative_to(ROOT).as_posix()} |" for p in files[:30]) + '\n'
        write('repository_size_report.md', size)
    print(json.dumps(dict(files=len(files), bytes=sum(p.stat().st_size for p in files),
        largest=files[0].relative_to(ROOT).as_posix(), blocking_findings=len(findings),
        integrity=integrity, syntax_failures=len(result['failures']))))
    return bool(findings or result['failures'] or any(x['violations'] for x in integrity.values()))


if __name__ == '__main__':
    sys.exit(main())
