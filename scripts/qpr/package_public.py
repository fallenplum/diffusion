"""Explicit QPR text-only allowlist using the unchanged project privacy rules."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,re
from pathlib import Path,PurePosixPath

SCANNER='4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98'
DENYLIST='d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867'


def allowed(name):
    p=PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name and p.suffix in ('.py','.json','.csv','.md') and (
        name.startswith(('src/qpr/','scripts/qpr/','tests/qpr/','configs/qpr/','reports/qpr/')))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(root,manifest,scanner,denylist):
    if sha(scanner)!=SCANNER or sha(denylist)!=DENYLIST:raise ValueError('PRIVACY_SOURCE_OR_DENYLIST_CHANGED')
    spec=importlib.util.spec_from_file_location('qpr_unchanged_original_scanner',scanner)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    raw=json.loads(Path(denylist).read_text(encoding='utf-8'));original.validate_denylist(raw)
    pattern=original.identifier_pattern(raw)
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'))['files']
    if len(rows)!=len({r['file'] for r in rows}) or not all(allowed(r['file']) for r in rows):raise ValueError('UNSAFE_SCOPE')
    before=original.ALLOWLIST;original.ALLOWLIST=frozenset(r['file'] for r in rows)
    try:actual=[original.scan_file(Path(root),r['file'],pattern) for r in rows]
    finally:original.ALLOWLIST=before
    if actual!=rows:raise ValueError('PUBLIC_BYTES_CHANGED_DURING_SCAN')
    return {'status':'PASS_ANONYMOUS_QPR_TEXT_ONLY','files':len(rows),'bytes':sum(r['bytes'] for r in rows),
            'manifest_sha256':sha(manifest),'scanner_sha256':SCANNER,'denylist_sha256':DENYLIST,
            'patient_identifiers_exported':False,'protected_datasets_weights_predictions_credentials_exported':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--scanner',required=True);p.add_argument('--denylist',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();result=scan(a.root,a.manifest,a.scanner,a.denylist)
    Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
