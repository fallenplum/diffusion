"""Explicit R5-only anonymous publication, preserving the R0-R4 guards.

This is a packaging tool added after training. It does not authorize a new
scientific file or alter the frozen R5 experiment and receipt checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import bgb_en_publication as inherited
from src.bgb_en.r5_protocol import verify_protocol_lock

SOURCES=tuple('src/bgb_en/'+name for name in (
    'r5_readout.py','r5_protocol.py','r5_training.py'))
SCRIPTS=tuple('scripts/bgb_en_'+name+'.py' for name in (
    'r5_common','r5_run','r5_qualify','r5_report','r5_replay','r5_schedule',
    'r5_report_recovery','r5_resume_reporting','r5_publication'))
CONFIGS=('configs/bgb_en/r5_main_v1.json','configs/bgb_en/r5_mechanism_v1.json')
TESTS=tuple('tests/test_bgb_en_r5_'+name+'.py' for name in (
    'readout','protocol','training','report','report_recovery'))
REPORT_NAMES=(
    'PROTOCOL_LOCK.json','R5_0_QUALIFICATION.json','DERIVATIVE_CURVES.csv',
    'RUN_LEDGER.csv','RUN_LEDGER.json','R5_MAIN.csv','R5_MECHANISM.csv',
    'R5_SEED_FOLD.csv','R5_READOUT_DIAGNOSTICS.csv','R5_ATTENTION_DIAGNOSTICS.csv',
    'R5_GRADIENT_DIAGNOSTICS.csv','R5_SUBGROUPS.csv',
    'R5_LEARNING_CURVES.csv','R5_FROZEN_CONTROLS.csv',
    'R5_DIRECT_RETRAINED_PAIRS.csv','R5_DIRECT_MAIN_PAIRS.csv',
    'R5_RISK_READOUT_DIAGNOSTICS.md','R5_DECISION.md','decision.json',
    'SOURCE_AUDIT.json','R5_REPORT_AUDIT.json','R5_CHECKPOINT_REPLAY_AUDIT.json',
    'R5_REPORT_SCHEMA_RECOVERY_main.json','R5_REPORT_SCHEMA_RECOVERY_final.json',
    'REPRODUCE.md','EXECUTION_RECOVERY.md','R5_RESOURCE_SUMMARY.json',
    'R5_INDEPENDENT_SUMMARY.md')
REPORTS=tuple('reports/bgb_en_r5/'+name for name in REPORT_NAMES)
ALLOWLIST=frozenset(SOURCES+SCRIPTS+CONFIGS+TESTS+REPORTS)
MANIFEST='reports/bgb_en_r5/PUBLICATION_MANIFEST.json'


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--denylist',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    before={str(f.relative_to(root)):inherited.sha256(f)
            for f in (root/'reports/bgb_en').rglob('*') if f.is_file()}
    deny=inherited.validate_denylist(json.loads(args.denylist.read_text()))
    inherited.validate_r0_qualification(root)
    lock=json.loads((root/'reports/bgb_en_r5/PROTOCOL_LOCK.json').read_text())
    verify_protocol_lock(lock,root)
    decision=json.loads((root/'reports/bgb_en_r5/decision.json').read_text())
    audit=json.loads((root/'reports/bgb_en_r5/R5_REPORT_AUDIT.json').read_text())
    inherited.require(decision['decision']!='INCONCLUSIVE'
        and audit['status']=='COMPLETE_REQUIRED_R5_TRAINING_AND_REPLAY'
        and audit['replay_passed'], 'COMPLETE_REPLAYED_R5_REQUIRED_BEFORE_FINAL_PUBLICATION')
    identifier_pattern=inherited.identifier_pattern(deny)
    original_allowlist=inherited.ALLOWLIST
    inherited.ALLOWLIST=ALLOWLIST
    try:
        files=[inherited.scan_file(root,name,identifier_pattern) for name in sorted(ALLOWLIST)]
    finally:
        inherited.ALLOWLIST=original_allowlist
    content=''.join(f['file']+'\0'+f['sha256']+'\0'+str(f['bytes'])+'\n' for f in files)
    result=dict(schema='bgb_en_r5_explicit_publication_v1',
        status='PASS_ANONYMOUS_EXPLICIT_FILES_ONLY',scientific_run_state='COMPLETE',
        contract_sha256=lock['contract_sha256'],files=files,files_count=len(files),
        total_bytes=sum(f['bytes'] for f in files),
        ordered_file_manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
        actual_private_identifier_groups_scanned={k:len(v) for k,v in deny['groups'].items()},
        credentials_patient_arrays_coordinates_and_weights_included=False,
        source_test_and_report_allowlist_explicit=True,
        historical_R0_R4_namespace_modified=False,git_operations_performed=0)
    after={str(f.relative_to(root)):inherited.sha256(f)
           for f in (root/'reports/bgb_en').rglob('*') if f.is_file()}
    inherited.require(before==after,'R0_R4_REPORT_CHANGED_DURING_R5_PUBLICATION')
    target=root/MANIFEST
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    # Independently scan the manifest itself after it is fully materialized.
    inherited.ALLOWLIST=ALLOWLIST|{MANIFEST}
    try:inherited.scan_file(root,MANIFEST,identifier_pattern)
    finally:inherited.ALLOWLIST=original_allowlist
    print(json.dumps(dict(status=result['status'],files=len(files),
                         manifest_sha256=inherited.sha256(target))))


if __name__=='__main__':main()
