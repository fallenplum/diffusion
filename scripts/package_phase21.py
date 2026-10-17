"""Package only new, complete, anonymous Phase21 evidence; never train or push.

The default dry run checks publication privacy and every inherited tracked byte.
Only --release requires all real completion gates and writes a public manifest
and a private mirror archive. Scientific negatives remain publishable.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
P=Path('/root/autodl-tmp/phase21_relative_private')
R=ROOT/'reports/phase21_relative_attention'
PARENT='5b2bb3645b125ca09e2e6825706978fc691981f4'
BRANCH='research/phase21-relative-attention-audit'
PARENT_REPORT='reports/phase20_selective_capacity'
CONFIG='configs/phase21_relative_attention.json'
SEEDS=(31,47,83)
FOLDS=(1,2,3,4,5)
ROLES=('inner','outer')
ARMS=('P16','S16','DIRECT')
MAX_BYTES=8*1024**2
EXTENSIONS={'.py','.json','.md','.csv'}
SENSITIVE_CONTENT=re.compile(
    rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|'
    rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|'
    rb'\bENS[GT]\d{11}(?:\.\d+)?\b|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}')
DENIED_FIELDS={
    'patient_id','patient_ids','patient_index','patient_indices','sample_id','sample_ids',
    'slide_id','slide_ids','barcode','barcodes','donor_indices','donor_ids',
    'risk','risks','raw_risk','raw_risks','risk_raw','patient_risk','patient_risks','anchor_risk',
    'prediction','predictions','raw_predictions','raw_prediction','target_value',
    'gene_ids','gene_symbols','selected_genes','selected_gene_ids','selected_gene_symbols',
    'coordinates','coords','raw_maps','raw_tokens','tokens','token_values','centroids',
    'prototype_features','pca_components','pca_state','model_state','state_dict',
    'dss_time','dss_event','pfi_time','pfi_event','survival_time','survival_event',
    'password','access_token','api_key','private_key','credentials'}
PRIVATE_NAME=re.compile(
    r'(?:^|_)(?:tmp|private|raw_predictions|patient_predictions|oof_predictions|'
    r'selected_genes|model_state|donor_indices)(?:_|\.|$)',re.I)


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.package.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def safe_path(relative):
    p=PurePosixPath(relative)
    require(not p.is_absolute() and p.parts and all(q not in ('','..','.') for q in p.parts)
            and '\\' not in relative and ':' not in relative,'unsafe relative path')
    path=ROOT.joinpath(*p.parts)
    require(path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(),'escaping/symlink path')
    return path


def allowed_path(relative):
    p=PurePosixPath(relative)
    if p.parts[:2]==('src','phase21_relative'):return len(p.parts)==3 and p.suffix=='.py'
    if p.parts[:1]==('configs',):return relative==CONFIG
    if p.parts[:1]==('tests',):return len(p.parts)==2 and p.name.startswith('test_phase21_') and p.suffix=='.py'
    if p.parts[:1]==('scripts',):
        return len(p.parts)==2 and re.fullmatch(r'[a-z][a-z0-9_]*phase21[a-z0-9_]*\.py',p.name) is not None
    return p.is_relative_to(PurePosixPath(R.relative_to(ROOT).as_posix())) and p.suffix in EXTENSIONS


def _denied_field(name):
    # Flattened anonymous diagnostic moments remain aggregate scalars.
    name=str(name).strip().lower()
    parts=name.split('.')
    return name in DENIED_FIELDS or any(part in DENIED_FIELDS for part in parts)


def audit_json(value,location='root',relative=''):
    if isinstance(value,dict):
        for key,item in value.items():
            require(not _denied_field(key),'protected JSON field: '+location+'.'+str(key))
            audit_json(item,location+'.'+str(key),relative)
    elif isinstance(value,list):
        if value and any(isinstance(q,(int,float)) and not isinstance(q,bool) for q in value):
            # These are fixed public configuration axes, never patient vectors.
            axis=location.removeprefix('root.configuration.').removeprefix('root.')
            allowed_axes=relative==CONFIG or relative==R.relative_to(ROOT).as_posix()+'/PROTOCOL_LOCK.json'
            expected={'seeds':list(SEEDS),'folds':list(FOLDS)}
            require(allowed_axes and axis in expected and value==expected[axis],
                    'raw numeric list in anonymous public JSON: '+location)
        for item in value:audit_json(item,location+'[]',relative)
    elif isinstance(value,float):require(math.isfinite(value),'nonfinite public JSON scalar: '+location)
    else:require(value is None or isinstance(value,(str,int,bool)),'unsupported JSON value: '+location)


def audit_public_file(path):
    path=Path(path);relative=path.relative_to(ROOT).as_posix()
    require(allowed_path(relative),'outside explicit new Phase21 allowlist: '+relative)
    safe_path(relative)
    require(path.is_file() and path.suffix in EXTENSIONS and path.stat().st_size<=MAX_BYTES,
            'unsupported/oversized public artifact: '+relative)
    require(not PRIVATE_NAME.search(path.stem),'private/temporary artifact filename: '+relative)
    raw=path.read_bytes();require(not SENSITIVE_CONTENT.search(raw),'sensitive content detected: '+relative)
    text=raw.decode('utf-8-sig')
    if path.suffix=='.json':audit_json(json.loads(text),relative=relative)
    elif path.suffix=='.csv':
        rows=csv.reader(io.StringIO(text));columns=next(rows,[])
        require(columns and len(set(columns))==len(columns),'missing/duplicate CSV columns: '+relative)
        require(not any(_denied_field(q) for q in columns),'patient-level CSV field: '+relative)
        for row in rows:
            require(len(row)==len(columns),'malformed aggregate CSV row: '+relative)
            require(not any(re.match(r'^\s*[\[{]',q) for q in row),'serialized raw array/object in public CSV: '+relative)
            require(not any(q.strip().lower() in ('nan','inf','+inf','-inf','infinity') for q in row),
                    'nonfinite aggregate CSV value: '+relative)
    return {'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def publication_paths(include_manifest=False):
    files=list((ROOT/'src/phase21_relative').glob('*.py'))+[ROOT/CONFIG]
    files+=list((ROOT/'tests').glob('test_phase21_*.py'))
    files+=[p for p in (ROOT/'scripts').glob('*phase21*.py') if allowed_path(p.relative_to(ROOT).as_posix())]
    files+=[p for p in R.rglob('*') if p.is_file() and (include_manifest or p.name!='ARTIFACT_MANIFEST.json')]
    require(files and all(p.is_file() for p in files),'new source/configuration missing')
    return sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())


def audit_parent_immutability():
    require(git('branch','--show-current').decode().strip()==BRANCH,'wrong publication branch')
    subprocess.check_call(['git','merge-base','--is-ancestor',PARENT,'HEAD'],cwd=ROOT)
    snapshot=read(P/'PARENT_TRACKED_HASHES.json')
    require(isinstance(snapshot,dict) and len(snapshot)==2362,'expected original 2362-file parent inventory')
    tree={}
    for record in git('ls-tree','-r','-z',PARENT).split(b'\0'):
        if not record:continue
        header,encoded=record.split(b'\t',1);mode,kind,oid=header.decode().split()
        require(kind=='blob' and mode in ('100644','100755'),'unsupported inherited tracked object')
        tree[encoded.decode('utf-8')]=oid
    require(set(tree)==set(snapshot),'private parent snapshot does not cover exact parent Git tree')
    for relative,expected in snapshot.items():
        path=safe_path(relative);require(path.is_file(),'removed inherited tracked file: '+relative)
        raw=path.read_bytes();blob=b'blob '+str(len(raw)).encode()+b'\0'+raw
        require(hashlib.sha256(raw).hexdigest()==expected and hashlib.sha1(blob).hexdigest()==tree[relative],
                'inherited tracked bytes changed: '+relative)
    manifest=read(ROOT/PARENT_REPORT/'ARTIFACT_MANIFEST.json')
    require(manifest==json.loads(git('show',PARENT+':'+PARENT_REPORT+'/ARTIFACT_MANIFEST.json'))
            and manifest['status']=='PASS','parent publication manifest changed')
    for row in manifest['public_files']:
        require(snapshot[row['path']]==row['sha256'],'parent public manifest SHA differs')
    scope=read(ROOT/PARENT_REPORT/'PUBLICATION_SCOPE_AUDIT.json')
    legacy=scope['unchanged_inherited_violations']
    require(len(legacy)==5 and all(PurePosixPath(q['path']).suffix.lower()=='.zip'
            and snapshot[q['path']]==q['sha256'] for q in legacy),'five old ZIPs must remain exact original bytes')
    for relative in [q.decode('utf-8') for q in git('ls-files','-z').split(b'\0') if q]:
        require(relative in snapshot or allowed_path(relative),'new tracked file outside publication allowlist: '+relative)
    return {'status':'PASS','parent_commit':PARENT,'parent_is_HEAD_ancestor':True,
        'inherited_tracked_files_verified':len(snapshot),'all_inherited_tracked_bytes_unchanged':True,
        'private_parent_inventory_sha256':sha(P/'PARENT_TRACKED_HASHES.json'),
        'parent_public_manifest_sha256':sha(ROOT/PARENT_REPORT/'ARTIFACT_MANIFEST.json'),
        'parent_public_manifest_files_verified':len(manifest['public_files']),
        'verification':'every inherited file SHA256 and Git blob identity; exact parent path set'},legacy


def execution_summary(private_root=None,identity=None):
    """Independently count actual ledgers; closed interrupted steps are bounds."""
    if private_root is None or identity is None:
        from src.phase21_relative import data
        private_root=data.P if private_root is None else Path(private_root)
        identity=data.source() if identity is None else identity
    private_root=Path(private_root)
    formal=[];ledger_hashes={};completed_steps=0;completed_process_seconds=0.
    expected={(role,seed,fold) for role in ROLES for seed in SEEDS for fold in FOLDS}
    for role,seed,fold in sorted(expected):
        name=f'DIRECT_seed{seed}_fold{fold}'
        receipt=read(private_root/role/(name+'.json'))
        require(receipt['status']=='PASS' and receipt['source']==identity
                and (receipt['role'],receipt['seed'],receipt['fold'],receipt['arm'])==(role,seed,fold,'DIRECT'),
                'incomplete/foreign completed DIRECT receipt')
        path=private_root/'attempts'/role/(name+'.json');attempts=read(path)['attempts']
        require(attempts and all(q['source']==identity and q['status']!='RUNNING' and isinstance(q['steps'],int)
                and q['steps']>=0 for q in attempts),'missing, running or foreign formal attempt ledger')
        success=[q for q in attempts if q['status']=='SUCCESS']
        epochs=int(receipt['summary']['actual_epochs'])
        require(len(success)==1 and success[0]['steps']==epochs,'successful optimizer steps differ from completed schedule')
        formal+=attempts;completed_steps+=epochs;ledger_hashes[role+'/'+name+'.json']=sha(path)
        seconds=receipt['summary'].get('seconds')
        require(isinstance(seconds,(int,float)) and math.isfinite(seconds) and seconds>=0,
                'completed fit runtime absent or invalid')
        completed_process_seconds+=seconds
    smoke_path=private_root/'SMOKE_LEDGER.json';smoke=read(smoke_path)['fits']
    require(smoke and all(q['source']==identity and q['status']!='RUNNING' and isinstance(q['steps'],int)
            and q['steps']>=0 for q in smoke),'missing, running or foreign diagnostic short-fit ledger')
    all_attempts=formal+smoke
    interrupted=[q for q in all_attempts if q['status'] not in ('SUCCESS','FAILED')]
    zero_interrupted=sum(q['steps']==0 for q in interrupted)
    positive=sum(q['steps']>0 for q in all_attempts)
    steps=sum(q['steps'] for q in all_attempts)
    return {'status':'PASS','source':identity,'new_completed_formal_neural_fits':30,
        'qualified_reused_neural_models':60,'new_convex_or_feature_fits':0,
        'formal_attempts_recorded':len(formal),'formal_successful_attempts':sum(q['status']=='SUCCESS' for q in formal),
        'formal_failed_attempts':sum(q['status']=='FAILED' for q in formal),
        'formal_interrupted_attempts':sum(q['status'] not in ('SUCCESS','FAILED') for q in formal),
        'formal_positive_committed_step_attempts':sum(q['steps']>0 for q in formal),
        'formal_completed_generator_steps':completed_steps,'formal_all_attempt_committed_generator_steps':sum(q['steps'] for q in formal),
        'formal_completed_fit_process_seconds':completed_process_seconds,
        'runtime_scope':'sum of completed fit process durations, not concurrent elapsed wall time or GPU-active seconds',
        'short_fit_attempts_recorded':len(smoke),'short_fits_with_committed_generator_steps':sum(q['steps']>0 for q in smoke),
        'short_generator_steps':sum(q['steps'] for q in smoke),
        'all_real_neural_fits_lower_bound':positive,'all_real_neural_fits_upper_bound':positive+zero_interrupted,
        'all_generator_steps_lower_bound':steps,'all_generator_steps_upper_bound':steps+len(interrupted),
        'interrupted_attempts_with_callback_commit_uncertainty':len(interrupted),
        'zero_committed_step_interrupted_attempts':zero_interrupted,'discriminator_optimizer_steps':0,
        'formal_attempt_ledger_sha256':ledger_hashes,'short_fit_ledger_sha256':sha(smoke_path),
        'fit_count_scope':'new positive-step attempts only; sixty frozen models are references; interruptions may have one uncommitted update'}


def _table(name):
    with (R/(name+'.csv')).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def verify_completion():
    from src.phase21_relative import data
    from scripts.validate_phase21 import gates,checkpoint
    from scripts.audit_phase21_saved import audit_identity,snapshot_inventory
    qualified,identity,config=gates(require_done=True)
    require(qualified.P==P and qualified.R==R and config['parent_commit']==PARENT
            and config['branch']==BRANCH,'foreign protocol/private path')
    evaluator=sha(ROOT/'scripts/validate_phase21.py')
    done=read(P/'EVALUATION_DONE.json')
    require(done['status']=='PASS' and done['source']==identity and done['evaluator_sha256']==evaluator
            and done['checkpoints']==90 and done['prediction_replays']==540 and done['bootstrap_draws']==20000
            and done['OOF_vectors']==48,'real independent evaluation is incomplete/stale')
    gate_names=('INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','BUDGET_VERIFICATION','MODEL_SOURCE_AUDIT',
                'PAIR_CREDIT_IDENTITY_AUDIT','BOOTSTRAP_AUDIT','EPOCH_LOCK_AUDIT','DIRECT_DIAGNOSTIC_AUDIT')
    gate={name:read(R/(name+'.json')) for name in gate_names}
    require(all(q['status']=='PASS' and q['source']==identity for q in gate.values()),'technical evaluation gate failed')
    for name in ('INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','DIRECT_DIAGNOSTIC_AUDIT'):
        require(gate[name]['evaluator_sha256']==evaluator,'evaluator source changed: '+name)
    independent=gate['INDEPENDENT_VALIDATION'];replay=gate['CHECKPOINT_REPLAY']
    require(independent['qualified_checkpoints']==90 and independent['new_neural_fits']==30
            and independent['qualified_reused_controls']==60 and independent['prediction_replays']==540
            and independent['complete_model_seed_OOF_vectors']==48 and independent['DSS_fold_rows']==240
            and independent['PFI_fold_rows']==240 and independent['independent_sksurv_every_fold'] is True
            and independent['repair_damage_net_identity'] is True and independent['no_new_feature_convex_or_teacher_fits'] is True
            and independent['outer_epoch_and_inner_file_receipt_lock_references_identical'] is True
            and independent['DIRECT_diagnostic_checkpoints']==30 and independent['DIRECT_diagnostic_partition_views']==60,
            'independent evaluation object/counts differ')
    require(replay['checkpoints']==90 and replay['prediction_vectors']==540 and replay['maximum_replay_error']==0
            and replay['all_modes_elementwise_identical'] is True and replay['anchor_elementwise_identical'] is True,
            'all-checkpoint bitwise replay not complete')
    bootstrap=gate['BOOTSTRAP_AUDIT']
    require(bootstrap['draws_requested']==bootstrap['draws_valid']==20000 and bootstrap['zero_denominator_draws']==0
            and bootstrap['primary_family_tests']==4 and bootstrap['risk_ensembling'] is False
            and bootstrap['within_fold_only'] is True and len(bootstrap['shared_patient_count_matrices'])==5
            and all(len(q['counts_sha256'])==64 for q in bootstrap['shared_patient_count_matrices']),
            'complete shared-patient 20k family-four bootstrap required')
    expected={(a,s,f,r) for a in ARMS for s in SEEDS for f in FOLDS for r in ROLES}
    rows=_table('CHECKPOINT_REPLAY');manifest=_table('CELL_MANIFEST')
    for table in (rows,manifest):
        require(len(table)==90 and {(q['arm'],int(q['seed']),int(q['fold']),q['role']) for q in table}==expected,
                'missing/duplicate model matrix')
    manifest_map={(q['arm'],int(q['seed']),int(q['fold']),q['role']):q for q in manifest}
    selection=read(R/'EPOCH_SELECTION_LOCK.json')
    require(selection['status']=='ALL15_INNER_EPOCHS_LOCKED' and selection['source']==identity
            and selection['no_outer_results_used'] is True,'inner-only epoch lock required')
    selections={(int(q['seed']),int(q['fold'])):q for q in selection['selections']}
    require(len(selection['selections'])==len(selections)==15,'incomplete/duplicate inner epoch lock')
    direct_inner=[q for q in manifest if q['arm']=='DIRECT' and q['role']=='inner']
    cap_fraction=sum(int(q['actual_epochs'])==config['training']['max_epochs'] for q in direct_inner)/15
    budget=gate['BUDGET_VERIFICATION']
    require(len(direct_inner)==15 and abs(budget['DIRECT_hit_maximum_fraction']-cap_fraction)<1e-12
            and budget['maximum_cap_fraction']==.2 and budget['budget_gate_pass']==(cap_fraction<=.2),
            'budget classification differs from actual locked inner schedules')
    source_counts={}
    for row in manifest:
        key=(row['fit_classification'],row['actual_fit_source']);source_counts[key]=source_counts.get(key,0)+1
    source_audit=gate['MODEL_SOURCE_AUDIT']
    require(source_audit['new_neural_fits']==30 and source_audit['reused_neural_models']==60
            and source_audit['new_convex_or_feature_fits']==0
            and {(q['fit_classification'],q['actual_fit_source']):q['models'] for q in source_audit['rows']}==source_counts,
            'actual training source accounting differs from qualified model matrix')
    direct_calls=0;direct_operator_calls=0
    for row in rows:
        key=(row['arm'],int(row['seed']),int(row['fold']),row['role']);a,s,f,r=key
        ck,receipt,path=checkpoint(a,s,f,r)
        require(receipt['status']=='PASS' and receipt['checkpoint_sha256']==sha(path)==row['checkpoint_sha256']
                and sha(path.with_suffix('.json'))==row['receipt_sha256']
                and ck['source']==row['actual_fit_source']==manifest_map[key]['actual_fit_source']
                and int(row['prediction_vectors_replayed'])==6 and float(row['maximum_replay_error'])==0,
                'qualified checkpoint/receipt changed after independent replay')
        if a=='DIRECT' and r=='outer':
            inner=P/'inner'/path.name;inner_receipt=read(inner.with_suffix('.json'));locked=selections[s,f]
            require(ck['inner_checkpoint_sha256']==receipt['inner_checkpoint_sha256']==sha(inner)
                    ==inner_receipt['checkpoint_sha256']==locked['checkpoint_sha256']
                    and ck['selected_epoch']==inner_receipt['selected_epoch']==locked['selected_epoch'],
                    'outer schedule/checkpoint differs from locked inner selection')
        if a=='DIRECT':
            diagnostic_path=R/'direct_checkpoints'/f'DIRECT_{r}_seed{s}_fold{f}.json'
            diagnostic=read(diagnostic_path);part=diagnostic['diagnostics']
            require(sha(diagnostic_path)==row['direct_diagnostic_sha256'] and diagnostic['status']=='PASS'
                    and diagnostic['source']==identity and diagnostic['evaluator_sha256']==evaluator
                    and diagnostic['checkpoint_sha256']==row['checkpoint_sha256']
                    and diagnostic['receipt_sha256']==row['receipt_sha256'] and diagnostic['model_values_unchanged'] is True
                    and diagnostic['model_gradients_created'] is False
                    and part['additional_gradient_calls']==part['optimizer_steps']==diagnostic['new_training_fits']==diagnostic['optimizer_calls']==0,
                    'DIRECT frozen diagnostic identity/read-only qualification differs')
            for partition,count_name in (('fit','fit_patients'),('heldout','heldout_patients')):
                require(part[partition]['status']=='PASS' and part[partition]['partition']==partition
                        and part[partition][count_name]>0,'DIRECT fit/heldout diagnostic missing')
            direct_calls+=part['additional_model_forward_calls'];direct_operator_calls+=part['additional_operator_calls']
        del ck
    direct_gate=gate['DIRECT_DIAGNOSTIC_AUDIT'];epoch_gate=gate['EPOCH_LOCK_AUDIT']
    require(direct_gate['checkpoints']==30 and direct_gate['partition_views']==60
            and direct_gate['fit_views']==direct_gate['heldout_views']==30
            and direct_gate['diagnostic_model_forward_calls']==direct_calls
            and direct_gate['diagnostic_operator_calls']==direct_operator_calls
            and direct_gate['additional_gradient_calls']==direct_gate['new_training_fits']==direct_gate['optimizer_calls']==0
            and direct_gate['diagnostic_CSV_sha256']==sha(R/'DIRECT_DIAGNOSTICS.csv'),
            'DIRECT complete diagnostic coverage/counts differ')
    require(epoch_gate['unique_inner_selections']==15 and len(epoch_gate['rows'])==15
            and epoch_gate['epoch_selection_lock_sha256']==sha(R/'EPOCH_SELECTION_LOCK.json')
            and epoch_gate['outer_epoch_and_inner_file_receipt_lock_references_identical'] is True
            and epoch_gate['no_outer_results_used'] is True,'independent outer epoch lineage gate incomplete')
    pairs=_table('PAIRED_COMPARISONS');primary=[q for q in pairs if q['family']=='primary']
    require(len(primary)==4 and {(q['comparison'],q['metric']) for q in primary}==
            {(c,m) for c in ('DIRECT-S16','DIRECT-P16') for m in ('Harrell','Uno60')}
            and all(int(float(q['family_tests']))==4 and all(math.isfinite(float(q[k])) for k in
                ('delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high')) for q in primary),
            'prelocked four paired tests or intervals missing')
    for name in ('FOLD_METRICS','PFI_FOLD_METRICS'):require(len(_table(name))==240,'incomplete fold metrics')
    for name in ('SEED_METRICS','PFI_SEED_METRICS'):require(len(_table(name))==48,'incomplete complete-seed metrics')
    inventory=snapshot_inventory(verify_all=True);audit_source,_=audit_identity()
    saved=read(R/'SAVED_S16_AUDIT.json')
    require(saved['status']=='PASS' and saved['audit_source']==audit_source and saved['checkpoint_count']==30
            and saved['prediction_vectors_replayed']==120 and saved['new_training_fits']==saved['optimizer_calls']==0
            and saved['immutable_original_inventory_sha256']==sha(P/'SAVED_S16_ORIGINAL_INVENTORY.json')
            and saved['diagnostic_CSV_sha256']==sha(R/'SAVED_S16_DIAGNOSTICS.csv'),
            'complete frozen fit/heldout S16 audit required')
    diag_calls=0
    for r in ROLES:
        for s in SEEDS:
            for f in FOLDS:
                name=f'S16_{r}_seed{s}_fold{f}';path=R/'saved_checkpoints'/(name+'.json')
                record=read(path);receipt=read(P/'saved_audits'/(name+'.json'))
                original=next(q for q in inventory['checkpoints'] if (q['role'],q['seed'],q['fold'])==(r,s,f))
                require(record['status']==receipt['status']=='PASS' and record['audit_source']==receipt['audit_source']==audit_source
                        and receipt['public_sha256']==sha(path) and record['checkpoint_sha256']==original['checkpoint_sha256']
                        and receipt['private_risks_sha256']==sha(P/'frozen_saved_risks'/(name+'.npz')),
                        'frozen per-checkpoint diagnosis changed')
                for partition,count_name in (('fit','fit_patients'),('heldout','heldout_patients')):
                    d=record['diagnostics'][partition]
                    require(d['status']=='PASS' and d['partition']==partition and d[count_name]>0
                            and d['additional_gradient_calls']==d['optimizer_steps']==0,
                            'frozen attention diagnostic partition missing')
                    diag_calls+=d['additional_model_forward_calls']
    require(saved['diagnostic_model_forward_calls']==diag_calls,'frozen diagnostic call accounting differs')
    execution=execution_summary(P,identity)
    require(read(R/'EXECUTION_COUNTS.json')==execution,'public actual fit/step accounting differs from private ledgers')
    qualification=read(R/'QUALIFICATION.json')
    require(qualification['tests_passed']==25 and qualification['real_short_fits']==execution['short_fit_attempts_recorded']
            and qualification['short_steps']==execution['short_generator_steps'],
            'actual preparation tests/short-fit counts changed')
    report=read(R/'REPORT_INPUT_AUDIT.json');decision=read(R/'decision.json')
    require(report['status']=='PASS' and report['source']==identity and report['files']
            and report['results_sha256']==sha(R/'RESULTS.md') and report['decision_sha256']==sha(R/'decision.json'),
            'final report is incomplete or stale')
    prefix=R.relative_to(ROOT).as_posix()+'/'
    required_report_inputs={CONFIG,'scripts/report_phase21.py'}
    required_report_inputs.update(prefix+name+'.json' for name in (*gate_names,'QUALIFICATION','PROTOCOL_LOCK',
        'EPOCH_SELECTION_LOCK','SAVED_S16_AUDIT','EXECUTION_COUNTS'))
    required_report_inputs.update(prefix+name+'.csv' for name in ('PAIRED_COMPARISONS','REPAIR_DAMAGE',
        'SEED_REPAIR_DAMAGE','SEED_METRICS','FOLD_METRICS','SAVED_S16_DIAGNOSTICS','DIRECT_DIAGNOSTICS'))
    require(required_report_inputs.issubset(report['files']),'final report does not bind all required real input artifacts')
    inherited=read(P/'PARENT_TRACKED_HASHES.json')
    for relative,expected_sha in report['files'].items():
        require((allowed_path(relative) or inherited.get(relative)==expected_sha)
                and sha(safe_path(relative))==expected_sha,'reported input file changed: '+relative)
    require(decision.get('source')==identity and isinstance(decision.get('status'),str) and decision['status'],
            'decision source/status missing; negative scientific results are permitted')
    return {'status':'PASS','source':identity,'scientific_decision':decision['status'],
        'new_completed_formal_neural_fits':30,'qualified_reused_neural_models':60,'new_convex_or_feature_fits':0,
        'qualified_checkpoint_replays':90,'evaluation_prediction_vector_replays':540,
        'additional_saved_S16_risk_vector_replays':120,'additional_saved_S16_diagnostic_model_calls':diag_calls,
        'additional_DIRECT_diagnostic_model_calls':direct_calls,'additional_DIRECT_diagnostic_operator_calls':direct_operator_calls,
        'primary_family_tests':4,'bootstrap_draws':20000,'complete_model_seed_OOF_vectors':48,
        'evaluator_sha256':evaluator,'audit_source':audit_source,'package_program_sha256':sha(Path(__file__)),
        'execution_counts_sha256':sha(R/'EXECUTION_COUNTS.json'),'report_input_audit_sha256':sha(R/'REPORT_INPUT_AUDIT.json'),
        'budget_gate_pass':gate['BUDGET_VERIFICATION']['budget_gate_pass'],
        'scientific_success_not_required_for_publication':True,'no_new_training_or_selection_by_packager':True}


def mirror_archive(paths,manifest):
    destination=P/'publication'/'phase21_public.tar.gz';destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    relative=path.relative_to(ROOT).as_posix();audit_public_file(path)
                    content=path.read_bytes();info=tarfile.TarInfo(relative)
                    info.size=len(content);info.mode=0o644;info.mtime=0;info.uid=info.gid=0
                    info.uname=info.gname='';archive.addfile(info,io.BytesIO(content))
    temporary.replace(destination)
    receipt={'status':'PASS','source':manifest['source'],'archive_name':destination.name,
        'archive_sha256':sha(destination),'archive_bytes':destination.stat().st_size,
        'public_manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json'),'files':len(paths),
        'archive_contains_public_manifest':True,'archive_is_private_transfer_artifact':True,
        'public_file_hashes':{path.relative_to(ROOT).as_posix():sha(path) for path in paths},
        'git_commit_or_push_performed':False}
    write(P/'publication'/'PACKAGE_RECEIPT.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',action='store_true',help='Require complete real evidence, then write manifest/private tar.')
    parser.add_argument('--check-completion',action='store_true',help='Also require full completion in a nonwriting dry run.')
    parser.add_argument('--dry-run',action='store_true',help='Default: privacy and parent preflight; writes nothing.')
    args=parser.parse_args();require(not (args.release and args.dry_run),'release and dry run conflict')
    parent,legacy=audit_parent_immutability()
    rows=[audit_public_file(path) for path in publication_paths()]
    completion=verify_completion() if args.release or args.check_completion else None
    if not args.release:
        print(json.dumps({'mode':'DRY_RUN','parent_status':parent['status'],'public_files_checked':len(rows),
            'completion_checked':completion is not None,'writes':0,'git_operations':0},sort_keys=True));return
    write(R/'PARENT_IMMUTABILITY_AUDIT.json',parent)
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'only new Phase21 source/config and anonymous aggregate reports',
        'base_commit':PARENT,'unchanged_inherited_violations':legacy,'new_artifacts_strict_privacy':True,
        'inherited_ZIPs_not_modified_or_readded':True,'no_repository_wide_privacy_PASS_claim':True,
        'excludes_temporary_files_and_private_data':True})
    rows=[audit_public_file(path) for path in publication_paths()]
    manifest={**completion,'parent_commit':PARENT,'branch':BRANCH,'public_files':rows,
        'manifest_does_not_hash_itself':True,'no_automatic_followup_training':True,
        'no_git_commit_or_push':True,'private_excluded':['patient risks/OOF','checkpoints/weights','raw arrays/maps/tokens',
            'patient/gene/donor identifiers','PCA/prototypes','credentials','temporary files']}
    write(R/'ARTIFACT_MANIFEST.json',manifest);audit_public_file(R/'ARTIFACT_MANIFEST.json')
    receipt=mirror_archive(publication_paths(include_manifest=True),manifest)
    require(all(sha(safe_path(q['path']))==q['sha256'] for q in rows),'public artifact changed during packaging')
    print('PHASE21_PACKAGE_PASS',len(rows),receipt['archive_sha256'],completion['scientific_decision'],flush=True)


if __name__=='__main__':main()
