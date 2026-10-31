"""M1--M3 text-only publication; unchanged original privacy rules and parent536."""
from pathlib import Path,PurePosixPath
from contextlib import contextmanager
import argparse,csv,hashlib,io,json,os,re,stat,subprocess,zipfile,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import bgb_en_publication as original

SCANNER='4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98'
DENYLIST='d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867'
BASE='4a913942012f4aabf975dada4b70ad7a7be5cfd0'
TREE='172286d5a59801e5bb0065010b4499b3e7960a48'
BASELINE='546f2b9ba8110f24cdcd5fc56f6460d5509fe472b05c12877e910fa295fe31e3'
MASTER='41f041bcb883d65285167a201183f40d41e9de1640ead2b40a9311f61f318af3'
ANALYSIS='903814ae0390fc43a26a64769c0073af27b2c96da6211b27a6df42f150626257'
DIAGNOSTIC='3e01a6c1ab63d00e0d3d45a100fe633bf5c64a2821fb9acfe34624f9aa7ef14b'
ROLES='536e8f95fd333655cd1320d5e8bbc48f910e8bfeaa3e70ec2b5d2b88b25ff363'
BRANCH='research/joint-trimodal-m1-m3'
REPORTS='reports/semantic_m123/'
RECEIPT=REPORTS+'PUBLICATION_SCAN_RECEIPT.json'
MANIFEST=REPORTS+'PUBLICATION_MANIFEST.json'


def require(v,msg):
    if not v:raise ValueError(msg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def new(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw)


def row(path,name):
    raw=Path(path).read_bytes()
    return {'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def allowed(name):
    p=PurePosixPath(name)
    if name=='configs/semantic_m123/master_lock.json':return False
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name and (
      name=='M1_M5_JOINT_TRIMODAL_REDESIGN_PROMPT.md' or
      re.fullmatch(r'(src/semantic_m123/[A-Za-z0-9_]+\.py|scripts/semantic_m123_[A-Za-z0-9_]+\.py|tests/test_semantic_m123_[A-Za-z0-9_]+\.py|configs/semantic_m123/[A-Za-z0-9_]+\.json)',name) is not None or
      (name.startswith(REPORTS) and len(p.parts)==3 and p.suffix in ('.md','.json','.csv')))


@contextmanager
def allow(rows):
    require(sha(original.__file__)==SCANNER,'ORIGINAL_SCANNER_CHANGED')
    old=original.ALLOWLIST;original.ALLOWLIST=frozenset(r['file'] for r in rows)
    try:yield
    finally:original.ALLOWLIST=old


def scan(root,rows,pattern):
    rows=sorted(rows,key=lambda x:x['file'])
    with allow(rows):
        actual=[original.scan_file(Path(root).resolve(),r['file'],pattern) for r in rows]
    require(actual==rows,'EXACT_PUBLIC_BYTES_CHANGED_DURING_SCAN')
    return rows


def verify_baseline(path):
    require(sha(path)==BASELINE,'PARENT536_BASELINE_CHANGED')
    d=read(path)
    require(d['commit']==BASE and d['tree']==TREE and d['file_count']==len(d['files'])==536,'WRONG_PARENT')
    return d


def verify_geometry(root,indexed):
    name='configs/semantic_m123/diagnostic_lock.json'
    require(indexed.get(name,{}).get('sha256')==DIAGNOSTIC,'DIAGNOSTIC_LOCK_NOT_EXACT')
    lock=read(Path(root)/name)
    require(lock['master_lock_sha256']==MASTER and lock['expected_model_forwards']==105,'DIAGNOSTIC_SCOPE_CHANGED')
    for name,h in lock['sources'].items():
        require(indexed.get(name,{}).get('sha256')==h,'DIAGNOSTIC_SOURCE_CLOSURE:'+name)
    audit=read(Path(root)/(REPORTS+'FROZEN_ATTENTION_GEOMETRY_AUDIT.json'))
    require(audit['schema']=='m123_frozen_attention_geometry_audit_v1' and audit['status']=='COMPLETE'
        and audit['actual_model_forwards']==audit['saved_model_states']==105
        and audit['master_lock_sha256']==MASTER and audit['diagnostic_lock_sha256']==DIAGNOSTIC
        and audit['source_sha256']==lock['sources'] and audit['survival_endpoint_arrays_decoded'] is False
        and audit['new_optimizer_fits']==audit['new_preprocessor_fits']==0,'FROZEN_GEOMETRY_NOT_ACTUALLY_COMPLETE')
    f=audit['file'];name=REPORTS+'FROZEN_ATTENTION_GEOMETRY.csv'
    require(f['file']=='FROZEN_ATTENTION_GEOMETRY.csv'
        and indexed.get(name,{}).get('sha256')==f['sha256'],'GEOMETRY_TABLE_NOT_EXACT')
    with (Path(root)/name).open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    arms=('M2_SINGLE','M2_EIGHT','M2_EIGHT_MEAN','M3_FULL','M3_LEARNED_QUERY','M3_NO_PRODUCT','M3_BIDIRECTIONAL')
    expected={(o,a,s) for o in range(1,6) for a in arms for s in (31,47,83)}
    actual={(int(r['outer']),r['arm'],int(r['seed'])) for r in rows}
    require(len(rows)==len(actual)==105 and actual==expected,'GEOMETRY_FOLD_SEED_ARM_COVERAGE')
    return {'frozen_geometry_forwards':105,'new_geometry_training_fits':0,'histological_ground_truth_claim':False}


def verify_parent_git(repo,base):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',GIT_NO_REPLACE_OBJECTS='1')
    prefix=['git','-c','protocol.allow=never','-C',str(repo)]
    def git(*args):return subprocess.run(prefix+list(args),check=True,capture_output=True,env=env,timeout=60).stdout
    def control():
        p=Path(git('rev-parse','--git-path','index').decode().strip())
        if not p.is_absolute():p=Path(repo)/p
        return {'HEAD':git('rev-parse','HEAD').decode().strip(),'index_sha256':sha(p) if p.exists() else None,
            'refs_sha256':hashlib.sha256(git('for-each-ref','--format=%(objectname) %(refname)')).hexdigest()}
    before=control()
    require(git('rev-parse',BASE+'^{tree}').decode().strip()==TREE,'FIXED_PARENT_TREE_CHANGED')
    entries={}
    for line in git('ls-tree','-rz','--full-tree',BASE).split(b'\0'):
        if not line:continue
        meta,name=line.split(b'\t',1);mode,kind,oid=meta.decode().split()
        entries[name.decode('utf-8')]=(mode,kind,oid)
    process=subprocess.Popen(prefix+['cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    for r in base['files']:
        require(entries.get(r['file']) in (('100644','blob',r['git_blob_sha1']),('100755','blob',r['git_blob_sha1'])),'OLD_PATH_TREE_CHANGED')
        process.stdin.write((r['git_blob_sha1']+'\n').encode());process.stdin.flush()
        header=process.stdout.readline().decode().strip().split()
        require(header==[r['git_blob_sha1'],'blob',str(r['bytes'])],'PARENT_BLOB_HEADER')
        raw=process.stdout.read(r['bytes']);require(process.stdout.read(1)==b'\n','PARENT_BLOB_TRAILER')
        require(hashlib.sha256(raw).hexdigest()==r['sha256'],'PARENT_RAW_BYTES_CHANGED')
        require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1'],'PARENT_GIT_BLOB_CHANGED')
    process.stdin.close();require(process.wait(timeout=30)==0,'PARENT_CAT_FILE_FAILED')
    require(control()==before,'CHECKOUT_OR_REFS_CHANGED')
    return {'verified_raw_Git_files':536,'verified_bytes':base['total_bytes'],'HEAD_index_refs_unchanged':True,'Git_writes':0}


def proof_closure(root,rows,base):
    indexed={r['file']:r for r in rows}
    require(len(indexed)==len(rows),'DUPLICATE_CANDIDATE')
    require(not set(indexed)&{r['file'] for r in base['files']},'CANNOT_OVERWRITE_PARENT536')
    for r in rows:
        require(set(r)=={'file','bytes','sha256'} and isinstance(r['bytes'],int) and 0<r['bytes']<=2*1024**2
            and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'BAD_CANDIDATE_METADATA')
        require(allowed(r['file']) and r['file'] not in (RECEIPT,MANIFEST),'OUTSIDE_EXPLICIT_NEW_NAMESPACE')
        require(row(Path(root)/r['file'],r['file'])==r,'CANDIDATE_BYTES_CHANGED')
    master_name='configs/semantic_m123/master_public.json';analysis_name='configs/semantic_m123/analysis_lock_v2.json'
    require(master_name in indexed,'PUBLIC_MASTER_PROJECTION_REQUIRED')
    require(indexed.get(analysis_name,{}).get('sha256')==ANALYSIS,'ANALYSIS_NOT_EXACT')
    # Keep execution paths private without weakening the original absolute-
    # path rule. Verify every other field against the actual unchanged lock.
    private_master=ROOT/'configs/semantic_m123/master_lock.json'
    require(sha(private_master)==MASTER,'ORIGINAL_PRIVATE_EXECUTION_LOCK_CHANGED')
    master=read(private_master);public=read(Path(root)/master_name);analysis=read(Path(root)/analysis_name)
    path_fields=('assets_path','legacy_root','parent_private','parent_split_plan','frozen_images')
    require(public['schema']=='m123_public_scientific_lock_projection_v1'
        and public['private_execution_lock_sha256']==MASTER
        and public['scientific_master']=={k:v for k,v in master.items() if k not in path_fields}
        and public['machine_path_placeholders']=={k:'${'+k.upper()+'}' for k in path_fields},
        'PUBLIC_LOCK_CHANGED_SCIENTIFIC_FIELDS_NOT_JUST_MACHINE_PATHS')
    old={r['file']:r for r in base['files']}
    for name,h in {**master['sources'],**analysis['analysis_sources']}.items():
        require((indexed.get(name) or old.get(name) or {}).get('sha256')==h,'SOURCE_CLOSURE:'+name)
    design=read(Path(root)/'configs/semantic_m123/design_precommit.json')
    require(indexed.get('configs/semantic_m123/design_precommit.json',{}).get('sha256')==master['design_precommit_sha256'],'DESIGN_CHANGED')
    require(indexed.get('M1_M5_JOINT_TRIMODAL_REDESIGN_PROMPT.md',{}).get('sha256')==design['user_prompt_sha256'],'USER_PROMPT_CHANGED')
    require(indexed.get(REPORTS+'01_REAL_ASSET_QUALIFICATION.json',{}).get('sha256')==master['qualification_sha256'],'QUALIFICATION_CHANGED')
    index=read(Path(root)/(REPORTS+'REPORT_INDEX.json'))
    require(index['schema']=='m123_anonymous_report_index_v1' and index['status']=='COMPLETE' and index['lock_sha256']==MASTER,'SCIENTIFIC_REPORT_INDEX')
    for r in index['files']:
        require(indexed.get(REPORTS+r['file'])=={**r,'file':REPORTS+r['file']},'MISSING_EXACT_REPORT')
    require(index['reporter_source_sha256']==indexed['scripts/semantic_m123_report.py']['sha256'],'REPORTER_NOT_BOUND')
    audit=read(Path(root)/(REPORTS+'09_ACTUAL_REPLAY_OFFICIAL_SCORER_AUDIT.json'))
    require(audit['status']=='PASS' and audit['registered_private_fit_artifacts_verified']==1100 and audit['outer_metric_cases']==205
        and audit['actual_official_API_calls']==410 and audit['new_preprocessor_fits']==0,'ACTUAL_AUDIT_NOT_COMPLETE')
    cost=read(Path(root)/(REPORTS+'10_COMPUTE_LEDGER.json'))
    require(cost['neural_initializations']==1050 and cost['Cox_fits']==50 and cost['new_teacher_fits']==0
        and cost['private_OOF_vectors']==41,'ACTUAL_FIT_LEDGER_NOT_COMPLETE')
    decision=read(Path(root)/(REPORTS+'decision.json'))
    require(decision['status']=='COMPLETE_LOCKED_M1_M2_M3' and decision['CI_bootstrap_pvalues'] is False
        and decision['M4_M5_trained'] is False,'SCIENTIFIC_DECISION_SCOPE_CHANGED')
    geometry=verify_geometry(root,indexed)
    role_name='configs/semantic_m123/readout_roles_lock.json'
    require(indexed.get(role_name,{}).get('sha256')==ROLES,'CHECKPOINT_ROLE_LOCK_CHANGED')
    role_lock=read(Path(root)/role_name)
    for n,h in role_lock['sources'].items():require(indexed.get(n,{}).get('sha256')==h,'ROLE_SOURCE_CLOSURE')
    roles=read(Path(root)/(REPORTS+'FROZEN_CHECKPOINT_TRAINING_ROLES_AUDIT.json'))
    require(roles['status']=='COMPLETE' and roles['actual_checkpoints_loaded']==150
        and roles['master_lock_sha256']==MASTER and roles['role_lock_sha256']==ROLES
        and roles['source_sha256']==role_lock['sources'] and roles['new_optimizer_fits']==roles['new_model_forwards']==0
        and roles['survival_endpoint_arrays_decoded'] is False,'ACTUAL_CHECKPOINT_ROLE_AUDIT_INCOMPLETE')
    role_table=REPORTS+'FROZEN_CHECKPOINT_TRAINING_ROLES.csv'
    require(indexed.get(role_table,{}).get('sha256')==roles['table_sha256'],'ROLE_TABLE_CHANGED')
    with (Path(root)/role_table).open(encoding='utf-8',newline='') as f:role_rows=list(csv.DictReader(f))
    require(len(role_rows)==150 and {(int(r['outer']),r['arm'],int(r['seed'])) for r in role_rows}
        =={(o,a,s) for o in range(1,6) for a in design['architectures']['arms'] for s in (31,47,83)},'ROLE_TABLE_COVERAGE')
    arithmetic=read(Path(root)/(REPORTS+'INDEPENDENT_AGGREGATE_ARITHMETIC_AUDIT.json'))
    require(arithmetic['status']=='PASS' and arithmetic['fold_rows']==205 and arithmetic['seed_vectors']==41
        and arithmetic['arm_summaries']==15 and arithmetic['paired_comparisons']==21 and arithmetic['repair_damage_rows']==205
        and arithmetic['patient_arrays_or_weights_read'] is False and arithmetic['new_optimizer_fits']==0
        and arithmetic['source_sha256']==indexed['scripts/semantic_m123_verify_reports.py']['sha256']
        and arithmetic['main_report_index_sha256']==indexed[REPORTS+'REPORT_INDEX.json']['sha256'],
        'INDEPENDENT_ARITHMETIC_AUDIT_NOT_BOUND')
    return {'master':MASTER,'analysis':ANALYSIS,'actual_registered_fits':1100,'actual_official_calls':410,
            'public_patient_arrays_or_weights':False,'independent_external_confirmation_claim':False,
            'actual_checkpoint_role_states':150,**geometry}


def plan(root,baseline,candidates,output):
    base=verify_baseline(baseline);rows=sorted(read(candidates)['files'],key=lambda x:x['file'])
    closure=proof_closure(root,rows,base)
    scan(root,rows,re.compile(r'(?!)'))
    for r in base['files']:require(row(Path(root)/r['file'],r['file'])=={k:r[k] for k in ('file','bytes','sha256')},'LOCAL_PARENT_MODIFIED')
    d={'schema':'m123_publication_plan_v1','status':'LOCAL_PREFLIGHT_REQUIRES_ACTUAL_PRIVATE_SCAN','base_commit':BASE,'base_tree':TREE,
        'branch':BRANCH,'parent536_baseline_sha256':BASELINE,'files':rows,'proof_closure':closure}
    new(output,encoded(d));return d


def stage(root,plan_path,baseline,archive):
    d=read(plan_path);base=verify_baseline(baseline)
    require(d['schema']=='m123_publication_plan_v1' and d['base_commit']==BASE and d['base_tree']==TREE
        and d['branch']==BRANCH and d['parent536_baseline_sha256']==BASELINE,'PUBLICATION_PLAN_CONTROL_CHANGED')
    proof_closure(root,d['files'],base)
    require(not Path(archive).exists(),'STAGE_ARCHIVE_EXISTS')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in d['files']:
            require(row(Path(root)/r['file'],r['file'])==r,'STAGE_SOURCE_CHANGED')
            z.writestr(r['file'],(Path(root)/r['file']).read_bytes())
        z.writestr('_publication/PLAN.json',Path(plan_path).read_bytes())
        z.writestr('_publication/BASELINE536.json',Path(baseline).read_bytes())
    return {'archive_sha256':sha(archive),'plan_sha256':sha(plan_path),'files':len(d['files'])}


def final_scan(archive,archive_sha,plan_path,plan_sha,baseline,denylist,repo,output):
    require(sha(archive)==archive_sha and sha(plan_path)==plan_sha,'STAGE_OR_PLAN_CHANGED')
    require(sha(original.__file__)==SCANNER and sha(denylist)==DENYLIST,'ESTABLISHED_PRIVACY_RULES_CHANGED')
    base=verify_baseline(baseline);d=read(plan_path)
    require(d['schema']=='m123_publication_plan_v1' and d['base_commit']==BASE and d['base_tree']==TREE
        and d['branch']==BRANCH and d['parent536_baseline_sha256']==BASELINE,'PUBLICATION_PLAN_CONTROL_CHANGED')
    pattern=original.identifier_pattern(original.validate_denylist(read(denylist)))
    require(not Path(output).exists(),'NEW_FINAL_DIRECTORY_REQUIRED')
    payload=Path(output)/'public';payload.mkdir(parents=True)
    require(len({r['file'] for r in d['files']})==len(d['files']),'DUPLICATE_PLANNED_FILE')
    for r in d['files']:
        require(allowed(r['file']) and r['file'] not in (RECEIPT,MANIFEST)
            and set(r)=={'file','bytes','sha256'} and isinstance(r['bytes'],int) and 0<r['bytes']<=2*1024**2,
            'UNSAFE_ARCHIVE_PLAN_PATH_OR_SIZE')
    expected={r['file']:r for r in d['files']}
    expected['_publication/PLAN.json']=row(plan_path,'_publication/PLAN.json')
    expected['_publication/BASELINE536.json']=row(baseline,'_publication/BASELINE536.json')
    with zipfile.ZipFile(archive) as z:
        require(not z.comment,'UNSCANNED_ZIP_METADATA')
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(expected),'ZIP_EXTRA_MISSING_OR_DUPLICATE')
        for info in z.infolist():
            mode=info.external_attr>>16
            require(not info.is_dir() and not stat.S_ISLNK(mode) and not(info.flag_bits&1)
                and not info.extra and not info.comment,'ZIP_UNSAFE_ENTRY')
            r=expected[info.filename];raw=z.read(info)
            require(len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],'ZIP_PAYLOAD_CHANGED')
            new(payload/info.filename,raw)
    proof_closure(payload,d['files'],base)
    parent=verify_parent_git(repo,base)
    rows=scan(payload,d['files'],pattern)
    controls=[expected['_publication/PLAN.json'],expected['_publication/BASELINE536.json']]
    scan(payload,controls,pattern)
    receipt={'schema':'m123_actual_publication_scan_v1','status':'PASS_ORIGINAL_SCANNER_AND_ESTABLISHED_PRIVATE_IDENTIFIER_RULES',
        'base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'plan_sha256':plan_sha,'archive_sha256':archive_sha,
        'original_scanner_sha256':SCANNER,'established_denylist_sha256':DENYLIST,'candidate_files_scanned_now':len(rows),
        'candidate_scan_reused':0,'control_files_scanned_now':2,'generated_payload_files_scanned_now':2,
        'parent536_Git_object_verification':parent,'proof_closure':d['proof_closure'],'Git_writes':0,
        'denylist_values_exported_or_registry_rebuilt':False}
    new(payload/RECEIPT,encoded(receipt));rr=row(payload/RECEIPT,RECEIPT)
    manifest={'schema':'m123_publication_manifest_v1','status':receipt['status'],'base_commit':BASE,'base_tree':TREE,
        'branch':BRANCH,'parent536_baseline_sha256':BASELINE,'plan_sha256':plan_sha,
        'files':sorted(rows+[rr],key=lambda x:x['file']),'manifest_does_not_hash_itself':True}
    new(payload/MANIFEST,encoded(manifest));mr=row(payload/MANIFEST,MANIFEST)
    scan(payload,[rr,mr],pattern)
    require(verify_parent_git(repo,base)==parent,'OLD_PARENT_CHANGED_DURING_SCAN')
    allrows=sorted(rows+[rr,mr],key=lambda x:x['file'])
    archiveout=Path(output)/'PUBLICATION.zip'
    with zipfile.ZipFile(archiveout,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in allrows:
            raw=(payload/r['file']).read_bytes()
            r['git_blob_sha1']=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            z.writestr(r['file'],raw)
    binding={'schema':'m123_final_publication_binding_v1','status':'PASS_ANONYMOUS_TEXT_ONLY_NO_GIT_WRITE',
        'base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'parent536_baseline_sha256':BASELINE,
        'candidate_files_scanned_now':len(rows),'candidate_scan_reused':0,'published_files':len(allrows),
        'publication_archive_sha256':sha(archiveout),'publication_manifest_sha256':mr['sha256'],
        'scan_receipt_sha256':rr['sha256'],'files':allrows,'Git_writes':0,
        'original_scanner_sha256':SCANNER,'established_denylist_sha256':DENYLIST}
    bp=Path(output)/'FINAL_BINDING.json';new(bp,encoded(binding))
    scan(Path(output),[row(bp,'FINAL_BINDING.json')],pattern)
    return binding


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['plan','stage','final-scan'])
    for k in ('root','baseline','candidates','plan','archive','expected-archive-sha256','expected-plan-sha256','denylist','git-repo','output'):
        p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='plan':d=plan(a.root,a.baseline,a.candidates,a.output);print(json.dumps({'candidate_files':len(d['files']),'plan_sha256':sha(a.output)}))
    elif a.command=='stage':print(json.dumps(stage(a.root,a.plan,a.baseline,a.archive)))
    else:
        b=final_scan(a.archive,a.expected_archive_sha256,a.plan,a.expected_plan_sha256,a.baseline,a.denylist,a.git_repo,a.output)
        print(json.dumps({k:v for k,v in b.items() if k!='files'}))


if __name__=='__main__':main()

