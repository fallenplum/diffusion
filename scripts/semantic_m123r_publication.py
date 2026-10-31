"""Explicit text-only repair publication with unchanged established scanner."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,re,stat,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import semantic_m123_publication as old

BASE='4b212b5fe3d1f86915ec41261fe10f817539d9af'
TREE='8896efa03efdafa3c3252c9a68b5f041e1842779'
MASTER='ba830987731efa23f82679d7182ff0c0fdbeacbd9767cd26d210cdf2cd93636e'
OLD_MASTER='a60de1b2ae380ce4a67b49e15d6b388b80f35b52190fb534fca3da449fac398f'
OLD_BINDING='8155dbf52131f47989e78bd85e5418403b786a9a7f15a9462f9c0c68a7c70bf2'
BRANCH='research/joint-trimodal-m123-repair'
REPORTS='reports/semantic_m123r/'
RECEIPT=REPORTS+'PUBLICATION_SCAN_RECEIPT.json';MANIFEST=REPORTS+'PUBLICATION_MANIFEST.json'
sha=old.sha;read=old.read;encoded=old.encoded;new=old.new;row=old.row;require=old.require


def allowed(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or 'master_lock' in p.name:return False
    return bool(re.fullmatch(r'(src/semantic_m123r/[A-Za-z0-9_]+\.py|scripts/semantic_m123r_[A-Za-z0-9_]+\.py|tests/test_semantic_m123r(?:_[A-Za-z0-9_]+)?\.py|configs/semantic_m123r/[A-Za-z0-9_]+\.json)',name)
       or name.startswith(REPORTS) and len(p.parts) in (3,4) and p.suffix in ('.md','.json','.csv'))


def baseline(parent536,prior61):
    require(sha(parent536)==old.BASELINE and sha(prior61)==OLD_BINDING,'PARENT_BINDING_CHANGED')
    b=read(parent536);p=read(prior61)
    require(b['file_count']==536 and p['published_files']==61 and p['status']=='PASS_ANONYMOUS_TEXT_ONLY_NO_GIT_WRITE','PARENT_SCOPE_CHANGED')
    rows=b['files']+p['files']
    require(len(rows)==len({r['file'] for r in rows})==597,'PARENT_REGISTERED597_COVERAGE')
    return {'commit':BASE,'tree':TREE,'file_count':597,'full_parent_tree_paths':4117,'files':rows}


def scientific_projection(path):
    require(sha(path)==MASTER,'EXECUTION_MASTER_CHANGED')
    m=read(path);fields=('parent_master','previous_study_private')
    return {'schema':'m123r_public_execution_projection_v1','private_execution_lock_sha256':MASTER,
       'scientific_master':{k:v for k,v in m.items() if k not in fields},
       'machine_path_placeholders':{k:'${'+k.upper()+'}' for k in fields}}


def closure(root,rows,base):
    root=Path(root);indexed={r['file']:r for r in rows};prior={r['file']:r for r in base['files']}
    require(len(indexed)==len(rows) and not set(indexed)&set(prior),'NEW_ONLY_CANDIDATES_REQUIRED')
    for r in rows:
        require(set(r)=={'file','bytes','sha256'} and allowed(r['file']) and r['file'] not in (RECEIPT,MANIFEST),'UNSAFE_CANDIDATE')
        require(row(root/r['file'],r['file'])==r,'CANDIDATE_CHANGED')
    projection=read(root/'configs/semantic_m123r/master_public.json')
    require(projection==scientific_projection(ROOT/'configs/semantic_m123r/master_lock.json'),'PUBLIC_PROJECTION_CHANGED_SCIENTIFIC_FIELDS')
    for name,h in projection['scientific_master']['sources'].items():
        require((indexed.get(name) or prior.get(name) or {}).get('sha256')==h,'TRAINING_SOURCE_CLOSURE:'+name)
    require(projection['scientific_master']['artifact_replay_amendment']['previous_execution_master_sha256']==OLD_MASTER,'FAILED_ATTEMPT_LINEAGE')
    required=('qualification/REPORT_INDEX.json','qualification/RESULTS.md','qualification/decision.json',
      'qualification/07_ACTUAL_AUDIT.json','learning/AUDIT.json','learning/REPORT_INDEX.json',
      'INDEPENDENT_ARITHMETIC_AUDIT.json','SOFTWARE_TEST_LEDGER.json','REPRODUCE.md','SCIENTIFIC_INTERPRETATION.md')
    for name in required:require(REPORTS+name in indexed,'REQUIRED_REPORT_MISSING:'+name)
    d=read(root/(REPORTS+'qualification/decision.json'));audit=read(root/(REPORTS+'qualification/07_ACTUAL_AUDIT.json'))
    require(d['status']=='COMPLETE_DEVELOPMENT_QUALIFICATION' and d['lock_sha256']==MASTER
        and d['CI_bootstrap_pvalues'] is False and d['M4_M5_trained'] is False,'QUALIFICATION_SCOPE_OR_STATE')
    require(audit['status']=='PASS' and audit['registered_real_fits']==540 and audit['new_optimizer_updates']==75600
        and audit['actual_selected_checkpoint_replays_after_retention']==144 and audit['actual_official_API_calls']==480
        and audit['outer1_test_endpoint_elements_decoded']==0,'ACTUAL_QUALIFICATION_NOT_COMPLETE')
    indices=['qualification','learning']+(['formal'] if d['formal_arms'] else [])
    for directory in indices:
        index=read(root/(REPORTS+directory+'/REPORT_INDEX.json'))
        require(index['status']=='COMPLETE','REPORT_INDEX_NOT_COMPLETE')
        for r in index['files']:
            name=REPORTS+directory+'/'+r['file']
            require(indexed.get(name)=={**r,'file':name},'REPORT_INDEX_BYTES_CHANGED:'+name)
    if d['formal_arms']:
        f=read(root/(REPORTS+'formal/decision.json'))
        require(f['status']=='COMPLETE_GATED_FORMAL_VALIDATION' and f['formal_arms']==d['formal_arms'] and f['lock_sha256']==MASTER,'GATED_FORMAL_INCOMPLETE')
    else:require(not any(n.startswith(REPORTS+'formal/') for n in indexed),'UNQUALIFIED_FORMAL_REPORT')
    return {'registered_qualification_fits':540,'registered_qualification_optimizer_updates':75600,'retired_failed_attempt_updates':10,
       'actual_selected_checkpoint_replays':144,'actual_qualification_official_API_calls':480,'formal_arms':d['formal_arms'],
       'no_CI_bootstrap_pvalues':True,'M4_M5_trained':False}


def stage(root,parent536,prior61,candidates,archive):
    base=baseline(parent536,prior61);rows=sorted(read(candidates)['files'],key=lambda x:x['file'])
    proof=closure(root,rows,base);old.scan(root,rows,re.compile(r'(?!)'))
    plan={'schema':'m123r_publication_stage_v1','base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'files':rows,'proof':proof,
      'parent536_sha256':sha(parent536),'prior61_sha256':sha(prior61)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in rows:z.writestr(r['file'],(Path(root)/r['file']).read_bytes())
        z.writestr('_publication/PLAN.json',encoded(plan));z.writestr('_publication/PARENT536.json',Path(parent536).read_bytes())
        z.writestr('_publication/PRIOR61.json',Path(prior61).read_bytes())
    return {'status':'STAGED_FOR_ACTUAL_PRIVATE_SCAN','archive_sha256':sha(archive),'candidate_files':len(rows),'bytes':sum(r['bytes'] for r in rows)}


def final_scan(archive,expected_archive_sha,denylist,output):
    require(sha(archive)==expected_archive_sha and sha(denylist)==old.DENYLIST,'STAGE_OR_DENYLIST_CHANGED')
    require(sha(old.original.__file__)==old.SCANNER,'ORIGINAL_SCANNER_CHANGED')
    output=Path(output);require(not output.exists(),'NEW_OUTPUT_REQUIRED');payload=output/'public';payload.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        require(not z.comment and len(z.namelist())==len(set(z.namelist())),'UNSAFE_ZIP_METADATA')
        plan=json.loads(z.read('_publication/PLAN.json'))
        require(plan['schema']=='m123r_publication_stage_v1' and plan['base_commit']==BASE and plan['base_tree']==TREE and plan['branch']==BRANCH,'PLAN_SCOPE')
        p536=z.read('_publication/PARENT536.json');p61=z.read('_publication/PRIOR61.json')
        require(hashlib.sha256(p536).hexdigest()==old.BASELINE and hashlib.sha256(p61).hexdigest()==OLD_BINDING,'PARENT_BINDINGS')
        base={'commit':BASE,'tree':TREE,'files':json.loads(p536)['files']+json.loads(p61)['files']}
        expected={r['file']:r for r in plan['files']}
        require(set(z.namelist())==set(expected)|{'_publication/PLAN.json','_publication/PARENT536.json','_publication/PRIOR61.json'},'ZIP_UNSCANNED_EXTRA')
        for info in z.infolist():
            require(not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16) and not info.extra and not info.comment and not(info.flag_bits&1),'UNSAFE_ZIP_ENTRY')
            if info.filename not in expected:continue
            r=expected[info.filename];require(allowed(info.filename),'ZIP_PATH_OUTSIDE_REPAIR_NAMESPACE')
            raw=z.read(info);require(len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],'ZIP_BYTES_CHANGED')
            new(payload/info.filename,raw)
    proof=closure(payload,plan['files'],base)
    pattern=old.original.identifier_pattern(old.original.validate_denylist(read(denylist)))
    old.scan(payload,plan['files'],pattern)
    receipt={'schema':'m123r_actual_private_scan_receipt_v1','status':'PASS_ANONYMOUS_TEXT_ONLY_NO_GIT_WRITE',
      'candidate_files_scanned_now':len(plan['files']),'candidate_scan_reused':0,'original_scanner_sha256':old.SCANNER,
      'established_denylist_sha256':old.DENYLIST,'execution_master_sha256':MASTER,'proof':proof,
      'base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'patient_arrays_or_weights_public':False}
    new(payload/RECEIPT,encoded(receipt))
    manifest={'schema':'m123r_anonymous_publication_manifest_v1','status':'COMPLETE_SCANNED_TEXT_ONLY',
      'base_commit':BASE,'base_tree':TREE,'branch':BRANCH,'files':plan['files']+[row(payload/RECEIPT,RECEIPT)]}
    new(payload/MANIFEST,encoded(manifest))
    allrows=manifest['files']+[row(payload/MANIFEST,MANIFEST)];old.scan(payload,allrows,pattern)
    binding={'schema':'m123r_publication_final_binding_v1','status':'PASS_ANONYMOUS_TEXT_ONLY_NO_GIT_WRITE','base_commit':BASE,'base_tree':TREE,
      'branch':BRANCH,'files':[{**r,'git_blob_sha1':hashlib.sha1(b'blob '+str(r['bytes']).encode()+b'\0'+(payload/r['file']).read_bytes()).hexdigest()} for r in sorted(allrows,key=lambda x:x['file'])],
      'published_files':len(allrows),'original_scanner_sha256':old.SCANNER,'established_denylist_sha256':old.DENYLIST}
    new(output/'FINAL_BINDING.json',encoded(binding))
    with zipfile.ZipFile(output/'SCANNED_PUBLICATION.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in allrows:z.writestr(r['file'],(payload/r['file']).read_bytes())
    return {'status':binding['status'],'files':len(allrows),'bytes':sum(r['bytes'] for r in allrows),
       'archive_sha256':sha(output/'SCANNED_PUBLICATION.zip'),'binding_sha256':sha(output/'FINAL_BINDING.json')}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('stage')
    for k in ('root','parent536','prior61','candidates','archive'):q.add_argument('--'+k,required=True)
    q=sub.add_parser('final-scan')
    for k in ('archive','expected-archive-sha','denylist','output'):q.add_argument('--'+k,required=True)
    a=p.parse_args()
    result=stage(a.root,a.parent536,a.prior61,a.candidates,a.archive) if a.cmd=='stage' else final_scan(a.archive,a.expected_archive_sha,a.denylist,a.output)
    print(json.dumps(result))


if __name__=='__main__':main()
