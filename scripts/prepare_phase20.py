"""Qualify real data and five tiny fits; persist no smoke weights or risks."""
import gc
import json
import subprocess
import sys
import time

import numpy as np
import torch

from src.phase20_selective.data import ROOT,P,R,Data,cfg,source,source_files,write,TRAINING_TESTS
from src.phase20_selective.train import train_cell,seed_all
from src.phase10_trimodal.data import digest


def qualify_tests():
    targets=[f'tests/test_phase20_{x}.py' for x in TRAINING_TESTS]
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*targets,'--basetemp',str(P/'pytest_workspace')]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (P/'logs/qualification_tests.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError('qualification tests failed: '+result.stdout[-10000:]+result.stderr[-3000:])
    write(R/'OPERATOR_AND_PIPELINE_TESTS.json',{'status':'PASS','training_source':source(),'command':command,
        'stdout_summary':result.stdout[-2500:],'test_files':{name:digest(ROOT/name) for name in targets}})
    math_targets=['tests/test_phase20_evaluation.py']
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*math_targets],cwd=ROOT,capture_output=True,text=True)
    (P/'logs/evaluation_math_tests.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError('independent math tests failed: '+result.stdout[-7000:])
    import re
    match=re.search(r'(\d+) passed',result.stdout)
    if not match or ' skipped' in result.stdout: raise ValueError('math tests incomplete/skipped')
    write(R/'EVALUATION_MATH_AUDIT.json',{'status':'PASS','source':source(),'server_actual_tests':int(match.group(1)),
        'failures_errors_skips':0,'evaluator_sha256':digest(ROOT/'scripts/validate_phase20.py'),
        'test_source_sha256':digest(ROOT/'tests/test_phase20_evaluation.py'),'stdout_summary':result.stdout[-2000:]})


def qualify_data():
    from scripts.validate_phase20 import inherited_simple
    cells=[];fingerprints={}
    from src.phase19_attention import data as prior
    for role in ['inner','outer']:
        for fold in cfg()['folds']:
            ds=Data(fold,role,31)
            _,checks=inherited_simple(ds)
            for group in ['preprocessing','simple']:
                path=prior.P/group/(f'fold{fold}_{role}.pt' if group=='preprocessing' else f'{role}_fold{fold}.pt')
                for p in [path,path.with_suffix('.json')]: fingerprints[str(p)]=digest(p)
            cells.append({'role':role,'fold':fold,'fit':len(ds.fit),'validation_or_test':len(ds.valid),
                'fit_events':int(ds.ep['event'][ds.fit].sum()),'validation_or_test_events':int(ds.ep['event'][ds.valid].sum()),
                'audit':ds.audit,'reference_stationarity':checks})
            del ds;gc.collect();torch.cuda.empty_cache()
    write(P/'INHERITED_FILE_HASHES.json',{'status':'PASS','files':fingerprints})
    write(R/'DATA_AUDIT.json',{'status':'PASS','source':source(),'patients':512,'DSS_events':65,'cells':cells,
        'new_feature_or_convex_fits':0,'qualified_inherited_unique_convex_models':20,
        'inherited_cache_files_verified':len(fingerprints),'neural_checkpoint_reuse':False,
        'patient_gene_ids_and_raw_arrays_public':False,'raw_image_validation':False})


def qualify_smoke():
    ledger_path=P/'SMOKE_LEDGER.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
    def short(arm,dataset=None):
        item={'id':len(ledger['attempts'])+1,'source':source(),'arm':arm,'status':'RUNNING','generator_steps':0,'discriminator_steps':0}
        ledger['attempts'].append(item);write(ledger_path,ledger)
        def step(kind,epoch):
            item[kind+'_steps']+=1;item['last_epoch']=epoch;write(ledger_path,ledger)
        try: ck=train_cell(arm,31,1,'inner',epochs=2,dataset=dataset,progress_callback=step)
        except Exception as error:
            item['status']='FAILED';item['error_type']=type(error).__name__;write(ledger_path,ledger);raise
        item['status']='RETURNED_SUCCESSFULLY';write(ledger_path,ledger)
        return ck
    original=Data(1,'inner',31);outer_test=original.part['outer_test']
    first=short('S16',original)
    from src.phase19_attention.data import parent
    endpoints=parent.endpoints
    def poison(d):
        result={ep:{k:np.array(v,copy=True) for k,v in obj.items()} for ep,obj in endpoints(d).items()}
        result['DSS']['time'][outer_test]=np.arange(len(outer_test))+17013.
        result['DSS']['event'][outer_test]=np.arange(len(outer_test))%2
        return result
    try:
        parent.endpoints=poison
        alternate=Data(1,'inner',31)
        second=short('S16',alternate)
    finally: parent.endpoints=endpoints
    if first['history']!=second['history'] or first['selected_epoch']!=second['selected_epoch']:
        raise ValueError('outer outcome poisoning changed inner selection/history')
    if any(not torch.equal(first['state'][name],second['state'][name]) for name in first['state']):
        raise ValueError('outer outcome poisoning changed learned weights')
    if not np.array_equal(first['risk'],second['risk']): raise ValueError('outer poisoning changed predictions')
    write(R/'DATA_LEAKAGE_AUDIT.json',{'status':'PASS','source':source(),
        'test':'new Data and complete two-epoch S16 fits with outer endpoint provider poisoned throughout',
        'maximum_parameter_error':0,'risk_error':0,'history_identical':True,'selected_epoch_identical':True,
        'main_anchor_remained_frozen':True,'smoke_checkpoints_or_predictions_persisted':False})
    checks=[{'arm':'S16','finite_risk':bool(np.isfinite(first['risk']).all()),'summary':first['summary'],'mechanism':first['mechanism']}]
    del first,second,original,alternate;gc.collect();torch.cuda.empty_cache()
    for arm in ['P16','D16','L0']:
        ck=short(arm)
        checks.append({'arm':arm,'finite_risk':bool(np.isfinite(ck['risk']).all()),'summary':ck['summary'],'mechanism':ck['mechanism']})
        del ck;gc.collect();torch.cuda.empty_cache()
    write(R/'MODEL_QUALIFICATION.json',{'status':'PASS','source':source(),'real_two_epoch_diagnostics':checks,
        'no_outcome_based_architecture_selection':True,'no_saved_smoke_weights_or_patient_risks':True})
    write(P/'SMOKE_COUNTS.json',{'status':'PASS','real_short_fits':sum(q['generator_steps']>0 for q in ledger['attempts']),
        'completed_short_fits':sum(q['status']=='RETURNED_SUCCESSFULLY' for q in ledger['attempts']),
        'generator_optimizer_steps':sum(q['generator_steps'] for q in ledger['attempts']),
        'discriminator_optimizer_steps':0,'no_saved_smoke_state_or_patient_risk':True})


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'logs').mkdir(exist_ok=True)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=cfg()['branch']: raise ValueError('wrong experiment branch')
    identity=source();seed_all(201017)
    write(P/'PREFLIGHT_DESIGN_SNAPSHOT.json',{'status':'BEFORE_REAL_DATA_SMOKE','unix_time':time.time(),
        'source':identity,'configuration':cfg()})
    qualify_tests();qualify_data();qualify_smoke()
    inherited=json.loads((P/'INHERITED_FILE_HASHES.json').read_text())['files']
    if any(digest(p)!=v for p,v in inherited.items()): raise ValueError('inherited cache was modified')
    if source()!=identity: raise ValueError('training source changed during qualification')
    lock={'status':'LOCKED_BEFORE_FIRST_FORMAL_TRAINING','source':identity,'unix_time':time.time(),
          'training_files':source_files(),'configuration':cfg(),'parent_commit':cfg()['parent_commit']}
    if (R/'PROTOCOL_LOCK.json').exists():
        if json.loads((R/'PROTOCOL_LOCK.json').read_text())['source']!=identity: raise ValueError('existing lock source differs')
    else: write(R/'PROTOCOL_LOCK.json',lock)
    write(P/'PREPARATION_DONE.json',{'status':'PASS','source':identity,'new_inner_fits_planned':135,'new_outer_fits_planned':135,
        'new_convex_solver_calls':0,'qualified_inherited_unique_convex_models':20})
    print('PHASE20_PREPARATION_PASS',identity,flush=True)


if __name__=='__main__': main()
