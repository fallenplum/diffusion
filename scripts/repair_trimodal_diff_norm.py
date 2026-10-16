"""Append-only repair for an omitted official Differential RMSNorm scale."""
import copy,json,hashlib,subprocess,sys,time,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
from src.trimodal_interaction.data import Data,P,R,ROOT,cfg,source,write,digest
from src.trimodal_interaction.model import Model
from src.trimodal_interaction import train

CONFIG=ROOT/'configs/trimodal_diff_norm_repair.json';PRIVATE=P/'a3f_norm_repair';REPORT=R/'a3f_norm_repair'

def repair_source():return hashlib.sha256((source()+digest(CONFIG)+digest(Path(__file__))).encode()).hexdigest()

def build(data,arm,w,config):
    assert arm=='A3';model=Model(data,arm,w,config)
    model.interaction.diff_norm=torch.nn.RMSNorm(32,eps=1e-5,elementwise_affine=True)
    assert model.interaction.diff_norm.weight.numel()==32
    return model

def fit(seed,fold):
    config=copy.deepcopy(cfg());config['training']['max_epochs']=160;train.cfg=lambda:config;train.Model=build
    result=train.train_cell('A3',seed,fold);result['summary']['arm']='A3F';result['implementation_source']=repair_source()
    ck=PRIVATE/'checkpoints'/f'A3F_seed{seed}_fold{fold}.pt';ck.parent.mkdir(parents=True,exist_ok=True);tmp=ck.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(ck)
    write(PRIVATE/'cells'/f'A3F_seed{seed}_fold{fold}.json',{'source':source(),'implementation_source':repair_source(),'checkpoint_sha256':digest(ck),'summary':result['summary']})

def evaluator():
    spec=importlib.util.spec_from_file_location('independent_trimodal_metrics',ROOT/'scripts/validate_trimodal_interaction.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def validate():
    config=copy.deepcopy(cfg());config['training']['max_epochs']=160;ev=evaluator();independent=ev.independent_functions();rows=[];summaries=[];audits=[];curves=[]
    for fold in config['folds']:
        for seed in config['seeds']:
            label=f'A3F_seed{seed}_fold{fold}';data=Data(fold,'inner',seed);ck=PRIVATE/'checkpoints'/f'{label}.pt';receipt=json.loads((PRIVATE/'cells'/f'{label}.json').read_text())
            assert receipt['implementation_source']==repair_source() and digest(ck)==receipt['checkpoint_sha256']
            q=torch.load(ck,map_location='cpu',weights_only=False);assert q['implementation_source']==repair_source() and q['source']==source()
            assert np.array_equal(q['fit'],data.fit) and np.array_equal(q['valid'],data.valid)
            ev.validate_history(q,config)
            model=build(data,'A3',np.zeros(data.f['X'].shape[1]),config).cuda();model.load_state_dict(q['state']);model.eval()
            with torch.no_grad():risk=train.forward(model,data,data.valid,16).cpu().numpy()
            error=float(abs(risk-q['risk']).max());assert error<2e-6
            loss=ev.independent_efron(risk,data.t[data.valid],data.e[data.valid]);assert abs(loss-q['summary']['validation_loss'])<2e-6
            assert 'interaction.diff_norm.weight' in q['state'] and q['state']['interaction.diff_norm.weight'].numel()==32
            rows.append(ev.metric_rows_for_risk(independent,risk,data,'A3F',seed,seed));summaries.append(q['summary']);audits.append({'seed':seed,'fold':fold,'risk_replay_error':error,'RMS_affine_scale_max_change':float(abs(q['state']['interaction.diff_norm.weight']-1).max()),'checkpoint_sha256':digest(ck)})
            curves += [{'arm':'A3F','seed':seed,'fold':fold,**row} for row in q['history']]
    frame=pd.DataFrame(rows);frame.to_csv(REPORT/'FOLD_METRICS.csv',index=False);ss=[]
    for seed,g in frame.groupby('seed'):
        ss.append({'arm':'A3F','seed':seed,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    seeds=pd.DataFrame(ss);seeds.to_csv(REPORT/'SEED_INNER_METRICS.csv',index=False);a=pd.DataFrame(summaries)
    summary={'arm':'A3F','role':'inner_feasibility_NOT_OOF','Harrell_mean_over_seed_algorithms':seeds.Harrell.mean(),'Uno60_mean_over_seed_algorithms':seeds.Uno60.mean(),'selected_epoch_median':a.selected_epoch.median(),'actual_epoch_median':a.actual_epochs.median(),'hit_maximum_fraction':a.hit_maximum.mean(),'parameters_registered_mean':a.parameters_registered.mean(),'parameters_gradient_active_mean':a.parameters_gradient_active.mean(),'peak_GPU_MiB_max':a.peak_GPU_MB.max(),'seconds_mean_concurrent_run':a.seconds.mean(),'real_training_cells_this_stage':len(a),'selected_model_cells':len(a),'evaluation_cells':len(a)}
    summary.update(Harrell_seed_min=seeds.Harrell.min(),Harrell_seed_max=seeds.Harrell.max(),Uno60_seed_min=seeds.Uno60.min(),Uno60_seed_max=seeds.Uno60.max())
    pd.DataFrame([summary]).to_csv(REPORT/'ARM_SUMMARY.csv',index=False);pd.DataFrame(audits).to_csv(REPORT/'CHECKPOINT_AUDIT.csv',index=False);pd.DataFrame(curves).to_csv(REPORT/'TRAINING_CURVES.csv',index=False)
    other=pd.read_csv(R/'budget160/SEED_INNER_METRICS.csv');other=other[other.arm=='A5'].set_index('seed');af=pd.read_csv(R/'budget160/FOLD_METRICS.csv');af=af[af.arm=='A5'].set_index(['seed','fold']);bf=frame.set_index(['seed','fold']);contrasts=[];gate=None
    for metric in ['Harrell','Uno60']:
        sd=other[metric]-seeds.set_index('seed')[metric];cd=af[metric]-bf[metric]
        row={'comparison':'A5-A3F','family':'primary_repaired','metric':metric,'mean_seed_pair_weighted_delta':float(sd.mean()),'positive_seed_algorithms':int((sd>0).sum()),'positive_inner_cells':int((cd>0).sum()),'inner_cells':len(cd),'positive_inner_cell_fraction':float((cd>0).mean()),'mean_unweighted_cell_delta':float(cd.mean()),'minimum_cell_delta':float(cd.min()),'maximum_cell_delta':float(cd.max()),'scope':'descriptive overlapping inner patients; bug-corrected norm, not an independently selected winner'}
        row.update({'seed_'+str(s)+'_pair_weighted_delta':float(sd.loc[s]) for s in config['seeds']});contrasts.append(row)
        if metric=='Harrell':gate={'comparison':'A5-A3F','each_seed_delta_above_locked_threshold':bool((sd>0).all()),'positive_cells':int((cd>0).sum()),'required_fraction':2/3,'pass':bool((sd>0).all() and (cd>0).mean()>=2/3)}
    pd.DataFrame(contrasts).to_csv(REPORT/'PAIRED_INNER_DIAGNOSTICS.csv',index=False)
    counts={'real_neural_training_cells':9,'real_baseline_only_convex_fits':0,'neural_initialization_ridge_solver_calls':9,'total_ridge_solver_calls':9,'real_completed_model_cells':9,'outer_fits':0,'outer_prediction_vectors':0,'patient_OOF_files':0,'checkpoint_replays':9};write(REPORT/'EXECUTION_COUNTS.json',counts)
    budget_ok=float(a.hit_maximum.mean())<=.2
    write(REPORT/'STAGE_DECISION.json',{'decision':'BUDGET_UNRESOLVED' if not budget_ok else 'PASS_A3F_COMPARISON' if gate['pass'] else 'REPAIRED_COMPARISON_NOT_SUPPORTED','budget_gate_pass':budget_ok,'maximum_epoch_fraction':float(a.hit_maximum.mean()),'primary_comparisons':[gate],'scope':'one repaired primary comparator, not whole model success'})
    write(REPORT/'VALIDATION.json',{'status':'PASS','source':source(),'implementation_source':repair_source(),'cells':9,'official_RMSNorm_affine_parameters':32,'actual_RMSNorm_affine_scale_updated':all(q['RMS_affine_scale_max_change']>0 for q in audits),'maximum_checkpoint_risk_error':max(q['risk_replay_error'] for q in audits),'maximum_sksurv_difference':float(frame.maximum_sksurv_error.max()),'all_reference_DSS_data_preprocessing_unchanged':True,'no_outer_evaluation':True,'no_confidence_intervals_or_p_values':True})
    print('A3F_VALIDATION_PASS',summary,gate,flush=True)

def main():
    if len(sys.argv)>1:return fit(int(sys.argv[1]),int(sys.argv[2]))
    assert json.loads((P/'budget160/DONE.json').read_text())['source']==source()
    REPORT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(exist_ok=True);lock={'implementation_source':repair_source(),'original_source':source(),'config':json.loads(CONFIG.read_text())}
    if (PRIVATE/'LOCK.json').exists():assert json.loads((PRIVATE/'LOCK.json').read_text())==lock
    write(PRIVATE/'LOCK.json',lock);write(REPORT/'PROTOCOL_LOCK.json',lock)
    jobs=[(s,f) for f in cfg()['folds'] for s in cfg()['seeds']];pending=[]
    for seed,fold in jobs:
        p=PRIVATE/'cells'/f'A3F_seed{seed}_fold{fold}.json'
        if not p.exists():pending.append((seed,fold))
        else:assert json.loads(p.read_text())['implementation_source']==repair_source()
    active=[]
    while pending or active:
        while pending and len(active)<3:
            seed,fold=pending.pop(0);h=(PRIVATE/f'seed{seed}_fold{fold}.log').open('w');p=subprocess.Popen([sys.executable,__file__,str(seed),str(fold)],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT);active.append((p,h))
        for item in list(active):
            p,h=item;code=p.poll()
            if code is None:continue
            h.close();active.remove(item)
            if code:
                for pp,hh in active:pp.terminate();hh.close()
                raise SystemExit('A3F_FIT_FAILED')
        time.sleep(1)
    while not (R/'budget160/VALIDATION.json').exists():time.sleep(3)
    assert json.loads((R/'budget160/VALIDATION.json').read_text())['status']=='PASS'
    validate();write(PRIVATE/'DONE.json',{'status':'PASS','source':source(),'implementation_source':repair_source(),'real_neural_fits':9,'ridge_initializations':9,'outer_fits':0})

if __name__=='__main__':main()
