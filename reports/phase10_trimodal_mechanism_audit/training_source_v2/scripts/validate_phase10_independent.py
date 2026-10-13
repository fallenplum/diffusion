"""Read private OOF/manifests/checkpoints, never trust summary CSV metrics."""
import argparse,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.phase10_trimodal.reference import components,uno_weights,metrics

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def validate(args):
    root=Path(__file__).resolve().parents[1];report=root/'reports/phase10_trimodal_mechanism_audit';report.mkdir(exist_ok=True,parents=True)
    d=dict(np.load(args.data,allow_pickle=True));out=Path(args.output);cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());folds=d['folds'];time=d['DSS_time'];event=d['DSS_event'];rows=[];fail=[];risks={};required=[a for a in cfg['arms'] if a.startswith('A')] if args.baseline else cfg['arms']
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    surv=lambda ev,t:np.array(list(zip(ev.astype(bool),t)),dtype=[('event','?'),('time','<f8')])
    weights=np.zeros(len(time))
    for fold in cfg['folds']:
        tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);weights[te]=uno_weights(time[tr],event[tr],time[te],event[te],cfg['tau_months'])
    for arm in required:
        for seed in cfg['seeds']:
            risk=np.full(len(time),np.nan)
            for fold in cfg['folds']:
                cp=out/'cells'/f'{arm}_seed{seed}_fold{fold}.json';pred=out/'oof'/f'{arm}_seed{seed}_fold{fold}.npz';mp=out/'manifests'/f'{arm}_seed{seed}_fold{fold}.json'
                try:
                    cell=json.loads(cp.read_text());m=json.loads(mp.read_text());z=dict(np.load(pred));te=z['test_indices'];tr=np.asarray(m['outer_train'],int);it=set(m['inner_train']);iv=set(m['inner_val']);expected=np.flatnonzero(folds==fold)
                    if not np.array_equal(te,expected) or set(tr)&set(te) or it&iv or it|iv!=set(tr):raise ValueError('split boundary failure')
                    if not np.isfinite(z['risk']).all() or cell['status']!='COMPLETE' or cell['prediction_sha256']!=digest(pred) or cell['private_manifest_sha256']!=digest(mp):raise ValueError('completion/hash failure')
                    if m['preprocess_inner']['fit_hash']!=hashlib.sha256(np.asarray(m['inner_train'],np.int64).tobytes()).hexdigest() or m['preprocess_outer']['fit_hash']!=hashlib.sha256(tr.tobytes()).hexdigest():raise ValueError('fit provenance failure')
                    if cell.get('outer_test_used_for_selection') is not False:raise ValueError('selection boundary failure')
                    for role in ['teacher_inner','teacher_outer']:
                        for ta in m.get(role) or []:
                            hold=set(ta['heldout_indices']);tf=set(ta['fit_indices']);si=set(ta['selection_fit']);sv=set(ta['selection_validation']);allowed=it if role=='teacher_inner' else set(tr)
                            if hold&(tf|si|sv) or si&sv or si|sv!=tf or tf|hold!=allowed:raise ValueError('teacher leakage')
                    if cell['actual_epochs'] and cell['actual_epochs']!=cell['selected_epoch']:raise ValueError('refit schedule mismatch')
                    ck=out/'checkpoints'/f'{arm}_seed{seed}_fold{fold}.pt'
                    if cell.get('checkpoint_sha256') and arm!='D0' and digest(ck)!=cell['checkpoint_sha256']:raise ValueError('checkpoint hash failure')
                    risk[te]=z['risk'];hn,hd=components(time[te],event[te],risk[te]);un,ud=components(time[te],event[te],risk[te],weights[te])
                    href=concordance_index_censored(event[te].astype(bool),time[te],risk[te])[0];uref=concordance_index_ipcw(surv(event[tr],time[tr]),surv(event[te],time[te]),risk[te],tau=60)[0]
                    if abs(hn/hd-href)>1e-10 or abs(un/ud-uref)>1e-10:raise ValueError('independent library metric disagreement')
                    rows.append({'arm':arm,'seed':seed,'fold':fold,'Harrell':hn/hd,'Uno60':un/ud,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'train_n':len(tr),'test_n':len(te),'train_events':int(event[tr].sum()),'test_events':int(event[te].sum()),'parameters':cell['total_parameters'],'epoch':cell['selected_epoch']})
                except Exception as ex:fail.append({'arm':arm,'seed':seed,'fold':fold,'error':str(ex)})
            if np.isfinite(risk).all():risks[arm,seed]=risk
    output={'status':'FAIL' if fail else 'PASS','scope':'baselines' if args.baseline else 'full matrix','cells_verified':len(rows),'expected_cells':len(required)*15,'failures':fail,'independent_library':'scikit-survival 0.23.1','metric_scope':'fold-local Harrell/Uno numerator-denominator; pooled sensitivity separately','dataset_sha256':digest(args.data)}
    pd.DataFrame(rows).to_csv(report/('baseline_fold_metrics.csv' if args.baseline else 'fold_metrics.csv'),index=False)
    pd.DataFrame(fail,columns=['arm','seed','fold','error']).to_csv(report/'FAILURES.csv',index=False)
    write(report/('baseline_independent_validation.json' if args.baseline else 'independent_validation.json'),output)
    if args.baseline and not fail:
        sys.path.insert(0,str(root/'scripts'));from run_phase10_trimodal import sources_hash
        write(out/'baseline_gate.json',{'status':'PASS','source_hash':sources_hash(),'evidence_sha256':digest(report/'baseline_independent_validation.json'),'cells':len(rows)})
    if fail:print(json.dumps(output));return 2
    if args.baseline:print(json.dumps({k:v for k,v in output.items() if k!='failures'}));return 0
    # Inner-selected C+P comparator, selected before reading any test metric.
    for seed in cfg['seeds']:
        risk=np.full(len(time),np.nan);choices=[]
        for fold in cfg['folds']:
            cells={a:json.loads((out/'cells'/f'{a}_seed{seed}_fold{fold}.json').read_text()) for a in ['A3','A4','A5']}
            best=max(cells,key=lambda a:cells[a]['selected_inner_Harrell']);ix=folds==fold;risk[ix]=risks[best,seed][ix];choices.append({'seed':seed,'fold':fold,'arm':best,'inner_Harrell':cells[best]['selected_inner_Harrell']})
        risks['CP_selected',seed]=risk
        pd.DataFrame(choices).to_csv(report/f'cp_selection_seed{seed}.csv',index=False)
    seedrows=[]
    for (arm,seed),risk in risks.items():
        m=metrics(time,event,risk,folds,weights);hn,hd=components(time,event,risk);un,ud=components(time,event,risk,weights)
        seedrows.append({'arm':arm,'seed':seed,**m,'pooled_Harrell':hn/hd,'pooled_Uno60':un/ud})
    pd.DataFrame(seedrows).to_csv(report/'seed_oof_metrics.csv',index=False)
    families=[('primary',cfg['primary_comparisons']),('secondary',cfg['secondary_comparisons'])];rng=np.random.default_rng(cfg['bootstrap']['seed']);draws=rng.multinomial(len(time),np.full(len(time),1/len(time)),size=cfg['bootstrap']['draws']);paired=[]
    # Cached pair credits allow exact multinomial duplicate weighting without
    # treating seed predictions as independent patients.
    from src.phase10_trimodal.reference import pairs
    pi,pj=pairs(time,event);keep=folds[pi]==folds[pj];pi,pj=pi[keep],pj[keep]
    kred=np.stack([np.where(np.abs(r[pi]-r[pj])<=1e-8,.5,r[pi]>r[pj]) for r in risks.values()],axis=1)
    boot_values={key:{'Harrell':[],'Uno60':[]} for key in risks}
    for start in range(0,len(draws),100):
        q=(draws[start:start+100,pi]*draws[start:start+100,pj]).astype(float);hd=q.sum(1);qu=q*weights[pi];ud=qu.sum(1)
        hv=(q@kred)/hd[:,None];uv=(qu@kred)/ud[:,None]
        for col,key in enumerate(risks):boot_values[key]['Harrell'].extend(hv[:,col]);boot_values[key]['Uno60'].extend(uv[:,col])
    for family,comparisons in families:
        for a,b in comparisons:
            for metric in ['Harrell','Uno60']:
                deltas=np.array([metrics(time,event,risks[a,s],folds,weights)[metric]-metrics(time,event,risks[b,s],folds,weights)[metric] for s in cfg['seeds']]);boot=np.mean([np.asarray(boot_values[a,s][metric])-np.asarray(boot_values[b,s][metric]) for s in cfg['seeds']],axis=0)
                lo,hi=np.nanquantile(boot,[.025,.975]);alpha=.05/(len(cfg['primary_comparisons'])*2) if family=='primary' else .05;simlo,simhi=np.nanquantile(boot,[alpha/2,1-alpha/2])
                paired.append({'family':family,'comparison':a+'-'+b,'metric':metric,'mean_delta':deltas.mean(),'seed31_delta':deltas[0],'seed47_delta':deltas[1],'seed83_delta':deltas[2],'positive_seeds':int((deltas>0).sum()),'CI95_low':lo,'CI95_high':hi,'simultaneous_low':simlo,'simultaneous_high':simhi,'bootstrap_draws':len(boot)})
    pd.DataFrame(paired).to_csv(report/'paired_comparisons.csv',index=False);pd.DataFrame(paired).to_csv(report/'bootstrap_intervals.csv',index=False)
    write(report/'bootstrap_provenance.json',{'scheme':cfg['bootstrap'],'counts_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),'shared_counts_all_models_seeds':True,'refit_variance_included':False})
    print(json.dumps(output));return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/root/phase7a_foundation_headroom_v1/private_dataset.npz');ap.add_argument('--output',default='/root/phase10_trimodal_private/formal');ap.add_argument('--baseline',action='store_true');raise SystemExit(validate(ap.parse_args()))
