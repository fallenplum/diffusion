"""Staged decisions read INNER checkpoints only; no outer predictions accessed."""
import sys,argparse,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
from src.phase17r_ssd.data import P,R,cfg,source,write,old
from scripts.run_phase17r import stem
from scripts.validate_phase17_independent import components,weights

def read(arm,seed,fold,scale=1.):
    p=P/'inner'/(stem(arm,seed,fold,scale)+'.pt');z=torch.load(p,map_location='cpu',weights_only=False)
    assert z['role']=='inner' and z['source']==source();return z

def records(z,d):
    ix=z['validation_or_test'];tr=z['fit'];ev=old.parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];w=weights(t[tr],e[tr],t[ix],e[ix])
    out=[]
    for condition,key in [('matched','risk'),('wrong','wrong_risk'),('SSM_off','SSM_off_risk'),('gamma_one','gamma_one_risk')]:
        hn,hd,*_=components(t[ix],e[ix],z[key],np.ones(len(ix)));un,ud,*_=components(t[ix],e[ix],z[key],w)
        out.append({'arm':z['arm'],'seed':z['seed'],'fold':z['fold'],'scale':z['scale'],'condition':condition,'Harrell':hn/hd,'Uno60':un/ud,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'selected_epoch':z['selected_epoch'],'best_validation_EMA':z['best_validation_EMA']})
    return out

def gate(arm,comparators,table,mechanisms):
    tc=cfg()['R3_gate'];q=pd.DataFrame([x for x in mechanisms if x['arm']==arm]);reasons=[]
    medstate=float(q.h_relative_L2.median());medrisk=float(q.wrong_context_relative_risk_drift.median());cells=int((q.h_relative_L2>=.001).sum())
    if medstate<tc['state_relative_L2_median_at_least']:reasons.append('state median below threshold')
    if cells<tc['state_relative_L2_at_least_0.001_cells']:reasons.append('state effect not repeated in12cells')
    if medrisk<tc['wrong_context_relative_risk_drift_median_at_least']:reasons.append('risk insensitive to patient context')
    checks=[];frame=pd.DataFrame(table)
    matched=frame[(frame.arm==arm)&(frame.condition=='matched')]
    for comparator in comparators+['WRONG_CONTEXT']:
        ref=frame[(frame.arm==arm)&(frame.condition=='wrong')] if comparator=='WRONG_CONTEXT' else frame[(frame.arm==comparator)&(frame.condition=='matched')]
        assert len(matched)==len(ref)==15
        mm=matched.merge(ref,on=['seed','fold'],suffixes=('_a','_b'));seed_gains=[]
        for s,g in mm.groupby('seed'):seed_gains.append(float(g.H_num_a.sum()/g.H_den_a.sum()-g.H_num_b.sum()/g.H_den_b.sum()))
        gain=float(np.mean(seed_gains));positive_seeds=int(np.sum(np.asarray(seed_gains)>0));positive_cells=int((mm.Harrell_a>mm.Harrell_b).sum())
        ok=gain>=tc['mean_inner_Harrell_gain_at_least'] and positive_seeds>=tc['positive_seeds_at_least'] and positive_cells>=tc['positive_cells_at_least']
        if not ok:reasons.append('predictive gate failed against '+comparator)
        checks.append({'comparator':comparator,'mean_inner_Harrell_gain':gain,'seed_gains':seed_gains,'positive_seeds':positive_seeds,'positive_cells':positive_cells,'pass':ok})
    return {'run':not reasons,'state_relative_L2_median':medstate,'state_effect_cells':cells,'relative_risk_drift_median':medrisk,'predictive_checks':checks,'reasons':reasons,'scope':'overlapping inner-validation engineering gate, not external significance'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['R1','R2'],required=True);a=p.parse_args();d=old.parent.load();cs=cfg();table=[];mechanisms=[];r1=[]
    for arm in cs['R1']:
        for s in cs['seeds']:
            for f in cs['folds']:
                z=read(arm,s,f);r1.append(z);table+=records(z,d);mechanisms.append({'arm':arm,'seed':s,'fold':f,'scale':1.,**z['mechanism']})
    q=pd.DataFrame([x for x in mechanisms if x['arm']=='R1_A8_noDirect']);trigger=cs['R2_trigger'];weak=bool(q.delta_over_base_RMS.median()<.001 or q.B_condition_over_base_RMS.median()<.001 or q.h_relative_L2.median()<.001)
    run2=bool(q.g_relative_L2.median()>=trigger['g_relative_L2_median_at_least'] and weak)
    r1gate=gate('R1_A8_noDirect',cs['R3_gate']['comparators_R1'],table,mechanisms)
    selected=[];candidates=[];r2gate=None
    if a.stage=='R2':
        assert run2
        for arm in cs['R2']['models']:
            for s in cs['seeds']:
                for f in cs['folds']:
                    zz=[read(arm,s,f,scale) for scale in cs['R2']['scales']];chosen=min(zz,key=lambda z:(z['best_validation_EMA'],z['scale']))
                    selected.append({'arm':arm,'seed':s,'fold':f,'scale':chosen['scale'],'selected_epoch':chosen['selected_epoch']})
                    for z in zz:candidates.append({'arm':arm,'seed':s,'fold':f,'scale':z['scale'],'validation_EMA':z['best_validation_EMA'],'selected_epoch':z['selected_epoch'],'selected':z['scale']==chosen['scale']})
                    table+=records(chosen,d);mechanisms.append({'arm':arm,'seed':s,'fold':f,'scale':chosen['scale'],**chosen['mechanism']})
        r2gate=gate('R2_A8',cs['R3_gate']['comparators_R2'],table,mechanisms)
    final=not run2 or a.stage=='R2';r3=r2gate if r2gate is not None else r1gate
    result={'source':source(),'decision_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':a.stage,'R1':{'inner_cells':60,'R3_gate':r1gate},'R2':{'run':run2,'trigger_population':'R1_A8_noDirect inner-train first32','g_relative_L2_median':float(q.g_relative_L2.median()),'delta_over_base_RMS_median':float(q.delta_over_base_RMS.median()),'B_condition_over_base_RMS_median':float(q.B_condition_over_base_RMS.median()),'h_relative_L2_median':float(q.h_relative_L2.median()),'selected':selected,'R3_gate':r2gate},'R3':r3,'R3_base_arm':'R2_A8' if r2gate is not None else 'R1_A8_noDirect','architecture_decisions_locked_before_outer':final,'outer_files_present_at_decision':len(list((P/'outer').glob('*.pt')))}
    assert result['outer_files_present_at_decision']==0,'outer data existed before architecture lock'
    pd.DataFrame(table).to_csv(R/'INNER_METRICS.csv',index=False);pd.DataFrame(mechanisms).to_csv(R/'INNER_MECHANISM.csv',index=False)
    if candidates:pd.DataFrame(candidates).to_csv(R/'R2_SELECTION.csv',index=False)
    write(R/('R1_DECISION.json' if a.stage=='R1' else 'R2_DECISION.json'),result);write(R/'STAGE_DECISION.json',result)
    print(json.dumps({'R2_run':run2,'R3_run':r3['run'],'R3_reasons':r3['reasons'],'architecture_lock':final}),flush=True)

if __name__=='__main__':main()
