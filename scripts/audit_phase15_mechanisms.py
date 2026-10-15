"""Post-fit diagnostics only; no outer error group feeds back into training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase15_complement.core import *
from evaluate_phase15 import independent_cs
from validate_phase11_independent import pairs,ipcw


def neff(values):
    values=np.asarray(values,float)
    return float(values.sum()**2/max(values@values,1e-30))


def main():
    d=parent.load();endpoint=parent.endpoints(d);supervision=[];shift=[];teacher_checks=[];rejected=0
    for ep in cfg()['endpoints']:
        t,e,valid=[endpoint[ep][k] for k in ['time','event','valid']]
        for seed in cfg()['seeds']:
            for fold in cfg()['folds']:
                for role in ['inner','outer']:
                    a=torch.load(PRIVATE/'anchors'/f'{ep}_seed{seed}_fold{fold}_{role}.pt',map_location='cpu',weights_only=False);fit,test=a['fit'],a['test'];i,j=honest_pairs(a,t,e)[:2];margin=a['oof'][fit[i]]-a['oof'][fit[j]]
                    for kind in ['ALL','SOFT','HARD']:
                        ww=weights(kind,margin,i,seed);mass=np.asarray([ww[i==v].sum() for v in np.unique(i)]);supervision.append({'endpoint':ep,'seed':seed,'fold':fold,'role':role,'kind':kind,'patients':len(fit),'events':int(e[fit].sum()),'eligible_pairs':int((ww>0).sum()),'contributing_events':int((mass>0).sum()),'event_mass_Neff':neff(mass),'maximum_event_mass_fraction':float(mass.max()/mass.sum())})
                    full=independent_cs(d,fit)[test]@a['w'];pi,pj=pairs(t[test],e[test]);fullmargin=full[pi]-full[pj];fc=np.where(abs(fullmargin)<=1e-8,.5,fullmargin>0);bad=fc<1
                    # Fixed outer train censoring weights for outer held-out diagnostics.
                    w=ipcw(t[fit],e[fit],t[test],e[test],60)[pi] if role=='outer' else np.ones(len(pi))
                    for k,teacher in enumerate(a['teachers']):
                        xx=independent_cs(d,teacher['train']);pred=xx[test]@teacher['w'];dd=pred[pi]-pred[pj];credit=np.where(abs(dd)<=1e-8,.5,dd>0);wrong=credit<1
                        for metric in (['Harrell','Uno60'] if role=='outer' else ['Harrell']):
                            ww=w if metric=='Uno60' else np.ones(len(pi));union=wrong|bad;inter=wrong&bad
                            shift.append({'endpoint':ep,'seed':seed,'fold':fold,'role':role,'teacher':k,'metric':metric,'weight_mass':float(ww.sum()),'teacher_correct_mass':float(ww@credit),'full_correct_mass':float(ww@fc),'wrong_intersection_mass':float(ww[inter].sum()),'wrong_union_mass':float(ww[union].sum()),'teacher_wrong_mass':float(ww[wrong].sum()),'full_wrong_mass':float(ww[bad].sum())})
                # Each teacher selection sees neither held patient's label.
                if ep=='DSS' and seed==31 and fold==1:
                    a=torch.load(PRIVATE/'anchors'/f'{ep}_seed{seed}_fold{fold}_inner.pt',map_location='cpu',weights_only=False)
                    for k,teacher in enumerate(a['teachers']):
                        train,held=teacher['train'],teacher['held'];tt=t.copy();ee=e.copy();tt[held]+=111;ee[held]=1-ee[held]
                        lam,scores=select_cs(d,tt,ee,train,seed+fold*100+k+1,2)
                        if lam!=teacher['lambda'] or scores!=teacher['selection']:raise RuntimeError('held pair label entered teacher selection')
                        xx,_=cs_features(d,train);w,_=fit_ridge(xx[train],tt[train],ee[train],lam)
                        if not np.array_equal(w,teacher['w']):raise RuntimeError('held label entered teacher model')
                        teacher_checks.append({'teacher':k,'held_labels_perturbed':len(held),'lambda_and_weights_bitwise_unchanged':True})
    pd.DataFrame(supervision).to_csv(REPORT/'SUPERVISION_EVENT_MASS.csv',index=False);pd.DataFrame(shift).to_csv(REPORT/'TEACHER_SHIFT.csv',index=False)
    # Intentional corruptions are in memory only, never mutate official files.
    fake={'fit':np.arange(6),'group':np.array([0,0,1,1,2,2]),'oof':np.arange(6,dtype=float),'teachers':[{'train':np.setdiff1d(np.arange(6),[k,k+1]),'held':np.array([k,k+1])} for k in [0,2,4]]}
    bad={**fake,'teachers':[dict(x) for x in fake['teachers']]};bad['teachers'][1]['train']=np.arange(6)
    try:honest_pairs(bad,np.arange(1,7),np.ones(6,int))
    except RuntimeError:rejected+=1
    s=pd.read_csv(REPORT/'SEED_OOF_METRICS.csv');f=pd.read_csv(REPORT/'REPAIR_DAMAGE.csv');maximum=0
    for row in f.itertuples():
        a=s[(s.endpoint==row.endpoint)&(s.gene==row.gene)&(s.seed==row.seed)&(s.model==row.model)][row.metric].iloc[0];b=s[(s.endpoint==row.endpoint)&(s.gene==row.gene)&(s.seed==row.seed)&(s.model=='CS')][row.metric].iloc[0];maximum=max(maximum,abs(row.net_delta-(a-b)))
    if maximum>1e-10:raise RuntimeError('repair-damage not equal net C-index difference')
    stability=[];fold_metrics=pd.read_csv(REPORT/'FOLD_METRICS.csv')
    for path in sorted((PRIVATE/'cells').glob('*.json')):
        c=json.loads(path.read_text());ep,gene,seed,fold=[c[k] for k in ['endpoint','gene','seed','fold']];a=torch.load(Path(c['teacher_anchor_paths']['inner']),map_location='cpu',weights_only=False);ev=endpoint[ep];test=a['test'];pi,pj=pairs(ev['time'][test],ev['event'][test]);dd=a['test_risk'][pi]-a['test_risk'][pj];inner_base=float(np.where(abs(dd)<=1e-8,.5,dd>0).mean())
        for model in ['JOINT','ALL','SOFT','SCALAR','RANDOM','WRONG']:
            one=next(row for row in c['selection'][model+'_rows'] if row['lambda']==1.)
            subset=fold_metrics[(fold_metrics.endpoint==ep)&(fold_metrics.gene==gene)&(fold_metrics.seed==seed)&(fold_metrics.fold==fold)]
            outer_base=float(subset[subset.model=='CS'].Harrell.iloc[0]);outer_fixed=float(subset[subset.model==model+'_FIXED1'].Harrell.iloc[0]);outer_delta=outer_fixed-outer_base
            stability.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'model':model,'selected_lambda':c['selection'][model],'inner_fixed1_delta':one['inner_Harrell']-inner_base,'outer_fixed1_delta':outer_delta,'inner_validation_events':int(ev['event'][test].sum()),'selected_OFF_despite_positive_outer_fixed1':c['selection'][model]=='OFF' and outer_delta>0,'scope':'diagnostic only;outer signs not used for selection'})
    pd.DataFrame(stability).to_csv(REPORT/'SELECTION_STABILITY.csv',index=False)
    result={'status':'PASS','actual_teacher_held_label_perturbations':teacher_checks,'event_mass_records':len(supervision),'teacher_shift_records':len(shift),'selection_stability_records':len(stability),'gain_damage_identity_max_error':maximum,'faults_rejected':rejected,'outer_errors_used_for_training':False,'scope':'post-fit exploratory diagnostics only;no model/config change'}
    write(REPORT/'MECHANISM_AUDIT.json',result);print(json.dumps(result))


if __name__=='__main__':main()
