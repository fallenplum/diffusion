"""Unify all50 arms and crosscheck every final fold against scikit-survival."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase13_independent import raw_endpoints
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from src.phase16_semantic.omics import *
from run_phase16_mil_two_stage import LP

def main():
    for p in [P/'EVALUATION_DONE.json',MP/'EVALUATION_DONE.json',LP/'EVALUATION_DONE.json']:
        if json.loads(p.read_text())['status']!='PASS':raise ValueError('independent stage incomplete')
    g5=R/'G5';mil=R/'MIL';a=pd.read_csv(g5/'UNIFIED_FOLD_METRICS.csv');b=pd.read_csv(mil/'FOLD_METRICS.csv');b=b[b.arm.str.startswith('MIL_TWO_STAGE')];fold=pd.concat([a,b],ignore_index=True);fold.to_csv(R/'UNIFIED_FOLD_METRICS.csv',index=False);s=pd.read_csv(g5/'UNIFIED_SEED_OOF_METRICS.csv');ss=pd.read_csv(mil/'SEED_OOF_METRICS.csv');ss=ss[ss.arm.str.startswith('MIL_TWO_STAGE')];seed=pd.concat([s,ss],ignore_index=True);seed.to_csv(R/'UNIFIED_SEED_OOF_METRICS.csv',index=False)
    models=pd.concat([pd.read_csv(R/'MODEL_SUMMARY.csv'),pd.read_csv(g5/'MODEL_SUMMARY.csv'),pd.read_csv(mil/'MODEL_SUMMARY.csv')],ignore_index=True);models.to_csv(R/'UNIFIED_MODEL_SUMMARY.csv',index=False);summary=seed.groupby(['endpoint','arm'])[['Harrell','Uno60']].mean().add_suffix('_mean').join(seed.groupby(['endpoint','arm'])[['Harrell','Uno60']].std().add_suffix('_seed_sd')).reset_index();summary.to_csv(R/'AGGREGATE_METRICS.csv',index=False)
    if len(fold)!=1500 or len(seed)!=300 or len(models)!=750 or seed.arm.nunique()!=50:raise ValueError('final arm inventory/count mismatch')
    d=parent.load();ev=raw_endpoints(d,{});riskcache={};maximum=0
    for row in fold.itertuples():
        key=(row.arm,int(row.seed),int(row.fold))
        if key not in riskcache:
            directory=MP if row.arm.startswith('G5_') else LP if row.arm.startswith('MIL_TWO_STAGE') else P if row.arm in arms() else base.P;c=json.loads((directory/'cells'/f'{label(row.arm,row.seed,row.fold)}.json').read_text())
            if sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('final checkpoint integrity')
            z=torch.load(c['artifact'],map_location='cpu',weights_only=False);riskcache[key]=(np.asarray(z['test_indices']),np.asarray(z['risk']))
        ids,rr=riskcache[key];t,e,valid=[ev[row.endpoint][k] for k in ['time','event','valid']];use=valid[ids];te=ids[use];tr=np.flatnonzero(valid&(d['folds']!=row.fold));r=rr[use];h=concordance_index_censored(e[te].astype(bool),t[te],r)[0];u=concordance_index_ipcw(Surv.from_arrays(e[tr].astype(bool),t[tr]),Surv.from_arrays(e[te].astype(bool),t[te]),r,tau=60)[0];err=max(abs(h-row.Harrell),abs(u-row.Uno60));maximum=max(maximum,err)
        if err>1e-10:raise ValueError('final metrics differ from scikit-survival')
    # Missing-block descriptive sensitivity; no platform identity is inferred.
    missing=np.isnan(d['raw_rna']).any(1);sub=[]
    from validate_phase11_independent import components,ipcw
    for ep in ['DSS','PFI']:
        t,e,valid=[ev[ep][k] for k in ['time','event','valid']]
        for arm in seed.arm.unique():
            for sd in [31,47,83]:
                for label_group,condition in [('missing3024',missing),('complete',~missing)]:
                    hn=hd=un=ud=0;events=number=0
                    for f in [1,2,3,4,5]:
                        ids,rr=riskcache[arm,sd,f];keep=valid[ids]&condition[ids];te=ids[keep];tr=np.flatnonzero(valid&(d['folds']!=f));iw=ipcw(t[tr],e[tr],t[te],e[te],60);a,b=components(t[te],e[te],rr[keep]);c,dd=components(t[te],e[te],rr[keep],iw);hn+=a;hd+=b;un+=c;ud+=dd;events+=int(e[te].sum());number+=len(te)
                    sub.append({'endpoint':ep,'arm':arm,'seed':sd,'missing_group':label_group,'n':number,'events':events,'Harrell':hn/hd if hd else None,'Uno60':un/ud if ud else None,'scope':'descriptive missing-pattern strata;not verified sequencing platform'})
    pd.DataFrame(sub).to_csv(R/'MISSING_BLOCK_SENSITIVITY.csv',index=False);write(R/'FINAL_INDEPENDENT_AUDIT.json',{'status':'PASS','arms':50,'outer_fold_evaluations':750,'actual_new_outer_fits_cumulative':690,'historical_references':60,'new_outer_fits_this_authorized_batch':540,'all_final_metric_records':1500,'all_final_sksurv_max_error':maximum,'patient_level_data_public':False,'G4':'BLOCKED_RAW_WSI_QA'});print('FINAL_TABLES_PASS',flush=True)
if __name__=='__main__':main()
