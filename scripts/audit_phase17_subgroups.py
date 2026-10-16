"""Descriptive platform-proxy and conditional-module sensitivity; no tuning."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase17_independent import pairs,components,weights
from src.phase17_ssd.data import P,R,write,parent
from src.phase17_ssd.train import Data,seed_all
from src.phase17_ssd.model import ConditionalS6

def main():
    d=parent.load();ep=parent.endpoints(d)['DSS'];t,e=ep['time'],ep['event'];proxy=np.isnan(d['raw_rna']).sum(1)==3024;folds=d['folds'];oof=dict(np.load(P/'OOF_PRIVATE.npz'));rows=[]
    for key,risk in oof.items():
        arm,seed=key.rsplit('__seed',1)
        for group,valid in [('RNA_missing_block203',proxy),('RNA_other309',~proxy)]:
            for fold in range(1,6):
                tr=np.flatnonzero(folds!=fold);te=np.flatnonzero((folds==fold)&valid);w=weights(t[tr],e[tr],t[te],e[te]);hn,hd,*_=components(t[te],e[te],risk[te],np.ones(len(te)));un,ud,*_=components(t[te],e[te],risk[te],w)
                rows.append({'arm':arm,'seed':int(seed),'fold':fold,'group':group,'n':len(te),'events':int(e[te].sum()),'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'Harrell':hn/hd if hd else np.nan,'Uno60':un/ud if ud else np.nan,'meaning':'missing-pattern proxy;RNA sequencing platform identity not directly verified'})
    tab=pd.DataFrame(rows);tab.to_csv(R/'RNA_MISSING_PATTERN_METRICS.csv',index=False);summ=[]
    for (a,s,g),q in tab.groupby(['arm','seed','group']):summ.append({'arm':a,'seed':s,'group':g,'Harrell':q.H_num.sum()/q.H_den.sum(),'Uno60':q.U_num.sum()/q.U_den.sum(),'events':int(q.events.sum()),'patients':int(q.n.sum())})
    pd.DataFrame(summ).to_csv(R/'RNA_MISSING_PATTERN_SEED.csv',index=False)
    # Frozen predictions: quantitative contribution to ordering, not merely norm.
    offrisks={};offrows=[]
    for path in sorted((P/'outer').glob('A8_*.pt')):
        z=torch.load(path,map_location='cpu',weights_only=False);fold,seed=z['fold'],z['seed'];seed_all(seed+100*fold);ds=Data(fold,'outer',512,'A8',seed);model=ConditionalS6(ds.f['C'].shape[1],ds.f['membership'],'A8',.25).cuda();model.load_state_dict(z['state']);model.eval();te=ds.valid;tr=ds.fit;w=weights(t[tr],e[tr],t[te],e[te])
        cases=[('SSM_off',{'module_off':True},None),('Omics_direct_off',{},'direct'),('gate_constant09',{'gate_override':.9},None),('delta_off',{},'delta_up'),('B_condition_off',{},'B_up')]
        for label,kw,layer in cases:
            hook=getattr(model,layer).register_forward_hook(lambda m,a,o:torch.zeros_like(o)) if layer else None
            try:
                with torch.no_grad():risk=ds.forward(model,te,**kw).cpu().numpy()
            finally:
                if hook:hook.remove()
            hn,hd,*_=components(t[te],e[te],risk,np.ones(len(te)));un,ud,*_=components(t[te],e[te],risk,w)
            offrows.append({'intervention':label,'seed':seed,'fold':fold,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'Harrell':hn/hd,'Uno60':un/ud,'retrained':False})
            offrisks.setdefault(label+'__seed'+str(seed),np.full(512,np.nan))[te]=risk
        print('FROZEN_ORDERING_SENSITIVITY',seed,fold,flush=True)
    np.savez_compressed(P/'FROZEN_INTERVENTION_OOF_PRIVATE.npz',**offrisks);tab=pd.DataFrame(offrows);tab.to_csv(R/'FROZEN_INTERVENTION_FOLD_METRICS.csv',index=False);ss=[]
    for (name,seed),q in tab.groupby(['intervention','seed']):ss.append({'intervention':name,'seed':seed,'Harrell':q.H_num.sum()/q.H_den.sum(),'Uno60':q.U_num.sum()/q.U_den.sum(),'retrained':False})
    pd.DataFrame(ss).to_csv(R/'FROZEN_INTERVENTION_SEED_METRICS.csv',index=False)
    write(R/'SUBGROUP_AUDIT.json',{'status':'PASS','missing_block_patients':int(proxy.sum()),'comparison_patients':int((~proxy).sum()),'scope':'descriptive only,no subgroup model selection;frozen interventions not refitted ablations','new_model_fits':0})

if __name__=='__main__':main()
