"""Additional requested descriptive contrasts; do not change primary family."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.phase10_trimodal.reference import metrics,pairs,uno_weights

def main():
 root=Path(__file__).resolve().parents[1];report=root/'reports/phase10_trimodal_mechanism_audit';out=Path('/root/phase10_trimodal_private/formal');d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());comparisons=[('B0','A6'),('C0','A6'),('B0','A3'),('C0','A3'),('D1','A3'),('C4','A3'),('B3','A3')];extra=[('D1','D0'),('D2','D0'),('D3','D0')];risks={};w=np.zeros(512)
 for fold in cfg['folds']:
  tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);w[te]=uno_weights(d['DSS_time'][tr],d['DSS_event'][tr],d['DSS_time'][te],d['DSS_event'][te])
 for arm in sorted(set(x for pair in comparisons+extra for x in pair)):
  for seed in cfg['seeds']:
   r=np.zeros(512)
   for fold in cfg['folds']:
    z=np.load(out/'oof'/f'{arm}_seed{seed}_fold{fold}.npz');r[z['test_indices']]=z['risk']
   risks[arm,seed]=r
 rng=np.random.default_rng(cfg['bootstrap']['seed']);draws=rng.multinomial(512,np.full(512,1/512),size=cfg['bootstrap']['draws']);i,j=pairs(d['DSS_time'],d['DSS_event']);keep=d['folds'][i]==d['folds'][j];i,j=i[keep],j[keep];keys=list(risks);credit=np.stack([np.where(abs(r[i]-r[j])<=1e-8,.5,r[i]>r[j]) for r in risks.values()],1);hv=[];uv=[]
 for start in range(0,len(draws),100):
  q=(draws[start:start+100,i]*draws[start:start+100,j]).astype(float);qu=q*w[i];hv.append((q@credit)/q.sum(1)[:,None]);uv.append((qu@credit)/qu.sum(1)[:,None])
 bs={'Harrell':np.concatenate(hv),'Uno60':np.concatenate(uv)};rows=[]
 for family,ps in [('requested_descriptive',comparisons),('imbalance_family',extra)]:
  for a,b in ps:
   for metric in bs:
    v=np.mean([bs[metric][:,keys.index((a,s))]-bs[metric][:,keys.index((b,s))] for s in cfg['seeds']],0);dd=[metrics(d['DSS_time'],d['DSS_event'],risks[a,s],d['folds'],w)[metric]-metrics(d['DSS_time'],d['DSS_event'],risks[b,s],d['folds'],w)[metric] for s in cfg['seeds']];lo,hi=np.quantile(v,[.025,.975]);alpha=.05/6 if family=='imbalance_family' else .05;sl,sh=np.quantile(v,[alpha/2,1-alpha/2]);rows.append({'family':family,'comparison':a+'-'+b,'metric':metric,'mean_delta':np.mean(dd),'CI95_low':lo,'CI95_high':hi,'family_low':sl,'family_high':sh,'positive_seeds':sum(np.array(dd)>0),'interpretation':'descriptive only; not added to main decision family'})
 a=pd.DataFrame(rows);a[a.family=='requested_descriptive'].to_csv(report/'additional_paired_comparisons.csv',index=False);a[a.family=='imbalance_family'].to_csv(report/'imbalance_family_intervals.csv',index=False);print('DONE descriptive requested pairs and imbalance-family intervals',flush=True)
if __name__=='__main__':main()
