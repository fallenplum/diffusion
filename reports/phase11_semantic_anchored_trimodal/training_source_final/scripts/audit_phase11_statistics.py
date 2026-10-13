"""Check optimized paired bootstrap against literal duplicated patient rows."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_phase11_independent import components,ipcw,pairs
REPORT=ROOT/'reports/phase11_semantic_anchored_trimodal';PRIVATE=Path('/root/phase11_semantic_anchor_private/formal')
cfg=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'))
d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));t=d['DSS_time'];e=d['DSS_event'];f=d['folds'];weights=np.zeros(512)
for fold in range(1,6):
    tr=f!=fold;te=f==fold;weights[te]=ipcw(t[tr],e[tr],t[te],e[te])
comparisons=[('B0','A7'),('C1','C0'),('P5','B0')];risks={}
for arm in set(sum([list(c) for c in comparisons],[])):
    for seed in cfg['seeds']:
        r=np.empty(512)
        for fold in range(1,6):
            z=dict(np.load(PRIVATE/'oof'/f'{arm}_seed{seed}_fold{fold}.npz'));r[z['test_indices']]=z['risk']
        risks[arm,seed]=r
pi,pj=pairs(t,e);keep=f[pi]==f[pj];pi,pj=pi[keep],pj[keep]
rng=np.random.default_rng(cfg['bootstrap']['seed']);maxerr=0;checks=0
for draw in range(25):
    count=rng.multinomial(512,np.ones(512)/512);q=count[pi]*count[pj];qu=q*weights[pi]
    for a,b in comparisons:
        optimized=[];literal=[]
        for seed in cfg['seeds']:
            credits=[];num=[0.,0.];den=[0.,0.]
            for arm in [a,b]:
                r=risks[arm,seed];diff=r[pi]-r[pj];credits.append(np.where(abs(diff)<=1e-8,.5,diff>0))
                sums=np.zeros(4)
                for fold in range(1,6):
                    ix=np.flatnonzero(f==fold);expanded=np.repeat(ix,count[ix])
                    hn,hd=components(t[expanded],e[expanded],r[expanded]);un,ud=components(t[expanded],e[expanded],r[expanded],weights[expanded]);sums+=np.array([hn,hd,un,ud])
                literal.append((sums[0]/sums[1],sums[2]/sums[3]))
            optimized.extend([(q@(credits[0]-credits[1]))/q.sum(),(qu@(credits[0]-credits[1]))/qu.sum()])
            exact=np.array(literal[-2])-np.array(literal[-1]);maxerr=max(maxerr,float(np.max(abs(exact-optimized[-2:]))));checks+=2
if maxerr>1e-12:raise RuntimeError('literal/credit bootstrap mismatch')
(REPORT/'LITERAL_BOOTSTRAP_AUDIT.json').write_text(json.dumps({'status':'PASS','draws_rechecked':25,'comparisons':comparisons,'scalar_differences_checked':checks,'max_abs_error':maxerr,'duplicate_patients_explicitly_expanded':True,'fixed_training_IPCW':True},indent=2));print('PASS literal bootstrap',maxerr)
