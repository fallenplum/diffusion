"""Independent sampled projection, text and concept-round-robin replay."""
import sys,json,sqlite3,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from src.phase16_semantic.core import *
from src.phase16_semantic.prompts import captions
from src.phase16_semantic.runtime import official_model

def main():
    base.seed_all(199);d=parent.load();model=official_model().half();weight=sha(MODEL/'model.safetensors');database=sqlite3.connect('file:/dev/shm/phase16_conch_z_text.sqlite?mode=ro',uri=True) if Path('/dev/shm/phase16_conch_z_text.sqlite').exists() else None;capsule=P/'QUERY_TEXT_REPLAY_CACHE.npz';bank={}
    if capsule.exists():
        saved=np.load(capsule);bank=dict(zip(saved['keys'].astype(str),saved['vectors']))
    cache=np.load('/dev/shm/phase16_conch_z_projected.npy',mmap_mode='r');projection=np.load(P/'official_projection.npy');meta=json.loads((P/'PROJECTION_CACHE.json').read_text());m=base.manifest();raw=np.load(m['features'],mmap_mode='r')
    if sha('/dev/shm/phase16_conch_z_projected.npy')!=meta['cache_sha256']:raise ValueError('projected cache changed')
    examples=np.linspace(0,len(raw)-1,1024).astype(int)
    with torch.no_grad():expected=F.normalize(torch.tensor(np.asarray(raw[examples]),device='cuda')@torch.tensor(projection,device='cuda'),dim=-1).cpu().numpy()
    projection_error=float(np.max(abs(expected-cache[examples])))
    if projection_error>1e-6:raise ValueError('independent projection replay')
    checks=0;max_text_error=0;rows=[]
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            part=parent.splits(d,'DSS',fold)
            for role in ['inner','outer']:
                fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test'];queries,audit=captions(d,fit,test,seed,fold);slide=(seed*13+fold*41+(role=='outer')*103)%len(m['slides']);s=m['slides'][slide];patient=s['patient'];a=s['offset'];b=a+s['n'];qmax=int(np.load(base.P/'selectors/quota_512.npy')[slide]);image=torch.tensor(np.asarray(cache[a:b]),device='cuda')
                for selector,texts in queries.items():
                    words=texts[patient];keys=[hashlib.sha256((weight+'|official_half_float32_normalized_v1|'+v).encode()).hexdigest() for v in words];emb=[]
                    for key in keys:
                        found=database.execute('SELECT value FROM embeddings WHERE key=?',(key,)).fetchone() if database is not None else None
                        value=np.frombuffer(found[0],np.float32).copy() if found else bank.get(key)
                        if value is None:raise ValueError('missing text provenance')
                        bank[key]=value;emb.append(value)
                    text=torch.tensor(np.stack(emb),device='cuda')
                    with torch.no_grad():
                        # Exact ranking replay uses stored encoder outputs;
                        # independent fresh encoding tolerates half/batch roundoff.
                        fresh=F.normalize(model.encode_text(model.tokenizer(words).cuda()).float(),dim=-1);err=float((fresh-text).abs().max());max_text_error=max(max_text_error,err);score=image@text.T;top=torch.topk(score,qmax,dim=0,sorted=True).indices.cpu().numpy().reshape(-1)
                    if err>2e-3:raise ValueError('text cache not reproducible by official frozen encoder')
                    _,first=np.unique(top,return_index=True);rank=top[np.sort(first)][:qmax]+a
                    for budget in cfg()['budgets']:
                        quota=int(np.load(base.P/'selectors'/f'quota_{budget}.npy')[slide]);actual=np.load(directory(seed,fold,role)/f'{selector}_{budget}.npy')[slide,:quota]
                        if not np.array_equal(actual,rank[:quota]):raise ValueError('independent semantic ROI ranking mismatch')
                        checks+=1
                    rows.append({'seed':seed,'fold':fold,'role':role,'selector':selector,'anonymous_slide_number':slide+1,'text_max_roundoff':err,'ROI_order_exact':True})
    import pandas as pd
    np.savez_compressed(capsule,keys=np.asarray(list(bank)),vectors=np.stack(list(bank.values())))
    pd.DataFrame(rows).to_csv(R/'QUERY_REPLAY_SAMPLES.csv',index=False);write(R/'QUERY_REPLAY_AUDIT.json',{'status':'PASS','ROI_budget_order_checks':checks,'projected_features_checked':1024,'projection_error':projection_error,'fresh_text_max_roundoff':max_text_error,'outcome_labels_used':False,'persistent_private_replay_capsule_sha256':sha(capsule),'persistent_capsule_vectors':len(bank),'sampling_scope':'one deterministic WSI perrole and selector;all selected rows have full ownership/hash checks elsewhere'});print('QUERY_REPLAY_PASS',checks,flush=True)
if __name__=='__main__':main()
