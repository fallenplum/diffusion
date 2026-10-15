"""Pinned official local runtime, exact projected features, hashed text cache."""
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModel
from src.phase16_semantic.core import *

def official_model():
    info=json.loads((MODEL/'DOWNLOAD_COMPLETE.json').read_text())
    for row in info['files']:
        if sha(MODEL/row['file'])!=row['sha256']:raise ValueError('official model source/hash changed')
    model,loading=AutoModel.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('official checkpoint did not load exactly: '+str(loading))
    loading={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
    write(R/'OFFICIAL_LOAD_AUDIT.json',{'status':'PASS','loading_info':loading,'official_files_verified':len(info['files']),'weights_sha256':sha(MODEL/'model.safetensors')});model=model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    return model

class TextCache:
    def __init__(self,model):
        self.model=model;self.maximum=0;self.generated=0;self.reused=0;self.model_sha=sha(MODEL/'model.safetensors');path=Path('/dev/shm/phase16_conch_z_text.sqlite');self.db=sqlite3.connect(path);self.db.execute('CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY,prompt TEXT NOT NULL,value BLOB NOT NULL)');self.db.commit()
    def encode(self,texts):
        texts=list(texts);keys=[hashlib.sha256((self.model_sha+'|official_half_float32_normalized_v1|'+s).encode()).hexdigest() for s in texts];lookup={};unique={k:s for k,s in zip(keys,texts)}
        keys_unique=list(unique)
        for start in range(0,len(keys_unique),400):
            kk=keys_unique[start:start+400];found=self.db.execute('SELECT key,value FROM embeddings WHERE key IN ('+','.join('?' for _ in kk)+')',kk).fetchall()
            for k,value in found:lookup[k]=np.frombuffer(value,dtype=np.float32)
        self.reused+=len(lookup);missing=[k for k in keys_unique if k not in lookup]
        for start in range(0,len(missing),cfg()['text_batch']):
            batch=missing[start:start+cfg()['text_batch']];words=[unique[k] for k in batch];lengths=[len(t) for t in self.model.tokenizer.tokenizer(words,truncation=False,padding=False)['input_ids']];maximum=max(lengths);self.maximum=max(self.maximum,maximum)
            if maximum>127:raise ValueError('conditioning text exceeds127tokens;silent truncation prohibited')
            with torch.inference_mode():
                token=self.model.tokenizer(words).cuda();value=self.model.encode_text(token,normalize=True).float();value=F.normalize(value,dim=-1);x=value.cpu().numpy()
            if not np.isfinite(x).all() or x.shape!=(len(batch),768):raise ValueError('text embedding invalid')
            self.db.executemany('INSERT OR IGNORE INTO embeddings VALUES(?,?,?)',[(k,unique[k],xx.tobytes()) for k,xx in zip(batch,x)]);self.db.commit()
            for k,xx in zip(batch,x):lookup[k]=xx
            self.generated+=len(batch)
            if start and start%4096==0:print('TEXT',self.generated,'CACHED',self.reused,flush=True)
        return np.stack([lookup[k] for k in keys])

def projected_cache(model):
    m=base.manifest();path=Path('/dev/shm/phase16_conch_z_projected.npy');meta=P/'PROJECTION_CACHE.json';weights_sha=sha(MODEL/'model.safetensors')
    if meta.exists() and path.exists():
        z=json.loads(meta.read_text())
        if z['weights_sha256']==weights_sha and z['candidate_sha256']==m['source_sha256'] and z['cache_sha256']==sha(path):return np.load(path,mmap_mode='r')
        raise ValueError('stale projection cache')
    x=np.load(m['features'],mmap_mode='r');out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=x.shape);projection=model.visual.proj_contrast.detach().float()
    with torch.inference_mode():
        for start in range(0,len(x),8192):out[start:start+8192]=F.normalize(torch.tensor(np.asarray(x[start:start+8192]),device='cuda')@projection,dim=-1).cpu().numpy()
    out.flush();write(meta,{'status':'PASS','weights_sha256':weights_sha,'candidate_sha256':m['source_sha256'],'cache_sha256':sha(path),'source_visual_cache_sha256':m['cache_sha256'],'operation':'L2normalize(L2normalized_raw_CONCH @ official_proj_contrast);homogeneous equivalent to raw projection','path':str(path)});return np.load(path,mmap_mode='r')

def round_robin(scores,k):
    # All concepts get an equal chance at each depth; no label-based concept weights.
    n=scores.shape[0];k=min(k,n);best=torch.topk(scores,k,dim=0,largest=True,sorted=True).indices.cpu().numpy();seen=set();result=[]
    for depth in range(k):
        for concept in range(best.shape[1]):
            value=int(best[depth,concept])
            if value not in seen:seen.add(value);result.append(value)
            if len(result)==k:return np.array(result,np.int64)
    raise ValueError('ROI union failed to reach budget')

def retrieve(projected,embeddings,budgets=(128,512)):
    m=base.manifest();slides=m['slides'];quota={b:np.load(base.P/'selectors'/f'quota_{b}.npy') for b in budgets};result={b:np.full((len(slides),int(quota[b].max())),-1,np.int64) for b in budgets};owners=np.load(base.P/'row_owners.npy',mmap_mode='r')
    with torch.inference_mode():
        for i,s in enumerate(slides):
            a=s['offset'];b=a+s['n'];image=torch.tensor(np.asarray(projected[a:b]),device='cuda');text=torch.tensor(embeddings[s['patient']],device='cuda');score=image@text.T;ranking=round_robin(score,int(quota[max(budgets)][i]))+a
            for budget in budgets:
                ids=ranking[:quota[budget][i]];base.check_rows(ids,owners,s['patient']);result[budget][i,:len(ids)]=ids
    return result
