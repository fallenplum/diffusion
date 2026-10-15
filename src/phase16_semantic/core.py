import hashlib,json
from pathlib import Path
import numpy as np,torch
from src.phase16_routing import core as base
ROOT=base.ROOT;P=Path('/root/phase16_semantic_private');R=ROOT/'reports/phase16_semantic_routing/semantic_completion';CONFIG=ROOT/'configs/phase16_semantic_continuation.json';MODEL=Path('/root/phase16_conch_z/model')
write=base.write;sha=base.sha;parent=base.parent
def cfg():return json.loads(CONFIG.read_text())
def source():
    files=[CONFIG]+[ROOT/'src/phase16_semantic'/n for n in ['core.py','prompts.py','runtime.py']]+[ROOT/'scripts'/n for n in ['run_phase16_semantic.py','prepare_phase16_semantic.py']]
    return hashlib.sha256((base.source()+''.join(str(p.relative_to(ROOT)).replace('\\','/')+sha(p) for p in files)).encode()).hexdigest()
def arms():return [f'{s}_{m}' for s in cfg()['G2_selectors']+cfg()['G3_selectors'] for m in cfg()['budgets']]
def label(arm,seed,fold):return f'{arm}_seed{seed}_fold{fold}'
def directory(seed,fold,role):return P/'retrieval'/f'seed{seed}_fold{fold}_{role}'
def check_partitions(fit,test,part):
    if set(fit)&set(test) or set(fit)|set(test)!=set(part):raise ValueError('retrieval partition mismatch')
def check_permutation(indices,perm):
    if set(indices)!=set(perm) or np.any(np.asarray(indices)==perm):raise ValueError('wrong prompt crossed partition or identity')
def tag(arm,seed,fold):return {'arm':arm,'seed':seed,'fold':fold,'source':source(),'candidate_sha256':base.manifest()['source_sha256'],'weights_sha256':next(r['sha256'] for r in json.loads((MODEL/'DOWNLOAD_COMPLETE.json').read_text())['files'] if r['file']=='model.safetensors')}
class SemanticBags(base.Bags):
    def __init__(self,path,budget,device='cuda'):
        self.m=base.manifest();self.device=device;self.dynamic=False;self.buckets=[];self.owner=[];self.quota=[]
        raw=np.load(self.m['features'],mmap_mode='r');slides=self.m['slides'];quota=np.load(base.P/'selectors'/f'quota_{budget}.npy');indices=np.load(path)
        for start in range(0,len(slides),24):
            order=sorted(range(len(slides)),key=lambda i:quota[i])[start:start+24];width=int(quota[order].max());x=np.zeros((len(order),width,768),np.float32);mask=np.zeros((len(order),width),bool);rowids=np.full((len(order),width),-1,np.int64)
            for j,i in enumerate(order):
                ids=indices[i,:quota[i]]
                if np.any(ids<0):raise ValueError('selected ROI missing')
                base.check_rows(ids,np.load(base.P/'row_owners.npy',mmap_mode='r'),slides[i]['patient']);x[j,:len(ids)]=raw[ids];mask[j,:len(ids)]=True;rowids[j,:len(ids)]=ids
            self.buckets.append((torch.tensor(x,device=device),torch.tensor(mask,device=device),torch.tensor(quota[order],device=device),rowids,order,width,quota[order]));self.owner.extend(slides[i]['patient'] for i in order)
        counts=np.bincount(self.owner,minlength=512);merge=np.zeros((512,len(self.owner)),np.float32)
        for j,i in enumerate(self.owner):merge[i,j]=1/counts[i]
        self.merge=torch.tensor(merge,device=device)
