import hashlib,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from src.phase13_rna import data as parent
from src.phase15_complement.core import cs_features
from src.phase10_trimodal.math import torch_cox,log_weights
from src.phase13_rna.math import inner_score

ROOT=Path(__file__).resolve().parents[2];P=Path('/root/phase16_routing_private');R=ROOT/'reports/phase16_semantic_routing';CONFIG=ROOT/'configs/phase16_semantic_routing.json'
def cfg():return json.loads(CONFIG.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=cast,allow_nan=False))
def source():
    paths=[CONFIG]+sorted((ROOT/'src/phase16_routing').glob('*.py'))+[ROOT/'scripts/run_phase16.py']
    return hashlib.sha256((parent.source_hash()+''.join(str(p.relative_to(ROOT)).replace('\\','/')+sha(p) for p in paths)).encode()).hexdigest()
def arm_names():return [f'{s}_{m}' for s in cfg()['active_selectors'] for m in cfg()['budgets']]+cfg()['anchors']
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def anchor(seed,fold,role):
    p=Path('/root/phase15_complement_private/anchors')/f'DSS_seed{seed}_fold{fold}_{role}.pt'
    z=torch.load(p,map_location='cpu',weights_only=False)
    expected=parent.splits(parent.load(),'DSS',fold)['inner_train' if role=='inner' else 'outer_train']
    if not np.array_equal(z['fit'],expected):raise ValueError('anchor partition mismatch')
    return z,p
def manifest():return json.loads((P/'CANDIDATE_CACHE.json').read_text())
def check_rows(rows,owner,patient):
    v=np.asarray(rows);v=v[v>=0]
    if len(v)!=len(set(v.tolist())) or np.any(np.asarray(owner)[v]!=patient):raise ValueError('cross-patient or duplicate ROI')
def check_geometry(actual,expected):
    for k in ['patch_size_level0','target_magnification','level0_magnification']:
        if actual.get(k)!=expected.get(k):raise ValueError('coordinate level/MPP mismatch')
def check_prompt_pair(part,perm):
    if set(part)!=set(perm) or np.any(np.asarray(part)==perm):raise ValueError('prompt permutation identity/partition')
def check_tag(tag,arm,seed,fold):
    if tag!={'arm':arm,'seed':seed,'fold':fold,'source':source(),'candidates':manifest()['source_sha256']}:raise ValueError('checkpoint identity mismatch')

class Bags:
    """Frozen visual features, grouped by WSI. No cross-patient normalization."""
    def __init__(self,selector,budget,seed,device='cuda'):
        self.m=manifest();self.device=device;self.dynamic=selector=='MIL';self.buckets=[];self.owner=[];self.quota=[]
        raw=np.load(self.m['features'],mmap_mode='r');slides=self.m['slides'];quota=np.load(P/'selectors'/f'quota_{budget}.npy')
        indices=None if self.dynamic else np.load(P/'selectors'/f'{selector}_seed{seed}_{budget}.npy')
        for start in range(0,len(slides),24):
            # Length sorting makes padding bounded for full candidate MIL.
            order=sorted(range(len(slides)),key=lambda i:slides[i]['n'] if self.dynamic else quota[i])[start:start+24]
            lengths=[slides[i]['n'] if self.dynamic else int(quota[i]) for i in order];width=max(lengths)
            x=np.zeros((len(order),width,768),np.float32);mask=np.zeros((len(order),width),bool);rowids=np.full((len(order),width),-1,np.int64)
            for j,i in enumerate(order):
                ids=np.arange(slides[i]['offset'],slides[i]['offset']+slides[i]['n']) if self.dynamic else indices[i,:quota[i]]
                if np.any(ids<0):raise ValueError('negative selected row')
                x[j,:len(ids)]=raw[ids];mask[j,:len(ids)]=True;rowids[j,:len(ids)]=ids
            self.buckets.append((torch.tensor(x,device=device),torch.tensor(mask,device=device),torch.tensor(quota[order],device=device),rowids,order,int(quota[order].max()),quota[order]))
            self.owner.extend(slides[i]['patient'] for i in order)
        counts=np.bincount(self.owner,minlength=512);merge=np.zeros((512,len(self.owner)),np.float32)
        for j,i in enumerate(self.owner):merge[i,j]=1/counts[i]
        self.merge=torch.tensor(merge,device=device)
    def pool(self,score,return_rows=False):
        outputs=[];chosen={}
        for x,mask,quota,ids,order,k,quota_np in self.buckets:
            values=x@score;values=values.masked_fill(~mask,-torch.inf)
            if self.dynamic:
                v,ix=values.topk(k,dim=1,sorted=True);keep=torch.arange(k,device=self.device)[None,:]<quota[:,None]
                xx=x.gather(1,ix[:,:,None].expand(-1,-1,768));v=v.masked_fill(~keep,-torch.inf)
                w=torch.softmax(v,1);pooled=(xx*w[:,:,None]).sum(1)
                if return_rows:
                    pos=ix.detach().cpu().numpy()
                    for j,o in enumerate(order):chosen[o]=ids[j,pos[j,:quota_np[j]]]
            else:
                w=torch.softmax(values,1);pooled=(x*w[:,:,None]).sum(1)
                if return_rows:
                    for j,o in enumerate(order):chosen[o]=ids[j,mask[j].cpu().numpy()]
            outputs.append(pooled)
        output=self.merge@torch.cat(outputs,0)
        return (output,chosen) if return_rows else output

class RoutingModel(nn.Module):
    def __init__(self,w):
        super().__init__();self.base=nn.Parameter(torch.tensor(w,dtype=torch.float32));self.score=nn.Parameter(torch.randn(768)*.01);self.roi=nn.Parameter(torch.randn(768)*.01)
    def forward(self,cs,bags):
        p=bags.pool(self.score);correction=cfg()['training']['residual_scale']*torch.tanh(p@self.roi)
        return cs@self.base+correction

def train(cs,bags,t,e,fit,valid,w,lam,epochs,seed,select):
    seed_all(seed);device=bags.device;model=RoutingModel(w).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=cfg()['training']['lr']);xx=torch.tensor(cs,dtype=torch.float32,device=device)
    fit=np.asarray(fit);a=log_weights(t[fit],e[fit],device);ee=torch.tensor(e[fit],device=device);v=log_weights(t[valid],e[valid],device) if select else None;ve=torch.tensor(e[valid],device=device) if select else None
    history=[];best=float('inf');best_epoch=0;best_state=None;ema=None;train_cfg=cfg()['training']
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();risk=model(xx,bags);data_loss=torch_cox(risk[fit],a,ee);penalty=lam/2*(model.base@model.base)+train_cfg['roi_l2']/2*((model.roi@model.roi)+(model.score@model.score));loss=data_loss+penalty
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();grad={k:float(vv.grad.norm()) for k,vv in model.named_parameters()};optimizer.step()
        entry={'epoch':epoch,'train_cox':float(data_loss.detach()),'penalized_loss':float(loss.detach()),'gradient':grad}
        if select:
            with torch.no_grad():rv=model(xx,bags)[valid];vl=float(torch_cox(rv,v,ve));ema=vl if ema is None else .3*vl+.7*ema;valc=inner_score(t[valid],e[valid],rv.cpu().numpy())
            entry.update(val_cox=vl,val_ema=ema,val_Harrell=valc)
            if ema<best:best=ema;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(entry)
    if select:model.load_state_dict(best_state)
    else:best_epoch=epochs
    return model,history,best_epoch
