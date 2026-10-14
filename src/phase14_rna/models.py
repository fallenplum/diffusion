import copy
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from src.phase13_rna.models import RNAOnly
from src.phase13_rna.math import fit_ridge
from src.phase10_trimodal.math import log_weights,torch_cox


def teacher_targets(x,t,e,seed):
    targets=np.empty(len(x));audit=[]
    for train,test in StratifiedKFold(3,shuffle=True,random_state=seed).split(x,e):
        w,info=fit_ridge(x[train],t[train],e[train],1.)
        risk=x[train]@w;sd=max(risk.std(),1e-8)
        targets[test]=(x[test]@w-risk.mean())/sd
        audit.append({'train':train.tolist(),'test':test.tolist(),'normalization_mean':float(risk.mean()),'normalization_sd':float(sd),'gradient_max':info['gradient_max']})
    return targets,audit


def ranking_kl(risk,target):
    ii,jj=torch.triu_indices(len(risk),len(risk),1,device=risk.device)
    p=torch.sigmoid(target[ii]-target[jj]).detach()
    logits=risk[ii]-risk[jj]
    entropy=-(p*torch.log(p.clamp_min(1e-7))+(1-p)*torch.log((1-p).clamp_min(1e-7)))
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits,p,reduction='none')-entropy).mean()


def fit_model(model,train,val,t,e,fit,valid,cfg,epoch=None,teacher=None,penalty=None):
    device=next(model.parameters()).device
    train=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in train]
    val=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in val]
    ee=torch.as_tensor(e[fit],device=device);la=log_weights(t[fit],e[fit],device)
    if len(valid):ve=torch.as_tensor(e[valid],device=device);vl=log_weights(t[valid],e[valid],device)
    target=None if teacher is None else torch.as_tensor(teacher,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    logs=[];best=None;state=None;ema=None
    for ep in range(1,(epoch or cfg['epochs'])+1):
        model.train();opt.zero_grad(set_to_none=True);risk=model(*train)
        if isinstance(risk,tuple):risk=risk[0]
        cox=torch_cox(risk,la,ee);kd=ranking_kl(risk,target) if target is not None else risk.sum()*0
        reg=penalty(model) if penalty else risk.sum()*0
        loss=cox+cfg['rank_weight']*kd+reg;loss.backward()
        grad=float(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm())
        connected=sum(p.numel() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip']);opt.step();model.eval()
        with torch.no_grad():
            rr=model(*train);rr=rr[0] if isinstance(rr,tuple) else rr;tl=float(torch_cox(rr,la,ee));vv=None
            if len(valid):
                vr=model(*val);vr=vr[0] if isinstance(vr,tuple) else vr;vv=float(torch_cox(vr,vl,ve))
                ema=vv if ema is None else cfg['ema']*vv+(1-cfg['ema'])*ema
                if best is None or ema<best[0]:best=(ema,ep);state=copy.deepcopy(model.state_dict())
        logs.append({'epoch':ep,'train_cox':tl,'train_objective_before':float(loss.detach()),'rank_KL':float(kd.detach()),'validation_cox':vv,'validation_EMA':ema,'gradient_norm':grad,'connected_parameters':connected})
        if not np.isfinite(tl+grad):raise RuntimeError('nonfinite training')
    if state is not None:model.load_state_dict(state)
    model.eval()
    return {'selected_epoch':best[1] if best else epoch,'actual_epochs':len(logs),'logs':logs,'parameters':sum(p.numel() for p in model.parameters()),'peak_cuda_MB':torch.cuda.max_memory_allocated()/2**20}


class Conditional(nn.Module):
    def __init__(self,dc,dr,w,interaction,cfg,seed):
        super().__init__();torch.manual_seed(seed)
        self.main=nn.Parameter(torch.tensor(w,dtype=torch.float32));self.dc=dc;self.interaction=interaction;self.bound=cfg['interaction_bound']
        if interaction:
            self.left=nn.Parameter(torch.randn(dc,cfg['interaction_rank'])*.02)
            self.right=nn.Parameter(torch.randn(dr,cfg['interaction_rank'])*.02)
            self.mix=nn.Parameter(torch.ones(cfg['interaction_rank'])/cfg['interaction_rank']**.5)
    def correction(self,c,r):
        if not self.interaction:return c[:,0]*0
        return self.bound*torch.tanh(((c@self.left)*(r@self.right)*self.mix).sum(1))
    def forward(self,c,r):return torch.cat([c,r],1)@self.main+self.correction(c,r)
    def regularization(self,base,cfg):
        value=.5*(base*self.main[:self.dc].square().sum()+cfg['probe_lambda']*self.main[self.dc:].square().sum())
        if self.interaction:value=value+cfg['interaction_penalty']*(self.left.square().mean()+self.right.square().mean()+self.mix.square().mean())
        return value


def derange(indices,strata,seed):
    indices=np.asarray(indices);out=indices.copy();rng=np.random.default_rng(seed)
    for s in np.unique(strata[indices]):
        positions=np.flatnonzero(strata[indices]==s)
        if len(positions)<2:raise ValueError('singleton wrong-patient stratum')
        shuffled=rng.permutation(positions);out[shuffled]=indices[np.roll(shuffled,1)]
    if np.any(out==indices) or set(out)!=set(indices):raise ValueError('invalid derangement')
    return out
