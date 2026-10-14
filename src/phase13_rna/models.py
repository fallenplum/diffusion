"""Identical RNA tokenizer initialized like the historical C4 producer."""
import copy
import numpy as np
import torch
from torch import nn
from src.phase12_c4.models import C4Model
from src.phase10_trimodal.math import log_weights,torch_cox
from src.phase10_trimodal.features import ahash


class RNAOnly(nn.Module):
    def __init__(self,clinical_mask,membership,seed):
        super().__init__();torch.manual_seed(seed)
        reference=C4Model(clinical_mask,membership,64,'real',False)
        self.encoder=reference.rna.encoder
        self.head=nn.Linear(512,1,bias=False)
    def forward(self,x):
        raw,processed=self.encoder(x)
        return self.head(processed.flatten(1)).squeeze(1),{'raw_full':raw.flatten(1),'processed_full':processed.flatten(1),'raw_mean':raw.mean(1),'processed_mean':processed.mean(1)}


def train_rna(r,clinical_mask,membership,fit,val,t,e,seed,cfg,epochs=None):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    model=RNAOnly(clinical_mask,membership,seed).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    train=torch.tensor(r[fit],device=device);event=torch.tensor(e[fit],device=device);loga=log_weights(t[fit],e[fit],device)
    if len(val):
        validation=torch.tensor(r[val],device=device);ve=torch.tensor(e[val],device=device);vl=log_weights(t[val],e[val],device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    logs=[];best=None;beststate=None;ema=None
    for epoch in range(1,(epochs or cfg['inner_epochs'])+1):
        model.train();opt.zero_grad(set_to_none=True);risk,_=model(train);loss=torch_cox(risk,loga,event);loss.backward()
        connected=sum(p.numel() for p in model.parameters() if p.grad is not None);nonzero=sum(int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None)
        gradient=float(torch.stack([p.grad.norm() for p in model.encoder.parameters() if p.grad is not None]).norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step();model.eval()
        with torch.no_grad():
            trisk,_=model(train);tl=float(torch_cox(trisk,loga,event));validationloss=None
            if len(val):
                vr,_=model(validation);validationloss=float(torch_cox(vr,vl,ve));ema=validationloss if ema is None else cfg['ema_alpha']*validationloss+(1-cfg['ema_alpha'])*ema
                if epoch>=cfg['minimum_epoch'] and (best is None or ema<best[0]):best=(ema,epoch);beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        logs.append({'epoch':epoch,'train_loss_before':float(loss.detach()),'train_loss_after':tl,'val_loss':validationloss,'val_EMA':ema,'encoder_gradient':gradient,'connected_parameters':connected,'nonzero_gradient_elements':nonzero})
        if not np.isfinite(tl) or not np.isfinite(gradient):raise RuntimeError('nonfinite RNA-only training')
    selected=best[1] if best else epochs
    if beststate is not None:model.load_state_dict(beststate)
    model.eval()
    info={'selected_epoch':selected,'actual_epochs':len(logs),'initial_state':initial,'logs':logs,'parameters':sum(p.numel() for p in model.parameters()),'fit_hash':ahash(np.asarray(fit,np.int64)),
          'peak_cuda_MB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0}
    return model,info


def extract_rna(model,r,ix):
    model.eval();device=next(model.parameters()).device
    with torch.no_grad():risk,taps=model(torch.tensor(r[ix],device=device))
    return risk.detach().cpu().numpy(),{k:v.detach().cpu().numpy() for k,v in taps.items()}
