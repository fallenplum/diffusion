"""Forward hooks expose exact C4 RNA retrieval BEFORE mean pooling."""
import numpy as np
import torch
from src.phase11_semantic.train import forward_partition


def fusion_taps(model,d,f,cache,ix):
    result={key:[] for key in ['raw_full','processed_full','raw_mean','processed_mean','RNApatch_full','RNApatch_mean','joint']}
    risks=[];device=next(model.parameters()).device;model.eval()
    for start in range(0,len(ix),64):
        ids=np.asarray(ix[start:start+64],int);p,mask=cache.batch(ids,device);valid=mask.any(-1);owner=valid.reshape(-1).nonzero().squeeze(1)//valid.shape[1];count=valid.sum(1)[:,None,None]
        captures={}
        def rna_hook(module,args,output):captures['raw'],captures['processed']=output
        def rp_hook(module,args,output):captures['RNApatch']=output
        def joined_hook(module,args):captures['joint']=args[0]
        handles=[model.rna.register_forward_hook(rna_hook),model.rp.register_forward_hook(rp_hook),model.head.register_forward_pre_hook(joined_hook)]
        try:
            with torch.no_grad():
                risk=model(torch.tensor(f['C'][ids],device=device),torch.tensor(f['S'][ids],device=device),torch.tensor(f['R'][ids],device=device),p,mask)
        finally:
            for handle in handles:handle.remove()
        rp=torch.zeros(len(ids),8,64,device=device).index_add(0,owner,captures['RNApatch'])/count
        if not torch.allclose(rp.mean(1),captures['joint'][:,192:256],atol=2e-6,rtol=0):raise RuntimeError('RNA Patch hook not exact joined slot')
        values={'raw_full':captures['raw'].flatten(1),'processed_full':captures['processed'].flatten(1),'raw_mean':captures['raw'].mean(1),'processed_mean':captures['processed'].mean(1),'RNApatch_full':rp.flatten(1),'RNApatch_mean':rp.mean(1),'joint':captures['joint']}
        for key,value in values.items():result[key].append(value.detach().cpu().numpy())
        risks.append(risk.detach().cpu().numpy())
    return np.concatenate(risks),{key:np.concatenate(values) for key,values in result.items()}
