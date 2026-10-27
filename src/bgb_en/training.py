"""Deterministic full-risk-set neural fits with frozen scalar anchors."""
from __future__ import annotations
import copy
import time
import numpy as np
import torch
from torch import nn
from .pathology import build_pathology_encoder
from .omics import SparsePathwayEncoder,shuffled_membership
from .survival import cached_full_risk_step,evaluate

class PatchBank:
    def __init__(self,cache,device):
        self.cache=cache;owner=np.asarray(cache['owner'])
        slots=max(int(np.sum(owner==i)) for i in range(512));width=cache['features'].shape[1];dim=cache['features'].shape[-1]
        x=np.zeros((512,slots,width,dim),np.float32);xy=np.zeros((512,slots,width,2),np.int64);mask=np.zeros((512,slots,width),bool)
        for patient in range(512):
            rows=np.flatnonzero(owner==patient);x[patient,:len(rows)]=cache['features'][rows]
            xy[patient,:len(rows)]=cache['coords'][rows];mask[patient,:len(rows)]=cache['mask'][rows]
        self.x=torch.from_numpy(x).to(device);self.xy=torch.from_numpy(xy).to(device);self.mask=torch.from_numpy(mask).to(device)
        regions=np.full((512,slots,width),-1,np.int64)
        if 'region_ids' in cache:
            for patient in range(512):
                rows=np.flatnonzero(owner==patient);regions[patient,:len(rows)]=cache['region_ids'][rows]
        else:
            regions[mask]=0
        self.region_ids=torch.from_numpy(regions).to(device)
        if 'coordinate_step' not in cache:raise ValueError('Patch cache lacks an audited coordinate step')
        steps=np.unique(cache['coordinate_step'])
        if len(steps)!=1 or steps[0]<=0:raise ValueError('Mixed slide grid steps require explicit per-slide normalization')
        self.step=float(steps[0])
        self.input_dim=dim

class MechanismModel(nn.Module):
    def __init__(self,stage,arm,prepared,membership,patch_bank,device,*,gamma=.1,backend='mamba',intervention=None):
        super().__init__();self.stage,self.arm,self.device,self.gamma=stage,arm,device,gamma
        self.patch_bank=patch_bank;self.intervention=intervention
        self.path=None;self.omics=None;self.interaction=None;self.geometry_cache={}
        feature=prepared['features'];self.inputs={k:torch.as_tensor(feature[k],device=device) for k in ('rna','rna_mask','cna_gain','cna_loss','cna_mask')}
        self.clinical=torch.as_tensor(feature['C'],device=device,dtype=torch.float32)
        self.offset=torch.as_tensor(prepared['CS'] if stage=='R1' else prepared['RNA'] if stage=='R2' else prepared['base'],device=device,dtype=torch.float32)
        self._rna_component=prepared['beta']*prepared['RNA'] if stage=='R3' else prepared['RNA'] if stage=='R2' else np.zeros(len(prepared['RNA']))
        self.fit_indices=np.asarray(prepared['train']);self.valid_indices=np.asarray(prepared['valid'])
        self.query_map=np.arange(512);self.context_mean=None
        if stage=='R1':
            path_arm=arm if arm in ('P1','P2','P3','P4','P5','P6') else 'P1'
            self.path=build_pathology_encoder(path_arm,input_dim=patch_bank.input_dim,width=64,state_dim=8,coordinate_step=patch_bank.step)
            self.risk_head=nn.Linear(32,1,bias=False)
            nn.init.normal_(self.risk_head.weight,std=.002)
        else:
            mode={'O3':'pool','O4':'set','O5':'ssm','O6':'set_ssm','O7':'cna_gate','O8':'factors','O9':'set'}.get(arm,'pool')
            if stage=='R3':mode='set_ssm' if arm=='F9' else 'pool'
            if arm in ('O9','F5_SHUFFLED_MEMBERSHIP'):membership=shuffled_membership(membership,seed=270099)[0]
            self.omics=SparsePathwayEncoder(membership,mode=mode,width=32,patterns=4,
                sequence_backend=backend,order_seed=270101,orders=4)
            if stage=='R3':
                from .interaction import BidirectionalInteraction
                self.path=build_pathology_encoder('P3' if arm=='F8' else 'P1',input_dim=patch_bank.input_dim,width=64,state_dim=8,coordinate_step=patch_bank.step)
                interaction_arm='F5' if arm in ('F5_WRONG_QUERY','F5_SHUFFLED_MEMBERSHIP','F5_UNI2','F11_UNI2') else arm
                self.interaction=BidirectionalInteraction(arm=interaction_arm,clinical_dim=feature['C'].shape[1])
                # The interaction consumes raw per-patch and per-program states.
                # Unused auxiliary heads are frozen and excluded from active budgets.
                self.path.mil.requires_grad_(False);self.path.output.requires_grad_(False)
                self.omics.risk_head.requires_grad_(False);self.omics.pool_query.requires_grad_(False)
        self.to(device)

    def _path(self,idx):
        key=tuple(idx.detach().cpu().tolist());bank=self.patch_bank
        x,xy,mask=bank.x[idx],bank.xy[idx],bank.mask[idx]
        regions=bank.region_ids[idx]
        if key not in self.geometry_cache:self.geometry_cache[key]=self.path.geometry(xy,mask,regions)
        return self.path(x,xy,mask,return_details=True,geometry=self.geometry_cache[key],region_ids=regions)

    def forward(self,idx,control=None):
        if not torch.is_tensor(idx):idx=torch.as_tensor(idx,device=self.device,dtype=torch.long)
        offset=self.offset[idx]
        p=self._path(idx) if self.path is not None else None
        diagnostics={}
        if self.stage=='R1':
            raw=self.risk_head(p['embedding']).squeeze(-1);diagnostics=p['diagnostics']
            # R1 uses a linear residual with magnitude regularization, not a saturated bound.
            delta=raw
        else:
            args={k:v[idx] for k,v in self.inputs.items()}
            if control=='whole_omics':
                other=torch.as_tensor(self.query_map[idx.detach().cpu().numpy()],device=self.device)
                args={k:v[other] for k,v in self.inputs.items()}
                component=torch.as_tensor(self._rna_component,device=self.device)
                offset=offset-component[idx]+component[other]
            o=self.omics(**args)
            if self.stage=='R2':raw=o['risk']
            else:
                context=None
                if control=='wrong_query' or self.intervention=='wrong_query':
                    other=torch.as_tensor(self.query_map[idx.detach().cpu().numpy()],device=self.device)
                    other_o=self.omics(**{k:v[other] for k,v in self.inputs.items()})
                    context=(other_o['tokens'],other_o['token_mask'])
                # Mean query is computed only from the current training partition.
                if control=='mean_query':
                    if self.context_mean is None:raise ValueError('Missing train-only query mean')
                    context=(self.context_mean.expand(len(idx),-1,-1),o['token_mask'])
                output=self.interaction(p['patch_states'],p['patch_mask'],o['tokens'],o['token_mask'],
                    clinical=self.clinical[idx],context_omics=context,
                    interaction_off=control=='interaction_off',permute_attention=control=='attention_permutation',return_details=True)
                raw=output['risk'];diagnostics=output.get('diagnostics',{})
                # Preserve exact additivity of the common unimodal heads.
                # Bounding their sum would itself introduce a cross-modal term.
                pool_p=output['pooled_path_risk'];pool_o=output['pooled_omics_risk']
                cross=output['OtoP_risk']+output['PtoO_risk']
                delta=self.gamma*(.25*torch.tanh(pool_p)+.25*torch.tanh(pool_o)+.5*torch.tanh(cross))
            if self.stage=='R2':delta=self.gamma*torch.tanh(raw)
        if control=='direct_off':offset=offset-torch.as_tensor(self._rna_component,device=self.device)[idx]
        return dict(risk=offset+delta,delta=delta,diagnostics=diagnostics)

    def regularization(self):
        return 1e-4*sum((p.square().sum() for p in self.parameters() if p.requires_grad))

    def training_query_mean(self,batch=16):
        if self.omics is None:return
        values=[]
        with torch.no_grad():
            for start in range(0,len(self.fit_indices),batch):
                idx=torch.as_tensor(self.fit_indices[start:start+batch],device=self.device)
                values.append(self.omics(**{k:v[idx] for k,v in self.inputs.items()})['tokens'])
        self.context_mean=torch.cat(values).mean(0,keepdim=True)

def predict(model,indices,batch_size=8,control=None):
    model.eval();risks=[];delta=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            out=model(torch.as_tensor(indices[start:start+batch_size],device=model.device),control=control)
            risks.append(out['risk'].detach().cpu().numpy());delta.append(out['delta'].detach().cpu().numpy())
    return np.concatenate(risks).astype(float),np.concatenate(delta).astype(float)

def train_neural(model,raw,prepared,*,epochs=80,batch_size=8,lr=.001,evaluate_epochs=(20,40,80)):
    start=time.monotonic();train,valid=prepared['train'],prepared['valid'];device=model.device
    # Fixed patient order, dropout=0; this is an exact full-risk objective via replay.
    batches=[torch.as_tensor(train[j:j+batch_size],device=device) for j in range(0,len(train),batch_size)]
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    curves=[];snapshots={}
    torch.cuda.reset_peak_memory_stats(device) if str(device).startswith('cuda') else None
    for epoch in range(1,epochs+1):
        step=cached_full_risk_step(model,batches,raw['t'][train],raw['e'][train],optimizer,device)
        if epoch in evaluate_epochs or epoch==epochs:
            tr,td=predict(model,train,batch_size);vr,vd=predict(model,valid,batch_size)
            train_metric=evaluate(raw['t'][train],raw['e'][train],raw['t'][train],raw['e'][train],tr,model.offset[train].detach().cpu().numpy())
            valid_metric=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],vr,model.offset[valid].detach().cpu().numpy())
            grads=[float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
            record=dict(epoch=epoch,**step,train=train_metric,valid=valid_metric,delta_train_SD=float(td.std()),delta_valid_SD=float(vd.std()),
                delta_train_bound_fraction=float(np.mean(np.abs(td)>.99*model.gamma)) if model.stage!='R1' else None,
                gradient_tensor_nonzero_ratio=float(np.mean(np.asarray(grads)>0)) if grads else 0.)
            curves.append(record)
            snapshots[epoch]=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},risk=vr,delta=vd,metrics=valid_metric)
    return dict(curves=curves,snapshots=snapshots,elapsed_seconds=time.monotonic()-start,
        parameters=sum(p.numel() for p in model.parameters()),peak_GPU_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
        epochs=epochs,optimizer_updates=epochs,full_riskset=True,gradient_cache=True)
