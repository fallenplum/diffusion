"""Adapters to pinned upstream implementations; no third-party source vendored.

MGCT batched port changes only tensor batch axes/masks and 6->8 signatures.
MambaMIL imports original SRMamba and MIL modules. The old fused helper is
ported to public causal_conv1d + selective_scan APIs with the same Euler input
discretization, NOT the exact-ZOH conditional S6 used by our proposed arms.
"""
import sys,types,importlib.util,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
P=Path('/root/autodl-tmp/phase17_ssd_private')
sys.path.insert(0,str(P/'paper_deps'))


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def mamba_inner_no_out(xz,conv1d_weight,conv1d_bias,x_proj_weight,delta_proj_weight,A,B=None,C=None,D=None,delta_bias=None,B_proj_bias=None,C_proj_bias=None,delta_softplus=True,checkpoint_lvl=1,**kwargs):
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    x,z=xz.chunk(2,dim=1);n=x.shape[-1];rank=delta_proj_weight.shape[1];state=A.shape[1]
    x=causal_conv1d_fn(x.contiguous(),conv1d_weight.squeeze(1).contiguous(),conv1d_bias,activation='silu')
    projected=F.linear(x.transpose(1,2),x_proj_weight)
    delta=F.linear(projected[:,:,:rank],delta_proj_weight).transpose(1,2).contiguous()
    B=projected[:,:,rank:rank+state].transpose(1,2).contiguous() if B is None else B
    C=projected[:,:,rank+state:rank+2*state].transpose(1,2).contiguous() if C is None else C
    if B_proj_bias is not None:B=B+B_proj_bias[None,:,None]
    if C_proj_bias is not None:C=C+C_proj_bias[None,:,None]
    return selective_scan_fn(x,delta,A,B,C,D,z=z.contiguous(),delta_bias=delta_bias,delta_softplus=delta_softplus)


def mambamil():
    # Load computational modules without the unrelated language-generation
    # __init__, whose transformers4 dependency conflicts with server HF5.
    package_root=types.ModuleType('mamba_ssm');package_root.__path__=[str(P/'paper_deps/mamba_ssm')];sys.modules['mamba_ssm']=package_root
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,mamba_inner_fn
    fake=types.ModuleType('mamba');fake.__path__=[];sys.modules['mamba']=fake
    package=types.ModuleType('mamba.mamba_ssm');package.__path__=[];sys.modules[package.__name__]=package
    ops=types.ModuleType('mamba.mamba_ssm.ops');ops.__path__=[];sys.modules[ops.__name__]=ops
    interface=types.ModuleType('mamba.mamba_ssm.ops.selective_scan_interface')
    interface.selective_scan_fn=selective_scan_fn;interface.mamba_inner_fn=mamba_inner_fn;interface.mamba_inner_fn_no_out_proj=mamba_inner_no_out
    sys.modules[interface.__name__]=interface
    base=P/'literature/MambaMIL'
    sr=load_file('pinned_srmamba',base/'mamba/mamba_ssm/modules/srmamba.py')
    bi=load_file('pinned_bimamba',base/'mamba/mamba_ssm/modules/bimamba.py')
    package.SRMamba=sr.SRMamba;package.BiMamba=bi.BiMamba;package.Mamba=Mamba
    model=load_file('pinned_mambamil',base/'models/MambaMIL.py')
    return model.MambaMIL(1536,1,.1,'relu',survival=False,layer=2,rate=10,type='SRMamba')


class PaperModel(nn.Module):
    def __init__(self,name,clinical_dim,membership):
        super().__init__();self.name=name;self.gamma=nn.Parameter(torch.tensor(.1));self.direct=nn.Linear(1536,1,bias=False);nn.init.zeros_(self.direct.weight)
        self.late=nn.Linear(clinical_dim,1,bias=False);nn.init.zeros_(self.late.weight)
        membership=torch.as_tensor(membership,dtype=torch.bool)
        for j in range(8):self.register_buffer('genes'+str(j),torch.where(membership[j])[0])
        if name=='MGCT':
            base=P/'literature/MGCT';sys.path.insert(0,str(base))
            module=load_file('pinned_MGCT',base/'models/MGCT.py')
            self.core=module.MGCT(stage1_num_layers=1,stage2_num_layers=2,n_classes=1,dropout=.1,omic_sizes=[int(v.sum())*5 for v in membership],fusion='concat')
            self.core.wsi_net[0]=nn.Linear(1536,256)
        elif name=='MambaMIL':self.core=mambamil()
        else:raise ValueError(name)

    @staticmethod
    def mgct_layer(layer,wsi,omic,mask=None,omic_mask=None):
        # Official weights and operations, with explicit batch dimension.
        a,_=layer.wsi_omic_attn(omic,wsi,wsi,key_padding_mask=mask)
        weights,features=layer.wsi_omic_pool(a.transpose(0,1));weights=weights.transpose(1,2)
        if omic_mask is not None:weights=weights.masked_fill(omic_mask[:,None,:],-1e4)
        weights=weights.softmax(-1)
        wa=layer.wsi_rho(layer.wsi_omic_ffn((weights@features).squeeze(1)))
        b,_=layer.omic_wsi_attn(wsi,omic,omic,key_padding_mask=omic_mask)
        weights,features=layer.omic_wsi_pool(b.transpose(0,1));weights=weights.transpose(1,2)
        if mask is not None:weights=weights.masked_fill(mask[:,None,:],-1e4)
        wb=layer.omic_rho(layer.omic_wsi_ffn((weights.softmax(-1)@features).squeeze(1)))
        return layer.fusion_layer(torch.cat([wa,wb],-1))

    def forward(self,visual,mask,clinical,r,g,l,rm,cm,anchor,**kwargs):
        if self.name=='MGCT':
            x=self.core.wsi_net(visual).transpose(0,1);values=torch.stack([r,g,l,rm,cm],-1)
            token=[]
            for j,net in enumerate(self.core.omic_net):token.append(net(values[:,getattr(self,'genes'+str(j))].flatten(1)))
            omic=torch.stack(token,0)
            for layer in self.core.stage1_layers:stage1=self.mgct_layer(layer,x,omic,~mask)
            # Preserve upstream's use of stage1 for each stage2 block; do not
            # silently fix the disconnected first stage2 block in the baseline.
            for layer in self.core.stage2_layers:stage2=self.mgct_layer(layer,stage1[None],x,None,~mask)
            value=self.core.classifier(self.core.act(self.core.cls_drop(stage2))).squeeze(-1)
        else:
            # Padding never enters the original SR-Mamba or attention pool.
            lens=mask.sum(1);items=[]
            for n in torch.unique(lens).tolist():
                ix=torch.where(lens==n)[0];logits=self.core(visual[ix,:n])[0].reshape(-1);items.append((ix,logits))
            value=torch.stack([next(v[(ix==j).nonzero(as_tuple=True)[0][0]] for ix,v in items if bool((ix==j).any())) for j in range(len(visual))])
        return anchor+self.direct(torch.cat([r,g,l],-1)).squeeze(-1)+self.late(clinical).squeeze(-1)+self.gamma*value
