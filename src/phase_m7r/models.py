from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from src.phase_m2.mcat import MCATModel, SixSignatureEncoder
from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.nn_blocks import ClinicalEncoder, RiskHead
from src.phase_m3r2.models import SimpleFusionModel
DIM=96
@contextmanager
def _isolated_seed(seed:int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed)); yield
class GatedABMIL(nn.Module):
    def __init__(self,dim=DIM,hidden=48):
        super().__init__(); self.projection=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,dim),nn.GELU()); self.v=nn.Linear(dim,hidden); self.u=nn.Linear(dim,hidden); self.w=nn.Linear(hidden,1); self.norm=nn.LayerNorm(dim)
    def forward(self,patches):
        h=self.projection(patches.float()); score=self.w(torch.tanh(self.v(h))*torch.sigmoid(self.u(h))).squeeze(-1); weights=torch.softmax(score,dim=1); return self.norm((weights.unsqueeze(-1)*h).sum(dim=1)),weights
class AdaptivePath(nn.Module):
    def __init__(self,dim=DIM):
        super().__init__(); self.projection=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,dim),nn.GELU()); self.norm=nn.LayerNorm(dim)
    def forward(self,patches):
        h=self.projection(patches.float()); return self.norm(h.mean(dim=1)),torch.full((h.shape[0],h.shape[1]),1.0/h.shape[1],device=h.device,dtype=h.dtype)
class FiLMModule(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(DIM,64),nn.GELU(),nn.Linear(64,DIM*2)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
@dataclass
class Streams: path:torch.Tensor; omic:torch.Tensor; clinical:torch.Tensor
class M7RRepresentation(nn.Module):
    def __init__(self,clinical_dim,widths,seed,fold,adaptive=False):
        super().__init__(); common_seed=170000000+(seed%1000000)*11+fold
        with _isolated_seed(common_seed):
            self.path_pool=AdaptivePath() if adaptive else GatedABMIL(); self.six_encoder=SixSignatureEncoder(widths,DIM)
            layer=nn.TransformerEncoderLayer(DIM,4,dim_feedforward=DIM*2,dropout=0.25,activation="relu",batch_first=True); self.omic_transformer=nn.TransformerEncoder(layer,num_layers=2); self.omic_attention_head=GatedAttentionPool(DIM); self.omic_rho=nn.Sequential(nn.Linear(DIM,DIM),nn.ReLU(),nn.Dropout(0.25)); self.clinical_encoder=ClinicalEncoder(clinical_dim,dim=DIM); self.fusion=nn.Sequential(nn.Linear(DIM*2,DIM),nn.ReLU(),nn.Linear(DIM,DIM),nn.ReLU()); self.clinical_gate=nn.Sequential(nn.Linear(DIM*2,DIM),nn.GELU(),nn.LayerNorm(DIM)); self.head=RiskHead(DIM)
    def encode(self,pathology,rna,cna,clinical,groups):
        path,weights=self.path_pool(pathology); omic=self.six_encoder(groups); omic=self.omic_transformer(omic); omic,_=self.omic_attention_head(omic); omic=self.omic_rho(omic); clinical_repr=self.clinical_encoder(clinical)[:,0]; return Streams(path,omic,clinical_repr),weights
class M7RModel(nn.Module):
    def __init__(self,variant,clinical_dim,widths,seed,fold):
        super().__init__();
        if variant not in {"A0","F1","F2","F3","F6"}: raise ValueError(variant)
        self.variant=variant; self.representation=M7RRepresentation(clinical_dim,widths,seed,fold,adaptive=variant=="A0"); interaction_seed=270000000+(seed%1000000)*13+fold
        with _isolated_seed(interaction_seed):
            self.residual=nn.Sequential(nn.Linear(DIM*2,64),nn.GELU(),nn.Linear(64,DIM)) if variant=="F2" else None; self.film_o=FiLMModule() if variant in {"F3","F6"} else None
    def _condition(self,path,omic):
        diag={}
        if self.variant in {"A0","F1"}: return path,omic,diag
        if self.variant=="F2": return path+self.residual(torch.cat([path,omic],dim=-1)),omic,diag
        gamma,beta=self.film_o(omic); path_mod=path+torch.tanh(gamma)*F.layer_norm(path,(DIM,))+beta; diag.update({"gamma_o":gamma,"beta_o":beta,"path_correction_norm":(path_mod-path).norm(dim=-1)}); return path_mod,omic,diag
    def forward(self,pathology,rna,cna,clinical,groups=None):
        if groups is None: raise ValueError("M7R models require six-signature groups")
        streams,_=self.representation.encode(pathology,rna,cna,clinical,groups); path,omic,_=self._condition(streams.path,streams.omic); fused=self.representation.fusion(torch.cat([path,omic],dim=-1)); context=self.representation.clinical_gate(torch.cat([fused,streams.clinical],dim=-1)); return self.representation.head(context),pathology.new_zeros(())
    @torch.no_grad()
    def modulation_diagnostics(self,pathology,rna,cna,clinical,groups):
        self.eval(); streams,_=self.representation.encode(pathology,rna,cna,clinical,groups); _,_,diag=self._condition(streams.path,streams.omic); return diag
    @torch.no_grad()
    def audit_forward(self,pathology,rna,cna,clinical,groups):
        self.eval(); streams,_=self.representation.encode(pathology,rna,cna,clinical,groups); path,omic,_=self._condition(streams.path,streams.omic); fused=self.representation.fusion(torch.cat([path,omic],dim=-1)); context=self.representation.clinical_gate(torch.cat([fused,streams.clinical],dim=-1)); return self.representation.head(context),pathology.new_zeros(())
def control_seed(method,seed,fold):
    from src.phase_m2.common import stable_int
    return 800000+stable_int(f"m7r-control:{method}:{seed}:{fold}",modulo=1900000000)
def build_model(variant,clinical_dim,widths,seed,fold):
    if variant=="F0": return MCATModel(clinical_dim,signature_widths=widths,variant="A0")
    if variant in {"A0","F1","F2","F3","F6"}: return M7RModel(variant,clinical_dim,widths,seed,fold)
    if variant=="B0":
        cs=control_seed(variant,seed,fold); return SimpleFusionModel(clinical_dim,variant="S3",variant_seed=cs,concat_seed=cs+1)
    raise ValueError(variant)

