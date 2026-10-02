from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from src.phase_m3r2.models import SimpleFusionModel, ClinicalOnlyModel
from src.phase_m2.nn_blocks import MultimodalBackbone

DIM=96
@contextmanager
def isolated_seed(seed:int):
    with torch.random.fork_rng(devices=[]): torch.manual_seed(int(seed)); yield

class ResidualRiskModel(nn.Module):
    """Frozen clinical offset plus additive low-capacity pathology/omics residuals."""
    def __init__(self, clinical_dim:int, variant:str, seed:int, fold:int):
        super().__init__(); self.variant=variant; self.backbone=MultimodalBackbone(clinical_dim,dim=DIM,path_tokens=32)
        with isolated_seed(700000+seed%1000000*17+fold):
            self.path_head=nn.Sequential(nn.Linear(DIM,32),nn.GELU(),nn.Linear(32,1)) if variant in {"RP","RPO"} else None
            self.omic_head=nn.Sequential(nn.Linear(DIM,32),nn.GELU(),nn.Linear(32,1)) if variant in {"RO","RPO"} else None
        for h in (self.path_head,self.omic_head):
            if h is not None: nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
    def forward(self,pathology,rna,cna,clinical,offset):
        enc=self.backbone.encode(pathology,rna,cna,clinical); delta=offset*0
        if self.path_head is not None: delta=delta+self.path_head(enc.pathology.mean(1)).squeeze(-1)
        if self.omic_head is not None: delta=delta+self.omic_head(enc.omics.mean(1)).squeeze(-1)
        return offset+delta,delta
    def delta_parts(self,pathology,rna,cna,clinical):
        enc=self.backbone.encode(pathology,rna,cna,clinical); p=enc.pathology.mean(1);o=enc.omics.mean(1)
        dp=self.path_head(p).squeeze(-1) if self.path_head is not None else torch.zeros(p.shape[0],device=p.device)
        do=self.omic_head(o).squeeze(-1) if self.omic_head is not None else torch.zeros(p.shape[0],device=p.device)
        return dp,do

def build_residual(variant,clinical_dim,seed,fold): return ResidualRiskModel(clinical_dim,variant,seed,fold)
