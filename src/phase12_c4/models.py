"""Identical C4 backbone; only constant RNA / risk-head controls differ."""
import torch
from torch import nn
from src.phase11_semantic.models import SemanticModel


class RNAControl(nn.Module):
    def __init__(self, encoder, constant=False):
        super().__init__();self.encoder=encoder;self.constant_mode=constant
        # Added AFTER complete backbone initialization; no extra RNG draw.
        if constant:self.constant=nn.Parameter(encoder.weight.detach().mean(1)*8)
    def forward(self,r,mode='msa'):
        if not self.constant_mode:return self.encoder(r,mode)
        raw=self.constant[None].expand(len(r),-1,-1)+self.encoder.bias
        raw=self.encoder.norm(torch.nn.functional.elu(raw))
        return raw,self.encoder.inter(raw,raw)


class ZeroAnchor(nn.Module):
    def __init__(self,dc):
        super().__init__();self.register_buffer('weight',torch.zeros(1,dc+32))
    def forward(self,x):return x.new_zeros((len(x),1))


class C4Model(SemanticModel):
    def __init__(self,clinical_mask,membership,d=64,kind='real',no_anchor=False):
        super().__init__('C4',clinical_mask,membership,d)
        self.rna=RNAControl(self.rna,kind=='null')
        if no_anchor:self.anchor=ZeroAnchor(clinical_mask.shape[1])
        self.control_kind=kind;self.no_anchor=no_anchor


def convert_parent_state(state):
    return {('rna.encoder.'+k[4:] if k.startswith('rna.') else k):v for k,v in state.items()}
