"""Low-capacity CONCH gated MIL and spatial niche residual Cox networks."""
from __future__ import annotations

import torch
from torch import nn


def segment_softmax(score, group, n_groups):
    score = score.reshape(-1)
    maxima = torch.full((n_groups,), -torch.inf, dtype=score.dtype, device=score.device)
    maxima.scatter_reduce_(0, group, score, reduce="amax", include_self=True)
    weights = torch.exp(score - maxima[group])
    denom = torch.zeros(n_groups, dtype=score.dtype, device=score.device).index_add_(0, group, weights)
    return weights / denom[group].clamp_min(1e-12)


def segment_pool(values, weights, group, n_groups):
    result = torch.zeros((n_groups, values.shape[-1]), dtype=values.dtype, device=values.device)
    return result.index_add_(0, group, values * weights[:, None])


class GatedAttention(nn.Module):
    def __init__(self, dim=128, attention_dim=32, dropout=0.05):
        super().__init__()
        self.v = nn.Linear(dim, attention_dim)
        self.u = nn.Linear(dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h, group, n_groups):
        score = self.w(self.dropout(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h)))).squeeze(-1)
        weight = segment_softmax(score, group, n_groups)
        return segment_pool(h, weight, group, n_groups), weight


class PatientRiskHead(nn.Module):
    def __init__(self, in_dim, clinical_dim, p4=False, dropout=0.05, alpha_init=0.05):
        super().__init__()
        self.p4 = p4
        self.query = nn.Linear(clinical_dim, 128, bias=False)
        self.register_buffer("constant_context", torch.ones(clinical_dim) / max(clinical_dim, 1) ** 0.5)
        self.key = nn.Linear(128, 128, bias=False)
        self.global_attention = GatedAttention(128, 32, dropout)
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 32), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def query_for_patient(self, clinical):
        context = clinical if self.p4 else self.constant_context.unsqueeze(0).expand(len(clinical), -1)
        return self.query(context)

    def residual(self, global_patient, rare_patient, clinical, static):
        return self.alpha * self.head(torch.cat([global_patient, rare_patient, clinical, static], dim=-1)).squeeze(-1)


class ConchGatedABMIL(nn.Module):
    """768D patches -> one gated attention vector per slide -> equal-slide patient pool."""
    def __init__(self, clinical_dim, dropout=0.05, alpha_init=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.project = nn.Linear(768, 128)
        self.activation = nn.GELU()
        self.attention = GatedAttention(128, 32, dropout)
        self.risk = nn.Sequential(nn.LayerNorm(128 + clinical_dim + 32), nn.Linear(128 + clinical_dim + 32, 32), nn.GELU(), nn.Linear(32, 1))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.clinical_dim = clinical_dim
        nn.init.normal_(self.risk[-1].weight, std=0.01)
        nn.init.zeros_(self.risk[-1].bias)

    def forward(self, batch, clinical, static):
        h = self.activation(self.project(self.norm(batch["features"])))
        slide_pool, weights = self.attention(h, batch["patch_slide"], batch["n_slides"])
        patient_pool = torch.zeros((batch["n_patients"], 128), device=h.device, dtype=h.dtype)
        patient_pool.index_add_(0, batch["slide_patient"], slide_pool)
        counts = torch.bincount(batch["slide_patient"], minlength=batch["n_patients"]).to(h.dtype)
        patient_pool /= counts[:, None].clamp_min(1)
        return self.alpha * self.risk(torch.cat([patient_pool, clinical, static], -1)).squeeze(-1), {"attention": weights, "patient_pool": patient_pool}


class SpatialNicheSurv(nn.Module):
    """One sparse region message layer plus global and rare-region attention."""
    def __init__(self, clinical_dim, p4=False, no_topology=False, dropout=0.05, rare_k=8, alpha_init=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(788)
        self.project = nn.Linear(788, 128)
        self.self_message = nn.Linear(128, 128)
        self.neighbor_message = nn.Linear(128, 128, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.no_topology = no_topology
        self.rare_k = rare_k
        self.niche = PatientRiskHead(256 + clinical_dim + 32, clinical_dim, p4, dropout, alpha_init)

    def forward(self, batch, clinical, static):
        x = self.project(self.norm(batch["features"]))
        adjacency = batch["adjacency"]
        neighbors = x if self.no_topology else torch.sparse.mm(adjacency, x)
        h = self.dropout(torch.nn.functional.gelu(self.self_message(x) + self.neighbor_message(neighbors)))
        slide_global, _ = self.niche.global_attention(h, batch["region_slide"], batch["n_slides"])
        q = self.niche.query_for_patient(clinical)
        q_slide = q[batch["slide_patient"]]
        keys = self.niche.key(h)
        score = (q_slide[batch["region_slide"]] * keys).sum(-1) / (128 ** 0.5)
        rare_slides = []
        for start, end in batch["slide_offsets"]:
            local = score[start:end]
            k = min(self.rare_k, len(local))
            idx = torch.topk(local, k=k, sorted=False).indices + start
            rare_slides.append(torch.sum(torch.softmax(score[idx], dim=0)[:, None] * h[idx], dim=0))
        rare_slide = torch.stack(rare_slides)
        global_patient = torch.zeros((batch["n_patients"], 128), device=h.device, dtype=h.dtype)
        rare_patient = torch.zeros_like(global_patient)
        global_patient.index_add_(0, batch["slide_patient"], slide_global)
        rare_patient.index_add_(0, batch["slide_patient"], rare_slide)
        counts = torch.bincount(batch["slide_patient"], minlength=batch["n_patients"]).to(h.dtype)
        global_patient /= counts[:, None].clamp_min(1)
        rare_patient /= counts[:, None].clamp_min(1)
        delta = self.niche.residual(global_patient, rare_patient, clinical, static)
        return delta, {"global_patient": global_patient, "rare_patient": rare_patient,
                       "alpha": self.niche.alpha, "risk_delta_abs_mean": delta.abs().mean(),
                       "region_embedding": h}


def efron_cox_loss(risk, time, event):
    loss = risk.new_zeros(())
    event_count = int(event.sum().detach().item())
    if event_count < 1:
        raise ValueError("Cox training partition has no DSS events")
    for time_value in torch.unique(time[event > 0], sorted=True):
        deaths = (time == time_value) & (event > 0)
        count = int(deaths.sum().item())
        risk_set = time >= time_value
        shift = risk[risk_set].max()
        exp_risk = torch.exp(risk - shift)
        denominator = exp_risk[risk_set].sum()
        tied = exp_risk[deaths].sum()
        for step in range(count):
            loss = loss + torch.log(denominator - (step / count) * tied) + shift
        loss = loss - risk[deaths].sum()
    return loss / event_count
