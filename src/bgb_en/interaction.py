"""Low-capacity R3 cross-modal message controls, without survival training.

The root trainer supplies the frozen CP/RNA risk offset and the common gamma.
This module returns an unbounded linear risk correction; it does not insert a
tanh, learn gamma, fit preprocessing, or read clinical/outcome labels.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


ARMS = ("F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")


def _require(value: Any, message: str) -> None:
    if not bool(value):
        raise ValueError(message)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exactly zero for absent evidence, including all-masked rows."""
    mask = torch.broadcast_to(mask.to(device=scores.device, dtype=torch.bool), scores.shape)
    present = mask.any(dim=dim, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~mask, -torch.inf), torch.zeros_like(scores))
    return torch.softmax(safe, dim=dim)*mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    typed_mask = mask.to(values.dtype).unsqueeze(-1)
    return (values*typed_mask).sum(dim)/typed_mask.sum(dim).clamp_min(1)


def path_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean within each WSI, followed by equal nonempty-WSI mean."""
    return masked_mean(masked_mean(values, mask, -2), mask.any(-1), -2)


class LowRankUpdate(nn.Module):
    """Small nonzero output initialization; exactly zero for a zero message."""
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.normal_(self.up.weight, std=.02)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(message))


class LowRankGate(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(2*width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.up.weight, std=.02)

    def forward(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(torch.nn.functional.gelu(self.down(torch.cat((state, message), -1)))))


def _validate_permutation(permutation: torch.Tensor | None, mask: torch.Tensor,
                          name: str) -> torch.Tensor | None:
    if permutation is None:
        return None
    _require(permutation.shape == mask.shape and permutation.dtype == torch.long,
             name+" must be a long tensor with the target-value mask shape")
    _require(permutation.device == mask.device, name+" must be on the evidence device")
    n = mask.shape[-1]
    _require(((permutation >= 0) & (permutation < n)).all(), name+" index out of range")
    expected = torch.arange(n, device=mask.device).expand_as(permutation)
    _require(torch.equal(permutation.sort(-1).values, expected), name+" must be a row-wise bijection")
    _require(torch.equal(mask.gather(-1, permutation), mask), name+" cannot exchange real values and padding")
    return permutation


class BidirectionalInteraction(nn.Module):
    """Pooled main effects plus independently inspectable directional messages.

    ``OtoP`` means an omics query reads *the current patient's patch values*,
    updating the omics slots. ``PtoO`` means a patch query reads the patient's
    molecular values, updating patch states.  F5 is one simultaneous update;
    F6 applies the exact same weights twice. F8/F9 equal F5 here: the external
    encoder, rather than this module, changes their visual/molecular backbone.

    Query-only interventions accept explicit ``query_omics``/``query_path``
    context. They never replace any own-patient keys, values, masks or pooled
    main effects. The distinct ``whole_omics`` control replaces all molecular
    evidence and therefore may change its main effect. Mean contexts must be
    computed by the trainer from its training partition.
    """
    def __init__(self, arm: str = "F5", patch_dim: int = 64, omics_dim: int = 32,
                 width: int = 32, rank: int = 4, topk: int = 64,
                 clinical_dim: int | None = None):
        super().__init__()
        _require(arm in ARMS, "unknown R3 interaction arm")
        _require(min(patch_dim, omics_dim, width, rank, topk) > 0, "positive interaction dimensions required")
        self.arm, self.width, self.topk = arm, width, topk
        self.patch_dim, self.omics_dim = patch_dim, omics_dim
        self.clinical_dim = clinical_dim  # Accepted for a shared trainer API; F2-F10 do not use it.
        self.use_otop = arm not in ("F2", "F4")
        self.use_ptoo = arm not in ("F2", "F3")
        self.rounds = 2 if arm == "F6" else 1
        self.competition = arm == "F7"
        self.reject = arm == "F10"
        self.patch_project = nn.Sequential(nn.Linear(patch_dim, width, bias=False), nn.LayerNorm(width))
        self.omics_project = nn.Sequential(nn.Linear(omics_dim, width, bias=False), nn.LayerNorm(width))
        self.pooled_path_head = nn.Linear(width, 1, bias=False)
        self.pooled_omics_head = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.pooled_path_head.weight, std=.02)
        nn.init.normal_(self.pooled_omics_head.weight, std=.02)
        if arm != "F2":
            # A common compatibility kernel is used by the two normalizations.
            self.path_match = nn.Linear(width, width, bias=False)
            self.omics_match = nn.Linear(width, width, bias=False)
        if self.use_otop:
            self.path_value = nn.Linear(width, width, bias=False)
            self.update_omics = LowRankUpdate(width, rank)
            self.otop_head = nn.Linear(width, 1, bias=False)
            nn.init.normal_(self.otop_head.weight, std=.02)
        if self.use_ptoo:
            self.omics_value = nn.Linear(width, width, bias=False)
            self.update_path = LowRankUpdate(width, rank)
            self.ptoo_head = nn.Linear(width, 1, bias=False)
            nn.init.normal_(self.ptoo_head.weight, std=.02)
        if self.reject:
            self.null_omics = nn.Linear(width, 1)
            self.null_path = nn.Linear(width, 1)
            self.gate_omics = LowRankGate(width, rank)
            self.gate_path = LowRankGate(width, rank)
            for null in (self.null_omics, self.null_path):
                nn.init.normal_(null.weight, std=.02)
                nn.init.zeros_(null.bias)

    @staticmethod
    def _project(module: nn.Module, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return module(torch.where(mask[..., None], values, torch.zeros_like(values)))*mask[..., None]

    def _omics_to_path(self, p: torch.Tensor, g: torch.Tensor, qg: torch.Tensor,
                       pm: torch.Tensor, gm: torch.Tensor, qgm: torch.Tensor,
                       permutation: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score = torch.einsum("bkd,bsnd->bskn", self.omics_match(qg), self.path_match(p))/math.sqrt(self.width)
        valid = pm[:, :, None, :] & gm[:, None, :, None] & qgm[:, None, :, None]
        null = score.new_zeros(score.shape[:-1])
        if self.reject:
            null_score = self.null_omics(qg).squeeze(-1)[:, None, :, None].expand(*score.shape[:-1], 1)
            extended = torch.cat((score, null_score), -1)
            null_valid = (gm & qgm)[:, None, :, None] & pm.any(-1)[:, :, None, None]
            weight_all = masked_softmax(extended, torch.cat((valid, null_valid), -1))
            weight, null = weight_all[..., :-1], weight_all[..., -1]
        elif self.competition:
            # Different molecular queries compete for a patch before each query
            # renormalizes its retained actual-patch mass. No value is pooled
            # or replaced before the fixed-top64 selection.
            weight = masked_softmax(score, valid, dim=-2)
            keep = min(self.topk, weight.shape[-1])
            chosen = torch.topk(weight.masked_fill(~valid, -1), keep, dim=-1).indices
            selected = torch.zeros_like(valid).scatter(-1, chosen, True) & valid
            weight = weight*selected
            weight = weight/weight.sum(-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
        else:
            weight = masked_softmax(score, valid)
        if permutation is not None:
            weight = weight.gather(-1, permutation[:, :, None, :].expand_as(weight))
        by_slide = torch.einsum("bskn,bsnd->bskd", weight, self.path_value(p))
        # Only real slides are averaged, never a patient-dependent slide sum.
        real_slide = pm.any(-1)
        message = (by_slide*real_slide[:, :, None, None]).sum(1)/real_slide.sum(1)[:, None, None].clamp_min(1)
        update = self.update_omics(message)
        gate = self.gate_omics(g, message) if self.reject else torch.ones_like(update)
        update = update*gate*(gm & qgm)[..., None]
        return update, {"attention": weight, "null": null, "message": message, "gate": gate, "score": score}

    def _path_to_omics(self, p: torch.Tensor, g: torch.Tensor, qp: torch.Tensor,
                       pm: torch.Tensor, gm: torch.Tensor, qpm: torch.Tensor,
                       permutation: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score = torch.einsum("bsnd,bkd->bsnk", self.path_match(qp), self.omics_match(g))/math.sqrt(self.width)
        valid = (pm & qpm)[..., None] & gm[:, None, None, :]
        null = score.new_zeros(score.shape[:-1])
        if self.reject:
            null_score = self.null_path(qp)
            null_valid = (pm & qpm)[..., None] & gm.any(-1)[:, None, None, None]
            weight_all = masked_softmax(torch.cat((score, null_score), -1), torch.cat((valid, null_valid), -1))
            weight, null = weight_all[..., :-1], weight_all[..., -1]
        else:
            weight = masked_softmax(score, valid)
        if permutation is not None:
            weight = weight.gather(-1, permutation[:, None, None, :].expand_as(weight))
        message = torch.einsum("bsnk,bkd->bsnd", weight, self.omics_value(g))
        update = self.update_path(message)
        gate = self.gate_path(p, message) if self.reject else torch.ones_like(update)
        update = update*gate*(pm & qpm)[..., None]
        return update, {"attention": weight, "null": null, "message": message, "gate": gate, "score": score}

    def forward(self, patch_states: torch.Tensor, patch_mask: torch.Tensor,
                omics_tokens: torch.Tensor, omics_mask: torch.Tensor,
                clinical: torch.Tensor | None = None, *,
                query_omics: torch.Tensor | None = None,
                query_omics_mask: torch.Tensor | None = None,
                query_path: torch.Tensor | None = None,
                query_path_mask: torch.Tensor | None = None,
                whole_omics: torch.Tensor | None = None,
                whole_omics_mask: torch.Tensor | None = None,
                context_omics: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
                context_omics_mask: torch.Tensor | None = None,
                permute_attention: bool = False,
                attention_patch_permutation: torch.Tensor | None = None,
                attention_omics_permutation: torch.Tensor | None = None,
                interaction_off: bool = False,
                return_details: bool = True) -> torch.Tensor | dict[str, Any]:
        p, g, pm, gm = patch_states, omics_tokens, patch_mask, omics_mask
        _require(p.ndim == 4 and p.shape[-1] == self.patch_dim, "patch_states must be [B,S,N,patch_dim]")
        _require(g.ndim == 3 and g.shape[-1] == self.omics_dim and g.shape[0] == p.shape[0],
                 "omics_tokens must be [B,K,omics_dim]")
        _require(pm.shape == p.shape[:-1] and gm.shape == g.shape[:-1] and pm.dtype == gm.dtype == torch.bool,
                 "evidence masks must be bool with matching token dimensions")
        _require(p.device == g.device == pm.device == gm.device, "evidence and masks must share a device")
        _require(p.shape[1] > 0 and p.shape[2] > 0 and g.shape[1] > 0, "use all-false masks, not zero-size token axes")
        if context_omics is not None:
            _require(query_omics is None and query_omics_mask is None, "context_omics and explicit query_omics are aliases; choose one")
            if isinstance(context_omics, tuple):
                _require(len(context_omics) == 2 and context_omics_mask is None, "context tuple is (tokens, mask)")
                query_omics, query_omics_mask = context_omics
            else:
                query_omics = context_omics
                # Tensor-only legacy callers assume identical token support.
                # Callers swapping patients with differing masks must supply
                # the actual source mask or the (tokens, mask) tuple.
                query_omics_mask = gm if context_omics_mask is None else context_omics_mask
        _require((whole_omics is None) == (whole_omics_mask is None), "whole-omics values and mask must be provided together")
        if whole_omics is not None:
            _require(query_omics is None and query_path is None, "whole-omics and query-only interventions must be separate")
            _require(whole_omics.shape == g.shape and whole_omics_mask.shape == gm.shape,
                     "whole-omics replacement must match the canonical shape")
            _require(whole_omics_mask.dtype == torch.bool, "whole-omics mask must be bool")
            _require(whole_omics.device == g.device == whole_omics_mask.device, "whole-omics replacement device mismatch")
            g, gm = whole_omics, whole_omics_mask
        _require(torch.isfinite(p[pm]).all() and torch.isfinite(g[gm]).all(), "real evidence contains NaN/Inf")
        _require((query_omics is None) == (query_omics_mask is None), "query-omics context needs its own mask")
        _require((query_path is None) == (query_path_mask is None), "query-path context needs its own mask")
        qg, qp = None, None
        if query_omics is not None:
            _require(query_omics.shape == g.shape and query_omics_mask.shape == gm.shape
                     and query_omics_mask.dtype == torch.bool, "query-omics shape or mask mismatch")
            _require(query_omics.device == g.device == query_omics_mask.device, "query-omics context device mismatch")
            _require(torch.isfinite(query_omics[query_omics_mask]).all(), "query-omics context contains NaN/Inf")
            qg = self._project(self.omics_project, query_omics, query_omics_mask)
        if query_path is not None:
            _require(query_path.shape == p.shape and query_path_mask.shape == pm.shape
                     and query_path_mask.dtype == torch.bool, "query-path shape or mask mismatch")
            _require(query_path.device == p.device == query_path_mask.device, "query-path context device mismatch")
            _require(torch.isfinite(query_path[query_path_mask]).all(), "query-path context contains NaN/Inf")
            qp = self._project(self.patch_project, query_path, query_path_mask)
        if permute_attention:
            _require(attention_patch_permutation is None, "boolean and explicit patch permutations are aliases; choose one")
            attention_patch_permutation = torch.arange(pm.shape[-1], device=pm.device).expand_as(pm).clone()
            # Fixed cyclic reassignment of valid patch weights to fixed values;
            # padding stays fixed and no RNG, outcome or clinical label is read.
            for batch in range(len(pm)):
                for slide in range(pm.shape[1]):
                    real = torch.where(pm[batch, slide])[0]
                    attention_patch_permutation[batch, slide, real] = real.roll(1)
        patch_permutation = _validate_permutation(attention_patch_permutation, pm, "patch attention permutation")
        omics_permutation = _validate_permutation(attention_omics_permutation, gm, "omics attention permutation")
        p0 = self._project(self.patch_project, p, pm)
        g0 = self._project(self.omics_project, g, gm)
        pooled_path_risk = self.pooled_path_head(path_mean(p0, pm)).squeeze(-1)
        pooled_omics_risk = self.pooled_omics_head(masked_mean(g0, gm, -2)).squeeze(-1)
        ps, gs = p0, g0
        trace = []
        if self.arm != "F2" and not interaction_off:
            for _ in range(self.rounds):
                # Both directions read the same pre-update states. There is no
                # accidental sequential use of an already-updated other branch.
                pd, gd = torch.zeros_like(ps), torch.zeros_like(gs)
                current: dict[str, Any] = {}
                if self.use_otop:
                    gd, current["omics_to_patch"] = self._omics_to_path(ps, gs, gs if qg is None else qg,
                        pm, gm, gm if qg is None else query_omics_mask, patch_permutation)
                if self.use_ptoo:
                    pd, current["path_to_omics"] = self._path_to_omics(ps, gs, ps if qp is None else qp,
                        pm, gm, pm if qp is None else query_path_mask, omics_permutation)
                ps, gs = ps+pd, gs+gd
                trace.append(current)
        zero = pooled_path_risk.new_zeros(pooled_path_risk.shape)
        otop_risk = self.otop_head(masked_mean(gs-g0, gm, -2)).squeeze(-1) if self.use_otop else zero
        ptoo_risk = self.ptoo_head(path_mean(ps-p0, pm)).squeeze(-1) if self.use_ptoo else zero
        interaction_risk = otop_risk+ptoo_risk
        delta = pooled_path_risk+pooled_omics_risk+interaction_risk
        if not return_details:
            return delta
        diagnostics: dict[str, Any] = {"arm": self.arm, "rounds": len(trace),
            "registered_parameters": self.parameter_count(), "clinical_used": False,
            "dropout": 0.0, "risk_readout": "linear; external gamma; no tanh",
            "omics_query_override": query_omics is not None,
            "path_query_override": query_path is not None,
            "whole_omics_replaced": whole_omics is not None,
            "patch_attention_assignment_permuted": patch_permutation is not None,
            "omics_attention_assignment_permuted": omics_permutation is not None,
            "interaction_off": bool(interaction_off)}
        for direction in ("omics_to_patch", "path_to_omics"):
            rows = [step[direction] for step in trace if direction in step]
            if rows:
                probability = rows[-1]["attention"]
                entropy = -(probability*probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(-1)
                null_probability = rows[-1]["null"]
                entropy = entropy-null_probability*null_probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
                diagnostics[direction+"_entropy"] = entropy.detach()
                diagnostics[direction+"_null"] = rows[-1]["null"].detach()
                diagnostics[direction+"_message_l2"] = rows[-1]["message"].norm(dim=-1).detach()
        return {"risk": delta, "delta_scalar": delta, "pooled_path_risk": pooled_path_risk,
                "pooled_omics_risk": pooled_omics_risk, "interaction_risk": interaction_risk,
                "OtoP_risk": otop_risk, "PtoO_risk": ptoo_risk,
                "path_initial": p0, "omics_initial": g0, "path_updated": ps, "omics_updated": gs,
                "trace": trace, "diagnostics": diagnostics}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_interaction(arm: str, **kwargs: Any) -> BidirectionalInteraction:
    return BidirectionalInteraction(arm=arm, **kwargs)


InteractionNetwork = BidirectionalInteraction
