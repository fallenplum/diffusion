"""Frozen-feature A0 and local A1/A2 spatial composition predictors.

All distances are physical micrometres. A1 and A2 use the same module; the
external objective's relative-loss coefficient is their only difference.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialPrediction:
    states: Tensor
    mu: Tensor
    log_mu: Tensor
    valid_spots: Tensor


def _dictionary(dictionary: Tensor) -> Tensor:
    d = torch.as_tensor(dictionary, dtype=torch.float32).detach().clone()
    if d.ndim != 2 or min(d.shape) < 1 or d.shape[1] > 1024:
        raise ValueError("dictionary must be nonempty [state, gene], G <= 1024")
    if not torch.isfinite(d).all() or (d < 0).any():
        raise ValueError("dictionary must be finite and nonnegative")
    if not torch.allclose(d.sum(1), torch.ones(d.shape[0]), atol=1e-6, rtol=1e-6):
        raise ValueError("dictionary rows must sum to one")
    return d


def _valid_features(features: Tensor, spot_mask: Tensor | None, feature_dim: int):
    if features.ndim != 2 or features.shape[1] != feature_dim or not features.is_floating_point():
        raise ValueError("features must be floating point [spot, frozen_feature_dimension]")
    valid = torch.ones(len(features), dtype=torch.bool, device=features.device) if spot_mask is None else spot_mask
    if valid.shape != (len(features),) or valid.dtype != torch.bool or valid.device != features.device:
        raise ValueError("spot_mask must be boolean [spot] on feature device")
    if not torch.isfinite(features[valid]).all():
        raise ValueError("valid frozen features must be finite")
    return torch.where(valid[:, None], features, 0.), valid


def physical_neighbors(coords_um: Tensor, wsi_ids, valid: Tensor, *,
                       radius_um: float, neighbor_cap: int = 16) -> tuple[Tensor, Tensor]:
    """Return radius-limited nearest neighbors, including self within the cap.

    Equal distances are resolved by physical x/y coordinates, independently of
    input order. Repeated coordinates inside one WSI are rejected as duplicate
    physical samples. WSI identifiers are never returned as model features.
    """
    if not math.isfinite(radius_um) or radius_um <= 0 or neighbor_cap < 1:
        raise ValueError("positive physical radius and neighbor cap are required")
    n = len(valid)
    if coords_um.shape != (n, 2) or coords_um.device != valid.device:
        raise ValueError("coords_um must be [spot,2] on feature device")
    if not coords_um.is_floating_point() or not torch.isfinite(coords_um[valid]).all():
        raise ValueError("valid coordinates must be finite physical micrometres")
    if isinstance(wsi_ids, Tensor):
        if wsi_ids.ndim != 1:
            raise ValueError("wsi_ids must be one-dimensional")
        wsi_ids = wsi_ids.detach().cpu().tolist()
    ids = list(wsi_ids)
    if len(ids) != n:
        raise ValueError("wsi_ids length must equal spot count")
    groups = {}
    for i in torch.where(valid)[0].tolist():
        value = ids[i]
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError("valid spots require a known WSI ID")
        try:
            groups.setdefault(value, []).append(i)
        except TypeError as exc:
            raise ValueError("WSI IDs must be scalar and hashable") from exc
    neighbors = torch.zeros((n, neighbor_cap), dtype=torch.long, device=valid.device)
    allowed = torch.zeros((n, neighbor_cap), dtype=torch.bool, device=valid.device)
    for positions in groups.values():
        idx = torch.tensor(positions, dtype=torch.long, device=valid.device)
        xy = coords_um[idx].double()
        if len(torch.unique(xy, dim=0)) != len(idx):
            raise ValueError("duplicate physical coordinates within one WSI")
        order = torch.argsort(xy[:, 1], stable=True)
        order = order[torch.argsort(xy[order, 0], stable=True)]
        canonical = idx[order]
        distances = torch.cdist(coords_um[idx].double(), coords_um[canonical].double())
        closest = torch.argsort(distances, dim=1, stable=True)[:, :min(neighbor_cap, len(idx))]
        k = closest.shape[1]
        neighbors[idx, :k] = canonical[closest]
        allowed[idx, :k] = distances.gather(1, closest) <= radius_um
    return neighbors, allowed


class _CompositionOutput(nn.Module):
    def __init__(self, dictionary: Tensor):
        super().__init__()
        self.register_buffer("dictionary", _dictionary(dictionary), persistent=True)

    def _output(self, logits: Tensor, valid: Tensor) -> SpatialPrediction:
        states = logits.softmax(-1)
        mu = states @ self.dictionary.to(dtype=states.dtype)
        # A numerical log floor is not an imputed observed count. The measured
        # data mask remains external to the predictor and objective.
        log_mu = mu.clamp_min(torch.finfo(mu.dtype).tiny).log()
        return SpatialPrediction(torch.where(valid[:, None], states, 0.),
                                 torch.where(valid[:, None], mu, 0.),
                                 torch.where(valid[:, None], log_mu, 0.), valid)


class LinearCompositionEncoder(_CompositionOutput):
    """A0: one linear map on frozen image features, then state/gene simplex."""
    def __init__(self, dictionary: Tensor, *, feature_dim: int = 1536):
        super().__init__(dictionary)
        self.feature_dim = feature_dim
        self.state_head = nn.Linear(feature_dim, self.dictionary.shape[0])

    def forward(self, features: Tensor, coords_um: Tensor | None = None,
                wsi_ids=None, spot_mask: Tensor | None = None) -> SpatialPrediction:
        x, valid = _valid_features(features, spot_mask, self.feature_dim)
        return self._output(self.state_head(x), valid)


class _LocalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must divide into heads")
        self.heads, self.head_dim = heads, hidden // heads
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.coordinate_bias = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.output = nn.Linear(hidden, hidden)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, hidden), nn.Dropout(dropout))

    def forward(self, x: Tensor, coords: Tensor, neighbors: Tensor, allowed: Tensor,
                valid: Tensor, radius_um: float) -> Tensor:
        n, hidden = x.shape
        if n == 0:
            return x
        q, k, v = self.qkv(self.norm1(x)).reshape(n, 3, self.heads, self.head_dim).unbind(1)
        key, value = k[neighbors], v[neighbors]
        score = (q[:, None] * key).sum(-1) / math.sqrt(self.head_dim)
        delta_xy = (coords[neighbors] - coords[:, None]) / radius_um
        score = score + self.coordinate_bias(delta_xy.to(x.dtype))
        score = score.masked_fill(~allowed[:, :, None], -torch.inf)
        score = torch.where(valid[:, None, None], score, 0.)
        attention = score.softmax(dim=1).masked_fill(~allowed[:, :, None], 0.)
        message = (self.attention_dropout(attention)[..., None] * value).sum(1).reshape(n, hidden)
        x = x + self.residual_dropout(self.output(message))
        x = x + self.ffn(self.norm2(x))
        return torch.where(valid[:, None], x, 0.)


class LocalSpatialEncoder(_CompositionOutput):
    """A1/A2: fixed two-block local architecture; radius requires a data lock.

    ``radius_um`` intentionally has no default. The radius is an external ST
    development choice, not an unspecified pixel or array-index distance.
    """
    def __init__(self, dictionary: Tensor, *, radius_um: float,
                 feature_dim: int = 1536, hidden: int = 128, layers: int = 2,
                 heads: int = 4, ffn: int = 256, dropout: float = .1,
                 neighbor_cap: int = 16):
        super().__init__(dictionary)
        if not math.isfinite(radius_um) or radius_um <= 0 or neighbor_cap < 1:
            raise ValueError("positive radius_um and neighbor_cap must be locked")
        self.feature_dim, self.radius_um, self.neighbor_cap = feature_dim, radius_um, neighbor_cap
        self.input_projection = nn.Linear(feature_dim, hidden)
        self.blocks = nn.ModuleList([_LocalBlock(hidden, heads, ffn, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.state_head = nn.Linear(hidden, self.dictionary.shape[0])

    def forward(self, features: Tensor, coords_um: Tensor, wsi_ids,
                spot_mask: Tensor | None = None) -> SpatialPrediction:
        x, valid = _valid_features(features, spot_mask, self.feature_dim)
        neighbors, allowed = physical_neighbors(coords_um, wsi_ids, valid,
                                                radius_um=self.radius_um, neighbor_cap=self.neighbor_cap)
        coords = torch.where(valid[:, None], coords_um, 0.)
        x = torch.where(valid[:, None], self.input_projection(x), 0.)
        for block in self.blocks:
            x = block(x, coords, neighbors, allowed, valid, self.radius_um)
        return self._output(self.state_head(self.final_norm(x)), valid)
