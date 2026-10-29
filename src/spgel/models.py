"""Small source-task models with a separately callable image-only branch.

Paired A5 scores recompute both conditional reads for EACH candidate pair.
Conditioning a positive embedding once and comparing it to unrelated positives
would expose the positive identity through its context; that shortcut is absent.
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import GeneBatch, ImageBatch, PROGRAMS, require


ARMS = ("S_A0", "S_A1", "S_A2", "S_A3", "S_A4", "S_A5")
NEURAL_ARMS = ARMS[1:]


def _seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "little") % (2**31 - 1)


def initialized(factory: Callable, seed: int, label: str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_seed(seed, label))
        return factory()


def masked_softmax(scores: torch.Tensor, valid: torch.Tensor):
    allowed = torch.broadcast_to(valid, scores.shape)
    active = allowed.any(-1)
    safe = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(active[..., None], safe, torch.zeros_like(safe))
    return safe.softmax(-1) * allowed, active


def symbol_identity(symbols) -> torch.Tensor:
    """Fixed numeric identity, not a claim of pretrained gene semantics."""
    return torch.tensor([[int(x) / 127.5 - 1 for x in hashlib.sha256(str(g).encode()).digest()[:8]]
                         for g in symbols], dtype=torch.float32)


class MeanGraphBlock(nn.Module):
    """A residual mean-aggregation GNN; padding/zero-neighbor rows are explicit."""
    def __init__(self, width):
        super().__init__()
        self.mix = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, center, neighbors, relative_xy, valid):
        # The local window includes self; GraphSAGE's neighbor mean excludes it.
        other = valid & (relative_xy.square().sum(-1) > 1e-10)
        mean = torch.where(other[..., None], neighbors, torch.zeros_like(neighbors)).sum(1)
        mean = mean / other.sum(1).clamp_min(1)[:, None]
        update = F.gelu(self.mix(torch.cat((center, mean), -1)))
        mixed = self.norm(center + update)
        return torch.where(other.any(1)[:, None], mixed, center)


class ImageBackbone(nn.Module):
    def __init__(self, width, hidden, *, use_graph, seed):
        super().__init__()
        self.width, self.use_graph = int(width), bool(use_graph)
        self.project = initialized(lambda: nn.Sequential(nn.Linear(1536, width), nn.LayerNorm(width), nn.GELU()),
                                   seed, "common_image_projection")
        self.feedforward = initialized(lambda: nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
            nn.Linear(hidden, width), nn.LayerNorm(width)), seed, "common_image_feedforward")
        self.graph = initialized(lambda: MeanGraphBlock(width), seed, "mean_graph") if use_graph else None

    def forward(self, batch: ImageBatch):
        batch.validate()
        h = self.project(batch.features)
        if self.graph is None:
            h = h + self.feedforward(h)
            return h, h[:, None], torch.ones((len(h), 1), dtype=torch.bool, device=h.device)
        valid = batch.neighbor_mask
        raw_neighbor = torch.where(valid[..., None], batch.neighbor_features, torch.zeros_like(batch.neighbor_features))
        n = self.project(raw_neighbor.reshape(-1, 1536)).reshape(*raw_neighbor.shape[:2], self.width)
        if self.graph is not None:
            h = self.graph(h, n, batch.neighbor_xy_um, valid)
        h = h + self.feedforward(h)
        # Exactly one center; discard the duplicate self entries in the registered graph.
        nonself = valid & (batch.neighbor_xy_um.square().sum(-1) > 1e-10)
        tokens = torch.cat((h[:, None], n), 1)
        mask = torch.cat((torch.ones((len(h), 1), dtype=torch.bool, device=h.device), nonself), 1)
        return h, tokens, mask


class ProgramWindowRead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = int(width)
        self.queries = nn.Parameter(torch.randn(4, width) * .02)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, tokens, valid, *, query=None, return_attention=False):
        # Supports both [B,N,D] and paired-candidate [A,C,N,D] windows.
        query = self.queries.expand(*tokens.shape[:-2], 4, self.width) if query is None else query
        require(query.shape == (*tokens.shape[:-2], 4, self.width) and valid.shape == tokens.shape[:-1],
                "PROGRAM_WINDOW_QUERY_OR_MASK_SHAPE")
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        score = torch.einsum("...kd,...nd->...kn", query, self.key(tokens)) / math.sqrt(self.width)
        weights, active = masked_softmax(score, valid[..., None, :])
        result = torch.einsum("...kn,...nd->...kd", weights, self.value(tokens))
        out = dict(tokens=result, program_mask=active)
        if return_attention:
            out["attention"] = weights
        return out


class OriginalGeneAttention(nn.Module):
    """Exact linear-key/value SparseSetAttn without a B×U×D materialization.

    Both keys and values contain expression. Algebraic factorization only avoids
    repeating their shared identity and expression slopes for every spot/pair.
    Signed up/down attention is normalized separately and then subtracted.
    """
    def __init__(self, up, down, symbols, *, width=128):
        super().__init__()
        up, down = torch.as_tensor(up, dtype=torch.bool), torch.as_tensor(down, dtype=torch.bool)
        require(up.shape == down.shape == (4, len(symbols)) and up.any(1).all() and not (up & down).any(),
                "ORIGINAL_GENE_SIGNED_MEMBERSHIP")
        self.width, self.genes = int(width), len(symbols)
        self.register_buffer("up", up)
        self.register_buffer("down", down)
        self.register_buffer("identity", symbol_identity(symbols))
        self.identity_key = nn.Linear(8, width, bias=False)
        self.identity_value = nn.Linear(8, width, bias=False)
        self.expression_key = nn.Parameter(torch.randn(width) * .02)
        self.expression_value = nn.Parameter(torch.randn(width) * .02)
        self.queries = nn.Parameter(torch.randn(4, width) * .02)

    def _side(self, score, observed, membership, values, id_value):
        allowed = observed[..., None, :] & membership
        weights, active = masked_softmax(score, allowed)
        identity = torch.einsum("...ku,ud->...kd", weights, id_value)
        expression = torch.sum(weights * values[..., None, :], -1)
        token = identity + expression[..., None] * self.expression_value
        return token, active, weights

    def read(self, genes: GeneBatch, *, query=None, candidates=None, return_attention=False):
        genes.validate(self.genes)
        b, device = len(genes.values), genes.values.device
        if candidates is None:
            candidates = torch.arange(b, device=device)[:, None]
        require(candidates.ndim == 2 and candidates.dtype == torch.long and candidates.device == device
                and ((candidates >= 0) & (candidates < b)).all(), "PAIRED_GENE_CANDIDATE_RANGE")
        if query is None:
            query = self.queries.expand(*candidates.shape, 4, self.width)
        require(query.shape == (*candidates.shape, 4, self.width), "PAIRED_GENE_QUERY_SHAPE")
        selected_values = genes.values[candidates]
        observed = genes.observed[candidates]
        selected_values = torch.where(observed, selected_values, torch.zeros_like(selected_values))
        identity = self.identity[genes.gene_ids]
        id_key, id_value = self.identity_key(identity), self.identity_value(identity)
        score = torch.einsum("ackd,ud->acku", query, id_key)
        score = score + torch.einsum("ackd,d->ack", query, self.expression_key)[..., None] * selected_values[..., None, :]
        score = score / math.sqrt(self.width)
        up = self.up[:, genes.gene_ids]
        down = self.down[:, genes.gene_ids]
        u, um, uw = self._side(score, observed, up, selected_values, id_value)
        d, dm, dw = self._side(score, observed, down, selected_values, id_value)
        valid = um & (dm | ~down.any(-1))
        token = torch.where(valid[..., None], u - d, torch.zeros_like(u))
        coverage = (observed[..., None, :] & (up | down)).sum(-1) / (up | down).sum(-1).clamp_min(1)
        out = dict(tokens=token, program_mask=valid, coverage=coverage)
        if return_attention:
            out.update(up_attention=uw, down_attention=dw)
        return out

    def forward(self, genes: GeneBatch, *, return_attention=False):
        result = self.read(genes, return_attention=return_attention)
        return {k: v[:, 0] for k, v in result.items()}


class SpatialProgramEncoder(nn.Module):
    """Fixed pilot arms. All predictions are image-only; pairs are another API."""
    def __init__(self, arm, up, down, symbols, *, width=128, hidden=128,
                 image_mlp_hidden=None, tissue_program_mean=None, seed=31):
        super().__init__()
        require(arm in NEURAL_ARMS and width == 128 and hidden >= 1, "SPGEL_NEURAL_ARM_OR_WIDTH")
        self.arm, self.width = arm, width
        self.has_slots = arm in ("S_A4", "S_A5")
        self.has_genes = arm in ("S_A3", "S_A4", "S_A5")
        self.bidirectional = arm == "S_A5"
        ff = image_mlp_hidden if arm == "S_A1" and image_mlp_hidden is not None else hidden
        self.backbone = ImageBackbone(width, ff, use_graph=arm != "S_A1", seed=seed)
        self.regression_head = initialized(lambda: nn.Linear(width, 4), seed, "ordinary_program_head") if not self.has_slots else None
        if self.has_slots:
            self.window = initialized(lambda: ProgramWindowRead(width), seed, "program_window")
            self.tissue_head = initialized(lambda: nn.Linear(width, 3), seed, "image_tissue_head")
            mean = np.zeros((3, 4), np.float32) if tissue_program_mean is None else np.asarray(tissue_program_mean, np.float32)
            require(mean.shape == (3, 4) and np.isfinite(mean).all(), "TRAIN_ONLY_TISSUE_PROGRAM_BACKGROUND")
            self.tissue_background = nn.Parameter(torch.tensor(mean.copy()))
            self.slot_weight = initialized(lambda: nn.Parameter(torch.randn(4, width) * .02), seed, "independent_slot_readout")
            self.slot_bias = nn.Parameter(torch.zeros(4))
        else:
            self.window = self.tissue_head = None
            self.register_parameter("tissue_background", None)
            self.register_parameter("slot_weight", None)
            self.register_parameter("slot_bias", None)
        self.gene_encoder = initialized(lambda: OriginalGeneAttention(up, down, symbols, width=width),
                                        seed, "original_gene_encoder") if self.has_genes else None
        if self.bidirectional:
            def small_projection():
                layer = nn.Linear(width, width, bias=False)
                nn.init.xavier_uniform_(layer.weight, gain=.1)
                return layer
            self.gene_to_image = initialized(small_projection, seed, "gene_to_image_query")
            self.image_to_gene = initialized(small_projection, seed, "image_to_gene_query")
        else:
            self.gene_to_image = self.image_to_gene = None

    def image_only_predict(self, image: ImageBatch, *, return_attention=False):
        """TASK A/C: signature deliberately accepts no genes, truth or depth."""
        require(isinstance(image, ImageBatch), "IMAGE_ONLY_REQUIRES_IMAGE_BATCH_NOT_PAIRED_DATA")
        h, windows, valid = self.backbone(image)
        out = dict(image_embedding=F.normalize(h, dim=-1, eps=1e-12),
                   window_tokens=windows, window_mask=valid)
        if self.has_slots:
            local = self.window(windows, valid, return_attention=return_attention)
            tissue_logits = self.tissue_head(h)
            tissue_probs = tissue_logits.softmax(-1)
            background = tissue_probs @ self.tissue_background
            residual = (local["tokens"] * self.slot_weight).sum(-1) / math.sqrt(self.width) + self.slot_bias
            out.update(program_predictions=background + residual, tissue_logits=tissue_logits,
                predicted_tissue_probabilities=tissue_probs, program_background=background,
                program_residual=residual, visual_program_tokens=local["tokens"])
            if return_attention:
                out["program_attention"] = local["attention"]
        else:
            out["program_predictions"] = self.regression_head(h)
        return out

    def forward(self, image: ImageBatch):
        return self.image_only_predict(image)

    def encode_gene_programs(self, genes: GeneBatch, *, paired_mode=False):
        require(paired_mode is True and self.gene_encoder is not None, "GENES_REQUIRE_EXPLICIT_PAIRED_MODE")
        return self.gene_encoder(genes)

    def paired_scores(self, image_outputs, genes: GeneBatch, image_rows, gene_rows, *, paired_mode=False,
                      return_details=False, _gene_base=None):
        """Score every candidate using its own image_i / measured genes_j pair.

        IDs/absolute positions/depth/tissue truth are not accepted. Candidate row
        tensors only gather already-encoded inputs, and are never embedded.
        """
        require(paired_mode is True and self.gene_encoder is not None, "PAIRED_SCORING_MODE_REQUIRED")
        require(image_rows.shape == gene_rows.shape and image_rows.ndim == 2 and image_rows.dtype == gene_rows.dtype == torch.long,
                "PAIRED_SCORE_ROW_SHAPES")
        n = len(genes.values)
        require(image_rows.device == gene_rows.device == genes.values.device
                and ((image_rows >= 0) & (image_rows < n)).all()
                and ((gene_rows >= 0) & (gene_rows < n)).all(), "PAIRED_SCORE_ROW_RANGE")
        molecular_base = self.gene_encoder(genes) if _gene_base is None else _gene_base
        require(molecular_base["tokens"].shape == (n, 4, self.width)
                and molecular_base["program_mask"].shape == (n, 4), "UNIMODAL_GENE_CACHE_SHAPE")
        if not self.bidirectional:
            molecular = molecular_base
            g, mask = molecular["tokens"], molecular["program_mask"]
            if self.arm == "S_A3":
                pooled = torch.where(mask[..., None], g, torch.zeros_like(g)).sum(1) / mask.sum(1).clamp_min(1)[:, None]
                g = F.normalize(pooled, dim=-1, eps=1e-12)
                score = (image_outputs["image_embedding"][image_rows] * g[gene_rows]).sum(-1)
                available = mask.any(1)[gene_rows]
            else:
                v = F.normalize(image_outputs["visual_program_tokens"], dim=-1, eps=1e-12)[image_rows]
                g = F.normalize(g, dim=-1, eps=1e-12)[gene_rows]
                m = mask[gene_rows]
                score = torch.where(m, (v * g).sum(-1), 0.).sum(-1) / m.sum(-1).clamp_min(1)
                available = m.any(-1)
            return dict(scores=score, valid=available)
        # Shared baseline tokens only encode one modality each.
        base_g = molecular_base
        g_context = base_g["tokens"][gene_rows]
        v_context = image_outputs["visual_program_tokens"][image_rows]
        windows = image_outputs["window_tokens"][image_rows]
        window_mask = image_outputs["window_mask"][image_rows]
        q_image = self.window.queries + self.gene_to_image(g_context)
        cond_v = self.window(windows, window_mask, query=q_image, return_attention=return_details)
        q_gene = self.gene_encoder.queries + self.image_to_gene(v_context)
        cond_g = self.gene_encoder.read(genes, query=q_gene, candidates=gene_rows, return_attention=return_details)
        mask = cond_v["program_mask"] & cond_g["program_mask"]
        score = F.cosine_similarity(cond_v["tokens"], cond_g["tokens"], dim=-1, eps=1e-12)
        result = dict(scores=torch.where(mask, score, 0.).sum(-1) / mask.sum(-1).clamp_min(1), valid=mask.any(-1))
        if return_details:
            result.update(conditioned_image_tokens=cond_v["tokens"], conditioned_gene_tokens=cond_g["tokens"],
                image_attention=cond_v["attention"], gene_up_attention=cond_g["up_attention"],
                gene_down_attention=cond_g["down_attention"], condition_is_recomputed_for_each_candidate=True)
        return result

    def descriptor(self):
        return dict(schema="spgel_source_program_encoder_v1", arm=self.arm, input_dim=1536,
            width=self.width, program_order=list(PROGRAMS), feature_unit="unit_L2_UNI2_float32",
            use_spatial_neighbor=self.backbone.use_graph, use_program_slots=self.has_slots,
            tissue_truth_is_prediction_input=False, depth_is_prediction_input=False,
            RNA_is_prediction_input=False, gene_branch_is_paired_only=self.has_genes,
            candidate_specific_dual_reread=self.bidirectional,
            gene_identity="fixed_symbol_SHA256_8D_not_medical_semantics",
            official_HVG_head_in_this_pilot=False,
            parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            registered_parameters=sum(p.numel() for p in self.parameters()))


def match_image_mlp_capacity(up, down, symbols, *, width=128, hidden=128):
    """Choose A1 width from parameter arithmetic only; no outcomes or fitting."""
    target = SpatialProgramEncoder("S_A5", up, down, symbols, width=width, hidden=hidden, seed=0)
    target_count = target.descriptor()["parameters"]
    counts = [SpatialProgramEncoder("S_A1", up, down, symbols, width=width, hidden=hidden,
                image_mlp_hidden=h, seed=0).descriptor()["parameters"] for h in (1, 2)]
    slope = counts[1] - counts[0]
    exact = (target_count - (counts[0] - slope)) / slope
    candidates = sorted({max(1, math.floor(exact)), max(1, math.ceil(exact))})
    rows = []
    for value in candidates:
        count = SpatialProgramEncoder("S_A1", up, down, symbols, width=width, hidden=hidden,
            image_mlp_hidden=value, seed=0).descriptor()["parameters"]
        rows.append(dict(hidden=value, parameters=count, absolute_difference=abs(count - target_count)))
    result = min(rows, key=lambda r: (r["absolute_difference"], r["hidden"]))
    require(result["absolute_difference"] / target_count <= .01, "IMAGE_MLP_CAPACITY_MATCH_EXCEEDS_ONE_PERCENT")
    return dict(**result, target_A5_parameters=target_count, selection_uses_labels=False,
                dummy_or_zero_gradient_parameters_added=False)
