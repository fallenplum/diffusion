"""P0/P1 components for the B0--B7 protocol, with no survival-label input.

These are representation components, not completed B0--B7 survival fits.
Clinical background regressions are fitted by the caller AFTER freezing this
model.  They must use this model's own latent coordinates and train-only OOF
predictions; no differently fitted model's latent arrays may be concatenated.

The recipe's simultaneous orthogonal-row and same-spectrum random prototype
requirements generally conflict.  ``rotate_prototypes`` explicitly implements
the same-Gram alternative: a seeded orthogonal rotation of the feature axes.
This preserves each row norm and all singular values without claiming that the
prototype rows themselves are orthogonal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARMS = ("B1", "B2", "B2_MATCHED_1H", "B2_MATCHED_4H", "B3", "B4", "B5",
        "B5_CLINICAL_CAPACITY_CONTROL", "B6a", "B7")


@dataclass(frozen=True)
class ModelConfig:
    patch_dim: int = 768
    conch_text_dim: int = 768
    titan_native_dim: int = 768
    titan_text_dim: int = 768
    titan_score_dim: int = 32
    clinical_dim: int = 6
    programs: int = 50
    slots: int = 8
    width: int = 64
    heads: int = 4
    rank: int = 2
    temperature: float = 0.2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.patch_dim, self.conch_text_dim, self.titan_native_dim,
               self.titan_text_dim, self.clinical_dim, self.programs,
               self.slots, self.width, self.heads, self.rank) < 1:
            raise ValueError("NONPOSITIVE_MODEL_DIMENSION")
        if self.width % self.heads or self.temperature <= 0:
            raise ValueError("INVALID_HEAD_OR_TEMPERATURE")


@dataclass
class ImageInputs:
    patch_values: Tensor              # B,N,raw CONCH dimension
    patch_aligned: Tensor             # B,N,official CONCH projected dimension
    patch_mask: Tensor                # B,N,boolean
    titan_native: Tensor              # B,D_native, patient aggregated upstream
    titan_aligned: Tensor             # B,D_text, official projection upstream
    titan_scores: Tensor              # B,32, identical input for every arm
    clinical: Tensor                  # B,fold-local encoded clinical columns
    patch_slide: Tensor | None = None # B,N,int; each slide is pooled separately


@dataclass
class OmicsInputs:
    values: Tensor                    # B,K,3: RNA / CNA gain / CNA loss
    measured: Tensor                  # B,K,3: observed support, not imputation


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """A completely masked row is exactly zero, including its gradient."""
    mask = mask.to(dtype=torch.bool).expand_as(logits)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def rotate_prototypes(prototypes: Tensor, seed: int) -> Tensor:
    """Frozen same-Gram pseudo-prototypes; never a patient-dependent transform."""
    if prototypes.ndim != 2 or not torch.isfinite(prototypes).all():
        raise ValueError("INVALID_PROTOTYPES")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(prototypes.shape[1], prototypes.shape[1],
                         generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(random)
    return (prototypes.detach().to(device="cpu", dtype=torch.float64)
            @ rotation).to(prototypes)


class SafeCrossAttention(nn.Module):
    """Ordinary multihead attention with explicit masks and all-masked handling."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, query: Tensor, values: Tensor, valid: Tensor):
        b, q, d = query.shape
        n = values.shape[1]
        h = self.heads
        qq = self.q(query).reshape(b, q, h, d // h).transpose(1, 2)
        kk = self.k(values).reshape(b, n, h, d // h).transpose(1, 2)
        vv = self.v(values).reshape(b, n, h, d // h).transpose(1, 2)
        weights = masked_softmax(qq @ kk.transpose(-1, -2) / math.sqrt(d // h),
                                 valid[:, None, None, :])
        message = (weights @ vv).transpose(1, 2).reshape(b, q, d)
        return self.out(message), weights


class SharedProjectionCrossAttention(nn.Module):
    """Ordinary scaled-dot attention on shared learned P/G projections.

    Its internal Q/K/V/output maps are identity maps, so it adds zero parameters
    beyond the same learned input projections available to the other arms.
    One-head and multihead versions provide two exactly parameter-matched
    attention controls; they are explicit adaptations, not the native QKVO
    implementation. FLOPs still need real-input profiling.
    """
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, query: Tensor, values: Tensor, valid: Tensor):
        b, q, d = query.shape
        n, h = values.shape[1], self.heads
        qq = query.reshape(b, q, h, d // h).transpose(1, 2)
        kk = values.reshape(b, n, h, d // h).transpose(1, 2)
        weights = masked_softmax(qq @ kk.transpose(-1, -2) / math.sqrt(d // h),
                                 valid[:, None, None, :])
        return (weights @ kk).transpose(1, 2).reshape(b, q, d), weights


class SemanticRepresentation(nn.Module):
    """Shared input budget, separate image-only prediction and paired scoring.

    B1's output risk features are purely additive.  B2 uses ordinary P<->G
    attention.  B3/B4 differ only in the two frozen prototype bases.  B5/B6a/B7
    share Phase-A encoding and differ only in frozen-readout feature operators
    or the caller's auxiliary positive assignment.  B7 uses independent
    single-modality squares in place of cross-modal products.

    B2's native attention has more parameters than the concept models; its two
    named MATCHED versions instead reuse the shared learned input projections.
    The clinical-capacity control spends the same external Ridge coefficients
    on a clinical-only additive basis instead of subtracting clinical means.
    Those external coefficients must be counted and lineage-audited by caller.
    """
    def __init__(self, arm: str, conch_prototypes: Tensor,
                 titan_prototypes: Tensor, relation_prior: Tensor, *,
                 config: ModelConfig | None = None, seed: int = 31):
        super().__init__()
        self.config = c = config or ModelConfig()
        if arm not in ARMS:
            raise ValueError("UNKNOWN_REPRESENTATION_ARM")
        self.arm = arm
        expected = ((c.slots, c.conch_text_dim), (c.slots, c.titan_text_dim),
                    (c.slots, c.programs))
        if (conch_prototypes.shape, titan_prototypes.shape, relation_prior.shape) != expected:
            raise ValueError("TEXT_OR_RELATION_SHAPE_MISMATCH")
        if not (torch.isfinite(conch_prototypes).all()
                and torch.isfinite(titan_prototypes).all()
                and torch.isfinite(relation_prior).all()) or (relation_prior < 0).any():
            raise ValueError("NONFINITE_OR_NEGATIVE_PRIOR")
        if (conch_prototypes.norm(dim=-1) == 0).any() or (titan_prototypes.norm(dim=-1) == 0).any():
            raise ValueError("ZERO_TEXT_PROTOTYPE")
        pseudo = arm in ("B1", "B2", "B2_MATCHED_1H", "B2_MATCHED_4H", "B3")
        cp = rotate_prototypes(conch_prototypes, seed + 1001) if pseudo else conch_prototypes
        tp = rotate_prototypes(titan_prototypes, seed + 2003) if pseudo else titan_prototypes
        self.register_buffer("conch_prototypes", F.normalize(cp.detach().clone(), dim=-1))
        self.register_buffer("titan_prototypes", F.normalize(tp.detach().clone(), dim=-1))
        self.register_buffer("relation_prior", relation_prior.detach().clone())
        # A single shared unknown-link strength, not S*K freely learned edges.
        if (relation_prior > 0).any() and (relation_prior == 0).any():
            self.unknown_logit = nn.Parameter(torch.tensor(-3.))
        else:
            self.register_buffer("unknown_logit", torch.tensor(-3.))
        # Isolate initialization from rotation generation and the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.raw_value = nn.Linear(c.patch_dim, c.width, bias=False)
            self.aligned_key = nn.Linear(c.conch_text_dim, c.width, bias=False)
            self.query_projection = nn.Linear(c.conch_text_dim, c.width, bias=False)
            self.query_residual = nn.Parameter(torch.zeros(c.slots, c.width))
            self.titan_query_projection = nn.Linear(c.titan_text_dim, c.width, bias=False)
            self.titan_query_residual = nn.Parameter(torch.zeros(c.slots, c.width))
            self.titan_aligned_key = nn.Linear(c.titan_text_dim, c.width, bias=False)
            self.native_projection = nn.Linear(c.titan_native_dim, c.width, bias=False)
            self.score_projection = nn.Linear(c.titan_score_dim, c.width, bias=False)
            self.clinical_encoder = nn.Linear(c.clinical_dim, c.width, bias=False)
            self.gene_encoder = nn.Linear(6, c.width, bias=False)
            self.program_identity = nn.Parameter(torch.randn(c.programs, c.width) * .01)
            self.program_head = nn.Linear(c.width * 2, c.programs)
            self.u = nn.Linear(c.width, c.rank, bias=False)
            self.v = nn.Linear(c.width, c.rank, bias=False)
            if arm == "B2":
                self.g_to_p = SafeCrossAttention(c.width, c.heads)
                self.p_to_g = SafeCrossAttention(c.width, c.heads)
            elif arm in ("B2_MATCHED_1H", "B2_MATCHED_4H"):
                heads = 1 if arm == "B2_MATCHED_1H" else c.heads
                self.g_to_p = SharedProjectionCrossAttention(heads)
                self.p_to_g = SharedProjectionCrossAttention(heads)
        self.dropout = nn.Dropout(c.dropout)

    def _validate_image(self, x: ImageInputs):
        c = self.config
        b, n, d = x.patch_values.shape
        if (d != c.patch_dim or x.patch_aligned.shape != (b, n, c.conch_text_dim)
                or x.patch_mask.shape != (b, n) or x.patch_mask.dtype != torch.bool
                or x.titan_native.shape != (b, c.titan_native_dim)
                or x.titan_aligned.shape != (b, c.titan_text_dim)
                or x.titan_scores.shape != (b, c.titan_score_dim)
                or x.clinical.shape != (b, c.clinical_dim)):
            raise ValueError("IMAGE_INPUT_SHAPE_OR_MASK")
        if x.patch_slide is not None:
            if x.patch_slide.shape != (b, n) or x.patch_slide.dtype != torch.long:
                raise ValueError("PATCH_SLIDE_SHAPE_OR_DTYPE")
            if (x.patch_slide[x.patch_mask] < 0).any():
                raise ValueError("NEGATIVE_VALID_SLIDE_INDEX")
            if x.patch_mask.any() and x.patch_slide[x.patch_mask].max() >= n:
                raise ValueError("SLIDE_INDICES_MUST_BE_COMPACT_PATIENT_LOCAL_IDS")
        for tensor in (x.titan_native, x.titan_aligned, x.titan_scores, x.clinical):
            if not torch.isfinite(tensor).all():
                raise ValueError("NONFINITE_GLOBAL_OR_CLINICAL_INPUT")
        if not torch.isfinite(x.patch_values[x.patch_mask]).all() or not torch.isfinite(x.patch_aligned[x.patch_mask]).all():
            raise ValueError("NONFINITE_VALID_PATCH")

    def image_only(self, x: ImageInputs, *, view: str = "combined"):
        """No OmicsInputs/targets accepted; stage/time/event cannot enter here."""
        self._validate_image(x)
        if view not in ("combined", "clinical", "pathology"):
            raise ValueError("UNKNOWN_IMAGE_ONLY_VIEW")
        c = self.config
        mask = x.patch_mask
        raw = torch.where(mask[..., None], x.patch_values, 0.)
        aligned = F.normalize(torch.where(mask[..., None], x.patch_aligned, 0.), dim=-1)
        q = self.query_projection(self.conch_prototypes) + self.query_residual
        logits = (aligned @ self.conch_prototypes.T) / c.temperature
        logits = logits + (self.aligned_key(aligned) @ q.T) / math.sqrt(c.width)
        # Preserve equal slide contribution rather than pooling all patches
        # into a slide-count/patch-count-dependent patient message.
        slide = x.patch_slide if x.patch_slide is not None else torch.zeros_like(mask, dtype=torch.long)
        weights = torch.zeros_like(logits)
        slide_count = torch.zeros(mask.shape[0], 1, 1, device=raw.device, dtype=raw.dtype)
        max_slide = int(slide[mask].max().item()) + 1 if mask.any() else 0
        for s in range(max_slide):
            valid = mask & (slide == s)
            weights = weights + masked_softmax(logits.transpose(1, 2), valid[:, None, :]).transpose(1, 2)
            slide_count = slide_count + valid.any(-1)[:, None, None].to(raw.dtype)
        weights = weights / slide_count.clamp_min(1)
        local = torch.einsum("bns,bnd->bsd", weights, self.raw_value(raw))
        tq = self.titan_query_projection(self.titan_prototypes) + self.titan_query_residual
        titan_logits = (F.normalize(x.titan_aligned, dim=-1) @ self.titan_prototypes.T)
        titan_logits = titan_logits + (self.titan_aligned_key(x.titan_aligned) @ tq.T) / math.sqrt(c.width)
        # Per-concept global evidence scales a common full-slide value. It is
        # not a local morphology label or a substitute for official projection.
        global_value = self.native_projection(x.titan_native) + self.score_projection(x.titan_scores)
        visual = local + titan_logits[..., None] * global_value[:, None, :]
        clinical = self.clinical_encoder(x.clinical)
        if view == "clinical":
            visual = visual * 0.
        elif view == "pathology":
            clinical = clinical * 0.
        pooled = visual.mean(1)
        prediction = self.program_head(self.dropout(torch.cat([pooled, clinical], -1)))
        return {"visual_tokens": visual, "clinical_embedding": clinical,
                "image_embedding": pooled, "program_prediction": prediction,
                "patch_attention": weights.transpose(1, 2),
                "titan_query_evidence": titan_logits, "image_only_view": view}

    def encode_omics(self, genes: OmicsInputs):
        c = self.config
        if (genes.values.ndim != 3 or genes.values.shape[1:] != (c.programs, 3)
                or genes.measured.shape != genes.values.shape
                or genes.measured.dtype != torch.bool):
            raise ValueError("OMICS_SHAPE_OR_MEASURED_MASK")
        if not torch.isfinite(genes.values[genes.measured]).all():
            raise ValueError("NONFINITE_MEASURED_OMICS")
        values = torch.where(genes.measured, genes.values, 0.)
        valid = genes.measured.any(-1)
        tokens = self.gene_encoder(torch.cat([values, genes.measured.to(values.dtype)], -1))
        tokens = (tokens + self.program_identity) * valid[..., None]
        embedding = tokens.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return {"gene_tokens": tokens, "gene_mask": valid, "gene_embedding": embedding}

    def _relations(self):
        return self.relation_prior + (self.relation_prior == 0) * torch.sigmoid(self.unknown_logit)

    def pair_scores(self, image_outputs: dict, gene_outputs: dict):
        """Score ALL image/candidate-gene pairs; caller controls positives.

        B2 recomputes its ordinary cross-attention for each candidate pair.
        Other arms score independent unimodal states, so no target RNA enters
        the image-only program predictor. No diagonal-identity assumption is
        embedded in this function.
        """
        image = image_outputs["visual_tokens"]
        gene, valid = gene_outputs["gene_tokens"], gene_outputs["gene_mask"]
        relations = self._relations()
        if self.arm.startswith("B2"):
            rows = []
            for i in range(len(image)):
                visual = image[i:i+1].expand(len(gene), -1, -1)
                dm, _ = self.g_to_p(visual, gene, valid)
                gm, _ = self.p_to_g(gene, visual, torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device))
                uu = F.normalize(self.u(visual + dm), dim=-1)
                vv = F.normalize(self.v(gene + gm), dim=-1)
                numer = torch.einsum("jsr,jkr,sk,jk->j", uu, vv, relations, valid.to(uu.dtype))
                denom = torch.einsum("sk,jk->j", relations, valid.to(relations.dtype)).clamp_min(1e-12)
                rows.append(numer / denom)
            scores = torch.stack(rows)
        else:
            uu = F.normalize(self.u(image), dim=-1)
            vv = F.normalize(self.v(gene), dim=-1)
            scores = torch.einsum("isr,jkr,sk,jk->ij", uu, vv, relations, valid.to(uu.dtype))
            denom = torch.einsum("sk,jk->j", relations, valid.to(relations.dtype)).clamp_min(1e-12)
            scores = scores / denom[None, :]
        return {"scores": scores / self.config.temperature,
                "valid": valid.any(-1)[None, :].expand_as(scores)}

    def forward(self, image: ImageInputs, genes: OmicsInputs, *,
                mu_visual: Tensor | None = None, mu_gene: Tensor | None = None,
                stage: str = "pretext", view: str = "combined"):
        if stage not in ("pretext", "frozen_readout"):
            raise ValueError("UNKNOWN_REPRESENTATION_STAGE")
        image_out = self.image_only(image, view=view)
        gene_out = self.encode_omics(genes)
        visual, gene = image_out["visual_tokens"], gene_out["gene_tokens"]
        if len(visual) != len(gene):
            raise ValueError("FORWARD_REQUIRES_MATCHED_BATCH_LENGTHS")
        pre_visual, pre_gene = visual, gene
        clinical_capacity = self.arm == "B5_CLINICAL_CAPACITY_CONTROL" and stage == "frozen_readout"
        residualize = self.arm in ("B5", "B6a", "B7") and stage == "frozen_readout"
        if residualize or clinical_capacity:
            if mu_visual is None or mu_gene is None:
                raise ValueError("FROZEN_READOUT_REQUIRES_TRAIN_ONLY_OOF_BACKGROUND")
            if self.training or any(p.requires_grad for p in self.parameters()):
                raise ValueError("BACKGROUND_READOUT_REQUIRES_FROZEN_EVAL_MODEL")
            if mu_visual.shape != visual.shape or mu_gene.shape != gene.shape:
                raise ValueError("BACKGROUND_LATENT_COORDINATE_SHAPE")
            if not torch.isfinite(mu_visual).all() or not torch.isfinite(mu_gene).all():
                raise ValueError("NONFINITE_BACKGROUND")
            if residualize:
                visual, gene = visual - mu_visual, gene - mu_gene
        elif mu_visual is not None or mu_gene is not None:
            raise ValueError("BACKGROUND_ONLY_ALLOWED_IN_FROZEN_B5_B6A_B7_READOUT")
        if self.arm.startswith("B2"):
            dm, _ = self.g_to_p(visual, gene, gene_out["gene_mask"])
            gm, _ = self.p_to_g(gene, visual, torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device))
            visual, gene = visual + dm, gene + gm
        uu, vv = self.u(visual), self.v(gene)
        relations = self._relations()[None] * gene_out["gene_mask"][:, None, :]
        denom = relations.sum((1, 2)).clamp_min(1e-12)[:, None]
        if self.arm == "B7":
            # Its visual component must not change when a patient's gene
            # measured mask changes. Use a fixed relation support/denominator
            # for both additive terms; gene tokens already carry their mask.
            fixed = self._relations()
            interaction = (torch.einsum("bsr,s->br", uu.square(), fixed.sum(1))
                           + torch.einsum("bkr,k->br", vv.square(), fixed.sum(0))) / fixed.sum().clamp_min(1e-12)
        elif self.arm == "B1":
            interaction = visual.new_zeros((len(visual), self.config.rank))
        else:
            interaction = torch.einsum("bsr,bkr,bsk->br", uu, vv, relations) / denom
        if clinical_capacity:
            # Same frozen Ridge coefficients as B5, all used; no clinical x
            # observed-visual/gene product is introduced in this extra term.
            fixed = self._relations()
            clinical_basis = (torch.einsum("bsr,s->br", self.u(mu_visual).square(), fixed.sum(1))
                              + torch.einsum("bkr,k->br", self.v(mu_gene).square(), fixed.sum(0))) / fixed.sum().clamp_min(1e-12)
            interaction = interaction + clinical_basis
        return {**image_out, **gene_out,
                "pre_residual_visual_tokens": pre_visual,
                "pre_residual_gene_tokens": pre_gene,
                "interaction_features": interaction,
                "additive_image": image_out["image_embedding"],
                "additive_gene": gene_out["gene_embedding"],
                "additive_clinical": image_out["clinical_embedding"],
                "clinical_residualization_applied": residualize,
                "clinical_additive_capacity_control_applied": clinical_capacity,
                "stage": stage,
                "mechanism_status": "FROZEN_FEATURE_OPERATOR" if stage == "frozen_readout" else "PRETEXT_UNRESIDUALIZED",
                "pair": self.pair_scores(image_out, gene_out)}

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        # Program head participates only in pretraining and is not a survival
        # predictor. Do not claim identical pretrain/inference parameter counts.
        auxiliary_only = sum(p.numel() for p in self.program_head.parameters())
        if self.arm == "B1":
            auxiliary_only += sum(p.numel() for p in self.u.parameters())
            auxiliary_only += sum(p.numel() for p in self.v.parameters())
            if isinstance(self.unknown_logit, nn.Parameter):
                auxiliary_only += self.unknown_logit.numel()
        return {"total": total, "pretraining_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "auxiliary_only_parameters": auxiliary_only,
                "prediction_active_parameters_static": total - auxiliary_only,
                "dss_trainable_in_this_module": 0,
                "prediction_used_requires_path_audit": True}
