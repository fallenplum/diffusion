"""Outcome-free P1-R operators, preserving every historical P1 source byte.

TEXTLESS uses independent random visual anchors and native visual statistics;
it never initializes from a text prototype or reads an explicit concept score.
The score-route control matches training first/second moments, not conditional
distributions or medical truth. Dual channels normalize measured known and
unknown edges separately; their mixing coefficient is an engineering control.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import ImageInputs, ModelConfig, SemanticRepresentation


@dataclass(frozen=True)
class RepairSpec:
    source_arm: str = "B4"
    query_mode: str = "true"
    score_route: str = "actual"
    relation_mode: str = "legacy"
    known_fraction: float = .5
    lambda_pair: float = .1
    shuffled_map: bool = False
    wrong_aux: bool = False

    def __post_init__(self):
        if self.query_mode not in ("true", "random", "textless", "visual_random"):
            raise ValueError("P1R_QUERY_MODE")
        if self.score_route not in ("actual", "train_moment_surrogate", "native_random"):
            raise ValueError("P1R_SCORE_ROUTE")
        if self.relation_mode not in ("legacy", "dual") or self.lambda_pair not in (0., .03, .1):
            raise ValueError("P1R_RELATION_OR_PAIR_WEIGHT")
        if self.relation_mode == "dual" and self.known_fraction not in (.5, .75):
            raise ValueError("P1R_PRELOCKED_DUAL_MIX")
        if self.query_mode == "textless" and self.score_route != "native_random":
            raise ValueError("TEXTLESS_MUST_NOT_KEEP_EXPLICIT_TEXT_SCORE_ROUTE")


def random_visual_basis(rows: int, columns: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(torch.randn(rows, columns, generator=generator), dim=-1)


class ScoreMomentSurrogate(nn.Module):
    """Fixed train-fitted native-visual surrogate for the 32-score route."""
    def __init__(self, projection, native_mean, whitening, coloring, score_mean,
                 fit_patients: int, moment_error: float):
        super().__init__()
        for name, tensor in (("projection", projection), ("native_mean", native_mean),
                             ("whitening", whitening), ("coloring", coloring),
                             ("score_mean", score_mean)):
            self.register_buffer(name, tensor.detach().float().clone())
        self.fit_patients = fit_patients
        self.moment_error = moment_error

    def forward(self, native):
        return ((native @ self.projection - self.native_mean)
                @ self.whitening @ self.coloring + self.score_mean)

    @classmethod
    def fit(cls, native_train: Tensor, score_train: Tensor, *, seed=2461033):
        if (native_train.ndim != 2 or score_train.ndim != 2
                or len(native_train) != len(score_train)
                or len(native_train) <= score_train.shape[1]
                or not torch.isfinite(native_train).all() or not torch.isfinite(score_train).all()):
            raise ValueError("SCORE_SURROGATE_TRAIN_SHAPE_OR_SUPPORT")
        x, y = native_train.detach().cpu().double(), score_train.detach().cpu().double()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection, _ = torch.linalg.qr(torch.randn(x.shape[1], y.shape[1], generator=generator, dtype=torch.float64), mode="reduced")
        z = x @ projection
        mz, my = z.mean(0), y.mean(0)
        cz = (z - mz).T @ (z - mz) / (len(z) - 1)
        cy = (y - my).T @ (y - my) / (len(y) - 1)
        ez, vz = torch.linalg.eigh(cz)
        ey, vy = torch.linalg.eigh(cy)
        if ez.min() <= ez.max() * 1e-10:
            raise ValueError("SCORE_SURROGATE_NATIVE_PROJECTION_RANK_DEFICIENT")
        whitening = (vz * ez.rsqrt()[None, :]) @ vz.T
        coloring = (vy * ey.clamp_min(0.).sqrt()[None, :]) @ vy.T
        prediction = (z - mz) @ whitening @ coloring + my
        cp = (prediction - prediction.mean(0)).T @ (prediction - prediction.mean(0)) / (len(y) - 1)
        error = max(float((prediction.mean(0) - my).abs().max()), float((cp - cy).abs().max()))
        if error > 1e-8 * max(1., float(cy.abs().max())):
            raise ValueError("SCORE_SURROGATE_FIRST_SECOND_MOMENT_MATCH_FAILED")
        return cls(projection, mz, whitening, coloring, my, len(x), error)


class RepairRepresentation(SemanticRepresentation):
    def __init__(self, conch_prototypes, titan_prototypes, relation_prior, *,
                 spec: RepairSpec, config: ModelConfig | None = None,
                 seed=31, score_surrogate: ScoreMomentSurrogate | None = None):
        config = config or ModelConfig()
        no_text_query = spec.query_mode in ("textless", "visual_random")
        if no_text_query:
            # Do not access values/norms/Gram of the supplied text prototypes.
            cp = random_visual_basis(config.slots, config.conch_text_dim, seed + 600007)
            tp = random_visual_basis(config.slots, config.titan_text_dim, seed + 600011)
            base_arm = spec.source_arm
        else:
            cp, tp = conch_prototypes, titan_prototypes
            base_arm = "B3" if spec.query_mode == "random" else spec.source_arm
        super().__init__(base_arm, cp, tp, relation_prior, config=config, seed=seed)
        self.spec = spec
        if spec.relation_mode == "dual" and isinstance(self.unknown_logit, nn.Parameter):
            value = self.unknown_logit.detach().clone()
            delattr(self, "unknown_logit")
            self.register_buffer("unknown_logit", value)
        if spec.query_mode == "textless":
            generator = torch.Generator(device="cpu").manual_seed(2461199)
            projection, _ = torch.linalg.qr(torch.randn(config.titan_native_dim, config.titan_score_dim,
                                                       generator=generator), mode="reduced")
            self.register_buffer("native_random_score_projection", projection)
        if spec.score_route == "train_moment_surrogate":
            if score_surrogate is None:
                raise ValueError("SCORE_ROUTE_CONTROL_REQUIRES_TRAIN_ONLY_FIT")
            self.score_surrogate = score_surrogate
        elif score_surrogate is not None:
            raise ValueError("UNEXPECTED_SCORE_SURROGATE")

    def image_only(self, image: ImageInputs, *, view="combined"):
        if self.spec.query_mode == "textless":
            if self.config.patch_dim != self.config.conch_text_dim or self.config.titan_native_dim != self.config.titan_text_dim:
                raise ValueError("TEXTLESS_VISUAL_SUBSTITUTION_REQUIRES_MATCHED_DIMENSIONS")
            image = replace(image,
                patch_aligned=F.normalize(image.patch_values, dim=-1),
                titan_aligned=F.normalize(image.titan_native, dim=-1),
                titan_scores=F.normalize(image.titan_native, dim=-1) @ self.native_random_score_projection)
        elif self.spec.score_route == "train_moment_surrogate":
            image = replace(image, titan_scores=self.score_surrogate(image.titan_native))
        return super().image_only(image, view=view)

    def channel_weights(self, gene_mask: Tensor):
        """Global measured slot×program channels, plus per-program fallbacks."""
        known = (self.relation_prior > 0)[None] & gene_mask[:, None, :]
        unknown = (self.relation_prior == 0)[None] & gene_mask[:, None, :]
        nk, nu = known.sum((1, 2)), unknown.sum((1, 2))
        wk = known.to(self.relation_prior.dtype) / nk.clamp_min(1)[:, None, None]
        wu = unknown.to(self.relation_prior.dtype) / nu.clamp_min(1)[:, None, None]
        mix = torch.full_like(nk, self.spec.known_fraction, dtype=self.relation_prior.dtype)
        mix = torch.where(nk == 0, 0., mix)
        mix = torch.where((nu == 0) & (nk > 0), 1., mix)
        weights = mix[:, None, None] * wk + (1 - mix)[:, None, None] * wu
        return weights, {"known_edge_count": nk, "unknown_edge_count": nu,
                         "known_effective_fraction": mix,
                         "valid": (nk + nu) > 0,
                         "program_no_known_fallback": gene_mask & ~known.any(1),
                         "program_no_unknown_fallback": gene_mask & ~unknown.any(1),
                         "known_weights": wk, "unknown_weights": wu}

    def pair_scores(self, image_outputs, gene_outputs):
        if self.spec.relation_mode == "legacy":
            return super().pair_scores(image_outputs, gene_outputs)
        image, gene = image_outputs["visual_tokens"], gene_outputs["gene_tokens"]
        weights, diagnostic = self.channel_weights(gene_outputs["gene_mask"])
        uu, vv = F.normalize(self.u(image), dim=-1), F.normalize(self.v(gene), dim=-1)
        known = torch.einsum("isr,jkr,jsk->ij", uu, vv, diagnostic["known_weights"])
        unknown = torch.einsum("isr,jkr,jsk->ij", uu, vv, diagnostic["unknown_weights"])
        fraction = diagnostic["known_effective_fraction"][None]
        score = fraction * known + (1 - fraction) * unknown
        return {"scores": score / self.config.temperature,
                "valid": diagnostic["valid"][None, :].expand_as(score),
                "known_unweighted_mean": known, "unknown_unweighted_mean": unknown,
                "known_weighted_contribution": fraction * known,
                "unknown_weighted_contribution": (1 - fraction) * unknown,
                "relation_diagnostics": diagnostic}

    def mechanism_from_tokens(self, visual, gene, gene_mask, *, additive=False):
        uu, vv = self.u(visual), self.v(gene)
        if self.spec.relation_mode == "dual":
            if additive:
                # Fixed support for the visual term avoids a hidden dependence
                # of visual features on the other modality's measured mask.
                fixed, _ = self.channel_weights(torch.ones_like(gene_mask))
                return (torch.einsum("bsr,bsk->br", uu.square(), fixed)
                        + torch.einsum("bkr,bsk->br", vv.square() * gene_mask[..., None], fixed))
            weights, _ = self.channel_weights(gene_mask)
            return torch.einsum("bsr,bkr,bsk->br", uu, vv, weights)
        if additive:
            fixed = self._relations()
            return (torch.einsum("bsr,s->br", uu.square(), fixed.sum(1))
                    + torch.einsum("bkr,k->br", vv.square() * gene_mask[..., None], fixed.sum(0))) / fixed.sum().clamp_min(1e-12)
        weights = self._relations()[None] * gene_mask[:, None, :]
        return torch.einsum("bsr,bkr,bsk->br", uu, vv, weights) / weights.sum((1, 2)).clamp_min(1e-12)[:, None]

    def forward(self, image, genes, *, view="combined", stage="pretext"):
        if stage != "pretext":
            raise ValueError("USE_EXPLICIT_FROZEN_READOUT_OPERATOR_AFTER_PHASE_A")
        result = super().forward(image, genes, view=view, stage="pretext")
        if self.spec.relation_mode == "dual":
            result["interaction_features"] = self.mechanism_from_tokens(
                result["pre_residual_visual_tokens"], result["pre_residual_gene_tokens"], result["gene_mask"])
        return result


def frozen_readout_operator(model, outputs, mu_visual=None, mu_gene=None, *, arm="B4"):
    """Preserve main effects; return exactly two active mechanism dimensions."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("READOUT_OPERATOR_REQUIRES_FROZEN_EVAL_ENCODER")
    import numpy as np
    original_visual = outputs["pre_residual_visual_tokens"]
    numpy_output = isinstance(original_visual, np.ndarray)
    device = next(model.parameters()).device
    visual = torch.as_tensor(original_visual, device=device)
    gene = torch.as_tensor(outputs["pre_residual_gene_tokens"], device=device)
    gene_mask = torch.as_tensor(outputs["gene_mask"], device=device, dtype=torch.bool)
    residual = arm in ("B5", "B6a", "B7")
    capacity = arm == "B5_CLINICAL_CAPACITY_CONTROL"
    if residual or capacity:
        mu_visual = torch.as_tensor(mu_visual, device=device) if mu_visual is not None else None
        mu_gene = torch.as_tensor(mu_gene, device=device) if mu_gene is not None else None
        if mu_visual is None or mu_gene is None or mu_visual.shape != visual.shape or mu_gene.shape != gene.shape:
            raise ValueError("READOUT_REQUIRES_SAME_COORDINATE_TRAIN_OOF_BACKGROUND")
        if not torch.isfinite(mu_visual).all() or not torch.isfinite(mu_gene).all():
            raise ValueError("NONFINITE_CLINICAL_BACKGROUND")
        if residual:
            visual = visual - mu_visual
            gene = torch.where(gene_mask[..., None], gene - mu_gene, torch.zeros_like(gene))
    elif mu_visual is not None or mu_gene is not None:
        raise ValueError("UNEXPECTED_CLINICAL_BACKGROUND")
    if arm in ("B1", "B7"):
        mechanism = model.mechanism_from_tokens(visual, gene, gene_mask, additive=True)
    else:
        mechanism = model.mechanism_from_tokens(visual, gene, gene_mask)
    if capacity:
        mechanism = mechanism + model.mechanism_from_tokens(mu_visual, mu_gene,
                                                            torch.ones_like(gene_mask), additive=True)
    if numpy_output:
        mechanism = mechanism.detach().cpu().numpy()
    return {"additive_image": outputs["additive_image"], "additive_gene": outputs["additive_gene"],
            "additive_clinical": outputs["additive_clinical"], "mechanism": mechanism,
            "interaction_features": mechanism,
            "clinical_residualization_applied": residual,
            "clinical_additive_control_applied": capacity}
