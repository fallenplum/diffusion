from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


CONCEPTS = 8
PATH_INPUT_DIM = 1536
PATH_PROJECTION_DIM = 16


@contextmanager
def isolated_seed(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


class ConceptFusionResidualModel(nn.Module):
    """Small concept-conditioned residual model on a frozen Clinical anchor."""

    def __init__(self, variant: str, seed: int, fold: int):
        super().__init__()
        allowed = {"K-O", "K-P", "K-ADD", "K-MSG", "K-SHUFFLE"}
        if variant not in allowed:
            raise ValueError(variant)
        self.variant = variant
        with isolated_seed(780000 + (seed % 1000000) * 17 + fold):
            # Shared pathology projection is the only high-dimensional map.
            self.path_projection = nn.Sequential(
                nn.LayerNorm(PATH_INPUT_DIM),
                nn.Linear(PATH_INPUT_DIM, PATH_PROJECTION_DIM),
                nn.GELU(),
            )
            self.concept_queries = nn.Parameter(torch.randn(CONCEPTS, PATH_PROJECTION_DIM) * 0.02)
            self.path_head = nn.Linear(PATH_PROJECTION_DIM, 1)
            self.omic_head = nn.Linear(2, 1)
            self.message_network = nn.Sequential(
                nn.Linear(4, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            # The lambda gates are deliberately near zero, while retaining a
            # nonzero gradient path for the residual heads at initialization.
            self.lambda_path = nn.Parameter(torch.full((CONCEPTS,), 1e-2))
            self.lambda_omics = nn.Parameter(torch.full((CONCEPTS,), 1e-2))
            self.lambda_message = nn.Parameter(torch.full((CONCEPTS,), 1e-2))

        # Small nonzero heads keep the residual near the frozen anchor while
        # allowing the concept-matched message branch to receive gradients
        # from the first Cox update.  The protocol freezes the anchor through
        # detach() and constrains the lambda gates near zero.
        nn.init.normal_(self.path_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.path_head.bias)
        nn.init.normal_(self.omic_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.omic_head.bias)
        nn.init.normal_(self.message_network[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.message_network[-1].bias)

    def pathology_concepts(self, pathology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.path_projection(pathology.float())
        logits = torch.einsum("bkd,cd->bkc", hidden, self.concept_queries) / (PATH_PROJECTION_DIM ** 0.5)
        attention = torch.softmax(logits, dim=1)
        concept_hidden = torch.einsum("bkc,bkd->bcd", attention, hidden)
        scores = self.path_head(concept_hidden).squeeze(-1)
        return scores, attention

    def omics_concepts(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.omic_head(evidence.float()).squeeze(-1)

    def forward(self, pathology: torch.Tensor, omics_evidence: torch.Tensor, anchor_offset: torch.Tensor):
        path_scores, attention = self.pathology_concepts(pathology)
        omic_scores = self.omics_concepts(omics_evidence)
        features = torch.stack(
            [path_scores, omic_scores, path_scores * omic_scores, (path_scores - omic_scores).abs()],
            dim=-1,
        )
        messages = self.message_network(features).squeeze(-1)
        if self.variant == "K-O":
            delta_parts = self.lambda_omics * omic_scores
        elif self.variant == "K-P":
            delta_parts = self.lambda_path * path_scores
        elif self.variant == "K-ADD":
            delta_parts = self.lambda_path * path_scores + self.lambda_omics * omic_scores
        else:
            delta_parts = self.lambda_message * messages
        delta = delta_parts.sum(dim=1)
        risk = anchor_offset.detach() + delta
        return risk, {
            "path_scores": path_scores,
            "omic_scores": omic_scores,
            "messages": messages,
            "delta_parts": delta_parts,
            "attention": attention,
        }


def build_model(variant: str, seed: int, fold: int) -> ConceptFusionResidualModel:
    return ConceptFusionResidualModel(variant, seed, fold)

