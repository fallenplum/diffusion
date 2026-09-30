from __future__ import annotations

import torch
from torch import nn

from .mgct import GatedAttentionPool
from .nn_blocks import MultimodalBackbone, RiskHead


class SixSignatureEncoder(nn.Module):
    """Six independent genomic signature networks as in official MCAT.

    Each official signature is one omic vector formed by concatenating its
    fold-local RNA and CNA symbol intersections.  MCAT therefore produces
    exactly six omic tokens, one per signature (the prior implementation
    incorrectly produced twelve separate RNA/CNA tokens).
    """

    def __init__(self, widths: tuple[tuple[int, int], ...], dim: int = 96) -> None:
        super().__init__()
        if len(widths) != 6:
            raise ValueError(f"MCAT requires six official signature groups, got {len(widths)}")
        if any(int(rna) <= 0 or int(cna) <= 0 for rna, cna in widths):
            raise ValueError(f"MCAT signature groups must all have nonzero RNA/CNA overlap: {widths}")
        self.widths = tuple((int(rna), int(cna)) for rna, cna in widths)
        self.sig_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(rna + cna, dim),
                nn.ELU(),
                nn.AlphaDropout(p=0.25),
                nn.Linear(dim, dim),
                nn.ELU(),
                nn.AlphaDropout(p=0.25),
            )
            for rna, cna in self.widths
        ])

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        # groups: [batch, six official signatures, max_group_width, 2], with
        # channel 0 RNA and channel 1 CNA.  Padding is never consumed because
        # each module slices its audited fold-local width.
        values = []
        for index, (rna_width, cna_width) in enumerate(self.widths):
            rna = groups[:, index, :rna_width, 0].float()
            cna = groups[:, index, :cna_width, 1].float()
            values.append(self.sig_networks[index](torch.cat([rna, cna], dim=-1)))
        return torch.stack(values, dim=1)


class MCATModel(nn.Module):
    """MCAT co-attention model kept separate from MOTCat."""

    def __init__(self, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...] | None = None, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        if signature_widths is None:
            signature_widths = tuple((1, 1) for _ in range(6))
        self.sig_networks = SixSignatureEncoder(signature_widths, dim)
        self.coattn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        path_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        omic_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        self.path_transformer = nn.TransformerEncoder(path_layer, num_layers=2)
        self.omic_transformer = nn.TransformerEncoder(omic_layer, num_layers=2)
        self.path_attention_head = GatedAttentionPool(dim)
        self.omic_attention_head = GatedAttentionPool(dim)
        self.path_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.omic_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.mm = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.clinical_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = RiskHead(dim)

    def forward(
        self,
        pathology: torch.Tensor,
        rna: torch.Tensor,
        cna: torch.Tensor,
        clinical: torch.Tensor,
        signature_groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        h_path = encoded.pathology
        if signature_groups is None:
            raise ValueError("MCAT requires official six-signature gene groups")
        h_omic = self.sig_networks(signature_groups)
        # MCAT: omic queries attend to the histology bag.
        h_path_coattn, _ = self.coattn(h_omic, h_path, h_path)
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path, _ = self.path_attention_head(h_path_trans)
        h_path = self.path_rho(h_path)
        h_omic_trans = self.omic_transformer(h_omic)
        h_omic, _ = self.omic_attention_head(h_omic_trans)
        h_omic = self.omic_rho(h_omic)
        fused = self.mm(torch.cat([h_path, h_omic], dim=-1))
        clinical_context = self.clinical_gate(torch.cat([fused, encoded.clinical[:, 0]], dim=-1))
        return self.head(clinical_context), pathology.new_zeros(())

