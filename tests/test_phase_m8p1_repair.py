import numpy as np
import torch

from src.phase_m8p1.clinical import validate_lineage
from src.phase_m8p1.models import build_model


def test_lineage_rejects_outer_test_overlap():
    try:
        validate_lineage([0, 1, 2, 3], [4], [0, 1], [2, 3], [3])
    except RuntimeError:
        return
    raise AssertionError("outer test overlap was accepted")


def test_residual_uses_frozen_offset_and_reconstructs():
    model = build_model("SPO", 2, 7, 1)
    assert float(model.path_head[-1].weight.abs().max()) == 0.0
    assert float(model.omic_head[-1].weight.abs().max()) == 0.0
    offset = torch.tensor([1.2, -0.4])
    path = torch.randn(2, 3, 1536)
    rna = torch.randn(2, 512)
    cna = torch.randn(2, 256)
    clinical = torch.randn(2, 2)
    risk, delta = model(path, rna, cna, clinical, offset)
    assert torch.allclose(risk, offset + delta)
    assert torch.allclose(offset, offset.detach())

