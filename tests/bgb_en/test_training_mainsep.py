"""Check the actual R3 trainer F2 risk stays additive after its bounds."""
from types import SimpleNamespace

import numpy as np
import torch

from src.bgb_en.training import MechanismModel


def test_f2_actual_training_instance_has_zero_cross_modal_difference():
    torch.manual_seed(9033991)
    rng = np.random.default_rng(9033991)
    # Four recombinations (P_A,O_A), (P_A,O_B), (P_B,O_A), (P_B,O_B).
    pa, pb = torch.randn(1, 6, 768), torch.randn(1, 6, 768)
    x = torch.stack((pa, pa, pb, pb))
    mask = torch.ones(4, 1, 6, dtype=torch.bool)
    xy = torch.stack((torch.arange(6), torch.zeros(6, dtype=torch.long)), -1)[None, None].expand(4, 1, 6, 2)
    bank = SimpleNamespace(x=x, mask=mask, xy=xy, region_ids=torch.zeros_like(mask, dtype=torch.long),
                           step=1., input_dim=768)
    ra, rb = rng.normal(size=(2, 12)).astype(np.float32)
    rna = np.stack((ra, rb, ra, rb))
    features = dict(rna=rna, rna_mask=np.ones_like(rna, bool),
                    cna_gain=np.zeros_like(rna), cna_loss=np.zeros_like(rna),
                    cna_mask=np.ones_like(rna, bool), C=np.zeros((4, 3), np.float32))
    prepared = dict(features=features, base=np.zeros(4), CS=np.zeros(4), RNA=np.zeros(4), beta=1.,
                    train=np.array([0, 1]), valid=np.array([2, 3]))
    membership = torch.zeros(3, 12, dtype=torch.bool)
    membership[0, :4] = True; membership[1, 4:8] = True; membership[2, 8:] = True
    net = MechanismModel("R3", "F2", prepared, membership, bank, "cpu").eval()
    output = net(torch.arange(4))
    delta = output["delta"]
    difference = delta[0]+delta[3]-delta[1]-delta[2]
    torch.testing.assert_close(difference, torch.zeros_like(difference), atol=1e-8, rtol=0)
    torch.testing.assert_close(output["risk"], delta, atol=0, rtol=0)
    # Context-only interventions cannot change the two pooled unimodal paths.
    net.query_map[:4] = np.array([1, 0, 3, 2])
    swapped = net(torch.arange(4), control="wrong_query")
    off = net(torch.arange(4), control="interaction_off")
    torch.testing.assert_close(delta, swapped["delta"], atol=0, rtol=0)
    torch.testing.assert_close(delta, off["delta"], atol=0, rtol=0)
