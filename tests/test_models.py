import numpy as np
import torch

from phase1_neural import ABMIL, ContextEncoder, EncoderConfig


def small_config(**overrides):
    values = {
        "input_dim": 1536,
        "dim": 16,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "use_coords": True,
        "max_tokens": 8,
    }
    values.update(overrides)
    return EncoderConfig(**values)


def test_context_encoder_forward_mask_and_backward():
    torch.manual_seed(7)
    features = torch.randn(2, 8, 1536, requires_grad=True)
    coords = torch.rand(2, 8, 2)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, [1, 5]] = True
    model = ContextEncoder(small_config())
    pooled, tokens = model(features, coords, mask)
    assert pooled.shape == (2, 16)
    assert tokens.shape == (2, 8, 16)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(tokens).all()
    model_no_coords = ContextEncoder(small_config(use_coords=False))
    pooled_no_coords, tokens_no_coords = model_no_coords(features.detach(), coords, None)
    assert pooled_no_coords.shape == pooled.shape
    assert tokens_no_coords.shape == tokens.shape
    pooled.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_abmil_forward_and_backward():
    torch.manual_seed(11)
    features = torch.randn(2, 8, 1536, requires_grad=True)
    model = ABMIL(small_config(use_coords=False))
    pooled, weights = model(features)
    assert pooled.shape == (2, 16)
    assert weights.shape == (2, 8)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    assert np.allclose(weights.detach().sum(dim=1).numpy(), 1.0)
    pooled.mean().backward()
    assert features.grad is not None
