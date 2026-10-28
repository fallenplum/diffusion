import math

import pytest
import torch

from scripts.bgb_en_r7_interface_audit import masked_scale


def test_rms_excludes_padding_and_preserves_zero_vectors():
    values = torch.tensor([[3., 4.], [0., 0.], [float("nan"), float("inf")]])
    result = masked_scale(values, torch.tensor([True, True, False]))
    assert result["coordinate_RMS"] == 2.5
    assert result["valid_vectors"] == 2
    assert result["zero_vectors"] == 1
    assert result["median_L2"] == 2.5


def test_empty_mask_does_not_make_fake_zero_measurement():
    result = masked_scale(torch.full((2, 4), float("nan")), torch.zeros(2, dtype=torch.bool))
    assert result["valid_vectors"] == 0
    assert result["coordinate_RMS"] is None


def test_invalid_measured_value_or_mask_is_rejected():
    with pytest.raises(ValueError, match="NONFINITE"):
        masked_scale(torch.tensor([[float("inf"), 0.]]), torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match="INVALID_MASK"):
        masked_scale(torch.zeros(2, 3), torch.ones(3, dtype=torch.bool))
