"""Allocation-only TITAN ALiBi optimization, checked against pinned official code.

The Euclidean grid distances, head slopes, float32 operations, prefix-token bias
and attention operations are unchanged. No patch subsampling or tiled attention.
"""
from __future__ import annotations

import inspect
import math
import textwrap
import types

import numpy as np
import torch


def optimize_alibi(model):
    vision = model.vision_encoder
    official = vision.get_alibi

    def slopes_for(n):
        if math.log2(n).is_integer():
            p = 2 ** (-2 ** -(math.log2(n) - 3))
            return [p * p ** i for i in range(n)]
        nearest = 2 ** math.floor(math.log2(n))
        return slopes_for(nearest) + slopes_for(2*nearest)[0::2][:n-nearest]

    def chunked(self, w, h, bg_mask=None):
        xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        if bg_mask is not None:
            selected = bg_mask.cpu().squeeze(0).numpy()
            xx, yy = xx[selected], yy[selected]
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        n = len(points)
        target_device = next(self.parameters()).device
        result = torch.zeros(1, self.num_heads, n+1, n+1, dtype=torch.float32, device=target_device)
        slopes = torch.tensor(slopes_for(self.num_heads), dtype=torch.float32).view(self.num_heads, 1, 1)
        for start in range(0, n, 128):
            stop = min(start+128, n)
            delta = points[start:stop, None, :] - points[None, :, :]
            distances = np.sqrt(np.sum(delta**2, axis=-1))
            # Calculate on CPU in the same float32 operation order as official.
            row_bias = torch.tensor(distances, dtype=torch.float32).unsqueeze(0) * slopes * -1
            result[0, :, start+1:stop+1, 1:] = row_bias.to(target_device)
        return result

    maximum = 0.0
    # Both a regular grid and irregular foreground mask exercise the exact
    # row order used by TITAN. Validation tensors remain in memory only.
    masks = [None, (torch.arange(256).reshape(1, 16, 16) % 3 != 0)]
    for mask in masks:
        reference = official(16, 16, mask)
        candidate = chunked(vision, 16, 16, mask).cpu()
        maximum = max(maximum, float((reference-candidate).abs().max()))
        if not torch.equal(reference, candidate):
            raise RuntimeError("chunked ALiBi differs from official float32 bias")
    original_forward = vision.forward_features.__func__
    source = textwrap.dedent(inspect.getsource(original_forward))
    needle = "attn_bias = attn_bias.repeat(x.shape[0], 1, 1, 1).type(x.dtype).to(x.device)"
    replacement = "attn_bias = (attn_bias if x.shape[0] == 1 else attn_bias.repeat(x.shape[0], 1, 1, 1)).type(x.dtype).to(x.device)"
    if source.count(needle) != 1:
        raise RuntimeError("pinned official attention allocation statement changed")
    namespace = dict(original_forward.__globals__)
    exec(compile(source.replace(needle, replacement), "<titan_allocation_only>", "exec"), namespace)
    vision.get_alibi = types.MethodType(chunked, vision)
    vision.forward_features = types.MethodType(namespace["forward_features"], vision)
    return {"official_float32_bias_max_abs_error": maximum, "bias_reference_tensors_bitwise_equal": True,
            "changes": ["Construct Euclidean head bias in 128-row chunks", "Allocate final float32 bias directly on GPU", "Avoid identity batch repeat for batch size one"],
            "unchanged": ["All patches and coordinates", "Float32 distance and slope multiplication", "All attention tokens and model parameters", "Official semantic projection"]}
