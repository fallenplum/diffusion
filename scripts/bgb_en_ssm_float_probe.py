"""Synthetic-only SSM masking arithmetic check; reads no experiment arrays."""
import json
import copy
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bgb_en.omics import DiagonalSelectiveSSM, MultiOrderSequence

torch.manual_seed(6)
block = DiagonalSelectiveSSM().double()
x = torch.randn(1, 4, 32, dtype=torch.float64)
valid = torch.tensor([[1, 0, 1, 0]]).bool()
full = block(x, valid)[:, [0, 2]]
compact = block(x[:, [0, 2]], torch.ones(1, 2).bool())
difference = (full - compact).abs()
print(json.dumps({'synthetic_only': True, 'torch': torch.__version__,
                  'dtype': str(full.dtype), 'maximum_absolute_error': difference.max().item(),
                  'maximum_output_magnitude': full.abs().max().item(),
                  'machine_epsilon': torch.finfo(full.dtype).eps,
                  'allclose_atol_1e_14_rtol_1e_14': torch.allclose(full, compact, atol=1e-14, rtol=1e-14)}, indent=2))

if '--official-mamba' in sys.argv:
    if not torch.cuda.is_available():
        raise RuntimeError('BLOCKED_OFFICIAL_MAMBA_GPU')
    torch.manual_seed(884)
    batched = MultiOrderSequence(50, backend='mamba', seed=270101).cuda()
    legacy = copy.deepcopy(batched)
    values = torch.randn(8, 50, 32, device='cuda', requires_grad=True)
    old_values = values.detach().clone().requires_grad_(True)
    valid = torch.zeros(8, 50, dtype=torch.bool, device='cuda')
    for row, n in enumerate([50, 50, 49, 49, 35, 35, 0, 50]):
        valid[row, torch.randperm(50, device='cuda')[:n]] = True
    before = [p.detach().clone() for p in batched.parameters()]
    torch.cuda.synchronize(); started = time.perf_counter()
    actual = batched(values, valid)
    actual.square().mean().backward()
    torch.cuda.synchronize(); batched_seconds = time.perf_counter() - started
    torch.cuda.synchronize(); started = time.perf_counter()
    normalized = legacy.norm(old_values)
    messages = []
    for order in legacy.orders:
        x, m = normalized[:, order], valid[:, order]
        out = torch.zeros_like(x)
        for row in range(len(x)):
            if m[row].any():
                out[row, m[row]] = legacy.scan(x[row, m[row]][None])[0]
        messages.append(out[:, torch.argsort(order)])
    expected = (old_values + torch.stack(messages).mean(0)) * valid[..., None]
    expected.square().mean().backward()
    torch.cuda.synchronize(); legacy_seconds = time.perf_counter() - started
    output_error = (actual - expected).abs().max().item()
    input_gradient_error = (values.grad - old_values.grad).abs().max().item()
    parameter_gradient_error = max((a.grad - b.grad).abs().max().item()
                                   for a, b in zip(batched.parameters(), legacy.parameters())
                                   if a.grad is not None or b.grad is not None)
    unchanged = all(torch.equal(p, q) for p, q in zip(before, batched.parameters()))
    report = {'synthetic_only': True, 'official_mamba': True, 'optimizer_steps': 0,
              'maximum_output_error': output_error, 'maximum_input_gradient_error': input_gradient_error,
              'maximum_parameter_gradient_error': parameter_gradient_error,
              'padded_outputs_exact_zero': bool((actual[~valid] == 0).all()), 'weights_unchanged': unchanged,
              'batched_scan_calls': 3 * 4, 'legacy_scan_calls': 7 * 4,
              'batched_forward_backward_seconds': batched_seconds, 'legacy_forward_backward_seconds': legacy_seconds}
    print(json.dumps(report, indent=2))
    if max(output_error, input_gradient_error, parameter_gradient_error) > 1e-6 or not unchanged:
        raise RuntimeError('FAIL_OFFICIAL_MAMBA_EQUAL_LENGTH_COMPACTION')
