"""Exact-ZOH local C_t--h_t readout, with bounded activation storage.

The input ``dt`` already includes the parent's write gate and patch mask.  This
module does not modify the transition, input, state matrix, gate, or skip path.
``m`` is used only in the requested readout weighting.  A gate therefore
appears in the transition exactly as in the parent and in the readout exactly
as specified by Phase18A; it is not multiplied into ``dt`` a second time.

Within each chunk, a stable affine prefix scan constructs its states.  Whole
chunks are recomputed by autograd checkpointing during backward.  Chunk-boundary
states stay connected, so gradients include the complete sequence (not TBPTT).
No [batch, full_length, channels, state_size] tensor is retained for training.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from src.phase17_ssd.ssm import reference_scan, zoh


def _local_chunk(u, dt, B, C, A, m, h0):
    """Return the exact chunk's readout numerator and final state.

    Parallel doubling composes ``(decay, input)`` affine transitions.  Decay
    exponents are sums of nonnegative dt times nonpositive A, avoiding unstable
    inverse cumulative products when the recurrence strongly forgets.
    """
    _, zoh_B = zoh(dt, A)
    h = zoh_B * B[:, :, None, :] * u[:, :, :, None]
    prefix_dt = dt
    step = 1
    while step < u.shape[1]:
        shifted_h = torch.cat([torch.zeros_like(h[:, :step]), h[:, :-step]], 1)
        h = h + torch.exp(prefix_dt[..., None] * A) * shifted_h
        shifted_dt = torch.cat(
            [torch.zeros_like(prefix_dt[:, :step]), prefix_dt[:, :-step]], 1
        )
        prefix_dt = prefix_dt + shifted_dt
        step *= 2
    h = h + torch.exp(prefix_dt[..., None] * A) * h0[:, None]
    numerator = ((h * C[:, :, None, :]).sum(-1) * m).sum(1)
    # Materialize the boundary state: a view would keep the entire chunk's
    # backing storage alive across all later chunks despite checkpointing.
    return numerator, h[:, -1].clone()


def local_readout_scan(
    u, dt, B, C, A, m, h0=None, chunk_size=32,
    checkpoint_chunks=True, return_final=False,
):
    """Compute ``sum_t m_t C_t h_t / clamp(sum_t m_t, 1e-8)``.

    Shapes: u/dt [B,N,D], B/C [B,N,S], A [D,S], m [B,N,1] (or
    [B,N]), optional h0 [B,D,S].  ``return_final`` additionally returns h_T
    for the same states used by this readout.  No learned parameters or output
    normalization are introduced.  Nonnegative dt and finite preprocessed
    inputs follow the unchanged parent model's contract.
    """
    if u.ndim != 3 or dt.shape != u.shape:
        raise ValueError("u and dt must have the same [batch,length,channels] shape")
    batch, length, channels = u.shape
    if length < 1 or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("length and chunk_size must be positive integers")
    if A.ndim != 2 or A.shape[0] != channels:
        raise ValueError("A must have [channels,state_size] shape")
    states = A.shape[1]
    if B.shape != (batch, length, states) or C.shape != B.shape:
        raise ValueError("B and C must have [batch,length,state_size] shape")
    if m.shape == (batch, length):
        m = m[:, :, None]
    if m.shape != (batch, length, 1):
        raise ValueError("m must have [batch,length,1] or [batch,length] shape")
    if h0 is None:
        h = u.new_zeros((batch, channels, states))
    elif h0.shape == (batch, channels, states):
        h = h0
    else:
        raise ValueError("h0 must have [batch,channels,state_size] shape")
    numerator = u.new_zeros((batch, channels))
    for start in range(0, length, int(chunk_size)):
        end = min(start + int(chunk_size), length)
        args = (u[:, start:end], dt[:, start:end], B[:, start:end],
                C[:, start:end], A, m[:, start:end], h)
        if checkpoint_chunks and torch.is_grad_enabled() and any(x.requires_grad for x in args):
            local_sum, h = checkpoint(_local_chunk, *args, use_reentrant=False)
        else:
            local_sum, h = _local_chunk(*args)
        numerator = numerator + local_sum
    y = numerator / m.sum(1).clamp_min(1e-8)
    return (y, h) if return_final else y


def explicit_local_readout(u, dt, B, C, A, m, h0=None):
    """Small-input oracle; deliberately stores all states, never used in fits."""
    hs = reference_scan(u, dt, B, A, h0=h0, states=True)
    if m.ndim == 2:
        m = m[:, :, None]
    y = ((hs * C[:, :, None, :]).sum(-1) * m).sum(1) / m.sum(1).clamp_min(1e-8)
    return y, hs[:, -1]


def operator_parity(device="cpu", dtype=torch.float64):
    """Execute output and all-input-gradient parity against explicit recurrence.

    The scalar probe includes y and h_T so both the readout and the returned
    boundary state receive gradients.  Model-parameter parity is separately
    exercised in tests/test_phase18a_readout.py.
    """
    dev = torch.device(device)
    generator = torch.Generator(device=dev).manual_seed(181803)
    rows = []
    for name, length, chunk, tiny, zero, padding in [
        ("N1", 1, 32, False, False, False),
        ("small", 7, 3, False, False, False),
        ("padding", 19, 8, False, False, True),
        ("tiny_dt", 11, 4, True, False, False),
        ("gate_zero", 9, 4, False, True, False),
        ("multiple_chunks", 65, 16, False, False, True),
    ]:
        def randn(*shape):
            return torch.randn(*shape, dtype=dtype, device=dev, generator=generator)
        u = randn(2, length, 4)
        gate = torch.sigmoid(randn(2, length, 1))
        valid = torch.ones_like(gate)
        if padding:
            valid[0, length // 2:] = 0
            valid[1, -2:] = 0
        if zero:
            gate.zero_()
        m = gate * valid
        dt = torch.nn.functional.softplus(randn(2, length, 4)) * m
        if tiny:
            dt = dt * 1e-12
        original = [u, dt, randn(2, length, 3), randn(2, length, 3),
                    -randn(4, 3).exp(), m, randn(2, 4, 3)]
        args = [x.detach().clone().requires_grad_() for x in original]
        oracle_args = [x.detach().clone().requires_grad_() for x in original]
        actual = local_readout_scan(*args[:6], h0=args[6], chunk_size=chunk, return_final=True)
        expected = explicit_local_readout(*oracle_args[:6], h0=oracle_args[6])
        probe_y, probe_h = randn(*actual[0].shape), randn(*actual[1].shape)
        actual_grad = torch.autograd.grad((actual[0] * probe_y).sum() + (actual[1] * probe_h).sum(), args)
        oracle_grad = torch.autograd.grad((expected[0] * probe_y).sum() + (expected[1] * probe_h).sum(), oracle_args)
        output_error = max(float((x - y).abs().max().detach().cpu()) for x, y in zip(actual, expected))
        gradient_error = max(float((x - y).abs().max().detach().cpu()) for x, y in zip(actual_grad, oracle_grad))
        tolerance = 1e-9 if dtype == torch.float64 else 2e-4
        passed = all(torch.allclose(x, y, atol=tolerance, rtol=tolerance) for x, y in zip(actual, expected))
        passed = passed and all(torch.allclose(x, y, atol=tolerance, rtol=tolerance) for x, y in zip(actual_grad, oracle_grad))
        if not passed:
            raise AssertionError(f"{device}/{name} output={output_error} gradient={gradient_error}")
        rows.append({"case": name, "length": length, "chunk_size": chunk,
                     "nonzero_h0": True, "output_max_abs": output_error,
                     "all_input_gradient_max_abs": gradient_error, "passed": True})
    return {"device": str(dev), "dtype": str(dtype), "oracle": "reference_scan(states=True)",
            "input_gradients": ["u", "dt", "B", "C", "A", "m", "h0"], "cases": rows,
            "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    result = {"float64": operator_parity(args.device, torch.float64),
              "float32": operator_parity(args.device, torch.float32)}
    Path(args.audit_output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
