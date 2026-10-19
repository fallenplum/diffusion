"""Synthetic official CUDA qualification only; no patient inputs or optimizer."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# A fresh process must perform normal package imports, before any namespace shim.
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda

sys.path.insert(0, '/root/phase16_routing_repo')
from src.phase17_ssd.losses import full_risk_efron

REPORT = Path('/root/autodl-tmp/phase28_reread_private/DEPENDENCY_QUALIFICATION.json')
torch.manual_seed(2801)
torch.cuda.manual_seed_all(2801)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weight_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_error(left, right):
    errors = []
    for (name, p), (other, q) in zip(left.named_parameters(), right.named_parameters()):
        assert name == other and ((p.grad is None) == (q.grad is None)), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-4, msg=name)
            errors.append(float((p.grad-q.grad).abs().max()))
    return max(errors, default=0.)


def core_reference(module, value):
    length = value.shape[1]
    xz = F.linear(value, module.in_proj.weight, module.in_proj.bias).transpose(1, 2)
    x, z = xz.chunk(2, 1)
    x = F.silu(F.conv1d(x, module.conv1d.weight, module.conv1d.bias,
                       padding=module.d_conv-1, groups=module.d_inner)[..., :length])
    mixed = F.linear(x.transpose(1, 2), module.x_proj.weight)
    dt, b, c = torch.split(mixed, [module.dt_rank, module.d_state, module.d_state], -1)
    dt = F.linear(dt, module.dt_proj.weight).transpose(1, 2)
    result = selective_scan_ref(x, dt, -torch.exp(module.A_log.float()), b.transpose(1, 2),
        c.transpose(1, 2), module.D.float(), z=z, delta_bias=module.dt_proj.bias.float(),
        delta_softplus=True, return_last_state=False)
    return F.linear(result.transpose(1, 2), module.out_proj.weight, module.out_proj.bias)


def main():
    result = {'status': 'RUNNING', 'source_kind': 'official_release_wheels',
              'normal_fresh_process_import': True, 'new_optimizer_steps': 0,
              'patient_data_or_endpoint_reads': 0, 'custom_S6_substitute': False,
              'last_state_used_for_objective': False, 'core': {
                  'd_model': 128, 'd_inner': 256, 'd_state': 16, 'd_conv': 4,
                  'expand': 2, 'use_fast_path': False}, 'scan_checks': []}
    env = Path(sys.prefix)
    assert env == Path('/root/autodl-tmp/phase28_reread_private/env')
    result['environment'] = {'prefix': str(env), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'GPU': torch.cuda.get_device_name(),
        'versions': {n: importlib.metadata.version(n) for n in
            ['mamba-ssm', 'causal-conv1d', 'transformers', 'tokenizers', 'huggingface-hub']}}
    result['official_modules'] = {n.__name__: {'path': n.__file__, 'sha256': sha(n.__file__)}
        for n in [mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda]}
    for length in (1, 7, 16):
        dims = (2, 256, length)
        a = [torch.randn(dims, device='cuda')*.1,
             torch.randn(dims, device='cuda')*.1-2,
             -torch.arange(1, 17, device='cuda').float()[None].repeat(256, 1),
             torch.randn(2, 16, length, device='cuda')*.1,
             torch.randn(2, 16, length, device='cuda')*.1,
             torch.randn(256, device='cuda')*.1,
             torch.randn(dims, device='cuda')*.1,
             torch.randn(256, device='cuda')*.1-2]
        left = [v.clone().requires_grad_() for v in a]
        right = [v.clone().requires_grad_() for v in a]
        def calculate(fn, args):
            u, dt, aa, b, c, d, z, bias = args
            return fn(u, dt, aa, b, c, d, z=z, delta_bias=bias,
                      delta_softplus=True, return_last_state=False)
        y, expected = calculate(selective_scan_fn, left), calculate(selective_scan_ref, right)
        probe = torch.randn_like(y)*.1
        gy = torch.autograd.grad((y*probe).sum(), left)
        gr = torch.autograd.grad((expected*probe).sum(), right)
        torch.testing.assert_close(y, expected, rtol=2e-3, atol=3e-5)
        for actual, reference in zip(gy, gr):
            torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-4)
        result['scan_checks'].append({'length': length,
            'forward_max_absolute_error': float((y-expected).abs().max()),
            'gradient_max_absolute_error': max(float((x-y).abs().max()) for x, y in zip(gy, gr)),
            'reference': 'official selective_scan_ref; exp(delta*A), delta*B*u'})
    model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2,
                  use_fast_path=False).cuda()
    reference = copy.deepcopy(model)
    before = weight_hash(model)
    x = torch.randn(2, 11, 128, device='cuda')*.1
    actual, expected = model(x), core_reference(reference, x)
    probe = torch.randn_like(actual)*.1
    (actual*probe).sum().backward(); (expected*probe).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-5)
    result['core_sequence_check'] = {
        'output_shape': list(actual.shape),
        'forward_max_absolute_error': float((actual-expected).abs().max()),
        'gradient_max_absolute_error': grad_error(model, reference),
        'parameters': sum(p.numel() for p in model.parameters()),
        'weights_unchanged': before == weight_hash(model), 'sequence_output_only': True}

    # Exercise true two-pass complete-Efron GradCache with dropout and RNG replay.
    class SyntheticRisk(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Mamba(128, d_state=16, d_conv=4, expand=2, use_fast_path=False)
            self.drop = nn.Dropout(.2)
            self.head = nn.Linear(128, 1)
        def forward(self, value):
            return self.head(self.drop(self.core(value)).mean(1)).flatten().double()
    ordinary = SyntheticRisk().cuda().train()
    cached = copy.deepcopy(ordinary)
    initial_hash = weight_hash(cached)
    values = torch.randn(6, 8, 128, device='cuda')*.1
    base = torch.linspace(-.2, .2, 6, device='cuda', dtype=torch.float64)
    times = np.array([1., 1., 2., 3., 3., 4.])
    statuses = np.array([1., 0., 1., 1., 0., 1.])
    torch.manual_seed(2802); torch.cuda.manual_seed_all(2802)
    plain = torch.cat([ordinary(values[j:j+2]) for j in range(0, 6, 2)])+base
    full_risk_efron(plain, times, statuses).backward()
    torch.manual_seed(2802); torch.cuda.manual_seed_all(2802)
    states, stored = [], []
    with torch.no_grad():
        for j in range(0, 6, 2):
            states.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
            stored.append(cached(values[j:j+2])+base[j:j+2])
    end_state = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    all_risk = torch.cat(stored)
    leaf = all_risk.detach().requires_grad_()
    upstream = torch.autograd.grad(full_risk_efron(leaf, times, statuses), leaf)[0]
    replay_error = 0.
    for batch, j in enumerate(range(0, 6, 2)):
        torch.set_rng_state(states[batch][0]); torch.cuda.set_rng_state(states[batch][1])
        risk = cached(values[j:j+2])+base[j:j+2]
        replay_error = max(replay_error, float((risk.detach()-stored[batch]).abs().max()))
        (risk*upstream[j:j+2]).sum().backward()
    torch.set_rng_state(end_state[0]); torch.cuda.set_rng_state(end_state[1])
    torch.testing.assert_close(plain.detach(), all_risk, rtol=0., atol=1e-10)
    result['complete_Efron_gradcache_check'] = {
        'synthetic_examples': 6, 'microbatches': 3, 'dropout': .2,
        'cached_vs_plain_risk_max_error': float((plain.detach()-all_risk).abs().max()),
        'rng_replayed_risk_max_error': replay_error,
        'parameter_gradient_max_error': grad_error(ordinary, cached),
        'weights_unchanged': initial_hash == weight_hash(cached),
        'RNG_restored_per_microbatch': True, 'all_synthetic_risk_set_used': True}
    result.update(status='PASS_OFFICIAL_CUDA_SEQUENCE_AND_FULL_EFRON_GRADCACHE',
                  qualification_script_sha256=sha(__file__))
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        error = {'status': 'FAIL_SYNTHETIC_QUALIFICATION', 'type': type(exc).__name__,
                 'message': str(exc)[:1000], 'patient_data_or_endpoint_reads': 0,
                 'new_optimizer_steps': 0, 'qualification_script_sha256': sha(__file__)}
        REPORT.write_text(json.dumps(error, indent=2)+'\n')
        print(json.dumps(error))
        raise
