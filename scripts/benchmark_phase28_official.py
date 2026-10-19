"""Two synthetic shape timings; no cohort, endpoints, optimizer, or weights output."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import torch
sys.path.insert(0, '/root/phase16_routing_repo')
from src.phase28_reread.model import BidirectionalReread
from src.phase28_reread.spatial import OfficialSpatialEncoder, build_plan

ROOT = Path('/root/phase16_routing_repo')
REPORT = Path('/root/autodl-tmp/phase28_reread_private/SYNTHETIC_PERFORMANCE.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def case(name):
    torch.manual_seed(2803); torch.cuda.manual_seed_all(2803)
    rng = np.random.default_rng(2803)
    genes = 26708
    membership = np.zeros((108, genes), bool)
    for j in range(50):
        use = rng.choice(genes, 150, replace=False)
        membership[2*j:2*j+2, use] = True
    membership[100:] = True
    if name == 'dense_4096':
        points = [(x, y) for y in range(64) for x in range(64)]
    else:
        points = [(tx*16+x, ty*16+y) for ty in range(8) for tx in range(8)
                  for y in range(8) for x in range(8)]
    plan = build_plan(np.asarray(points, np.int64)*512)
    model = BidirectionalReread(genes, membership, OfficialSpatialEncoder(), arm='C11').cuda().train()
    # Synthetic nonzero risk heads exercise gradients upstream, without fitting.
    with torch.no_grad():
        model.main_path.coefficient.fill_(.01); model.main_gene.axis.fill_(.01)
    values = torch.randn(genes, 3, device='cuda')*.1
    mask = torch.rand(genes, 3, device='cuda') < .95
    ids = torch.arange(genes, device='cuda')
    slides = [{'features': torch.randn(4096, 1536, device='cuda')*.1, 'plan': plan}]
    before = weight_hash(model)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    output = model(values, mask, ids, slides, offset=.1)
    torch.cuda.synchronize(); forward = time.perf_counter()-begun
    begun = time.perf_counter()
    output['risk'].backward()
    torch.cuda.synchronize(); backward = time.perf_counter()-begun
    allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    grad_count = sum(p.numel() for p in model.parameters() if p.grad is not None)
    del output
    begun = time.perf_counter()
    with torch.no_grad():
        prediction = model(values, mask, ids, slides, offset=.1)['risk']
    torch.cuda.synchronize(); no_grad = time.perf_counter()-begun
    assert bool(torch.isfinite(prediction)) and bool(finite)
    result = {'name': name, 'patches': 4096, 'genes': genes, 'molecular_slots': 108,
        'regions': plan['audit']['selected_regions'], 'path_slots': 2*plan['audit']['selected_regions'],
        'rounds': 2, 'molecular_measurement_fraction': .95,
        'forward_seconds': forward, 'backward_seconds': backward,
        'no_grad_forward_seconds': no_grad,
        'full_gradcache_example_seconds': no_grad+forward+backward,
        'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
        'parameters': sum(p.numel() for p in model.parameters()),
        'parameters_with_gradient': grad_count, 'gradients_finite': bool(finite),
        'weights_unchanged': before == weight_hash(model),
        'new_optimizer_steps': 0,
        'estimate_80epochs_327single_WSI_examples_30models_seconds':
            (no_grad+forward+backward)*80*327*30,
        'estimate_scope': 'serial synthetic compute only; excludes H5 I/O, unequal WSI/patch counts, validation, controls, source replay, optimizer overhead, concurrency contention'}
    del prediction, model, values, mask, ids, slides
    torch.cuda.empty_cache()
    return result


def main():
    assert sys.prefix == '/root/autodl-tmp/phase28_reread_private/env'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    result = {'status': 'PASS_SYNTHETIC_ONLY', 'new_optimizer_steps': 0,
        'patient_data_or_endpoint_reads': 0, 'weights_or_arrays_saved': False,
        'no_architecture_selection_from_timing': True,
        'normal_official_Mamba_import': True, 'test_only_scan_passed': False,
        'GPU': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'env_prefix': sys.prefix, 'cases': [],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [ROOT/'src/phase28_reread/model.py', ROOT/'src/phase28_reread/spatial.py']},
        'script_sha256': sha(__file__), 'measurement_repeats_per_case': 1,
        'timing_confidence': 'LOW: single cold-gradient pass, synthetic evidence; memory shape audit only'}
    for name in ('dense_4096', 'sparse64regions_4096'):
        result['cases'].append(case(name))
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        REPORT.write_text(json.dumps({'status': 'FAIL_SYNTHETIC_BENCHMARK',
            'type': type(exc).__name__, 'message': str(exc)[:1000],
            'new_optimizer_steps': 0, 'patient_data_or_endpoint_reads': 0,
            'script_sha256': sha(__file__)}, indent=2)+'\n')
        raise
