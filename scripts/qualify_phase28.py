"""Source-bound synthetic CUDA integration; never reads patient outcomes."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch
from src.phase28_reread import runtime
from src.phase28_reread.spatial import build_plan
from src.phase28_reread.train import make_model, forward_patients, regularizer, model_hash
from src.phase28_reread.training_math import efron_loss, risk_gradient_cache


def seal(value):
    value.update(source=runtime.source(), source_files=runtime.source_files(),
                 script_path='scripts/qualify_phase28.py', script_sha256=runtime.sha(__file__))
    return value


def dependency():
    script = ROOT/'scripts/qualify_phase28_official_mamba.py'
    completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True)
    receipt = runtime.PRIVATE/'DEPENDENCY_QUALIFICATION.json'
    raw = json.loads(receipt.read_text())
    if completed.returncode or not raw['status'].startswith('PASS'):
        raise RuntimeError('official dependency numerical qualification failed; private receipt retained')
    modules = {name: {'filename': Path(v['path']).name, 'sha256': v['sha256']}
               for name, v in raw['official_modules'].items()}
    official_sources = {
        'mamba_ssm.modules.mamba_simple': 'a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3',
        'mamba_ssm.ops.selective_scan_interface': 'c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc',
        'causal_conv1d.causal_conv1d_interface': '7286f93996561532dac8452de1e75bd44a31baf5acbb3ad4fd48b87fbc7fdc94'}
    for name, expected in official_sources.items():
        actual = Path(importlib.util.find_spec(name).origin)
        if runtime.sha(actual) != expected:
            raise AssertionError('installed source differs from verified official release commit')
        modules[name] = {'filename': actual.name, 'sha256': expected}
    public = {key: raw[key] for key in ('status', 'source_kind', 'normal_fresh_process_import',
              'new_optimizer_steps', 'patient_data_or_endpoint_reads', 'custom_S6_substitute',
              'last_state_used_for_objective', 'core', 'scan_checks', 'core_sequence_check',
              'complete_Efron_gradcache_check')}
    public.update(official_modules=modules, official_script_path=str(script.relative_to(ROOT)).replace('\\', '/'),
                  official_script_sha256=runtime.sha(script), packages=raw['environment']['versions'],
                  torch=raw['environment']['torch'], CUDA=raw['environment']['cuda'], GPU=raw['environment']['GPU'],
                  official_Mamba_commit='95d8aba8a8c75aedcaa6143713b11e745e7cd0d9',
                  official_causal_conv1d_commit='82867a9d2e6907cc0f637ac6aff318f696838548')
    runtime.atomic_json(runtime.REPORT/'DEPENDENCY_QUALIFICATION.json', seal(public))
    return public


class SyntheticData:
    def __init__(self):
        generator = torch.Generator().manual_seed(28008)
        self.molecular = torch.randn(4, 26708, 3, generator=generator)
        self.molecular_mask = torch.rand(4, 26708, 3, generator=generator) > .08
        self.molecular_mask[:, :7] = False
        self.gene_ids = torch.arange(26708)
        self.base = torch.tensor([.2, -.1, .6, -.4], dtype=torch.float64)
        self.slot_membership = torch.zeros(108, 26708, dtype=torch.bool)
        for k in range(100):
            self.slot_membership[k, torch.arange((k//2)*47, (k//2)*47+130)] = True
        self.slot_membership[100:] = True
        coords = np.array([(512*x, 512*y) for x in range(6) for y in range(5)
                           if not (x == 2 and y == 2)], dtype=np.int64)
        plan = build_plan(coords)
        self.slides = []
        for patient in range(4):
            values = []
            for slide in range(2 if patient == 0 else 1):
                raw = torch.randn(len(coords), 1536, generator=generator)
                values.append({'features': raw[plan['selected_indices']], 'plan': plan})
            self.slides.append(values)
    def load_patient_slides(self, patient, device=None):
        return [{'features': row['features'].to(device), 'plan': row['plan']} for row in self.slides[patient]]


def integration():
    if not torch.cuda.is_available():
        raise RuntimeError('official CUDA qualification requires CUDA; no fallback')
    ds = SyntheticData(); cases = []
    times, events = np.array([1., 1., 3., 5.]), np.array([1, 1, 0, 1])
    for arm in runtime.ARMS:
        model = make_model(ds, arm, 31, 1)
        # Activate the otherwise initially zero main heads only for the gradient
        # equality check. This is not the formal initialization or a fitted head.
        with torch.no_grad():
            model.main_path.coefficient.normal_(std=.01)
            model.main_gene.axis.normal_(std=.01)
        # Parameter deepcopy drops custom attributes such as Mamba A_log's
        # _no_weight_decay. Instantiate the same official module anew, then
        # copy tensor state so both objectives retain identical parameter rules.
        other = make_model(ds, arm, 31, 1)
        other.load_state_dict(model.state_dict())
        if {n for n, p in model.named_parameters() if getattr(p, '_no_weight_decay', False)} != {
                n for n, p in other.named_parameters() if getattr(p, '_no_weight_decay', False)}:
            raise AssertionError('plain/cache official parameter metadata differs')
        initial = model_hash(model.state_dict())
        before = ds.base.clone()
        complete = forward_patients(model, ds, np.arange(4))
        objective = efron_loss(complete['risk'], times, events)+regularizer(model)
        objective.backward()
        audit = risk_gradient_cache(other, lambda patients: forward_patients(other, ds, patients),
                    np.arange(4), times, events, microbatch=1, regularizer=lambda: regularizer(other))
        largest = 0.
        for (name, parameter), (other_name, second) in zip(model.named_parameters(), other.named_parameters()):
            if name != other_name or (parameter.grad is None) != (second.grad is None):
                raise AssertionError('plain/cache gradient paths disagree')
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all() or not torch.isfinite(second.grad).all():
                    raise AssertionError('nonfinite parameter gradient')
                largest = max(largest, float((parameter.grad-second.grad).abs().max()))
        if largest > 5e-5 or audit['maximum_absolute_component_replay_error'] > 1e-6:
            raise AssertionError('complete-model gradient cache parity exceeds preregistered tolerance')
        if model_hash(other.state_dict()) != initial or not torch.equal(ds.base, before):
            raise AssertionError('qualification changed weights or external offset')
        cases.append({'arm': arm, 'allocated_parameters': other.parameter_counts()['allocated_trainable'],
                      'synthetic_patients': 4, 'full_gene_count': 26708, 'slots': 108,
                      'full_vs_cache_gradient_max_absolute_error': largest, 'audit': audit,
                      'weights_unchanged': True, 'external_offset_unchanged': True,
                      'official_parameter_metadata_preserved': True})
        del model, other, complete, objective
        torch.cuda.empty_cache()
    model = make_model(ds, 'C11', 31, 1)
    raw = ds.molecular[0].to('cuda').requires_grad_()
    encoded = model.encode(raw, ds.molecular_mask[0].to('cuda'), ds.gene_ids.to('cuda'), ds.load_patient_slides(0, 'cuda'))
    output = model.read_encoded(encoded, ds.base[0], details=True)
    rna = torch.autograd.grad(output['path_interaction'], raw, retain_graph=True)[0]
    patch = torch.autograd.grad(output['gene_interaction'], encoded['path_evidence'], retain_graph=True)
    route = {'path_interaction_to_RNA_gradient_L2': float(rna[:, 0].norm()),
             'path_interaction_to_all_molecular_channels_gradient_L2': float(rna.norm()),
             'gene_interaction_to_original_context_patch_gradient_L2': float(torch.cat(patch).norm()),
             'initial_path_risk_correction': float(output['path_interaction'].detach()),
             'initial_gene_risk_correction': float(output['gene_interaction'].detach())}
    if not all(np.isfinite(v) for v in route.values()) or any(route[k] <= 0 for k in list(route)[:3]):
        raise AssertionError('cross direction data gradient missing or nonfinite')
    off = model.read_encoded(encoded, ds.base[0], interaction_off=True)
    if not torch.equal(off['risk'], off['base']):
        raise AssertionError('zero-interaction exact own-main fallback failed')
    return cases, route


def main():
    if Path(sys.prefix).resolve() != runtime.ENV_PYTHON.parent.parent.resolve():
        raise RuntimeError('qualification must run inside the isolated official environment')
    identity = runtime.source()
    dep = dependency()
    cases, gradients = integration()
    if runtime.source() != identity:
        raise RuntimeError('training source changed during qualification')
    result = seal({'status': 'PASS_SYNTHETIC_COMPLETE_MODEL_CUDA_AND_FULL_RISK_GRADIENT_CACHE',
                   'patient_data_or_endpoint_reads': 0, 'new_optimizer_steps': 0,
                   'weights_or_raw_arrays_saved': False, 'cases': cases,
                   'cross_direction_gradients': gradients, 'readout_activation_for_parity_only': True,
                   'formal_initialization_unchanged': True, 'Clinical_directly_conditions_local_queries': False,
                   'dependency_receipt_sha256': runtime.sha(runtime.REPORT/'DEPENDENCY_QUALIFICATION.json'),
                   'does_not_establish_DSS_gain_or_histologic_semantics': True})
    runtime.atomic_json(runtime.REPORT/'NUMERICAL_QUALIFICATION.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(cases), 'source': identity}))


if __name__ == '__main__':
    main()
