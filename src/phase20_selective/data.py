"""Read-only qualified inputs and exact, frozen Phase19 convex offsets."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.phase19_attention import data as prior
from src.phase10_trimodal.data import digest

ROOT = Path(__file__).resolve().parents[2]
P = Path('/root/autodl-tmp/phase20_selective_private')
R = ROOT/'reports/phase20_selective_capacity'
CONFIG = ROOT/'configs/phase20_selective_attention.json'
write = prior.write
TRAINING_TESTS = ['operators', 'model', 'data', 'training', 'diagnostics']


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    initial = list((ROOT/'src/phase20_selective').glob('*.py'))
    initial += [CONFIG, ROOT/'scripts/run_phase20.py']
    initial += [ROOT/f'tests/test_phase20_{name}.py' for name in TRAINING_TESTS]
    files = dict(prior.source_files())
    files.update({p.relative_to(ROOT).as_posix(): digest(p) for p in prior._import_closure(initial)})
    return files


def source():
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reference_audit(ds):
    path = prior.P/'simple'/f'{ds.role}_fold{ds.fold}.pt'
    receipt_path = path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt['source'] != cfg()['upstream_source'] or digest(path) != receipt['checkpoint_sha256']:
        raise ValueError('inherited convex reference has invalid source/hash')
    bank = torch.load(path, map_location='cpu', weights_only=False)
    expected_audit = json.loads(json.dumps(
        {k:v for k,v in ds.upstream_audit.items() if k != 'wrong_controls'},
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), sort_keys=True))
    if bank['source'] != cfg()['upstream_source'] or bank['preprocess_audit'] != expected_audit:
        raise ValueError('inherited reference feature provenance mismatch')
    if not np.array_equal(bank['fit'], ds.fit) or not np.array_equal(bank['valid'], ds.valid):
        raise ValueError('inherited convex reference split mismatch')
    for arm, key in [('J0','X'), ('J_CS','CS')]:
        value = bank['models'][arm]
        actual = np.asarray(ds.f[key][ds.valid] @ value['w'], np.float32)
        if not np.array_equal(actual, value['risk']):
            raise ValueError('inherited convex risk differs from feature/weight formula')
    audit = {'status':'PASS','source':cfg()['upstream_source'],
             'checkpoint_sha256':digest(path), 'receipt_sha256':digest(receipt_path),
             'unique_inherited_convex_models':2,'new_convex_solver_calls':0,
             'both_valid_risks_bitwise_reconstructed':True}
    return bank, audit


def add_anchor(ds):
    bank, audit = reference_audit(ds)
    w = np.asarray(bank['models']['J0']['w'], np.float64)
    fit_risk = np.asarray(ds.f['X'][ds.fit], np.float64) @ w
    anchor = np.zeros(len(ds.f['X']), np.float32)
    anchor[ds.fit] = fit_risk.astype(np.float32)
    anchor[ds.valid] = bank['models']['J0']['risk']
    ds.f['ANCHOR'] = anchor
    ds.ft['ANCHOR'] = torch.as_tensor(anchor, device=ds.visual.device)
    ds.anchor_w = w.copy()
    ds.anchor_sd_fit = float(np.std(fit_risk, ddof=0))
    ds.anchor_reference = audit
    ds.inherited_simple = bank
    return ds


class Data(prior.Data):
    def __init__(self, fold, role, seed, device='cuda'):
        if prior.source() != cfg()['upstream_source']:
            raise ValueError('Phase19 training/data ancestry changed')
        bank_path = prior.P/'preprocessing'/f'fold{fold}_{role}.pt'
        receipt_path = bank_path.with_suffix('.json')
        if not bank_path.is_file() or not receipt_path.is_file():
            raise ValueError('read-only inherited preprocessing prerequisite missing; no automatic refit')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('sha256') != digest(bank_path) or
            receipt.get('provenance',{}).get('source') != cfg()['upstream_source']):
            raise ValueError('inherited preprocessing source/hash mismatch')
        super().__init__(fold, role, seed, device=device)
        self.upstream_audit = copy.deepcopy(self.audit)
        if self.audit['source'] != cfg()['upstream_source']:
            raise ValueError('wrong inherited preprocessing source')
        self.audit = {**self.audit, 'source':source(), 'upstream_source':cfg()['upstream_source'],
                      'upstream_audit':self.upstream_audit,
                      'feature_reuse':'read-only fit-local Phase19 inputs, no new feature fitting',
                      'neural_checkpoint_reuse':False}
        add_anchor(self)
        self.gradient_probe_indices = self.fit[np.unique(np.linspace(0, len(self.fit)-1, min(8,len(self.fit))).astype(int))]
        member = np.asarray(self.f['membership'], bool)
        union = member.any(0)
        # Feature ordering is independently verified by prior.blocks in the
        # collector; this describes membership only, not gene identities.
        self.audit['interaction_input'] = {
            'tokens':24,'RNA_programs':8,'CNA_gain_programs':8,'CNA_loss_programs':8,
            'program_gene_counts':member.sum(1).tolist(),
            'RNA512_genes_in_program_union':int(union.sum()),
            'RNA512_genes_outside_program_union':int((~union).sum()),
            'filters':16, 'rank1_gain':4.,
            'rank_limit_per_program':np.minimum(member.sum(1),16).tolist(),
            'RNA_CNA_separate':True,'gain_loss_separate':True,
            'no_PCA_or_survival_distillation':True,
            'representation_scope':'same complete within-program measured genes; learned signed normalized filters; imposed rank1 versus uncompressed16 outputs'}
        self.audit['frozen_anchor'] = {**self.anchor_reference, 'fit_risk_population_SD':self.anchor_sd_fit,
            'SD_formula':'std(float64(X_fit)@float64(inherited_J0_w), ddof=0)',
            'valid_offset':'bitwise inherited_J0_valid_risk',
            'main_trainable':False,'patient_raw_offsets_public':False}
        self.audit['probes'] = {'all_fit_risk_attention_patients':len(self.fit),
                               'gradient_query_patients':len(self.gradient_probe_indices),
                               'gradient_query_hash':prior.ahash(self.gradient_probe_indices),
                               'sampling':'eight deterministic equally-spaced fit positions; deduplicated across checkpoints in evaluation'}
