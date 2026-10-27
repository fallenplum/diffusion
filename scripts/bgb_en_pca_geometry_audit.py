"""Independent saved-coefficient replay of the nine CPU PCA geometry controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_checkpoint_audit import require, sha256, verify_replayed_risk, metrics_from_risk, verify_metric_fields
from scripts.bgb_en_pca_geometry import source_contract, enforce_lock, canonical, write
from src.phase10_trimodal.reference import reference_efron


def independent_objective_gradient(x, time, event, coefficient, lam=1.):
    """Event-time Efron loop; no training weights, automatic gradients or solver."""
    x, time, event, w = np.asarray(x, float), np.asarray(time), np.asarray(event), np.asarray(coefficient, float)
    risk = x @ w; deaths = int(event.sum())
    require(deaths > 0 and np.isfinite(risk).all(), 'INVALID_GRADIENT_AUDIT_INPUT')
    exp_risk = np.exp(risk - risk.max()); dr = -event.astype(float)
    for current in np.unique(time[event == 1]):
        tied = (time == current) & (event == 1); at_risk = time >= current; count = int(tied.sum())
        for removed in range(count):
            weight = at_risk.astype(float) - removed / count * tied.astype(float)
            contribution = weight * exp_risk
            dr += contribution / contribution.sum()
    gradient = x.T @ (dr / deaths) + lam * w
    objective = reference_efron(time, event, risk) + lam / 2. * float(w @ w)
    return float(objective), gradient


def run(args):
    torch.set_num_threads(2)
    contract, preparations, raw = source_contract(args)
    enforce_lock(args, contract)
    repair_lock = None
    if args.repair_private is not None:
        from scripts.bgb_en_pca_geometry_solver_repair import ensure_repair_lock
        repair_lock = ensure_repair_lock(args, contract)
    rows = []
    for dimension in [32,128,256]:
        for part in [0,1,2]:
            directory = args.private / 'cells' / f'PCA{dimension}_UNWHITENED_p{part}'
            source_private = args.private
            if not (directory / 'COMPLETE.json').exists() and repair_lock is not None:
                require(dict(dimension=dimension,partition=part) in repair_lock['contract']['config']['locked_units'], 'UNLOCKED_REPAIR_UNIT')
                directory = args.repair_private / 'cells' / f'PCA{dimension}_UNWHITENED_p{part}'
                source_private = args.repair_private
            if not (directory / 'COMPLETE.json').exists(): continue
            receipt = json.loads((directory / 'COMPLETE.json').read_text())
            identity = dict(lock_sha256=sha256(source_private / 'LOCK.json'), partition=part, dimension=dimension,
                            source_contract_sha256=canonical(contract))
            require(receipt['identity'] == identity and receipt['artifact_sha256'] == sha256(directory / 'model.pt'), 'NEW_GEOMETRY_ARTIFACT_SOURCE_CHANGED')
            saved = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
            require(saved['identity'] == identity and saved['actual_optimizer_fits'] == 1
                    and saved['PCA_refits'] == saved['teacher_refits'] == 0, 'NEW_GEOMETRY_MODEL_ACCOUNTING_CHANGED')
            prepared = preparations[part]; train, valid = prepared['train'], prepared['valid']
            require(np.array_equal(saved['train'], train) and np.array_equal(saved['valid'], valid), 'NEW_GEOMETRY_SPLIT_CHANGED')
            original_dir = args.core_private / 'cells' / f'R2_outer1_inner{part}_O2_PCA{dimension}_seed31'
            original = torch.load(original_dir / 'model.pt', map_location='cpu', weights_only=False)
            pca = original['feature_state']
            for key in ('components','mean','post_mean','post_scale'):
                require(np.array_equal(saved['saved_PCA_source'][key], pca[key]), 'NEW_GEOMETRY_SAVED_PCA_DRIFT')
            x = (np.asarray(prepared['features']['R'], float) - pca['mean']) @ pca['components'].T
            coef = np.asarray(saved['coefficient']); require(coef.shape == (dimension,), 'NEW_GEOMETRY_COEF_DIMENSION_MISMATCH')
            risk = x @ coef
            error = max(verify_replayed_risk(saved['risk'], risk[valid], len(valid), 2e-9),
                        verify_replayed_risk(saved['risk_train'], risk[train], len(train), 2e-9))
            gene_coef = pca['components'].T @ coef
            require(np.allclose(saved['gene_coefficient'], gene_coef, rtol=0, atol=1e-12), 'NEW_GEOMETRY_GENE_COEF_DRIFT')
            loss, grad = independent_objective_gradient(x[train], raw['t'][train], raw['e'][train], coef)
            require(abs(loss - saved['optimizer']['loss']) <= 1e-10 and np.max(np.abs(grad)) <= 2e-5,
                    'NEW_GEOMETRY_INDEPENDENT_OBJECTIVE_OR_GRADIENT_FAILED')
            metrics = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], risk[valid], prepared['RNA'][valid])
            metric_error = max(verify_metric_fields(metrics, receipt['metrics']), verify_metric_fields(metrics, saved['metrics']))
            original_comparison = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], risk[valid], original['risk'])
            metric_error = max(metric_error, verify_metric_fields(original_comparison, receipt['new_vs_original_repair_damage']))
            scale = receipt['diagnostics']['new_scale']
            scale_error = max(abs(float(risk[train].mean())-scale['mean']), abs(float(risk[train].std())-scale['sd']))
            require(scale_error <= 1e-10, 'NEW_GEOMETRY_RISK_SCALE_NOT_OWN_TRAIN')
            rows.append(dict(dimension=dimension, partition=part, status='PASS',
                numerical_source='strict_precision_retry' if source_private!=args.private else 'original_solver',
                artifact_sha256=sha256(directory/'model.pt'), receipt_sha256=sha256(directory/'COMPLETE.json'),
                maximum_risk_replay_error=error, independent_metric_max_error=metric_error,
                independent_objective_error=abs(loss-saved['optimizer']['loss']), independent_gradient_max=float(np.max(np.abs(grad))),
                own_train_risk_scale_error=scale_error, original_PCA_source_exact=True))
    result = dict(schema='bgb_en_pca_geometry_independent_replay_v1',
        status='PASS_ALL_NINE_SAVED_COX_REPLAYS' if len(rows)==9 else 'PASS_AVAILABLE_PARTIAL_REPLAY',
        source_contract_sha256=canonical(contract), lock_sha256=sha256(args.private/'LOCK.json'),
        audit_script_sha256=sha256(__file__), independent_reference_sha256=sha256(ROOT/'src/phase10_trimodal/reference.py'),
        original_solver_attempts=9, original_valid_units=7 if repair_lock else len(rows),
        precision_retry_attempts=sum((args.repair_private/'cells'/f"PCA{u['dimension']}_UNWHITENED_p{u['partition']}"/'START.json').exists() for u in repair_lock['contract']['config']['locked_units']) if repair_lock else 0,
        actual_saved_Cox_coefficients_replayed=len(rows),
        audit_optimizer_fits=0, PCA_refits=0, teacher_refits=0, GPU_executions=0,
        outer_test_evaluated=False, changes_R4_finalists=False, patient_arrays_or_identifiers_exported=False, units=rows)
    write(args.output, result)
    print(json.dumps({k:result[k] for k in ('status','actual_saved_Cox_coefficients_replayed','audit_optimizer_fits','PCA_refits','teacher_refits')}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True); parser.add_argument('--core-private', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--repair-private', type=Path)
    parser.add_argument('--repair-config', type=Path, default=ROOT/'configs/bgb_en/pca_geometry_solver_repair_v1.json')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--geometry-config', type=Path, default=ROOT/'configs/bgb_en/pca_geometry_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/bgb_en/R2_PCA_GEOMETRY_REPLAY_AUDIT.json')
    run(parser.parse_args())


if __name__=='__main__': main()
