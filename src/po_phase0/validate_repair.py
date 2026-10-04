from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.phase_m2.common import censoring_survival, lookup_censor_survival
from src.po_phase0.pathway_repair import cox_loss_fast
from src.po_phase0.runner import cox_loss_grouped


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(repo: Path, report: Path, runtime: Path) -> None:
    fold = pd.read_csv(report / "fold_cell_metrics.csv")
    mechanism = pd.read_csv(report / "mechanism_fold_metrics.csv")
    oof = pd.read_csv(report / "seed_oof_metrics.csv")
    profiles = json.loads((report / "model_profiles.json").read_text())['profiles']
    bootstrap = json.loads((report / "paired_patient_bootstrap.json").read_text())
    canonical = report.parent / 'po_phase0_pathway_representation_probe'
    canonical_manifest = json.loads((canonical / 'manifest_pathway_rna.json').read_text())
    canonical_fold = pd.read_csv(canonical / 'fold_pathway_rna.csv')
    canonical_summary = pd.read_csv(canonical / 'summary_pathway_rna.csv')
    expected_params = {x['model_field']: int(x['params']) for x in profiles}
    expected_models = set(expected_params)
    checks = {}
    checks['pathway_cell_count'] = bool(len(fold) == 45 and all((fold.groupby('method').size() == 15).tolist()))
    checks['pathway_model_fields_match'] = bool((fold['method'] == fold['model_field']).all() and set(fold['method']) == expected_models)
    checks['pathway_params_match_profiles'] = bool(all(int(r.params) == expected_params[r.method] for r in fold.itertuples()))
    checks['rna_full_selection_15_cells'] = bool((fold.query("method == 'Pathway-RNA'").recovery_mode == 'full_selection_refit').all())
    checks['canonical_rna_manifest_correct'] = bool(canonical_manifest.get('model_field') == 'Pathway-RNA' and canonical_manifest.get('model_class') == 'PathwayEncoder' and int(canonical_manifest.get('params')) == 509990 and canonical_manifest.get('full_selection_rerun') is True)
    checks['canonical_rna_files_are_not_15_cell_seed_mean'] = bool(len(canonical_fold) == 15 and set(canonical_fold['model_field']) == {'Pathway-RNA'} and 'seed_mean' not in set(canonical_summary.columns) and set(canonical_summary['aggregation']) == {'mean/std of three per-seed five-fold OOF metrics'})
    checks['replays_all_exact'] = bool(fold.query("method != 'Pathway-RNA'").replay_metric_match.all())
    checks['oof_rows_and_seeds'] = bool(len(oof) == 9 and set(oof.groupby('method').size()) == {3} and set(oof.groupby(['method', 'seed']).size()) == {1})
    checks['mechanism_matrix_90_cells'] = bool(len(mechanism) == 90 and set(mechanism.groupby('method').size()) == {15})
    checks['mechanism_methods'] = bool(set(mechanism['method']) == {'RNA-Linear', 'RNA-LN-Linear', 'RNA-PCA32-Linear', 'RNA-PathwayMean-Linear', 'RNA-MLP', 'RNA-MLP-noLN'})
    checks['paired_bootstrap_has_harrell_and_uno'] = bool(all(k in bootstrap for k in ('delta_harrell_mean', 'delta_harrell_ci95_low', 'delta_harrell_ci95_high', 'delta_uno_mean', 'delta_uno_ci95_low', 'delta_uno_ci95_high')))
    checks['public_reports_have_no_patient_ids'] = True
    for path in report.iterdir():
        if path.is_file() and b'TCGA-' in path.read_bytes():
            checks['public_reports_have_no_patient_ids'] = False
    cache = runtime / 'oof_cache'
    coverage = {}
    for model in sorted(expected_models):
        stem = model.replace('+', '_').replace('-', '_')
        for seed in (20260904, 20260905, 20260906):
            idx = []
            for fold_id in range(1, 6):
                with np.load(cache / f'{stem}_s{seed}_f{fold_id}.npz') as z:
                    idx.extend(np.asarray(z['test_idx'], dtype=int).tolist())
            coverage[f'{model}/{seed}'] = len(idx) == 512 and len(set(idx)) == 512
    checks['all_oof_cache_cover_512_once'] = bool(all(coverage.values()))

    # The vectorized control loss is algebraically checked against the
    # official tied-time grouped implementation on a fixed random fixture.
    torch.manual_seed(20261004)
    times = torch.tensor([1., 2., 2., 3., 5., 5., 7.])
    events = torch.tensor([1., 0., 1., 1., 0., 1., 0.])
    risks = torch.randn(7)
    checks['fast_cox_matches_grouped'] = bool(torch.allclose(cox_loss_fast(times, events, risks), cox_loss_grouped(times, events, risks), atol=1e-7, rtol=1e-7))

    # Git HEAD is the pre-experiment archive; this protects the two raw cell
    # files from accidental overwrite during the repair run.
    preserved = {}
    for rel in ('reports/po_phase0_pathway_representation_probe/fold_pathway_rna_cna.csv', 'reports/po_phase0_pathway_representation_probe/summary_pathway_rna_cna.csv', 'reports/po_phase0_pathway_representation_probe/manifest_pathway_rna_cna.json', 'reports/po_phase0_pathway_representation_probe/fold_signature.csv', 'reports/po_phase0_pathway_representation_probe/summary_signature.csv', 'reports/po_phase0_pathway_representation_probe/manifest_signature.json'):
        current = (repo / rel).read_bytes()
        archived = subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=repo)
        preserved[rel] = {'current_sha256': digest(current), 'head_sha256': digest(archived), 'unchanged': current == archived}
    checks['existing_raw_files_unchanged'] = bool(all(x['unchanged'] for x in preserved.values()))
    payload = {'status': bool(all(checks.values())), 'checks': checks, 'oof_coverage': coverage, 'preserved_raw_hashes': preserved}
    (report / 'validation.json').write_text(json.dumps(payload, indent=2) + '\n')
    if not payload['status']:
        raise SystemExit(json.dumps(payload, indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--runtime', required=True, type=Path)
    a = p.parse_args()
    main(a.repo, a.report, a.runtime)
