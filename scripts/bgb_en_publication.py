"""Fail-closed publication inventory for explicit anonymous BG-BEN artifacts.

This tool performs no Git staging, commit or push. It never includes files by a
directory wildcard. Private identifier lists remain separate from its public
manifest; a failed scan reports the file and rule, never the matched value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np


SOURCES = (
    'src/bgb_en/__init__.py', 'src/bgb_en/data.py', 'src/bgb_en/protocol.py',
    'src/bgb_en/survival.py', 'src/bgb_en/baselines.py', 'src/bgb_en/pathology.py',
    'src/bgb_en/omics.py', 'src/bgb_en/interaction.py', 'src/bgb_en/external.py',
    'src/bgb_en/training.py', 'src/bgb_en/replay.py', 'src/bgb_en/mechanism_audit.py',
)
CONFIGS = ('configs/bgb_en/prototype_v1.json', 'configs/bgb_en/path_budget_sensitivity_v1.json',
           'configs/bgb_en/budget_sensitivity_v1.json', 'configs/bgb_en/pca_geometry_v1.json',
           'configs/bgb_en/pca_geometry_solver_repair_v1.json',
           'configs/bgb_en/F7_deterministic_runtime_v1.json',
           'configs/bgb_en/matched_support_v1.json',
           'configs/bgb_en/assets.example.json',
           'configs/bgb_en/execution_ledger_inputs.example.json')
SCRIPTS = (
    'scripts/bgb_en_asset_audit.py', 'scripts/bgb_en_audit_transfer_norms.py',
    'scripts/bgb_en_bind_geometry.py', 'scripts/bgb_en_gene_audit.py',
    'scripts/bgb_en_literature_search.py', 'scripts/bgb_en_lock.py',
    'scripts/bgb_en_prepare_patches.py', 'scripts/bgb_en_pretrain_external.py',
    'scripts/bgb_en_run.py', 'scripts/bgb_en_schedule.py',
    'scripts/bgb_en_ssm_float_probe.py', 'scripts/bgb_en_topology_audit.py',
    'scripts/bgb_en_publication.py', 'scripts/bgb_en_outer.py',
    'scripts/bgb_en_report_inner.py', 'scripts/bgb_en_pair_replay.py',
    'scripts/bgb_en_checkpoint_audit.py', 'scripts/bgb_en_aux_run.py',
    'scripts/bgb_en_report_outer.py', 'scripts/bgb_en_missingness.py', 'scripts/bgb_en_plot.py',
    'scripts/bgb_en_batch.py', 'scripts/bgb_en_replay_feature_audit.py',
    'scripts/bgb_en_nested.py', 'scripts/bgb_en_final_neural.py',
    'scripts/bgb_en_path_sensitivity.py',
    'scripts/bgb_en_report_aux.py', 'scripts/bgb_en_make_batch_plan.py',
    'scripts/bgb_en_resource_monitor.py', 'scripts/bgb_en_external_controls.py',
    'scripts/bgb_en_anchor_cast_audit.py', 'scripts/bgb_en_audit_external_replay.py',
    'scripts/bgb_en_omics_sensitivity.py',
    'scripts/bgb_en_report_path_sensitivity.py', 'scripts/bgb_en_backup_private.py',
    'scripts/bgb_en_outer_audit.py', 'scripts/bgb_en_mechanism_replay.py',
    'scripts/bgb_en_select_outer.py', 'scripts/bgb_en_budget_diagnostics.py',
    'scripts/bgb_en_aux_projection_drift.py',
    'scripts/bgb_en_environment.py', 'scripts/bgb_en_outer_schedule.py',
    'scripts/bgb_en_make_budget_plan.py', 'scripts/bgb_en_after_batch.py',
    'scripts/bgb_en_budget_retrain.py', 'scripts/bgb_en_summarize_budget.py',
    'scripts/bgb_en_pca_geometry.py',
    'scripts/bgb_en_pca_geometry_audit.py',
    'scripts/bgb_en_f7_determinism_diagnostic.py',
    'scripts/bgb_en_f7_runtime_repair.py', 'scripts/bgb_en_f7_recovery_schedule.py',
    'scripts/bgb_en_recovery_audit.py', 'scripts/bgb_en_pca_geometry_solver_repair.py',
    'scripts/bgb_en_execution_ledger.py',
    'scripts/bgb_en_recovered_gate.py',
    'scripts/bgb_en_report_f7_runtime.py',
    'scripts/bgb_en_patch_throughput.py', 'scripts/bgb_en_o9_drift_diagnostic.py',
    'scripts/bgb_en_report_o9_drift.py', 'scripts/bgb_en_merge_mechanism_reports.py',
    'scripts/bgb_en_parameter_audit.py',
    'scripts/bgb_en_input_coverage.py', 'scripts/bgb_en_matched_support.py',
    'scripts/bgb_en_matched_support_plan.py',
    'scripts/bgb_en_lineage_inventory.py',
    'scripts/bgb_en_publication_dependencies.py',
    'scripts/bgb_en_matched_support_io_plan.py', 'scripts/bgb_en_report_matched_support.py',
    'scripts/bgb_en_replay_matched_support.py', 'scripts/bgb_en_precompute_nested_prepare.py',
    'scripts/bgb_en_report_prepare_prefetch.py',
    'scripts/bgb_en_budget_parent_drift_audit.py', 'scripts/bgb_en_budget_saved_chain_audit.py',
    'scripts/bgb_en_test_suite_audit.py',
)
TESTS = (
    'tests/bgb_en/test_common.py', 'tests/bgb_en/test_data.py',
    'tests/bgb_en/test_external.py', 'tests/bgb_en/test_interaction.py',
    'tests/bgb_en/test_omics.py', 'tests/bgb_en/test_pathology.py',
    'tests/bgb_en/test_training_mainsep.py', 'tests/bgb_en/test_publication.py',
    'tests/bgb_en/test_report_inner.py',
    'tests/bgb_en/test_checkpoint_audit.py', 'tests/bgb_en/test_report_outer.py',
    'tests/bgb_en/test_outer_guard.py', 'tests/bgb_en/test_missingness.py', 'tests/bgb_en/test_plot.py',
    'tests/bgb_en/test_path_sensitivity.py',
    'tests/bgb_en/test_report_aux.py', 'tests/bgb_en/test_external_controls.py',
    'tests/bgb_en/test_omics_sensitivity.py', 'tests/bgb_en/test_outer_pipeline_review.py',
    'tests/bgb_en/test_report_path_sensitivity.py', 'tests/bgb_en/test_outer_audit.py',
    'tests/bgb_en/test_mechanism_audit.py',
    'tests/bgb_en/test_budget_diagnostics.py',
    'tests/bgb_en/test_aux_projection_drift.py',
    'tests/bgb_en/test_budget_retrain.py', 'tests/bgb_en/test_budget_summary.py',
    'tests/bgb_en/test_pca_geometry.py',
    'tests/bgb_en/test_f7_runtime_repair.py', 'tests/bgb_en/test_f7_determinism_diagnostic.py',
    'tests/bgb_en/test_execution_ledger.py',
    'tests/bgb_en/test_recovery_audit.py',
    'tests/bgb_en/test_report_f7_runtime.py',
    'tests/bgb_en/test_patch_throughput.py', 'tests/bgb_en/test_o9_drift_diagnostic.py',
    'tests/bgb_en/test_o9_drift_report.py', 'tests/bgb_en/test_merge_mechanism_reports.py',
    'tests/bgb_en/test_parameter_audit.py',
    'tests/bgb_en/test_input_coverage.py', 'tests/bgb_en/test_matched_support.py',
    'tests/bgb_en/test_matched_support_plan.py',
    'tests/bgb_en/test_lineage_inventory.py',
    'tests/bgb_en/test_publication_dependencies.py',
    'tests/bgb_en/test_matched_support_io_plan.py', 'tests/bgb_en/test_report_matched_support.py',
    'tests/bgb_en/test_replay_matched_support.py', 'tests/bgb_en/test_precompute_nested_prepare.py',
    'tests/bgb_en/test_budget_parent_drift_audit.py', 'tests/bgb_en/test_budget_saved_chain_audit.py',
    'tests/bgb_en/test_test_suite_audit.py',
)
REPORT_NAMES = (
    'R0_ASSET_AUDIT.md', 'R0_ASSET_AUDIT.json', 'R0_GENE_SOURCES.json',
    'R0_TOPOLOGY_QUALIFICATION.json', 'R0_TOPOLOGY_QUALIFICATION.md',
    'R0_EXTERNAL_TRANSFER_NORM_QUALIFICATION.json', 'R0_EXTERNAL_TRANSFER_NORM_QUALIFICATION.md',
    'R0_OMICS_NUMERICAL_QUALIFICATION.json', 'R0_REAL_INPUT_QUALIFICATION.json',
    'R0_COMPUTE_QUALIFICATION.json', 'R0_COMPUTE_COST.json', 'PROTOCOL_LOCK.json',
    'LITERATURE_NOVELTY_AUDIT.md', 'LITERATURE_SEARCH.json',
    'R1_RESULTS.md', 'R1_ARCH_ABLATION.csv', 'R1_COST.csv', 'R1_FEATURE_INTEGRITY.md',
    'R2_RESULTS.md', 'R2_RESULTS.csv', 'R2_RNA_RETENTION.csv',
    'R2_PATHWAY_MAPPING.md', 'R2_ORDER_SENSITIVITY.csv',
    'R3_RESULTS.md', 'R3_MODEL_MATRIX.csv', 'R3_MECHANISM_ABLATIONS.csv',
    'R3_PAIRING_CONTROLS.csv', 'R3_TRAINING_DIAGNOSTICS.csv',
    'R4_FINAL_RESULTS.md', 'R4_PAIRED_COMPARISONS.csv', 'R4_FOLD_SEED_METRICS.csv',
    'R4_COMPUTE_COST.csv', 'DECISION.md', 'EXPERIMENT_LINEAGE.json', 'REPRODUCE.md',
    'RUNNING_STATUS.json', 'EXTERNAL_PRETRAINING_RESULTS.json',
    'EXTERNAL_PRETRAINING_RESULTS.md', 'SOURCE_QUALIFICATION.json',
    'INNER_REPORT_AUDIT.json', 'MECHANISM_REPLAY.json', 'OUTER_SELECTION_LOCK.json',
    'CHECKPOINT_AUDIT.json', 'R4_REPORT_AUDIT.json', 'PROTOCOL_ACCOUNTING_ADDENDUM.json',
    'R5_PROMPT_DRAFT.md', 'MISSINGNESS_SENSITIVITY.md', 'MISSINGNESS_SENSITIVITY.csv',
    'MISSINGNESS_SENSITIVITY_AUDIT.json', 'figures/PLOT_MANIFEST.json',
    'SAVED_PREPROCESSING_REPLAY.json', 'ANCHOR_CAST_AUDIT.json',
    'PATH_BUDGET_SENSITIVITY_READINESS.md', 'PATH_BUDGET_SENSITIVITY_READINESS.json',
    'AUXILIARY_TRANSFER_RESULTS.json', 'AUXILIARY_TRANSFER_RESULTS.md',
    'AUXILIARY_TRANSFER_FOLD_METRICS.csv', 'AUXILIARY_TRANSFER_SEED_METRICS.csv',
    'AUXILIARY_TRANSFER_PAIRED.csv', 'EXTERNAL_CHECKPOINT_REPLAY.json',
    'EXTERNAL_SUPERVISION_PROVENANCE.md', 'R2_OMICS_SENSITIVITY_QUALIFICATION.json',
    'R4_PIPELINE_REVIEW.json',
    'INNER_REPORT_FOR_SELECTION.json', 'ADDITIONAL_PLAN_LOCK.json',
    'R3_MECHANISM_REPLAY_SUMMARY.json',
    'AUX_PROJECTION_DRIFT.json', 'AUX_PROJECTION_DRIFT.md',
    'EXTERNAL_CHECKPOINT_REPLAY_WITH_CONTROLS.json',
    'BUDGET_PLAN_LOCK.json',
    'R2_BUDGET_SENSITIVITY.json', 'R2_BUDGET_SENSITIVITY.csv', 'R2_BUDGET_SENSITIVITY.md',
    'R2_BUDGET_REPLAY.csv',
    'R2_PCA_GEOMETRY_CONTROL.json', 'R2_PCA_GEOMETRY_CONTROL.csv', 'R2_PCA_GEOMETRY_CONTROL.md',
    'R2_PCA_GEOMETRY_REPLAY_AUDIT.json',
    'DELIVERY_GAPS.md',
    'F7_DETERMINISM_TRIAGE.md', 'EXECUTION_LEDGER.json', 'EXECUTION_LEDGER.md',
    'F7_CHECKPOINT_REPLAY.json', 'F7_RUNTIME_REPAIR_REPORT.json', 'F7_RUNTIME_REPAIR_REPORT.md',
    'PATCH_THROUGHPUT_AUDIT.json', 'PATCH_THROUGHPUT_AUDIT.csv', 'PATCH_THROUGHPUT_AUDIT.md',
    'O9_FLOAT32_DRIFT_NOTE.md', 'O9_FLOAT32_DRIFT_DETERMINISTIC.json',
    'O9_FLOAT32_DRIFT_DEFAULT.json', 'O9_FLOAT32_DRIFT_AGGREGATE.json',
    'R3_MECHANISM_DIAGNOSTIC_NOTE.md', 'REPORT_PLOT_REFRESH_AUDIT.json',
    'PARAMETER_BUDGET_AUDIT.json', 'PARAMETER_BUDGET_AUDIT.csv', 'PARAMETER_BUDGET_AUDIT.md',
    'R2_INPUT_COVERAGE_AUDIT.json', 'R2_INPUT_COVERAGE_AUDIT.md',
    'MATCHED_SUPPORT_NULL_PLAN.md', 'MATCHED_SUPPORT_NULL_LOCK.json',
    'MATCHED_SUPPORT_PUBLIC_PLAN.json', 'MATCHED_SUPPORT_LAUNCH_LOCK.json',
    'BUDGET_DIAGNOSTICS.json', 'BUDGET_DIAGNOSTICS.md',
    'SOURCE_PUBLICATION_CLOSURE_AUDIT.json',
    'MATCHED_SUPPORT_IO_RUNTIME_LOCK.json', 'MATCHED_SUPPORT_RESULTS.json',
    'MATCHED_SUPPORT_RESULTS.md', 'MATCHED_SUPPORT_METRICS.csv',
    'MATCHED_SUPPORT_COMPARISONS.csv', 'MATCHED_SUPPORT_CHECKPOINT_AUDIT.csv',
    'MATCHED_SUPPORT_REPLAY_QUALIFICATION.json',
    'R4_CPU_PREPARE_PREFETCH_AUDIT.json', 'R4_CPU_PREPARE_PREFETCH.md',
    'PATCH_THROUGHPUT_R4_INDEX_QUALIFICATION.json',
    'BUDGET_O4_FAILED_PARENT80_AUDIT.json', 'R2_BUDGET_SAVED_CHAIN_AUDIT.json',
    'R2_BUDGET_SAVED_CHAIN_AUDIT.md', 'R2_BUDGET_SAVED_CHAIN_AUDIT_LOCK.json',
    'MATCHED_SUPPORT_RISK_REPLAY.json', 'TEST_SUITE_AUDIT.json', 'TEST_SUITE_AUDIT.md',
    'R2_OMICS_SENSITIVITY_AUDIT.json', 'R2_OMICS_SENSITIVITY.csv',
    'R2_OMICS_SENSITIVITY_RETENTION.csv', 'R2_OMICS_SENSITIVITY.md',
    'OUTER_AUDIT_READINESS.json', 'OUTER_CHECKPOINT_AUDIT.json',
    'R4_RESTART_INTEGRITY_20261028.json',
    'PATH_BUDGET_SENSITIVITY_RESULTS.json', 'PATH_BUDGET_SENSITIVITY_RESULTS.md',
    'PATH_BUDGET_SENSITIVITY_FOLD_METRICS.csv', 'PATH_BUDGET_SENSITIVITY_SEED_METRICS.csv',
    'PATH_BUDGET_SENSITIVITY_PAIRED.csv', 'PATH_BUDGET_SENSITIVITY_LEARNING_CURVES.csv',
    'PATH_BUDGET_SENSITIVITY_RISK_DISTRIBUTION.csv',
)
REPORTS = tuple('reports/bgb_en/' + name for name in REPORT_NAMES)
FIGURE_SOURCES = {
    'R1_PERFORMANCE_COST': 'R1_ARCH_ABLATION.csv', 'R2_PERFORMANCE_COST': 'R2_RESULTS.csv',
    'R3_PERFORMANCE_COST': 'R3_MODEL_MATRIX.csv', 'R2_RNA_RETENTION': 'R2_RNA_RETENTION.csv',
    'R3_MECHANISM_DELTAS': 'R3_MECHANISM_ABLATIONS.csv',
    'R3_PAIR_REPAIR_DAMAGE': 'R3_PAIRING_CONTROLS.csv',
    'R3_RISK_SHIFT_REPAIR_DAMAGE': 'R3_MODEL_MATRIX.csv',
    'PATCH_THROUGHPUT': 'PATCH_THROUGHPUT_AUDIT.csv',
}
FIGURES = tuple('reports/bgb_en/figures/'+name+'.'+extension
                for name in FIGURE_SOURCES for extension in ('svg','png'))
ALLOWLIST = frozenset(SOURCES + CONFIGS + SCRIPTS + TESTS + REPORTS + FIGURES)
MANIFEST_NAME = 'reports/bgb_en/PUBLICATION_MANIFEST.json'
CANONICAL_COHORT_SHA256 = '4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1'
CANONICAL_PATIENT_SHA256 = '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f'
CANONICAL_ST_PREPARED_SHA256 = 'b5a2a44862123f629314cb28ad66abae437d7d08e923fb6128f82f7e9059c025'
CANONICAL_ST_IDENTIFIER_SHA256 = '6bddc982cedee88008d21b0ee500b1f8e48144d1fb595afe576118905e420894'

PATTERNS = {
    'TCGA patient/slide identifier': re.compile(r'(?<![A-Za-z0-9])TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Za-z0-9.-]+)?'),
    'Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    'GitHub credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'private SSH host': re.compile(r'connect\.west[a-z0-9]*\.seetacloud\.com', re.I),
    'private key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'private absolute path': re.compile(r'(?:/(?:root|home|lustre|autodl-tmp)/[A-Za-z0-9_]|[A-Za-z]:[\\/](?:Users|develop|diffusion|dataset)[\\/])'),
}
RAW_JSON_KEYS = frozenset(('patient_ids', 'patient_id', 'barcodes', 'barcode', 'rowids',
                           'patient_indices', 'train_indices', 'test_indices', 'fit_indices',
                           'valid_indices', 'coordinates', 'coords', 'risk_vector', 'OOF_predictions',
                           'predictions', 'RNA_matrix', 'CNA_matrix', 'state_dict', 'optimizer_state'))
ARRAY_ONLY_PRIVATE_KEYS = frozenset(('risk', 'risks', 'pred', 'prediction', 'weights',
                                     'coefficients', 'features', 'embedding', 'embeddings'))


class PublicationError(ValueError):
    pass


def require(value: Any, message: str):
    if not bool(value):
        raise PublicationError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def build_denylist(cohort: str | Path, stage_b: str | Path) -> dict[str, Any]:
    """Read exact private identifier fields only, never an endpoint or features."""
    require(sha256(cohort) == CANONICAL_COHORT_SHA256, 'canonical cohort identity mismatch')
    require(sha256(stage_b) == CANONICAL_ST_PREPARED_SHA256, 'qualified Stage B identity mismatch')
    with np.load(cohort, allow_pickle=False) as archive:
        patient = archive['patient_ids'].astype(str)
    require(len(patient) == 512 and len(np.unique(patient)) == 512
            and hashlib.sha256('\n'.join(patient).encode()).hexdigest() == CANONICAL_PATIENT_SHA256,
            'canonical patient registry mismatch')
    groups = {'patient': patient.tolist()}
    with np.load(stage_b, allow_pickle=False) as archive:
        for key in ('donor', 'section', 'barcodes'):
            require(key in archive.files, 'qualified Stage B identifier field missing')
            groups[key] = np.unique(archive[key].astype(str)).tolist()
    require(len(groups['donor']) == 6 and len(groups['section']) == 12
            and len(groups['barcodes']) > 1000, 'private ST registry coverage mismatch')
    require(all(value and '\x00' not in value for values in groups.values() for value in values),
            'empty or binary identifier in private registry')
    return {'schema': 'bgb_en_private_publication_denylist_v1',
            'cohort_sha256': CANONICAL_COHORT_SHA256, 'patient_order_sha256': CANONICAL_PATIENT_SHA256,
            'stage_b_sha256': CANONICAL_ST_PREPARED_SHA256, 'groups': groups}


def validate_denylist(value: dict[str, Any]) -> dict[str, Any]:
    require(value.get('schema') == 'bgb_en_private_publication_denylist_v1'
            and value.get('cohort_sha256') == CANONICAL_COHORT_SHA256
            and value.get('patient_order_sha256') == CANONICAL_PATIENT_SHA256
            and value.get('stage_b_sha256') == CANONICAL_ST_PREPARED_SHA256,
            'private denylist provenance mismatch')
    groups = value.get('groups', {})
    require(set(groups) == {'patient', 'donor', 'section', 'barcodes'}, 'private denylist groups mismatch')
    patient = groups['patient']
    require(len(patient) == len(set(patient)) == 512
            and hashlib.sha256('\n'.join(patient).encode()).hexdigest() == CANONICAL_PATIENT_SHA256,
            'private denylist canonical patient ordering mismatch')
    require(len(groups['donor']) == 6 and len(groups['section']) == 12 and len(groups['barcodes']) > 1000,
            'private denylist ST coverage mismatch')
    require(all(isinstance(item, str) and item for values in groups.values() for item in values),
            'invalid private denylist identifier')
    st = {key: groups[key] for key in ('donor', 'section', 'barcodes')}
    require(hashlib.sha256(json.dumps(st, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == CANONICAL_ST_IDENTIFIER_SHA256, 'private denylist ST identifier values/order changed')
    return value


def identifier_pattern(denylist: dict[str, Any]):
    values = set(item for group in denylist['groups'].values() for item in group)
    # Exact-token boundaries prevent a short donor token from matching a
    # substring inside an unrelated word. Long barcodes/TCGA IDs also match
    # inside structured composites and retain their private full identifiers.
    tokens = []
    for value in sorted(values, key=lambda item: (-len(item), item)):
        escaped = re.escape(value)
        tokens.append(escaped if len(value) >= 8 else r'(?<![A-Za-z0-9])' + escaped + r'(?![A-Za-z0-9])')
    return re.compile('|'.join(tokens))


def validate_public_json(value: Any, *, context='root', allow_cache_hashes=False):
    if isinstance(value, dict):
        hash_context = context == 'root.new_feature_only_cache.array_sha256' or bool(
            re.fullmatch(r'root\.caches\.patch_(?:regions|grid|rand|diverse|uni2)\.array_sha256', context))
        if allow_cache_hashes and hash_context:
            known = {'features', 'coords', 'mask', 'rowids', 'owner', 'quota',
                     'full_mean', 'selected_mean', 'coordinate_step', 'region_ids'}
            require(set(value) <= known and {'features', 'coords', 'mask', 'owner'} <= set(value)
                    and all(isinstance(item, str) and re.fullmatch('[0-9a-f]{64}', item) for item in value.values()),
                    'invalid explicitly scoped cache-hash dictionary')
            return
        for key, item in value.items():
            require(key not in RAW_JSON_KEYS, 'private row-level JSON field: ' + context + '.' + key)
            require(not (key in ARRAY_ONLY_PRIVATE_KEYS and isinstance(item, list)),
                    'private raw array JSON field: ' + context + '.' + key)
            validate_public_json(item, context=context + '.' + key, allow_cache_hashes=allow_cache_hashes)
    elif isinstance(value, list):
        require(len(value) <= 1000, 'unbounded public JSON array: ' + context)
        require(not (len(value) > 128 and all(isinstance(item, (int, float, list)) for item in value)),
                'large numeric array is not an anonymous aggregate report: ' + context)
        for item in value:
            validate_public_json(item, context=context, allow_cache_hashes=allow_cache_hashes)
    elif isinstance(value, float):
        require(np.isfinite(value), 'nonfinite public JSON scalar: ' + context)


def validate_figure_binding(root: Path, relative: str):
    path = root/relative
    require(relative in FIGURES, 'unknown binary/vector figure name')
    manifest_path = root/'reports/bgb_en/figures/PLOT_MANIFEST.json'
    require(manifest_path.is_file(), 'figure requires anonymous plot provenance manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    require(manifest.get('schema') == 'bgb_en_anonymous_plot_manifest_v1'
            and manifest.get('status') == 'COMPLETE_AVAILABLE_FIGURES_ONLY'
            and manifest.get('private_assets_read') is False
            and manifest.get('scientific_completion_inferred') is False,
            'plot provenance is not the qualified anonymous renderer contract')
    require(manifest.get('plotting_source_sha256') == sha256(root/'scripts/bgb_en_plot.py'),
            'plot rendering code differs from provenance')
    entries = [row for row in manifest.get('plots', []) if row.get('name') == path.stem]
    require(len(entries) == 1, 'figure has absent/duplicate provenance entry')
    entry = entries[0]
    sources = entry.get('source_files', [])
    require(len(sources) == 1 and sources[0].get('file') == FIGURE_SOURCES[path.stem],
            'figure source is not its explicit anonymous aggregate CSV')
    source = root/'reports/bgb_en'/sources[0]['file']
    require(source.is_file() and sha256(source) == sources[0].get('sha256'), 'figure source CSV hash changed')
    files = entry.get('files', [])
    require({item.get('file') for item in files} == {path.stem+'.svg', path.stem+'.png'} and len(files) == 2,
            'figure requires the exact PNG/vector pair')
    for item in files:
        image = path.parent/item['file']
        require(image.is_file() and not image.is_symlink() and sha256(image) == item.get('sha256')
                and image.stat().st_size == item.get('bytes'), 'figure pair bytes differ from manifest')
    return entry


def scan_file(root: Path, relative: str, id_pattern) -> dict[str, Any]:
    require(relative in ALLOWLIST, 'file is not in the explicit BG-BEN publication allowlist')
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and '..' not in pure.parts, 'unsafe publication relative path')
    path = root.joinpath(*pure.parts)
    require(not path.is_symlink() and path.is_file(), 'missing/symlink publication file: ' + relative)
    require(path.resolve().is_relative_to(root.resolve()), 'publication path escapes root')
    # Symlinked parents may otherwise redirect a nominally whitelisted path.
    require(not any(parent.is_symlink() for parent in [path.parent, *path.parents]
                    if parent != root.parent and parent.is_relative_to(root)),
            'publication parent is a symlink: ' + relative)
    binding = validate_figure_binding(root, relative) if relative in FIGURES else None
    if path.suffix.lower() == '.png':
        require(relative in FIGURES, 'unknown binary image blocked')
        raw = path.read_bytes()
        require(raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw) <= 8*1024**2, 'invalid/oversized diagnostic PNG')
        declared = next(item for item in binding['files'] if item['file'] == path.name)
        require(hashlib.sha256(raw).hexdigest() == declared['sha256'], 'PNG changed after provenance verification')
        # The paired SVG keeps searchable labels and numeric chart geometry;
        # both it and the exact source CSV undergo the ordinary privacy scan.
        scan_file(root, relative[:-4]+'.svg', id_pattern)
        source = 'reports/bgb_en/'+FIGURE_SOURCES[path.stem]
        scan_file(root, source, id_pattern)
        return {'file': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    require(path.suffix.lower() in ('.py', '.json', '.md', '.csv', '.svg'), 'binary artifact extension blocked')
    require(path.suffix.lower() != '.svg' or relative in FIGURES, 'unknown SVG blocked')
    before = path.stat()
    require(before.st_size <= 2*1024**2, 'oversized publication artifact: ' + relative)
    try:
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
    except (UnicodeDecodeError, OSError) as exc:
        raise PublicationError('publication file is not readable UTF8 text: ' + relative) from exc
    require('\x00' not in text, 'binary content in publication file: ' + relative)
    for label, pattern in PATTERNS.items():
        require(pattern.search(text) is None, 'privacy rule failed [' + label + ']: ' + relative)
    require(id_pattern.search(text) is None, 'actual private patient/ST identifier found: ' + relative)
    if path.suffix.lower() == '.svg':
        require('<image' not in text and 'data:image' not in text
                and re.search(r'(?:xlink:)?href=[\"\x27](?!#)', text) is None,
                'SVG may not embed raw raster images or external media')
    if path.suffix.lower() == '.json':
        try:
            value = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise PublicationError('malformed publication JSON: ' + relative) from exc
        validate_public_json(value, allow_cache_hashes=relative in (
            'reports/bgb_en/R0_ASSET_AUDIT.json', 'reports/bgb_en/R0_TOPOLOGY_QUALIFICATION.json'))
    if path.suffix.lower() == '.csv':
        rows = list(csv.reader(io.StringIO(text)))
        require(rows and len(rows) <= 20000, 'empty/oversized aggregate CSV: ' + relative)
        require(len(rows[0]) == len(set(rows[0])) and len(rows[0]) <= 150, 'CSV header invalid: ' + relative)
        require(not set(rows[0]) & RAW_JSON_KEYS, 'private row-level CSV header: ' + relative)
        require(all(len(row) == len(rows[0]) for row in rows), 'ragged aggregate CSV: ' + relative)
    inspected_sha = hashlib.sha256(raw).hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and after.st_size == len(raw) and sha256(path) == inspected_sha,
            'publication file changed during inspection: ' + relative)
    return {'file': relative, 'sha256': inspected_sha, 'bytes': len(raw)}


def validate_r0_qualification(root: Path):
    """Bind public R0 claims to the actual frozen cohort and loader evidence."""
    asset_path = root/'reports/bgb_en/R0_ASSET_AUDIT.json'
    topology_path = root/'reports/bgb_en/R0_TOPOLOGY_QUALIFICATION.json'
    require(asset_path.is_file() and topology_path.is_file(), 'qualified R0 asset/topology evidence missing')
    asset = json.loads(asset_path.read_text(encoding='utf-8-sig'))
    topology = json.loads(topology_path.read_text(encoding='utf-8-sig'))
    require(asset.get('schema') == 'bgb_en_R0_asset_audit_v1'
            and asset.get('status') == 'PASS_COHORT_AND_AVAILABLE_PATCH_ASSETS'
            and asset.get('model_fits') == 0 and asset.get('patient_identifiers_public') is False
            and asset.get('private_paths_public') is False, 'R0 public asset qualification is not valid')
    cohort = asset.get('cohort', {})
    require(cohort.get('patients') == 512 and cohort.get('DSS_events') == 65
            and cohort.get('outer_folds') == 5
            and cohort.get('hashes', {}).get('cohort_sha256') == CANONICAL_COHORT_SHA256
            and cohort.get('hashes', {}).get('patient_order_sha256') == CANONICAL_PATIENT_SHA256,
            'R0 public cohort source qualification differs')
    require(asset.get('CONCH', {}).get('slides') == 519 and asset.get('CONCH', {}).get('patches') == 1278642
            and asset.get('CONCH', {}).get('full_original_H5_hashes_verified') is True
            and asset.get('UNI2', {}).get('slides') == 520 and asset.get('UNI2', {}).get('patches') == 3295129,
            'R0 original patch source qualification differs')
    require(topology.get('schema') == 'bgb_en_R0_topology_qualification_v1'
            and topology.get('status') == 'PASS_INPUTS_FROZEN_BEFORE_FITS'
            and topology.get('outcome_reads') == topology.get('model_fits') == 0
            and topology.get('risk_labels_used_to_choose_patch_selector') is False
            and topology.get('main_selector') == 'REGIONS512', 'R0 topology was not qualified before fitting')
    require(topology.get('source_data_module_sha256') == sha256(root/'src/bgb_en/data.py'),
            'published data loader differs from qualified frozen R0 loader')
    require(set(topology.get('caches', {})) == {'patch_regions', 'patch_grid', 'patch_rand', 'patch_diverse', 'patch_uni2'},
            'R0 explicit cache qualification coverage differs')
    for key, cache in topology['caches'].items():
        require(cache.get('status') == 'COMPLETE' and cache.get('outcome_reads') == 0,
                'R0 cache is not qualified: ' + key)
        expected = {'512': 520} if key == 'patch_uni2' else {'1024': 519}
        require(cache.get('coordinate_step_counts') == expected, 'R0 native hop qualification differs: ' + key)
    return {'status': 'PASS_FROZEN_R0_SOURCE_QUALIFICATION', 'asset_report_sha256': sha256(asset_path),
            'topology_report_sha256': sha256(topology_path),
            'data_loader_sha256': topology['source_data_module_sha256']}


def audit(root: str | Path, denylist: dict[str, Any], *, run_state='RUNNING', requested=None):
    root = Path(root).resolve()
    require(run_state in ('RUNNING', 'PARTIAL', 'COMPLETE', 'BLOCKED'), 'invalid scientific run state')
    denylist = validate_denylist(denylist)
    id_pattern = identifier_pattern(denylist)
    requested = sorted(ALLOWLIST) if requested is None else list(requested)
    require(len(requested) == len(set(requested)), 'duplicate publication request')
    require(all(name in ALLOWLIST for name in requested), 'request contains unknown publication file')
    files, absent = [], []
    for relative in sorted(requested):
        if not (root/relative).exists():
            absent.append(relative)
            continue
        files.append(scan_file(root, relative, id_pattern))
    required = set(SOURCES + CONFIGS) | {'reports/bgb_en/R0_ASSET_AUDIT.json', 'reports/bgb_en/R0_ASSET_AUDIT.md'}
    require(required <= {item['file'] for item in files}, 'required BG-BEN code/config/R0 audit missing')
    qualification = validate_r0_qualification(root)
    if run_state == 'COMPLETE':
        require({'reports/bgb_en/R1_RESULTS.md', 'reports/bgb_en/R2_RESULTS.md', 'reports/bgb_en/R3_RESULTS.md',
                 'reports/bgb_en/DECISION.md', 'reports/bgb_en/EXPERIMENT_LINEAGE.json',
                 'reports/bgb_en/REPRODUCE.md'} <= {item['file'] for item in files},
                'COMPLETE requires actual stage results, decision, lineage and reproduction reports')
    known = set(ALLOWLIST) | {MANIFEST_NAME}
    ignored = []
    for directory in ('src/bgb_en', 'configs/bgb_en', 'tests/bgb_en', 'reports/bgb_en'):
        parent = root/directory
        if parent.exists():
            # Discovery is only to report excluded files; it never grants
            # permission or adds them to the publication manifest.
            ignored += [str(path.relative_to(root)).replace('\\', '/') for path in parent.rglob('*')
                        if path.is_file() and str(path.relative_to(root)).replace('\\', '/') not in known
                        and '__pycache__' not in path.parts]
    content = ''.join(item['file']+'\0'+item['sha256']+'\0'+str(item['bytes'])+'\n' for item in files)
    return {'schema': 'bgb_en_publication_manifest_v1', 'status': 'PASS_ANONYMOUS_EXPLICIT_FILES_ONLY',
            'scientific_run_state': run_state, 'scientific_completion_inferred_from_privacy_audit': False,
            'files': files, 'files_count': len(files), 'total_bytes': sum(item['bytes'] for item in files),
            'ordered_file_manifest_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'actual_private_identifier_groups_scanned': {key: len(values) for key, values in denylist['groups'].items()},
            'private_registry_provenance': {key: denylist[key] for key in ('cohort_sha256', 'patient_order_sha256', 'stage_b_sha256')},
            'R0_source_qualification': qualification,
            'public_private_registry_values': False, 'credentials_patient_data_weights_and_raw_arrays_included': False,
            'unknown_files_grant_publication_permission': False,
            'excluded_unknown_file_count': len(ignored), 'absent_optional_explicit_file_count': len(absent),
            'git_operations_performed': 0,
            'manifest_self_hash': 'not self-referential; hash the written manifest separately'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--stage-b-prepared', type=Path)
    parser.add_argument('--denylist', type=Path, help='Private registry produced from exact qualified source assets')
    parser.add_argument('--export-denylist', type=Path, help='Private destination outside source/config/test/report trees')
    parser.add_argument('--run-state', choices=('RUNNING', 'PARTIAL', 'COMPLETE', 'BLOCKED'), default='RUNNING')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.denylist:
        denylist = validate_denylist(json.loads(args.denylist.read_text(encoding='utf-8-sig')))
    else:
        require(args.cohort is not None and args.stage_b_prepared is not None, 'exact source assets or a qualified private denylist are required')
        denylist = build_denylist(args.cohort, args.stage_b_prepared)
    if args.export_denylist:
        resolved = args.export_denylist.resolve()
        require(not any(resolved.is_relative_to((args.root/folder).resolve()) for folder in ('src', 'scripts', 'configs', 'tests', 'reports')),
                'denylist must remain outside all public artifact trees')
        args.export_denylist.parent.mkdir(parents=True, exist_ok=True)
        args.export_denylist.write_text(json.dumps(denylist, allow_nan=False)+'\n', encoding='utf-8')
    result = audit(args.root, denylist, run_state=args.run_state)
    output = args.output if args.output else args.root/MANIFEST_NAME
    require(output.resolve() != (args.root/'configs/bgb_en/prototype_v1.json').resolve(), 'output cannot replace scientific configuration')
    require(output.resolve() == (args.root/MANIFEST_NAME).resolve()
            or not output.resolve().is_relative_to(args.root.resolve()),
            'in-workspace publication output must be the exact publication manifest path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'scientific_run_state': result['scientific_run_state'],
                      'files': result['files_count'], 'bytes': result['total_bytes'],
                      'ordered_file_manifest_sha256': result['ordered_file_manifest_sha256'],
                      'written_manifest_sha256': sha256(output)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except PublicationError as exc:
        print(json.dumps({'status': 'FAIL_CLOSED_PUBLICATION', 'reason': str(exc)}), flush=True)
        raise SystemExit(2)
