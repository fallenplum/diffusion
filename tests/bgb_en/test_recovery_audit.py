import json
from pathlib import Path
import pytest
from scripts.bgb_en_recovery_audit import (certify_recovery, verify_recovery_receipt, sha256,
    matrix_inventory, RUNTIME_SOURCES)
from scripts.bgb_en_f7_runtime_repair import original_inventory


def fixture_files(tmp_path):
    root, private, archive = tmp_path/'repo', tmp_path/'formal', tmp_path/'archive'
    for folder in (root, private, archive):
        folder.mkdir()
    config = json.loads(Path('configs/bgb_en/prototype_v1.json').read_text())
    config_path = root/'protocol.json'; config_path.write_text(json.dumps(config))
    lock = json.loads(Path(RUNTIME_SOURCES[-1]).read_text())
    for name in (*RUNTIME_SOURCES, 'scripts/bgb_en_recovery_audit.py'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lock) if name == RUNTIME_SOURCES[-1] else 'synthetic immutable source\n')
    (root/'core.py').write_text('synthetic core closure\n')
    lineage = dict(protocol_sha256=sha256(config_path), source={'core.py': sha256(root/'core.py')}, cohort_sha256='synthetic')
    for part in (0, 1, 2):
        path = private/'prepared'/f'outer1_inner{part}.pt'; path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'synthetic prepared, no tensor loading')
        path.with_suffix('.json').write_text(json.dumps(dict(lineage=lineage, sha256=sha256(path))))
    for part in (0, 1, 2):
        for seed in (31, 47):
            name = f'R3_outer1_inner{part}_F7_seed{seed}'
            folder = archive/name; folder.mkdir()
            if name in lock['expected_original_failed']:
                row = dict(lineage=lineage, type='RuntimeError', message='Gradient-cache forward was not deterministic')
                (folder/'FAILED.json').write_text(json.dumps(row))
            else:
                (folder/'model.pt').write_bytes(b'original model')
                row = dict(lineage=lineage, arm='F7', epochs=80, artifact_sha256=sha256(folder/'model.pt'))
                (folder/'COMPLETE.json').write_text(json.dumps(row))
    origin = dict(schema='bgb_en_F7_original_inventory_v1', sources={name: sha256(root/name) for name in RUNTIME_SOURCES},
                  original_cells=original_inventory(archive, lock))
    manifest = tmp_path/'origin.json'; manifest.write_text(json.dumps(origin))
    backend = dict(mode='deterministic', CUBLAS_WORKSPACE_CONFIG=':4096:8', deterministic_algorithms=True,
        deterministic_warn_only=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
        cudnn_benchmark=False, cudnn_deterministic=True, matmul_precision='highest')
    for stage in ('R1', 'R2', 'R3'):
        for arm in config[stage]['arms']+config[stage].get('additional_real_retraining', []):
            convex = arm in ('P0', 'F0', 'F1', 'O0', 'O1') or arm.startswith('O2_')
            for part in (0, 1, 2):
                for seed in ([31] if convex else (31, 47)):
                    folder = private/'cells'/f'{stage}_outer1_inner{part}_{arm}_seed{seed}'
                    folder.mkdir(parents=True)
                    (folder/'model.pt').write_bytes(b'new matrix model')
                    complete = dict(status='COMPLETE', stage=stage, arm=arm, outer_fold=1, inner_partition=part, seed=seed,
                        lineage=lineage, outer_test_evaluated=False, artifact_sha256=sha256(folder/'model.pt'), epochs=80)
                    (folder/'COMPLETE.json').write_text(json.dumps(complete))
                    if stage == 'R3' and arm == 'F7':
                        prep = private/'prepared'/f'outer1_inner{part}.pt'
                        runtime = dict(schema='bgb_en_F7_runtime_repair_receipt_v1', status='COMPLETE_F7_DETERMINISTIC_FRESH_FIT',
                            partition=part, seed=seed, core_lineage=lineage, artifact_sha256=complete['artifact_sha256'],
                            core_receipt_sha256=sha256(folder/'COMPLETE.json'), actual_new_fits=1, actual_new_updates=80,
                            teacher_refits=0, optimizer_resumed=False, gradient_cache_threshold=2e-5, threshold_unchanged=True,
                            sources=origin['sources'], backend=backend, original_inventory_sha256=sha256(manifest),
                            prepared_origin_sha256=sha256(prep), prepared_receipt_sha256=sha256(prep.with_suffix('.json')))
                        (folder/'RUNTIME_REPAIR_COMPLETE.json').write_text(json.dumps(runtime))
    (private/'R3_SCHEDULE.json').write_text(json.dumps(dict(status='PARTIAL_FAILED_UNITS_CONTINUE_INDEPENDENT_MECHANISMS')))
    (private/'SCHEDULER_RECEIPT.json').write_text(json.dumps(dict(status='FINISHED_CHECK_UNIT_FAILURES')))
    return root, private, archive, manifest, config_path


def test_recovery_requires_186_new_current_cells_and_replays_all_archive_runtime_hashes(tmp_path):
    root, private, archive, manifest, config = fixture_files(tmp_path)
    old_schedule_sha = sha256(private/'R3_SCHEDULE.json')
    receipt = certify_recovery(private, manifest, archive, config_path=config, root=root)
    assert receipt['core_unique_cells'] == 186
    assert receipt['per_stage_unique_cells'] == dict(R1=57, R2=57, R3=72)
    assert receipt['actual_new_updates'] == 480 and receipt['original_failed_updates'] is None
    assert len(receipt['repaired_cells']) == 6 and receipt['original_R3_schedule_still_partial']
    path = tmp_path/'RECOVERY_RECEIPT.json'; path.write_text(json.dumps(receipt))
    assert verify_recovery_receipt(path, private, config_path=config, root=root) == receipt
    assert sha256(private/'R3_SCHEDULE.json') == old_schedule_sha
    # A stale/newly replaced checkpoint cannot be admitted by a boolean COMPLETE.
    target = private/'cells'/'R3_outer1_inner0_F7_seed31'/'model.pt'
    target.write_bytes(b'mutated model')
    with pytest.raises(ValueError, match='ARTIFACT_SHA_DIFFERS'):
        verify_recovery_receipt(path, private, config_path=config, root=root)


def test_recovery_cannot_hide_partial_current_matrix_or_rewrite_old_partial_schedule(tmp_path):
    root, private, archive, manifest, config = fixture_files(tmp_path)
    missing = private/'cells'/'R3_outer1_inner0_F7_seed31'/'COMPLETE.json'
    missing.rename(missing.with_suffix('.hidden'))
    with pytest.raises(ValueError, match='INCOMPLETE_AFTER_RECOVERY'):
        matrix_inventory(private, json.loads(config.read_text()))
    missing.with_suffix('.hidden').rename(missing)
    (private/'R3_SCHEDULE.json').write_text(json.dumps(dict(status='COMPLETE')))
    with pytest.raises(ValueError, match='NOT_PRESERVED_PARTIAL'):
        certify_recovery(private, manifest, archive, config_path=config, root=root)
