import importlib.util
import json
from pathlib import Path
import pytest
from scripts.bgb_en_merge_mechanism_reports import merge_reports, copy_private_reports, ARMS, AUDIT_SOURCES
from scripts.bgb_en_recovery_audit import sha256


def fixture_reports(tmp_path):
    spec = importlib.util.spec_from_file_location('recovery_fixture_for_mechanism_merge', Path(__file__).with_name('test_recovery_audit.py'))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    root, private, _, _, config = fixture.fixture_files(tmp_path)
    for name in (*AUDIT_SOURCES, 'scripts/bgb_en_merge_mechanism_reports.py'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic audit source')
    audit_sources = {name: sha256(root/name) for name in AUDIT_SOURCES}
    inputs = [private/'mechanism_default', private/'mechanism_F7']
    for input_root in inputs:
        input_root.mkdir()
        inventory = []
        for arm in ARMS:
            if (arm == 'F7') != (input_root == inputs[1]):
                continue
            for part in (0, 1, 2):
                for seed in (31, 47):
                    name = f'R3_outer1_inner{part}_{arm}_seed{seed}'
                    cell = private/'cells'/name; complete = json.loads((cell/'COMPLETE.json').read_text())
                    contract = dict(artifact_sha256=complete['artifact_sha256'],
                        prepared_sha256=sha256(private/'prepared'/f'outer1_inner{part}.pt'),
                        audit_sources=audit_sources, patient_limit=8, device='cuda')
                    runtime = None
                    if arm == 'F7':
                        runtime = json.loads((cell/'RUNTIME_REPAIR_COMPLETE.json').read_text())
                        contract.update(runtime_repair_receipt_sha256=sha256(cell/'RUNTIME_REPAIR_COMPLETE.json'), runtime_backend=runtime['backend'])
                    amplitude = {name: dict(SD=.1) for name in ('interaction', 'OtoP_risk_raw', 'PtoO_risk_raw', 'pooled_path', 'pooled_omics')}
                    amplitude['trained_delta_SD_over_offset_SD'] = .2
                    for branch in ('interaction', 'pooled_path', 'pooled_omics'):
                        amplitude['bounded_'+branch+'_readout_derivative'] = dict(mean=.02)
                    for field in ('interaction_raw_abs_gt2_fraction', 'interaction_readout_derivative_below_one_percent_max_fraction',
                                  'pooled_path_raw_abs_gt2_fraction', 'pooled_omics_raw_abs_gt2_fraction'):
                        amplitude[field] = 0.
                    row = dict(status='PASS_FINAL80_MECHANISM_REPLAY', optimizer_updates=0,
                        identity=dict(arm=arm, outer_fold=1, partition=part, seed=seed), contract=contract,
                        diagnostic_validation_patients=8, full_validation_patients=16,
                        sampled_final80_risk_replay_max_abs=0., amplitude=amplitude, attention={},
                        loss_chain_gradients={probe: dict(regularization_included=False, pooled_main_effects_included=False,
                            parameter_gradients=[], input_gradients={kind: dict(gradient_L2=0., connected=False)
                                for kind in ('RNA_input', 'patch_input')}) for probe in ('interaction', 'OtoP_drop_one', 'PtoO_drop_one')})
                    (input_root/(name+'.json')).write_text(json.dumps(row))
                    inventory.append({**row['identity'], 'final_epoch': 80, 'artifact_verified': True, 'row_partition_verified': True})
        (input_root/'INVENTORY.json').write_text(json.dumps(dict(status='COMPLETE_REQUESTED_MECHANISM_REPLAYS',
            inventory=inventory, completed_mechanism_cells=len(inventory), inventoried_cells=len(inventory),
            optimizer_updates=0, actual_weights_changed=False, audit_sources=audit_sources,
            F7_runtime_repair=input_root == inputs[1], explicit_runtime_backend=runtime['backend'] if runtime else None)))
    return root, private, config, inputs


def test_merge_separates_backends_and_requires_all_66_before_complete(tmp_path):
    root, private, config, inputs = fixture_reports(tmp_path)
    report, sources = merge_reports(inputs, private, config_path=config, root=root)
    assert report['status'] == 'COMPLETE_VERIFIED_66_MECHANISM_REPLAYS'
    assert report['completed_mechanism_cells'] == 66 and len(report['per_arm']) == 11
    assert len(report['per_arm_seed']) == 22 and len(report['per_arm_partition']) == 33
    assert sorted(row['cells'] for row in report['runtime_groups']) == [6, 60]
    partial, _ = merge_reports(inputs[:1], private, config_path=config, root=root)
    assert partial['status'] == 'PARTIAL_MISSING_MECHANISM_REPLAYS' and len(partial['missing_cells']) == 6
    output = tmp_path/'public'/'summary.json'
    combined = tmp_path/'private_combined'
    copy_private_reports(combined, sources, input_roots=inputs, public_output=output, script_sha=report['merge_script_sha256'])
    assert len(list(combined.glob('R3_*.json'))) == 66
    assert all(sha256(combined/path.name) == row['replay_json_sha256'] for path, row in sources)
    assert str(private) not in json.dumps(report)


def test_merge_rejects_duplicate_identity_and_source_or_artifact_drift(tmp_path):
    root, private, config, inputs = fixture_reports(tmp_path)
    with pytest.raises(ValueError, match='DUPLICATE_MECHANISM_IDENTITY'):
        merge_reports([inputs[0], inputs[0]], private, config_path=config, root=root)
    path = private/'cells'/'R3_outer1_inner0_F2_seed31'/'model.pt'
    path.write_bytes(b'changed artifact')
    with pytest.raises(ValueError, match='ARTIFACT_SHA_DIFFERS'):
        merge_reports(inputs, private, config_path=config, root=root)


def test_private_copies_cannot_be_written_into_public_output_directory(tmp_path):
    with pytest.raises(ValueError, match='MUST_BE_PRIVATE'):
        copy_private_reports(tmp_path/'public'/'copies', [], input_roots=[tmp_path/'input'],
                             public_output=tmp_path/'public'/'summary.json', script_sha='fixed')
