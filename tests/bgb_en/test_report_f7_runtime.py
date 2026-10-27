import copy
import importlib.util
import json
from pathlib import Path
import pytest
from scripts.bgb_en_report_f7_runtime import diagnostic_summary, build_report, markdown
from scripts.bgb_en_recovery_audit import certify_recovery, sha256


def diagnostic_batch(error=0., first=None):
    record = dict(first_non_bitwise_intermediate=first, risk_within_unchanged_gradient_cache_threshold=error <= 2e-5,
        intermediate=[dict(name='omics.rna:0', bitwise_equal=first is None, max_abs_error=error),
                      dict(name='risk', bitwise_equal=error == 0, max_abs_error=error),
                      dict(name='selected_message', bitwise_equal=error == 0, max_abs_error=error)],
        topk=dict(score_max_abs_error=error, changed_support_query_rows=int(error > 0),
            symmetric_difference_selected_keys=2*int(error > 0), eligible_query_rows=2,
            attention_TV_mean=error, attention_TV_max=2*error))
    boundary = dict(rows_with_more_than_topk_keys=2, minimum_margin=.0001, mean_margin=.001, exact_boundary_ties=0)
    return dict(training_batch_ordinal=99999, boundary_no_grad=boundary, boundary_grad=boundary,
                comparisons={mode: [copy.deepcopy(record)] for mode in ('no_grad_repeat', 'grad_repeat', 'no_grad_vs_grad')})


def test_summary_keeps_actual_comparison_errors_and_does_not_export_private_rows():
    doc = dict(status='PASS_SAMPLED_READONLY_FORWARD', batches=[diagnostic_batch(1e-7, 'omics.rna:0')],
               private_path='DO_NOT_EXPORT_PRIVATE_PATH', patient_ids=['DO_NOT_EXPORT_PATIENT'])
    summary = diagnostic_summary(doc, include_comparisons=True)
    assert summary['comparisons'] == 3 and summary['comparisons_with_support_flip'] == 3
    assert summary['maximum_risk_error'] == 1e-7
    assert summary['first_non_bitwise_counts'] == {'omics.rna:0': 3}
    text = json.dumps(summary)
    assert 'DO_NOT_EXPORT' not in text and 'training_batch_ordinal' not in text
    empty = diagnostic_summary(dict(status='FAILED_DIAGNOSTIC_BACKEND_QUALIFICATION', batches=[]))
    assert empty['all_compared_tensors_bitwise_equal'] is None
    assert empty['all_risks_within_original_threshold'] is None


def test_summary_does_not_trust_a_forged_threshold_or_first_divergence_flag():
    doc = dict(status='PASS_SAMPLED_READONLY_FORWARD', batches=[diagnostic_batch(3e-5, 'omics.rna:0')])
    doc['batches'][0]['comparisons']['no_grad_repeat'][0]['risk_within_unchanged_gradient_cache_threshold'] = True
    with pytest.raises(ValueError, match='RISK_THRESHOLD_FIELD_DIFFERS'):
        diagnostic_summary(doc)


def test_full_report_reverifies_recovery_certificate_and_preserves_logged_guard_boundary(tmp_path):
    spec = importlib.util.spec_from_file_location('recovery_fixture_for_f7_report', Path(__file__).with_name('test_recovery_audit.py'))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    root, private, archive, manifest, config = fixture.fixture_files(tmp_path)
    (root/'scripts/bgb_en_report_f7_runtime.py').write_text('immutable synthetic reporter source')
    work = tmp_path/'work'; work.mkdir()
    origin = json.loads(manifest.read_text())
    for part in (0, 1, 2):
        for seed in (31, 47):
            name = f'R3_outer1_inner{part}_F7_seed{seed}'
            cell = private/'cells'/name
            complete = json.loads((cell/'COMPLETE.json').read_text())
            complete['curves'] = [dict(epoch=epoch, replay_max=0.) for epoch in (20, 40, 80)]
            (cell/'COMPLETE.json').write_text(json.dumps(complete))
            runtime = json.loads((cell/'RUNTIME_REPAIR_COMPLETE.json').read_text())
            old = next(row for row in origin['original_cells'] if row['cell'] == name)
            qual = dict(status='PASS_SAMPLED_READONLY_FORWARD', optimizer_updates=0, neural_fits=0, backpropagations=0,
                arm='F7', partition=part, seed=seed, threshold=2e-5, state='initial', device='cuda',
                core_lineage=complete['lineage'], prepared_sha256=sha256(private/'prepared'/f'outer1_inner{part}.pt'),
                original_receipt_sha256={key: value for key, value in old['files'].items() if key.endswith('.json')},
                diagnostic_sources={'scripts/bgb_en_f7_determinism_diagnostic.py': sha256(root/'scripts/bgb_en_f7_determinism_diagnostic.py')},
                backend=runtime['backend'], batches=[diagnostic_batch()],
                all_comparisons_within_threshold=True, all_comparisons_bitwise_equal=True)
            qual_path = work/f'CUDA_QUAL_p{part}_s{seed}.json'; qual_path.write_text(json.dumps(qual))
            runtime.update(core_receipt_sha256=sha256(cell/'COMPLETE.json'), qualification_sha256=sha256(qual_path))
            (cell/'RUNTIME_REPAIR_COMPLETE.json').write_text(json.dumps(runtime))
            (work/f'RUNTIME_TRAIN_p{part}_s{seed}.json').write_text(json.dumps(runtime))
            if part == 0 and seed == 31:
                original = copy.deepcopy(qual)
                original.update(state='final80', checkpoint_artifact_sha256=old['files']['model.pt'])
                original['backend'] = {**runtime['backend'], 'mode': 'inherited'}
                original['private_path'] = str(private)
                original['patient_ids'] = ['DO_NOT_EXPORT_PATIENT']
                (work/'ORIGINAL_FORWARD.json').write_text(json.dumps(original))
    receipt = certify_recovery(private, manifest, archive, config_path=config, root=root)
    recovery_path = work/'RECOVERY_RECEIPT.json'; recovery_path.write_text(json.dumps(receipt))
    (work/'QUEUE_RECEIPT.json').write_text(json.dumps(dict(status='SIX_FITS_COMPLETE_PENDING_INDEPENDENT_CERTIFICATION',
        actual_new_fits=6, actual_new_updates=480, teacher_refits=0, original_schedule_modified=False)))
    report = build_report(work, private, recovery_path, config_path=config, root=root)
    assert report['recovery']['actual_new_optimizer_updates'] == 480
    assert report['training_guard']['observed_logged_points'] == 18
    assert report['training_guard']['all_epoch_numerical_errors_available'] is False
    assert report['causal_interpretation']['exact_original_CUDA_cause_localized'] is False
    assert report['all_six_initial_qualifications_bitwise_equal']
    assert len(report['deterministic_initial_CUDA_qualifications']) == 6
    text = json.dumps(report)+markdown(report)
    assert 'DO_NOT_EXPORT' not in text and str(private) not in text and str(work) not in text
    # The report is not allowed to trust an edited COMPLETE certificate alone.
    (private/'cells'/'R3_outer1_inner0_F7_seed31'/'model.pt').write_bytes(b'changed after certificate')
    with pytest.raises(ValueError, match='ARTIFACT_SHA_DIFFERS'):
        build_report(work, private, recovery_path, config_path=config, root=root)
