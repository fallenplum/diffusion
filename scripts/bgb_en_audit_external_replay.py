"""Independent read-only raw-v1/unitnorm-v2 checkpoint and annotation replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def natural(donor, section):
    weight = np.zeros(len(donor), float)
    identities = sorted(set(donor))
    for identity in identities:
        slides = sorted(set(section[donor == identity]))
        for slide in slides:
            rows = np.flatnonzero((donor == identity) & (section == slide))
            weight[rows] = 1/(len(identities)*len(slides)*len(rows))
    return weight


def forward(x, state):
    x = torch.as_tensor(np.array(x, copy=True))
    h = F.linear(x, state['project.0.weight'], state['project.0.bias'])
    h = F.layer_norm(h, (64,), state['project.1.weight'], state['project.1.bias'], eps=1e-5)
    h = F.gelu(h, approximate='none')
    return (F.linear(h, state['program.weight'], state['program.bias']).numpy(),
            F.linear(h, state['tissue.weight'], state['tissue.bias']).numpy())


def metric(y, pred, tissue, logits, donor, section, mean, scale):
    target = (y-mean)/scale
    weight = natural(donor, section)
    error = np.abs(pred-target)
    huber = np.where(error <= 1, .5*error**2, error-.5)
    recalls = [float((logits.argmax(-1)[tissue == k] == k).mean()) for k in range(3)]
    return dict(program_standardized_Huber=float(np.dot(weight, huber.mean(-1))),
                program_standardized_MSE=float(np.dot(weight, ((pred-target)**2).mean(-1))),
                tissue_balanced_accuracy=float(np.mean(recalls)), tissue_recall=recalls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--stage-a-root', type=Path, required=True)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--unitnorm-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.source_root/'manifest.json').read_text())
    data_sha = sha(args.source_root/'prepared.npz')
    assert data_sha == manifest['data_sha256']
    with np.load(args.source_root/'prepared.npz', allow_pickle=False) as f:
        data = {key: np.array(f[key]) for key in ('X', 'program_y', 'tissue_y', 'donor', 'section', 'barcodes')}
    raw_x = data['X']
    norm = np.linalg.norm(raw_x, axis=1)
    unit_x = raw_x/norm[:, None]
    am = json.loads((args.stage_a_root/'manifest.json').read_text())
    assert sha(args.stage_a_root/'prepared.npz') == manifest['stage_a_data_sha256'] == am['data_sha256']
    assert sha(args.stage_a_root/'annotation_mapping.npz') == am['annotation_mapping_sha256']
    with np.load(args.stage_a_root/'prepared.npz', allow_pickle=False) as f:
        assert np.array_equal(f['y'], data['tissue_y'])
        for key in ('donor', 'section', 'barcodes'):
            assert np.array_equal(f[key], data[key])
    with np.load(args.stage_a_root/'annotation_mapping.npz', allow_pickle=False) as f:
        mapped = np.array([am['classes'].index(am['label_mapping'][v]) if am['label_mapping'][v] in am['classes'] else -1
                           for v in f['raw_labels']], np.int64)
        assert np.array_equal(mapped, f['y'])
        assert np.array_equal(mapped[mapped >= 0], data['tissue_y'])
    annotation = dict(status='PASS_ELEMENTWISE_INDEPENDENT_TISSUE_LABEL_LINEAGE', teacher_proxy=False,
        source='author-released Pathology_SpotAnnotations.zip; independent pathologist tissue annotation',
        source_url='https://zenodo.org/records/7760264', annotation_archive_sha256=am['annotation_archive_sha256'],
        mapping_sha256=am['mapping_sha256'], primary_labels_replayed=13011,
        source_classes=am['classes'], tissue_counts=np.bincount(data['tissue_y']).tolist(),
        evidence_boundary='coarsened spot tissue classes; not cell fractions, pure cell sources or clinical validation')
    completed = []
    donors = np.unique(data['donor'])
    for version, directory, x in [('raw_v1_not_transferred', args.raw_run, raw_x),
                                  ('unitnorm_v2_eligible_for_transfer', args.unitnorm_run, unit_x)]:
        for index in range(7):
            folder = directory/f'fit_{index:02d}'
            receipt = json.loads((folder/'COMPLETE.json').read_text())
            assert receipt['status'] == 'COMPLETE_FIXED_SOURCE_FIT'
            assert receipt['epochs'] == 20 and receipt['optimizer_updates'] == 20
            for name, digest in receipt['artifacts'].items():
                assert sha(folder/name) == digest
            state = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
            assert state['contract']['lineage']['source_data_sha256'] == data_sha
            if version.startswith('unitnorm'):
                assert state['schema'] == 'bgb_en_external_source_checkpoint_v2_unitnorm'
                assert state['input_normalization'].startswith('float32_per_row_L2')
            with np.load(folder/'PRIVATE_PARTITION.npz', allow_pickle=False) as f:
                train, valid = np.array(f['train']), np.array(f['valid'])
                expected = np.arange(len(x)) if index == 6 else np.flatnonzero(data['donor'] != donors[index])
                held = np.empty(0, int) if index == 6 else np.flatnonzero(data['donor'] == donors[index])
                assert np.array_equal(train, expected) and np.array_equal(valid, held)
                weight = natural(data['donor'][train], data['section'][train])
                mean = np.sum(data['program_y'][train]*weight[:, None], axis=0)
                scale = np.sqrt(np.sum((data['program_y'][train]-mean)**2*weight[:, None], axis=0))
                mean_error = float(np.abs(mean-state['target_mean'].numpy()).max())
                scale_error = float(np.abs(scale-state['target_scale'].numpy()).max())
                assert mean_error < 1e-12 and scale_error < 1e-12
                mass = np.bincount(data['tissue_y'][train], weights=weight, minlength=3)
                class_error = float(np.abs(1/(3*mass)-state['class_weights'].numpy()).max())
                assert class_error < 1e-12
                with torch.no_grad():
                    pred, logits = forward(x, state['state_dict'])
                replay = metric(data['program_y'][train], pred[train], data['tissue_y'][train], logits[train],
                                data['donor'][train], data['section'][train], mean, scale)
                metrics_error = max(abs(replay[key]-receipt['train_metrics'][key])
                                    for key in ('program_standardized_Huber', 'program_standardized_MSE', 'tissue_balanced_accuracy'))
                prediction_error = None
                valid_metrics_error = None
                if len(valid):
                    prediction_error = max(float(np.abs(pred[valid]-f['program_standardized']).max()),
                                           float(np.abs(logits[valid]-f['tissue_logits']).max()))
                    valid_replay = metric(data['program_y'][valid], pred[valid], data['tissue_y'][valid], logits[valid],
                                          data['donor'][valid], data['section'][valid], mean, scale)
                    valid_metrics_error = max(abs(valid_replay[key]-receipt['valid_metrics'][key])
                                              for key in ('program_standardized_Huber', 'program_standardized_MSE', 'tissue_balanced_accuracy'))
                    assert prediction_error < 3e-5 and valid_metrics_error < 2e-6
                assert metrics_error < 2e-6
            completed.append(dict(version=version, fit_index=index, epochs=20, optimizer_updates=20,
                source_train_rows=len(train), source_heldout_rows=len(valid),
                target_mean_replay_max_abs=mean_error, target_scale_replay_max_abs=scale_error,
                class_weight_replay_max_abs=class_error, train_metric_replay_max_abs=metrics_error,
                heldout_prediction_replay_max_abs=prediction_error, heldout_metric_replay_max_abs=valid_metrics_error))
    controls = []
    for directory in args.control_run:
        folder = directory/'fit_06'
        receipt = json.loads((folder/'COMPLETE.json').read_text())
        assert receipt['status'] == 'COMPLETE_NEGATIVE_SOURCE_FIT' and receipt['epochs'] == 20 and receipt['optimizer_updates'] == 20
        for name, digest in receipt['artifacts'].items():
            assert sha(folder/name) == digest
        state = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        control = state['lineage']['source_control']
        assert state['lineage']['source_data_sha256'] == data_sha
        with np.load(folder/'PRIVATE_TARGET_MAP.npz', allow_pickle=False) as f:
            mapping = np.array(f['target_map'])
        assert np.array_equal(data['tissue_y'][mapping], data['tissue_y'])
        if control == 'WRONG_DONOR':
            assert np.all(data['donor'][mapping] != data['donor'])
        else:
            assert control == 'SPATIAL_SHUFFLE'
            assert np.array_equal(np.sort(mapping), np.arange(len(mapping)))
            assert np.array_equal(data['donor'][mapping], data['donor'])
            assert np.array_equal(data['section'][mapping], data['section'])
        weight = natural(data['donor'], data['section'])
        mean = np.sum(data['program_y']*weight[:, None], axis=0)
        scale = np.sqrt(np.sum((data['program_y']-mean)**2*weight[:, None], axis=0))
        assert np.array_equal(mean, state['target_mean'].numpy()) and np.array_equal(scale, state['target_scale'].numpy())
        with torch.no_grad():
            pred, logits = forward(unit_x, state['state_dict'])
        true_metrics = metric(data['program_y'], pred, data['tissue_y'], logits, data['donor'], data['section'], mean, scale)
        rematched_metrics = metric(data['program_y'][mapping], pred, data['tissue_y'], logits, data['donor'], data['section'], mean, scale)
        true_error = max(abs(true_metrics[key]-receipt['original_true_target_training_metrics'][key])
                         for key in ('program_standardized_Huber', 'program_standardized_MSE', 'tissue_balanced_accuracy'))
        rematched_error = max(abs(rematched_metrics[key]-receipt['rematched_training_target_metrics'][key])
                              for key in ('program_standardized_Huber', 'program_standardized_MSE', 'tissue_balanced_accuracy'))
        assert max(true_error, rematched_error) < 2e-6
        controls.append(dict(control=control, epochs=20, optimizer_updates=20,
            true_source_target_scaler_elementwise_equal=True, tissue_target_unchanged=True,
            true_target_training_metric_replay_max_abs=true_error, rematched_target_training_metric_replay_max_abs=rematched_error,
            target_map_same_tissue_all_rows=True, true_target_evaluation_is_heldout=False))
    result = dict(status='PASS_EXTERNAL_CHECKPOINT_REPLAY', audit_optimizer_updates=0, audit_script_sha256=sha(__file__),
        actual_recorded_source_fits=14+len(controls), actual_recorded_source_optimizer_updates=280+20*len(controls),
        raw_v1_transfer_eligible_under_v2_contract=False,
        migration_audit_boundary='input/weight eligibility; no TCGA auxiliary checkpoint lineage inspected here',
        source_data_sha256=data_sha,
        raw_source_median_L2=float(np.median(norm)), normalized_source_median_L2=float(np.median(np.linalg.norm(unit_x, axis=1))),
        annotation=annotation, fits=completed, negative_source_controls=controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
