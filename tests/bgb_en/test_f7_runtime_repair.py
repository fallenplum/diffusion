import copy
import json
from pathlib import Path
import pytest
from scripts.bgb_en_f7_runtime_repair import original_inventory, validate_forward_qualification
from scripts.bgb_en_f7_determinism_diagnostic import sha


def test_original_inventory_requires_all_six_and_preserves_three_failed_attempts(tmp_path):
    lock = json.loads(Path('configs/bgb_en/F7_deterministic_runtime_v1.json').read_text())
    for partition in lock['inner_partitions']:
        for seed in lock['seeds']:
            name = f'R3_outer1_inner{partition}_F7_seed{seed}'
            folder = tmp_path/name; folder.mkdir()
            if name in lock['expected_original_failed']:
                row = dict(lineage={'test': 'fixed'}, type='RuntimeError', message='Gradient-cache forward was not deterministic')
                (folder/'FAILED.json').write_text(json.dumps(row))
            else:
                (folder/'model.pt').write_bytes(b'fixed synthetic artifact, not torch weights')
                row = dict(lineage={'test': 'fixed'}, arm='F7', epochs=80, artifact_sha256=sha(folder/'model.pt'))
                (folder/'COMPLETE.json').write_text(json.dumps(row))
    first = original_inventory(tmp_path, lock)
    assert len(first) == 6 and sum(row['original_status'] == 'FAILED' for row in first) == 3
    assert original_inventory(tmp_path, lock) == first
    (tmp_path/lock['expected_original_failed'][0]/'COMPLETE.json').write_text('{}')
    with pytest.raises(ValueError, match='success/failure inventory'):
        original_inventory(tmp_path, lock)


def test_cpu_qualification_or_relaxed_guard_cannot_enable_cuda_retraining():
    value = dict(status='PASS_SAMPLED_READONLY_FORWARD', optimizer_updates=0, arm='F7', partition=1, seed=47,
        device='cuda', state='initial', core_lineage={'test': 'fixed'}, threshold=2e-5, all_comparisons_within_threshold=True,
        diagnostic_sources={'scripts/bgb_en_f7_determinism_diagnostic.py': 'fixed-sha'},
        backend=dict(mode='deterministic', deterministic_algorithms=True, deterministic_warn_only=False,
                     CUBLAS_WORKSPACE_CONFIG=':4096:8', matmul_allow_tf32=False, cudnn_allow_tf32=False))
    validate_forward_qualification(value, 1, 47, {'test': 'fixed'}, 'fixed-sha')
    wrong = copy.deepcopy(value); wrong['device'] = 'cpu'
    with pytest.raises(ValueError, match='CPU does not qualify CUDA'):
        validate_forward_qualification(wrong, 1, 47, {'test': 'fixed'}, 'fixed-sha')
    wrong = copy.deepcopy(value); wrong['threshold'] = 1e-3
    with pytest.raises(ValueError, match='threshold differs'):
        validate_forward_qualification(wrong, 1, 47, {'test': 'fixed'}, 'fixed-sha')
