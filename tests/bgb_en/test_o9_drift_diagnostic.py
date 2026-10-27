import numpy as np
import torch
import json
import sys
from pathlib import Path
from scripts.bgb_en_o9_drift_diagnostic import numerical_error, pair_changes, state_hash


def test_risk_error_below_two_micro_units_can_still_change_pair_credit():
    time, event = np.array([1., 2., 3.]), np.ones(3, int)
    saved, current = np.array([0., 1e-7, -1.]), np.array([2e-7, 1e-7, -1.])
    assert numerical_error(saved, current)['within_original_2e_minus6_gate']
    assert pair_changes(time, event, saved, current)['changed_pair_credits'] == 1
    assert pair_changes(time, event, saved, current)['net_credit_change'] == 1.


def test_error_and_state_checks_expose_risk_drift_and_buffer_mutation():
    assert numerical_error(np.zeros(3), np.full(3, 3e-6))['within_original_2e_minus6_gate'] is False
    module = torch.nn.Linear(2, 1)
    module.register_buffer('edge_gene', torch.tensor([1, 2, 3]))
    before = state_hash(module)
    assert state_hash(module) == before
    module.edge_gene[0] = 5
    assert state_hash(module) != before


def test_actual_three_cell_driver_binds_frozen_R2_gamma_and_replays_all_five(tmp_path, monkeypatch):
    from scripts import bgb_en_o9_drift_diagnostic as diagnostic
    from scripts import bgb_en_run as driver
    from src.bgb_en import data, training
    torch.set_num_threads(2)
    config = json.loads(Path('configs/bgb_en/prototype_v1.json').read_text())
    assert config['R2']['gamma'] == .3
    rng = np.random.default_rng(801)
    n, genes = 8, 12
    features = dict(rna=rng.normal(size=(n, genes)).astype(np.float32), rna_mask=np.ones((n, genes), bool),
        cna_gain=np.zeros((n, genes), np.float32), cna_loss=np.zeros((n, genes), np.float32),
        cna_mask=np.ones((n, genes), bool), C=np.zeros((n, 2), np.float32))
    prepared = dict(features=features, train=np.arange(4), valid=np.arange(4, 8), base=np.zeros(n),
        RNA=np.linspace(-.5, .5, n), CS=np.zeros(n), beta=1., lineage={'synthetic': 'fixed'})
    member = torch.arange(genes)[None, :]//4 == torch.arange(3)[:, None]
    raw = dict(t=np.array([2., 20., 30., 40., 3., 5., 8., 12.]), e=np.ones(n, int))
    private = tmp_path/'formal'
    for part in (1, 2):
        path = private/'prepared'/f'outer1_inner{part}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prepared, path)
        path.with_suffix('.json').write_text(json.dumps(dict(lineage=prepared['lineage'], sha256=driver.sha256(path))))
    for part, seed, _ in diagnostic.CELLS:
        folder = private/'cells'/f'R2_outer1_inner{part}_O9_seed{seed}'; folder.mkdir(parents=True)
        torch.manual_seed(seed)
        model = training.MechanismModel('R2', 'O9', prepared, member, None, 'cpu', gamma=config['R2']['gamma']).eval()
        with torch.no_grad():
            model.omics.risk_head.weight.zero_(); model.omics.risk_head.weight[0, 0] = 1.
        risk, _ = training.predict(model, prepared['valid'], batch_size=8)
        wrong = training.MechanismModel('R2', 'O9', prepared, member, None, 'cpu', gamma=.1).eval()
        wrong.load_state_dict(model.state_dict(), strict=True)
        wrong_risk, _ = training.predict(wrong, prepared['valid'], batch_size=8)
        assert np.max(np.abs(risk-wrong_risk)) > 2e-6  # This fixture detects the original omission.
        torch.save(dict(train=prepared['train'], valid=prepared['valid'], lineage=prepared['lineage'],
                        snapshots={80: dict(state_dict=model.state_dict(), risk=risk)}), folder/'model.pt')
        (folder/'COMPLETE.json').write_text(json.dumps(dict(lineage=prepared['lineage'],
                                                           artifact_sha256=driver.sha256(folder/'model.pt'))))
    monkeypatch.setattr(driver, 'load_inputs', lambda _args: ({'patch_regions': 'synthetic'}, config, raw, member, prepared['lineage']))
    monkeypatch.setattr(data, 'read_patch_cache', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(training, 'PatchBank', lambda *_args: None)
    output = tmp_path/'diagnostic.json'
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--assets', 'unused.json', '--private', str(private),
                                    '--backend', 'inherited', '--device', 'cpu', '--output', str(output)])
    diagnostic.main()
    report = json.loads(output.read_text())
    assert len(report['cells']) == 3
    for row in report['cells']:
        assert row['gamma'] == config['R2']['gamma'] and row['gamma_matches_frozen_config']
        assert len(row['repeats']) == 5 and len(row['cross_repeat_comparisons']) == 10
        assert all(value['status'] == 'FORWARD_COMPLETE' and value['max_abs_error'] == 0 for value in row['repeats'])
        assert row['weights_bitwise_unchanged']
