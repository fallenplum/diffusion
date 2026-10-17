"""Real aggregate-only diagnostics on label-free synthetic model/data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase20_selective import diagnostics, train
from src.phase20_selective.model import SelectiveModel


def synthetic():
    rng = np.random.default_rng(202020)
    n, g = 10, 128
    membership = np.zeros((8,g),bool)
    for k in range(8):
        membership[k,k*16:(k+1)*16] = True
    f = {key:rng.normal(size=(n,d)).astype(np.float32)
         for key,d in [('C',3),('R',g),('G',g),('L',g)]}
    f['RM'] = np.ones((n,g),np.float32)
    f['CM'] = np.ones((n,g),np.float32)
    f['RM'][1,:16] = 0
    f['CM'][2,16:24] = 0
    f['X'] = np.concatenate([f['C'],f['R']],1)
    weight = np.linspace(-.03,.03,f['X'].shape[1])
    f['ANCHOR'] = np.asarray(f['X'].astype(np.float64)@weight,np.float32)
    f['membership'] = membership
    owner = np.array([0,0,1,2,3,4,5,6,7,8,9])
    visual = torch.as_tensor(rng.normal(size=(len(owner),4,1536)),dtype=torch.float32)
    mask = torch.ones((len(owner),4),dtype=torch.bool)
    mask[0,2:] = False
    fit, valid = np.arange(8), np.array([8,9])
    wrong = np.concatenate([np.roll(fit,1),valid[::-1]])
    ds = SimpleNamespace(f=f,ft={k:torch.as_tensor(v) for k,v in f.items()},
        owner=owner,visual=visual,mask=mask,fit=fit,valid=valid,
        wrong_c=wrong.copy(),wrong_o=wrong.copy(),anchor_w=weight,
        anchor_sd_fit=float(np.std(f['X'][fit].astype(np.float64)@weight)),
        gradient_probe_indices=diagnostics.gradient_probe_indices(fit),
        blocks={'C':(0,3),'R':(3,3+g)},audit={'source':'label-free-synthetic'})
    def check(indices):
        value = np.asarray(indices,int)
        if len(np.unique(value)) != len(value) or not np.isin(value,np.arange(n)).all():
            raise ValueError('invalid synthetic indices')
        return value
    ds.check_indices = check
    assert not hasattr(ds,'ep')
    return ds


def config():
    path = Path(__file__).resolve().parents[1]/'configs/phase20_selective_attention.json'
    result = json.loads(path.read_text(encoding='utf8'))
    result['training']['dropout'] = 0.
    return result


def model_for(ds, arm):
    torch.manual_seed(2020)
    return SelectiveModel(3,ds.f['membership'],ds.anchor_w,
        ds.f['C'][ds.fit].mean(0),ds.anchor_sd_fit,arm,config())


def test_hierarchical_selection_is_patient_equal_not_query_pooled():
    gate = np.zeros((3,2,4),np.float32)
    gate[0,0,0] = 1; gate[0,1,:3] = 1
    gate[1,0,:] = 1
    gate[2,:,0:2] = 1
    patch = np.array([[1,1],[1,0],[1,1]],bool)
    omics = np.ones((3,4),bool)
    eligible = patch[:,:,None]&omics[:,None,:]
    diag = {'patch_mask':patch,'omics_mask':omics,
        'correction_eligible_mask':eligible,'gate_score':np.zeros_like(gate),
        'gate_p_to_o':gate,'gate_o_to_p':gate.transpose(0,2,1)}
    stats = diagnostics._Stats()
    diagnostics._selection_summary(stats,diag,np.array([0,0,1]),np.array([0,1]))
    out = stats.output()
    assert out['po_selectivity_support_patient_mean'] == pytest.approx(.625)
    assert out['po_selectivity_support_query_mean'] == pytest.approx(.6)
    assert out['po_selectivity_support_patient_count'] == 2
    assert out['po_selectivity_support_WSI_count'] == 3
    assert out['po_selectivity_support_query_count'] == 5
    assert out['po_selectivity_raw_score_degenerate_patient_mean'] == 1.


def test_selection_counts_low_candidate_and_zero_score_rows():
    gate = np.zeros((1,3,3),np.float32)
    eligible = np.array([[[1,1,1],[1,0,0],[0,0,0]]],bool)
    diag = {'patch_mask':np.ones((1,3),bool),'omics_mask':np.ones((1,3),bool),
        'correction_eligible_mask':eligible,'gate_score':np.zeros_like(gate),
        'gate_p_to_o':gate,'gate_o_to_p':gate.transpose(0,2,1)}
    stats = diagnostics._Stats()
    diagnostics._selection_summary(stats,diag,np.array([0]),np.array([0]))
    out = stats.output()
    assert out['po_selectivity_support_query_count'] == 1
    assert out['po_selectivity_all_rows_support_query_count'] == 3
    assert out['po_selectivity_eligible_lt2_patient_mean'] == pytest.approx(2/3)
    assert out['po_selectivity_raw_score_degenerate_patient_mean'] == 1.


def test_real_inspection_reads_no_labels_and_preserves_state_and_gradients(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    ds = synthetic(); model = model_for(ds,'S16')
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter,.125)
    state = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    before_features = {k:v.clone() for k,v in ds.ft.items()}
    out = diagnostics.inspect_model(model,ds)
    assert model.training
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),gradients))
    assert all(torch.equal(v,ds.ft[k]) for k,v in before_features.items())
    assert out['survival_labels_read'] is False
    assert out['raw_maps_persisted'] is False and out['new_probe_or_teacher_fits'] == 0
    assert out['fit_probe_patients'] == 8 and out['fit_probe_WSIs'] == 9
    assert out['isolated_gradient_query_patients'] == 8
    assert out['isolated_gradient_query_WSIs'] == 9
    assert out['isolated_correction_clinical_gradient_L2'] > 0
    assert out['isolated_correction_RNA_gradient_L2'] > 0
    assert out['constantC_off_max_absolute_error'] == 0
    assert out['pair_branch_cap_violation'] == 0
    assert out['correction_branch_cap_violation'] == 0
    assert out['RNA_measured_filter_rank_before_max'] == 16
    assert out['RNA_measured_response_map_rank_after_max'] == 16
    assert out['RNA_anchor_beta_projection_available'] is True
    for direction in ['po','op']:
        assert out[direction+'_gate_mass_error_max'] < 1e-5
        assert out[direction+'_TV_bound_violation_max'] < 1e-6
        assert out[direction+'_unselected_probability_abs_error_max'] == 0
        assert out[direction+'_selectivity_support_patient_count'] == 8
    assert all(np.isfinite(v) for v in out.values() if isinstance(v,(int,float)))
    assert all(not isinstance(v,(list,np.ndarray,torch.Tensor,dict)) for v in out.values())


@pytest.mark.parametrize('arm', ['P1','P16','L0'])
def test_no_correction_is_unavailable_not_a_claim_of_broken_gradient(monkeypatch,arm):
    monkeypatch.setattr(train,'cfg',config)
    ds = synthetic(); model = model_for(ds,arm)
    out = diagnostics.inspect_model(model,ds)
    assert out['isolated_correction_input_gradient_available'] is False
    assert out['isolated_correction_clinical_gradient_L2'] == 0
    assert out['isolated_correction_RNA_gradient_L2'] == 0
    assert out['correction_risk_SD'] == 0
    assert out['matched_wrongC_risk_MAE'] == 0
    assert out['matched_wrongO_risk_MAE'] == 0
    assert out['gate_budget_applicable'] is False
    if arm == 'P1':
        assert out['RNA_measured_response_map_rank_after_max'] == 1
        assert out['summary16_empirical_after_before_energy_ratio'] != pytest.approx(1.,abs=1e-4)
    if arm in ['P1','P16']:
        assert out['B_fullOmicsWrong_risk_absolute_change_mean'] > 0
        assert out['B_uniformPair_risk_absolute_change_mean'] > 0
    assert not any('patient_indices' in k or 'raw_risk' in k for k in out)


def test_covariance_inspection_never_mutates_input_views():
    original = np.arange(48,dtype=float).reshape(4,3,4)
    saved = original.copy()
    result = diagnostics._covariance_stats(original[:,1:])
    assert np.array_equal(original,saved)
    assert result['covariance_numerical_rank'] == 1
    assert result['effective_rank'] == pytest.approx(1.)
