"""Finite synthetic controls for R0/T2; no patient or outcome cache is read."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts import semantic_b0b7_p1r_diagnostics as d
from src.semantic_b0b7.models import ImageInputs, OmicsInputs, ModelConfig, SemanticRepresentation


def synthetic():
    torch.manual_seed(19)
    n, patches = 5, 7
    config = ModelConfig(patch_dim=6, conch_text_dim=6, titan_native_dim=6,
                         titan_text_dim=6, titan_score_dim=3, clinical_dim=2,
                         programs=3, slots=2, width=4, heads=2, rank=2, dropout=0.)
    cp, tp = torch.randn(2,6), torch.randn(2,6)
    prior = torch.tensor([[1.,0.,0.],[0.,.5,0.]])
    model = SemanticRepresentation('B3', cp, tp, prior, config=config, seed=31).eval()
    image = ImageInputs(patch_values=torch.randn(n,patches,6), patch_aligned=torch.randn(n,patches,6),
                        patch_mask=torch.ones(n,patches,dtype=torch.bool),
                        patch_slide=torch.zeros(n,patches,dtype=torch.long),
                        titan_native=torch.randn(n,6), titan_aligned=torch.randn(n,6),
                        titan_scores=torch.randn(n,3), clinical=torch.randn(n,2))
    measured = torch.ones(n,3,3,dtype=torch.bool); measured[0,2] = False
    genes = OmicsInputs(torch.randn(n,3,3), measured)
    return model,image,genes,cp,tp


def test_capacity_control_uses_mean_information_only_and_keeps_clinical():
    rng=np.random.default_rng(9)
    slots=rng.normal(size=(12,8,64));clinical=rng.normal(size=(12,64))
    changed=slots.copy();changed[:,0]+=1.;changed[:,1]-=1.
    _,a,state=d.feature_pair('T2_capacity_control',slots,clinical,slots,clinical)
    _,b,_=d.feature_pair('T2_capacity_control',slots,clinical,changed,clinical)
    assert a.shape==(12,576) and np.allclose(a,b,rtol=0,atol=1e-12)
    assert np.array_equal(a[:,-64:],clinical)
    _,a,_=d.feature_pair('T2_keep_slots',slots,clinical,slots,clinical)
    _,b,_=d.feature_pair('T2_keep_slots',slots,clinical,changed,clinical)
    assert not np.array_equal(a,b)


def test_polynomial_statistics_do_not_fit_to_validation_values():
    rng=np.random.default_rng(31)
    fit=rng.normal(size=(15,8,64));valid=rng.normal(size=(6,8,64));a=rng.normal(size=(15,64));b=rng.normal(size=(6,64))
    train1,_,s1=d.feature_pair('T2_capacity_control',fit,a,valid,b)
    train2,_,s2=d.feature_pair('T2_capacity_control',fit,a,valid+1e5,b-1e4)
    assert np.array_equal(train1,train2)
    assert np.array_equal(s1['mean64_center'],s2['mean64_center'])
    assert np.array_equal(s1['mean64_scale'],s2['mean64_scale'])


def test_cross_products_fixed_unique_and_no_square_duplicates():
    pairs=d.cross_pairs()
    assert pairs.shape==(384,2) and np.all(pairs[:,0]<pairs[:,1])
    assert len(set(map(tuple,pairs)))==384 and np.array_equal(pairs,d.cross_pairs())


def test_alpha_selection_has_locked_stronger_ridge_tie_break():
    assert d.alpha_choice({.01:1.,.1:1.+1e-13,1.:2.})==.1
    assert d.alpha_choice({.01:1.,.1:1.+2e-12,1.:2.})==.01
    with pytest.raises(ValueError,match='INVALID_CV_ALPHA_METRIC'):
        d.alpha_choice({1.:float('nan')})


def test_known_unknown_decomposition_respects_actual_pair_mask():
    model,image,genes,*_=synthetic()
    out=model(image,genes)
    with torch.no_grad():r=d.relation_diagnostics(model,out)
    assert r['decomposition_max_abs_error']<2e-6
    assert r['known_edges']==2 and r['unknown_edges']==4
    assert r['per_program'][2]['known_edges']==0
    assert r['per_program'][2]['measured_patients']==4
    assert r['normalized_known_mass']['mean']+r['normalized_unknown_mass']['mean']==pytest.approx(1.)


def test_backward_norms_do_not_fabricate_disconnected_program_omics_gradient():
    model,image,genes,*_=synthetic();out=model(image,genes)
    before=d.state_digest(model)
    r=d.gradient_diagnostics(model,out,torch.randn(5,3),torch.ones(5,3,dtype=torch.bool),
                             ~torch.eye(5,dtype=torch.bool),None)
    assert r['parameter_groups']['gene_encoder.weight']['program_connected'] is False
    assert r['parameter_groups']['gene_encoder.weight']['pair_connected'] is True
    assert r['parameter_groups']['u.weight']['program_norm']==0.
    assert r['parameter_groups']['u.weight']['pair_norm']>0.
    assert r['backward_calls']==2 and d.state_digest(model)==before


def test_validation_RNA_permutation_cannot_change_image_prediction():
    model,image,genes,*_=synthetic()
    with torch.no_grad():
        out=model(image,genes)
        r=d.frozen_route_interventions(model,(image,genes,torch.randn(5,3),torch.ones(5,3,dtype=torch.bool)),
                                      out,image.titan_scores[:3].mean(0,keepdim=True))
    assert r['RNA_permutation_prediction_bytes_identical']
    assert set(r['B3_score_branch'])=={'zero','training_mean'}
    assert r['B3_score_branch']['zero']['prediction_change']['std']>0.


def test_image_score_native_and_clinical_routes_exactly_reconstruct():
    model,image,genes,cp,tp=synthetic()
    with torch.no_grad():r=d.text_route(model,image,model(image,genes),{'conch_prototypes':cp,'titan_prototypes':tp})
    assert r['image_route_decomposition_max_abs_error']<2e-6
    assert r['program_head_decomposition_max_abs_error']<2e-6
    assert r['conch_same_Gram_max_abs_error']<1e-6


def test_saved_ridge_replay_and_unobserved_target_values_do_not_change_fit():
    rng=np.random.default_rng(7);x=rng.normal(size=(14,9));z=rng.normal(size=(5,9));y=rng.normal(size=(14,3))
    mask=np.ones((14,3),bool);mask[:3,0]=False;changed=y.copy();changed[~mask]=1e12
    pred,state=d.original.fit_masked_ridge(x,z,y,mask,alpha=1.)
    alt,_=d.original.fit_masked_ridge(x,z,changed,mask,alpha=1.)
    assert np.array_equal(pred,alt)
    assert np.allclose(d.ridge_predict(state,z),pred,rtol=0,atol=1e-12)


def test_constant_representation_rank_has_explicit_zero():
    r=d.rank_description(np.ones((9,12)))
    assert r['numerical_rank']==0 and r['energy_entropy_effective_rank']==0.
