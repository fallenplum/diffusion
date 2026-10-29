"""Synthetic contracts only; no real cohort, source weights or experiment fits."""
import hashlib

import numpy as np
import pytest
import torch

from src.bgb_en_r8.cox import efron_log_weights, efron_loss_gradient, fit_protected_offset_cox
from src.bgb_en_r8.evidence import FrozenWSIEvidence, decode_frozen_source_cache, pool_frozen_evidence
from src.bgb_en_r8.features import BlockPCA, MaskedPCA, build_low_capacity_features


FIT = hashlib.sha256(b"synthetic-train-identities-not-patients").hexdigest()


def evidence(seed=11):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(3, 3, 7, 64, generator=g)
    q = torch.randn(3, 3, 7, 4, 64, generator=g)
    s = torch.randn(3, 3, 7, 4, generator=g)
    m = torch.ones(3, 3, 7, dtype=torch.bool)
    m[0, 2] = False
    m[1, 1, 3:] = False
    m[2, 0, 1:] = False
    return FrozenWSIEvidence(h, q, s, m)


def read(e, **kwargs):
    return pool_frozen_evidence(e, source_prior_scale=.1, cosine_temperature=1., **kwargs)


def test_equal_slide_weighting_and_padding_not_patch_count_weighting():
    e = evidence()
    h, q, s, m = e.states.clone(), e.program_keys.clone(), e.program_scores.clone(), e.patch_mask.clone()
    h[~m] = q[~m] = s[~m] = torch.nan
    out = pool_frozen_evidence(FrozenWSIEvidence(h, q, s, m), source_prior_scale=0., cosine_temperature=1.)
    expected = []
    for b in range(3):
        means = [e.states[b, j, m[b, j]].mean(0) for j in range(3) if m[b, j].any()]
        expected.append(torch.stack(means).mean(0))
    assert torch.allclose(out["image_program_embeddings"], torch.stack(expected)[:, None].expand(-1, 4, -1), atol=2e-7)
    assert torch.equal(out["valid_slide_count"], torch.tensor([2, 3, 3]))
    assert torch.isfinite(out["attention"]).all() and torch.all(out["attention"][~m[:, :, None].expand(-1,-1,4,-1)] == 0)


def test_repeating_entire_slide_set_does_not_change_patient_evidence():
    e = evidence(); a = read(e)
    d = FrozenWSIEvidence(*(torch.cat((x, x), 1) for x in (e.states,e.program_keys,e.program_scores,e.patch_mask)))
    b = read(d)
    assert torch.allclose(a["image_program_scores"], b["image_program_scores"], atol=2e-7)
    assert torch.allclose(a["image_program_embeddings"], b["image_program_embeddings"], atol=2e-7)


def test_joint_patch_reindexing_is_readout_invariant():
    e = evidence(); p = torch.tensor([5, 0, 3, 2, 6, 1, 4]); a = read(e)
    d = FrozenWSIEvidence(e.states[:,:,p], e.program_keys[:,:,p], e.program_scores[:,:,p], e.patch_mask[:,:,p])
    b = read(d)
    assert torch.allclose(a["image_program_embeddings"], b["image_program_embeddings"], atol=3e-7)
    assert torch.allclose(a["attention"][:,:,:,p], b["attention"], atol=3e-7)


def test_query_changes_only_reads_and_missing_query_exactly_falls_back():
    e = evidence(); original = [x.clone() for x in (e.states,e.program_keys,e.program_scores,e.patch_mask)]
    q = torch.randn(3,4,64, generator=torch.Generator().manual_seed(72)); valid = torch.ones(3,4,dtype=torch.bool)
    a = read(e); b = read(e, query=q, query_mask=valid)
    assert (a["attention"]-b["attention"]).abs().max() > 1e-5
    for x, y in zip((e.states,e.program_keys,e.program_scores,e.patch_mask), original):
        assert torch.equal(x,y)
    missing = read(e, query=torch.full_like(q, torch.nan), query_mask=torch.zeros_like(valid))
    assert torch.equal(a["attention"], missing["attention"])
    assert torch.equal(a["image_program_embeddings"], missing["image_program_embeddings"])
    assert b["diagnostics"]["source_visual_forward_uses_RNA"] is False


def test_empty_patient_and_trainable_evidence_fail_closed():
    e = evidence(); m=e.patch_mask.clone();m[0]=False
    with pytest.raises(ValueError, match="NO_VALID_VISUAL"):
        read(FrozenWSIEvidence(e.states,e.program_keys,e.program_scores,m))
    with pytest.raises(ValueError, match="MUST_BE_FROZEN"):
        read(FrozenWSIEvidence(e.states.requires_grad_(True),e.program_keys,e.program_scores,e.patch_mask))


def test_missing_cache_keys_are_recovered_only_by_exact_frozen_source():
    from src.bgb_en.r6_source_model import SourceProgramModel, PROGRAMS
    from src.bgb_en.r6_pathology import source_model_state_sha256
    from src.bgb_en.r6_omics import SparseProgramEncoder, mapping_from_programs
    mapping=mapping_from_programs([dict(id=p,up_genes=["G"+str(k)],down_genes=[]) for k,p in enumerate(PROGRAMS)])
    source=SourceProgramModel("S7",init_seed=31,tissue_prior=torch.full((3,),1/3),
        gene_reader=SparseProgramEncoder(mapping)).requires_grad_(False).eval()
    x=torch.nn.functional.normalize(torch.randn(2*2*3,1536,generator=torch.Generator().manual_seed(92)),dim=-1)
    with torch.no_grad():
        out=source(x)
    cache=dict(states=out["embedding"].reshape(2,2,3,64),program_score_prior=out["program_predictions"].reshape(2,2,3,4),
               patch_mask=torch.ones(2,2,3,dtype=torch.bool))
    frozen,receipt=decode_frozen_source_cache(source,cache,expected_source_state_sha256=source_model_state_sha256(source),batch_size=5)
    assert receipt["score_max_abs_error"]<=2e-6
    assert torch.allclose(frozen.program_keys,out["queries"].reshape(2,2,3,4,64),atol=2e-7)
    assert receipt["new_optimizer_fits"]==0 and receipt["source_visual_input_uses_RNA"] is False
    cache["program_score_prior"]=cache["program_score_prior"].clone();cache["program_score_prior"][0,0,0,0]+=.1
    with pytest.raises(ValueError,match="DO_NOT_REPLAY"):
        decode_frozen_source_cache(source,cache,expected_source_state_sha256=source_model_state_sha256(source))


def test_observed_zero_is_not_missing_and_eval_does_not_refit_pca():
    x=np.array([[0.,3.],[10.,5.],[999.,7.]])
    mask=np.array([[1,1],[1,1],[0,1]],bool)
    p=MaskedPCA.fit(x,components=2,measured_mask=mask,fit_identity_sha256=FIT)
    assert p.mean[0]==5. and np.array_equal(p.observed_count,[2,3])
    before=[a.copy() for a in (p.mean,p.sd,p.components)]
    a=p.transform(np.array([[np.nan,9.],[5.,11.]]),np.array([[0,1],[1,1]],bool))
    assert np.isfinite(a).all()
    p.transform(np.array([[1e9,-1e9]]))
    assert all(np.array_equal(x,y) for x,y in zip((p.mean,p.sd,p.components),before))
    assert p.fit_rows==3 and p.audit()["outcomes_used"] is False


def test_program_pca_preserves_four_positions_and_missing_program_is_train_mean():
    rng=np.random.default_rng(51);x=rng.normal(size=(40,4,7));m=np.ones((40,4),bool);m[:5,2]=False
    a=BlockPCA.fit(x,components_per_program=2,program_mask=m,fit_identity_sha256=FIT)
    out=a.transform(x,m)
    assert out.shape==(40,4,2) and np.array_equal(out[:5,2],np.zeros((5,2)))
    b=BlockPCA.fit(x,components_per_program=2,program_mask=m,fit_identity_sha256=FIT)
    assert np.array_equal(out,b.transform(x,m))
    m[:,1]=False
    with pytest.raises(ValueError,match="NO_TRAIN_MEASUREMENTS"):
        BlockPCA.fit(x,components_per_program=2,program_mask=m,fit_identity_sha256=FIT)


def test_query_only_and_program_only_controls_preserve_recipient_main_effects():
    rng=np.random.default_rng(21);h=rng.normal(size=(15,4,2));g=rng.normal(size=h.shape)
    main=build_low_capacity_features(h,g,mode="additive")
    matched=build_low_capacity_features(h,g,mode="bilinear_rank2")
    wrong=build_low_capacity_features(h,g,mode="bilinear_rank2",query_gene2=np.roll(g,1,axis=0))
    wp=build_low_capacity_features(h,g,mode="bilinear_rank2",program_permutation=[1,2,3,0])
    none=build_low_capacity_features(h,g,mode="additive_quadratic")
    assert main.shape==(15,16) and matched.shape==wrong.shape==wp.shape==none.shape==(15,32)
    assert all(np.array_equal(x[:,:16],main) for x in (matched,wrong,wp,none))
    assert not np.array_equal(matched[:,16:],wrong[:,16:])
    assert not np.array_equal(matched[:,16:],wp[:,16:])
    assert np.array_equal(none[:,16:],main**2)
    with pytest.raises(ValueError,match="DERANGEMENT"):
        build_low_capacity_features(h,g,mode="bilinear_rank2",program_permutation=[0,2,3,1])


def test_explicit_bilinear_coefficients_equal_four_rank_two_matrices():
    rng=np.random.default_rng(7);h=rng.normal(size=(25,4,2));g=rng.normal(size=h.shape);w=rng.normal(size=(4,2,2))
    z=build_low_capacity_features(h,g,mode="bilinear_rank2")
    assert np.allclose(z[:,16:]@w.reshape(-1),np.einsum("nki,kij,nkj->n",h,w,g))
    assert all(np.linalg.matrix_rank(x)<=2 for x in w)


def test_efron_ties_value_and_gradient_match_independent_small_reference():
    t=np.array([2.,2.,3.,5.,6.]);e=np.array([1,1,0,1,0]);r=np.array([.1,.4,-.2,.3,.8])
    la,ev=efron_log_weights(t,e);value,grad=efron_loss_gradient(r,la,ev)
    raw=0.
    for at in (2.,5.):
        events=np.flatnonzero((t==at)&(e==1));d=len(events)
        raw+=sum(np.log(np.exp(r[t>=at]).sum()-ell/d*np.exp(r[events]).sum()) for ell in range(d))-r[events].sum()
    assert abs(value-raw/e.sum())<1e-13
    numeric=[]
    for k in range(len(r)):
        a=r.copy();b=r.copy();a[k]+=1e-6;b[k]-=1e-6
        numeric.append((efron_loss_gradient(a,la,ev)[0]-efron_loss_gradient(b,la,ev)[0])/2e-6)
    assert np.allclose(grad,numeric,atol=5e-10) and abs(grad.sum())<1e-13
    p=np.array([3,1,4,0,2]);lp,ep=efron_log_weights(t[p],e[p]);vp,gp=efron_loss_gradient(r[p],lp,ep)
    assert abs(value-vp)<1e-13 and np.allclose(grad[p],gp,atol=1e-14)


def sample_cox(seed=221):
    rng=np.random.default_rng(seed);x=rng.normal(size=(100,8));base=rng.normal(size=100)
    t=np.round(np.exp(-1.2*x[:,0]+.2*rng.normal(size=100)),1);e=(rng.random(100)<.75).astype(int)
    return x,t,e,base


def test_full_risk_convex_fit_meets_kkt_and_training_sd_protection():
    x,t,e,base=sample_cox();saved=base.copy()
    m=fit_protected_offset_cox(x,t,e,base,gamma=.03,ridge_lambda=1.,fit_identity_sha256=FIT)
    d=m.delta(x);assert np.array_equal(base,saved)
    assert d.std()/base.std()<=.03+1e-10
    assert m.receipt["kkt_stationarity_max"]<5e-6 and m.receipt["kkt_complementarity"]<5e-6
    assert m.receipt["registered_coefficients"]==8 and m.receipt["neural_updates"]==0
    assert abs(d.mean())<1e-12
    assert np.allclose(m.predict(x,base),base+d)
    assert m.receipt["validation_or_test_labels_used"] is False


def test_evaluation_uses_train_scale_without_validation_sd_clipping():
    x,t,e,base=sample_cox();m=fit_protected_offset_cox(x,t,e,base,gamma=.03,ridge_lambda=1.,fit_identity_sha256=FIT)
    old=[a.copy() for a in (m.mean,m.sd,m.coefficient)]
    test=m.mean+200*(x-m.mean)
    assert np.std(m.delta(test))/np.std(base)>.03*20
    assert all(np.array_equal(a,b) for a,b in zip((m.mean,m.sd,m.coefficient),old))


def test_degenerate_features_yield_offset_identity_not_fabricated_signal():
    x,t,e,base=sample_cox();x[:]=2.
    m=fit_protected_offset_cox(x,t,e,base,gamma=.1,ridge_lambda=1.,fit_identity_sha256=FIT)
    assert np.array_equal(m.predict(x,base),base)
    assert m.receipt["active_variance_columns"]==m.receipt["train_design_rank"]==0


def test_insufficient_optimizer_budget_or_invalid_labels_fail_closed():
    x,t,e,base=sample_cox()
    with pytest.raises(ValueError,match="KKT_OR_SD_BOUND"):
        fit_protected_offset_cox(x,t,e,base,gamma=.3,ridge_lambda=.1,fit_identity_sha256=FIT,maxiter=1)
    with pytest.raises(ValueError,match="ENDPOINT_INVALID"):
        fit_protected_offset_cox(x,t,np.zeros_like(e),base,gamma=.1,ridge_lambda=1.,fit_identity_sha256=FIT)
