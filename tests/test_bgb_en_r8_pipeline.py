"""Synthetic pipeline lineage and finite-arm controls, no real data."""
import hashlib

import numpy as np
import pytest

from src.bgb_en_r8.pipeline import (ARMS, PARAMETERS, FrozenPatientEvidence,
    FrozenPartitionTransform, validate_query_partition_mapping,signed_mean_dispersion_audit)

FIT=hashlib.sha256(b"synthetic-outer-train-subset").hexdigest()


def data(seed=21,random=False):
    rng=np.random.default_rng(seed);h=rng.normal(size=(30,4,64));g=rng.normal(size=h.shape)
    mask=np.ones((30,4),bool);mask[:3,2]=False;g[~mask]=np.nan
    return FrozenPatientEvidence(rng.normal(size=(30,4)),h,h+.02*rng.normal(size=h.shape),g,mask,
        hashlib.sha256(str(seed).encode()).hexdigest(),"R6_S7_RANDOM_NO_ST_locked" if random else "R6_S7_locked")


def query_map():
    p=np.arange(30);p[:20]=np.roll(p[:20],1);p[20:]=np.roll(p[20:],1)
    return p


def test_all_prelocked_arms_have_exact_registered_capacity_and_shared_main_effects():
    d=data();p=FrozenPartitionTransform.fit(d,np.arange(20),fit_identity_sha256=FIT)
    base=p.features(d,"R8-4");matched=p.features(d,"R8-6")
    wrong=p.features(d,"R8-9_WRONG_PATIENT",query_mapping=query_map())
    wp=p.features(d,"R8-8b_WRONG_PROGRAM");no=p.features(d,"R8-10_NO_INTERACTION")
    assert all(np.array_equal(x[:,:16],base) for x in (matched,wrong,wp,no))
    assert not np.array_equal(matched[:,16:],wrong[:,16:])
    assert not np.array_equal(matched[:,16:],wp[:,16:])
    for arm in ARMS:
        if arm=="R8-8a_RANDOM_SOURCE":continue
        out=p.features(d,arm,query_mapping=query_map() if arm=="R8-9_WRONG_PATIENT" else None)
        assert out.shape==(30,PARAMETERS[arm]) and np.isfinite(out).all()
    assert p.audit()["unsupervised_PCA_fits"]==10
    assert p.audit()["target_outcomes_used"] is False


def test_validation_perturbation_cannot_change_any_training_transform():
    d=data();a=FrozenPartitionTransform.fit(d,np.arange(20),fit_identity_sha256=FIT)
    changed=FrozenPatientEvidence(*(x.copy() for x in (d.image_scores,d.image_tokens,d.conditioned_image_tokens,d.gene_tokens,d.gene_program_mask)),
        d.source_state_sha256,d.source_version)
    for x in (changed.image_scores,changed.image_tokens,changed.conditioned_image_tokens,changed.gene_tokens):x[20:]*=1000
    b=FrozenPartitionTransform.fit(changed,np.arange(20),fit_identity_sha256=FIT)
    assert np.array_equal(a.features(d,"R8-6")[:20],b.features(changed,"R8-6")[:20])
    assert np.array_equal(a.image_pca8.mean,b.image_pca8.mean)
    assert np.array_equal(a.gene_blocks2.blocks[0].components,b.gene_blocks2.blocks[0].components)


def test_random_source_requires_separate_qualified_state_and_transform():
    d=data();r=data(92,True)
    a=FrozenPartitionTransform.fit(d,np.arange(20),fit_identity_sha256=FIT)
    b=FrozenPartitionTransform.fit(r,np.arange(20),fit_identity_sha256=FIT)
    assert b.features(r,"R8-8a_RANDOM_SOURCE").shape==(30,32)
    with pytest.raises(ValueError,match="SOURCE_STATE_CHANGED"):
        a.features(r,"R8-8a_RANDOM_SOURCE")
    with pytest.raises(ValueError,match="VERSION_MISMATCH"):
        a.features(d,"R8-8a_RANDOM_SOURCE")
    with pytest.raises(ValueError,match="VERSION_MISMATCH"):
        b.features(r,"R8-6")


def test_wrong_patient_mapping_is_strict_within_each_train_validation_partition():
    valid=validate_query_partition_mapping(query_map(),np.arange(20),np.arange(20,30),30)
    assert valid["self_pairs"]==valid["train_validation_crossings"]==0
    with pytest.raises(ValueError,match="CROSSES_TRAIN_VALIDATION"):
        validate_query_partition_mapping(np.roll(np.arange(30),1),np.arange(20),np.arange(20,30),30)
    with pytest.raises(ValueError,match="HAS_SELF_PAIR"):
        validate_query_partition_mapping(np.arange(30),np.arange(20),np.arange(20,30),30)


def test_signed_proxy_respects_observed_zero_and_missing_signed_component():
    import torch
    from src.bgb_en.r6_omics import GeneValueBank,mapping_from_programs
    definitions=[dict(id="P0",up_genes=["A"],down_genes=["B"]),
        dict(id="P1",up_genes=["B"],down_genes=[]),dict(id="P2",up_genes=["C"],down_genes=[]),
        dict(id="P3",up_genes=["D"],down_genes=[])]
    mapping=mapping_from_programs(definitions);bank=GeneValueBank(["A","B","C","D"],mapping.gene_symbols)
    values=torch.tensor([[0.,2.,3.,4.],[0.,2.,3.,4.]])
    mask=torch.ones_like(values,dtype=torch.bool);mask[1,1]=False
    measurements=bank(values,mask);proxy=signed_mean_dispersion_audit(measurements,mapping)
    assert proxy["values"][0,0,0]==-2. and proxy["program_mask"][0,0]
    assert not proxy["program_mask"][1,0] and not proxy["program_mask"][1,1]
    assert proxy["program_mask"][1,2] and proxy["values"][1,2,0]==3.
    assert proxy["diagnostics"]["fed_to_any_locked_DSS_readout"] is False
