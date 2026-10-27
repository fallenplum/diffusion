import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts import bgb_en_patch_throughput as throughput


def test_forward_counts_separate_inner_r1_r3_and_sealed_outer():
    r1=throughput.admissions('INNER_R1',80,3,100,50,3,2,8)
    r3=throughput.admissions('INNER_R3',80,3,100,50,3,2,8)
    outer=throughput.admissions('OUTER_R3_FINAL',80,0,100,50,3,2,8)
    assert r1['train_complete_patient_passes']==163
    assert r1['validation_complete_patient_passes']==3
    assert r1['logical_valid_node_admissions']==16450
    assert r1['actual_visual_forward_invocations']==166
    assert r3['logical_valid_node_admissions']==16750
    assert r3['validation_complete_patient_passes']==9
    assert outer['logical_valid_node_admissions']==16350
    assert outer['train_complete_patient_passes']==160
    assert outer['validation_complete_patient_passes']==7


def test_cache_counts_valid_mask_over_multiple_slides_without_opening_features(tmp_path,monkeypatch):
    owner=np.array([0,0,1],np.int64)
    mask=np.array([[1,1],[1,0],[1,0]],bool)
    np.save(tmp_path/'owner.npy',owner);np.save(tmp_path/'mask.npy',mask)
    (tmp_path/'features.npy').write_bytes(b'not a valid numpy file; must never be opened')
    (tmp_path/'coords.npy').write_bytes(b'not a valid numpy file; must never be opened')
    receipt=dict(schema='bgb_en_native_patch_cache_v1',status='COMPLETE',patients=2,
        source_order='canonical patient index then source slide',feature_shape=[3,2,768],selected_patches=4,
        budget_per_patient=2,selector='REGIONS',
        array_sha256={key:throughput.sha(tmp_path/(key+'.npy')) for key in ('owner','mask')},
        source_h5_collection_sha256='a'*64)
    (tmp_path/'READY.json').write_text(json.dumps(receipt))
    counts,audit=throughput.patch_counts(tmp_path,2)
    assert np.array_equal(counts,[3,1])
    assert audit['features_array_opened'] is False and audit['coordinates_array_opened'] is False
    assert audit['selected_valid_nodes']==4
    assert audit['budget_per_patient']==2 and audit['visual_feature_dim']==768 and audit['cache_selector']=='REGIONS'
    del receipt['source_order']
    catalog=tmp_path/'inventory.json'
    catalog.write_text(json.dumps({'chosen_conch':[{'patient_index':1,'stem':'c'},
        {'patient_index':0,'stem':'b'},{'patient_index':0,'stem':'a'}]}))
    receipt['source_inventory_sha256']=throughput.sha(catalog)
    (tmp_path/'READY.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='inventory'):
        throughput.patch_counts(tmp_path,2)
    counts,audit=throughput.patch_counts(tmp_path,2,catalog)
    assert np.array_equal(counts,[3,1]) and audit['source_inventory_sha256']==throughput.sha(catalog)
    wrong=json.loads(catalog.read_text());wrong['chosen_conch'][0]['patient_index']=0
    catalog.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='inventory'):
        throughput.patch_counts(tmp_path,2,catalog)
    (tmp_path/'mask.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        throughput.patch_counts(tmp_path,2)


def test_private_indices_extracted_once_then_no_mixed_bundle_reload(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle.pt';receipt=tmp_path/'bundle.json';destination=tmp_path/'private'/'indices.npz'
    train=np.array([0,2]);valid=np.array([1])
    torch.save(dict(train=train,valid=valid,fit_hash=throughput.array_hash(train),valid_hash=throughput.array_hash(valid),
                    features={'example_measurement':np.array([4.,5.,6.])}),bundle)
    receipt.write_text(json.dumps({'sha256':throughput.sha(bundle)}))
    a,b,proof=throughput.index_cache(bundle,receipt,destination,3,'a'*64,create=True)
    assert np.array_equal(a,train) and np.array_equal(b,valid)
    with np.load(destination,allow_pickle=False) as archive:assert set(archive.files)=={'train','valid'}
    def forbidden(*args,**kwargs):raise AssertionError('cached index analysis must not re-deserialize bundle')
    monkeypatch.setattr(torch,'load',forbidden)
    assert throughput.index_cache(bundle,receipt,destination,3,'a'*64)[2]==proof


def test_rna_only_is_zero_while_references_and_unknown_failure_are_na():
    assert throughput.classify_nonvisual({'status':'COMPLETE','category':'NEURAL','stage':'R2'})=='SUPPORTED_ZERO_VISUAL_WORK'
    assert throughput.classify_nonvisual({'status':'COMPLETE','category':'NEURAL'},'budget')=='SUPPORTED_ZERO_VISUAL_WORK'
    assert throughput.classify_nonvisual({'status':'COMPLETE','category':'REFERENCE','stage':'R2'}).startswith('NA_')
    assert throughput.classify_nonvisual({'status':'FAILED','category':'UNKNOWN_ATTEMPT'}).startswith('NA_')
    assert throughput.classify_nonvisual({'status':'COMPLETE','category':'EXTERNAL_SOURCE'}).startswith('NOT_APPLICABLE_')


def test_canonical_outer_reads_only_folds_and_checks_final_contract(tmp_path):
    path=tmp_path/'cohort.npz'
    folds=np.tile(np.arange(1,6),2)
    # An object member would fail under allow_pickle=False if the auditor
    # accessed it. It must touch only folds, not IDs or measurements.
    np.savez(path,folds=folds,patient_ids=np.array([{'forbidden':'object'}],dtype=object))
    registry={'patients':10,'hashes':{'cohort_sha256':throughput.sha(path),'patient_order_sha256':'a'*64}}
    actual,evidence=throughput.canonical_folds({'cohort':path},registry)
    assert np.array_equal(actual,folds) and evidence['cohort_members_accessed']==['folds']
    train,test=throughput.canonical_outer_indices(folds,3)
    contract={'train_hash':throughput.array_hash(train),'test_hash':throughput.array_hash(test)}
    assert np.array_equal(throughput.canonical_outer_indices(folds,3,contract)[1],[2,7])
    contract['train_hash']='0'*64
    with pytest.raises(ValueError,match='contract indices'):
        throughput.canonical_outer_indices(folds,3,contract)


def test_nested_indices_cannot_include_sealed_outer_patient(tmp_path):
    root=tmp_path/'outer'/'nested_inner'/'prepared';root.mkdir(parents=True)
    train=np.array([0,2]);valid=np.array([3,4]);folds=np.arange(1,6)
    stem=root/'outer2_inner0';lineage={'cohort_sha256':'a'*64}
    torch.save(dict(train=train,valid=valid,fit_hash=throughput.array_hash(train),
                    valid_hash=throughput.array_hash(valid)),stem.with_suffix('.pt'))
    stem.with_suffix('.json').write_text(json.dumps({'status':'COMPLETE','lineage':lineage,
        'sha256':throughput.sha(stem.with_suffix('.pt'))}))
    registry={'patients':5,'hashes':{'cohort_sha256':'a'*64,'patient_order_sha256':'b'*64}}
    args=(tmp_path/'unused_core',tmp_path/'indices',2,0,registry)
    a,b,_=throughput.index_identity(*args,create=True,outer_private=tmp_path/'outer',folds=folds)
    assert np.array_equal(a,train) and np.array_equal(b,valid)
    with pytest.raises(ValueError,match='partition its canonical outer train'):
        throughput.index_identity(*args,outer_private=tmp_path/'outer',folds=np.array([2,1,3,4,5]))


def test_matched_support_lineage_strips_only_the_additional_wrapper():
    original={'source':{'a':'b'},'cohort_sha256':'a'*64}
    wrapped={**original,'matched_support_auxiliary':{'null':{'hash':'a'}}}
    assert throughput.original_lineage({'lineage':wrapped})==original
    assert 'matched_support_auxiliary' in wrapped
