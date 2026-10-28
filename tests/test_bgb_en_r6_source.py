import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from src.bgb_en.r6_source import (SourceContract,SourceSpotDataset,composite_identity,
    measured_gene_bank,normalize_features,program_index,scores_from_gene_bank,spatial_graph,
    write_private_cache,load_private_cache)


def test_composite_identity_repeated_barcodes_need_section():
    sc,dc,audit=composite_identity(['s1','s1','s2'],['b1','b2','b1'],['d1','d1','d2'])
    assert audit['unique_composite_keys']==3 and audit['globally_unique_barcode_strings']==2
    assert np.array_equal(sc,[0,0,1]) and np.array_equal(dc,[0,0,1])
    with pytest.raises(ValueError,match='DUPLICATE_SECTION_BARCODE'):
        composite_identity(['s1','s1'],['b1','b1'],['d1','d1'])
    with pytest.raises(ValueError,match='CROSSES_DONORS'):
        composite_identity(['s1','s1'],['b1','b2'],['d1','d2'])


def test_gene_bank_keeps_full_library_and_measured_zero():
    raw=sparse.csr_matrix([[0,5,95],[2,0,98]],dtype=np.float64)
    count,mask,value,library,detected=measured_gene_bank(raw,['A','B','background'],['A','B','C'])
    assert np.array_equal(count.toarray(),[[0,5,0],[2,0,0]])
    assert np.array_equal(mask,[[True,True,False],[True,True,False]])
    assert np.array_equal(library,[100,100]) and np.array_equal(detected,[2,2])
    assert value[0,1]==pytest.approx(np.log1p(500))
    assert value[0,0]==0 and mask[0,0] and not mask[0,2]
    with pytest.raises(ValueError,match='DUPLICATE_GENE'):
        measured_gene_bank(raw,['A','A','background'],['A'])


def test_program_missing_support_returns_zero_invalid_not_nan():
    gene,up,down=program_index([dict(id='p1',up_genes=['A'],down_genes=[]),
                               dict(id='p2',up_genes=['B'],down_genes=['C'])])
    assert gene.tolist()==['A','B','C']
    score,valid=scores_from_gene_bank(np.zeros((2,3)),np.array([[True,False,False],[False,False,False]]),
        up,down,minimum_genes=1,minimum_coverage=1)
    assert np.isfinite(score).all() and np.array_equal(score,np.zeros((2,2)))
    assert np.array_equal(valid,[[True,False],[False,False]])


def test_member_value_changes_only_its_program():
    gene,up,down=program_index([dict(id='p1',up_genes=['A'],down_genes=[]),
                               dict(id='p2',up_genes=['B'],down_genes=[])])
    mask=np.ones((1,2),bool);x=np.array([[1.,2.]])
    before,_=scores_from_gene_bank(x,mask,up,down,minimum_genes=1,minimum_coverage=1)
    x[0,0]=3.
    after,_=scores_from_gene_bank(x,mask,up,down,minimum_genes=1,minimum_coverage=1)
    assert after[0,0]!=before[0,0] and after[0,1]==before[0,1]


def test_neighbors_are_same_section_and_mean_unit_is_explicit():
    raw=np.array([[3.,0.],[0.,4.],[8.,6.]],dtype=np.float32)
    x,_=normalize_features(raw)
    index,count,mean,mean_unit=spatial_graph(x,[[0,0],[1,0],[0,0]],[0,0,1],radius_um=2.,maximum_neighbors=2)
    assert np.array_equal(count,[2,2,1]) and np.array_equal(index[2],[2,-1])
    assert np.allclose(mean[0],[.5,.5]) and np.allclose(mean_unit[0],[2**-.5,2**-.5])
    legacy_raw_mean=raw[:2].mean(0);legacy_raw_mean/=np.linalg.norm(legacy_raw_mean)
    assert not np.allclose(mean_unit[0],legacy_raw_mean)
    for i in range(3):assert np.all(np.array([0,0,1])[index[i,:count[i]]]==[0,0,1][i])
    with pytest.raises(ValueError,match='NOT_UNIT'):
        spatial_graph(raw,[[0,0],[1,0],[0,0]],[0,0,1])


def test_no_zero_feature_or_duplicate_within_slide_coordinate():
    with pytest.raises(ValueError,match='ZERO_OR_NONFINITE'):
        normalize_features(np.zeros((1,2),np.float32))
    x,_=normalize_features(np.ones((2,2),np.float32))
    with pytest.raises(ValueError,match='DUPLICATE_WITHIN_SECTION'):
        spatial_graph(x,[[0,0],[0,0]],[0,0])


def small_dataset():
    x,_=normalize_features(np.array([[3.,0.],[0.,4.]],np.float32))
    ni,nc,mean,unit=spatial_graph(x,[[0,0],[1,0]],[0,0],radius_um=2.,maximum_neighbors=2)
    c=SourceContract(fresh_certificate_sha256='a'*64,primary_spots=2,donors=1,sections=1)
    audit=dict(prepared_data_sha256=c.prepared_data_sha256,prepared_manifest_sha256=c.prepared_manifest_sha256,
        program_definition_sha256=c.program_definition_sha256,fresh_certificate_sha256=c.fresh_certificate_sha256,
        primary_spots=2,source_donors=1,source_sections=1,full_source_genes=c.full_source_genes,
        radius_um=c.radius_um,maximum_neighbors_including_self=c.maximum_neighbors,
        program_replay_tolerance=c.program_replay_tolerance)
    return SourceSpotDataset(features_unit=x,counts=sparse.csr_matrix([[0.,2.],[3.,0.]]),
        measured_mask=np.ones((2,2),bool),log1p_cp10k=np.ones((2,2),np.float64),
        program_targets=np.ones((2,1),np.float64),tissue_y=np.array([0,1]),
        program_up=np.ones((1,2),bool),program_down=np.zeros((1,2),bool),gene_symbols=np.array(['A','B']),
        coords_um=np.array([[0,0],[1,0]],np.float64),section_codes=np.zeros(2,np.int64),
        donor_codes=np.zeros(2,np.int64),library_size=np.array([10.,20.]),n_detected_genes=np.array([2,2]),
        neighbors=ni,neighbor_counts=nc,neighbor_mean_of_unit=mean,neighbor_mean_unit=unit,audit=audit),c


def test_private_cache_roundtrip_integrity_and_no_overwrite(tmp_path):
    dataset,contract=small_dataset();target=tmp_path/'private_cache'
    report=write_private_cache(dataset,target,repository_root=tmp_path/'repo',raw_root=tmp_path/'raw')
    assert report['status']=='COMPLETE_PRIVATE_DERIVED_CACHE'
    recovered=load_private_cache(target,contract,expected_manifest_sha256=report['manifest_sha256'])
    assert np.array_equal(recovered.features_unit,dataset.features_unit)
    assert np.array_equal(recovered.counts.toarray(),dataset.counts.toarray())
    assert np.array_equal(recovered.tissue_y,dataset.tissue_y)
    assert recovered.model_gene_values([0]).dtype==np.float32
    with pytest.raises(ValueError,match='NO_OVERWRITE'):
        write_private_cache(dataset,target,repository_root=tmp_path/'repo',raw_root=tmp_path/'raw')
    with (target/'gene_counts.private.npz').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='PAYLOAD_CHANGED'):
        load_private_cache(target,contract,expected_manifest_sha256=report['manifest_sha256'])


def test_private_cache_never_written_in_repository_or_raw(tmp_path):
    dataset,_=small_dataset()
    with pytest.raises(ValueError,match='OUTSIDE_REPOSITORY'):
        write_private_cache(dataset,tmp_path/'repo'/'bad',repository_root=tmp_path/'repo',raw_root=tmp_path/'raw')
    with pytest.raises(ValueError,match='OUTSIDE_REPOSITORY'):
        write_private_cache(dataset,tmp_path/'raw'/'bad',repository_root=tmp_path/'repo',raw_root=tmp_path/'raw')


def test_cache_cannot_cross_geometry_lock_or_forge_manifest(tmp_path):
    from dataclasses import replace
    dataset,contract=small_dataset();target=tmp_path/'private_cache'
    report=write_private_cache(dataset,target,repository_root=tmp_path/'repo',raw_root=tmp_path/'raw')
    changed=replace(contract,radius_um=100.)
    with pytest.raises(ValueError,match='GEOMETRY_OR_MEASUREMENT_CONTRACT_CHANGED'):
        load_private_cache(target,changed,expected_manifest_sha256=report['manifest_sha256'])
    manifest=target/'manifest.private.json';data=json.loads(manifest.read_text());data['status']='FORGED'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='MANIFEST_SHA_CHANGED'):
        load_private_cache(target,contract,expected_manifest_sha256=report['manifest_sha256'])
    with pytest.raises(ValueError,match='CANNOT_BE_RELAXED'):
        replace(contract,program_replay_tolerance=1e-5)
