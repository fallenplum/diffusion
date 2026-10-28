"""R6 raw-gene identity, sparse sign groups and mask/query invariants only."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.bgb_en.r6_omics import (GeneValueBank, GeneMeasurements, GeneEvidenceTransform,
    SparseProgramEncoder, OriginalGeneRereader, SignedProgramMapping,
    mapping_from_programs, load_source_program_mapping, PROGRAM_ORDER)


def programs():
    return [dict(id='p0', up_genes=['A', 'B'], down_genes=['C']),
            dict(id='p1', up_genes=['D', 'E'], down_genes=[]),
            dict(id='p2', up_genes=['B'], down_genes=['A']),
            dict(id='p3', up_genes=['F'], down_genes=[])]


def fixture(dtype=torch.float64):
    mapping=mapping_from_programs(programs())
    bank=GeneValueBank(['background', 'A', 'B', 'C', 'D', 'E', 'F'], mapping.gene_symbols)
    x=torch.tensor([[.1, 1., 2., 3., 4., 5., 6.], [.2, 3., 2., 1., 5., 4., 8.]],dtype=dtype)
    mask=torch.ones_like(x,dtype=torch.bool);mask[1,3]=False
    x[~mask]=float('nan')
    torch.manual_seed(61031)
    stem=GeneEvidenceTransform().to(dtype)
    encoder=SparseProgramEncoder(mapping,gene_encoder=stem).to(dtype)
    reader=OriginalGeneRereader(mapping,gene_encoder=stem).to(dtype)
    evidence=torch.randn(2,4,64,dtype=dtype)
    return mapping,bank,x,mask,stem,encoder,reader,evidence


def test_fixed_source_mapping_is_four_programs_not_fifty_hallmarks():
    path=Path(__file__).resolve().parents[1]/'configs/sami_surv_stage_b_programs_v1.json'
    mapping=load_source_program_mapping(path)
    audit=mapping.public_audit()
    assert mapping.program_names==PROGRAM_ORDER and audit['union_genes']==760 and audit['signed_edges']==765
    assert audit['union_gene_order_sha256']=='0e56b530399d863077cff1caba105b97ce49b5fa7dfbc728a1603bd29f62be85'
    assert [p['up_defined'] for p in audit['per_program']]==[308,74,25,25]
    assert [p['down_defined'] for p in audit['per_program']]==[279,54,0,0]
    assert not audit['Hallmark_tokens_claimed_ST_aligned']
    with pytest.raises(ValueError,match='SHA_CHANGED'):
        load_source_program_mapping(path,expected_sha256='0'*64)


def test_raw_rna_gather_preserves_measured_zero_and_absent_symbols():
    bank=GeneValueBank(['background','B','A'],['A','B','C'])
    raw=torch.tensor([[9.,float('nan'),0.]])
    mask=torch.tensor([[True,False,True]])
    batch=bank(raw,mask)
    assert batch.values.shape==(1,3,1)
    assert torch.equal(batch.values,torch.zeros_like(batch.values))
    assert torch.equal(batch.measured_mask[:,:,0],torch.tensor([[True,False,False]]))
    assert sum(p.numel() for p in bank.parameters())==0
    assert bank.public_audit()['RNA_union_absent']==1
    assert torch.isnan(raw[0,1]) and raw[0,0]==9


def test_original_columns_and_gene_ids_joint_reordering_is_invariant():
    _,bank,raw,mask,_,encoder,_,_=fixture()
    before=encoder(bank(raw,mask))
    order=torch.tensor([6,2,0,4,1,5,3])
    reordered=bank(raw[:,order],mask[:,order],gene_ids=order)
    assert torch.equal(before['tokens'],encoder(reordered)['tokens'])
    with pytest.raises(ValueError,match='BIJECTION'):
        bank(raw,mask,gene_ids=torch.zeros(7,dtype=torch.long))
    with pytest.raises(ValueError,match='BIJECTION'):
        bank(raw,mask,gene_ids=torch.arange(7,dtype=torch.float32))


def test_cna_is_off_by_default_and_gain_loss_masks_are_independent():
    bank=GeneValueBank(['A','B'],['A','B','C'])
    x=torch.tensor([[0.,2.]]);mask=torch.ones_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='CNA_DISABLED'):
        bank(x,mask,cna_gain=x)
    with pytest.raises(ValueError,match='CNA_DISABLED'):
        GeneValueBank(['A','B'],['A'],cna_gene_symbols=['A'])
    enabled=GeneValueBank(['A','B'],['A','B','C'],enable_cna=True,cna_gene_symbols=['C','B','A'])
    gain=torch.tensor([[1.,2.,float('nan')]]);loss=torch.tensor([[3.,0.,4.]])
    gm=torch.tensor([[True,True,False]]);lm=torch.tensor([[False,True,True]])
    batch=enabled(x,mask,cna_gain=gain,cna_loss=loss,cna_gain_mask=gm,cna_loss_mask=lm)
    assert torch.equal(batch.values,torch.tensor([[[0.,0.,4.],[2.,2.,0.],[0.,1.,0.]]]))
    assert torch.equal(batch.measured_mask,torch.tensor([[[True,False,True],[True,True,True],[False,True,False]]]))
    order=torch.tensor([2,0,1])
    changed=enabled(x,mask,cna_gain=gain[:,order],cna_loss=loss[:,order],cna_gain_mask=gm[:,order],
                    cna_loss_mask=lm[:,order],cna_gene_ids=order)
    assert torch.equal(batch.values,changed.values) and torch.equal(batch.measured_mask,changed.measured_mask)
    with pytest.raises(ValueError,match='SEPARATE_EXPLICIT_MASKS'):
        enabled(x,mask,cna_gain=gain,cna_loss=loss,cna_gain_mask=gm)
    with pytest.raises(ValueError,match='CNA_ENCODER'):
        GeneEvidenceTransform()(batch)


def test_shared_stem_parameter_budget_and_identity_are_explicit():
    mapping,bank,raw,mask,stem,encoder,reader,_=fixture()
    network=nn.ModuleDict(dict(encoder=encoder,reader=reader))
    assert encoder.gene_encoder is reader.gene_encoder
    assert sum(p.numel() for p in stem.parameters())==8832
    assert sum(p.numel() for p in network.parameters())==13184
    assert encoder.queries.numel()==256 and reader.query_projection.weight.numel()==4096
    assert not bank._identity.requires_grad and not bank.state_dict()
    enabled=GeneEvidenceTransform(enable_cna=True)
    assert sum(p.numel() for p in enabled.parameters())==9088


def test_cna_channel_gradients_respect_separate_masks_and_all_missing_null():
    mapping=mapping_from_programs([dict(id='p',up_genes=['A'],down_genes=['B'])])
    bank=GeneValueBank(['A','B'],mapping.gene_symbols,enable_cna=True,cna_gene_symbols=['B','A'])
    rna=torch.tensor([[1.,float('nan')],[float('nan'),float('nan')]],requires_grad=True)
    rna_mask=torch.tensor([[True,False],[False,False]])
    gain=torch.tensor([[float('nan'),2.],[float('nan'),float('nan')]],requires_grad=True)
    loss=torch.tensor([[3.,float('nan')],[float('nan'),float('nan')]],requires_grad=True)
    gm=torch.tensor([[False,True],[False,False]]);lm=torch.tensor([[True,False],[False,False]])
    batch=bank(rna,rna_mask,cna_gain=gain,cna_loss=loss,cna_gain_mask=gm,cna_loss_mask=lm)
    stem=GeneEvidenceTransform(enable_cna=True)
    reader=OriginalGeneRereader(mapping,gene_encoder=stem)
    output=reader(torch.randn(2,1,64),batch)
    assert not output['program_mask'][1].any() and output['tokens'][1].abs().sum()==0
    output['tokens'].square().sum().backward()
    for value,mask in ((rna,rna_mask),(gain,gm),(loss,lm)):
        assert torch.isfinite(value.grad).all() and value.grad[~mask].abs().sum()==0
        assert value.grad[mask].abs().sum()>0


def test_keys_and_values_depend_on_original_measured_gene_values():
    _,bank,raw,mask,stem,_,_,_=fixture()
    initial=stem(bank(raw,mask));changed=raw.clone();changed[0,1]+=.7
    later=stem(bank(changed,mask))
    assert not torch.equal(initial['keys'][0,0],later['keys'][0,0])
    assert not torch.equal(initial['values'][0,0],later['values'][0,0])
    assert torch.equal(initial['keys'][:,1:],later['keys'][:,1:])
    assert torch.equal(initial['values'][1],later['values'][1])


def test_signed_groups_normalize_separately_and_missing_down_is_not_negative_truth():
    mapping,bank,raw,mask,_,encoder,_,_=fixture()
    result=encoder(bank(raw,mask))
    assert result['weights'].shape==(2,len(mapping.edge_gene))
    for k in range(4):
        for at,sign in enumerate((1,-1)):
            edge=(mapping.edge_program==k)&(mapping.edge_sign==sign)
            mass=result['weights'][:,edge].sum(-1)
            assert torch.equal(mass>0,result['component_mask'][:,k,at])
            torch.testing.assert_close(mass[result['component_mask'][:,k,at]],
                                       torch.ones_like(mass[result['component_mask'][:,k,at]]))
    assert not result['component_defined'][1,1] and not result['component_mask'][:,1,1].any()
    assert torch.equal(result['down_tokens'][:,1],torch.zeros_like(result['down_tokens'][:,1]))
    assert (result['signed_weights'][:,mapping.edge_sign<0]<=0).all()
    assert torch.equal(result['tokens'],result['up_tokens']-result['down_tokens'])


def test_all_missing_values_have_exact_zero_outputs_weights_and_finite_zero_gradients():
    _,bank,raw,_,_,encoder,reader,evidence=fixture()
    mask=torch.zeros_like(raw,dtype=torch.bool);x=torch.full_like(raw,float('nan'),requires_grad=True)
    batch=bank(x,mask)
    result=encoder(batch);reread=reader(evidence,batch)
    for row in (result,reread):
        assert torch.isfinite(row['tokens']).all() and not row['program_mask'].any()
        assert torch.equal(row['tokens'],torch.zeros_like(row['tokens']))
        assert torch.equal(row['weights'],torch.zeros_like(row['weights']))
    (result['tokens'].sum()+reread['tokens'].sum()).backward()
    assert torch.equal(x.grad,torch.zeros_like(x.grad))
    for parameter in nn.ModuleDict(dict(encoder=encoder,reader=reader)).parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()==0


def test_gene_absent_from_canonical_vocabulary_is_never_read():
    mapping=mapping_from_programs(programs());bank=GeneValueBank(['background'],mapping.gene_symbols)
    batch=bank(torch.ones(2,1),torch.ones(2,1,dtype=torch.bool))
    result=SparseProgramEncoder(mapping)(batch)
    assert torch.equal(result['tokens'],torch.zeros_like(result['tokens']))
    assert not result['program_mask'].any()


def test_changing_one_gene_changes_only_member_programs_with_fixed_query_and_preprocessing():
    _,bank,raw,mask,_,encoder,reader,evidence=fixture()
    original=bank(raw,mask);changed=raw.clone();changed[0,1]+=.9;modified=bank(changed,mask)
    for operator in (lambda batch:encoder(batch),lambda batch:reader(evidence,batch)):
        before,after=operator(original)['tokens'],operator(modified)['tokens']
        assert not torch.equal(before[0,0],after[0,0]) and not torch.equal(before[0,2],after[0,2])
        assert torch.equal(before[:,[1,3]],after[:,[1,3]])
        assert torch.equal(before[1],after[1])


def test_rereader_query_context_changes_selection_not_own_gene_values_or_masks():
    _,bank,raw,mask,stem,_,reader,evidence=fixture()
    batch=bank(raw,mask);before_values=batch.values.clone();before_mask=batch.measured_mask.clone()
    keys=stem(batch)['keys'].clone();source_mask=torch.ones(2,4,dtype=torch.bool);source_mask[0,2]=False
    normal=reader(evidence,batch)
    swapped=reader(evidence,batch,query_context=evidence.flip(0),query_mask=source_mask)
    assert not torch.equal(normal['weights'],swapped['weights'])
    assert torch.equal(before_values,batch.values) and torch.equal(before_mask,batch.measured_mask)
    assert torch.equal(keys,stem(batch)['keys'])
    assert torch.equal(swapped['tokens'][0,2],torch.zeros_like(swapped['tokens'][0,2]))
    assert not swapped['program_mask'][0,2]
    with pytest.raises(ValueError,match='SOURCE_MASK'):
        reader(evidence,batch,query_context=evidence.flip(0))


def test_no_patient_reads_another_patients_gene_values():
    _,bank,raw,mask,_,_,reader,evidence=fixture()
    before=reader(evidence,bank(raw,mask))['tokens']
    changed=raw.clone();changed[1]=99.
    after=reader(evidence,bank(changed,mask))['tokens']
    assert torch.equal(before[0],after[0])


def test_vocabulary_identity_values_masks_and_edge_reindexing_preserve_outputs():
    mapping,bank,raw,mask,stem,encoder,reader,evidence=fixture()
    batch=bank(raw,mask);order=torch.tensor([5,2,4,0,3,1])
    renamed=mapping.reindex_genes(order);reordered=batch.reindex(order)
    other_encoder=SparseProgramEncoder(renamed,gene_encoder=stem).double()
    other_reader=OriginalGeneRereader(renamed,gene_encoder=stem).double()
    with torch.no_grad():
        other_encoder.queries.copy_(encoder.queries)
        other_reader.query_projection.weight.copy_(reader.query_projection.weight)
    torch.testing.assert_close(encoder(batch)['tokens'],other_encoder(reordered)['tokens'],atol=1e-14,rtol=1e-12)
    torch.testing.assert_close(reader(evidence,batch)['tokens'],other_reader(evidence,reordered)['tokens'],atol=1e-14,rtol=1e-12)
    with pytest.raises(ValueError,match='VOCABULARY_ORDER'):
        reader(evidence,reordered)


def test_full_members_default_and_top64_is_the_same_sparse_kernel_control():
    genes=[f'G{i:03d}' for i in range(100)]
    mapping=mapping_from_programs([dict(id='large',up_genes=genes[:60],down_genes=genes[60:]),
                                   dict(id='small',up_genes=genes[:8],down_genes=[])])
    bank=GeneValueBank(genes,mapping.gene_symbols);raw=torch.randn(2,100)
    mask=torch.ones_like(raw,dtype=torch.bool);mask[1]=False
    encoder=SparseProgramEncoder(mapping);batch=bank(raw,mask)
    full=encoder(batch);small=encoder(batch,topk=64)
    assert full['all_members_used'] and full['topk_per_program_before_signed_normalization'] is None
    assert not small['all_members_used'] and (small['retained_counts'].sum(-1)<=64).all()
    assert torch.equal(full['tokens'][:,1],small['tokens'][:,1])
    assert torch.isfinite(small['tokens']).all() and not small['program_mask'][1].any()
    assert full['weights'].shape==(2,108) and not full['dense_gene_gene_attention_used']


def test_nonfinite_observed_and_pooled_token_replacements_fail_closed():
    mapping,bank,raw,mask,_,encoder,reader,evidence=fixture()
    bad=raw.clone();bad[0,1]=float('nan')
    with pytest.raises(ValueError,match='NONFINITE_OBSERVED'):
        bank(bad,mask)
    with pytest.raises(ValueError,match='NOT_POOLED'):
        encoder(torch.ones(2,4,64))
    with pytest.raises(ValueError,match='NOT_POOLED'):
        reader(evidence,torch.ones(2,4,64))
    batch=bank(raw,mask)
    with pytest.raises(ValueError,match='NOT_RAW_GENE_CHANNELS'):
        replace(batch,values=torch.ones(2,6,64),measured_mask=torch.ones(2,6,64,dtype=torch.bool)).validate()


def test_masked_measured_gradients_and_query_gradients_follow_only_available_edges():
    mapping,bank,raw,mask,_,_,reader,evidence=fixture()
    raw.requires_grad_(True);evidence.requires_grad_(True)
    result=reader(evidence,bank(raw,mask));result['tokens'].square().sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[~mask].abs().sum()==0
    assert raw.grad[:,0].abs().sum()==0  # background has no program edge
    assert raw.grad[mask].abs().sum()>0 and evidence.grad.abs().sum()>0
    assert reader.query_projection.weight.grad.abs().sum()>0


def test_invalid_membership_and_raw_gene_contracts_are_rejected():
    with pytest.raises(ValueError,match='CONFLICTING'):
        mapping_from_programs([dict(id='p',up_genes=['A'],down_genes=['A'])])
    with pytest.raises(ValueError,match='SIGN'):
        SignedProgramMapping(('p',),('A',),torch.tensor([0]),torch.tensor([0]),torch.tensor([0]))
    with pytest.raises(ValueError,match='UNIQUE'):
        GeneValueBank(['A','A'],['A'])
    bank=GeneValueBank(['A'],['A'])
    with pytest.raises(ValueError,match='MATRIX_OR_MEASUREMENT_MASK'):
        bank(torch.ones(2,4,64),torch.ones(2,4,64,dtype=torch.bool))
