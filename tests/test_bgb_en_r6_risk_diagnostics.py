"""Anonymous synthetic receipts only; no target labels, models or optimization."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/bgb_en_r6_risk_diagnostics.py'
spec=importlib.util.spec_from_file_location('r6_risk_metadata_test',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def score(value=.6,den=100.,repair=.1,damage=.05):
    return dict(Harrell=value,Uno60=value,Harrell_num=value*den,Harrell_den=den,
        Uno_num=value*den*2,Uno_den=den*2,repair=repair,damage=damage,
        Uno_repair=repair,Uno_damage=damage,risk_shift_sd_ratio=.025,
        secret_path='PRIVATE_SHOULD_NOT_EXPORT',risk=[1234.])


def describe(sd,avg=0.):
    return dict(n=8,mean=avg,sd=sd,min=-sd,q10=-sd/2,median=0.,q90=sd/2,max=sd,
                subject_identifiers=['PRIVATE_SHOULD_NOT_EXPORT'])


def raw_receipt(arm,part,seed):
    active=list(module.GROUPS)
    if arm in ('B2','B3'):
        active=[x for x in active if x not in module.CROSS_ONLY_GROUPS]
    elif arm in ('B4','B11'):
        active.remove('core.gene_rereader')
    paths={}
    for path in ('cross_delta','risk'):
        paths[path]={name:dict(tensors=2,nonzero=1,L2_squared=4. if path=='cross_delta' else 9.,
            parameter_values=['PRIVATE_SHOULD_NOT_EXPORT']) for name in active}
    magnitudes={name:describe(.25 if name=='delta' else .1 if name=='cross_delta' else .2 if name=='main_delta' else .03)
                for name in module.SCALARS}
    curves=[]
    for epoch in module.EPOCHS:
        # Unequal support catches an incorrect mean of per-partition C.
        den=(10.,30.,100.)[part]
        value=(.2,.4,.8)[part]+(seed-31)/1000.
        curves.append(dict(epoch=epoch,optimizer_updates_completed=epoch,cox_loss=1./epoch,
            parameter_regularizer=.01,total_risk_penalty=.02,cross_risk_penalty=.003,gradient_norm=.4,
            risk_cache_replay_max=0.,cross_cache_replay_max=0.,
            train_SD_delta_over_offset=.1+epoch/1000,valid_SD_delta_over_offset=.2+epoch/1000,
            train=score(value,den),valid=score(value,den),
            hidden_risk_arrays=['PRIVATE_SHOULD_NOT_EXPORT'],source_origin={'path':'PRIVATE_SHOULD_NOT_EXPORT'}))
    controls={name:dict(risk_SD_ratio=.01,frozen_not_retrained=True,whole_modality_replaced=name=='whole_omics',
        metrics_vs_base=score(.55),relative_to_trained=score(.6),
        original_query_sources=['PRIVATE_SHOULD_NOT_EXPORT']) for name in module.CONTROLS}
    return dict(status='COMPLETE_R6_TARGET_FIT',contract={'contract_sha256':'a'*64},arm=arm,partition=part,seed=seed,
        epochs=80,optimizer_updates=80,curves=curves,metrics_vs={'CP_STRONG':score(),'CP_SCALAR':score()},
        risk_diagnostics={split:dict(quantities=magnitudes,SD_total_delta_over_offset=.125,
            SD_cross_delta_over_offset=.05,source_origin='PRIVATE_SHOULD_NOT_EXPORT') for split in ('train','validation')},
        gradient_probe=dict(scope=module.PROBE_SCOPE,patients=8,groups_by_path=paths,new_optimizer_updates=0,
                            raw_gradient=['PRIVATE_SHOULD_NOT_EXPORT']),
        frozen_controls=controls,source_origin={'path':'PRIVATE_SHOULD_NOT_EXPORT'},
        patient_ids=['PRIVATE_SHOULD_NOT_EXPORT'],absolute_path='PRIVATE_SHOULD_NOT_EXPORT')


@pytest.fixture
def corpus(tmp_path):
    cells=tmp_path/'private_cells';ledger=[]
    for arm in module.ARMS:
        for part in module.PARTITIONS:
            for seed in module.SEEDS:
                path=cells/f'{arm}_p{part}_seed{seed}'/'COMPLETE.json'
                dump(path,raw_receipt(arm,part,seed))
                ledger.append(dict(arm=arm,partition=part,seed=seed,status='PASS_FIT_AND_ACTUAL_REPLAY',
                    actual_replay_passed=True,actual_optimizer_updates=80,actual_new_optimizer_fits=1,
                    receipt_sha256=module.sha(path)))
    ledger += [dict(arm=arm,partition=part,seed=seed,status='PASS_ZERO_FIT_REFERENCE',
        actual_replay_passed=True,actual_optimizer_updates=0,actual_new_optimizer_fits=0)
        for arm in ('B0','B1') for part in module.PARTITIONS for seed in module.SEEDS]
    report=dict(schema='bgb_en_r6_target_inner_report_v1',contract_sha256='a'*64,
        actual_replay=dict(complete=True,status='PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY',sha256='c'*64),
        qualification=dict(passed=True,sha256='b'*64),source=dict(complete_source_and_full_transfer=True),ledger=ledger,
        confirmed_new_target_fits=90,confirmed_target_updates=7200,unknown_cost_units=0,direct_pair_audit=dict(complete=True),
        outer_results=dict(status='NOT_RUN'),decision=dict(outer_start_allowed=False,candidate_supported=False),
        paired_comparisons={'locked_negative_comparison':{'delta':-.001}})
    report_path=tmp_path/'R6_4_INNER_MATRIX.json';dump(report_path,report)
    return dict(root=tmp_path,cells=cells,report=report_path,raw_report=report,white=tmp_path/'ANONYMOUS.json')


def test_complete_export_analyze_uses_only_whitelist_and_saved_epochs(corpus):
    envelope=module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    assert len(envelope['units'])==90
    assert 'PRIVATE_SHOULD_NOT_EXPORT' not in corpus['white'].read_text()
    result=module.analyze(corpus['report'],corpus['white'],corpus['root']/'diagnostics')
    assert result['status']=='COMPLETE_SAVED_METADATA_DIAGNOSTICS'
    assert len(result['units'])==90 and len(result['learning_curves'])==30
    assert result['coverage']['main_cross_directional_SD_epochs20_40']=='NOT_SAVED'
    assert result['coverage']['gradient_epochs20_40']=='NOT_SAVED'
    assert all(result[name]==0 for name in ('new_optimizer_fits','new_optimizer_updates','new_model_forwards','new_backward_passes'))
    assert result['epoch_or_scale_or_model_selection'] is False
    assert result['existing_rank_credit_analysis_recomputed_or_replaced'] is False
    b7=next(row for row in result['units'] if row['arm']=='B7')
    assert b7['final_magnitude']['validation']['effective_anchor_scale_inferred']==2.
    assert b7['final_magnitude']['validation']['SD_main_delta_over_anchor']==.1
    assert b7['gradient_probe']['risk']['core.spatial_adapter']['L2']==3.
    assert b7['gradient_probe']['cross_delta']['core.spatial_adapter']['L2']==2.
    assert b7['gradient_probe']['main_gradient_not_inferred_by_subtraction']
    assert b7['gradient_probe']['patients']==8 and b7['gradient_probe']['scope']==module.PROBE_SCOPE
    assert 'main' not in b7['gradient_probe']
    assert b7['saved_scalar_distributions']['validation']['quantities']['cross_delta']['sd']==.1
    assert [row['epoch'] for row in b7['budget_snapshots']]==[20,40,80]
    assert result['interpretations']['step_loss_and_penalties_pre_update_snapshot_metrics_post_update']
    assert b7['frozen_interventions']['wrong_query']['trained_minus_control_Harrell']==pytest.approx(.05)
    assert b7['frozen_interventions']['wrong_query']['trained_minus_control_repair']==.1
    assert next(row for row in result['units'] if row['arm']=='B9')['trained_query_mode']=='wrong_query'
    b2=next(row for row in result['arm_summaries'] if row['arm']=='B2')
    assert b2['gradient_paths']['risk']['core.query_condition']['status']=='STRUCTURAL_NOT_TRAINABLE'
    assert b2['gradient_paths']['risk']['core.query_condition']['L2']['mean'] is None
    output=(corpus['root']/'diagnostics/R6_RISK_DIAGNOSTICS.json').read_text()
    assert 'PRIVATE_SHOULD_NOT_EXPORT' not in output


def test_gate_before_complete_root_io_and_no_output(corpus):
    corpus['raw_report']['actual_replay']['complete']=False;dump(corpus['report'],corpus['raw_report'])
    with pytest.raises(ValueError,match='BEFORE_DIAGNOSTIC_IO'):
        module.export_receipts(corpus['report'],corpus['root']/'not_existing',corpus['white'])
    assert not corpus['white'].exists()


def test_missing_or_duplicate_actual_cell_rejected(corpus):
    report=corpus['raw_report'];report['ledger'][0]=copy.deepcopy(report['ledger'][1]);dump(corpus['report'],report)
    with pytest.raises(ValueError,match='108_LEDGER'):
        module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])


def test_original_complete_byte_changes_rejected(corpus):
    path=corpus['cells']/'B7_p0_seed31/COMPLETE.json'
    path.write_text(path.read_text()+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='ORIGINAL_COMPLETE_SHA_CHANGED'):
        module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])


def test_anonymous_receipt_ledger_and_report_sha_binding(corpus):
    envelope=module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    envelope['units'][0]['original_COMPLETE_sha256']='d'*64;dump(corpus['white'],envelope)
    with pytest.raises(ValueError,match='WHITELIST_COMPLETE_SHA_NOT_IN_LEDGER'):
        module.analyze(corpus['report'],corpus['white'],corpus['root']/'out')
    envelope['units'][0]['original_COMPLETE_sha256']=corpus['raw_report']['ledger'][0]['receipt_sha256']
    envelope['report_sha256']='d'*64;dump(corpus['white'],envelope)
    with pytest.raises(ValueError,match='WHITELIST_ORIGIN_CHANGED'):
        module.analyze(corpus['report'],corpus['white'],corpus['root']/'out')


def test_seed_aggregate_sums_numerators_denominators_not_mean_C(corpus):
    module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    result=module.analyze(corpus['report'],corpus['white'],corpus['root']/'out')
    curve=result['learning_curves'][0]
    expected=sum(c*d for c,d in zip((.2,.4,.8),(10.,30.,100.)))/140.
    assert curve['inner_validation_ranking']['per_seed'][0]['Harrell']==pytest.approx(expected)
    assert curve['inner_validation_ranking']['per_seed'][0]['Harrell']!=pytest.approx((.2+.4+.8)/3)
    expected_seed_mean=expected+sum((s-31)/1000 for s in module.SEEDS)/3
    assert curve['inner_validation_ranking']['Harrell']==pytest.approx(expected_seed_mean)
    assert curve['inner_validation_ranking']['Uno60']==pytest.approx(expected_seed_mean)
    assert all(row['best_epoch_not_selected'] for row in result['learning_curves'])


def test_direct_frozen_repairs_are_trained_minus_control_not_reverse():
    controls={'wrong_query':dict(risk_SD_ratio=.1,frozen_not_retrained=True,whole_modality_replaced=False,
        metrics_vs_base=module.metric(score(.55)),relative_to_trained=module.metric(score(.6,repair=.01,damage=.06)))}
    with pytest.raises(ValueError,match='DIRECT_FROZEN_REPAIR_DAMAGE_IDENTITY_FAILED'):
        module.frozen_values(controls,module.metric(score(.6)))
    controls['wrong_query']['relative_to_trained']=module.metric(score(.7))
    with pytest.raises(ValueError,match='FROZEN_TRAINED_SCORE_CHANGED'):
        module.frozen_values(controls,module.metric(score(.6)))


def test_missing_probe_never_becomes_zero_or_partial_mean(corpus):
    path=corpus['cells']/'B7_p0_seed31/COMPLETE.json';raw=json.loads(path.read_text());raw.pop('gradient_probe');dump(path,raw)
    ledger=next(r for r in corpus['raw_report']['ledger'] if module.key(r)==('B7',0,31));ledger['receipt_sha256']=module.sha(path)
    dump(corpus['report'],corpus['raw_report'])
    module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    result=module.analyze(corpus['report'],corpus['white'],corpus['root']/'out')
    assert result['status']=='PARTIAL_SAVED_METADATA_DIAGNOSTIC_COVERAGE'
    assert result['coverage']['gradient_probe_cells']==89
    arm=next(row for row in result['arm_summaries'] if row['arm']=='B7')
    grad=arm['gradient_paths']['risk']['core.query_condition']
    assert grad['status']=='MISSING_PROBE_COVERAGE' and grad['L2']['available']==8 and grad['L2']['mean'] is None


def test_unknown_anchor_scale_stays_missing_not_zero():
    result=module.final_magnitude(dict(quantities={n:describe(0.) for n in module.SCALARS},
        SD_total_delta_over_offset=0.,SD_cross_delta_over_offset=0.))
    assert result['effective_anchor_scale_inferred'] is None
    assert result['SD_main_delta_over_anchor'] is None
    assert result['SD_cross_over_main'] is None


def test_existing_export_and_diagnostics_never_overwritten(corpus):
    module.export_receipts(corpus['report'],corpus['cells'],corpus['white']);original=corpus['white'].read_bytes()
    with pytest.raises(ValueError,match='EXISTING_OUTPUT_PRESERVED'):
        module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    assert corpus['white'].read_bytes()==original
    output=corpus['root']/'out';module.analyze(corpus['report'],corpus['white'],output)
    result=(output/'R6_RISK_DIAGNOSTICS.json').read_bytes()
    with pytest.raises(ValueError,match='EXISTING_OUTPUT_PRESERVED'):
        module.analyze(corpus['report'],corpus['white'],output)
    assert (output/'R6_RISK_DIAGNOSTICS.json').read_bytes()==result


def test_script_imports_only_stdlib_and_cli_is_runnable(corpus):
    tree=ast.parse(SCRIPT.read_text(encoding='utf-8'))
    imports={n.name.split('.')[0] for row in ast.walk(tree) if isinstance(row,ast.Import) for n in row.names}
    imports|={row.module.split('.')[0] for row in ast.walk(tree) if isinstance(row,ast.ImportFrom)}
    assert not imports & {'torch','numpy','src','scripts'}
    run=subprocess.run([sys.executable,str(SCRIPT),'--mode','export','--report',str(corpus['report']),
        '--complete-root',str(corpus['cells']),'--whitelist',str(corpus['white'])],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['new_optimizer_fits']==0
    run=subprocess.run([sys.executable,str(SCRIPT),'--mode','analyze','--report',str(corpus['report']),
        '--whitelist',str(corpus['white']),'--output',str(corpus['root']/'cli')],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['status']=='COMPLETE_SAVED_METADATA_DIAGNOSTICS'


def test_lossless_arm_split_roundtrip_and_part_tamper_rejection(corpus,monkeypatch):
    module.export_receipts(corpus['report'],corpus['cells'],corpus['white'])
    result=module.analyze(corpus['report'],corpus['white'],corpus['root']/'original')
    assert module.encoded_size(result)<2*1024**2
    # Tight synthetic transport bound forces the same no-field-deletion branch.
    monkeypatch.setattr(module,'PUBLIC_FILE_LIMIT',900000)
    output=corpus['root']/'split';module.write_diagnostic_bundle(output,result)
    index=module.read(output/'R6_RISK_DIAGNOSTICS.json')
    assert index['schema']=='bgb_en_r6_risk_diagnostic_index_v1'
    assert len(index['parts'])==10
    assert all((output/row['file']).stat().st_size<=900000 for row in index['parts'])
    assert module.read_diagnostic_bundle(output/'R6_RISK_DIAGNOSTICS.json')==result
    part=output/index['parts'][0]['file'];part.write_bytes(part.read_bytes()+b' ')
    with pytest.raises(ValueError,match='PART_SHA_CHANGED'):
        module.read_diagnostic_bundle(output/'R6_RISK_DIAGNOSTICS.json')
