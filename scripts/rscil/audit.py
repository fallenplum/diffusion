"""Independent read-only audit and reporting; never trains or selects models."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.rscil.geometry import EfronRiskSets,fit_projection
from src.rscil.protocol import synthetic_gate
from src.rscil.report import write_json


def theory(output):
    receipt=json.loads((output/'SOFTWARE_TEST_LEDGER.json').read_text(encoding='utf-8'))
    if receipt['status']!='PASS':raise ValueError('ACTUAL_SOFTWARE_RECEIPT_REQUIRED')
    supplied=json.loads((output/'reference/supplied_linear_results.json').read_text(encoding='utf-8'))
    rerun=json.loads((output/'reference/recomputed_linear_results.json').read_text(encoding='utf-8'))
    if len(supplied['rows'])!=36 or len(rerun['rows'])!=36:raise ValueError('REFERENCE_COVERAGE')
    reference_error=max(abs(a[k]-b[k]) for a,b in zip(supplied['rows'],rerun['rows']) for k in ('base_C','raw_C','ortho_C'))
    if reference_error>1e-12:raise ValueError('REFERENCE_RANKING_MISMATCH')
    torch.manual_seed(119);x=torch.randn(14,7,dtype=torch.float64)
    t=torch.tensor([1,1,2,2,2,3,4,4,5,5,5,6,7,8],dtype=torch.float64)
    e=torch.tensor([1,0,1,1,0,1,0,1,1,0,1,0,1,0]);beta=torch.arange(7,dtype=torch.float64)*.05
    rs=EfronRiskSets(t,e)
    h=torch.autograd.functional.hessian(lambda p:rs.loss(x@p,normalize=False),beta)
    info=rs.information(x@beta,x);a,d=fit_projection(x[:,:3],x[:,3:],geometry='riskset',risk=x@beta,risksets=rs)
    wb=beta[:3];wz=beta[3:]
    mapping_error=float((x@beta-(x[:,:3]@(wb+a@wz)+(x[:,3:]-x[:,:3]@a)@wz)).abs().max())
    report={'role':'software_math_qualification_not_prediction_evidence','theory_pass':True,
            'Hessian_sum_information_max_abs':float((h-info).abs().max()),
            'information_min_eigenvalue':float(torch.linalg.eigvalsh(info).min()),
            'linear_class_mapping_max_abs':mapping_error,'ridge_identity_max_abs':d['ridge_identity_error'],
            'tests_local':receipt['local']['passed'],'tests_server':receipt['server']['passed'],'tests_cover_reference':True,
            'reference_linear_36_rows_rerun':True,'reference_max_C_difference':reference_error,
            'literal_exact_zero_not_claimed_for_ridge':True}
    if max(report['Hessian_sum_information_max_abs'],mapping_error,d['ridge_identity_error'])>=1e-8:raise ValueError('THEORY_FAIL')
    write_json(output/'THEORY_QUALIFICATION.json',report)
    return report


def table(path,rows):
    if not rows:raise ValueError('EMPTY_REPORT')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit(input_dir,retired_dir,output):
    lock=json.loads((ROOT/'configs/rscil/synthetic_lock.json').read_text(encoding='utf-8'))
    execution_hash=hashlib.sha256((ROOT/'configs/rscil/run_lock.json').read_bytes()).hexdigest()
    expected=set(itertools.product(lock['sample_sizes'],lock['scenarios'],lock['seeds']))
    cells=[json.loads(p.read_text(encoding='utf-8')) for p in input_dir.glob('[0-9]*.json')]
    keys=[(x['rows'][0]['n'],x['rows'][0]['scenario'],x['rows'][0]['seed']) for x in cells]
    if len(cells)!=96 or len(set(keys))!=96 or set(keys)!=expected:raise ValueError('INCOMPLETE_96_CELLS')
    rows=[];curves=defaultdict(list);diagnostic=[];checkpoints=[];updates=0;replays=0;max_replay=0.;warm_max=0.
    for x in cells:
        if x['lock_sha256']!=execution_hash:raise ValueError('SOURCE_BINDING_CHANGED')
        if [r['arm'] for r in x['rows']]!=lock['arms']:raise ValueError('MISSING_ARM')
        if not x['no_TCGA_loaded'] or not x['test_labels_seen_only_in_scoring_after_epoch_lock']:raise ValueError('SCOPE_FAIL')
        for r,a in zip(x['rows'],x['audit']):
            if r['arm']!=a['arm'] or r['selected_epoch'] not in (30,60,90):raise ValueError('SELECTION_RULE_FAIL')
            if abs(r['Harrell']-r['Harrell_num']/r['Harrell_den'])>1e-14:raise ValueError('SCORING_ARITHMETIC_FAIL')
            if abs(r['warm_error'])>=1e-10 or r['disk_replay_max_abs']>1e-12:raise ValueError('REPLAY_OR_WARM_FAIL')
            rows.append(r);warm_max=max(warm_max,r['warm_error']);max_replay=max(max_replay,r['disk_replay_max_abs'])
            for stage in a['stages']:
                if stage['optimizer_updates']!=90 or len(stage['history'])!=90:raise ValueError('TRUNCATED_FIT')
                receipts=stage['all_snapshot_disk_replays']
                if [ss['epoch'] for ss in receipts]!=[0,10,30,60,90]:raise ValueError('MISSING_SNAPSHOT_REPLAY')
                for ss in receipts:
                    err=max(ss['train_max_abs'],ss['evaluation_max_abs']);max_replay=max(max_replay,err)
                    if err>1e-12:raise ValueError('SNAPSHOT_REPLAY_FAIL')
                replays+=len(receipts);updates+=90
                h=stage['history'];coxs=[ss['Cox'] for ss in h];geom=[ss['geometry'] for ss in h if ss['geometry']]
                grad=max(ss['grad_Z_Cox_norm'] for ss in h)
                if r['arm'] in ('S1','S2','S3') and grad<=0:raise ValueError('INTERACTION_GRADIENT_DID_NOT_ACTIVATE')
                if stage['updates']['generator']<80:raise ValueError('GENERATOR_NOT_TRAINED')
                diagnostic.append({'n':r['n'],'scenario':r['scenario'],'seed':r['seed'],'arm':r['arm'],'phase':stage['phase'],
                                   'base_updates':stage['updates']['base'],'generator_updates':stage['updates']['generator'],'output_updates':stage['updates']['output'],
                                   'max_grad_Z_Cox':grad,'first_train_Cox':coxs[0],'last_train_Cox':coxs[-1],
                                   'max_geometry_condition_number':max((q['condition_number'] for q in geom),default=0),
                                   'geometry_evaluations':stage['geometry_evaluations']})
                for ss in h:
                    curves[r['n'],r['scenario'],r['arm']].append({'seed':r['seed'],'phase':stage['phase'],'epoch':ss['epoch'],
                        'train_Cox':ss['Cox'],'objective':ss['objective'],'grad_Z_Cox_norm':ss['grad_Z_Cox_norm'],
                        'main_SD':ss['main_SD'],'interaction_SD':ss['interaction_SD'],
                        'information_condition_number':ss['geometry'].get('condition_number',0),
                        'ridge_identity_error':ss['geometry'].get('ridge_identity_error',0)})
                checkpoints.append({'n':r['n'],'scenario':r['scenario'],'seed':r['seed'],'arm':r['arm'],'phase':stage['phase'],
                                    'snapshots':receipts,'validation_curve':a['validation_curve'] if stage['phase']=='selection' else [],
                                    'selected_epoch':r['selected_epoch'],'selected_tensor_sha256':r['selected_model_tensor_sha256']})
    rows=sorted(rows,key=lambda r:(r['n'],r['scenario'],r['seed'],r['arm']))
    gate=synthetic_gate(rows);software=theory(output)
    decision={'theory_pass':software['theory_pass'],'synthetic_neural_pass':gate['passed'],'inner_pass':None,
              'outer_completed':False,'originality_status':'NOT_ESTABLISHED',
              'claim_status':gate['status'],'stage':'SYNTHETIC_COMPLETED','TCGA_fits':0,'TCGA_endpoint_elements_decoded':0,
              'no_CI_bootstrap_pvalues':True,'larger_N_not_used_to_rescue':True}
    write_json(output/'decision.json',decision);write_json(output/'synthetic_gate.json',gate)
    write_json(output/'synthetic_all_repetitions.json',{'role':'synthetic_neural','rows':rows,'lock_sha256':execution_hash})
    compact=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    table(output/'synthetic_all_repetitions.csv',compact);table(output/'training_effectiveness.csv',diagnostic)
    (output/'curves').mkdir(exist_ok=True)
    for (n,sc,arm),values in curves.items():
        table(output/'curves'/f'{n}_{sc}_{arm}.csv',[v for v in values if v['epoch'] in (1,10,11,30,60,61,90)])
    for n in lock['sample_sizes']:
        write_json(output/f'checkpoint_summaries_{n}.json',{'role':'synthetic_only','max_replay_error':max_replay,
                                                          'fits':[x for x in checkpoints if x['n']==n]})
    retired=[json.loads(p.read_text()) for p in retired_dir.glob('[0-9]*.json')] if retired_dir else []
    old_equal=0;old_max=0.
    lookup={(r['n'],r['scenario'],r['seed'],r['arm']):r for r in rows}
    for cell in retired:
        for r in cell['rows']:
            q=lookup[r['n'],r['scenario'],r['seed'],r['arm']]
            old_equal+=int(r['selected_model_tensor_sha256']==q['selected_model_tensor_sha256'])
            old_max=max(old_max,abs(r['Harrell']-q['Harrell']),abs(r['Cox_NLL']-q['Cox_NLL']))
    ledger={'formal_result_cells':480,'formal_neural_fits_including_selection_and_refit':960,'formal_neural_optimizer_updates':updates,
            'formal_current_fit_linear_warm_teachers':192,'reference_linear_Cox_fits':180,
            'LBFGS_warm_teacher_actual_iteration_count':'not_instrumented;max100_each;not_claimed_as100_actual_updates',
            'all_snapshot_disk_checkpoint_replays':replays,'extra_selected_checkpoint_replays':480,
            'checkpoint_replay_max_abs':max_replay,'warm_max_abs':warm_max,
            'retired_v1_completed_cells':len(retired),'retired_v1_neural_fits':len(retired)*10,'retired_v1_neural_updates':len(retired)*900,
            'retired_vs_v2_selected_tensor_hashes_equal':old_equal,'retired_vs_v2_compared_models':len(retired)*5,
            'retired_vs_v2_max_metric_difference':old_max,'retired_not_used_in_primary_gate':True,
            'software_qualification_neural_fits':10,'software_qualification_neural_updates':300,
            'software_qualification_warm_teacher_fits':15,
            'all_checkpoint_weights_removed_after_replay':True,'TCGA_fits':0}
    write_json(output/'fit_ledger.json',ledger)
    summary=[]
    for n,sc,arm in itertools.product(lock['sample_sizes'],lock['scenarios'],lock['arms']):
        z=[r for r in rows if (r['n'],r['scenario'],r['arm'])==(n,sc,arm)]
        summary.append({'n':n,'scenario':sc,'arm':arm,'repetitions':len(z),'Harrell_mean':float(np.mean([r['Harrell'] for r in z])),
                        'Cox_NLL_mean':float(np.mean([r['Cox_NLL'] for r in z])),
                        'selected_epoch30_count':sum(r['selected_epoch']==30 for r in z),
                        'selected_epoch60_count':sum(r['selected_epoch']==60 for r in z),
                        'selected_epoch90_count':sum(r['selected_epoch']==90 for r in z)})
    table(output/'synthetic_summary.csv',summary)
    print(json.dumps({'decision':decision,'ledger':ledger,'gate':gate},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--retired');p.add_argument('--output',default=str(ROOT/'reports/rscil'));p.add_argument('--theory-only',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.theory_only:theory(out)
    else:audit(Path(a.input),Path(a.retired) if a.retired else None,out)
