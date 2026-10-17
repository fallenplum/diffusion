"""Package complete Phase20 evidence without training, selecting or pushing.

Scientific negatives and budget-unresolved outcomes remain publishable. All
technical completeness, provenance, aggregate-statistic and privacy gates must
pass. Inherited tracked bytes are immutable, including historical reports.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase20_selective.data import P, R, cfg, source, source_files, write
from src.phase20_selective.provenance import require_record, require_attempt, verify_qualified_tests
from src.phase10_trimodal.data import digest
from scripts.package_phase19 import (require, read, safe_path, _json_privacy,
    SENSITIVE_CONTENT, DENIED_FIELDS, MAX_BYTES)
sys.path.insert(0,str(ROOT/'src'))
from reproducibility import protected_path_violations

BRANCH = 'research/phase20-selective-capacity-audit'
PARENT = '065a6a8dea4f181f5b6e9fb492ed3ebb56f6c68c'
PARENT_REPORT = 'reports/phase19_clinical_trilinear_attention'
LEGACY_SOURCE = 'd1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192'
PRE_HISTORY_EVALUATOR = '05d081ee25c1bd6fac58781eacf4c7b0f4005e24682d234c84940a362086ded3'
ARMS = ['L0','P1','P16','S1','S16','D16','F16','WC16','WO16']
METRICS = ['Harrell','Uno60']
EXTENSIONS = {'.py','.json','.md','.csv'}
SCRIPT_NAMES = {name+'_phase20.py' for name in ['prepare','run','schedule','start','validate','report','repair','package','publish']}
SCRIPT_NAMES.add('audit_phase20_history_reference.py')
GATES = ['DATA_AUDIT','DATA_LEAKAGE_AUDIT','MODEL_QUALIFICATION','OPERATOR_AND_PIPELINE_TESTS',
    'EVALUATION_MATH_AUDIT','INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','BOOTSTRAP_AUDIT',
    'CONTRAST_POINT_AUDIT','BUDGET_VERIFICATION','FIT_AND_REFERENCE_COUNTS','PROBE_COVERAGE_AUDIT',
    'SELECTIVITY_BEHAVIOR_AUDIT','EXECUTION_COUNTS','REPORT_INPUT_AUDIT',
    'AUDIT_REPAIR_COMPATIBILITY','AUDIT_REPAIR_TESTS','HISTORICAL_REFERENCE_REPAIR_AUDIT']


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def allowed_path(relative):
    p = PurePosixPath(relative)
    if p.parts[:2] == ('src','phase20_selective'):
        return len(p.parts)==3 and p.suffix=='.py'
    if p.parts[:1] == ('configs',):
        return relative == 'configs/phase20_selective_attention.json'
    if p.parts[:1] == ('scripts',):
        return len(p.parts)==2 and p.name in SCRIPT_NAMES
    if p.parts[:1] == ('tests',):
        return len(p.parts)==2 and p.name.startswith('test_phase20_') and p.suffix=='.py'
    return p.is_relative_to(PurePosixPath(R.relative_to(ROOT).as_posix())) and p.suffix in EXTENSIONS


def _audit_public_json(value,relative):
    """Exempt only two literal formulas at their exact prelocked locations."""
    value=copy.deepcopy(value)
    container=None
    if relative=='configs/phase20_selective_attention.json':
        container=value
    elif relative==R.relative_to(ROOT).as_posix()+'/PROTOCOL_LOCK.json':
        require(isinstance(value,dict) and isinstance(value.get('configuration'),dict),
                'missing exact original protocol configuration')
        container=value['configuration']
    if container is not None:
        formulas={
            'base':'anchor_sd*tanh(base_head(matched_base_messages))',
            'selective':'.25*anchor_sd*tanh(bias_free_Linear(concatenated_delta_messages)); no trainable multiplier or further head after this cap'}
        for section,expected in formulas.items():
            require(isinstance(container.get(section),dict) and container[section].get('risk')==expected
                    and 'risk_formula' not in container[section],'unrecognized configuration risk formula')
            container[section]['risk_formula']=container[section].pop('risk')
    _json_privacy(value)


def audit_public_file(path):
    relative = path.relative_to(ROOT).as_posix()
    require(allowed_path(relative),'file outside explicit Phase20 allowlist: '+relative)
    safe_path(relative)
    require(path.is_file() and path.suffix in EXTENSIONS and path.stat().st_size < MAX_BYTES,
            'unsupported or oversized public artifact: '+relative)
    require(not protected_path_violations([relative]),'protected public path: '+relative)
    require(not re.search(r'(?:OOF_PRIVATE|raw_predictions|patient_predictions|selected_genes|model_state|private_cache)',
                          path.name,re.IGNORECASE),'private artifact filename: '+relative)
    raw = path.read_bytes()
    require(not SENSITIVE_CONTENT.search(raw),'sensitive content detected: '+relative)
    text = raw.decode('utf-8-sig')
    if path.suffix=='.json':
        _audit_public_json(json.loads(text),relative)
    elif path.suffix=='.csv':
        columns = next(csv.reader(text.splitlines()),[])
        require(not {c.strip().lower() for c in columns}&DENIED_FIELDS,'patient-level CSV field: '+relative)
    return {'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def _frame(name):
    return pd.read_csv(R/(name+'.csv'))


def _keys(frame, fields):
    require(not frame.duplicated(fields).any(),'duplicate aggregate cell identity')
    return set(frame[fields].itertuples(index=False,name=None))


def execution_bounds(formal_attempts, smoke_real, smoke_steps):
    """Committed callbacks are lower bounds for killed attempts, never zero-fill.

    A serial optimizer callback follows each update. A killed worker can have
    at most one additional update before its next persisted callback. This
    uncertainty changes the positive-fit count only for zero-committed records.
    """
    positive=sum(int(q.get('generator_steps',0))>0 for q in formal_attempts)
    committed=sum(int(q.get('generator_steps',0)) for q in formal_attempts)
    interrupted=[q for q in formal_attempts if q['status'] not in ['RETURNED_SUCCESSFULLY','FAILED']]
    zero=sum(int(q.get('generator_steps',0))==0 for q in interrupted)
    return {
        'formal_zero_committed_step_interrupted_attempts':zero,
        'formal_real_fit_attempts_lower_bound':positive,
        'formal_real_fit_attempts_upper_bound':positive+zero,
        'formal_interrupted_attempts_with_step_commit_uncertainty':len(interrupted),
        'formal_all_attempt_generator_optimizer_steps_lower_bound':committed,
        'formal_all_attempt_generator_optimizer_steps_upper_bound':committed+len(interrupted),
        'all_real_neural_fits_lower_bound':positive+smoke_real,
        'all_real_neural_fits_upper_bound':positive+smoke_real+zero,
        'all_recorded_generator_optimizer_steps_lower_bound':committed+smoke_steps,
        'all_recorded_generator_optimizer_steps_upper_bound':committed+smoke_steps+len(interrupted)}


def gpu_summary(samples,role):
    """Recompute observed-interval telemetry; offline gaps remain unknown."""
    ordered=sorted(samples,key=lambda q:float(q['unix_time']))
    require(len(ordered)>1,'insufficient GPU samples')
    times=np.asarray([q['unix_time'] for q in ordered],float)
    gaps=np.diff(times)
    require(np.isfinite(times).all() and (gaps>=0).all() and gaps.sum()>0,'invalid GPU sample times')
    observed=(gaps>0)&(gaps<=60.)
    unknown=gaps>60.
    require(observed.any(),'no observed GPU interval within 60 seconds')
    seconds=float(gaps[observed].sum())
    result={'role':role,'samples':len(ordered),'sampled_interval_seconds':seconds,
        'wall_clock_span_seconds':float(gaps.sum()),'observed_interval_count':int(observed.sum()),
        'maximum_counted_interval_seconds':60.,'excluded_unknown_gap_count':int(unknown.sum()),
        'excluded_unknown_gap_seconds':float(gaps[unknown].sum()),'zero_length_interval_count':int((gaps==0).sum()),
        'maximum_running_workers':int(max(q['running'] for q in ordered))}
    for key in ['GPU_percent','memory_MiB']:
        values=np.asarray([q[key] for q in ordered],float)
        require(np.isfinite(values).all(),'nonfinite GPU telemetry')
        result[key+'_time_weighted_mean']=float(np.sum(values[:-1][observed]*gaps[observed])/seconds)
        result[key+'_maximum']=float(values.max())
    return result


def _verify_gpu_profile():
    summary=read(R/'GPU_PROFILE.json');frame=_frame('GPU_PROFILE')
    require(summary['status']=='PASS' and len(summary['rows'])==2 and len(frame)==2
            and {q['role'] for q in summary['rows']}=={'inner','outer'} and set(frame.role)=={'inner','outer'},
            'complete anonymous GPU summaries required')
    for role in ['inner','outer']:
        expected=gpu_summary(read(P/(role+'_GPU.json')),role)
        public=next(q for q in summary['rows'] if q['role']==role)
        csv_row=frame[frame.role==role].iloc[0]
        for key,value in expected.items():
            if key=='role':
                continue
            require(np.isfinite([public[key],csv_row[key]]).all()
                    and np.isclose(public[key],value,rtol=1e-10,atol=1e-10)
                    and np.isclose(csv_row[key],value,rtol=1e-10,atol=1e-10),
                    'GPU summary differs from observed-only intervals: '+role+'.'+key)


def _verify_historical_reference_repair(audit, math_audit, identity, evaluator):
    """Bind a historical-only tolerance repair to actual tests and frozen data."""
    archive=P/'recovery/history_reference_evaluation'
    snapshot=read(archive/'SNAPSHOT.json')
    require(audit['source']==identity and audit['evaluator_before_sha256']==PRE_HISTORY_EVALUATOR
            and audit['evaluator_after_sha256']==evaluator and evaluator!=PRE_HISTORY_EVALUATOR
            and snapshot['source']==identity and snapshot['original_evaluator_sha256']==PRE_HISTORY_EVALUATOR,
            'historical repair evaluator/source chain differs')
    require(audit['audit_program_sha256']==digest(ROOT/'scripts/audit_phase20_history_reference.py')
            and audit['math_audit_sha256']==digest(R/'EVALUATION_MATH_AUDIT.json'),
            'historical repair qualification program/math receipt changed')
    require(snapshot['training_source_files']==source_files()
            and snapshot['repair_lock_sha256']==digest(R/'AUDIT_REPAIR_LOCK.json')
            and snapshot['training_qualification_sha256']==digest(R/'AUDIT_REPAIR_TESTS.json')
            and audit['training_source_files_unchanged'] is True
            and audit['original_math_test_bytes_unchanged'] is True
            and audit['current_risks_and_historical_OOF_modified'] is False
            and audit['new_training_fits']==audit['new_prediction_replays']==0,
            'historical-only repair changed frozen training qualification or predictions')
    evidence=audit['original_failure_evidence']
    expected_evidence={'evaluator_before.py','math_audit_before.json','independent_failure_before.log',
                       'pipeline_failure_before.log','pipeline_failed_before.json'}
    require(set(evidence)==expected_evidence and set(snapshot['files'])==expected_evidence,
            'original historical-reference failure evidence incomplete')
    for name,expected in evidence.items():
        require(digest(archive/name)==expected==snapshot['files'][name]['sha256'],
                'original evaluator/failure archive changed: '+name)
    require(digest(archive/'evaluator_before.py')==PRE_HISTORY_EVALUATOR,
            'archived pre-repair evaluator identity differs')
    old_math=read(archive/'math_audit_before.json')
    math_sha=digest(ROOT/'tests/test_phase20_evaluation.py')
    history_sha=digest(ROOT/'tests/test_phase20_history_reference.py')
    require(old_math['evaluator_sha256']==PRE_HISTORY_EVALUATOR and old_math['server_actual_tests']==7
            and old_math['failures_errors_skips']==0 and old_math['test_source_sha256']==math_sha
            and snapshot['original_math_test_sha256']==audit['original_math_test_sha256']==math_sha
            and audit['history_regression_test_sha256']==history_sha,
            'original seven tests or separate historical regression source changed')
    require(audit['actual_math_tests']==7 and audit['actual_history_regression_tests']==5
            and audit['actual_total_evaluation_tests']==12 and audit['failures_errors_skips']==0
            and math_audit['historical_reference_regression_tests']==5
            and math_audit['historical_reference_test_sha256']==history_sha
            and math_audit['total_actual_evaluation_tests']==12,
            'actual seven mathematical plus five history test evidence incomplete')
    require(set(audit['actual_test_runs'])=={'math','history'},'actual test suites differ')
    for label,count,test_sha in [('math',7,math_sha),('history',5,history_sha)]:
        item=audit['actual_test_runs'][label]
        junit=archive/(label+'_actual_tests.xml')
        require(item['actual_tests']==count and item['failures_errors_skips']==0
                and item['test_source_sha256']==test_sha and item['junit_sha256']==digest(junit),
                'actual history-repair test receipt changed')
        cases=list(ET.parse(junit).iter('testcase'))
        require(len(cases)==count and not any(list(case) for case in cases),
                'history-repair junit has a failure, skip, error or wrong case count')
    require(math_audit['junit_sha256']==audit['actual_test_runs']['math']['junit_sha256'],
            'fresh mathematical receipt and actual junit differ')
    before=ast.parse((archive/'evaluator_before.py').read_text(encoding='utf-8'))
    after=ast.parse((ROOT/'scripts/validate_phase20.py').read_text(encoding='utf-8'))
    functions=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    old,new=functions(before),functions(after)
    require(set(new)-set(old)=={'historical_reference_identity'} and not set(old)-set(new)
            and all(ast.dump(old[name])==ast.dump(new[name]) for name in old if name!='collect'),
            'historical-only repair modified existing replay/primary evaluation functions')
    def without_history(node):
        node=copy.deepcopy(node)
        node.body=[part for part in node.body if not (isinstance(part,ast.For)
            and any(isinstance(child,ast.Name) and child.id=='historic' for child in ast.walk(part)))]
        return ast.dump(node)
    require(without_history(old['collect'])==without_history(new['collect']),
            'collector changed outside the historical-reference loop')
    other=lambda tree:[ast.dump(node) for node in tree.body if not isinstance(node,ast.FunctionDef)]
    require(other(before)==other(after),'historical repair changed module constants/imports')
    ast_fields=['all_preexisting_functions_except_collect_unchanged','collect_outside_historical_loop_unchanged',
        'replay_AST_identical','primary_metrics_bootstrap_and_contrasts_AST_identical','module_constants_imports_unchanged']
    require(all(audit['AST'][key] is True for key in ast_fields)
            and audit['AST']['only_added_function']=='historical_reference_identity','published AST proof differs')
    preserved=audit['preserved_replay']
    require(preserved['checkpoint_count']==270 and preserved['prediction_vectors']==1740 and preserved['maximum_error']==0
            and preserved['new_forward_calls']==preserved['new_training_fits']==0,'saved current-model replay proof differs')
    replay_paths={}
    for i in range(4):
        paths=[P/f'REPLAY_SHARD_{i}.json']+[R/f'{name}_{i}.csv'
                for name in ['CHECKPOINT_REPLAY','FEATURE_RECONSTRUCTION','CONVEX_REFERENCE_AUDIT']]
        replay_paths.update({path.name:path for path in paths})
    require(set(preserved['files'])==set(replay_paths),'saved 16 replay artifacts incomplete')
    require(all(digest(path)==preserved['files'][name] for name,path in replay_paths.items()),
            'saved replay artifacts changed during statistics-only recovery')
    base=Path('/root/phase16r_light_rna_private')
    require(digest(base/'OOF_PRIVATE.npz')==audit['original_historical_OOF_sha256'],
            'original historical OOF cache was replaced or cast')
    rule=audit['published_reference_rule']
    require(rule['absolute_tolerance']==1e-9
            and rule['Phase19_validator_sha256']==digest(ROOT/'scripts/validate_phase19.py')
            and rule['Phase16R_independent_validator_sha256']==digest(ROOT/'scripts/validate_phase16r_light_rna.py'),
            'historical rule does not cite the unchanged published evaluators')
    cells=pd.DataFrame(audit['historical_cells'])
    expected={(s,f) for s in [31,47,83] for f in [1,2,3,4,5]}
    require(audit['historical_cell_count']==15 and len(cells)==15 and _keys(cells,['seed','fold'])==expected,
            'historical-reference audit must cover all 15 original cells')
    published=pd.read_csv(ROOT/'reports/phase19_clinical_trilinear_attention/HISTORICAL_LOCAL_REFERENCE_AUDIT.csv').set_index(['seed','fold'])
    current=_frame('HISTORICAL_LOCAL_REFERENCE_AUDIT').set_index(['seed','fold'])
    require(len(current)==15 and set(current.index)==expected,'collected historical-reference table incomplete')
    boolean_fields=['patient_fold_elementwise_identical','shape_identical','finite','all_pair_strict_order_identical',
                    'all_pair_tie_classification_identical','matches_published_Phase19_audit']
    for row in cells.to_dict('records'):
        key=(int(row['seed']),int(row['fold']))
        checkpoint=base/'artifacts'/f'seed{key[0]}_fold{key[1]}.pt'
        receipt=read(base/'cells'/f'seed{key[0]}_fold{key[1]}.json')
        require(all(row[field] is True for field in boolean_fields) and row['OOF_values_changed_or_cast'] is False
                and row['stored_dtype']==row['OOF_dtype']=='float64'
                and row['published_Phase19_replay_absolute_tolerance']==1e-9 and row['pair_tie_tolerance']==1e-8
                and 0<=row['maximum_error']<=1e-9
                and row['artifact_sha256']==row['receipt_artifact_sha256']==receipt['sha256']==digest(checkpoint),
                'historical cell identity, original dtype, ranking or replay bound differs')
        for table in [published,current]:
            value=table.loc[key]
            require(value.artifact_sha256==row['artifact_sha256'] and abs(value.maximum_error-row['maximum_error'])<1e-24,
                    'published/collected historical cell differs from qualified repair proof')
    require(abs(audit['maximum_historical_reference_error']-float(cells.maximum_error.max()))<1e-24,
            'maximum historical replay error differs from actual cells')


def _verify_metrics_decision(config,identity,gates):
    folds,seeds = _frame('FOLD_METRICS'),_frame('SEED_METRICS')
    require(len(folds)==300 and len(seeds)==60,'complete 60-seed-vector/300-fold matrix required')
    labels = set(ARMS+['J0','J_CS','HIST_CS_S32_CONCH32']+
        ['S16__'+q for q in ['WRONG_C','WRONG_O','CONSTANT_C','CORRECTION_OFF']]+
        [a+'__'+q for a in ['P1','P16'] for q in ['FULL_OMICS_WRONG','UNIFORM_PAIR']])
    require(_keys(seeds,['arm','seed'])=={(a,s) for a in labels for s in config['seeds']},'OOF label/seed coverage differs')
    require(_keys(folds,['arm','seed','fold'])=={(a,s,f) for a in labels for s in config['seeds'] for f in config['folds']},
            'OOF label/seed/fold coverage differs')
    for (arm,seed),group in folds.groupby(['arm','seed']):
        require(group.patients.sum()==512 and group.events.sum()==65,'OOF patient/event coverage differs')
        row = seeds[(seeds.arm==arm)&(seeds.seed==seed)].iloc[0]
        require(abs(group.H_num.sum()/group.H_den.sum()-row.Harrell)<1e-12
                and abs(group.U_num.sum()/group.U_den.sum()-row.Uno60)<1e-12,'per-seed num/den aggregation differs')
    for prefix in ['PFI_']:
        require(_keys(_frame(prefix+'SEED_METRICS'),['arm','seed'])==_keys(seeds,['arm','seed'])
                and _keys(_frame(prefix+'FOLD_METRICS'),['arm','seed','fold'])==_keys(folds,['arm','seed','fold']),
                'frozen secondary-endpoint transfer matrix incomplete')
    paired,perseed = _frame('PAIRED_COMPARISONS'),_frame('SEED_PAIRED_CONTRASTS')
    families = {
        'primary':(config['primary_contrast_coefficients'],14),
        'secondary':({q:{q.split('-',1)[0]:1,q.split('-',1)[1]:-1} for q in config['secondary_comparisons']},16),
        'exploratory_representation':(config['exploratory_representation_contrast_coefficients'],10)}
    lookup = seeds.set_index(['arm','seed'])
    fold_lookup = folds.set_index(['arm','seed','fold'])
    for family,(contrasts,size) in families.items():
        rows = paired[paired.family==family]
        require(_keys(rows,['comparison','metric'])=={(c,m) for c in contrasts for m in METRICS}
                and set(rows.family_tests)=={size},'paired family differs: '+family)
        sr = perseed[perseed.family==family]
        require(_keys(sr,['comparison','metric','seed'])=={(c,m,s) for c in contrasts for m in METRICS for s in config['seeds']},
                'per-seed contrast family differs')
        for row in rows.itertuples(index=False):
            expected=[]
            for seed in config['seeds']:
                value=sum(w*lookup.loc[(a,seed),row.metric] for a,w in contrasts[row.comparison].items())
                saved=sr[(sr.comparison==row.comparison)&(sr.metric==row.metric)&(sr.seed==seed)].iloc[0]
                require(abs(value-saved.delta)<1e-11,'per-seed statistic differs from its predictions')
                expected.append(value)
            require(abs(np.mean(expected)-row.delta)<1e-11 and int((np.asarray(expected)>0).sum())==row.positive_seeds,
                    'paired point/seed statistic differs')
            fold_values=[sum(w*fold_lookup.loc[(a,s,f),row.metric] for a,w in contrasts[row.comparison].items())
                         for s in config['seeds'] for f in config['folds']]
            require(row.fold_cells==15 and row.positive_fold_cells==int((np.asarray(fold_values)>0).sum()),
                    'reported fold-cell direction differs from paired fold predictions')
            require(row.draws_requested==20000 and row.draws_valid==gates['BOOTSTRAP_AUDIT']['valid']
                    and np.isfinite([row.delta,row.CI95_low,row.CI95_high,row.Bonferroni_low,row.Bonferroni_high]).all()
                    and row.Bonferroni_low<=row.CI95_low<=row.CI95_high<=row.Bonferroni_high,
                    'paired interval or resampling object invalid')
    decision=read(R/'decision.json');primary=paired[paired.family=='primary']
    require(decision['source']==identity and decision['primary_family_tests']==14
            and decision['independent_external_confirmation'] is False,'decision provenance differs')
    caps={q['arm']:q['budget_gate_pass'] for q in gates['BUDGET_VERIFICATION']['rows']}
    groups={
        'representation':(['P16-P1'],['P1','P16']),
        'selective_clinical_correction':(['S16-'+a for a in ['P16','D16','F16','WC16','WO16']],['P16','S16','D16','F16','WC16','WO16']),
        'representation_by_correction_interaction':([config['primary_contrasts'][-1]],['P1','P16','S1','S16'])}
    for name,(comparisons,arms) in groups.items():
        q=primary[primary.comparison.isin(comparisons)]
        direction=bool(((q.delta>0)&(q.positive_seeds>=2)).all())
        corrected=bool((q.Bonferroni_low>0).all());sufficient=all(caps[a] for a in arms)
        status=('BUDGET_UNRESOLVED' if not sufficient else 'WITHIN_COHORT_CORRECTED_SUPPORT' if direction and corrected
                else 'DIRECTIONAL_CANDIDATE' if direction else 'NOT_SUPPORTED_IN_THIS_ARCHITECTURE')
        axis=decision['axes'][name]
        if name=='selective_clinical_correction':
            require(axis['predictive_contrast_status']==status,'predictive A status differs')
            behavior=gates['SELECTIVITY_BEHAVIOR_AUDIT']['behavior_gate_pass']
            require(axis['selectivity_behavior_gate_pass']==behavior,'selectivity behavior status differs')
            if sufficient and not behavior:
                status='CONDITIONED_MODULATION_CANDIDATE_WITHOUT_SPARSE_SELECTION' if direction else 'SELECTIVE_MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE'
        require(axis['status']==status and axis['budget_sufficient']==sufficient
                and axis['direction_and_seed_gate']==direction and axis['all_corrected_intervals_positive']==corrected,
                'decision axis differs from locked measured gates: '+name)
    require(decision['status']==decision['axes']['selective_clinical_correction']['status'],'overall decision changed')
    means=seeds.groupby('arm')[METRICS].mean()
    no_harm=all(means.loc['S16',m]>=means.loc[a,m] for a in ['J0','J_CS'] for m in METRICS)
    utility=decision['practical_prediction_value']
    require(utility['all_J0_J_CS_mean_no_harm']==no_harm and utility['status']==(
        'MEAN_NO_HARM_CANDIDATE' if no_harm else 'NO_FINAL_PREDICTIVE_INCREMENT_DEMONSTRATED'),
        'practical axis differs from actual strong-reference means')
    for q in utility['checks']:
        require(abs(q['delta_S16_minus_reference']-(means.loc['S16',q['metric']]-means.loc[q['reference'],q['metric']]))<1e-11,
                'practical contrast point differs')


def verify_completion():
    config,identity=cfg(),source()
    require(config['branch']==BRANCH and config['parent_commit']==PARENT and config['arms']==ARMS
            and config['seeds']==[31,47,83] and config['folds']==[1,2,3,4,5],'locked experiment matrix differs')
    inventory_path=P/'ANCHOR_AUDIT_REPAIR_COMPATIBILITY.json'
    inventory=read(inventory_path)
    require(inventory['status']=='PASS' and inventory['source']==identity and inventory['legacy_source']==LEGACY_SOURCE
            and identity!=LEGACY_SOURCE,'missing/stale explicit D1 to D2 compatibility inventory')
    done,prep,lock=require_record(P/'TRAINING_DONE.json'),require_record(P/'PREPARATION_DONE.json'),require_record(R/'PROTOCOL_LOCK.json')
    require(done['status']==prep['status']=='PASS' and done['source']==identity
            and prep['source']==lock['source']==LEGACY_SOURCE,
            'training/preparation/source incomplete')
    require(lock['status']=='LOCKED_BEFORE_FIRST_FORMAL_TRAINING' and lock['configuration']==config
            and lock['training_files']==inventory['legacy_source_files'],'original D1 protocol/source lock changed')
    repair_lock=require_record(R/'AUDIT_REPAIR_LOCK.json')
    require(repair_lock['status']=='LOCKED_BEFORE_RESTART_AFTER_AUDIT_ONLY_REPAIR'
            and repair_lock['source']==identity and repair_lock['training_files']==source_files()
            and repair_lock['compatibility_audit_sha256']==digest(R/'AUDIT_REPAIR_COMPATIBILITY.json')
            and repair_lock['qualification_sha256']==digest(R/'AUDIT_REPAIR_TESTS.json'),
            'audit-only D2 repair was not explicitly locked before restart')
    for key,value in [('new_inner_fits',135),('new_outer_fits',135),('new_formal_fits',270),
            ('new_unique_convex_fits',0),('qualified_inherited_unique_convex_models',20),('old_neural_checkpoints_used',0)]:
        require(done[key]==value,'formal completion count differs: '+key)
    require(prep['new_convex_solver_calls']==0 and prep['qualified_inherited_unique_convex_models']==20,
            'preparation refitted supposedly inherited convex models')
    inherited_hashes=read(P/'INHERITED_FILE_HASHES.json')
    require(inherited_hashes['status']=='PASS' and inherited_hashes['files']
            and all(digest(path)==expected for path,expected in inherited_hashes['files'].items()),
            'read-only inherited preprocessing/convex artifacts changed after preparation')
    gates={}
    for name in GATES:
        item=require_record(R/(name+'.json'))
        require(item['status']=='PASS',
                'missing/stale technical gate: '+name)
        gates[name]=item
    verify_qualified_tests(gates['OPERATOR_AND_PIPELINE_TESTS'])
    repair=gates['AUDIT_REPAIR_COMPATIBILITY']
    require(repair['source']==identity and repair['inventory_sha256']==digest(inventory_path)
            and repair['same_phase_retained_checkpoint_count']==189 and repair['retained_inner']==135
            and repair['retained_outer']==54 and repair['formal_nonanchor_prediction_vectors_checked']==1029
            and repair['bitwise_equal'] is True,'D1 retention/nonanchor compatibility proof incomplete')
    math=gates['EVALUATION_MATH_AUDIT'];evaluator=digest(ROOT/'scripts/validate_phase20.py')
    require(math['server_actual_tests']==7 and math['failures_errors_skips']==0
            and math['evaluator_sha256']==evaluator and math['test_source_sha256']==digest(ROOT/'tests/test_phase20_evaluation.py'),
            'independent actual math test evidence differs')
    _verify_historical_reference_repair(gates['HISTORICAL_REFERENCE_REPAIR_AUDIT'],math,identity,evaluator)
    evaluation=read(P/'EVALUATION_DONE.json')
    require(evaluation['status']=='PASS' and evaluation['source']==identity and evaluation['evaluator_sha256']==evaluator
            and evaluation['actual_new_checkpoint_replays']==270 and evaluation['bootstrap_draws']==20000,
            'independent evaluation pipeline receipt missing/stale')
    for key,value in [('context_modes_per_checkpoint',5),('additional_anchor_replays',270),
            ('additional_representation_replays',120),('actual_total_prediction_vectors_replayed',1740),
            ('model_seed_vectors',60),('outer_fold_metric_rows',300),
            ('retained_pre_audit_repair_checkpoints',189),('post_audit_repair_checkpoints',81)]:
        require(evaluation[key]==value,'evaluation replay/count evidence differs: '+key)
    require(evaluation['test_source_sha256']==digest(ROOT/'tests/test_phase20_evaluation.py'),
            'evaluation used a different independent test source')
    require(gates['DATA_AUDIT']['patients']==512 and gates['DATA_AUDIT']['DSS_events']==65
            and gates['DATA_AUDIT']['new_feature_or_convex_fits']==0,'cohort or read-only reuse differs')
    valid=gates['INDEPENDENT_VALIDATION']
    require(valid['model_seed_vectors']==60 and valid['outer_fold_metric_rows']==300 and valid['primary_tests']==14
            and valid['independent_sksurv_Harrell_Uno60_each_fold'] is True and valid['risk_ensembling'] is False
            and valid['previous_neural_models_counted_as_current_OOF'] is False
            and valid['retained_pre_audit_repair_checkpoints']==189 and valid['post_audit_repair_checkpoints']==81,
            'OOF or mixed-source metric qualification incomplete')
    replay=gates['CHECKPOINT_REPLAY']
    require(replay['new_neural_checkpoints']==270 and replay['modes_per_checkpoint']==5
            and replay['additional_anchor_replays']==270 and replay['additional_representation_replays']==120
            and replay['maximum_error']<2e-6 and replay['anchor_elementwise_identical'] is True
            and replay['constantC_equals_correction_off'] is True and replay['frozen_main_buffer'] is True,
            '1740 prediction-vector replays or exact anchor qualification incomplete')
    boot=gates['BOOTSTRAP_AUDIT'];validdraws=boot['valid']
    require(boot['draws']==20000 and 19800<=validdraws<=20000 and boot['zero_denominator_draws']==20000-validdraws
            and boot['primary_family_tests']==14 and boot['risk_ensembling'] is False,'shared-patient Bonferroni14 bootstrap incomplete')
    matrices=boot['shared_patient_count_matrices']
    require(len(matrices)==5 and {q['fold'] for q in matrices}==set(config['folds'])
            and all(re.fullmatch('[0-9a-f]{64}',q['counts_sha256']) for q in matrices),'five shared resampling-matrix identities required')
    require(gates['CONTRAST_POINT_AUDIT']['primary_family_tests']==14,'point/bootstrap statistical object differs')
    fit=gates['FIT_AND_REFERENCE_COUNTS']
    for key,value in [('new_inner_neural_fits',135),('new_outer_neural_fits',135),('new_unique_convex_models',0),
            ('qualified_inherited_unique_convex_models',20),('convex_seed_role_references',60),
            ('historical_LOCAL_outer_references',15),('previous_neural_models_counted_as_new_fit_or_OOF',0),
            ('new_native_method_fits',0),('new_PCA_or_KMeans_fits',0),
            ('retained_pre_audit_repair_formal_fits',189),('post_audit_repair_formal_fits',81)]:
        require(fit[key]==value,'fit/reference count differs: '+key)
    require(fit['actual_fit_source_counts']=={LEGACY_SOURCE:189,identity:81},'fit-count original sources differ')
    manifest,compute=_frame('CELL_MANIFEST'),_frame('COMPUTE_PROFILE')
    neural=manifest[manifest.fit_classification=='new_neural_fit'].copy()
    inherited=manifest[manifest.fit_classification=='qualified_inherited_convex_reference']
    require(len(manifest)==290 and len(neural)==270 and len(inherited)==20,'complete formal/reference manifest required')
    neural['seed']=neural.seed.astype(int)
    expected={(a,s,f,r) for a in ARMS for s in config['seeds'] for f in config['folds'] for r in ['inner','outer']}
    require(_keys(neural,['arm','seed','fold','role'])==expected and _keys(compute,['arm','seed','fold','role'])==expected
            and set(neural.source)=={identity},'formal identity/evaluation source differs')
    require('actual_fit_source' in neural.columns,'original checkpoint sources must be explicit')
    actual_sources=[]
    for q in neural.to_dict('records'):
        checkpoint=P/q['role']/f"{q['arm']}_seed{int(q['seed'])}_fold{int(q['fold'])}.pt"
        receipt=require_record(checkpoint.with_suffix('.json'),checkpoint_path=checkpoint)
        require(receipt['status']=='PASS' and receipt['checkpoint_sha256']==digest(checkpoint)==q['checkpoint_sha256']
                and receipt['source']==q['actual_fit_source']
                and (receipt['arm'],int(receipt['seed']),int(receipt['fold']),receipt['role'])==
                    (q['arm'],int(q['seed']),int(q['fold']),q['role']),
                'checkpoint is not the exact enumerated D1 artifact or a qualified current D2 fit')
        actual_sources.append((q['role'],q['actual_fit_source']))
    require(sum(s==LEGACY_SOURCE for _,s in actual_sources)==189 and sum(s==identity for _,s in actual_sources)==81
            and sum(r=='inner' and s==LEGACY_SOURCE for r,s in actual_sources)==135
            and sum(r=='outer' and s==LEGACY_SOURCE for r,s in actual_sources)==54
            and all(s==LEGACY_SOURCE or (r=='outer' and s==identity) for r,s in actual_sources),
            'retained D1 versus fresh D2 completed-cell inventory differs')
    replay_cells=_frame('CHECKPOINT_REPLAY')
    require(_keys(replay_cells,['arm','seed','fold','role'])==expected,'mixed-source replay cell matrix differs')
    replay_lookup=replay_cells.set_index(['arm','seed','fold','role'])
    for q in neural.to_dict('records'):
        replay_row=replay_lookup.loc[q['arm'],int(q['seed']),int(q['fold']),q['role']]
        require(replay_row.actual_fit_source==q['actual_fit_source']
                and replay_row.checkpoint_sha256==q['checkpoint_sha256'],
                'replay actual source/checkpoint bytes differ from retained/fresh manifest')
    require(_keys(inherited,['arm','fold','role'])=={(a,f,r) for a in ['J0','J_CS'] for f in config['folds'] for r in ['inner','outer']}
            and set(inherited.source)=={config['upstream_source']},'read-only inherited reference identities differ')
    selection=require_record(R/'EPOCH_SELECTION_LOCK.json')
    require(selection['source']==LEGACY_SOURCE and selection['status']=='ALL135_INNER_SELECTIONS_LOCKED'
            and selection['no_outer_scores_used'] is True,'all-inner preouter epoch selection not locked')
    selected=pd.DataFrame(selection['selections'])
    require(_keys(selected,['arm','seed','fold'])=={(a,s,f) for a in ARMS for s in config['seeds'] for f in config['folds']},
            'epoch-lock cell matrix differs')
    for q in selected.itertuples(index=False):
        matches=neural[(neural.arm==q.arm)&(neural.seed==q.seed)&(neural.fold==q.fold)]
        require(len(matches)==2 and set(matches.selected_epoch)=={q.epoch}
                and matches[matches.role=='inner'].iloc[0].checkpoint_sha256==q.checkpoint_sha256,
                'fresh outer schedule/checkpoint identity differs from locked inner selection')
    prior=require_record(R/'BUDGET_AUDIT.json');budget=gates['BUDGET_VERIFICATION']
    require(prior['source']==LEGACY_SOURCE and prior['no_outer_results_used'] is True
            and prior['status'] in ['PASS','BUDGET_UNRESOLVED'] and budget['original_preouter_audit_preserved'] is True,
            'preouter budget evidence changed')
    inner_done=require_record(P/'inner_DONE.json')
    outer_done=require_record(P/'outer_DONE.json')
    require(inner_done['status']==outer_done['status']=='PASS' and inner_done['source']==LEGACY_SOURCE
            and outer_done['source']==identity and inner_done['formal_fits']==outer_done['formal_fits']==135,
            'preserved inner completion or resumed full outer completion differs')
    require(len(budget['rows'])==9 and {q['arm'] for q in budget['rows']}==set(ARMS),'budget-arm coverage differs')
    for q in budget['rows']:
        rows=compute[(compute.arm==q['arm'])&(compute.role=='inner')]
        fraction=float((rows.actual_epochs==config['training']['max_epochs']).mean())
        require(len(rows)==15 and q['inner_cells']==15 and abs(q['cap_fraction']-fraction)<1e-12
                and abs(prior['inner_cap_fractions'][q['arm']]-fraction)<1e-12
                and q['budget_gate_pass']==(fraction<=config['engineering_gate']['maximum_cap_fraction_per_arm']),
                'budget claim differs from actual optimizer schedule')
    coverage=gates['PROBE_COVERAGE_AUDIT']
    require(len(coverage['rows'])==18 and {(q['arm'],q['role']) for q in coverage['rows']}=={(a,r) for a in ARMS for r in ['inner','outer']}
            and all(q['checkpoint_count']==15 for q in coverage['rows']) and coverage['additional_forward_or_gradient_calls']==0
            and coverage['raw_patient_indices_public'] is False,'aggregate-only all-fit/eight-query coverage incomplete')
    for q in coverage['rows']:
        require(q['isolated_input_gradient_probe_per_checkpoint_patients_min']==8
                and q['isolated_input_gradient_probe_per_checkpoint_patients_max']==8
                and q['isolated_input_gradient_available']==(q['arm'] not in ['L0','P1','P16']),
                'fixed eight queries or unavailable baseline gradients misrepresented')
    behavior=gates['SELECTIVITY_BEHAVIOR_AUDIT'];rows=behavior['rows']
    require(len(rows)==15 and {(q['seed'],q['fold']) for q in rows}=={(s,f) for s in config['seeds'] for f in config['folds']}
            and all(q['arm']=='S16' and q['role']=='inner' for q in rows),'behavior gate used wrong checkpoints')
    for direction in ['po','op']:
        median=float(np.median([q[direction+'_selectivity_support_patient_mean'] for q in rows]))
        claim=behavior['directions'][direction]
        require(abs(claim['median_selected_inner_checkpoint_patient_mean_support']-median)<1e-12
                and claim['behavior_gate_pass']==(median<.75),'sparse behavior median differs')
    require(behavior['behavior_gate_pass']==all(q['behavior_gate_pass'] for q in behavior['directions'].values()),
            'behavior-axis boolean differs')
    _verify_metrics_decision(config,identity,gates)
    ledger,smoke=read(P/'SMOKE_LEDGER.json'),read(P/'SMOKE_COUNTS.json')
    attempts=ledger['attempts'];execution=gates['EXECUTION_COUNTS']
    require(attempts and len({q['id'] for q in attempts})==len(attempts) and all(q['status']!='RUNNING' for q in attempts),
            'short-fit execution ledger incomplete')
    real=sum(q.get('generator_steps',0)>0 for q in attempts);steps=sum(q.get('generator_steps',0) for q in attempts)
    require(smoke['status']=='PASS' and smoke['real_short_fits']==real and smoke['generator_optimizer_steps']==steps
            and smoke['discriminator_optimizer_steps']==0 and smoke['no_saved_smoke_state_or_patient_risk'] is True,
            'actual diagnostic updates differ')
    require(done['smoke']==smoke,'training completion cites stale short-fit counts')
    formal_attempts=[];formal_receipts=[]
    for q in neural.to_dict('records'):
        key=(q['arm'],int(q['seed']),int(q['fold']),q['role'])
        path=P/'attempts'/q['role']/f"{q['arm']}_seed{int(q['seed'])}_fold{int(q['fold'])}.json"
        current=read(path)['attempts']
        require(current,'missing actual formal attempt ledger')
        for index,item in enumerate(current):
            require_attempt(item,path,index)
            require((item['arm'],item['seed'],item['fold'],item['role'])==key
                    and int(item['generator_steps'])>=0 and item['status']!='RUNNING',
                    'formal attempt identity/source/steps or closure differ')
        successful=[item for item in current if item['status']=='RETURNED_SUCCESSFULLY']
        require(len(successful)==1 and successful[0]['generator_steps']==int(q['actual_epochs'])
                and successful[0]['source']==q['actual_fit_source'],
                'completed checkpoint schedule differs from actual successful attempt steps')
        formal_attempts.extend(current)
        formal_receipts.append({'arm':q['arm'],'role':q['role'],'seed':int(q['seed']),
            'fold':int(q['fold']),'ledger_sha256':digest(path),'attempts':len(current)})
    formal_real=sum(q.get('generator_steps',0)>0 for q in formal_attempts)
    formal_steps=sum(int(q.get('generator_steps',0)) for q in formal_attempts)
    completed_steps=int(compute.actual_epochs.sum())
    require(formal_real>=270 and formal_steps>=completed_steps,'formal attempts undercount completed checkpoint matrix')
    for key,value in [('new_formal_neural_fits',270),('new_convex_fits',0),('qualified_inherited_unique_convex_models',20),
            ('smoke_attempts_recorded',len(attempts)),('smoke_real_short_fits_with_generator_steps',real),
            ('smoke_generator_optimizer_steps',steps),('formal_generator_optimizer_steps',completed_steps),
            ('formal_attempts_recorded',len(formal_attempts)),('formal_real_fit_attempts_with_generator_steps',formal_real),
            ('formal_all_attempt_generator_optimizer_steps',formal_steps),
            ('formal_failed_or_unfinished_attempts_with_generator_steps',sum(q['status']!='RETURNED_SUCCESSFULLY' and q.get('generator_steps',0)>0 for q in formal_attempts)),
            ('formal_partial_attempt_generator_optimizer_steps',formal_steps-completed_steps),
            ('formal_discriminator_optimizer_steps',0),('all_recorded_real_neural_fits',formal_real+real),
            ('all_recorded_generator_optimizer_steps',formal_steps+steps),
            ('retained_pre_audit_repair_formal_fits',189),('completed_post_audit_repair_formal_fits',81)]:
        require(execution[key]==value,'reported optimizer/fit counts differ: '+key)
    require(execution['completed_formal_actual_fit_source_counts']=={LEGACY_SOURCE:189,identity:81},
            'reported successful attempt original sources differ')
    require(execution['formal_attempt_ledger_audit']==formal_receipts,
            'report formal-attempt receipt hashes differ from actual private ledgers')
    for key,value in execution_bounds(formal_attempts,real,steps).items():
        require(execution[key]==value,'reported interrupted-attempt uncertainty bound differs: '+key)
    _verify_gpu_profile()
    report=gates['REPORT_INPUT_AUDIT']
    require(report['reporter_sha256']==digest(ROOT/'scripts/report_phase20.py')
            and report['results_sha256']==digest(R/'RESULTS.md') and report['decision_sha256']==digest(R/'decision.json'),
            'reporter or generated result identity changed')
    require('HISTORICAL_REFERENCE_REPAIR_AUDIT.json' in report['files'],
            'final report omitted historical-reference repair evidence')
    for name,expected in report['files'].items():
        require('/' not in name and '\\' not in name and digest(R/name)==expected,'reported evidence input changed')
    for name in ['RESULTS.md','DECISION.md','PROTOCOL.md','LITERATURE_AUDIT.md','FAILURE_LOG.md','REPRODUCE.md']:
        require((R/name).is_file() and (R/name).stat().st_size>0,'required report missing: '+name)
    return {'status':'PASS','source':identity,'formal_new_neural_fits':270,'new_convex_fits':0,
        'inherited_unique_convex_models':20,'primary_family_tests':14,'bootstrap_draws':20000,
        'retained_original_source':LEGACY_SOURCE,'retained_original_source_checkpoints':189,
        'fresh_current_source_checkpoints':81,
        'private_audit_repair_inventory_sha256':digest(inventory_path),
        'audit_repair_lock_sha256':digest(R/'AUDIT_REPAIR_LOCK.json'),
        'historical_reference_repair_audit_sha256':digest(R/'HISTORICAL_REFERENCE_REPAIR_AUDIT.json'),
        'historical_reference_audit_program_sha256':digest(ROOT/'scripts/audit_phase20_history_reference.py'),
        'prediction_vector_replays':1740,'scientific_decision':read(R/'decision.json')['status'],
        'private_training_done_sha256':digest(P/'TRAINING_DONE.json'),
        'private_evaluation_done_sha256':digest(P/'EVALUATION_DONE.json'),
        'private_smoke_ledger_sha256':digest(P/'SMOKE_LEDGER.json')}


def audit_parent_immutability():
    require(git('branch','--show-current').decode().strip()==BRANCH,'wrong publication branch')
    subprocess.check_call(['git','merge-base','--is-ancestor',PARENT,'HEAD'],cwd=ROOT)
    inherited={}
    for record in git('ls-tree','-r','-z',PARENT).split(b'\0'):
        if not record:
            continue
        header,encoded=record.split(b'\t',1);mode,kind,oid=header.decode().split()
        require(kind=='blob' and mode in ['100644','100755'],'unsupported inherited tracked object')
        relative=encoded.decode('utf-8');path=safe_path(relative)
        require(path.is_file(),'inherited tracked file removed: '+relative)
        raw=path.read_bytes();blob=b'blob '+str(len(raw)).encode()+b'\0'+raw
        require(hashlib.sha1(blob).hexdigest()==oid,'inherited tracked bytes changed: '+relative)
        inherited[relative]=hashlib.sha256(raw).hexdigest()
    manifest=read(ROOT/PARENT_REPORT/'ARTIFACT_MANIFEST.json')
    frozen=json.loads(git('show',PARENT+':'+PARENT_REPORT+'/ARTIFACT_MANIFEST.json'))
    require(manifest==frozen and manifest['status']=='PASS','parent publication manifest changed')
    for row in manifest['public_files']:
        require(inherited.get(row['path'])==row['sha256'],'parent public artifact changed: '+row['path'])
    scope=json.loads(git('show',PARENT+':'+PARENT_REPORT+'/PUBLICATION_SCOPE_AUDIT.json'))
    legacy=scope['unchanged_inherited_violations']
    require(len(legacy)==5 and all(Path(q['path']).suffix.lower()=='.zip' for q in legacy),
            'legacy exception must be exactly five inherited ZIPs')
    tracked=[p.decode() for p in git('ls-files','-z').split(b'\0') if p]
    require(set(protected_path_violations(tracked))=={q['path'] for q in legacy},'repository guard found new violations')
    require(all(inherited.get(q['path'])==q['sha256'] for q in legacy),'legacy protected bytes changed')
    return {'status':'PASS','parent_commit':PARENT,'inherited_tracked_files_verified':len(inherited),
        'parent_public_manifest_files_verified':len(manifest['public_files']),
        'all_inherited_tracked_bytes_unchanged':True,'verification':'Git blob identity and SHA256 of every inherited tracked file'},legacy


def publication_paths():
    files=list((ROOT/'src/phase20_selective').glob('*.py'))
    files += [ROOT/'configs/phase20_selective_attention.json']+[ROOT/'scripts'/n for n in SCRIPT_NAMES]
    files += list((ROOT/'tests').glob('test_phase20_*.py'))
    files += [p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']
    require(all(p.is_file() for p in files),'required new source/report file missing')
    return sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())


def main():
    completion=verify_completion();parent,legacy=audit_parent_immutability()
    write(R/'PARENT_IMMUTABILITY_AUDIT.json',parent)
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'only new Phase20 source/config and anonymous aggregate reports',
        'base_commit':PARENT,'repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS','unchanged_inherited_violations':legacy,
        'new_artifacts_strict_privacy':True,'no_repository_wide_PASS_claim':True})
    rows=[audit_public_file(p) for p in publication_paths()]
    write(R/'ARTIFACT_MANIFEST.json',{**completion,'public_files':rows,'parent_commit':PARENT,'branch':BRANCH,
        'scientific_axis_truth_not_required_for_publication':True,
        'private_excluded':['patient OOF and raw risks','trained weights','raw matrices','patient/gene identifiers',
            'donor mappings','PCA/Token states','raw attention maps','foundation weights','credentials'],
        'no_automatic_followup_training':True})
    audit_public_file(R/'ARTIFACT_MANIFEST.json')
    print('PHASE20_PACKAGE_PASS',len(rows),completion['scientific_decision'],flush=True)


if __name__=='__main__':
    main()
