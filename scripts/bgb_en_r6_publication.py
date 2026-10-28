"""Explicit anonymous R6 delivery; private arrays/weights never enter a tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import bgb_en_publication as inherited

SOURCES=tuple('src/bgb_en/r6_'+x+'.py' for x in (
    'source','source_model','omics','pathology','survival','training'))
SCRIPTS=tuple('scripts/bgb_en_r6_'+x+'.py' for x in (
    'qualify','pretrain','lock_source','compute_qualify','source_eligibility',
    'schedule_source','report_source','common','run','qualify_target',
    'schedule_target','replay','report_target','retention','export_retention_audit',
    'independent_source_review','independent_retention_review','independent_target_review',
    'source_depth_audit','attention_diagnostics','risk_diagnostics','execution_ledger','plot_diagnostics',
    'subgroups','compact_reports','publication'))
CONFIGS=tuple('configs/bgb_en/r6_'+x+'.json' for x in (
    'architecture_intent_v1','source_v1','inner_v1','retention_v1'))
TESTS=tuple('tests/test_bgb_en_r6_'+x+'.py' for x in (
    'source','source_model','omics','pathology','pretrain','survival','training',
    'common','replay','report_source','report_target','retention','random_gate','subgroups',
    'independent_target_review','source_depth_audit','attention_diagnostics','risk_diagnostics'))+(
    'tests/bgb_en/test_r6_execution_ledger.py','tests/bgb_en/test_r6_plot_diagnostics.py',
    'tests/bgb_en/test_r6_publication_figures.py')
REPORT_NAMES=(
    'R6_0_DATA_QUALIFICATION.md','R6_0_DATA_QUALIFICATION.json',
    'R6_0_SOURCE_CACHE_NUMERICAL_REPLAY.json','R6_0_SOURCE_CACHE_REPLAY.json',
    'R6_0_SOURCE_DATASET_QUALIFICATION.json','R6_SOURCE_TARGET_ELIGIBILITY.json',
    'R6_SOURCE_COHORT_PROVENANCE.md','R6_REAL_SOURCE_COMPUTE_QUALIFICATION.json',
    'R6_1_SOURCE_SUPERVISION.md','R6_1_SOURCE_SUPERVISION.json','R6_1_SOURCE_SUPERVISION.csv',
    'R6_SOURCE_DECISION.json','R6_SOURCE_REPLAY_AUDIT.json','R6_SOURCE_CELL_METRICS.csv',
    'R6_SOURCE_COSTS.csv','R6_SOURCE_GATE_AUDIT.csv',
    'R6_2_OMICS_RETENTION.json','R6_2_OMICS_RETENTION.csv','R6_2_OMICS_RETENTION.md',
    'R6_3_REAL_GENE_REREAD_AUDIT.json','R6_3_REAL_GENE_REREAD_AUDIT.md',
    'PROTOCOL_LOCK.json','R6_4_INNER_MATRIX.md','R6_4_INNER_MATRIX.json','R6_4_INNER_MATRIX.csv',
    'R6_SEED_FOLD.csv','R6_MECHANISM_ABLATIONS.csv','R6_COSTS.csv',
    'R6_DECISION.md','decision.json','R6_5_OUTER_RESULTS.md','R6_5_OUTER_RESULTS.csv',
    'SOURCE_AUDIT.json','R6_REPRODUCE.md','R6_EXECUTION_LEDGER.json',
    'R6_INDEPENDENT_SUMMARY.md','R6_SUBGROUP_METRICS.csv',
    'R6_SUBGROUP_SEED_SUMMARY.csv','R6_SUBGROUP_SUMMARY.csv',
    'R6_SUBGROUP_AUDIT.json','R6_SUBGROUPS.md','R6_RETENTION_INDEPENDENT_REPLAY_AUDIT.json',
    'R6_SOURCE_INDEPENDENT_REVIEW.json','R6_SOURCE_INDEPENDENT_REVIEW.md',
    'R6_RETENTION_INDEPENDENT_REVIEW.json','R6_RETENTION_INDEPENDENT_REVIEW.md',
    'R6_TARGET_INDEPENDENT_REVIEW.json','R6_TARGET_INDEPENDENT_REVIEW.md',
    'R6_SOURCE_DEPTH_AUDIT.json','R6_SOURCE_DEPTH_AUDIT.csv',
    'R6_ATTENTION_DIAGNOSTICS.json','R6_RISK_DIAGNOSTICS.json','R6_RISK_DIAGNOSTICS.md',
    'R6_RESOURCE_SUMMARY.json','R6_ARM_EXECUTION_STATUS.csv','R6_OPERATOR_SCOPE.md')
REPORTS=tuple('reports/bgb_en_r6/'+n for n in REPORT_NAMES)
DIAGNOSTIC_ARMS=('B2','B3','B4','B5','B6','B7','B8','B9','B11','B13')
ATTENTION_PARTS=tuple('reports/bgb_en_r6/R6_ATTENTION_DIAGNOSTICS_'+arm+'.json' for arm in DIAGNOSTIC_ARMS)
OPTIONAL_RISK_PARTS=frozenset('reports/bgb_en_r6/R6_RISK_DIAGNOSTICS_'+arm+'.json' for arm in DIAGNOSTIC_ARMS)
FIGURE_NAMES=('source_huber','source_within_tissue_rho','retention_Harrell',
    'target_Harrell_delta_B1','target_Uno60_delta_B1','risk_branch_SD','query_attention_TV')
FIGURE_FILES=frozenset('reports/bgb_en_r6/figures/'+name+ext for name in FIGURE_NAMES for ext in ('.svg','.png'))
FIGURE_TEXT=('reports/bgb_en_r6/figures/R6_FIGURE_BINDINGS.json','reports/bgb_en_r6/figures/FIGURE_README.md')
REQUIRED=frozenset(SOURCES+SCRIPTS+CONFIGS+TESTS+REPORTS+ATTENTION_PARTS+FIGURE_TEXT)|FIGURE_FILES
ALLOWLIST=REQUIRED|OPTIONAL_RISK_PARTS
AUDIT='reports/bgb_en_r6/SOURCE_PUBLICATION_AUDIT.json'
MANIFEST='reports/bgb_en_r6/PUBLICATION_MANIFEST.json'


def scan_bound_figures(root,pattern):
    """Only seven aggregate plots, their exact source bytes and generator."""
    base=root/'reports/bgb_en_r6/figures'
    binding=json.loads((base/'R6_FIGURE_BINDINGS.json').read_text(encoding='utf-8-sig'))
    inherited.require(binding.get('schema')=='bgb_en_r6_figure_bindings_v1'
        and binding.get('status')=='COMPLETE_7_ANONYMOUS_DIAGNOSTIC_FIGURES'
        and binding.get('figure_count')==7,'FIGURE_BINDING_SCHEMA_CHANGED')
    generator=binding['generator']
    inherited.require(generator['file']=='scripts/bgb_en_r6_plot_diagnostics.py'
        and inherited.sha256(root/generator['file'])==generator['sha256'],'FIGURE_GENERATOR_CHANGED')
    source_map={
        'source_huber':{'R6_1_SOURCE_SUPERVISION.json','R6_SOURCE_CELL_METRICS.csv'},
        'source_within_tissue_rho':{'R6_1_SOURCE_SUPERVISION.json','R6_SOURCE_CELL_METRICS.csv'},
        'retention_Harrell':{'R6_2_OMICS_RETENTION.json'},
        'target_Harrell_delta_B1':{'R6_4_INNER_MATRIX.json'},
        'target_Uno60_delta_B1':{'R6_4_INNER_MATRIX.json'},
        'query_attention_TV':{'R6_ATTENTION_DIAGNOSTICS.json'}|
            {'R6_ATTENTION_DIAGNOSTICS_'+a+'.json' for a in DIAGNOSTIC_ARMS}}
    risk=json.loads((root/'reports/bgb_en_r6/R6_RISK_DIAGNOSTICS.json').read_text(encoding='utf-8-sig'))
    risk_files={'R6_RISK_DIAGNOSTICS.json'}
    if risk.get('schema')=='bgb_en_r6_risk_diagnostic_index_v1':
        parts=risk.get('parts',[])
        inherited.require(len(parts)==10 and {p.get('arm') for p in parts}==set(DIAGNOSTIC_ARMS)
            and all(p.get('file')=='R6_RISK_DIAGNOSTICS_'+p['arm']+'.json' for p in parts),
            'FIGURE_RISK_SOURCE_INDEX_CHANGED')
        risk_files|={p['file'] for p in parts}
    else:
        inherited.require(risk.get('schema')=='bgb_en_r6_risk_diagnostics_v1',
            'FIGURE_RISK_SOURCE_INDEX_CHANGED')
    source_map['risk_branch_SD']=risk_files
    expected_sources=set().union(*source_map.values())
    sources=binding.get('source_files',[])
    inherited.require(isinstance(sources,list) and len(sources)==len(expected_sources)
        and {s.get('file') for s in sources}==expected_sources,
        'FIGURE_EXACT_NONEMPTY_SOURCE_SET_REQUIRED')
    for source in sources:
        inherited.require(Path(source['file']).name==source['file'] and '/' not in source['file']
            and '\\' not in source['file'],'FIGURE_SOURCE_BASENAME_REQUIRED')
        original=root/'reports/bgb_en_r6'/source['file'];p=original.resolve()
        inherited.require(not original.is_symlink() and p.is_relative_to((root/'reports/bgb_en_r6').resolve())
            and str(p.relative_to(root)).replace('\\','/') in ALLOWLIST
            and p.stat().st_size==source['bytes'] and inherited.sha256(p)==source['sha256'],
            'FIGURE_SOURCE_BYTES_OR_ALLOWED_PATH_CHANGED')
    figures=binding['figures']
    inherited.require(len(figures)==7 and {x['basename'] for x in figures}==set(FIGURE_NAMES),
        'SEVEN_FIXED_AGGREGATE_FIGURES_REQUIRED')
    rows=[]
    for fig in figures:
        inherited.require(fig.get('one_axes') is True and fig.get('uncertainty_intervals') is False
            and fig.get('pvalues') is False,'FIGURE_STATS_CONTRACT_CHANGED')
        own_sources=fig.get('source_files',[])
        inherited.require(isinstance(own_sources,list) and len(own_sources)==len(source_map[fig['basename']])
            and set(own_sources)==source_map[fig['basename']], 'FIGURE_PER_CHART_SOURCE_BINDING_CHANGED')
        inherited.require(len(fig['outputs'])==2
            and {x['file'] for x in fig['outputs']}=={fig['basename']+'.png',fig['basename']+'.svg'},
            'PAIRED_SVG_PNG_REQUIRED')
        for item in fig['outputs']:
            original=base/item['file'];p=original.resolve();relative=str(p.relative_to(root)).replace('\\','/')
            inherited.require(p.parent==base.resolve() and relative in FIGURE_FILES and not original.is_symlink(),
                'FIGURE_PATH_NOT_EXPLICITLY_ALLOWED')
            data=p.read_bytes()
            inherited.require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
                and len(data)<=8*1024**2,'FIGURE_BYTES_CHANGED')
            if p.suffix=='.png':
                inherited.require(data.startswith(b'\x89PNG\r\n\x1a\n'),'INVALID_BOUND_PNG')
            else:
                text=data.decode('utf-8');xml=ET.fromstring(text)
                inherited.require(xml.tag.endswith('svg') and not any(node.tag.split('}')[-1].lower()
                    in ('image','script','foreignobject','animate') for node in xml.iter()),'SVG_EXTERNAL_OR_RASTER_CONTENT')
                inherited.require(all(value.startswith('#') for node in xml.iter()
                    for attribute,value in node.attrib.items() if attribute.rsplit('}',1)[-1].lower()=='href'),
                    'SVG_EXTERNAL_HREF_FORBIDDEN')
                inherited.require(re.search(r'(?:xlink:)?href=[\"\x27](?!#)',text) is None
                    and pattern.search(text) is None,'SVG_LINK_OR_PRIVATE_IDENTIFIER')
                for label,rule in inherited.PATTERNS.items():
                    inherited.require(rule.search(text) is None,'FIGURE_PRIVACY_RULE_'+label)
            rows.append(dict(file=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    return rows


def identifier_group_counts(denied):
    """Anonymous scalar coverage; never reuse a raw identifier field name."""
    return {name+'_count':len(values) for name,values in denied['groups'].items()}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--denylist',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    denied=inherited.validate_denylist(json.loads(args.denylist.read_text(encoding='utf-8-sig')))
    inherited.validate_r0_qualification(root)
    source=json.loads((root/'reports/bgb_en_r6/R6_1_SOURCE_SUPERVISION.json').read_text(encoding='utf-8-sig'))
    target=json.loads((root/'reports/bgb_en_r6/R6_4_INNER_MATRIX.json').read_text(encoding='utf-8-sig'))
    inherited.require(source['counts'].get('PASS_COMPLETE_AND_REPLAYED')==147,
                      'COMPLETE_147_SOURCE_REPLAYS_REQUIRED')
    inherited.require(target['counts'].get('PASS_FIT_AND_ACTUAL_REPLAY')==90
        and target['counts'].get('PASS_ZERO_FIT_REFERENCE')==18,
        'COMPLETE_90_TARGET_AND_18_REFERENCE_REPLAYS_REQUIRED')
    # Optional followups are a separate prelocked stage, never silently counted
    # as completed by the base 90-cell manifest.
    inherited.require(not target['decision'].get('actual_optional_followup_required_before_external_stage'),
                      'POSITIVE_PROTOTYPE_OPTIONAL_STAGE_MUST_BE_SEPARATELY_DELIVERED')
    # Both diagnostic formats reconstruct the same full scalar object; no
    # oversized report is made publishable by dropping cells or relaxing scans.
    from scripts.bgb_en_r6_risk_diagnostics import read_diagnostic_bundle
    risk_index=root/'reports/bgb_en_r6/R6_RISK_DIAGNOSTICS.json'
    read_diagnostic_bundle(risk_index)
    index=json.loads(risk_index.read_text(encoding='utf-8-sig'))
    declared=frozenset('reports/bgb_en_r6/'+p['file'] for p in index.get('parts',[]))
    inherited.require(declared<=OPTIONAL_RISK_PARTS and
        {n for n in OPTIONAL_RISK_PARTS if (root/n).exists()}==declared,
        'DIAGNOSTIC_PARTS_MUST_MATCH_EXPLICIT_VALIDATED_INDEX')
    from scripts.bgb_en_r6_attention_diagnostics import read_diagnostics
    attention=read_diagnostics(root/'reports/bgb_en_r6/R6_ATTENTION_DIAGNOSTICS.json')
    inherited.require(attention.get('status')=='COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS'
        and attention.get('diagnostic_forward_calls')==450 and attention.get('new_optimizer_fits')==0,
        'COMPLETE_INFERENCE_ONLY_ATTENTION_AUDIT_REQUIRED')
    old_snapshot={str(f.relative_to(root)):inherited.sha256(f)
                  for directory in ('reports/bgb_en','reports/sami_surv_stage_b')
                  for f in (root/directory).rglob('*') if f.is_file()}
    pattern=inherited.identifier_pattern(denied)
    previous=inherited.ALLOWLIST;inherited.ALLOWLIST=ALLOWLIST|{AUDIT,MANIFEST}
    try:
        files=[inherited.scan_file(root,n,pattern) for n in sorted((REQUIRED|declared)-FIGURE_FILES)]
        files.extend(scan_bound_figures(root,pattern));files.sort(key=lambda x:x['file'])
        audit=dict(schema='bgb_en_r6_source_publication_audit_v1',
            status='PASS_EXPLICIT_ANONYMOUS_FILES_ONLY',source_replayed_logical_units=147,
            source_formal_optimizer_fits=135,target_replayed_neural_fits=90,
            target_zero_fit_references=18,
            source_and_target_decisions_are_separate=True,
            medical_or_causal_claims_inferred_from_publication_scan=False,
            private_identifiers_coordinates_values_risks_and_weights_included=False,
            actual_private_identifier_groups_scanned=identifier_group_counts(denied),
            files_count_before_audit=len(files),git_operations_performed=0)
        (root/AUDIT).write_text(json.dumps(audit,indent=2)+'\n')
        files.append(inherited.scan_file(root,AUDIT,pattern));files.sort(key=lambda x:x['file'])
        content=''.join(f['file']+'\0'+f['sha256']+'\0'+str(f['bytes'])+'\n' for f in files)
        manifest=dict(schema='bgb_en_r6_publication_manifest_v1',status=audit['status'],
            scientific_run_state='COMPLETE_PRELOCKED_BASE_MATRIX',files=files,
            files_count=len(files),total_bytes=sum(f['bytes'] for f in files),
            ordered_file_manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
            unknown_files_grant_publication_permission=False,
            raw_data_or_models_included=False,git_operations_performed=0)
        (root/MANIFEST).write_text(json.dumps(manifest,indent=2)+'\n')
        inherited.scan_file(root,MANIFEST,pattern)
    finally:inherited.ALLOWLIST=previous
    after={str(f.relative_to(root)):inherited.sha256(f)
           for directory in ('reports/bgb_en','reports/sami_surv_stage_b')
           for f in (root/directory).rglob('*') if f.is_file()}
    inherited.require(after==old_snapshot,'HISTORICAL_REPORT_CHANGED_DURING_PUBLICATION')
    print(json.dumps(dict(status=manifest['status'],files=len(files),
        manifest_sha256=inherited.sha256(root/MANIFEST))))


if __name__=='__main__':main()
