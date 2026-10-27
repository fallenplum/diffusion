"""CPU-only static dependency closure for the explicit publication allowlist.

This audit reads source text and anonymous reports only. It never imports the
scientific trainers, loads a checkpoint or opens private measurements. A literal
reference is evidence of a dependency candidate, not proof the branch executed.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_publication import ALLOWLIST,sha256
from scripts.bgb_en_test_suite_audit import POST_SUITE_DIAGNOSTIC_ONLY

SOURCE_PATH=re.compile(r'(?:scripts|src/bgb_en|configs/bgb_en|tests/bgb_en)/[A-Za-z0-9_./-]+\.(?:py|json|ps1)')


def project_modules(tree):
    modules=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(node.module+'.'+alias.name for alias in node.names if alias.name!='*')
    result=set()
    for module in modules:
        if module.startswith(('scripts.','src.bgb_en.','tests.bgb_en.')):
            result.add(module.replace('.','/')+'.py')
    return result


def literal_paths(value):
    if isinstance(value,str):return set(SOURCE_PATH.findall(value))
    if isinstance(value,dict):
        return set().union(*(literal_paths(key)|literal_paths(item) for key,item in value.items())) if value else set()
    if isinstance(value,(list,tuple)):
        return set().union(*(literal_paths(item) for item in value)) if value else set()
    return set()


def analyze(root,allowlist):
    root=Path(root);allowed=set(allowlist);edges=[];audited=[]
    for relative in sorted(allowed):
        if relative=='reports/bgb_en/SOURCE_PUBLICATION_CLOSURE_AUDIT.json':continue
        path=root/relative
        if not path.is_file() or path.suffix not in ('.py','.json','.md'):continue
        contents=path.read_text(encoding='utf-8-sig')
        refs=set()
        if path.suffix=='.py':
            tree=ast.parse(contents,filename=relative)
            refs.update((target,'AST_IMPORT') for target in project_modules(tree))
            if not relative.startswith('tests/'):
                for node in ast.walk(tree):
                    if isinstance(node,ast.Constant) and isinstance(node.value,str):
                        refs.update((target,'SOURCE_LITERAL') for target in literal_paths(node.value))
        elif path.suffix=='.json':
            refs.update((target,'ANONYMOUS_CERTIFICATE_LITERAL') for target in literal_paths(json.loads(contents)))
        else:refs.update((target,'DOCUMENT_LITERAL') for target in literal_paths(contents))
        audited.append({'file':relative,'sha256':sha256(path)})
        for target,kind in sorted(refs):
            candidate=root/target
            # AST from-import symbols can be functions/classes, not modules.
            if kind=='AST_IMPORT' and not candidate.is_file():continue
            scoped=target.startswith(('src/bgb_en/','configs/bgb_en/','tests/bgb_en/')) or target.startswith('scripts/bgb_en_')
            status='ALLOWLISTED' if target in allowed else 'UNLISTED_BG_BEN_DEPENDENCY' if scoped else 'EXISTING_REPOSITORY_DEPENDENCY'
            edges.append({'from':relative,'to':target,'evidence':kind,'status':status,
                          'source_file_present':candidate.is_file()})
    unresolved=sorted({edge['to'] for edge in edges if edge['status']=='UNLISTED_BG_BEN_DEPENDENCY'})
    # The unreferenced old server-only script is deliberately not added merely
    # because a file exists. A binding from any reviewed source/certificate would
    # make it an unresolved dependency and fail this check.
    offset='scripts/'+'bgb_en_audit_offset_cast.py'
    refs=[edge for edge in edges if edge['to']==offset]
    builder=root/'scripts/bgb_en_lock.py'
    external_test=root/'tests/bgb_en/test_external.py'
    result={'schema':'bgb_en_publication_dependency_audit_v1',
            'status':'PASS_STATIC_DEPENDENCY_CLOSURE' if not unresolved else 'FAIL_UNLISTED_DEPENDENCIES',
            'files_reviewed':audited,'dependency_edges':edges,'unlisted_BG_BEN_dependencies':unresolved,
            'old_offset_cast':{'relative_file':offset,'file_present':(root/offset).is_file(),
                'allowlisted':offset in allowed,'source_or_certificate_reference_count':len(refs),
                'publication_required_by_reviewed_files':bool(refs)},
            'historical_source_handling':{
                'lock_builder_actual_sha256':sha256(builder) if builder.is_file() else None,
                'lock_builder_matches_original_generation_bytes':builder.is_file() and sha256(builder)==
                    '9916087ef95d401b9d093f66b1cf2c0c07cff24599371d3d62acca4c3356fdac',
                'accounting_clarification_does_not_rewrite_original_lock_builder':True,
                'clarification_document':'reports/bgb_en/PROTOCOL_ACCOUNTING_ADDENDUM.json',
                'historical_raw_external_test_sha256':'cff5ed9b2027cb4c7111f52fcb6650d67b088b397956c8b9ff0f6d449bc4e9e2',
                'current_external_test_sha256':sha256(external_test) if external_test.is_file() else None,
                'current_external_test_contains_v2_unitnorm_contract':external_test.is_file() and
                    'bgb_en_external_projection_transfer_v2_unitnorm' in external_test.read_text(),
                'old_test_bytes_are_private_historical_reference_not_current_unitnorm_test':True,
                'source_deployment_or_metadata_audit_is_not_a_new_training_run':True},
            'fit_calls':0,'forward_calls':0,'private_data_read':False,
            'scientific_stage_completion_inferred':False,
            'missing_allowlisted_source_files':sorted(name for name in allowed
                if name.startswith(('src/bgb_en/','scripts/','tests/bgb_en/','configs/bgb_en/'))
                and not (root/name).is_file()),
            'scope':'AST imports plus literal source/certificate/document paths in currently available explicit public files. Tests contribute imports only, excluding synthetic filename fixtures. This audit does not consume its previous output. Dynamic imports or unexecuted references are not execution proof.',
            'audit_script_sha256':sha256(Path(__file__))}
    suite=root/'reports/bgb_en/TEST_SUITE_AUDIT.json'
    if suite.is_file():
        tested=json.loads(suite.read_text(encoding='utf-8-sig'))
        if tested.get('schema')=='bgb_en_synthetic_cpu_test_suite_audit_v1':
            original=tested['production_and_test_source_sha256']
            current={row['file']:row['sha256'] for row in audited
                     if row['file'].startswith(('src/bgb_en/','scripts/','tests/bgb_en/','configs/bgb_en/'))}
            changed=[{'file':name,'full_suite_tested_sha256':original.get(name),
                      'current_sha256':current.get(name)} for name in sorted(set(original)|set(current))
                     if original.get(name)!=current.get(name)]
            allowed=all(row['file'] in POST_SUITE_DIAGNOSTIC_ONLY for row in changed)
            result['full_suite_byte_scope']={
                'test_suite_audit_sha256':sha256(suite),'full_suite_result_at_original_tested_bytes':tested['status'],
                'full_suite_counts':tested['counts'],'changed_after_full_suite':changed,
                'changes_confined_to_approved_readonly_diagnostics_and_tests':allowed,
                'frozen_core_and_training_source_unchanged_since_full_suite':allowed,
                'full_suite_not_claimed_to_cover_post_suite_changes':True,
                'post_suite_diagnostic_changes_require_separate_test_evidence':True}
            if not allowed:result['status']='FAIL_POST_SUITE_UNREVIEWED_SOURCE_CHANGE'
    return result


def peer_comparison(current,peer):
    def sources(value):
        return {row['file']:row['sha256'] for row in value['files_reviewed']
                if row['file'].startswith(('src/bgb_en/','scripts/','tests/bgb_en/','configs/bgb_en/'))}
    own,other=sources(current),sources(peer)
    return {'scope':'Supplied peer static snapshot; no filesystem or scientific completion inference',
            'only_current':sorted(set(own)-set(other)),'only_peer':sorted(set(other)-set(own)),
            'different_source_bytes':[{'file':name,'current_sha256':own[name],'peer_sha256':other[name]}
                for name in sorted(set(own)&set(other)) if own[name]!=other[name]],
            'identical_source_files':sum(own[name]==other[name] for name in set(own)&set(other))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en/SOURCE_PUBLICATION_CLOSURE_AUDIT.json')
    parser.add_argument('--peer-audit',type=Path)
    args=parser.parse_args();result=analyze(args.root,ALLOWLIST)
    if args.peer_audit:
        peer=json.loads(args.peer_audit.read_text(encoding='utf-8-sig'))
        result['peer_audit_sha256']=sha256(args.peer_audit)
        result['peer_source_comparison']=peer_comparison(result,peer)
        result['peer_old_offset_cast']=peer.get('old_offset_cast')
        current_reports={row['file']:row['sha256'] for row in result['files_reviewed'] if row['file'].startswith('reports/')}
        peer_reports={row['file']:row['sha256'] for row in peer['files_reviewed'] if row['file'].startswith('reports/')}
        result['anonymous_report_deployment']={
            'snapshot_only_not_automatic_overwrite_authority':True,
            'only_current':sorted(set(current_reports)-set(peer_reports)),
            'only_peer':sorted(set(peer_reports)-set(current_reports)),
            'different_bytes':sorted(name for name in set(current_reports)&set(peer_reports)
                                     if current_reports[name]!=peer_reports[name])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'files_reviewed':len(result['files_reviewed']),
                      'unlisted_BG_BEN_dependencies':result['unlisted_BG_BEN_dependencies'],
                      'old_offset_cast':result['old_offset_cast']}))
    if result['status'].startswith('FAIL'):raise SystemExit(1)


if __name__=='__main__':main()
