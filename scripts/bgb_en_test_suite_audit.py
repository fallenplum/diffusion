"""Run or summarize the exact synthetic CPU test suite without private data.

The raw pytest log, JUnit XML, runtime hook and execution receipt remain private.
Only counts, package/thread metadata and relative production/test SHA256s are
published. Existing run paths are never deleted or silently reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_publication import SOURCES,SCRIPTS,CONFIGS,TESTS,sha256


def require(value,message):
    if not bool(value):raise ValueError(message)


def source_snapshot(root):
    root=Path(root)
    return {name:sha256(root/name) for name in sorted(set(SOURCES+SCRIPTS+CONFIGS+TESTS))
            if (root/name).is_file()}


def junit_counts(path):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
    require(bool(suites),'JUnit contains no test suite counters')
    result={name:sum(int(row.get(name,'0')) for row in suites)
            for name in ('tests','failures','errors','skipped')}
    result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
    require(result['passed']>=0,'JUnit test counters are inconsistent')
    return result


def warning_count(log):
    matches=re.findall(r'\b(\d+) warnings?\b',log)
    return int(matches[-1]) if matches else 0


def execute(basetemp):
    basetemp=Path(basetemp);parent=basetemp.parent;stem=basetemp.name
    paths={name:parent/(stem+suffix) for name,suffix in
        (('log','.log'),('junit','.xml'),('execution','_execution.json'),
         ('runtime','_runtime'),('threads','_threads.json'))}
    require(not basetemp.exists() and all(not path.exists() for path in paths.values()),
            'Final test path already exists; preserve it and choose an explicit new run name')
    parent.mkdir(parents=True,exist_ok=True);paths['runtime'].mkdir()
    # sitecustomize applies only to this subprocess through its private
    # PYTHONPATH. It records actual torch settings before pytest imports tests.
    hook="""import json, os
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
Path(os.environ['BGB_TEST_THREAD_RECEIPT']).write_text(json.dumps({
 'torch_num_threads':torch.get_num_threads(),
 'torch_num_interop_threads':torch.get_num_interop_threads(),
 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'torch_version':torch.__version__}))
"""
    (paths['runtime']/'sitecustomize.py').write_text(hook)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',BGB_TEST_THREAD_RECEIPT=str(paths['threads']))
    environment['PYTHONPATH']=str(paths['runtime'])+os.pathsep+environment.get('PYTHONPATH','')
    before=source_snapshot(ROOT);start=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    command=[sys.executable,'-m','pytest','-q','tests/bgb_en','--basetemp',str(basetemp),
             '--junitxml',str(paths['junit'])]
    with paths['log'].open('xb') as log:
        completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    value={'schema':'bgb_en_private_cpu_test_execution_v1','returncode':completed.returncode,
        'started_utc':started,'elapsed_seconds':time.monotonic()-start,
        'private_exact_command':command,'private_basetemp':str(basetemp),
        'private_log':str(paths['log']),'private_junit':str(paths['junit']),
        'private_thread_receipt':str(paths['threads']),
        'source_before':before,'source_after':source_snapshot(ROOT),
        'no_existing_run_deleted':True,'public_test_scope':'tests/bgb_en',
        'environment':{key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    paths['execution'].write_text(json.dumps(value,indent=2)+'\n')
    return paths['execution']


def summarize(execution):
    execution=Path(execution);value=json.loads(execution.read_text())
    require(value['schema']=='bgb_en_private_cpu_test_execution_v1','Unknown CPU suite execution schema')
    log_path=Path(value['private_log']);junit_path=Path(value['private_junit'])
    counts=junit_counts(junit_path);counts['warnings']=warning_count(log_path.read_text(errors='replace'))
    thread_path=Path(value['private_thread_receipt']);threads=json.loads(thread_path.read_text())
    stable=value['source_before']==value['source_after']
    current=source_snapshot(ROOT);still_matches=current==value['source_after']
    actual_cpu=threads['torch_num_threads']==threads['torch_num_interop_threads']==2 and threads['CUDA_VISIBLE_DEVICES']==''
    success=value['returncode']==0 and counts['failures']==counts['errors']==0 and stable and still_matches and actual_cpu
    return {'schema':'bgb_en_synthetic_cpu_test_suite_audit_v1',
        'status':'PASS_SYNTHETIC_CPU_TEST_SUITE' if success else 'FAIL_SYNTHETIC_CPU_TEST_SUITE',
        'scientific_stage_status':'RUNNING_OR_AS_REPORTED_BY_LINEAGE_NOT_INFERRED_FROM_UNIT_TESTS',
        'created_utc':datetime.now(timezone.utc).isoformat(),'started_utc':value['started_utc'],
        'test_scope':value['public_test_scope'],'counts':counts,'pytest_exit_code':value['returncode'],
        'elapsed_seconds':value['elapsed_seconds'],'environment':value['environment'],
        'actual_torch_runtime':threads,'source_unchanged_during_suite':stable,
        'current_source_still_matches_tested_bytes':still_matches,
        'production_and_test_source_sha256':value['source_after'],
        'execution_receipt_sha256':sha256(execution),'private_log_sha256':sha256(log_path),
        'private_junit_sha256':sha256(junit_path),'private_thread_receipt_sha256':sha256(thread_path),
        'warnings_count_source':'pytest terminal final summary; no warning summary means zero',
        'synthetic_tensor_forward_and_optimizer_operations_may_be_exercised':True,
        'new_real_research_fits':0,'real_patient_or_outer_data_inputs_supplied':False,
        'GPU_hidden':actual_cpu,'full_log_tracebacks_and_test_runtime_paths_public':False,
        'test_success_is_not_model_replay_or_scientific_completion':True,
        'no_existing_run_deleted':value['no_existing_run_deleted'],
        'auditor_sha256':sha256(Path(__file__))}


def write_report(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'TEST_SUITE_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    counts=result['counts']
    lines=['# 合成CPU测试合并审计','',f"状态：**{result['status']}**。",'',
        f"实际通过 {counts['passed']}、失败 {counts['failures']}、错误 {counts['errors']}、跳过 {counts['skipped']}、警告 {counts['warnings']}；pytest退出码 {result['pytest_exit_code']}。",'',
        f"实测单次套件墙钟 {result['elapsed_seconds']:.3f} 秒。GPU对该进程隐藏，实际torch计算/interop线程均为2；BLAS/OMP/MKL/NumExpr线程环境均为2。",'',
        f"测试前后生产源、配置与测试字节一致：{result['source_unchanged_during_suite']}；报告时仍为同一批字节：{result['current_source_still_matches_tested_bytes']}。完整逐文件SHA见JSON。",'',
        '测试使用合成输入；可能执行小型张量前向或优化步骤，但没有新增真实研究拟合，也未给测试进程提供患者/外折数据入口。原始日志、JUnit、traceback与运行目录均保留私有，仅公开计数、环境和来源哈希。',
        '', '**优势：**对最终合并源码实际执行，而不是将各代理历史PASS相加。',
        '', '**局限：**单元测试通过不能替代真实checkpoint回放、医学语义真值或独立外部验证，也不把整体研究状态改为COMPLETE。',
        '', '**置信度：高，仅限本次明确源码字节上的合成CPU测试结果。**','']
    if result.get('prior_attempts'):
        lines+=['## 保留的首次环境失败','',
            '不是首轮全通过。首次套件334项通过、2项失败；两项均因服务器未部署已有匿名O9诊断fixture而发生FileNotFoundError。补齐既有有效V2 JSON后，在全新目录用完全相同的生产/配置/测试字节重跑，336项全部通过。首轮私有日志、JUnit和执行收据均保留；未通过修改测试或阈值消除失败。','',
            '合并测试之后允许的绘图器/只读审计与对应测试变化另列哈希差异；336项证据对应原明确测试字节，后续诊断变化不伪称已经重跑全部套件。','']
    (output/'TEST_SUITE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')


POST_SUITE_DIAGNOSTIC_ONLY={
    'scripts/bgb_en_plot.py','scripts/bgb_en_patch_throughput.py','scripts/bgb_en_publication.py',
    'scripts/bgb_en_test_suite_audit.py','scripts/bgb_en_publication_dependencies.py',
    'tests/bgb_en/test_plot.py','tests/bgb_en/test_patch_throughput.py',
    'tests/bgb_en/test_publication.py','tests/bgb_en/test_test_suite_audit.py',
    'tests/bgb_en/test_publication_dependencies.py',
}


def add_history(result,prior_executions,current):
    """Append metadata; never rewrite original counts or their tested bytes."""
    require(result['status']=='PASS_SYNTHETIC_CPU_TEST_SUITE','History must retain a real completed PASS receipt')
    original=result['production_and_test_source_sha256'];history=[]
    for path in prior_executions:
        path=Path(path);value=json.loads(path.read_text())
        require(value['source_before']==value['source_after']==original,
                'Prior suite was not the identical production/config/test byte set')
        counts=junit_counts(value['private_junit']);counts['warnings']=warning_count(Path(value['private_log']).read_text(errors='replace'))
        failures=ET.parse(value['private_junit']).getroot().findall('.//failure')
        require(value['returncode']!=0 and counts['failures']==2 and counts['errors']==0
                and len(failures)==2 and all('FileNotFoundError' in (node.get('message','')+(node.text or ''))
                    and 'O9_FLOAT32_DRIFT_DEFAULT.json' in (node.get('message','')+(node.text or '')) for node in failures),
                'Prior failure does not match the independently identified missing-fixture case')
        history.append(dict(counts=counts,pytest_exit_code=value['returncode'],
            source_identical_to_successful_rerun=True,execution_receipt_sha256=sha256(path),
            private_log_sha256=sha256(value['private_log']),private_junit_sha256=sha256(value['private_junit']),
            reason='Two tests could not read an existing anonymous O9 diagnostic JSON missing from server deployment',
            recovery='Copy two existing valid V2 anonymous O9 JSON fixtures, then rerun unchanged source in a new directory',
            source_or_test_or_threshold_changed_for_recovery=False))
    changed=[{'file':name,'tested_sha256':original.get(name),'current_sha256':current.get(name)}
             for name in sorted(set(original)|set(current)) if original.get(name)!=current.get(name)]
    require(all(row['file'] in POST_SUITE_DIAGNOSTIC_ONLY for row in changed),
            'Post-suite source change is outside the approved readonly diagnostic/test scope')
    result.update(prior_attempts=history,first_attempt_was_not_all_passed=True,
        original_test_counts_and_tested_source_hashes_preserved=True,
        current_source_match_flag_is_at_original_report_creation=True,
        post_suite_diagnostic_source_changes=changed,
        post_suite_changes_are_not_claimed_to_have_run_the_full_suite=True,
        history_augmentation_source_sha256=sha256(Path(__file__)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('run','report','history'),required=True)
    parser.add_argument('--basetemp',type=Path)
    parser.add_argument('--execution',type=Path)
    parser.add_argument('--prior-execution',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en')
    args=parser.parse_args()
    if args.stage=='history':
        require(bool(args.prior_execution),'History requires exact retained private execution receipts')
        report=args.output/'TEST_SUITE_AUDIT.json'
        result=add_history(json.loads(report.read_text()),args.prior_execution,source_snapshot(ROOT))
        write_report(result,args.output)
        print(json.dumps({'status':result['status'],'prior_attempts':len(result['prior_attempts']),
                          'post_suite_diagnostic_changes':len(result['post_suite_diagnostic_source_changes'])}));return
    if args.stage=='run':
        require(args.basetemp is not None,'--basetemp is required for a new explicit run')
        execution=execute(args.basetemp)
    else:
        require(args.execution is not None,'--execution is required to report an existing run')
        execution=args.execution
    result=summarize(execution);write_report(result,args.output)
    print(json.dumps({'status':result['status'],'counts':result['counts'],
        'source_unchanged_during_suite':result['source_unchanged_during_suite'],
        'current_source_still_matches_tested_bytes':result['current_source_still_matches_tested_bytes']}))
    if result['status'].startswith('FAIL'):raise SystemExit(1)


if __name__=='__main__':main()
