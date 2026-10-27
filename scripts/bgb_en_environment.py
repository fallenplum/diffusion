"""Anonymous runtime/dependency evidence and real-input qualification extract."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original=json.loads(args.qualification.read_text())
    if original['status']!='PASS_REAL_INPUT_FORWARD_BACKWARD' or original['optimized_models']!=0 or original['weights_saved'] is not False:
        raise ValueError('The input is not the saved zero-optimizer real-input qualification')
    allowed=('stage','arm','patients','real_patch_nodes','seconds_forward_backward','replay_max','trainable_parameters','nonzero_gradient_tensors','gradient_tensors')
    qual=dict(status=original['status'],scientific_result=False,optimized_models=0,survival_updates=0,weights_saved=False,
        original_qualification_sha256=sha(args.qualification),models=[{key:model[key] for key in allowed if key in model} for model in original['models']],
        extract_scope='aggregate cost/gradient counts only; no per-patient geometry exported')
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','scikit-learn','mamba-ssm','h5py','triton')}
    query=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    name,memory=query.stdout.splitlines()[0].rsplit(',',1)
    dependencies={}
    for relative in ('src/phase10_trimodal/math.py','src/phase10_trimodal/reference.py'):
        tracked=subprocess.run(['git','show','HEAD:'+relative],cwd=ROOT,check=True,capture_output=True).stdout
        dependencies[relative]=dict(sha256=sha(ROOT/relative),matches_git_parent_bytes=hashlib.sha256(tracked).hexdigest()==sha(ROOT/relative))
    report=dict(status='PASS_RUNTIME_AND_DEPENDENCY_INVENTORY',python=sys.version.split()[0],packages=versions,
        reported_GPU_name=name.strip(),reported_GPU_total_MiB=float(memory),
        hardware_scope='runtime-reported device; no retail hardware-capacity inference',
        imported_reference_dependencies=dependencies,formal_Mamba_backend='official mamba_ssm.Mamba',
        new_optimizer_fits=0,new_patient_reads=0,script_sha256=sha(__file__),
        resource_boundary='process peaks in fit receipts; monitor CPU/RAM reflect host procfs rather than a dedicated process quota')
    (args.output/'R0_REAL_INPUT_QUALIFICATION.json').write_text(json.dumps(qual,indent=2)+'\n')
    (args.output/'R0_COMPUTE_QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],python=report['python'],dependencies_unchanged=all(row['matches_git_parent_bytes'] for row in dependencies.values()))))


if __name__=='__main__':main()
