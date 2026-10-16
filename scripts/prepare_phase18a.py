import json,subprocess,sys,platform
import torch
from src.phase18a_readout.data import P,R,ROOT,cfg,source,write,digest,old

def main():
    c=cfg();assert c['parent_training']==old.cfg()['training'];assert c['patch_budget']==old.cfg()['patch_budget']==512
    assert old.source()==c['parent_training_source']
    assert (R/'READOUT_PAPER_CODE_MAPPING.md').exists()
    check=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_phase18a_readout.py','tests/test_phase18a_model.py'],cwd=ROOT,capture_output=True,text=True)
    if check.returncode:raise RuntimeError(check.stdout+check.stderr)
    assert '27 passed' in check.stdout
    cuda=json.loads((R/'OPERATOR_PARITY_CUDA.json').read_text());profile=json.loads((R/'SCAN_IMPLEMENTATION_PROFILE.json').read_text())
    write(R/'OPERATOR_PARITY.json',{'status':'PASS','source':source(),'tests':27,'stdout':check.stdout.strip(),'test_exit':check.returncode,'test_file_hashes':{p:digest(ROOT/p) for p in ['tests/test_phase18a_readout.py','tests/test_phase18a_model.py']},'CPU_and_CUDA_scan_oracle':cuda,'production_shape_implementation_benchmark':profile,'full_model_final_parent_identical':True,'local_model_parameter_and_input_gradients_oracle':True,'additional_trainable_parameters':0,'chunk_size_locked':c['scan']['chunk_size'],'scope':'operator/model/parity only; synthetic benchmark, not real-data optimization'})
    lock={'status':'LOCKED_BEFORE_FIRST_FORMAL_TRAINING','source':source(),'parent_source':old.source(),'parent_commit':c['parent_commit'],'config':c}
    P.mkdir(parents=True,exist_ok=True)
    if (P/'LOCK.json').exists():assert json.loads((P/'LOCK.json').read_text())==lock
    write(P/'LOCK.json',lock);write(R/'PROTOCOL_LOCK.json',lock)
    write(R/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'GPU_total_MiB':torch.cuda.get_device_properties(0).total_memory/2**20,'source':source(),'parent_training_environment_reuse_audited_separately':True})
    print('LOCKED',source(),check.stdout.strip(),flush=True)

if __name__=='__main__':main()
