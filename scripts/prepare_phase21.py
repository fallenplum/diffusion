"""Qualify only the changed operator and inherited unchanged pipeline."""
import gc
import json
import subprocess
import sys
import time
import numpy as np
import torch
from src.phase21_relative.data import ROOT,P,R,cfg,source,source_files,write,Data
from src.phase21_relative.train import train_cell
from src.phase10_trimodal.data import digest

def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'logs').mkdir(exist_ok=True)
    if (R/'PROTOCOL_LOCK.json').exists(): raise ValueError('locked; do not overwrite qualification')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=cfg()['branch']: raise ValueError('wrong branch')
    identity=source();write(P/'PREPARATION_START.json',{'source':identity,'configuration':cfg(),'unix_time':time.time()})
    targets=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_phase21_*.py'))
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*targets],cwd=ROOT,capture_output=True,text=True)
    (P/'logs/tests.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'skipped' in result.stdout: raise ValueError('tests failed: '+result.stdout[-5000:]+result.stderr[-1000:])
    import re
    test_count=int(re.search(r'(\d+) passed',result.stdout).group(1))
    ds=Data(1,'inner',31);outer=ds.part['outer_test']
    smoke=[];fits=[]
    def short(dataset):
        item={'source':identity,'steps':0,'status':'RUNNING'};fits.append(item);write(P/'SMOKE_LEDGER.json',{'fits':fits})
        def step(kind,epoch): item['steps']+=1;write(P/'SMOKE_LEDGER.json',{'fits':fits})
        ck=train_cell('DIRECT',31,1,'inner',epochs=2,dataset=dataset,progress_callback=step)
        item['status']='SUCCESS';write(P/'SMOKE_LEDGER.json',{'fits':fits});return ck
    first=short(ds)
    from src.phase19_attention.data import parent
    original=parent.endpoints
    def poisoned(d):
        result={ep:{k:np.array(v,copy=True) for k,v in obj.items()} for ep,obj in original(d).items()}
        result['DSS']['time'][outer]=np.arange(len(outer))+17013
        result['DSS']['event'][outer]=np.arange(len(outer))%2
        return result
    try:
        parent.endpoints=poisoned;alternate=Data(1,'inner',31);second=short(alternate)
    finally: parent.endpoints=original
    if first['history']!=second['history'] or first['selected_epoch']!=second['selected_epoch']: raise ValueError('outer poisoning changed selection')
    if any(not torch.equal(v,second['state'][k]) for k,v in first['state'].items()): raise ValueError('outer poisoning changed weights')
    if not np.array_equal(first['risk'],second['risk']): raise ValueError('outer poisoning changed risks')
    qualification={'status':'PASS','source':identity,'tests_passed':test_count,'test_stdout':result.stdout[-1500:],
          'test_files':{name:digest(ROOT/name) for name in targets},'real_short_fits':len(fits),
          'short_steps':sum(x['steps'] for x in fits),'outer_endpoint_poisoning':'history,selection,weights,risks bitwise identical',
          'same_parameter_layout_and_same_initialization':'tested P16/S16/DIRECT',
          'new_forward_AST':'only conservative_update symbol replaced by direct_update',
          'training_schedule':'identical inherited training configuration and Efron loop',
          'summary':first['summary'],'smoke_weights_or_patient_predictions_persisted':False}
    del first,second,ds,alternate;gc.collect();torch.cuda.empty_cache()
    if identity!=source(): raise ValueError('qualification source changed')
    write(R/'QUALIFICATION.json',qualification)
    write(R/'PROTOCOL_LOCK.json',{'status':'LOCKED_BEFORE_FIRST_NEW_FORMAL_FIT','source':identity,
          'unix_time':time.time(),'configuration':cfg(),'training_files':source_files(),
          'qualification_sha256':digest(R/'QUALIFICATION.json')})
    print('PHASE21_QUALIFIED',identity,flush=True)

if __name__=='__main__': main()
