"""Explicit, source-identified inner-only schedule sensitivity; original stays immutable."""
import copy,json,hashlib,subprocess,sys,time
from pathlib import Path
from src.trimodal_interaction.data import P,R,ROOT,cfg,source,write,digest

CONFIG=ROOT/'configs/trimodal_interaction_budget160.json'

def budget_source():
    return hashlib.sha256((source()+digest(CONFIG)+digest(Path(__file__))).encode()).hexdigest()

def cell(arm,seed,fold):
    import torch
    from src.trimodal_interaction import train
    label=f'{arm}_seed{seed}_fold{fold}';original=json.loads((P/'cells'/f'{label}.json').read_text());base=cfg();bc=json.loads(CONFIG.read_text())
    assert original['source']==source()
    oldpath=P/'checkpoints'/f'{label}.pt';assert digest(oldpath)==original['checkpoint_sha256']
    target=P/'budget160/cells'/f'{label}.json';target.parent.mkdir(parents=True,exist_ok=True)
    if original['summary']['actual_epochs']<base['training']['max_epochs']:
        write(target,{**original,'budget_source':budget_source(),'reused_reference':True,'checkpoint_path':str(oldpath),'reason':'same early stopping completes before either cap'});return
    bc160=copy.deepcopy(base);bc160['training']['max_epochs']=bc['max_epochs']
    train.cfg=lambda:bc160
    result=train.train_cell(arm,seed,fold);old=torch.load(oldpath,map_location='cpu',weights_only=False)
    fields=['epoch','train_loss','penalty','validation_loss','validation_loss_EMA','validation_Harrell','gradient_norm','query_gradient','gate_gradient']
    max_error=0.0
    for a,b in zip(old['history'],result['history']):
        for k in fields:max_error=max(max_error,abs(a[k]-b[k]))
    if len(result['history'])<len(old['history']) or max_error>1e-7:raise ValueError('original schedule prefix changed '+str(max_error))
    result['budget_source']=budget_source();result['original_prefix_max_error']=max_error
    ck=P/'budget160/checkpoints'/f'{label}.pt';ck.parent.mkdir(exist_ok=True);tmp=ck.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(ck)
    write(target,{'source':source(),'budget_source':budget_source(),'reused_reference':False,'checkpoint_path':str(ck),'checkpoint_sha256':digest(ck),'summary':result['summary'],'original_prefix_max_error':max_error})

def main():
    if len(sys.argv)>1:return cell(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    c=cfg();done=json.loads((P/'DONE.json').read_text());assert done['source']==source()
    output=P/'budget160';output.mkdir(exist_ok=True);(R/'budget160').mkdir(exist_ok=True)
    lock={'budget_source':budget_source(),'original_source':source(),'config':json.loads(CONFIG.read_text()),'original_cells_complete_before_budget_repair':True}
    if (output/'LOCK.json').exists():assert json.loads((output/'LOCK.json').read_text())==lock
    write(output/'LOCK.json',lock);write(R/'budget160/PROTOCOL_LOCK.json',lock)
    jobs=[];trigger={}
    for arm in c['arms']:
        rows=[json.loads((P/'cells'/f'{arm}_seed{seed}_fold{fold}.json').read_text())['summary'] for fold in c['folds'] for seed in ([31] if arm=='A0' else c['seeds'])]
        trigger[arm]=sum(q['hit_maximum'] for q in rows)/len(rows)
        for fold in c['folds']:
            for seed in ([31] if arm=='A0' else c['seeds']):jobs.append((arm,seed,fold))
    if not any(v>.2 for k,v in trigger.items() if k!='A0'):raise ValueError('budget sensitivity not triggered')
    write(R/'budget160/TRIGGER.json',{'status':'TRIGGERED','cap40_fraction_per_arm':trigger,'outer_results_seen':False,'criterion':'per arm >.2'})
    pending=[]
    for arm,seed,fold in jobs:
        receipt=output/'cells'/f'{arm}_seed{seed}_fold{fold}.json'
        if receipt.exists():
            q=json.loads(receipt.read_text());assert q['budget_source']==budget_source() and digest(q['checkpoint_path'])==q['checkpoint_sha256']
        else:pending.append((arm,seed,fold))
    active=[]
    while pending or active:
        while pending and len(active)<6:
            arm,seed,fold=pending.pop(0);label=f'{arm}_seed{seed}_fold{fold}';handle=(output/f'{label}.log').open('w')
            child=subprocess.Popen([sys.executable,str(Path(__file__)),arm,str(seed),str(fold)],stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT);active.append((child,handle,label))
        for item in list(active):
            p,h,label=item;code=p.poll()
            if code is None:continue
            h.close();active.remove(item)
            if code:
                for pp,hh,_ in active:pp.terminate();hh.close()
                raise SystemExit('BUDGET_FAILED '+label)
            print('BUDGET_DONE',label,len(pending),flush=True)
        time.sleep(1)
    records=[json.loads(p.read_text()) for p in (output/'cells').glob('*.json')]
    write(output/'DONE.json',{'status':'ALL_BUDGET_CELLS_VERIFIED','budget_source':budget_source(),'source':source(),'cells':len(records),'new_real_neural_fits':sum(not r['reused_reference'] for r in records),'evaluation_references':sum(r['reused_reference'] for r in records),'outer_fits':0})
    print('BUDGET_ALL_DONE',flush=True)

if __name__=='__main__':main()
