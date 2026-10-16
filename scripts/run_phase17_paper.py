import sys,argparse,time,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from src.phase17_ssd.train import Data,seed_all
from src.phase17_ssd.data import P,R,cfg,source,write,parent
from src.phase17_ssd.losses import full_risk_efron
from src.phase13_rna.math import inner_score
from src.phase10_trimodal.data import digest
from phase17_paper_models import PaperModel

def paper_source():
    import hashlib
    return hashlib.sha256((source()+digest(Path(__file__))+digest(ROOT/'scripts/phase17_paper_models.py')).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['MGCT','MambaMIL'],required=True);p.add_argument('--seed',type=int,default=31);p.add_argument('--fold',type=int,default=1);p.add_argument('--stage',choices=['pilot','inner','outer'],default='pilot');a=p.parse_args()
    gate=json.loads((R/'PAPER_IMPLEMENTATION_AUDIT.json').read_text());assert next(v for v in gate['methods'] if v['method']==a.model)['status']=='PASS'
    stem=f'{a.model}_seed{a.seed}_fold{a.fold}';target=P/('paper_'+a.stage)/(stem+'.pt');target.parent.mkdir(exist_ok=True);cell=target.with_suffix('.json')
    if cell.exists():assert json.loads(cell.read_text())['paper_source']==paper_source();return
    role='outer' if a.stage=='outer' else 'inner';budget=128 if a.stage=='pilot' else 512;tc=cfg()['training'];epochs=tc['max_epochs']
    if a.stage=='outer':
        lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert lock['training_source']==source()
        previous=torch.load(P/'paper_inner'/(stem+'.pt'),map_location='cpu',weights_only=False);assert previous['paper_source']==paper_source();epochs=previous['selected_epoch']
    seed_all(a.seed+a.fold*100);ds=Data(a.fold,role,budget,'A2',a.seed);model=PaperModel(a.model,ds.f['C'].shape[1],ds.f['membership']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    ev=parent.endpoints(ds.d)['DSS'];t,e=ev['time'],ev['event'];history=[];best=float('inf');best_epoch=0;best_state=None;ema=None;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    batch=128 if a.model=='MGCT' else 32
    def predict(ix,training=False):return torch.cat([ds.forward(model,ix[k:k+batch],training=training) for k in range(0,len(ix),batch)])
    for ep in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True);risk=predict(ds.fit,True);dl=full_risk_efron(risk,t[ds.fit],e[ds.fit]);reg=.5*model.direct.weight.square().sum();loss=dl+reg
        if not torch.isfinite(loss):raise ValueError('nonfinite paper loss')
        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));active=sum(p.numel() for p in model.parameters() if p.grad is not None);opt.step()
        row={'epoch':ep,'train_loss':float(dl.detach()),'penalty':float(reg.detach()),'gradient_norm':gn}
        if role=='inner':
            model.eval()
            with torch.no_grad():v=predict(ds.valid);vl=float(full_risk_efron(v,t[ds.valid],e[ds.valid]));hc=inner_score(t[ds.valid],e[ds.valid],v.cpu().numpy())
            ema=vl if ema is None else .2*vl+.8*ema;row.update(validation_loss=vl,validation_loss_EMA=ema,validation_Harrell=hc)
            if ema<best-1e-8:best=ema;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        if ep==1 or ep%20==0:print('PAPER_TRAIN',a.model,a.seed,a.fold,a.stage,ep,round(row['train_loss'],5),row.get('validation_Harrell','NA'),round(time.monotonic()-start,1),flush=True)
        if role=='inner' and ep-best_epoch>=tc.get('early_stopping_patience',10000):break
    if role=='inner':model.load_state_dict(best_state)
    else:best_epoch=epochs
    model.eval()
    with torch.no_grad():risk=predict(ds.valid).cpu().numpy()
    summary={'registered_parameters':sum(p.numel() for p in model.parameters()),'last_step_gradient_parameters':active,'seconds':time.monotonic()-start,'peak_GPU_MB':torch.cuda.max_memory_allocated()/2**20,'gamma':float(model.gamma.detach())}
    z={'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arm':a.model,'seed':a.seed,'fold':a.fold,'role':role,'budget':budget,'source':source(),'paper_source':paper_source(),'fit':ds.fit,'validation_or_test':ds.valid,'selected_epoch':best_epoch,'history':history,'preprocess_audit':ds.audit,'anchor_w':ds.anchor_w,'anchor_risk':ds.anchor.cpu().numpy(),'risk':risk,'summary':summary}
    tmp=target.with_suffix('.tmp');torch.save(z,tmp);tmp.replace(target)
    write(cell,{'status':'COMPLETE','source':source(),'paper_source':paper_source(),'checkpoint_sha256':digest(target),'selected_epoch':best_epoch,'summary':summary,'outer_test_selection':False})
    print('PAPER_COMPLETE',a.model,a.stage,stem,best_epoch,flush=True)

if __name__=='__main__':main()
