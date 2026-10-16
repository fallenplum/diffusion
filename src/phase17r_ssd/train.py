import time
import numpy as np
import torch
from .data import Data,seed_all,cfg,source
from .model import RepairS6
from src.phase17_ssd.losses import full_risk_efron
from src.phase13_rna.math import inner_score

def grad_group(norms,prefixes):
    return float(sum(v*v for k,v in norms.items() if any(k.startswith(p) for p in prefixes))**.5)

def train_cell(arm,seed,fold,role,scale=1.,epochs=None,direction='forward'):
    seed_all(seed+100*fold);tc=cfg()['training'];ds=Data(fold,role,seed,direction)
    model=RepairS6(ds.f['C'].shape[1],ds.f['membership'],arm,scale,tc['dropout']).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    event=__import__('src.phase13_rna.data',fromlist=['endpoints']).endpoints(ds.d)['DSS'];t,e=event['time'],event['event']
    history=[];best=float('inf');best_epoch=0;best_state=None;ema=None
    length=epochs or tc['max_epochs'];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,length+1):
        model.train();opt.zero_grad(set_to_none=True);risk=ds.forward(model,ds.fit,training=True)
        data_loss=full_risk_efron(risk,t[ds.fit],e[ds.fit])
        reg=.5*tc['direct_l2']*model.direct.weight.square().sum() if model.direct is not None else data_loss.new_zeros(())
        loss=data_loss+reg
        if not torch.isfinite(loss):raise ValueError('nonfinite full-risk loss')
        loss.backward()
        norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip']));opt.step()
        row={'epoch':ep,'train_loss':float(data_loss.detach()),'penalty':float(reg.detach()),'gradient_norm':gn,'gamma':float(model.gamma.detach()),'g_attention_grad':grad_group(norms,['q.','k.','v.','attn_out.']),'delta_grad':grad_group(norms,['delta_']),'B_condition_grad':grad_group(norms,['B_down.','B_up.']),'write_grad':grad_group(norms,['write.']),'molecular_shared_grad':grad_group(norms,['molecular.']),'late_grad':grad_group(norms,['late.']),'direct_grad':grad_group(norms,['direct.']),'readout_grad':grad_group(norms,['readout.']),'gamma_grad':norms.get('gamma',0.)}
        if role=='inner':
            model.eval()
            with torch.no_grad():v=ds.forward(model,ds.valid);vl=float(full_risk_efron(v,t[ds.valid],e[ds.valid]));hc=inner_score(t[ds.valid],e[ds.valid],v.cpu().numpy())
            alpha=tc['validation_ema_alpha'];ema=vl if ema is None else alpha*vl+(1-alpha)*ema
            row.update(validation_loss=vl,validation_loss_EMA=ema,validation_Harrell=hc)
            if ema<best-1e-8:best=ema;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        if ep==1 or ep%20==0:print('TRAIN',arm,scale,seed,fold,role,ep,round(row['train_loss'],5),row.get('validation_Harrell','NA'),round(time.monotonic()-start,1),flush=True)
        if role=='inner' and ep-best_epoch>=tc['early_stopping_patience']:break
    if role=='inner':model.load_state_dict(best_state)
    else:best_epoch=length
    model.eval()
    with torch.no_grad():
        pred=ds.forward(model,ds.valid);wrong=ds.forward(model,ds.valid,wrong_inference=True)
        off=ds.forward(model,ds.valid,module_off=True);saved_gamma=model.gamma.clone();model.gamma.fill_(1.)
        gamma_one=ds.forward(model,ds.valid);model.gamma.copy_(saved_gamma)
    from .diagnostics import diagnose
    mechanism=diagnose(model,ds)
    selected_row=history[best_epoch-1]
    return {'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arm':arm,'scale':scale,'seed':seed,'fold':fold,'role':role,'direction':direction,'budget':cfg()['patch_budget'],'source':source(),'fit':ds.fit,'validation_or_test':ds.valid,'selected_epoch':best_epoch,'history':history,'selected_training_row':selected_row,'preprocess_audit':ds.audit,'anchor_w':ds.anchor_w,'anchor_risk':ds.anchor.cpu().numpy(),'wrong_map':ds.wrong,'risk':pred.cpu().numpy(),'wrong_risk':wrong.cpu().numpy(),'SSM_off_risk':off.cpu().numpy(),'gamma_one_risk':gamma_one.cpu().numpy(),'mechanism':mechanism,'best_validation_EMA':best if role=='inner' else None,'summary':{'registered_parameters':sum(p.numel() for p in model.parameters()),'last_step_gradient_parameters':sum(p.numel() for k,p in model.named_parameters() if k in norms),'gradient_norms_last_step':norms,'seconds':time.monotonic()-start,'peak_GPU_MB':torch.cuda.max_memory_allocated()/2**20,'gamma':float(model.gamma.detach())}}
