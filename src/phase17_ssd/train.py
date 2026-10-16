import copy,time,random,json
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .data import P,R,cfg,write,source,molecular_features,wrong_map,parent
from .model import ConditionalS6
from .losses import full_risk_efron
from src.phase10_trimodal.math import fit_ridge
from src.phase13_rna.math import inner_score


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


class Data:
    def __init__(self,fold,role,budget,arm,seed):
        self.d=parent.load();self.part=parent.splits(self.d,'DSS',fold);self.fit=self.part['inner_train' if role=='inner' else 'outer_train'];self.valid=self.part['inner_val' if role=='inner' else 'outer_test']
        self.f,self.audit=molecular_features(self.d,fold,role);self.budget=budget
        v=P/'visual';self.owner=np.load(v/'owner.npy');mask=np.load(v/'mask.npy')
        # Nested geometry-only uniform subsampling of the locked 512-ROI set.
        xx=np.load(v/'u.npy',mmap_mode='r');self.visual=torch.zeros(len(mask),budget,1536,device='cuda')
        mm=np.zeros((len(mask),budget),bool);cap=np.zeros(len(mask),int)
        for pi in range(512):
            ss=np.flatnonzero(self.owner==pi);nn=mask[ss].sum(1);use=np.zeros(len(ss),int)
            for _ in range(min(budget,int(nn.sum()))):
                score=nn/(use+1);score[use>=nn]=-1;use[np.argmax(score)]+=1
            cap[ss]=use
        for i,q in enumerate(cap):
            take=np.linspace(0,mask[i].sum()-1,q,dtype=int);self.visual[i,:q]=torch.as_tensor(np.asarray(xx[i,take],np.float32),device='cuda');mm[i,:q]=True
        self.mask=torch.as_tensor(mm,device='cuda');self.ft={k:torch.as_tensor(v,device='cuda') for k,v in self.f.items() if k!='membership'}
        cs=np.c_[self.f['C'],self.f['S']]
        if arm=='A11_NO_S32':cs=self.f['C']
        if arm=='A11_NO_CLIN':cs=self.f['S']
        w,self.anchor_diagnostic=fit_ridge(cs[self.fit],parent.endpoints(self.d)['DSS']['time'][self.fit],parent.endpoints(self.d)['DSS']['event'][self.fit],cfg()['training']['anchor_lambda'])
        self.anchor=torch.as_tensor(cs@w,device='cuda',dtype=torch.float32);self.anchor_w=w
        self.wrong=np.arange(512)
        for ix in [self.fit,self.valid]:self.wrong[ix]=wrong_map(self.d,ix,seed+1000+fold)[ix]
        self.arm=arm;self.role=role

    def forward(self,model,indices,training=False,order='original',gate_override=None,module_off=False,wrong_inference=False,diagnostic=False):
        outputs=[];diags=[];indices=np.asarray(indices,int)
        chunk=cfg()['training']['microbatch_patients']
        for start in range(0,len(indices),chunk):
            patients=indices[start:start+chunk];slides=np.flatnonzero(np.isin(self.owner,patients));owners=self.owner[slides];f=self.ft
            x=self.visual[slides];mask=self.mask[slides]
            if order!='original':
                x=x.clone()
                for k in range(len(slides)):
                    n=int(mask[k].sum());p=torch.arange(n-1,-1,-1,device='cuda') if order=='reverse' else torch.as_tensor(np.random.default_rng(1731+int(slides[k])).permutation(n),device='cuda')
                    x[k,:n]=x[k,p]
            own=[f[k][owners] for k in ['R','G','L','RM','CM']]
            ctx=self.wrong[owners] if self.arm=='A10' or wrong_inference else owners
            context=tuple(f[k][ctx] for k in ['R','G','L','RM','CM'])
            def call(xx,ma,cc,rr,gg,ll,rmm,cmm,aa,*co):
                return model(xx,ma,cc,rr,gg,ll,rmm,cmm,aa,context_omics=co,gate_override=gate_override,module_off=module_off,diagnostic=diagnostic)
            args=(x,mask,f['C'][owners],*own,self.anchor[owners],*context)
            result=checkpoint(call,*args,use_reentrant=False) if training else call(*args)
            if diagnostic:result,dg=result;diags.append({k:v.detach().cpu() for k,v in dg.items()})
            # Equal slide weights, no latent mixing between independently fitted encoders.
            outputs.append(torch.stack([result[torch.as_tensor(owners==pi,device='cuda')].mean() for pi in patients]))
        risk=torch.cat(outputs)
        return (risk,diags) if diagnostic else risk


def train_cell(arm,seed,fold,role,budget,epochs=None):
    seed_all(seed+fold*100);tc=cfg()['training'];ds=Data(fold,role,budget,arm,seed)
    model=ConditionalS6(ds.f['C'].shape[1],ds.f['membership'],arm,tc['dropout']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    ev=parent.endpoints(ds.d)['DSS'];t,e=ev['time'],ev['event'];history=[];best=float('inf');best_epoch=0;best_state=None;ema=None
    length=epochs or tc['max_epochs'];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,length+1):
        model.train();opt.zero_grad(set_to_none=True)
        risk=ds.forward(model,ds.fit,training=True)
        data_loss=full_risk_efron(risk,t[ds.fit],e[ds.fit])
        reg=tc['RNA_direct_l2']*.5*model.direct.weight.square().sum()
        loss=data_loss+reg
        if not torch.isfinite(loss):raise ValueError('nonfinite full-risk loss')
        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip']))
        norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
        opt.step()
        row={'epoch':ep,'train_loss':float(data_loss.detach()),'penalty':float(reg.detach()),'gradient_norm':gn,'gamma':float(model.gamma.detach())}
        if role=='inner':
            model.eval()
            with torch.no_grad():v=ds.forward(model,ds.valid);vl=float(full_risk_efron(v,t[ds.valid],e[ds.valid]));hc=inner_score(t[ds.valid],e[ds.valid],v.cpu().numpy())
            ema=vl if ema is None else .2*vl+.8*ema;row.update(validation_loss=vl,validation_loss_EMA=ema,validation_Harrell=hc)
            if ep>=min(tc['minimum_epoch'],length) and ema<best-1e-8:best=ema;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        if ep==1 or ep%20==0:print('TRAIN',arm,seed,fold,role,ep,round(row['train_loss'],5),row.get('validation_Harrell','NA'),round(time.monotonic()-start,1),flush=True)
    if role=='inner':model.load_state_dict(best_state)
    else:best_epoch=length
    model.eval()
    # Inner result contains only inner-train/validation risks, NEVER outer-test.
    with torch.no_grad():pred,diag=ds.forward(model,ds.valid,diagnostic=True)
    gradients={k:v for k,v in norms.items()};registered=sum(p.numel() for p in model.parameters());active=sum(p.numel() for k,p in model.named_parameters() if k in gradients)
    summary={'registered_parameters':registered,'last_step_gradient_parameters':active,'gradient_norms':gradients,'seconds':time.monotonic()-start,'peak_GPU_MB':torch.cuda.max_memory_allocated()/2**20,'gamma':float(model.gamma.detach())}
    states={'m':[],'dt':[],'B':[],'conditional_delta':[],'conditional_B':[],'g':[],'h':[],'ssm_risk':[]}
    for q in diag:
        for key in states:states[key].append(q[key].float().reshape(-1).numpy())
    mechanism={}
    for key,items in states.items():
        a=np.concatenate(items);mechanism.update({key+'_mean':float(a.mean()),key+'_std':float(a.std()),key+'_absmax':float(abs(a).max())})
    mechanism['gate_below_001']=float(np.mean(np.concatenate(states['m'])<.01));mechanism['gate_above_099']=float(np.mean(np.concatenate(states['m'])>.99))
    ck={'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'fold':fold,'role':role,'budget':budget,'source':source(),'fit':ds.fit,'validation_or_test':ds.valid,'selected_epoch':best_epoch,'history':history,'preprocess_audit':ds.audit,'anchor_w':ds.anchor_w,'anchor_risk':ds.anchor.cpu().numpy(),'wrong_map':ds.wrong,'risk':pred.cpu().numpy(),'summary':summary,'mechanism':mechanism}
    return ck
