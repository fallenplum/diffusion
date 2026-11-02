"""One full-risk-set DSS fit; own RNA main effect never changes under controls."""
from __future__ import annotations
import argparse,copy,hashlib,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.qpr.protocol import require_stage
from src.qpr.survival_model import FullRelation
from src.rscil.geometry import EfronRiskSets
from src.bgb_en.survival import evaluate


def mismatch(data,rows):
    rows=np.asarray(rows,int);out=np.arange(512);exact=0
    for i in rows:
        pool=[j for j in data['fit'] if j!=i and data['cancer'][j]==data['cancer'][i] and data['stage'][j]==data['stage'][i]]
        if pool:exact+=1
        else:pool=[j for j in data['fit'] if j!=i and data['cancer'][j]==data['cancer'][i]]
        if not pool:pool=[j for j in data['fit'] if j!=i]
        if not pool:raise ValueError('NO_LEGAL_WRONG_QUERY_DONOR')
        out[i]=pool[int.from_bytes(hashlib.blake2b(str(int(i)).encode(),digest_size=8).digest(),'little')%len(pool)]
    return out,{'mapping_fraction':1.,'exact_cancer_stage_fraction':exact/len(rows),'donors_training_partition_only':True,'no_event_matching':True}


class Evidence:
    def __init__(self,data,arm):
        self.rows=np.r_[data['fit'],data['evaluation']];self.ntrain=len(data['fit']);self.data=data
        self.program=torch.tensor(data['program'],dtype=torch.float64,device='cuda')
        self.c=torch.tensor(data['clinical'],dtype=torch.float64,device='cuda');self.s=torch.tensor(data['s32'],dtype=torch.float64,device='cuda')
        self.mask=torch.tensor(data['program_mask'],device='cuda')
        self.mapping,self.mapping_audit=mismatch(data,self.rows)
        self.owner_to_slides={int(o):[i for i,g in enumerate(data['graphs']) if g['owner']==o] for o in self.rows}
        if any(not v for v in self.owner_to_slides.values()):raise ValueError('PATIENT_WITHOUT_SLIDE')
        view='wrong' if arm=='B4' else 'content' if arm=='B2' else 'true'
        maxe=max(len(g[view][0]) for g in data['graphs']);ns=len(data['graphs'])
        self.k=torch.zeros(ns,maxe,32,device='cuda');self.v=torch.zeros(ns,maxe,64,device='cuda');self.edge_mask=torch.zeros(ns,maxe,dtype=torch.bool,device='cuda')
        for i,g in enumerate(data['graphs']):
            k,v=g[view];n=len(k);self.k[i,:n]=torch.as_tensor(k,device='cuda');self.v[i,:n]=torch.as_tensor(v,device='cuda');self.edge_mask[i,:n]=True
        self.arm=arm;self.constant=None

    def constant_query(self,model):
        with torch.no_grad():
            rows=torch.as_tensor(self.data['fit'],device='cuda');q=model.query(self.program[rows],self.c[rows],self.s[rows]);m=self.mask[rows]
            self.constant=(q*m[...,None]).sum(0)/m.sum(0)[:,None].clamp_min(1)

    def forward(self,model,positions):
        owners=self.rows[np.asarray(positions,int)];ids=[];local=[]
        for i,o in enumerate(owners):
            for sid in self.owner_to_slides[int(o)]:ids.append(sid);local.append(i)
        real=torch.tensor(owners[np.asarray(local)],device='cuda');donor=torch.tensor(self.mapping[owners[np.asarray(local)]],device='cuda') if self.arm=='B5' else real
        si=torch.tensor(ids,device='cuda');index=torch.tensor(local,device='cuda')
        delta=model(self.program[donor],self.c[real],self.s[real],self.k[si],self.v[si],self.edge_mask[si],self.mask[donor],constant_query=self.constant)
        total=delta.new_zeros(len(owners));count=delta.new_zeros(len(owners));total.index_add_(0,index,delta);count.index_add_(0,index,torch.ones_like(delta))
        return total/count

    @torch.no_grad()
    def predict(self,model,positions):
        return torch.cat([self.forward(model,positions[i:i+8]) for i in range(0,len(positions),8)])


def main(a):
    torch.set_num_threads(1);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    require_stage('gate2',a.reports);lock=json.loads(Path(a.lock).read_text())
    if lock['status']!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('G2_NOT_LOCKED')
    for name,h in lock['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('G2_SOURCE_CHANGED:'+name)
    receipt=json.loads(Path(a.cache_receipt).read_text());h=hashlib.sha256(Path(a.cache).read_bytes()).hexdigest()
    if h!=receipt['cache_sha256'] or receipt['outer1_test_endpoint_elements_decoded']!=0:raise ValueError('G2_DATA_IDENTITY')
    out=Path(a.output)/f'inner{a.inner}_{a.arm}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.private.json').exists():raise ValueError('EXISTING_COMPLETED_MODEL')
    with Path(a.cache).open('rb') as f:data=pickle.load(f)
    store=Evidence(data,a.arm);model=FullRelation(data['clinical'].shape[1],a.arm).cuda();store.constant_query(model)
    nt=len(data['fit']);nv=len(data['evaluation']);tr=np.arange(nt);va=np.arange(nt,nt+nv)
    r0=torch.as_tensor(data['r0_fit'],device='cuda');r0v=torch.as_tensor(data['r0_evaluation'],device='cuda')
    rs=EfronRiskSets(data['train_time'],data['train_event'],device='cuda');rv=EfronRiskSets(data['evaluation_time'],data['evaluation_event'],device='cuda')
    named=list(model.named_parameters());qparams=[p for n,p in named if n.startswith('query.')];heads=[p for n,p in named if not n.startswith('query.')]
    optimizer=torch.optim.AdamW([{'params':qparams,'lr':3e-5},{'params':heads,'lr':3e-4}],weight_decay=1e-4)
    sd=r0.std(unbiased=False)+1e-6;history=[];snapshots=[];best=None;bestloss=float('inf');started=time.monotonic();updates={'query':0,'head':0}
    def snapshot(epoch):
        nonlocal best,bestloss
        model.eval();model.stage(epoch)
        with torch.no_grad():dt=store.predict(model,tr);dv=store.predict(model,va);vt=float(rv.loss(r0v+dv))
        path=out/f'epoch{epoch:03d}.private.pt';torch.save({'state':model.state_dict(),'epoch':epoch,'constant_query':store.constant},path)
        restored=FullRelation(data['clinical'].shape[1],a.arm).cuda();saved=torch.load(path,weights_only=True);restored.load_state_dict(saved['state']);restored.stage(epoch);restored.eval()
        replay=max(float((store.predict(restored,tr)-dt).abs().max()),float((store.predict(restored,va)-dv).abs().max()))
        if replay>1e-10:raise ValueError('G2_DISK_REPLAY_FAILED')
        metrics=evaluate(data['train_time'],data['train_event'],data['evaluation_time'],data['evaluation_event'],(r0v+dv).cpu().numpy(),data['r0_evaluation'])
        row={'epoch':epoch,'validation_Cox':vt,'train_Cox':float(rs.loss(r0+dt)),
             'train_delta_SD':float(dt.std(unbiased=False)),'validation_delta_SD':float(dv.std(unbiased=False)),
             'normalized_train_delta_abs_p50_p95':torch.quantile((dt/sd).abs(),torch.tensor([.5,.95],dtype=torch.float64,device='cuda')).cpu().tolist(),
             'metrics':metrics,'query_updates':updates['query'],'head_updates':updates['head'],'replay_max_abs':replay,
             'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        snapshots.append(row)
        np.savez_compressed(out/f'epoch{epoch:03d}_predictions.private.npz',fit_delta=dt.cpu().numpy(),evaluation_delta=dv.cpu().numpy(),evaluation_risk=(r0v+dv).cpu().numpy())
        if vt<bestloss:bestloss=vt;best=epoch
    snapshot(0)
    for epoch in range(1,81):
        model.train();model.stage(epoch);optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():d=store.predict(model,tr)
        leaf=d.detach().requires_grad_();lc=rs.loss(r0+leaf);shrink=.01*((leaf/sd)**2).mean();gradient=torch.autograd.grad(lc+shrink,leaf)[0].detach()
        coxgrad=torch.autograd.grad(rs.loss(r0+leaf),leaf)[0].detach()
        before={n:p.detach().clone() for n,p in named if p.requires_grad};probe={};max_replay=0.
        for start in range(0,nt,8):
            ix=tr[start:start+8];dd=store.forward(model,ix);max_replay=max(max_replay,float((dd.detach()-d[ix]).abs().max()))
            if start==0 and epoch in (1,20,21,40,80):
                active=[(n,p) for n,p in named if p.requires_grad]
                gg=torch.autograd.grad(torch.dot(dd,coxgrad[ix]),[p for _,p in active],retain_graph=True,allow_unused=True)
                for group,prefix in [('query','query.'),('head','')]:
                    vals=[g.square().sum() for (n,_),g in zip(active,gg) if g is not None and (n.startswith('query.') if group=='query' else not n.startswith('query.'))]
                    probe[group+'_Cox_gradient_first_batch_norm']=float(torch.stack(vals).sum().sqrt()) if vals else 0.
            torch.dot(dd,gradient[ix]).backward()
        if max_replay>1e-10:raise ValueError('FULL_RISK_GRADIENT_CACHE_NOT_EXACT')
        explicit_l2=1e-4*sum(p.square().sum() for n,p in named if p.requires_grad);explicit_l2.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
        if not np.isfinite(norm) or not torch.isfinite(lc+shrink+explicit_l2):raise FloatingPointError('NONFINITE_G2_UPDATE')
        optimizer.step()
        for group in updates:
            updates[group]+=int(any(not torch.equal(p.detach(),before[n]) for n,p in named if n in before and (n.startswith('query.') if group=='query' else not n.startswith('query.'))))
        history.append({'epoch':epoch,'train_Cox':float(lc.detach()),'shrink_penalty':float(shrink.detach()),'explicit_L2':float(explicit_l2.detach()),'grad_norm_before_clip':norm,'gradient_cache_replay_error':max_replay,**probe})
        (out/'PROGRESS.private.json').write_text(json.dumps({'epoch':epoch,'actual_parameter_updates':updates,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
        if epoch in (10,20,40,80):snapshot(epoch)
        if epoch%10==0:print(json.dumps({'inner':a.inner,'arm':a.arm,'seed':a.seed,'epoch':epoch}),flush=True)
    baseline=evaluate(data['train_time'],data['train_event'],data['evaluation_time'],data['evaluation_event'],data['r0_evaluation'])
    selected=next(s for s in snapshots if s['epoch']==best)
    complete={'inner':a.inner,'arm':a.arm,'seed':a.seed,'status':'COMPLETE','role':'OUTER1_TRAIN_SIDE_DEVELOPMENT',
              'selected_epoch':best,'selected':selected,'baseline_metrics':baseline,'snapshots':snapshots,'history':history,
              'total_parameters':sum(p.numel() for p in model.parameters()),'effective_parameters':model.effective_parameters(),
              'optimizer_updates':80,'actual_parameter_updates':updates,'elapsed_seconds':time.monotonic()-started,
              'cache_sha256':h,'formal_lock_sha256':hashlib.sha256(Path(a.lock).read_bytes()).hexdigest(),
              'wrong_query_mapping':store.mapping_audit if a.arm=='B5' else None,'CPU_validation_labels_not_in_optimizer':True,
              'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'original_RNA_main_risk_unchanged':True}
    (out/'COMPLETE.private.json').write_text(json.dumps(complete,indent=2,allow_nan=False)+'\n');print(json.dumps({'inner':a.inner,'arm':a.arm,'seed':a.seed,'status':'COMPLETE'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('cache','cache-receipt','output','reports','lock'):p.add_argument('--'+x,required=True)
    p.add_argument('--inner',required=True,type=int);p.add_argument('--seed',required=True,type=int);p.add_argument('--arm',required=True);main(p.parse_args())
