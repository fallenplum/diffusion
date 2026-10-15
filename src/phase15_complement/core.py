import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit
from scipy.optimize import minimize
import torch
from src.phase13_rna import data as parent
from src.phase13_rna.math import fit_ridge,block_fit,inner_score
from src.phase10_trimodal.features import clinical,scale,ahash

ROOT=Path(__file__).resolve().parents[2];PRIVATE=Path('/root/phase15_complement_private');REPORT=ROOT/'reports/phase15_rna_complementarity';CONFIG=ROOT/'configs/phase15_complementarity.json'
MODELS=['CS','CS_LEGACY','RNA_ONLY','JOINT','ALL','SOFT','HARD','RANDOM','WRONG','NO_OFFSET','SCALAR','JOINT_FIXED1','ALL_FIXED1','SOFT_FIXED1','SCALAR_FIXED1','RANDOM_FIXED1','WRONG_FIXED1']
def cfg():return json.loads(CONFIG.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False,default=cast))
def source():
    files=[CONFIG,ROOT/'src/phase15_complement/core.py',ROOT/'scripts/run_phase15.py',ROOT/'scripts/gate_phase15.py']
    return hashlib.sha256((parent.source_hash()+''.join(str(p.relative_to(ROOT))+sha(p) for p in files)).encode()).hexdigest()
def name(ep,gene,seed,fold):return f'{ep}_{gene}_seed{seed}_fold{fold}'


def cs_features(d,fit):
    raw,coding=clinical(d,fit);c,cc=scale(raw,np.asarray(fit));s,ss=scale(np.asarray(d['static'],float),np.asarray(fit))
    return np.c_[c,s],{'coding':coding,'C':cc,'S':ss,'fit_indices':np.asarray(fit),'fit_hash':ahash(np.asarray(fit,np.int64))}


def rna_scale(x,fit):
    with np.errstate(invalid='ignore'):median=np.nanmedian(x[fit],axis=0)
    median=np.where(np.isfinite(median),median,0);v=np.where(np.isfinite(x),x,median);mean=v[fit].mean(0);sd=v[fit].std(0);sd=np.where(sd>1e-8,sd,1)
    return (v-mean)/sd,{'median':median,'mean':mean,'sd':sd,'fit_indices':np.asarray(fit)}


def patient_pairs(t,e):
    return np.where((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))


def select_cs(d,t,e,fit,seed,folds):
    scores={lam:[0.,0.] for lam in cfg()['anchor_grid']}
    for a,b in StratifiedKFold(folds,shuffle=True,random_state=seed).split(fit,e[fit]):
        train,valid=fit[a],fit[b];x,_=cs_features(d,train);i,j=patient_pairs(t[valid],e[valid])
        for lam in scores:
            w,_=fit_ridge(x[train],t[train],e[train],lam);delta=(x[valid]@w)[i]-(x[valid]@w)[j];scores[lam][0]+=np.where(abs(delta)<=1e-8,.5,delta>0).sum();scores[lam][1]+=len(i)
    winner=max(scores,key=lambda lam:scores[lam][0]/scores[lam][1])
    return winner,scores


def anchor(d,ep,fold,seed,role):
    target=PRIVATE/'anchors'/f'{ep}_seed{seed}_fold{fold}_{role}.pt'
    if target.exists():
        z=torch.load(target,map_location='cpu',weights_only=False)
        if z['source_hash']!=source():raise RuntimeError('stale anchor')
        return z
    ev=parent.endpoints(d)[ep];t,e=ev['time'],ev['event'];part=parent.splits(d,ep,fold);fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test']
    group=np.full(len(d['patient_ids']),-1,int);oof=np.full(len(group),np.nan);teachers=[]
    for k,(aa,bb) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+fold*100).split(fit,e[fit])):
        train,held=fit[aa],fit[bb];lam,scores=select_cs(d,t,e,train,seed+fold*100+k+1,2);x,audit=cs_features(d,train);w,info=fit_ridge(x[train],t[train],e[train],lam)
        oof[held]=x[held]@w;group[held]=k;teachers.append({'train':train,'held':held,'lambda':lam,'selection':scores,'preprocessing':audit,'w':w,'optimizer':info})
    lam,scores=select_cs(d,t,e,fit,seed+fold*100,3);x,audit=cs_features(d,fit);w,info=fit_ridge(x[fit],t[fit],e[fit],lam)
    z={'fit':fit,'test':test,'oof':oof,'group':group,'teachers':teachers,'lambda':lam,'selection':scores,'w':w,'preprocessing':audit,'optimizer':info,'full_fit_risk':x[fit]@w,'test_risk':x[test]@w,'source_hash':source()}
    target.parent.mkdir(parents=True,exist_ok=True);torch.save(z,target);return z


def honest_pairs(a,t,e):
    fit=a['fit'];i,j=patient_pairs(t[fit],e[fit]);keep=a['group'][fit[i]]==a['group'][fit[j]];i,j=i[keep],j[keep]
    if not len(i):raise RuntimeError('no honest comparable pairs')
    for k,teacher in enumerate(a['teachers']):
        rows=a['group'][fit[i]]==k;endpoints=set(fit[i[rows]])|set(fit[j[rows]])
        if endpoints&set(teacher['train']) or not endpoints<=set(teacher['held']):raise RuntimeError('pair teacher leakage')
    return i,j,a['oof'][fit[i]]-a['oof'][fit[j]]


def weights(kind,margin,i,seed):
    if kind in ['ALL','SCALAR']:return np.ones(len(i))
    if kind=='HARD':return (margin<=0).astype(float)
    w=expit(-margin)
    if kind=='RANDOM':
        rng=np.random.default_rng(seed)
        for event in np.unique(i):
            positions=np.flatnonzero(i==event);w[positions]=rng.permutation(w[positions])
    return w


def pair_fit(x,ii,jj,offset,weight,lam):
    dimension=x.shape[1]
    if weight.sum()==0:return np.zeros(dimension),{'gradient_max':0.,'iterations':0,'no_eligible_pairs':True}
    device='cuda' if torch.cuda.is_available() else 'cpu';D=torch.as_tensor(x[ii]-x[jj],dtype=torch.float64,device=device);b=torch.as_tensor(offset,dtype=torch.float64,device=device);q=torch.as_tensor(weight/weight.sum(),dtype=torch.float64,device=device)
    def objective(w):
        v=torch.as_tensor(w,dtype=torch.float64,device=device);m=b+D@v;loss=(torch.nn.functional.softplus(-m)*q).sum()+lam/2*(v@v);gradient=D.T@(-torch.sigmoid(-m)*q)+lam*v
        return float(loss),gradient.cpu().numpy()
    result=minimize(objective,np.zeros(dimension),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-13,'gtol':1e-7,'maxls':50});w=result.x.copy();loss,grad=objective(w)
    if abs(grad).max()>2e-5 or not np.isfinite(loss):raise RuntimeError('pair objective not stationary')
    return w,{'gradient_max':float(abs(grad).max()),'iterations':int(result.nit),'loss':loss,'pairs':len(ii),'weight_sum':float(weight.sum()),'parameters':dimension}


def rna_teacher(x,a,t,e):
    fit=a['fit'];oof=np.full(len(x),np.nan);teachers=[]
    for teacher in a['teachers']:
        train,held=teacher['train'],teacher['held'];v,st=rna_scale(x,train);w,info=fit_ridge(v[train],t[train],e[train],1.);risk=v[train]@w;sd=max(risk.std(),1e-8);oof[held]=(v[held]@w-risk.mean())/sd
        teachers.append({'train':train,'held':held,'preprocessing':st,'w':w,'risk_mean':risk.mean(),'risk_sd':sd,'optimizer':info})
    v,st=rna_scale(x,fit);w,info=fit_ridge(v[fit],t[fit],e[fit],1.);risk=v[fit]@w;sd=max(risk.std(),1e-8)
    return {'oof':oof,'teachers':teachers,'w':w,'preprocessing':st,'risk_mean':risk.mean(),'risk_sd':sd,'test':(v[a['test']]@w-risk.mean())/sd,'fit':risk,'optimizer':info}
