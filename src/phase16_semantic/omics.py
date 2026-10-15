"""Small fixed molecular representations; common IDs, separate RNA/CNA evidence."""
import warnings
import numpy as np,torch
from scipy.special import expit
from src.phase16_semantic.core import *
from src.phase10_trimodal.features import gene_fit
MCONFIG=ROOT/'configs/phase16_omics_continuation.json';MP=Path('/root/phase16_omics_private')
def mcfg():return json.loads(MCONFIG.read_text())
def msource():return hashlib.sha256((source()+sha(MCONFIG)+sha(Path(__file__))+sha(ROOT/'scripts/run_phase16_omics.py')).encode()).hexdigest()
def armdefs():
    out={}
    for selector in mcfg()['selectors']:
        for m in mcfg()['budgets']:
            for kind in mcfg()['inputs']:out[f'G5_{selector}_{m}_{kind}']={'selector':selector,'budget':m,'kind':kind,'random':False}
        for kind in ['RC_GATED']:out[f'G5_{selector}_512_RANDOM_{kind}']={'selector':selector,'budget':512,'kind':kind,'random':True}
    for kind in mcfg()['omics_only_controls']:out['G5_'+kind]={'selector':None,'budget':0,'kind':kind,'random':False}
    if len(out)!=16:raise ValueError('G5 arm inventory')
    return out
def nanmean(x):
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(x,axis=1)
def inputs(d,fit,fold,kind,random_mask):
    rr=np.asarray(d['rna_genes'],float);cc=np.asarray(d['cna_genes'],float);rnames=d['rna_gene_symbols'].astype(str);cnames=d['cna_gene_symbols'].astype(str);clookup={g:j for j,g in enumerate(cnames)};full_z,rst=gene_fit(rr,fit);cgood=np.isfinite(cc[fit]).mean(0)>=.9;eligible=np.array([j for j,g in enumerate(rnames) if rst['usable'][j] and g in clookup and cgood[clookup[g]]]);ci=np.array([clookup[rnames[j]] for j in eligible]);rawR=rr[:,eligible];rawC=cc[:,ci];rz=(rawR-rst['mean'][eligible])/np.where(rst['sd'][eligible]>1e-8,rst['sd'][eligible],1);lookup={g:i for i,g in enumerate(rnames[eligible])};rlookup={g:i for i,g in enumerate(rnames)};groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());permutation=np.arange(len(eligible))
    if random_mask:
        mean=rst['mean'][eligible];sd=rst['sd'][eligible];qm=np.quantile(mean,[.2,.4,.6,.8]);qs=np.quantile(sd,[.2,.4,.6,.8]);bins=np.searchsorted(qm,mean)*5+np.searchsorted(qs,sd);rng=np.random.default_rng(169781+fold)
        for value in np.unique(bins):ix=np.flatnonzero(bins==value);permutation[ix]=rng.permutation(ix)
    full_permutation=np.arange(len(rnames));full_permutation[eligible]=eligible[permutation]
    extra=np.setdiff1d(np.flatnonzero(rst['usable']),eligible)
    if random_mask and len(extra):
        mean=rst['mean'][extra];sd=rst['sd'][extra];bins=np.searchsorted(np.quantile(mean,[.2,.4,.6,.8]),mean)*5+np.searchsorted(np.quantile(sd,[.2,.4,.6,.8]),sd);rng=np.random.default_rng(169952+fold)
        for b in np.unique(bins):ix=extra[bins==b];full_permutation[ix]=rng.permutation(ix)
    rfeat=[];gain=[];loss=[];rmiss=[];cmiss=[];mapping=[]
    for name,genes in groups.items():
        original=np.array([lookup[g] for g in genes if g in lookup],int);ix=permutation[original]
        if len(ix)<10:raise ValueError('insufficient common pathway genes')
        r=rz[:,ix];c=rawC[:,ix];g=np.maximum(c,0);l=np.maximum(-c,0)
        if kind=='RC_GATED':
            gate=expit(r);g=g*gate;l=l*gate
        full_indices=full_permutation[np.array([rlookup[g] for g in genes if g in rlookup and rst['usable'][rlookup[g]]],int)]
        rfeat.append(full_z[:,full_indices].mean(1));gain.append(nanmean(g));loss.append(nanmean(l));rmiss.append(1-np.isfinite(rr[:,full_indices]).mean(1));cmiss.append(1-np.isfinite(c).mean(1));mapping.append({'group':name,'RNA_indices':eligible[ix],'CNA_indices':ci[ix],'RNA_head_indices':full_indices,'original_member_indices':eligible[original],'size':len(ix)})
    Rscore=np.stack(rfeat,1);G=np.stack(gain,1);L=np.stack(loss,1);RM=np.stack(rmiss,1);CM=np.stack(cmiss,1)
    if kind in ['RNA','RNA_ONLY']:x=np.c_[Rscore,RM]
    elif kind=='CNA_ONLY':x=np.c_[G,L,CM]
    else:x=np.c_[Rscore,G,L,RM,CM]
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(x[fit],axis=0)
    median=np.where(np.isfinite(median),median,0);x=np.where(np.isfinite(x),x,median);mu=x[fit].mean(0);sd=x[fit].std(0);sd=np.where(sd>1e-8,sd,1);v=(x-mu)/sd
    return v,{'fit':fit,'kind':kind,'random_mask':random_mask,'mapping':mapping,'eligible_RNA_indices':eligible,'eligible_CNA_indices':ci,'permutation':permutation,'RNA_head_permutation':full_permutation,'median':median,'mean':mu,'sd':sd,'RNA_gene_mean':rst['mean'][eligible],'RNA_gene_sd':rst['sd'][eligible],'RNA_missing_block_handling':'same query RNA scores;training median imputation;explicit missing fractions','CNA_gain_loss_separate':True}

class OmicsModel(base.RoutingModel):
    def __init__(self,w,dimension):super().__init__(w);self.mol=torch.nn.Parameter(torch.zeros(dimension))
    def forward(self,cs,bags,mol):return super().forward(cs,bags)+.25*torch.tanh(mol@self.mol)

def train_model(cs,bags,mol,t,e,fit,test,w,lam,epochs,seed,select):
    from src.phase10_trimodal.math import log_weights,torch_cox
    from src.phase13_rna.math import inner_score
    base.seed_all(seed);model=OmicsModel(w,mol.shape[1]).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.tensor(cs,dtype=torch.float32,device='cuda');mm=torch.tensor(mol,dtype=torch.float32,device='cuda');a=log_weights(t[fit],e[fit],'cuda');ee=torch.tensor(e[fit],device='cuda');v=log_weights(t[test],e[test],'cuda') if select else None;ve=torch.tensor(e[test],device='cuda') if select else None;history=[];best=float('inf');best_epoch=epochs;best_state=None;ema=None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();risk=model(x,bags,mm);data=torch_cox(risk[fit],a,ee);penalty=lam/2*(model.base@model.base)+.001/2*((model.roi@model.roi)+(model.score@model.score))+.1/2*(model.mol@model.mol);loss=data+penalty
        if not torch.isfinite(loss):raise ValueError('G5 nonfinite training')
        loss.backward();grad={k:float(vv.grad.norm()) for k,vv in model.named_parameters()};optimizer.step();row={'epoch':epoch,'train_cox':float(data.detach()),'penalized_loss':float(loss.detach()),'gradient':grad}
        if select:
            with torch.no_grad():r=model(x,bags,mm)[test];vl=float(torch_cox(r,v,ve));ema=vl if ema is None else .3*vl+.7*ema;valc=inner_score(t[test],e[test],r.cpu().numpy())
            row.update(val_cox=vl,val_ema=ema,val_Harrell=valc)
            if ema<best:best=ema;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
    if select:model.load_state_dict(best_state)
    return model,history,best_epoch
