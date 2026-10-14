"""Complete inner/outer pipelines; shared inputs, private replayable artifacts."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.decomposition import PCA
from src.phase13_rna import data as parent
from src.phase13_rna.math import block_fit,fit_ridge,linear_map
from src.phase13_rna.models import RNAOnly,extract_rna
from src.phase14_rna.models import teacher_targets,fit_model,Conditional,derange

PRIVATE=Path('/root/phase14_rna_private');REPORT=ROOT/'reports/phase14_rna_retention'
CONFIG=ROOT/'configs/phase14_rna_retention.json'
MODELS=['RAW512','RAW64','INIT64','INITPOOL64','TOKEN512','TOKEN64','POOL64','RANK512','RANK64','RANKPOOL64','CS','COXRAW','COX','COXPOOL','WRONG_COX','ADDITIVE','INTERACT','WRONG_ADDITIVE','WRONG_INTERACT']


def cfg():return json.loads(CONFIG.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source():
    paths=[CONFIG,ROOT/'scripts/run_phase14.py',ROOT/'src/phase14_rna/models.py',ROOT/'scripts/gate_phase14.py']
    return hashlib.sha256((parent.source_hash()+''.join(str(p.relative_to(ROOT))+sha(p) for p in paths)).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))
def unit(ep,gene,seed,fold):return f'{ep}_{gene}_seed{seed}_fold{fold}'


def index_map(indices,permutation):
    lookup={int(v):i for i,v in enumerate(indices)}
    if len(lookup)!=len(indices) or set(lookup)!=set(permutation):raise ValueError('index map partition mismatch')
    return np.asarray([lookup[int(v)] for v in permutation],int)


def reduce_rep(train,test,dimension=None):
    mean=train.mean(0);sd=train.std(0);sd=np.where(sd>1e-8,sd,1);a=(train-mean)/sd;b=(test-mean)/sd
    state={'input_mean':mean,'input_sd':sd}
    if dimension:
        p=PCA(dimension,svd_solver='full').fit(a);a=p.transform(a);b=p.transform(b)
        state.update({'pca_mean':p.mean_,'components':p.components_,'explained_ratio':float(p.explained_variance_ratio_.sum())})
    mu=a.mean(0);ss=a.std(0);ss=np.where(ss>1e-8,ss,1)
    state.update({'output_mean':mu,'output_sd':ss,'dimension':a.shape[1]})
    return ((a-mu)/ss).astype(np.float32),((b-mu)/ss).astype(np.float32),state


def apply_rep(x,state):
    x=(x-state['input_mean'])/state['input_sd']
    if 'components' in state:x=(x-state['pca_mean'])@state['components'].T
    return ((x-state['output_mean'])/state['output_sd']).astype(np.float32)


def role_data(ep,gene,fold,role):
    d=parent.load();part=parent.splits(d,ep,fold);f,m,a=parent.cached_features(d,ep,fold,role)
    fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test']
    return f,m,a,fit,test


def run(ep,gene,seed,fold):
    configuration=cfg();name=unit(ep,gene,seed,fold);out=PRIVATE/'cells'/f'{name}.json'
    if out.exists():
        cell=json.loads(out.read_text())
        if cell['source_hash']!=source() or sha(cell['artifact'])!=cell['artifact_sha256']:raise RuntimeError('stale cell')
        return
    gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=source():raise RuntimeError('gate absent/stale')
    started=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    d=parent.load();ev=parent.endpoints(d)[ep];t,e=ev['time'],ev['event'];part=parent.splits(d,ep,fold)
    features={};members={};audits={};fits={};tests={};representations={};states={};producers={};transforms={};probes={};risks={};diagnostics=[]
    for role in ['inner','outer']:
        f,m,a,fit,test=role_data(ep,gene,fold,role);features[role]=f;members[role]=m;audits[role]=a;fits[role]=fit;tests[role]=test
        representations[role]={}
        for model,dimension in [('RAW512',None),('RAW64',64)]:
            aa,bb,ss=reduce_rep(f[gene][fit],f[gene][test],dimension);representations[role][model]=(aa,bb);transforms[role+'_'+model]=ss
        target,audit=teacher_targets(f[gene][fit].astype(float),t[fit],e[fit],seed)
        teacher_w,teacher_info=fit_ridge(f[gene][fit],t[fit],e[fit],1.)
        states[role+'_teacher']={'w':teacher_w,'targets':target,'audit':audit,'fit_indices':fit,'optimizer':teacher_info}
        for producer,constrained in [('TOKEN',False),('RANK',True)]:
            torch.manual_seed(seed);np.random.seed(seed);model=RNAOnly(m['clinical_mask'],m[gene],seed).cuda()
            if producer=='TOKEN':
                _,itaps=extract_rna(model,f[gene],fit);_,jtaps=extract_rna(model,f[gene],test)
                for kind,field,dim in [('INIT64','processed_full',64),('INITPOOL64','processed_mean',None)]:
                    aa,bb,ss=reduce_rep(itaps[field],jtaps[field],dim);representations[role][kind]=(aa,bb);transforms[role+'_'+kind]=ss
            info=fit_model(model,[f[gene][fit]],[f[gene][test]] if role=='inner' else [],t,e,fit,test if role=='inner' else np.array([],int),configuration,
                           epoch=None if role=='inner' else producers['inner_'+producer]['selected_epoch'],teacher=target if constrained else None)
            producers[role+'_'+producer]=info;states[role+'_'+producer]={k:v.cpu() for k,v in model.state_dict().items()}
            _,a=extract_rna(model,f[gene],fit);_,b=extract_rna(model,f[gene],test)
            for kind,field,dim in [(producer+'512','processed_full',None),(producer+'64','processed_full',64),('POOL64' if producer=='TOKEN' else 'RANKPOOL64','processed_mean',None)]:
                aa,bb,ss=reduce_rep(a[field],b[field],dim);representations[role][kind]=(aa,bb);transforms[role+'_'+kind]=ss
            del model;torch.cuda.empty_cache()
    tr,te=fits['outer'],tests['outer'];it,iv=fits['inner'],tests['inner']
    for model in MODELS[:10]:
        a,b=representations['outer'][model];w,diag=fit_ridge(a,t[tr],e[tr],1.);risks[model]=b@w;probes[model]={'w':w,'optimizer':diag}
        # A scalar teacher decoding probe is diagnostic, not an outcome oracle.
        teacher=states['outer_teacher']['w'];train_target=features['outer'][gene][tr]@teacher;test_target=features['outer'][gene][te]@teacher
        tw=linear_map(a,train_target-train_target.mean(),1.);decoded=b@tw+train_target.mean()
        delta=decoded[:,None]-decoded[None,:];reference=test_target[:,None]-test_target[None,:];upper=np.triu(np.ones(delta.shape,bool),1)
        nonzero=upper&(abs(reference)>1e-8);credit=np.where(abs(delta[nonzero])<=1e-8,.5,np.sign(delta[nonzero])==np.sign(reference[nonzero]))
        r2=1-np.sum((decoded-test_target)**2)/max(np.sum((test_target-test_target.mean())**2),1e-20)
        diagnostics.append({'model':model,'teacher_R2':float(r2),'teacher_pair_agreement':float(credit.mean()),'dimension':a.shape[1]})
        probes[model].update({'teacher_decoder':tw,'teacher_train_mean':float(train_target.mean())})
    cs={role:np.c_[features[role]['C'],features[role]['S']] for role in ['inner','outer']}
    base={role:json.loads((parent.PRIVATE/'formal/cells'/f'{ep}_CS_seed31_fold{fold}.json').read_text())['lambda'] for role in ['inner','outer']}
    cw,cdiag=fit_ridge(cs['outer'][tr],t[tr],e[tr],base['outer']);risks['CS']=cs['outer'][te]@cw;probes['CS']={'w':cw,'optimizer':cdiag}
    anchor=np.load(parent.PRIVATE/'formal/oof'/f'{ep}_CS_seed31_fold{fold}.npz')['risk']
    if abs(anchor-risks['CS']).max()>2e-6:raise RuntimeError('CS anchor identity failure')
    strata=np.char.add(d['cancer_type'].astype(str),np.where(np.isnan(d['rna_genes']).any(1),'_missing','_complete'))
    permutations={}
    for role in ['inner','outer']:
        for side,ix in [('fit',fits[role]),('test',tests[role])]:permutations[role+'_'+side]=derange(ix,strata,seed+fold*100+(0 if role=='inner' else 500)+(0 if side=='fit' else 50))
    # Reindex only within the same producer partition, preserving its latent coordinates.
    for model,representation,wrong in [('COXRAW','RAW64',False),('COX','RANK64',False),('COXPOOL','RANKPOOL64',False),('WRONG_COX','RANK64',True)]:
        for role in ['inner','outer']:
            fit,test=fits[role],tests[role];a,b=representations[role][representation]
            if wrong:a=a[index_map(fit,permutations[role+'_fit'])];b=b[index_map(test,permutations[role+'_test'])]
            w,diag=block_fit(np.c_[cs[role][fit],a],t[fit],e[fit],np.r_[np.full(cs[role].shape[1],base[role]),np.ones(a.shape[1])])
            probes[role+'_'+model]={'w':w,'optimizer':diag}
            if role=='outer':risks[model]=np.c_[cs[role][test],b]@w
    downstream={}
    for arm,interaction,wrong in [('ADDITIVE',False,False),('INTERACT',True,False),('WRONG_ADDITIVE',False,True),('WRONG_INTERACT',True,True)]:
        for role in ['inner','outer']:
            fit,test=fits[role],tests[role];a,b=representations[role]['RANK64']
            if wrong:a=a[index_map(fit,permutations[role+'_fit'])];b=b[index_map(test,permutations[role+'_test'])]
            w=probes[role+('_WRONG_COX' if wrong else '_COX')]['w'];model=Conditional(cs[role].shape[1],64,w,interaction,configuration,seed).cuda()
            info=fit_model(model,[cs[role][fit],a],[cs[role][test],b] if role=='inner' else [],t,e,fit,test if role=='inner' else np.array([],int),configuration,
                           epoch=None if role=='inner' else downstream['inner_'+arm]['selected_epoch'],penalty=lambda q:q.regularization(base[role],configuration))
            downstream[role+'_'+arm]=info;states[role+'_'+arm]={k:v.cpu() for k,v in model.state_dict().items()}
            if role=='outer':
                with torch.no_grad():
                    cc=torch.tensor(cs[role][test],device='cuda');rr=torch.tensor(b,device='cuda');risk=model(cc,rr);correction=model.correction(cc,rr)
                risks[arm]=risk.cpu().numpy();info['test_correction_max']=float(correction.abs().max());info['test_correction_RMS']=float(correction.square().mean().sqrt())
                if info['test_correction_max']>configuration['interaction_bound']+1e-6:raise RuntimeError('risk bound violated')
            del model;torch.cuda.empty_cache()
    artifact=PRIVATE/'artifacts'/f'{name}.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'states':states,'transforms':transforms,'probes':probes,'risks':risks,'partitions':part,'permutations':permutations,'test_time':t[te],'test_event':e[te],'producers':producers,'downstream':downstream,'source_hash':source()},artifact)
    write(out,{'status':'COMPLETE','endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'source_hash':source(),'protocol_sha256':sha(CONFIG),'artifact':str(artifact),'artifact_sha256':sha(artifact),
               'models':MODELS,'partitions':{k:v.tolist() for k,v in part.items()},'preprocessing':audits,'outer_test_selection':False,'producers':producers,'downstream':downstream,'decodability':diagnostics,
               'CS_anchor_max_error':float(abs(anchor-risks['CS']).max()),'seconds':time.monotonic()-started,'artifact_bytes':artifact.stat().st_size})
    print('COMPLETE',name,artifact.stat().st_size,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True);ap.add_argument('--gene',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();run(a.endpoint,a.gene,a.seed,a.fold)
