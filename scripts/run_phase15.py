"""Fit-local anchor, pair-honest residual RNA, no outer-error supervision."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase15_complement.core import *
from src.phase14_rna.models import derange

KINDS=['JOINT','ALL','SOFT','HARD','RANDOM','WRONG','NO_OFFSET','SCALAR']


def inputs(d,ep,gene,fold,role):
    meta,z=parent.parent_features(fold,role);key={'TOP':'top','HALL':'enriched','MATCH':'matched'}[gene]
    indices=np.asarray(meta['top_indices'] if gene=='TOP' else meta['phase12'][key+'_indices'],int)
    return np.asarray(d['rna_genes'][:,indices],float),indices


def prepare(ep,seed,fold):
    d=parent.load()
    for role in ['inner','outer']:anchor(d,ep,fold,seed,role)
    print('ANCHORS_COMPLETE',ep,seed,fold,flush=True)


def run(ep,gene,seed,fold):
    target=PRIVATE/'cells'/f'{name(ep,gene,seed,fold)}.json'
    if target.exists():
        c=json.loads(target.read_text())
        if c['source_hash']!=source() or sha(c['artifact'])!=c['artifact_sha256']:raise RuntimeError('stale cell')
        return
    gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=source():raise RuntimeError('hard gate absent')
    started=time.monotonic();torch.set_num_threads(2);d=parent.load();configuration=cfg();ev=parent.endpoints(d)[ep];t,e=ev['time'],ev['event'];part=parent.splits(d,ep,fold)
    data={};audits={};RNA={};transforms={};anchors={};pairs={};permutations={};coefficients={};selection={};optimizer={};risks={};indices={}
    strata=np.char.add(d['cancer_type'].astype(str),np.where(np.isnan(d['raw_rna']).any(1),'_missing','_complete'))
    for role in ['inner','outer']:
        a=anchor(d,ep,fold,seed,role);anchors[role]=a;fit,test=a['fit'],a['test'];raw,idx=inputs(d,ep,gene,fold,role);indices[role]=idx;v,st=rna_scale(raw,fit);transforms[role]=st;data[role]=v
        cs,ca=cs_features(d,fit);audits[role]={'CS':ca,'RNA_fit_hash':ahash(fit.astype(np.int64)),'RNA_gene_indices_hash':ahash(idx),'RNA_selected_missing_train':int(np.isnan(raw[fit]).sum()),'RNA_gene_universe':'inherited role-training unsupervised selection;no outcome selection'}
        rr=rna_teacher(raw,a,t,e);RNA[role]=rr;ii,jj,margin=honest_pairs(a,t,e);pairs[role]={'i':ii,'j':jj,'margin':margin,'event_owners':len(np.unique(ii))}
        for label,ix in [('fit',fit),('test',test)]:permutations[role+'_'+label]=derange(ix,strata,seed+fold*100+(0 if role=='inner' else 500)+(0 if label=='fit' else 50))
        # Each method sees the sameRNA and same honest pair set. Selection only inner-val.
        for kind in KINDS:
            values=configuration['rna_grid'] if role=='inner' else sorted({1.,selection[kind]},key=str)
            candidates=[]
            for lam in values:
                if lam=='OFF':w=None;risk=a['test_risk'];diagnostic={'off':True}
                elif kind=='JOINT':
                    w,diagnostic=block_fit(np.c_[cs[fit],v[fit]],t[fit],e[fit],np.r_[np.full(cs.shape[1],a['lambda']),np.full(512,lam)])
                    risk=np.c_[cs[test],v[test]]@w
                else:
                    ww=weights(kind,margin,ii,seed+fold*100+901);offset=np.zeros_like(margin) if kind=='NO_OFFSET' else margin
                    design=v
                    if kind=='WRONG':
                        design=v.copy();design[fit]=v[permutations[role+'_fit']];design[test]=v[permutations[role+'_test']]
                    if kind=='SCALAR':
                        design=np.zeros((len(v),1));design[fit,0]=rr['oof'][fit];design[test,0]=rr['test']
                    w,diagnostic=pair_fit(design[fit],ii,jj,offset,ww,float(lam));risk=a['test_risk']+design[test]@w
                coefficients[role+'_'+kind+'_'+str(lam)]=w;optimizer[role+'_'+kind+'_'+str(lam)]=diagnostic
                if role=='inner':candidates.append({'lambda':lam,'inner_Harrell':inner_score(t[test],e[test],risk),'parameters':0 if w is None else len(w)})
                else:
                    if lam==selection[kind]:risks[kind]=risk
                    if lam==1. and kind in ['JOINT','ALL','SOFT','SCALAR','RANDOM','WRONG']:risks[kind+'_FIXED1']=risk
            if role=='inner':
                best=max(row['inner_Harrell'] for row in candidates);tied=[r for r in candidates if abs(r['inner_Harrell']-best)<=1e-12]
                winner=next((r for r in tied if r['lambda']=='OFF'),tied[0]);selection[kind]=winner['lambda'];selection[kind+'_rows']=candidates
    te=anchors['outer']['test'];risks['CS']=anchors['outer']['test_risk'];risks['RNA_ONLY']=RNA['outer']['test']
    legacy=parent.PRIVATE/'formal/oof'/f'{ep}_CS_seed31_fold{fold}.npz';old=np.load(legacy)
    if not np.array_equal(old['test_indices'],te):raise RuntimeError('legacy patient order')
    risks['CS_LEGACY']=old['risk']
    if set(risks)!=set(MODELS):raise RuntimeError('model inventory missing '+str(set(MODELS)-set(risks)))
    artifact=PRIVATE/'artifacts'/f'{name(ep,gene,seed,fold)}.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'source_hash':source(),'partitions':part,'indices':indices,'RNA':RNA,'transforms':transforms,'pairs':pairs,'permutations':permutations,'coefficients':coefficients,'selection':selection,'optimizer':optimizer,'risks':risks,'test_time':t[te],'test_event':e[te]},artifact)
    write(target,{'status':'COMPLETE','endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'source_hash':source(),'protocol_sha256':sha(CONFIG),'artifact':str(artifact),'artifact_sha256':sha(artifact),'outer_test_selection':False,'models':MODELS,'selection':selection,'optimizer':optimizer,'preprocessing':audits,'teacher_anchor_paths':{role:str(PRIVATE/'anchors'/f'{ep}_seed{seed}_fold{fold}_{role}.pt') for role in ['inner','outer']},'seconds':time.monotonic()-started,'artifact_bytes':artifact.stat().st_size})
    print('COMPLETE',name(ep,gene,seed,fold),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True);ap.add_argument('--gene',default='HALL');ap.add_argument('--seed',required=True,type=int);ap.add_argument('--fold',required=True,type=int);ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    prepare(a.endpoint,a.seed,a.fold) if a.prepare else run(a.endpoint,a.gene,a.seed,a.fold)
