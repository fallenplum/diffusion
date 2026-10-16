import sys,json,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np,pandas as pd
from validate_phase17_independent import independent_cs,anchor_gradient
from audit_phase17_preprocessing import scaling
from src.phase17_ssd.data import P,R,parent,write

def gradient(x,t,e,w,penalty):
    r=x@w;v=np.exp(r-r.max());g=-x[e.astype(bool)].sum(0)
    for u in np.unique(t[e==1]):
        deaths=(t==u)&(e==1);at=t>=u;number=deaths.sum()
        for j in range(number):
            a=at.astype(float)-j/number*deaths;p=a*v;g+=(x*p[:,None]).sum(0)/p.sum()
    return g/e.sum()+penalty*w

def main():
    d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];raw=np.load(P/'visual/u.npy',mmap_mode='r');mask=np.load(P/'visual/mask.npy');owner=np.load(P/'visual/owner.npy');nslide=np.bincount(owner);U=np.zeros((512,1536))
    for k,pi in enumerate(owner):U[pi]+=np.asarray(raw[k,mask[k]],np.float32).mean(0)/nslide[pi]
    rows=[];cloc={s:k for k,s in enumerate(d['cna_gene_symbols'].astype(str))}
    for fold in range(1,6):
        saved=torch.load(P/'simple'/f'fold{fold}.pt',map_location='cpu',weights_only=False);part=parent.splits(d,'DSS',fold)
        role_design={}
        for role in ['inner','outer']:
            st=saved['preprocessing'][role];fit=part['inner_train' if role=='inner' else 'outer_train'];meta,_=parent.parent_features(fold,role);ix=np.asarray(meta['phase12']['enriched_indices'],int);r,_=scaling(np.asarray(d['rna_genes'],float),fit);r=r[:,ix].astype(np.float32);c,s=independent_cs(d,fit)
            cna=np.full((512,512),np.nan)
            for k,sy in enumerate(d['rna_gene_symbols'].astype(str)[ix]):
                if sy in cloc:cna[:,k]=d['cna_genes'][:,cloc[sy]]
            gain,_=scaling(np.maximum(cna,0),fit);loss,_=scaling(np.maximum(-cna,0),fit);o=np.c_[r,gain.astype(np.float32),loss.astype(np.float32)]
            mean=U[fit].mean(0);sd=U[fit].std(0);sd[sd<1e-8]=1;assert np.max(abs(mean-st['visual_scale']['mean']))<1e-12 and np.max(abs(sd-st['visual_scale']['sd']))<1e-12
            u=(U-mean)/sd;pc=(u-u[fit].mean(0))@st['visual_components'].T;pm=pc[fit].mean(0);ps=pc[fit].std(0);ps[ps<1e-8]=1;v=(pc-pm)/ps;cs=np.c_[c,s]
            role_design[role]={'CS':cs,'V':v,'O':o}
        train,test=part['outer_train'],part['outer_test'];f=role_design['outer']
        for arm in ['A0','A1']:
            q=saved['models'][arm];x=f['CS'] if arm=='A0' else np.c_[f['CS'],f['V'],f['O']];rr=x[test]@q['w'];error=float(abs(rr-q['risk']).max());assert error<1e-8
            penalty=np.full(x.shape[1],10.) if arm=='A0' else np.r_[np.full(f['CS'].shape[1],10.),np.full(32,q['selected']['visual_lambda']),np.full(1536,q['selected']['omics_lambda'])]
            ge=float(abs(gradient(x[train].astype(float),t[train],e[train],q['w'],penalty)).max());assert ge<2e-5
            rows.append({'arm':arm,'fold':fold,'risk_max_error':error,'independent_objective_gradient_max':ge,'actual_fit':True,'seed_aliases':'31;47;83'})
    pd.DataFrame(rows).to_csv(R/'SIMPLE_CHECKPOINT_REPLAY.csv',index=False);write(R/'SIMPLE_CHECKPOINT_REPLAY.json',{'status':'PASS','actual_outer_fits':10,'max_prediction_error':max(r['risk_max_error'] for r in rows),'max_gradient':max(r['independent_objective_gradient_max'] for r in rows),'features':'independent rawRNA/CNA/Clinical and saved trainfit PCA transform'})
    print('SIMPLE_CHECKPOINT_REPLAY_PASS',rows,flush=True)

if __name__=='__main__':main()
