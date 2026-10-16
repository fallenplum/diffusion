"""Same-protocol convex safety lines; train-only selection, no stacked in-sample risks."""
import sys,json,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from sklearn.decomposition import PCA
from src.phase17_ssd.data import P,R,parent,cfg,source,write,molecular_features
from src.phase10_trimodal.features import scale
from src.phase13_rna.math import block_fit,inner_score
from src.phase10_trimodal.math import fit_ridge
from src.phase10_trimodal.data import digest

def feature(fold,role):
    d=parent.load();part=parent.splits(d,'DSS',fold);fit=part['inner_train' if role=='inner' else 'outer_train'];f,a=molecular_features(d,fold,role)
    raw=np.load(P/'visual/u.npy',mmap_mode='r');mask=np.load(P/'visual/mask.npy');owner=np.load(P/'visual/owner.npy');u=np.zeros((512,1536));count=np.bincount(owner)
    for k,pi in enumerate(owner):u[pi]+=np.asarray(raw[k,mask[k]],np.float32).mean(0)/count[pi]
    u,st=scale(u,fit);pca=PCA(32,svd_solver='full').fit(u[fit]);v,vs=scale(pca.transform(u),fit)
    return {'CS':np.c_[f['C'],f['S']],'V':v,'O':np.c_[f['R'],f['G'],f['L']]}, {'omics':a,'visual_components':pca.components_,'visual_scale':st,'PC_scale':vs}

def main():
    torch.set_num_threads(2)
    lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert lock['training_source']==source()
    d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event']
    for fold in cfg()['folds']:
        target=P/'simple'/f'fold{fold}.pt';target.parent.mkdir(exist_ok=True)
        if target.exists():continue
        part=parent.splits(d,'DSS',fold);it,iv,ot,te=[part[k] for k in ['inner_train','inner_val','outer_train','outer_test']]
        fi,ai=feature(fold,'inner');fo,ao=feature(fold,'outer');models={};trials=[]
        w,dg=fit_ridge(fo['CS'][ot],t[ot],e[ot],10.);models['A0']={'risk':fo['CS'][te]@w,'w':w,'parameters':len(w),'optimizer':dg,'lambda':10.}
        # Clinical/S32 penalty10 common to the neural offsets; other blocks have a small locked grid.
        for lv,lo in itertools.product([1.,10.],[1.,10.]):
            x=np.c_[fi['CS'],fi['V'],fi['O']];penalty=np.r_[np.full(fi['CS'].shape[1],10.),np.full(32,lv),np.full(1536,lo)]
            w,dg=block_fit(x[it],t[it],e[it],penalty);score=inner_score(t[iv],e[iv],x[iv]@w);trials.append({'visual_lambda':lv,'omics_lambda':lo,'inner_Harrell':score,'optimizer':dg})
        selected=max(trials,key=lambda z:(z['inner_Harrell'],z['omics_lambda'],z['visual_lambda']))
        x=np.c_[fo['CS'],fo['V'],fo['O']];penalty=np.r_[np.full(fo['CS'].shape[1],10.),np.full(32,selected['visual_lambda']),np.full(1536,selected['omics_lambda'])]
        w,dg=block_fit(x[ot],t[ot],e[ot],penalty);models['A1']={'risk':x[te]@w,'w':w,'parameters':len(w),'optimizer':dg,'selected':selected}
        torch.save({'models':models,'trials':trials,'source':source(),'parts':part,'preprocessing':{'inner':ai,'outer':ao}},target)
        write(target.with_suffix('.json'),{'status':'COMPLETE','source':source(),'checkpoint_sha256':digest(target),'outer_fits':2,'inner_fits':4,'seed_references':cfg()['seeds'],'duplicated_seed_fits':False})
        print('SIMPLE_DONE',fold,flush=True)

if __name__=='__main__':main()
