"""Same-fold, same-objective RNA probes. Deterministic seeds are references."""
import sys,os,json,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase16_diagnostic.core import *
from src.phase10_trimodal.math import fit_ridge
from src.phase13_rna.math import inner_score

def fold_run(fold):
    target=P/'folds'/f'fold{fold}.pt';cell=target.with_suffix('.json')
    if cell.exists():
        z=json.loads(cell.read_text())
        if z['source']!=source() or sha(target)!=z['sha256']:raise ValueError('stale RNA diagnostic')
        return
    d=parent.load();parts=parent.splits(d,'DSS',fold);ep=parent.endpoints(d)['DSS'];t,e=ep['time'],ep['event'];it,iv,ot,te=[parts[k] for k in ['inner_train','inner_val','outer_train','outer_test']]
    xi,si=features(d,fold,'inner');xo,so=features(d,fold,'outer');models={};trials={}
    for rep in cfg()['representations']:
        choices=[]
        for lam in cfg()['grid']:
            if lam=='OFF':w=np.zeros(xi[rep].shape[1]);diag={'gradient_max':0.,'OFF':True}
            else:w,diag=fit_ridge(xi[rep][it],t[it],e[it],lam)
            choices.append({'lambda':lam,'inner_Harrell':inner_score(t[iv],e[iv],xi[rep][iv]@w),'optimizer':diag})
        best=max(c['inner_Harrell'] for c in choices);tied=[c for c in choices if abs(c['inner_Harrell']-best)<=1e-12]
        chosen='OFF' if any(c['lambda']=='OFF' for c in tied) else max(c['lambda'] for c in tied)
        trials[rep]=choices
        for schedule,lam in [('FIXED1',1.),('SELECTED',chosen)]:
            if lam=='OFF':w=np.zeros(xo[rep].shape[1]);diag={'gradient_max':0.,'OFF':True}
            else:w,diag=fit_ridge(xo[rep][ot],t[ot],e[ot],lam)
            models[rep+'__'+schedule]={'w':w,'lambda':lam,'risk':xo[rep][te]@w,'optimizer':diag,'parameters':len(w)}
    target.parent.mkdir(exist_ok=True);torch.save({'source':source(),'fold':fold,'parts':parts,'states':{'inner':si,'outer':so},'features':{'inner':xi,'outer':xo},'trials':trials,'models':models,'endpoints':{k:v[te] for k,v in ep.items()}},target)
    write(cell,{'status':'COMPLETE','source':source(),'sha256':sha(target),'models':18,'deterministic_seed_references':[31,47,83],'outer_test_selection':False});print('RNA_FOLD_DONE',fold,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(fold_run,cfg()['folds']))
    write(P/'RNA_TRAINING_DONE.json',{'status':'PASS','unique_outer_fits':90,'arms':18,'seed_fold_evaluation_slots':270,'seed_references_not_independent':True,'source':source()})
