"""Pre-outcome freeze, synthetic/dataflow gates; no smoke predictions saved."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from run_phase14 import *
from src.phase14_rna.models import ranking_kl,teacher_targets,derange,fit_model


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    configuration=cfg();d=parent.load();ev=parent.endpoints(d);checks={}
    for ep in configuration['endpoints']:
        for fold in configuration['folds']:
            for role in ['inner','outer']:
                f,m,a,fit,test=role_data(ep,'HALL',fold,role)
                if set(fit)&set(test):raise ValueError('partition leakage')
                strata=np.char.add(d['cancer_type'].astype(str),np.where(np.isnan(d['rna_genes']).any(1),'_missing','_complete'))
                for ix in [fit,test]:
                    wrong=derange(ix,strata,31+fold)
                    assert np.array_equal(strata[ix],strata[wrong])
        checks[ep+'_valid']={'n':int(ev[ep]['valid'].sum()),'events':int(ev[ep]['event'][ev[ep]['valid']].sum())}
    f,m,a,fit,test=role_data('DSS','HALL',1,'inner');t,e=ev['DSS']['time'],ev['DSS']['event'];c={**configuration,'epochs':2}
    target,audit=teacher_targets(f['HALL'][fit],t[fit],e[fit],31)
    for row in audit:assert not set(row['train'])&set(row['test'])
    def smoke(tt,ee):
        torch.manual_seed(31);q=RNAOnly(m['clinical_mask'],m['HALL'],31).cuda()
        info=fit_model(q,[f['HALL'][fit]],[f['HALL'][test]],tt,ee,fit,test,c,teacher=target)
        return {k:v.cpu().clone() for k,v in q.state_dict().items()}
    first=smoke(t,e);tt=t.copy();ee=e.copy();outer=parent.splits(d,'DSS',1)['outer_test'];tt[outer]+=123;ee[outer]=1-ee[outer];second=smoke(tt,ee)
    assert all(torch.equal(first[k],second[k]) for k in first)
    checks['outer_test_label_perturbation']='bitwise-identical two-epoch selected state'
    yy=torch.tensor([-.5,.2,.8],requires_grad=True);loss=ranking_kl(yy,yy.detach());assert abs(float(loss))<1e-6;loss.backward();assert yy.grad.abs().max()<1e-6
    checks['ranking_KL_identity']='PASS'
    assert np.array_equal(index_map(np.array([9,2,8]),np.array([8,9,2])),[2,0,1])
    checks['unsorted_inner_index_mapping']='PASS'
    from src.phase14_rna.models import Conditional
    q=Conditional(4,64,np.zeros(68),True,configuration,31)
    cc=torch.randn(20,4)*100;rr=torch.randn(20,64)*100
    assert q.correction(cc,rr).abs().max()<=.25
    assert torch.equal(q.correction(cc,torch.zeros_like(rr)),torch.zeros(20))
    checks['pure_bilinear_and_bound']='PASS'
    out={'status':'PASS','source_hash':source(),'protocol_sha256':sha(CONFIG),'checks':checks,'smoke_predictions_saved':False,'parent_commit':configuration['parent_commit']}
    write(REPORT/'HARD_GATES.json',out);write(PRIVATE/'PROTOCOL_LOCK.json',out);print(json.dumps(out))


if __name__=='__main__':main()
