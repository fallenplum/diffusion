"""Lock source, audit training partitions and prepare private fit-local caches."""
import json,subprocess,platform
import torch,numpy as np
from src.trimodal_interaction.data import Data,P,R,ROOT,cfg,source,write,digest


def main():
    c=cfg();s=source();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    expected={'status':'LOCKED_BEFORE_REAL_MODEL_FITTING','source':s,'config_sha256':digest(ROOT/'configs/trimodal_interaction_v1.json'),'parent_commit':c['parent_commit'],'config':c}
    if (P/'LOCK.json').exists() and json.loads((P/'LOCK.json').read_text())!=expected:raise ValueError('lock changed')
    write(P/'LOCK.json',expected);write(R/'PROTOCOL_LOCK.json',expected)
    results=[]
    for fold in c['folds']:
        data=Data(fold,'inner',31)
        assert not np.intersect1d(data.active,data.part['outer_test']).size
        assert np.isnan(data.ep['time'][data.part['outer_test']]).all()
        assert np.all(data.f['X'][data.part['outer_test']]==0)
        try:data.check_indices(data.part['outer_test'][:1])
        except ValueError:pass
        else:raise AssertionError('held-out prediction allowed')
        assert np.array_equal(np.sort(data.wrong[data.fit]),np.sort(data.fit))
        assert np.array_equal(np.sort(data.wrong[data.valid]),np.sort(data.valid))
        assert np.all(data.wrong[data.active]!=data.active)
        write(R/f'DATA_FOLD{fold}_AUDIT.json',data.audit);results.append({'fold':fold,'fit':len(data.fit),'validation':len(data.valid),'events_fit':int(data.e[data.fit].sum()),'events_validation':int(data.e[data.valid].sum()),'outer_labels_exposed':False})
    write(R/'DATA_BOUNDARY_AUDIT.json',{'status':'PASS','folds':results,'source':s,'scope':'Current wrapper rejects held-out indices; historical cohort was previously explored; not an external validation'})
    write(R/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'GPU_total_MB':torch.cuda.get_device_properties(0).total_memory/2**20,'config_sha256':expected['config_sha256'],'source':s})
    print('READY',s,results,flush=True)

if __name__=='__main__':main()
