"""Aggregate synthetic reporting; no patient-level artifacts."""
import json
from pathlib import Path


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def harrell(time,event,risk):
    import numpy as np
    t,e,r=np.asarray(time),np.asarray(event,bool),np.asarray(risk)
    pairs=e[:,None]&(t[:,None]<t[None,:])
    denominator=int(pairs.sum())
    if denominator==0:raise ValueError('NO_COMPARABLE_PAIRS')
    numerator=float(((r[:,None]>r[None,:])*pairs).sum()+.5*((r[:,None]==r[None,:])*pairs).sum())
    return {'Harrell':numerator/denominator,'Harrell_num':numerator,'Harrell_den':denominator}
