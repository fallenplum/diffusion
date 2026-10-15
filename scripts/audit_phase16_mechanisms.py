"""Frozen risk-component accounting and independent low-capacity Cox gradients."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase16_independent import cs_independent
from src.phase16_semantic.omics import *

def literal_gradient(x,t,e,w,lam):
    r=x@w;g=-e.astype(float)
    for u in np.unique(t[e==1]):
        deaths=(t==u)&(e==1);at=t>=u;n=int(deaths.sum());shift=r[at].max();v=np.exp(r-shift)
        for k in range(n):a=at.astype(float)-k/n*deaths;g+=a*v/(a@v)
    return x.T@g/e.sum()+lam*w

def main():
    d=parent.load();ev=parent.endpoints(d)['DSS'];components=[];selection=[];gradients=[];maximum=0
    for arm in arms()+list(armdefs()):
        directory=P if arm in arms() else MP
        for seed in [31,47,83]:
            for fold in [1,2,3,4,5]:
                c=json.loads((directory/'cells'/f'{label(arm,seed,fold)}.json').read_text());z=torch.load(c['artifact'],map_location='cpu',weights_only=False)
                if arm in armdefs() and armdefs()[arm]['selector'] is None:
                    for role,st in z['states'].items():
                        feature=torch.load(st['feature_path'],map_location='cpu',weights_only=False)['x'];fit,test=st['fit'],st['test'];lam=st['lambda_value'];selection.append({'arm':arm,'seed':seed,'fold':fold,'role':role,'lambda':lam})
                        options=z['selection'] if role=='inner' else [{'lambda':lam,'w':st['w']}]
                        for row in options:
                            if row['lambda']=='OFF':continue
                            grad=literal_gradient(feature[fit],ev['time'][fit],ev['event'][fit],row['w'],float(row['lambda']));error=float(abs(grad).max());maximum=max(maximum,error);gradients.append({'arm':arm,'seed':seed,'fold':fold,'role':role,'lambda':row['lambda'],'independent_gradient_max':error})
                            if error>3e-5:raise ValueError('low-capacity Cox not stationary')
                    continue
                st=z['states']['outer'];fit,test=st['fit'],st['test'];cs=cs_independent(d,fit);weights=st['model'];csrisk=(cs.astype(np.float32)@weights['base'].numpy())[test];a,ap=base.anchor(seed,fold,'outer');molecular=np.zeros(len(test))
                if arm.startswith('G5_'):
                    features=torch.load(st['feature_path'],map_location='cpu',weights_only=False)['x'];molecular=.25*np.tanh(features.astype(np.float32)@weights['mol'].numpy())[test]
                roi=z['risk']-csrisk-molecular;components.append({'arm':arm,'seed':seed,'fold':fold,'ROI_risk_sd':float(roi.std()),'ROI_risk_mean':float(roi.mean()),'molecular_risk_sd':float(molecular.std()),'CS_reoptimization_change_sd':float((csrisk-a['test_risk']).std()),'scope':'prediction accounting only;not retrained or causal ablation'})
    pd.DataFrame(components).to_csv(R/'RISK_COMPONENTS.csv',index=False);pd.DataFrame(selection).to_csv(R/'OMICS_ONLY_SELECTION.csv',index=False);pd.DataFrame(gradients).to_csv(R/'OMICS_ONLY_INDEPENDENT_GRADIENTS.csv',index=False);write(R/'MECHANISM_AUDIT.json',{'status':'PASS','risk_component_records':len(components),'low_capacity_independent_gradient_max':maximum,'omics_only_optimization_verified':True,'NN_budget_sufficiency_claimed':False});print('MECHANISM_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
