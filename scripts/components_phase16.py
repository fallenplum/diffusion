"""Frozen-checkpoint prediction accounting; no new fits or configuration choice."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import components,ipcw
from validate_phase16_independent import cs_independent
from src.phase16_routing.core import *
def main():
    d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];rows=[];metrics=[]
    for selector in cfg()['active_selectors']:
        for budget in cfg()['budgets']:
            arm=f'{selector}_{budget}'
            for seed in cfg()['seeds']:
                for fold in cfg()['folds']:
                    c=json.loads((P/'cells'/f'{arm}_seed{seed}_fold{fold}.json').read_text());z=torch.load(c['artifact'],map_location='cpu',weights_only=False);st=z['states']['outer'];tr,te=st['fit'],st['test'];xx=cs_independent(d,tr);w=st['model']['base'].numpy();a,path=anchor(seed,fold,'outer');csbase=z['base_only_risk'];delta=z['risk']-csbase;initial=a['test_risk'];nos=z['risk']-xx[te,-32:]@w[-32:];iw=ipcw(t[tr],e[tr],t[te],e[te],60)
                    rows.append({'arm':arm,'seed':seed,'fold':fold,'ROI_correction_mean':float(delta.mean()),'ROI_correction_sd':float(delta.std()),'ROI_correction_max_abs':float(abs(delta).max()),'CS_base_drift_sd':float((csbase-initial).std()),'CS_coefficient_drift_norm':float(np.linalg.norm(w-a['w'])),'S32_zeroing_prediction_change_sd':float((nos-z['risk']).std()),'scope':'frozen prediction accounting;not retrained causal ablation'})
                    for condition,risk in [('full',z['risk']),('no_ROI_inference',csbase),('zero_S32_inference',nos)]:
                        hn,hd=components(t[te],e[te],risk);un,ud=components(t[te],e[te],risk,iw);metrics.append({'arm':arm,'seed':seed,'fold':fold,'condition':condition,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud})
    pd.DataFrame(rows).to_csv(R/'RISK_COMPONENTS.csv',index=False);df=pd.DataFrame(metrics);seedrows=[]
    for (arm,seed,condition),g in df.groupby(['arm','seed','condition']):seedrows.append({'arm':arm,'seed':seed,'condition':condition,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum(),'scope':'DSS frozen inference ablation;not trained controls'})
    pd.DataFrame(seedrows).to_csv(R/'INFERENCE_ABLATION_SEED_METRICS.csv',index=False);print('COMPONENTS_PASS',len(rows),flush=True)
if __name__=='__main__':main()
