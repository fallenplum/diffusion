"""Document inner-only decisions and freeze before any outer predictions."""
import sys,json,copy,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
from src.phase17_ssd.data import P,R,CONFIG,cfg,source,write
from src.phase10_trimodal.data import digest

def prepare():
    assert not list((P/'outer').glob('*.pt'))
    rows=[]
    for arm in ['A3','A7','A8']:
        for fold in [1,2]:
            for name,path in [('T0',P/'pilot'/f'{arm}_seed31_fold{fold}.pt'),('T1',P/'tuning'/f'T1_{arm}_fold{fold}.pt')]:
                z=torch.load(path,map_location='cpu',weights_only=False);best=min(z['history'],key=lambda r:r['validation_loss_EMA'])
                rows.append({'preset':name,'arm':arm,'fold':fold,'best_validation_EMA':best['validation_loss_EMA'],'epoch':best['epoch'],'Harrell_at_EMA_optimum':best['validation_Harrell']})
    table=pd.DataFrame(rows);table.to_csv(R/'INNER_HYPERPARAMETER_SELECTION.csv',index=False)
    means=table.groupby('preset').best_validation_EMA.mean();chosen=means.idxmin();conf=cfg();conf['training']['early_stopping_patience']=40
    conf['training'].update(dropout=.25 if chosen=='T1' else .1,weight_decay=.01 if chosen=='T1' else .001)
    conf['training']['preset_selection']='common dropout/WD preset;minimum mean EMA validation Cox across A3/A7/A8 x2 inner folds;no outer labels'
    conf['model']['FiLM']='rank8,1664 parameters,matched to delta+B adapters;common module initialization preserved'
    CONFIG.write_text(json.dumps(conf,indent=2)+'\n')
    repairs=[]
    for fold in [1,2]:
        old=torch.load(P/'pilot'/f'A8_seed31_fold{fold}.pt',map_location='cpu',weights_only=False);new=torch.load(P/'repair_inner'/f'A8_seed31_fold{fold}.pt',map_location='cpu',weights_only=False)
        curve_error=max(abs(a['validation_loss_EMA']-b['validation_loss_EMA']) for a,b in zip(old['history'],new['history']))
        repairs.append({'fold':fold,'old_selected_epoch':old['selected_epoch'],'new_selected_epoch':new['selected_epoch'],'old_selected_validation_Harrell':old['history'][old['selected_epoch']-1]['validation_Harrell'],'new_selected_validation_Harrell':new['history'][new['selected_epoch']-1]['validation_Harrell'],'identical_training_curve_max_error':curve_error})
    pd.DataFrame(repairs).to_csv(R/'INNER_SELECTION_REPAIR.csv',index=False)
    with (R/'INNER_DIAGNOSIS.md').open('a',encoding='utf8') as f:
        f.write('\n## 实施后的统一协议修复\n\n解除minimum_epoch16的下限；A8两个内层折旧/新已实际重跑，训练曲线最大差异为'+str(max(v['identical_training_curve_max_error'] for v in repairs))+'。具体选中epoch和Harrell见INNER_SELECTION_REPAIR.csv。\n\n')
        f.write('共同dropout/weight_decay预设只比较T0=(0.1,0.001)、T1=(0.25,0.01)，在A3/A7/A8×两个内层折上按最小验证Cox EMA的均值选择，选中'+chosen+'；两者均值='+str(means.to_dict())+'。不是按外折选超参。\n\n')
        f.write('首轮22个模型均跑满160轮，验证最优在1–6轮，未见后期恢复。因此正式内层保留160上限，采用40轮无EMA改善的早停；最终外层精确重训内层选择的epoch，学习率schedule一致。FiLM参数审计发现8320对1664不匹配，改为rank8的1664参数并补做两个内层检查；其余母干初始化保持不变。该修复在任何外折生成前完成。\n\n没有增加第二项结构性修复；参数/选择协议修复不以外折反馈驱动。\n')
    write(P/'PRELOCK_DECISIONS.json',{'selected_preset':chosen,'preset_means':means.to_dict(),'training_source':source(),'repair_folds':repairs,'formal_outer_predictions_at_decision':0})
    print('PRELOCK_DECISIONS',chosen,means.to_dict(),source(),flush=True)

def lock():
    assert not list((P/'outer').glob('*.pt'));assert json.loads((R/'SSM_OPERATOR_AUDIT.json').read_text())['status']=='PASS';assert json.loads((R/'DATA_LEAKAGE_AUDIT.json').read_text())['status']=='PASS'
    for fold in [1,2]:
        q=torch.load(P/'repair_inner'/f'A4_seed31_fold{fold}.pt',weights_only=False,map_location='cpu');assert q['source']==source()
    arms=cfg()['arms'];rows=[]
    descriptions={'A0':'Clinical+S32 Ridge lambda10','A1':'joint convex Clinical+S32+UNI32+RNA512/CNA512gain/loss','A2':'vanilla WSI S6 + late clinical/own omics','A3':'same cross attention g -> input -> vanilla S6','A4':'rank8 FiLM matched1664 active adapter parameters','A5':'conditional delta only','A6':'conditional B only','A7':'conditional delta+B','A8':'conditional delta+B+write gate','A9':'no individual omics state context;own direct+late retained','A10':'wrong patient state omics;own direct+late retained','A11_NO_S32':'A8 separately refitted without S32 anchor','A11_NO_CLIN':'A8 separately refitted without Clinical anchor/context'}
    for arm in arms:rows.append({'arm':arm,'description':descriptions[arm],'seeds':'31;47;83','outer_folds':'1;2;3;4;5','patch_budget':512,'primary_role':'primary if A8-A3 or A8-A7;otherwise secondary','outer_prediction_before_lock':False})
    pd.DataFrame(rows).to_csv(R/'ABLATION_REGISTRY.csv',index=False)
    paths=[CONFIG]+sorted((ROOT/'src/phase17_ssd').glob('*.py'))+[ROOT/'scripts'/name for name in ['run_phase17.py','run_phase17_paper.py','phase17_paper_models.py','run_phase17_simple.py','validate_phase17_independent.py']]
    write(R/'PROTOCOL_LOCK.json',{'status':'LOCKED_BEFORE_OUTER','training_source':source(),'files':{str(p.relative_to(ROOT)):digest(p) for p in paths},'registry_sha256':digest(R/'ABLATION_REGISTRY.csv'),'primary':cfg()['primary_comparisons'],'outer_predictions':0,'paper_baselines':['MGCT adapted','MambaMIL with late adapter'],'TransSurv':'BLOCKED:no verifiable official implementation/multiscale inputs','time_unix':time.time()})
    print('PROTOCOL_LOCKED',source(),flush=True)

if __name__=='__main__':
    if '--lock' in sys.argv:lock()
    else:prepare()
