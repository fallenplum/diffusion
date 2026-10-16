"""Only inner artifacts; refuses to inspect outer artifacts for repair decisions."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase17_ssd.data import P,R,cfg,write,source,parent
from src.phase13_rna.math import inner_score

def main():
    done=json.loads((P/'pilot_DONE.json').read_text());assert done['status']=='PASS'
    assert not list((P/'outer').glob('*.pt')) if (P/'outer').exists() else True
    d=parent.load();ev=parent.endpoints(d)['DSS'];rows=[]
    for path in sorted((P/'pilot').glob('*.pt')):
        z=torch.load(path,map_location='cpu',weights_only=False);assert z['source']==source()
        h=z['history'];best_raw=min(h,key=lambda v:v['validation_loss']);best_ema=min(h,key=lambda v:v['validation_loss_EMA']);v=z['validation_or_test']
        rows.append({'arm':z['arm'],'fold':z['fold'],'epochs':len(h),'selected_epoch':z['selected_epoch'],'raw_best_epoch':best_raw['epoch'],'EMA_best_unrestricted_epoch':best_ema['epoch'],'selected_Harrell':inner_score(ev['time'][v],ev['event'][v],z['risk']),'first_train_loss':h[0]['train_loss'],'last_train_loss':h[-1]['train_loss'],'first_val_loss':h[0]['validation_loss'],'last_val_loss':h[-1]['validation_loss'],**z['mechanism'],**{k:x for k,x in z['summary'].items() if k!='gradient_norms'}})
    tab=pd.DataFrame(rows);tab.to_csv(R/'INNER_PILOT_DIAGNOSTICS.csv',index=False)
    early=int((tab.EMA_best_unrestricted_epoch<cfg()['training']['minimum_epoch']).sum());total=len(tab)
    display=tab[['arm','fold','selected_epoch','EMA_best_unrestricted_epoch','selected_Harrell','first_train_loss','last_train_loss','first_val_loss','last_val_loss']]
    table='| '+' | '.join(display.columns)+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in display.itertuples(index=False,name=None))
    lines=['# INNER_DIAGNOSIS — 锁定前内层诊断','','本文件仅使用seed31两个内层折、128 patch预算；没有读取或生成外折预测。','',f'实际完成{total}个内层cell，每个训练160 epochs。{early}/{total}个cell的验证损失EMA全程最优点早于原minimum_epoch=16。', '',table,'','## 解释边界','','训练损失下降而验证损失上升支持过拟合/优化目标不一致的诊断，但不证明65个DSS事件是唯一原因。只比较训练期的内层目标，不能用这些结果预判外折性能。','','## 定向修复候选','','若原minimum_epoch=16排除多个内层更优早期epoch，R1只解除该人为下限，保留最大160、EMA验证Cox、架构、正则和输入。先用已保存逐epoch曲线与旧/新同协议内层选择比较；正式每折需完整运行锁定schedule。其他结构修复必须另有真实内层机制证据，不自动添加。']
    (R/'INNER_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(P/'INNER_DIAGNOSIS_READY.json',{'source':source(),'cells':total,'early_epoch_exclusions':early,'outer_predictions':0})
    print(tab[['arm','fold','selected_epoch','EMA_best_unrestricted_epoch','selected_Harrell']].to_string(index=False),flush=True)

if __name__=='__main__':main()
