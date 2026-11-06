"""命令行接口：合成演示→训练→冻结模型→预测CSV→指标JSON。"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from .demo import create
from .io import read_npz
from .model import train_model,predict_model,save_model,load_model
from .reference import evaluate

def _digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run_demo(out,n=144,seed=20261107):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=out/'synthetic_features.npz';synth=create(data,n=n,seed=seed)
    raw=read_npz(data,require_outcome=True)
    rng=np.random.default_rng(seed);order=rng.permutation(n);train=np.sort(order[:int(.75*n)]);test=np.sort(order[int(.75*n):])
    # 整个训练流程只读取train位置的DSS; test结局仅用于最终合成演示评估。
    state=train_model(raw,train,seed=27011)
    model_file=out/'r4_model.json';save_model(state,model_file)
    state2=load_model(model_file)
    pred=predict_model(raw,state2)
    csv_file=out/'synthetic_predictions.csv'
    with csv_file.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['synthetic_patient_id','risk_score','cp_risk','rna_risk','subset'])
        for i in range(n):
            writer.writerow([raw['patient_id'][i],f'{pred["risk_score"][i]:.10f}',f'{pred["cp_risk"][i]:.10f}',f'{pred["rna_risk"][i]:.10f}','train' if i in set(train) else 'test'])
    metrics=evaluate(raw['t'][train],raw['e'][train],raw['t'][test],raw['e'][test],pred['risk_score'][test])
    summary={'software':'多模态肿瘤生存风险预测与分析软件','version':'V1.0',
             'status':'SYNTHETIC_SMOKE_TEST_ONLY','explicit_no_TCGA_result':True,
             'dataset':synth,'fit_patients':len(train),'test_patients':len(test),
             'test_events':int(raw['e'][test].sum()),'beta_rna':state2['beta'],
             'metrics_synthetic_only':metrics,'files':{'dataset':data.name,'model':model_file.name,'predictions':csv_file.name},
             'sha256':{'dataset':_digest(data),'model':_digest(model_file),'predictions':_digest(csv_file)}}
    (out/'demo_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description='多模态肿瘤生存风险预测与分析软件 V1.0')
    sub=p.add_subparsers(dest='cmd',required=True)
    demo=sub.add_parser('demo',help='合成数据端到端演示');demo.add_argument('--out',default='demo_output');demo.add_argument('--patients',type=int,default=144)
    train=sub.add_parser('train',help='从授权特征训练模型');train.add_argument('--input',required=True);train.add_argument('--output',required=True);train.add_argument('--seed',type=int,default=27011)
    infer=sub.add_parser('predict',help='从冻结模型对预编码特征预测风险');infer.add_argument('--input',required=True);infer.add_argument('--model',required=True);infer.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.cmd=='demo':
        print(json.dumps(run_demo(a.out,a.patients),ensure_ascii=False,indent=2));return
    if a.cmd=='train':
        raw=read_npz(a.input,require_outcome=True);state=train_model(raw,np.arange(len(raw['t'])),seed=a.seed)
        save_model(state,a.output);print(json.dumps({'model':a.output,'train_patients':len(raw['t']),'beta_rna':state['beta']},ensure_ascii=False));return
    raw=read_npz(a.input,require_outcome=False);state=load_model(a.model);result=predict_model(raw,state)
    with open(a.output,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','risk_score','cp_risk','rna_risk'])
        for i,pid in enumerate(raw['patient_id']):w.writerow([pid]+[f'{result[k][i]:.10f}' for k in ('risk_score','cp_risk','rna_risk')])
    print(json.dumps({'output':a.output,'patients':len(raw['patient_id']),'risk_is_relative_not_diagnostic':True},ensure_ascii=False))

if __name__=='__main__':main()
