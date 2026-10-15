"""Post hoc INNER-ONLY budget diagnostic; never alters the formal predictions."""
import sys,time,concurrent.futures,subprocess,os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.phase16_routing.core import *
def one(arm,fold):
    seed_all(31);d=parent.load();ev=parent.endpoints(d)['DSS'];a,path=anchor(31,fold,'inner');cs,_=cs_features(d,a['fit']);selector,mm=arm.rsplit('_',1);bags=Bags(selector,int(mm),31);started=time.time();model,history,selected=train(cs,bags,ev['time'],ev['event'],a['fit'],a['test'],a['w'],a['lambda'],320,31+fold*100,True)
    values=[x['val_ema'] for x in history];old=int(np.argmin(values[:80]))+1;row={'arm':arm,'seed':31,'fold':fold,'diagnostic_only':True,'original_budget':80,'diagnostic_budget':320,'inner_selected80':old,'inner_selected320':selected,'inner_min_EMA80':min(values[:80]),'inner_min_EMA320':min(values),'train_loss_epoch80':history[79]['train_cox'],'train_loss_epoch320':history[-1]['train_cox'],'inner_val_C80':history[old-1]['val_Harrell'],'inner_val_C320':history[selected-1]['val_Harrell'],'outer_test_prediction_performed':False,'seconds':time.time()-started,'scope':'post hoc diagnostic after formal evaluation;no model selection or formal prediction changed'}
    write(P/'budget_diagnostic'/f'{arm}_fold{fold}.json',row);print('BUDGET_DIAG',arm,fold,flush=True)
def main():
    tasks=[(f'{s}_{m}',fold) for s in cfg()['active_selectors'] for m in cfg()['budgets'] for fold in [1,2]];env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def call(task):
        arm,fold=task;subprocess.run([sys.executable,__file__,'--arm',arm,'--fold',str(fold)],env=env,check=True,cwd=ROOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(call,tasks))
    rows=[json.loads(p.read_text()) for p in sorted((P/'budget_diagnostic').glob('*.json'))]
    if len(rows)!=16 or any(r['outer_test_prediction_performed'] for r in rows):raise ValueError('budget diagnostic scope')
    pd.DataFrame(rows).to_csv(R/'BUDGET_DIAGNOSTIC.csv',index=False);write(R/'BUDGET_DIAGNOSTIC.json',{'status':'PASS_DIAGNOSTIC_ONLY','inner_fits':16,'outer_fits':0,'prediction_changes':0,'models_or_predictions_retained':False,'original_formal_budget80_unchanged':True,'scope':'post hoc nonconfirmatory;seed31fold1/2;320epochs fixed once;no further budget search'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--fold',type=int);a=p.parse_args();one(a.arm,a.fold) if a.arm else main()
