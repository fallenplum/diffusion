"""G1 numerical gate uses every locked seed; no selection of favorable runs."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.qpr.protocol import require_stage


def evaluate_gate1(source,output):
    output=Path(output);require_stage('gate1',output.parent)
    cells=[json.loads((Path(source)/f'seed{s}.json').read_text()) for s in range(9101,9111)]
    look=[{r['arm']:r for r in x['results']} for x in cells]
    a2=np.array([x['A2']['real']['RMSE_probe'] for x in look]);a4=np.array([x['A4']['real']['RMSE_probe'] for x in look])
    l2=np.array([x['A2']['real']['logZ_MAE_probe'] for x in look]);l4=np.array([x['A4']['real']['logZ_MAE_probe'] for x in look])
    fraction=float((a2.mean()-a4.mean())/max(a2.mean(),1e-20));positive=int((a4<a2).sum())
    topology=all(t['teacher_gap']>1e-12 and t['student_gap']>1e-12 for x in cells for t in x['topology_effects'])
    structure=all(x['A4']['active_slots_min']>=2 and x['A4']['max_slot_mass']<=.999 for x in look)
    numerical=all(not x['DSS_labels_read'] and max(r['metric_replay_max_abs'] for r in x['checkpoint_replays'])<=1e-12 for x in cells)
    passed=fraction>=.15 and positive>=8 and l4.mean()<=1.05*l2.mean() and topology and structure and numerical
    decision={'fidelity_go':bool(passed),'status':'FIDELITY_GO' if passed else 'FIDELITY_NO_GO','replications':10,
              'main_metric':'real_patient_equal_slide_probe_RMSE','A2_mean_RMSE':float(a2.mean()),'A4_mean_RMSE':float(a4.mean()),
              'fraction_RMSE_reduction':fraction,'positive_seeds':positive,'logZ_A2':float(l2.mean()),'logZ_A4':float(l4.mean()),
              'topology_counterexample_pass':topology,'mass_structure_pass':structure,'numerical_and_no_DSS_pass':numerical,
              'neural_fits':20,'KMeans_fit_calls':10,'gate2_independently_allowed':True,'gate3_allowed':False}
    output.mkdir(parents=True,exist_ok=True);(output/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    rows=[]
    for x in cells:
        for r in x['results']:rows.append({'seed':x['seed'],'arm':r['arm'],**{k:v for k,v in r['real'].items() if not isinstance(v,list)},
                                           'neural_fits':r['neural_fits'],'trained_epochs':r.get('trained_epochs',0),'optimizer_updates':r.get('optimizer_updates',0),'selected_epoch':r.get('selected_epoch',0)})
    with (output/'query_error.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'ALL_SEED_RESULTS.json').write_text(json.dumps(cells,indent=2)+'\n')
    print(json.dumps(decision))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if a.stage!='gate1':raise ValueError('EVALUATOR_NOT_QUALIFIED_FOR_OTHER_STAGE')
    evaluate_gate1(a.source,a.output)
