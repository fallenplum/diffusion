"""Descriptive post-selection summaries from anonymous reports only."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.reports);inputs={}
    def rows(stem):
        out=[]
        for q in sorted(root.glob(stem+'*.csv')):
            inputs[q.name]=hashlib.sha256(q.read_bytes()).hexdigest()
            with q.open(encoding='utf-8',newline='') as f:out.extend(csv.DictReader(f))
        if not out:raise ValueError('MISSING_AGGREGATE_TABLE:'+stem)
        return out
    def mean(group,key):
        v=[float(r[key]) for r in group if r[key]!='']
        return statistics.mean(v) if v else None
    geometry=rows('FROZEN_ATTENTION_GEOMETRY');gm=[]
    for arm in sorted({r['arm'] for r in geometry}):
        group=[r for r in geometry if r['arm']==arm]
        gm.append({'arm':arm,'fold_seed_cells':len(group),**{k:mean(group,k) for k in
            ('effective_patch_fraction_mean','attention_pair_cosine_mean','slot_effective_rank_mean','centered_slot_energy_fraction_mean')}})
    budget=rows('11_INNER_400_VS200_BUDGET_');bs=[]
    for arm in sorted({r['arm'] for r in budget}):
        group=[r for r in budget if r['arm']==arm]
        item={'arm':arm,'correlated_candidate_cells':len(group)}
        for k in ('Harrell_400_minus200','Uno60_400_minus200'):
            v=[float(r[k]) for r in group if r[k]!=''];item[k+'_mean']=statistics.mean(v);item[k+'_positive_cells']=sum(x>0 for x in v)
        bs.append(item)
    role=rows('FROZEN_CHECKPOINT_TRAINING_ROLES');rs=[]
    for arm in sorted({r['arm'] for r in role}):
        group=[r for r in role if r['arm']==arm];counts=collections.Counter(int(r['selected_epoch']) for r in group)
        rs.append({'arm':arm,'final_fits':len(group),'selected_epochs':[{'epoch':k,'count':v} for k,v in sorted(counts.items())],
            'exact_zero_local_output_matrices':sum(r['local_output_matrix_exactly_zero']=='True' for r in group)})
    query=rows('08_FROZEN_QUERY_INTERVENTIONS_');qs=[]
    for kind in sorted({r['kind'] for r in query}):
        v=[float(r['matched_minus_intervened_risk_sd']) for r in query if r['kind']==kind]
        qs.append({'kind':kind,'fold_seed_cells':len(v),'mean_risk_difference_sd':statistics.mean(v),
            'minimum_risk_difference_sd':min(v),'maximum_risk_difference_sd':max(v)})
    damage=rows('08_PAIR_REPAIR_DAMAGE_');ds=[]
    for arm in ('M1_WARM','M2_EIGHT_MEAN','M3_FULL'):
        for seed in (31,47,83):
            group=[r for r in damage if r['arm']==arm and int(r['seed'])==seed]
            ds.append({'arm':arm,'seed':seed,**{k:sum(float(r[k]) for r in group) for k in
                ('Harrell_repair','Harrell_damage','Harrell_den','Uno_repair','Uno_damage','Uno_den')}})
    out={'schema':'m123_descriptive_mechanism_summary_v1','status':'COMPLETE','geometry':gm,'budget':bs,
        'checkpoint_roles':rs,'frozen_query_sensitivity':qs,'within_fold_repair_damage':ds,
        'aggregation_scope':'equal_fold_seed_means_of_patient_averages_for_geometry; candidate_cell_means_for_budget; pair_sums_for_repairs',
        'post_selection_descriptive_not_new_independent_performance_evidence':True,'new_optimizer_fits':0,
        'CI_bootstrap_pvalues':False,'patient_arrays_or_weights_read':False,
        'input_file_sha256':inputs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'COMPLETE','geometry_arms':len(gm),'budget_arms':len(bs),'new_fits':0}))


if __name__=='__main__':main()
