"""Anonymous arithmetic and predeclared structural gates, not statistical CI."""
import csv,io,math


def aggregate(rows):
    h=math.fsum(r['Harrell_num'] for r in rows)/math.fsum(r['Harrell_den'] for r in rows)
    ok=all(r['Uno_num'] is not None and r['Uno_den'] is not None and r['Uno_den']>0 for r in rows)
    return {'Harrell':h,'Uno60':math.fsum(r['Uno_num'] for r in rows)/math.fsum(r['Uno_den'] for r in rows) if ok else None}


def comparison(a,b):
    bb={r['seed']:r for r in b};one=next(iter(bb.values())) if len(bb)==1 else None
    d=[{'seed':r['seed'],'Harrell':r['Harrell']-(one or bb[r['seed']])['Harrell'],
        'Uno60':r['Uno60']-(one or bb[r['seed']])['Uno60'] if r['Uno60'] is not None and (one or bb[r['seed']])['Uno60'] is not None else None} for r in a]
    h=sum(x['Harrell'] for x in d)/len(d);u=sum(x['Uno60'] for x in d)/len(d) if all(x['Uno60'] is not None for x in d) else None
    both=sum(x['Harrell']>0 and x['Uno60'] is not None and x['Uno60']>0 for x in d)
    return {'Harrell_delta':h,'Uno60_delta':u,'positive_both_seeds':both,'seed_deltas':d,
        'direction_pass':h>0 and u is not None and u>0 and both>=2,'CI_bootstrap_pvalues':False}


def structure_pass(arm,node):
    u=node['actual_update_counts'];g=node['maximum_observed_training_Cox_only_gradients'];d=node['diagnostics']['validation'];tr=node['diagnostics']['train']
    if arm in ('R2','R3'):return u.get('base',0)>=20 and g.get('base',0)>1e-12 and tr['base_drift_ratio']<=.1
    keys=('patch',) if arm=='R4' else ('patch','mil_query','mil_attention')
    if not all(u.get(k,0)>=20 and g.get(k,0)>1e-12 for k in keys):return False
    if arm=='R4':return d['local_risk_sd']/d['anchor_sd']>=.001
    def geometry(prefix):
        names=('slot_effective_rank_mean','centered_slot_energy_fraction_mean','attention_pair_cosine_mean','effective_patch_fraction_mean')
        if any(d.get(prefix+'_'+n) is None for n in names):return False
        return d[prefix+'_slot_effective_rank_mean']>=1.1 and d[prefix+'_centered_slot_energy_fraction_mean']>=.0001 and d[prefix+'_attention_pair_cosine_mean']<=.995 and d[prefix+'_effective_patch_fraction_mean']<=.98
    if not (geometry('MIL') and d['MIL_risk_dependence_ratio']>=.001):return False
    if arm!='R6':return True
    return (all(u.get(k,0)>=20 and g.get(k,0)>1e-12 for k in ('molecular_query','cross_attention'))
        and geometry('CROSS') and d['wrong_pair_eligible']>=max(10,d['CROSS_patients_total']/2)
        and d['wrong_query_risk_dependence_ratio']>=.001 and d['wrong_query_attention_L1_mean']>=.02)


def csv_shards(root,stem,rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    if len(fields)>149:raise ValueError('PUBLIC_COLUMN_LIMIT')
    def line(row=None):
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n')
        if row is None:w.writeheader()
        else:w.writerow(row)
        return s.getvalue()
    header=line();text=header;size=len(text.encode());part=1;n=0
    for r in rows:
        value=line(r);added=len(value.encode())
        if size+added>1_800_000 or n>=18000:
            (root/f'{stem}_{part:02}.csv').write_text(text,encoding='utf-8',newline='');part+=1;text=header;size=len(header.encode());n=0
        text+=value;size+=added;n+=1
    (root/f'{stem}_{part:02}.csv').write_text(text,encoding='utf-8',newline='')
