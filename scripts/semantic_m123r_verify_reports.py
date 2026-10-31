"""Independent public-table arithmetic; no patient arrays or model weights."""
from pathlib import Path
import argparse,csv,hashlib,json,math


def require(v,msg):
    if not v:raise ValueError(msg)


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def near(a,b,tol=1e-12):return a is None and b is None or a is not None and b is not None and abs(a-b)<=tol


def tables(root,stem):
    out=[]
    for p in sorted(Path(root).glob(stem+'_*.csv')):
        with p.open(encoding='utf-8',newline='') as f:out.extend(csv.DictReader(f))
    require(out,'MISSING_TABLE:'+stem);return out


def number(row,key):return None if row.get(key) in (None,'') else float(row[key])


def ratios(rows):
    require(len(rows)==3 and sorted(int(r['inner']) for r in rows)==[0,1,2],'EXACT_THREE_INNER_ROWS')
    result={}
    for metric,prefix in (('Harrell','Harrell'),('Uno60','Uno')):
        nums=[number(r,prefix+'_num') for r in rows];dens=[number(r,prefix+'_den') for r in rows]
        for r,n,d in zip(rows,nums,dens):require(near(number(r,metric),None if n is None or d is None or d<=0 else n/d),'FOLD_COMPONENT_RATIO')
        result[metric]=math.fsum(nums)/math.fsum(dens) if all(n is not None and d is not None and d>0 for n,d in zip(nums,dens)) else None
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.report)
    index=read(root/'REPORT_INDEX.json')
    for r in index['files']:require(sha(root/r['file'])==r['sha256'] and (root/r['file']).stat().st_size==r['bytes'],'REPORT_INDEX_BYTES_CHANGED')
    full=tables(root,'01_ALL_FIXED_CANDIDATE_METRICS');sel=tables(root,'03_SELECTED_INNER_FOLD_METRICS')
    require(len(full)==3240 and len(sel)==240,'FULL_QUALIFICATION_TABLE_COVERAGE')
    identity=lambda r:(r['arm'],int(r['seed']),int(r['inner']),number(r,'a'),number(r,'b'),number(r,'n'),int(r['epoch']))
    fullmap={identity(r):r for r in full};require(len(fullmap)==3240,'DUPLICATE_CANDIDATE_ROW')
    summary=read(root/'05_SEED_METRICS.json');pairs=read(root/'06_PAIRED_COMPARISONS.json');decision=read(root/'decision.json')
    groups={}
    for r in sel:groups.setdefault((r['track'],r['arm'],int(r['seed'])),[]).append(r)
    require(len(groups)==80,'OOF_SEED_VECTOR_COUNT')
    for (track,arm,seed),rr in groups.items():
        actual=ratios(rr);expected=next(x for x in summary[track][arm] if x['seed']==seed)
        for metric in actual:require(near(actual[metric],expected[metric]),'SEED_COMPONENT_AGGREGATION')
        for r in rr:
            reference=next(x for x in groups[track,'R0',0] if x['inner']==r['inner'])
            d=(number(r,'Harrell_repair')-number(r,'Harrell_damage'))/number(r,'Harrell_den')
            require(near(d,number(r,'Harrell')-number(reference,'Harrell')),'REPAIR_MINUS_DAMAGE_IDENTITY')
        if arm not in ('R0','R1') and '_' not in arm:
            for r in rr:
                q=fullmap[identity(r)]
                for metric in ('Harrell','Uno60'):require(near(number(r,metric),number(q,'validation_'+metric)),'SELECTED_METRIC_NOT_SAME_CELL')
    selection_checks=0
    for arm in ('R2','R3','R4','R5','R6','R7','R8','R9'):
        for seed in (31,47,83):
            candidates={}
            for r in full:
                if r['arm']==arm and int(r['seed'])==seed:candidates.setdefault((number(r,'a'),number(r,'b'),number(r,'n'),int(r['epoch'])),[]).append(r)
            for track in ('EARLY_ANY','ENCODER_TRAINED'):
                options=[]
                for (aa,bb,nn,epoch),rr in candidates.items():
                    require(len(rr)==3,'CANDIDATE_THREE_INNERS')
                    needed=['base'] if arm in ('R2','R3') else ['patch'] if arm=='R4' else ['patch','mil_query','mil_attention']
                    if arm in ('R6','R7','R8','R9'):needed+=['molecular_query','cross_attention']
                    if track=='ENCODER_TRAINED' and (epoch<30 or not all(number(r,'actual_updates_'+k)>=20 for r in rr for k in needed)):continue
                    mm=[]
                    for r in rr:mm.append({'inner':r['inner'],**{k.removeprefix('validation_'):v for k,v in r.items() if k.startswith('validation_')}})
                    metric=ratios(mm);options.append(((metric['Harrell'],metric['Uno60'] if metric['Uno60'] is not None else -math.inf,bb,aa,nn,-epoch),(aa,bb,nn,epoch)))
                chosen=max(options)[1]
                rr=groups[track,arm,seed]
                require(all((number(r,'a'),number(r,'b'),number(r,'n'),int(r['epoch']))==chosen for r in rr),'GLOBAL_THREE_INNER_SELECTION_CHANGED')
                selection_checks+=1
    paired_checks=0
    for track,cc in pairs.items():
        for name,d in cc.items():
            aa,bb=name.split(' minus ');a_rows=summary[track][aa];b_rows=summary[track][bb];ds=[]
            for r in a_rows:
                b=b_rows[0] if len(b_rows)==1 else next(x for x in b_rows if x['seed']==r['seed'])
                ds.append((r['Harrell']-b['Harrell'],r['Uno60']-b['Uno60']))
            h=math.fsum(x[0] for x in ds)/len(ds);u=math.fsum(x[1] for x in ds)/len(ds);n=sum(x>0 and y>0 for x,y in ds)
            require(near(h,d['Harrell_delta']) and near(u,d['Uno60_delta']) and n==d['positive_both_seeds'] and d['direction_pass']==(h>0 and u>0 and n>=2),'PAIRED_COMPARISON_ARITHMETIC')
            paired_checks+=1
    controls={'M1_GROUP':('R2',('R1',)),'M1_FUNCTION':('R3',('R1','R2')),'MIL':('R5',('R3','R4')),
       'MOLECULAR':('R6',('R5','R7','R8','R9','R6_WRONG_QUERY','R6_MEAN_QUERY'))}
    for name,(arm,cc) in controls.items():
        g=decision['gates'][name]
        require(g['direction_pass']==all(pairs['ENCODER_TRAINED'][arm+' minus '+c]['direction_pass'] for c in cc),'GATE_DIRECTION_MISMATCH')
        require(g['passed']==(g['direction_pass'] and g['structure_pass']),'GATE_BOOLEAN_MISMATCH')
    result={'status':'PASS','full_candidate_snapshot_rows':len(full),'selected_metric_rows':len(sel),'seed_vectors':len(groups),
      'global_three_inner_selection_checks':selection_checks,'paired_comparison_checks':paired_checks,
      'patient_arrays_or_weights_read':False,'new_optimizer_fits':0,'source_sha256':sha(__file__),'main_report_index_sha256':sha(root/'REPORT_INDEX.json'),
      'execution_lock_sha256':decision['lock_sha256'],'CI_bootstrap_pvalues':False}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
