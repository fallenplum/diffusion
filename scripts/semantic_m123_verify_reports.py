"""Independent aggregate arithmetic audit; no patient data or model selection."""
from pathlib import Path
import argparse,csv,hashlib,json,math


def require(v,message):
    if not v:raise ValueError(message)


def close(a,b,message):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
        and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=2e-12),message)


def grouped(folds):
    groups={};seen=set()
    for row in folds:
        key=(row['arm'],int(row['seed']),int(row['outer']))
        require(key not in seen,'DUPLICATE_FOLD');seen.add(key)
        groups.setdefault(key[:2],[]).append(row)
    out={}
    for key,rows in groups.items():
        require(sorted(int(r['outer']) for r in rows)==[1,2,3,4,5],'INCOMPLETE_FIVE_FOLDS')
        d={}
        for name,numerator,denominator in (('Harrell','Harrell_num','Harrell_den'),('Uno60','Uno_num','Uno_den')):
            nn=math.fsum(float(r[numerator]) for r in rows);dd=math.fsum(float(r[denominator]) for r in rows)
            require(dd>0,'NONPOSITIVE_PAIR_DENOMINATOR')
            for r in rows:close(r[name],float(r[numerator])/float(r[denominator]),'FOLD_COMPONENT_RATIO')
            d[name]=nn/dd
        out[key]=d
    return out


def verify_aggregates(folds,summaries,pairs,repairs):
    byseed=grouped(folds)
    require(len(summaries)==len({r['arm'] for r in summaries}),'DUPLICATE_SUMMARY_ARM')
    means={}
    for row in summaries:
        arm=row['arm'];seeds=sorted(s for a,s in byseed if a==arm)
        require(sorted(int(x['seed']) for x in row['seed_results'])==seeds,'SUMMARY_SEED_COVERAGE')
        for sr in row['seed_results']:
            for metric in ('Harrell','Uno60'):close(sr[metric],byseed[arm,int(sr['seed'])][metric],'SEED_COMPONENT_RATIO')
        d={m:math.fsum(byseed[arm,s][m] for s in seeds)/len(seeds) for m in ('Harrell','Uno60')}
        for m in d:close(row[m],d[m],'EQUAL_SEED_MEAN')
        means[arm]=d
    require(set(means)=={a for a,_ in byseed},'SUMMARY_ARM_COVERAGE')
    for p in pairs:
        a,b=p['comparison'].split(' minus ')
        for m in ('Harrell','Uno60'):close(p[m+'_delta'],means[a][m]-means[b][m],'PAIRED_POINT_ESTIMATE')
        expected={s for aa,s in byseed if aa==a}
        require({int(x['seed']) for x in p['seed_deltas']}==expected,'PAIRED_SEED_COVERAGE')
        reference_seeds={s for aa,s in byseed if aa==b}
        for sr in p['seed_deltas']:
            s=int(sr['seed']);bs=next(iter(reference_seeds)) if len(reference_seeds)==1 else s
            for m in ('Harrell','Uno60'):close(sr[m],byseed[a,s][m]-byseed[b,bs][m],'PAIRED_SEED_DIFFERENCE')
        for name in ('Harrell','Uno60'):
            require(p['positive_'+name+'_seeds']==sum(s[name]>0 for s in p['seed_deltas']),'POSITIVE_SEED_COUNT')
        require(p['positive_both_seeds']==sum(s['Harrell']>0 and s['Uno60']>0 for s in p['seed_deltas']),'POSITIVE_BOTH_COUNT')
    reference={int(r['outer']):r for r in folds if r['arm']=='M0'}
    index={(r['arm'],int(r['seed']),int(r['outer'])):r for r in folds}
    seen=set()
    for r in repairs:
        k=(r['arm'],int(r['seed']),int(r['outer']));require(k not in seen,'DUPLICATE_REPAIR_ROW');seen.add(k)
        target=index[k];base=reference[k[2]]
        for metric,prefix,den in (('Harrell','Harrell','Harrell_den'),('Uno60','Uno','Uno_den')):
            close(r[den],target[den],'REPAIR_DENOMINATOR')
            net=(float(r[prefix+'_repair'])-float(r[prefix+'_damage']))/float(r[den])
            close(net,float(target[metric])-float(base[metric]),'REPAIR_MINUS_DAMAGE_IDENTITY')
    require(seen==set(index),'REPAIR_COVERAGE')
    return {'fold_rows':len(folds),'seed_vectors':len(byseed),'arm_summaries':len(means),'paired_comparisons':len(pairs),
        'repair_damage_rows':len(repairs)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.reports)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(name):return json.loads((root/name).read_bytes())
    index=read('REPORT_INDEX.json')
    for r in index['files']:
        q=root/r['file'];require(q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'REPORT_INDEX_BYTES_CHANGED')
    def tables(stem):
        result=[]
        for q in sorted(root.glob(stem+'_*.csv')):
            with q.open(newline='',encoding='utf-8') as f:result.extend(csv.DictReader(f))
        return result
    result=verify_aggregates(tables('04_OUTER_FOLD_METRICS'),read('05_SEED_METRICS.json'),
        read('06_PRELOCKED_PAIRED_COMPARISONS.json'),tables('08_PAIR_REPAIR_DAMAGE'))
    require(result=={'fold_rows':205,'seed_vectors':41,'arm_summaries':15,'paired_comparisons':21,'repair_damage_rows':205},'REGISTERED_SCOPE_COUNTS')
    audit={'schema':'m123_independent_aggregate_arithmetic_v1','status':'PASS',**result,
        'source_sha256':sha(Path(__file__)),'main_report_index_sha256':sha(root/'REPORT_INDEX.json'),
        'patient_arrays_or_weights_read':False,'new_model_or_parameter_selection':False,'new_optimizer_fits':0,
        'scope':'component_arithmetic_and_repair_damage_only_not_an_additional_raw_data_replay'}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
