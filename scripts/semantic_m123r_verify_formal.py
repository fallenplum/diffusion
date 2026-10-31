"""Independent formal arithmetic and fold directions from public tables only."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123_verify_reports import grouped,close,require
from src.semantic_m123r.reporting import csv_shards


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output-newdir',required=True)
    a=p.parse_args();root=Path(a.report);read=lambda n:json.loads((root/n).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    index=read('REPORT_INDEX.json')
    for r in index['files']:require(sha(root/r['file'])==r['sha256'] and (root/r['file']).stat().st_size==r['bytes'],'FORMAL_REPORT_INDEX_BYTES')
    def table(stem):
        rows=[]
        for q in sorted(root.glob(stem+'_*.csv')):
            with q.open(encoding='utf-8',newline='') as f:rows.extend(csv.DictReader(f))
        require(rows,'MISSING_FORMAL_TABLE');return rows
    folds=table('01_FORMAL_FOLD_METRICS');repairs=table('03_PAIR_REPAIR_DAMAGE')
    summary=read('04_FORMAL_SEED_METRICS.json');paired=read('05_FORMAL_COMPARISONS.json');d=read('decision.json')
    require(d['status']=='COMPLETE_GATED_FORMAL_VALIDATION','FORMAL_INCOMPLETE')
    directions=[];totalseeds=paircount=0
    for track in ('EARLY_ANY','ENCODER_TRAINED'):
        rr=[r for r in folds if r['track']==track];seeds=grouped(rr);totalseeds+=len(seeds)
        for arm,ss in summary[track].items():
            require({x['seed'] for x in ss}=={s for ar,s in seeds if ar==arm},'FORMAL_SEED_COVERAGE')
            for x in ss:
                for metric in ('Harrell','Uno60'):close(x[metric],seeds[arm,x['seed']][metric],'FORMAL_SEED_ARITHMETIC')
        lookup={(r['arm'],int(r['seed']),int(r['outer'])):r for r in rr}
        for name,pp in paired[track].items():
            aa,bb=name.split(' minus ');s1=summary[track][aa];s2=summary[track][bb];ds=[]
            for x in s1:
                y=s2[0] if len(s2)==1 else next(v for v in s2 if v['seed']==x['seed'])
                delta={'seed':x['seed'],**{m:x[m]-y[m] for m in ('Harrell','Uno60')}};ds.append(delta)
                exact=next(v for v in pp['seed_deltas'] if v['seed']==x['seed'])
                for metric in ('Harrell','Uno60'):close(delta[metric],exact[metric],'FORMAL_PAIRED_SEED_DELTA')
                for outer in range(1,6):
                    u=lookup[aa,x['seed'],outer];v=lookup[bb,y['seed'],outer]
                    dd={m:float(u[m])-float(v[m]) for m in ('Harrell','Uno60')}
                    directions.append({'track':track,'comparison':name,'seed':x['seed'],'outer':outer,
                        'Harrell_delta':dd['Harrell'],'Uno60_delta':dd['Uno60'],'positive_both':dd['Harrell']>0 and dd['Uno60']>0})
            for metric in ('Harrell','Uno60'):close(math.fsum(x[metric] for x in ds)/len(ds),pp[metric+'_delta'],'FORMAL_PAIRED_MEAN_DELTA')
            n=sum(x['Harrell']>0 and x['Uno60']>0 for x in ds)
            require(n==pp['positive_both_seeds'],'FORMAL_POSITIVE_BOTH_SEEDS')
            require(pp['direction_pass']==(pp['Harrell_delta']>0 and pp['Uno60_delta']>0 and n>=2),'FORMAL_DIRECTION_BOOLEAN')
            paircount+=1
        seen=set()
        for x in repairs:
            if x['track']!=track:continue
            key=(x['arm'],int(x['seed']),int(x['outer']),x['metric']);require(key not in seen,'DUPLICATE_FORMAL_REPAIR');seen.add(key)
            target=lookup[key[:3]];reference=lookup['R0',0,key[2]];den='Harrell_den' if key[3]=='Harrell' else 'Uno_den'
            close(x['den'],target[den],'FORMAL_REPAIR_DENOMINATOR')
            net=(float(x['repair_num'])-float(x['damage_num']))/float(x['den'])
            close(net,float(target[key[3]])-float(reference[key[3]]),'FORMAL_REPAIR_DAMAGE_IDENTITY')
        require(len(seen)==2*len(rr),'FORMAL_REPAIR_COVERAGE')
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    csv_shards(out,'01_PAIRED_FOLD_DIRECTIONS',directions)
    audit={'status':'PASS_INDEPENDENT_FORMAL_ARITHMETIC','fold_metric_rows':len(folds),'five_fold_seed_vectors':totalseeds,
        'paired_comparisons_checked':paircount,'paired_fold_directions':len(directions),'repair_damage_rows_checked':len(repairs),
        'patient_arrays_or_weights_read':False,'new_optimizer_fits':0,'new_hyperparameter_or_scale_selection':False,
        'source_sha256':sha(__file__),'main_formal_report_index_sha256':sha(root/'REPORT_INDEX.json'),'CI_bootstrap_pvalues':False}
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    files=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    (out/'REPORT_INDEX.json').write_text(json.dumps({'status':'COMPLETE','schema':'m123r_formal_arithmetic_report_v1','files':files},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()
