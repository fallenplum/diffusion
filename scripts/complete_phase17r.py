"""Wait for staged training, then replay and independently evaluate; no auto-push."""
import sys,os,time,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17r_ssd.data import P,R,source,write

def main():
    while not (P/'TRAINING_DONE.json').exists():
        if (P/'PROGRESS.json').exists():
            q=json.loads((P/'PROGRESS.json').read_text())
            if q['failures']:raise RuntimeError(q['failures'])
        time.sleep(10)
    done=json.loads((P/'TRAINING_DONE.json').read_text());assert done['status']=='PASS' and done['source']==source()
    children=[]
    for shard in range(4):
        h=(P/f'REPLAY_{shard}.log').open('w');p=subprocess.Popen([sys.executable,'scripts/replay_phase17r.py','--shard',str(shard),'--shards','4'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT);children.append((p,h))
    subprocess.run([sys.executable,'scripts/validate_phase17r.py'],cwd=ROOT,check=True)
    for p,h in children:
        code=p.wait();h.close();assert code==0,'checkpoint replay failed'
    import pandas as pd
    rows=pd.concat([pd.read_csv(R/f'CHECKPOINT_REPLAY_{i}.csv') for i in range(4)],ignore_index=True);assert len(rows)==done['outer_fits'];rows.to_csv(R/'CHECKPOINT_REPLAY.csv',index=False)
    for i in range(4):(R/f'CHECKPOINT_REPLAY_{i}.csv').unlink()
    write(R/'CHECKPOINT_REPLAY.json',{'status':'PASS','source':source(),'checkpoints':len(rows),'maximum_error':float(rows.max_risk_replay_error.max()),'conditions':['matched','wrong-context','SSM-off','gamma-one'],'all_from_real_saved_checkpoints':True})
    subprocess.run([sys.executable,'scripts/audit_phase17r_leakage.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/audit_phase17r_budget.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/report_phase17r.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/plot_phase17r.py'],cwd=ROOT,check=True)
    write(P/'AUDITS_DONE.json',{'status':'PASS','source':source(),'figures':'actual visual inspection and privacy packaging still required before push'})
    print('AUDITS_DONE_REQUIRES_VISUAL_QA',flush=True)

if __name__=='__main__':main()
