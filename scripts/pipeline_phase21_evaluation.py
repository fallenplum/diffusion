"""Wait for the finite training matrix, replay then independently evaluate."""
import json
import subprocess
import sys
import time
from src.phase21_relative.data import P,ROOT,source,write

def main():
    while not (P/'TRAINING_DONE.json').exists():
        if (P/'FAILED.json').exists(): raise RuntimeError('training failed; inspect private attempt ledger')
        time.sleep(10)
    done=json.loads((P/'TRAINING_DONE.json').read_text())
    if done['status']!='PASS' or done['source']!=source(): raise ValueError('training incomplete/stale')
    live=[]
    for shard in range(8):
        log=(P/'logs'/f'evaluation_replay_{shard}.log').open('a')
        child=subprocess.Popen([sys.executable,'scripts/validate_phase21.py','--replay','--shard',str(shard),'--shards','8'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        live.append((child,log,shard))
    while live:
        for row in list(live):
            child,log,shard=row;code=child.poll()
            if code is None: continue
            log.close();live.remove(row)
            if code:
                for other,stream,_ in live: other.terminate();stream.close()
                write(P/'EVALUATION_FAILED.json',{'stage':'replay','shard':shard,'exit':code});raise RuntimeError('replay failed')
            print('REPLAY_SHARD_DONE',shard,flush=True)
        if live: time.sleep(10)
    child=subprocess.run([sys.executable,'scripts/validate_phase21.py'],cwd=ROOT)
    if child.returncode:
        write(P/'EVALUATION_FAILED.json',{'stage':'statistics','exit':child.returncode});raise RuntimeError('statistics failed')
    print('ALL_PHASE21_EVALUATION_DONE',flush=True)

if __name__=='__main__': main()
