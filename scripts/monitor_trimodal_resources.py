"""Aggregate resource telemetry only; not a completion or performance check."""
import os,sys,time,subprocess
from src.trimodal_interaction.data import P,ROOT,write

if '--worker' not in sys.argv:
    h=(P/'RESOURCE_MONITOR.log').open('a')
    p=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
    print('MONITOR',p.pid)
else:
    rows=[];begin=time.monotonic()
    while time.monotonic()-begin<14400:
        value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        rows.append({'time':time.time(),'GPU_memory_MiB':float(value[0]),'GPU_utilization_percent':float(value[1]),'budget160_started':(P/'budget160/LOCK.json').exists(),'scope':'periodic total-device sample; monitor started during stage40'})
        write(P/'RESOURCE_TELEMETRY.json',rows)
        if (P/'budget160/DONE.json').exists():break
        time.sleep(5)
