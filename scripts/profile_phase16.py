"""Measured forward latency/operator FLOPs; not a claim of full FLOP accounting."""
import sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase16_routing.core import *
def main():
    seed_all(31);d=parent.load();rows=[]
    for selector in cfg()['active_selectors']:
        for budget in cfg()['budgets']:
            arm=f'{selector}_{budget}';c=json.loads((P/'cells'/f'{arm}_seed31_fold1.json').read_text());z=torch.load(c['artifact'],map_location='cpu',weights_only=False);st=z['states']['outer'];bags=Bags(selector,budget,31);cs,_=cs_features(d,st['fit']);xx=torch.tensor(cs,dtype=torch.float32,device='cuda');model=RoutingModel(st['model']['base'].numpy()).cuda();model.load_state_dict(st['model']);torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(2):model(xx,bags)
                torch.cuda.synchronize();started=time.perf_counter()
                for _ in range(10):model(xx,bags)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-started)/10
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as p:model(xx,bags)
                torch.cuda.synchronize()
            score_flops=2*sum(x.numel() for x,*_ in bags.buckets);pool_flops=2*sum(x.shape[0]*(k if bags.dynamic else x.shape[1])*768 for x,mask,quota,ids,order,k,qn in bags.buckets);merge_flops=2*512*len(bags.owner)*768;head_flops=2*512*(768+len(st['model']['base']))
            rows.append({'arm':arm,'patients_per_forward':512,'latency_ms':elapsed*1000,'profiled_operator_FLOPs':sum(x.flops for x in p.key_averages()),'analytic_linear_attention_FLOPs':score_flops+pool_flops+merge_flops+head_flops,'analytic_full_candidate_score_FLOPs':score_flops,'GPU_allocated_peak_MiB':torch.cuda.max_memory_allocated()/1024**2,'includes_topk_search':selector=='MIL','FLOPs_scope':'analytic padded linear MACs x2;profiler undercounts MV;topk/softmax/mask/I/O excluded;forward only'})
            del bags,model,xx;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(R/'COMPUTE_PROFILE.csv',index=False);print('COMPUTE_PROFILE_PASS',flush=True)
if __name__=='__main__':main()
