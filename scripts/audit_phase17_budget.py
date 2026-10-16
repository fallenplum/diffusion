import sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from src.phase17_ssd.data import R,write
from src.phase17_ssd.model import ConditionalS6
from phase17_paper_models import PaperModel

def main():
    torch.manual_seed(1791);torch.set_num_threads(2);rows=[]
    mask=np.zeros((8,512),bool)
    for j in range(8):mask[j,j*64:(j+1)*64]=True
    for name,batch in [('A8',128),('MGCT',128),('MambaMIL',32)]:
        model=ConditionalS6(41,mask,'A8',.25).cuda() if name=='A8' else PaperModel(name,41,mask).cuda()
        x=torch.randn(batch,512,1536,device='cuda');m=torch.ones(batch,512,device='cuda',dtype=torch.bool);c=torch.randn(batch,41,device='cuda');r=torch.randn(batch,512,device='cuda');g=r.abs()*.1;l=r.abs()*.2;obs=torch.ones_like(r);anchor=torch.zeros(batch,device='cuda')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();out=model(x,m,c,r,g,l,obs,obs,anchor);out.mean().backward();torch.cuda.synchronize()
        rows.append({'method':name,'patients_per_microbatch':batch,'N':512,'forward_backward_seconds':time.monotonic()-start,'peak_allocated_MB':torch.cuda.max_memory_allocated()/2**20,'finite':bool(torch.isfinite(out).all())})
        del model,x,m,c,r,g,l,obs,anchor,out;gc.collect();torch.cuda.empty_cache()
    write(R/'FULL_BUDGET_AUDIT.json',{'status':'PASS','checks':rows,'scope':'synthetic full512 budget;no OOF or smoke checkpoints retained'})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
