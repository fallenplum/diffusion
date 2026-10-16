import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from phase17_paper_models import PaperModel,mamba_inner_no_out
from src.phase17_ssd.data import P,R,write

def main():
    torch.manual_seed(1708);torch.set_num_threads(2);rows=[]
    membership=np.zeros((8,512),bool)
    for k in range(8):membership[k,k*64:(k+1)*64]=True
    for name in ['MGCT','MambaMIL']:
        try:
            m=PaperModel(name,41,membership).cuda();m.eval();x=torch.randn(2,16,1536,device='cuda');mask=torch.ones(2,16,device='cuda',dtype=torch.bool)
            c=torch.randn(2,41,device='cuda');r=torch.randn(2,512,device='cuda');args=(x,mask,c,r,r*.1,r*.2,torch.ones_like(r),torch.ones_like(r),torch.zeros(2,device='cuda'))
            out=m(*args);out.sum().backward();grads=[p.grad for p in m.parameters() if p.grad is not None]
            if not torch.isfinite(out).all() or not all(torch.isfinite(g).all() for g in grads):raise ValueError('nonfinite forward/backward')
            q={'method':name,'status':'PASS','forward_shape':list(out.shape),'parameters':sum(p.numel() for p in m.parameters()),'parameters_with_gradient':sum(p.numel() for p in m.parameters() if p.grad is not None)}
            if name=='MGCT':
                # B=1 port equivalence to the untouched official layer and gradients.
                layer=m.core.stage1_layers[0];wa=torch.randn(16,1,256,device='cuda',requires_grad=True);om=torch.randn(8,1,256,device='cuda',requires_grad=True)
                native=layer(wa,om)[0];port=m.mgct_layer(layer,wa,om).squeeze(0)
                err=float((native-port).abs().max());g1=torch.autograd.grad(native.sum(),[wa,om],retain_graph=True);g2=torch.autograd.grad(port.sum(),[wa,om]);ge=max(float((a-b).abs().max()) for a,b in zip(g1,g2))
                assert err<1e-5 and ge<1e-5;q.update(batched_port_output_error=err,batched_port_gradient_error=ge)
            rows.append(q)
        except Exception as exc:rows.append({'method':name,'status':'BLOCKED','error':repr(exc)})
    write(R/'PAPER_IMPLEMENTATION_AUDIT.json',{'methods':rows,'source':'pinned official repositories;feature/Cox/clinical adapters disclosed'})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
