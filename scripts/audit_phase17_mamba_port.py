import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,torch.nn.functional as F
from phase17_paper_models import mambamil,mamba_inner_no_out
from src.phase17_ssd.data import R,write

def literal(xz,cw,cb,xw,dw,A,D,db):
    x,z=xz.chunk(2,1);x=F.silu(F.conv1d(x,cw,cb,padding=cw.shape[-1]-1,groups=x.shape[1])[...,:x.shape[-1]])
    values=F.linear(x.transpose(1,2),xw);rank=dw.shape[1];s=A.shape[1];dt=F.softplus(F.linear(values[:,:,:rank],dw,db));B=values[:,:,rank:rank+s];C=values[:,:,rank+s:]
    h=torch.zeros(len(x),x.shape[1],s,device=x.device);out=[]
    for t in range(x.shape[-1]):
        h=torch.exp(dt[:,t,:,None]*A)*h+dt[:,t,:,None]*B[:,t,None,:]*x[:,:,t,None]
        y=(h*C[:,t,None,:]).sum(-1)+x[:,:,t]*D;out.append(y*F.silu(z[:,:,t]))
    return torch.stack(out,-1)

def main():
    mambamil();torch.manual_seed(1712);rows=[]
    for n in [1,16,128,512]:
        args=[torch.randn(1,8,n,device='cuda',requires_grad=True),torch.randn(4,1,4,device='cuda',requires_grad=True)*.1,torch.randn(4,device='cuda',requires_grad=True)*.1,torch.randn(2+2*3,4,device='cuda',requires_grad=True)*.1,torch.randn(4,2,device='cuda',requires_grad=True)*.1,-torch.rand(4,3,device='cuda',requires_grad=True),torch.randn(4,device='cuda',requires_grad=True),torch.randn(4,device='cuda',requires_grad=True)-2]
        xz,cw,cb,xw,dw,A,D,db=args
        a=mamba_inner_no_out(xz,cw,cb,xw,dw,A,D=D,delta_bias=db);b=literal(*args)
        ga=torch.autograd.grad(a.square().sum(),args,retain_graph=True);gb=torch.autograd.grad(b.square().sum(),args)
        fe=float((a-b).abs().max());ge=max(float((x-y).abs().max()) for x,y in zip(ga,gb));assert fe<3e-5 and ge<2e-4
        rows.append({'N':n,'forward_max_error':fe,'gradient_max_error':ge,'reference':'literal causal depthwise conv + Euler input selective recurrence + gated D skip'})
    write(R/'MAMBAMIL_KERNEL_PORT_AUDIT.json',{'status':'PASS','checks':rows,'scope':'ported original SRMamba fast helper uses official released causal_conv1d and selective_scan kernels;SR reordering unchanged','exact_ZOH_used':False})
    print('MAMBAMIL_KERNEL_PORT_PASS',rows,flush=True)

if __name__=='__main__':main()
