"""Executable hard gates. Fails closed; never fabricates PASS on exceptions."""
import sys,json,time,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase17_ssd.ssm import reference_scan,cuda_final_scan
from src.phase17_ssd.model import ConditionalS6
from src.phase17_ssd.losses import full_risk_efron
from src.phase17_ssd.data import R,P,write,molecular_features,parent,wrong_map


def run():
    torch.manual_seed(1701);torch.set_num_threads(2);rows=[]
    for n in [1,16,128,512]:
        args=[torch.randn(1,n,4,device='cuda',dtype=torch.float64,requires_grad=True),
              (torch.rand(1,n,4,device='cuda',dtype=torch.float64)*.2).requires_grad_(),
              torch.randn(1,n,3,device='cuda',dtype=torch.float64,requires_grad=True),
              (-torch.rand(4,3,device='cuda',dtype=torch.float64)).requires_grad_()]
        h0=torch.randn(1,4,3,device='cuda',dtype=torch.float64,requires_grad=True)
        h=reference_scan(*args,h0=h0);v=cuda_final_scan(*args,h0=h0)
        gh=torch.autograd.grad(h.square().sum(),args+[h0],retain_graph=True)
        gv=torch.autograd.grad(v.square().sum(),args+[h0])
        fe=float((h-v).abs().max());ge=max(float((a-b).abs().max()) for a,b in zip(gh,gv))
        if fe>1e-9 or ge>1e-8:raise ValueError(('scan mismatch',n,fe,ge))
        hold=reference_scan(args[0],torch.zeros_like(args[1]),args[2],args[3],h0=h0)
        if float((hold-h0).abs().max())>1e-6:raise ValueError('m0 hold failed')
        rows.append({'N':n,'batch':1,'dtype':'float64','forward_max':fe,'gradient_max':ge,'m0_hold_error':float((hold-h0).abs().max())})
    # Float32/mixed input; state transitions explicitly compute in float32.
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        u=torch.randn(2,512,8,device='cuda',dtype=dtype);dt=torch.rand_like(u)*.1;B=torch.randn(2,512,4,device='cuda',dtype=dtype);A=-torch.rand(8,4,device='cuda')
        a=reference_scan(u.float(),dt.float(),B.float(),A);b=cuda_final_scan(u.float(),dt.float(),B.float(),A)
        err=float((a-b).abs().max());assert err<3e-5
        rows.append({'N':512,'batch':2,'input_dtype':str(dtype),'accumulator':'float32','forward_max':err})
    aa=[torch.randn(1,3,2,dtype=torch.double,requires_grad=True),torch.rand(1,3,2,dtype=torch.double,requires_grad=True),torch.randn(1,3,2,dtype=torch.double,requires_grad=True),(-torch.rand(2,2,dtype=torch.double)*1e-7).requires_grad_()]
    finite_difference=torch.autograd.gradcheck(cuda_final_scan,aa,eps=1e-6,atol=3e-5,rtol=1e-4)
    if not finite_difference:raise ValueError('finite difference')
    membership=np.ones((8,512),bool);model=ConditionalS6(41,membership,'A8').cuda();
    with torch.no_grad():model.delta_up.weight.normal_(0,.03);model.B_up.weight.normal_(0,.03)
    visual=torch.randn(2,16,1536,device='cuda');mask=torch.ones(2,16,device='cuda',dtype=torch.bool);mask[0,-3:]=False
    c=torch.randn(2,41,device='cuda');r=torch.randn(2,512,device='cuda');g=r*.1;l=-r*.2;rm=torch.ones_like(r);cm=rm.clone();rm[0,:30]=0;cm[1]=0
    risk,diag=model(visual,mask,c,r,g,l,rm,cm,torch.zeros(2,device='cuda'),diagnostic=True)
    diag['g'].retain_grad();risk.sum().backward()
    gradient={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    for k in ['delta_up.weight','delta_down.weight','B_up.weight','B_down.weight','clinical.weight','molecular.logits','write.weight']:
        if gradient.get(k,0)<=0:raise ValueError('missing real conditioning gradient '+k)
    # Set a common model to vanilla and verify m=1/U=0 has identical outputs.
    model.eval();vanilla=copy.deepcopy(model);vanilla.arm='A2'
    with torch.no_grad():
        model.delta_up.weight.zero_();model.B_up.weight.zero_()
        one=model(visual,mask,c,r,g,l,rm,cm,torch.zeros(2,device='cuda'),gate_override=1)
        base=vanilla(visual,mask,c,r,g,l,rm,cm,torch.zeros(2,device='cuda'))
    eq=float((one-base).abs().max());assert eq<1e-6
    # Independently enumerate tied Efron risk sets with explicit scalar loops.
    t=np.array([1.,2.,2.,2.,4.,5.]);e=np.array([1,1,1,0,0,1]);x=np.array([.1,-.3,.7,.4,-.2,.05])
    total=0.
    for tt in np.unique(t[e==1]):
        event=(t==tt)&(e==1);riskset=t>=tt;d=event.sum();total-=x[event].sum()
        for k in range(d):total+=np.log(np.exp(x[riskset]).sum()-k/d*np.exp(x[event]).sum())
    expected=total/e.sum();observed=float(full_risk_efron(torch.tensor(x),t,e));assert abs(expected-observed)<1e-12
    # Actual feature builders cannot read outer-test endpoint values: mutate raw labels.
    d=parent.load();features,audit=molecular_features(d,1,'inner');altered={k:v.copy() for k,v in d.items()};te=parent.splits(d,'DSS',1)['outer_test']
    altered['DSS_event'][te]=1-altered['DSS_event'][te];altered['DSS_time'][te]=altered['DSS_time'][te]*9+77
    features2,audit2=molecular_features(altered,1,'inner')
    assert all(np.array_equal(features[k],features2[k]) for k in features)
    parts=parent.splits(d,'DSS',1)
    for key in ['inner_train','inner_val','outer_train','outer_test']:wrong_map(d,parts[key],1703)
    write(R/'DATA_LEAKAGE_AUDIT.json',{'status':'PARTIAL_PASS','feature_outer_label_mutation':'PASS','split_local_derangements':'PASS','complete_training_checkpoint_mutation_test':'PENDING','RNA_missing_patients':int((np.isnan(d['raw_rna']).sum(1)==3024).sum()),'RNA_missing_columns':3024,'patient_OOF_public':False})
    out={'status':'PASS','implementation':'conditional S6 exact ZOH;torch CUDA reverse cumulative sum,NOT Mamba2 SSD kernel','scan_checks':rows,'finite_difference':bool(finite_difference),'conditional_parameter_gradients':gradient,'g_gradient_norm':float(diag['g'].grad.norm()),'m1_zero_U_vanilla_error':eq,'tied_Efron_error':abs(expected-observed),'missing_mask':'executed;all-CNA missing batch row','CUDA':torch.version.cuda,'torch':torch.__version__}
    write(R/'SSM_OPERATOR_AUDIT.json',out);print(json.dumps({'status':'PASS','checks':len(rows),'g_gradient_norm':out['g_gradient_norm']}),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as exc:
        write(R/'SSM_OPERATOR_AUDIT.json',{'status':'INVALID','error':repr(exc)})
        raise
