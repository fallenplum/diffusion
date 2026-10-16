import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.phase17_ssd.data import R,write
from src.phase17_ssd.model import ConditionalS6

def main():
    torch.manual_seed(1709);mask=np.zeros((8,512),bool)
    for j in range(8):mask[j,j*64:(j+1)*64]=True
    base=ConditionalS6(41,mask,'A8',.25).cuda().double().eval()
    with torch.no_grad():base.delta_up.weight.normal_(0,.05);base.B_up.weight.normal_(0,.05);base.write.weight.normal_(0,.05)
    x=torch.randn(2,16,1536,device='cuda',dtype=torch.double);valid=torch.ones(2,16,device='cuda',dtype=torch.bool);c=torch.randn(2,41,device='cuda',dtype=torch.double);r=torch.randn(2,512,device='cuda',dtype=torch.double);g=r.abs()*.2;l=r.abs()*.1;obs=torch.ones_like(r);a=torch.zeros(2,device='cuda',dtype=torch.double);outputs={};rows=[]
    for arm in ['A2','A3','A4','A5','A6','A7','A8','A9','A10','A11_NO_CLIN']:
        model=copy.deepcopy(base);model.arm=arm;out=model(x,valid,c,r,g,l,obs,obs,a,context_index=torch.tensor([1,0],device='cuda') if arm=='A10' else None);out.sum().backward();outputs[arm]=out.detach().cpu().numpy()
        rows.append({'arm':arm,'registered_parameters':sum(p.numel() for p in model.parameters()),'elements_in_tensors_with_grad':sum(p.numel() for p in model.parameters() if p.grad is not None),'nonzero_gradient_entries':sum(int((p.grad.abs()>1e-12).sum()) for p in model.parameters() if p.grad is not None)})
    differences={a+'-'+b:float(abs(outputs[a]-outputs[b]).max()) for a,b in [('A8','A7'),('A7','A5'),('A7','A6'),('A3','A2'),('A4','A2'),('A8','A9'),('A8','A10')]}
    print('MEASURED_DIFFERENCES',differences,flush=True)
    # This gate tests a computational path, not clinical effect size. In double
    # precision a nonzero tiny wrong-context effect is valid path evidence and
    # still weak patient-specific sensitivity, explicitly reported as such.
    assert all(v>1e-12 for v in differences.values())
    film=sum(p.numel() for n,p in base.named_parameters() if n.startswith('film_'));conditional=sum(p.numel() for n,p in base.named_parameters() if n.startswith(('delta_','B_down','B_up')));assert film==conditional==1664
    write(R/'ABLATION_OPERATOR_AUDIT.json',{'status':'PASS','real_forward_differences_after_nonzero_adapter_activation':differences,'parameter_counts':rows,'FiLM_adapter_parameters':film,'delta_B_adapter_parameters':conditional,'note':'registered common mother graph includes inactive branches;gradient-bearing tensors and actually nonzero entries reported separately;synthetic checks are not survival performance'})
    print('ABLATION_OPERATOR_PASS',differences,flush=True)

if __name__=='__main__':main()
