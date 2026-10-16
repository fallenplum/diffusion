"""Real parameter/gradient and full-A8 parity checks, no exploratory checkpoint."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.phase17r_ssd.data import P,R,cfg,source,write,seed_all,old
from src.phase17r_ssd.model import RepairS6
from src.phase17_ssd.model import ConditionalS6

def main():
    R.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True);seed_all(42)
    assert old.source()==cfg()['parent_training_source']
    manifest=json.loads((old.R/'ARTIFACT_MANIFEST.json').read_text())
    for entry in manifest['public_files']:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    d=old.parent.load();assert len(d['patient_ids'])==512 and int(d['DSS_event'].sum())==65
    visual=old.prepare_visual();assert visual['patients']==512 and visual['slides']==519
    membership=np.ones((8,512),bool);B,N=2,16
    visual=torch.randn(B,N,1536,device='cuda');mask=torch.ones(B,N,device='cuda',dtype=torch.bool);mask[0,-3:]=False
    clinical=torch.randn(B,12,device='cuda');mol=[torch.randn(B,512,device='cuda') for _ in range(3)];masks=[torch.ones(B,512,device='cuda')]*2;anchor=torch.randn(B,device='cuda')
    args=(visual,mask,clinical,*mol,*masks,anchor)
    seed_all(13);parent_model=ConditionalS6(12,membership,'A8',.25).cuda().eval();seed_all(13);full=RepairS6(12,membership,'R1_A8_full').cuda().eval()
    assert parent_model.state_dict().keys()==full.state_dict().keys()
    for k in parent_model.state_dict():assert torch.equal(parent_model.state_dict()[k],full.state_dict()[k]),k
    a=parent_model(*args);b=full(*args);parity=float((a-b).abs().max());assert parity<1e-7
    a.sum().backward();b.sum().backward()
    for (k,p),(j,q) in zip(parent_model.named_parameters(),full.named_parameters()):
        assert k==j
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-7,rtol=1e-6),k
    rows=[]
    for arm in cfg()['R1']+cfg()['R2']['models']:
        seed_all(13);model=RepairS6(12,membership,arm,.25).cuda().eval()
        if arm!='R1_A8_full':assert model.direct is None and not any(k.startswith('direct.') for k in model.state_dict())
        if 'A8' in arm:
            with torch.no_grad():model.delta_up.weight.fill_(.02);model.B_up.weight.fill_(.02)
        if arm=='R2_FiLM':
            with torch.no_grad():model.film_up.weight.fill_(.02)
        if arm=='R2_Input':
            with torch.no_grad():model.input_up.weight.fill_(.02)
        risk,d=model(*args,diagnostic=True)
        if 'A8' in arm or arm.startswith('R2'):
            gg=torch.autograd.grad(d['ssm_risk'].sum(),d['g'],retain_graph=True)[0];assert gg.abs().max()>0,arm
        else:gg=torch.zeros(1,device='cuda')
        risk.sum().backward()
        if arm.startswith('R2'):
            prefixes=['delta_down.','delta_up.','B_down.','B_up.'] if arm=='R2_A8' else ['film_down.','film_up.'] if arm=='R2_FiLM' else ['input_down.','input_up.']
            size=sum(p.numel() for k,p in model.named_parameters() if any(k.startswith(z) for z in prefixes));assert size==1664
            if arm=='R2_A8':assert d['conditional_delta'].abs().max()<=.25 and d['conditional_B'].abs().max()<=.25
        else:size=None
        _,off=model(*args,gate_override=0.,diagnostic=True);assert float(off['h'].abs().max())<1e-7
        rows.append({'arm':arm,'registered_parameters':sum(p.numel() for p in model.parameters()),'matched_adapter_parameters':size,'isolated_SSM_g_gradient_absmax':float(gg.abs().max()),'noDirect':model.direct is None,'finite':bool(torch.isfinite(risk).all())})
    write(R/'OPERATOR_AUDIT.json',{'status':'PASS','source':source(),'parent_manifest_files_unchanged':len(manifest['public_files']),'patients':512,'DSS_events':65,'WSIs':519,'visual_cache_all_source_hashes_verified':True,'full_A8_parent_output_max_error':parity,'full_A8_parent_parameter_and_gradient_parity':True,'gate_zero_h_error_below':1e-7,'rows':rows,'parent_exact_ZOH_and_Cox_audits':'inherited unchanged and hash-locked; no fake Mamba2 SSD claim'})
    write(R/'PROTOCOL_LOCK.json',{'status':'LOCKED_BEFORE_FIRST_TRAINING','source':source(),'config':cfg(),'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'outer_scores_not_available':not any((P/'outer').glob('*.pt'))})
    print('OPERATOR_AND_LOCK_PASS',source(),flush=True)

if __name__=='__main__':main()
