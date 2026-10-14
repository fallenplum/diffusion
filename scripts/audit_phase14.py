"""Budget, actual interaction gradients, literal bootstrap and artifact audits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from run_phase14 import *
from src.phase10_trimodal.math import torch_cox,log_weights
from validate_phase11_independent import pairs,components,ipcw


def main():
    torch.set_num_threads(2);configuration=cfg();d=parent.load();endpoints=parent.endpoints(d);rows=[];hashes=[];fail=[];budgets=[]
    for path in sorted((PRIVATE/'cells').glob('*.json')):
        cell=json.loads(path.read_text());ep,gene,seed,fold=[cell[k] for k in ['endpoint','gene','seed','fold']];z=torch.load(cell['artifact'],map_location='cpu',weights_only=False)
        if z['source_hash']!=source() or cell['protocol_sha256']!=sha(CONFIG):raise RuntimeError('artifact version')
        hashes.append({'pipeline':path.stem,'artifact_sha256':sha(cell['artifact']),'cell_sha256':sha(path),'artifact_bytes':Path(cell['artifact']).stat().st_size})
        for family,infos in [('producer',z['producers']),('readout',z['downstream'])]:
            for name,info in infos.items():
                role,arm=name.split('_',1)
                if role=='outer':
                    if info['actual_epochs']!=infos['inner_'+arm]['selected_epoch']:raise RuntimeError('training schedule mismatch')
                else:
                    logs=info['logs'];rawmin=min(logs,key=lambda r:r['validation_EMA'])['epoch']
                    if info['selected_epoch']!=rawmin or info['actual_epochs']!=80:raise RuntimeError('epoch selection mismatch')
                    slope=float(np.polyfit(np.arange(10),[r['validation_EMA'] for r in logs[-10:]],1)[0])
                    budgets.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'family':family,'arm':arm,'selected_epoch':rawmin,'ceiling_selected':rawmin==80,'final10_EMA_slope':slope,'ceiling_and_improving':rawmin==80 and slope<0})
        f,m,a,tr,te=role_data(ep,gene,fold,'outer');ev=endpoints[ep];t,e=ev['time'],ev['event'];cs=np.c_[f['C'],f['S']];producer=RNAOnly(m['clinical_mask'],m[gene],seed).cuda();producer.load_state_dict(z['states']['outer_RANK']);_,taps=extract_rna(producer,f[gene],tr);del producer
        r=apply_rep(taps['processed_full'],z['transforms']['outer_RANK64']);base=json.loads((parent.PRIVATE/'formal/cells'/f'{ep}_CS_seed31_fold{fold}.json').read_text())['lambda']
        for arm,interaction,wrong in [('ADDITIVE',False,False),('INTERACT',True,False),('WRONG_ADDITIVE',False,True),('WRONG_INTERACT',True,True)]:
            rr=r[index_map(tr,z['permutations']['outer_fit'])] if wrong else r;q=Conditional(cs.shape[1],64,np.zeros(cs.shape[1]+64),interaction,configuration,seed).cuda();q.load_state_dict(z['states']['outer_'+arm]);q.eval()
            cc=torch.tensor(cs[tr],device='cuda');rna=torch.tensor(rr,device='cuda');risk=q(cc,rna);loss=torch_cox(risk,log_weights(t[tr],e[tr],'cuda'),torch.tensor(e[tr],device='cuda'))+q.regularization(base,configuration);loss.backward()
            gradients={k:float(p.grad.norm()) for k,p in q.named_parameters()};correction=q.correction(cc,rna).detach()
            rows.append({'endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'arm':arm,'objective_gradient_norm':float(np.sqrt(sum(v*v for v in gradients.values()))),'main_gradient':gradients['main'],'left_gradient':gradients.get('left',0),'right_gradient':gradients.get('right',0),'mix_gradient':gradients.get('mix',0),'train_correction_max':float(correction.abs().max()),'train_correction_RMS':float(correction.square().mean().sqrt()),'base_weight_norm':float(q.main[:cs.shape[1]].detach().norm()),'RNA_weight_norm':float(q.main[cs.shape[1]:].detach().norm())})
            del q
    pd.DataFrame(rows).to_csv(REPORT/'INTERACTION_GRADIENTS.csv',index=False);pd.DataFrame(budgets).to_csv(REPORT/'BUDGET_AUDIT.csv',index=False);pd.DataFrame(hashes).to_csv(REPORT/'PRIVATE_ARTIFACT_HASHES.csv',index=False)
    # Ten literal patient resamples for each endpoint; primary five HALL contrasts.
    comparisons=[('RANK64','TOKEN64'),('POOL64','TOKEN64'),('INTERACT','ADDITIVE'),('INTERACT','COX'),('INTERACT','WRONG_INTERACT')];maximum=0;checked=0
    for ep in configuration['endpoints']:
        ev=endpoints[ep];t,e,valid=[ev[k] for k in ['time','event','valid']];weights=np.zeros(512);risks={}
        for fold in configuration['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));weights[te]=ipcw(t[tr],e[tr],t[te],e[te],60)
            for seed in configuration['seeds']:
                cell=json.loads((PRIVATE/'cells'/f'{unit(ep,"HALL",seed,fold)}.json').read_text());z=torch.load(cell['artifact'],map_location='cpu',weights_only=False)
                for model in {x for pair in comparisons for x in pair}:
                    if (model,seed) not in risks:risks[model,seed]=np.zeros(512)
                    risks[model,seed][te]=z['risks'][model]
        local=np.flatnonzero(valid);ii,jj=pairs(t[local],e[local]);ii,jj=local[ii],local[jj];keep=d['folds'][ii]==d['folds'][jj];ii,jj=ii[keep],jj[keep];rng=np.random.default_rng(configuration['bootstrap_seed']+(ep=='PFI'))
        for draw in range(10):
            counts=np.zeros(512,int);counts[local]=rng.multinomial(len(local),np.ones(len(local))/len(local));q=counts[ii]*counts[jj]
            for a,b in comparisons:
                for seed in configuration['seeds']:
                    delta_a=risks[a,seed][ii]-risks[a,seed][jj];delta_b=risks[b,seed][ii]-risks[b,seed][jj];credit=np.where(abs(delta_a)<=1e-8,.5,delta_a>0)-np.where(abs(delta_b)<=1e-8,.5,delta_b>0)
                    for metric in ['Harrell','Uno60']:
                        ww=np.ones(len(q)) if metric=='Harrell' else weights[ii];matrix=float((q*ww)@credit/(q*ww).sum());numerator=0.;denominator=0.
                        for fold in configuration['folds']:
                            ix=np.flatnonzero(valid&(d['folds']==fold));expanded=np.repeat(ix,counts[ix]);w=None if metric=='Harrell' else weights[expanded]
                            an,dd=components(t[expanded],e[expanded],risks[a,seed][expanded],w);bn,bd=components(t[expanded],e[expanded],risks[b,seed][expanded],w);numerator+=an-bn;denominator+=dd
                        error=abs(matrix-numerator/denominator);maximum=max(maximum,error);checked+=1
                        if error>1e-10:raise RuntimeError('literal bootstrap mismatch')
    # Deliberate failures never write or mutate formal artifacts.
    rejected=0
    for ix,perm in [(np.array([2,1]),np.array([1,3])),(np.array([1,1]),np.array([1,1]))]:
        try:index_map(ix,perm)
        except ValueError:rejected+=1
    from evaluate_phase14 import compare
    try:compare(np.array([1.,2.]),np.array([1.,2.1]),'injected risk')
    except RuntimeError:rejected+=1
    if rejected!=3:raise RuntimeError('fault rejection failure')
    result={'status':'PASS','pipelines':len(hashes),'schedule_checks':len(budgets),'literal_bootstrap_checks':checked,'literal_bootstrap_max_error':maximum,'rejected_faults':rejected,'interaction_gradient_records':len(rows),'source_hash':source()}
    write(REPORT/'ADDITIONAL_AUDIT.json',result);print(json.dumps(result))


if __name__=='__main__':main()
