"""RNA-free shared-coordinate visual probes; all PCA and scaling fit-local."""
import sys,json,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,h5py
from sklearn.decomposition import PCA
from src.phase16_diagnostic.core import *
from src.phase10_trimodal.features import scale
from src.phase10_trimodal.math import fit_ridge
from src.phase13_rna.math import inner_score

VC=ROOT/'configs/phase16_visual_diagnostic.json'
def vsource():return hashlib.sha256((source()+sha(VC)+sha(Path(__file__))).encode()).hexdigest()

def prepare():
    target=P/'PAIRED_VISUAL.npz';audit=P/'PAIRED_VISUAL.json'
    if audit.exists():
        a=json.loads(audit.read_text())
        if a['source']!=vsource() or sha(target)!=a['sha256']:raise ValueError('paired visual data changed')
        return dict(np.load(target))
    meta=base.manifest();inv=json.loads((base.P/'ASSET_INVENTORY.json').read_text());con=sorted(inv['chosen_conch'],key=lambda r:(r['patient_index'],r['stem']));uni=[r for r in inv['h5'] if r['kind']=='UNI2H_CANDIDATE'];rank=np.load(base.P/'selectors/GRID_seed31_1024.npy');maps=[];counts=[]
    for i,s in enumerate(meta['slides']):
        mp=base.P/'coordinate_maps'/f'slide_{i+1:04d}.npz'
        if not mp.exists():raise ValueError('missing computational coordinate map')
        m=dict(np.load(mp));eligible=np.unique(m['conch_row']);maps.append(m);counts.append(len(eligible))
    quotas=np.zeros(len(counts),int)
    for patient in range(512):
        ss=np.array([i for i,s in enumerate(meta['slides']) if s['patient']==patient]);nn=np.asarray(counts)[ss];used=np.zeros(len(ss),int)
        if nn.sum()==0:raise ValueError('patient has no matched visual data; cannot claim full-cohort comparison')
        for _ in range(min(512,int(nn.sum()))):q=nn/(used+1);q[used>=nn]=-1;used[q.argmax()]+=1
        quotas[ss]=used
    C=np.zeros((512,768));U=np.zeros((512,1536));wsi_n=np.zeros(512);rows=[];selection={}
    for i,(s,c,m) in enumerate(zip(meta['slides'],con,maps)):
        q=int(quotas[i]);patient=s['patient'];eligible=np.unique(m['conch_row']);matches=[r for r in uni if r['stem']==c['stem'] and r['patient_index']==patient]
        if not matches:raise ValueError('UNI2 source mapping absent')
        u=sorted(matches,key=lambda r:len(r['path']))[0]
        if sha(c['path'])!=str(m['conch_h5_sha256']) or sha(u['path'])!=str(m['uni2_h5_sha256']):raise ValueError('coordinate map source file changed')
        if not q:continue
        with h5py.File(c['path'],'r') as h:cx=h['features'][:].reshape(-1,768).astype(np.float64);xy=h['coords'][:].reshape(-1,2)
        first=rank[i][rank[i]>=0]-s['offset'];first=first[np.isin(first,eligible)];left=np.setdiff1d(eligible,first);left=left[np.lexsort((xy[left,1],xy[left,0]))];take=np.r_[first,left][:q];fallback=max(0,q-len(first));selection[str(i)]=take
        keep=np.isin(m['conch_row'],take);rr=m['conch_row'][keep];jj=m['uni2_row'][keep];area=m['intersection_area'][keep].astype(float)
        with h5py.File(u['path'],'r') as h:ux=h['features'][:].reshape(-1,1536).astype(np.float64)
        cx/=np.maximum(np.linalg.norm(cx,axis=1)[:,None],1e-12);ux/=np.maximum(np.linalg.norm(ux,axis=1)[:,None],1e-12)
        order=np.argsort(take);sorted_take=take[order];rowpos=order[np.searchsorted(sorted_take,rr)];aligned=np.zeros((q,1536));den=np.bincount(rowpos,weights=area,minlength=q)
        np.add.at(aligned,rowpos,ux[jj]*area[:,None]);aligned/=den[:,None];aligned/=np.maximum(np.linalg.norm(aligned,axis=1)[:,None],1e-12)
        C[patient]+=cx[take].mean(0);U[patient]+=aligned.mean(0);wsi_n[patient]+=1
        rows.append({'anonymous_slide_number':i+1,'candidate_conch_rows':s['n'],'matched_candidate_rows':len(eligible),'selected_patches':q,'lexicographic_fallback_rows':fallback,'minimum_matched_area':float(den.min()),'mean_UNI2_subpatches':float(len(jj)/q)})
        if i%80==0:print('VISUAL_SOURCE_SLIDE',i,flush=True)
    if np.any(wsi_n==0):raise ValueError('empty patient visual pool')
    C/=wsi_n[:,None];U/=wsi_n[:,None];np.savez_compressed(target,C=C,U=U);np.savez_compressed(P/'PAIRED_VISUAL_SELECTION_PRIVATE.npz',**selection)
    pd=None
    import pandas as pd
    pd.DataFrame(rows).to_csv(R/'PAIRED_VISUAL_COVERAGE.csv',index=False);write(audit,{'status':'PASS','source':vsource(),'sha256':sha(target),'patients':512,'slides':len(rows),'selected_patches':sum(r['selected_patches'] for r in rows),'fallback_rows':sum(r['lexicographic_fallback_rows'] for r in rows),'raw_image_QA':False});write(R/'PAIRED_VISUAL_AUDIT.json',json.loads(audit.read_text()));print('PAIRED_VISUAL_READY',flush=True)
    return {'C':C,'U':U}

def visual_features(d,data,fit):
    c,cs=scale(data['C'],fit);u,us=scale(data['U'],fit);pc=PCA(32,svd_solver='full').fit(c[fit]);pu=PCA(32,svd_solver='full').fit(u[fit]);cc=pc.transform(c);uu=pu.transform(u)
    arrays={'CONCH32':cc,'UNI32':uu,'DUAL32':np.c_[cc[:,:16],uu[:,:16]],'DUAL64':np.c_[cc,uu]};csx,st=base.cs_features(d,fit);result={};scalers={}
    for rep,x in arrays.items():
        z,ss=scale(x,fit);scalers[rep]=ss
        result['VISUAL_ONLY__'+rep]=z;result['CS_PLUS_VISUAL__'+rep]=np.c_[csx,z]
    result['CS']=csx
    return result,{'fit':fit,'C_scale':cs,'U_scale':us,'C_components':pc.components_,'C_pca_mean':pc.mean_,'U_components':pu.components_,'U_pca_mean':pu.mean_,'scalers':scalers,'CS':st}

def run(fold,data):
    target=P/'visual_folds'/f'fold{fold}.pt';cell=target.with_suffix('.json')
    if cell.exists():
        q=json.loads(cell.read_text())
        if q['source']!=vsource() or sha(target)!=q['sha256']:raise ValueError('stale visual fit')
        return
    d=parent.load();parts=parent.splits(d,'DSS',fold);t,e=[parent.endpoints(d)['DSS'][k] for k in ['time','event']];it,iv,ot,te=[parts[k] for k in ['inner_train','inner_val','outer_train','outer_test']]
    xi,si=visual_features(d,data,it);xo,so=visual_features(d,data,ot);models={};trials={}
    for rep in xi:
        rows=[]
        for lam in cfg()['grid']:
            if lam=='OFF':w=np.zeros(xi[rep].shape[1]);dg={'OFF':True,'gradient_max':0.}
            else:w,dg=fit_ridge(xi[rep][it],t[it],e[it],lam)
            rows.append({'lambda':lam,'inner_Harrell':inner_score(t[iv],e[iv],xi[rep][iv]@w),'optimizer':dg})
        best=max(r['inner_Harrell'] for r in rows);tied=[r for r in rows if abs(r['inner_Harrell']-best)<1e-12];chosen='OFF' if any(r['lambda']=='OFF' for r in tied) else max(r['lambda'] for r in tied);trials[rep]=rows
        for schedule,lam in [('FIXED1',1.),('SELECTED',chosen)]:
            if lam=='OFF':w=np.zeros(xo[rep].shape[1]);dg={'OFF':True,'gradient_max':0.}
            else:w,dg=fit_ridge(xo[rep][ot],t[ot],e[ot],lam)
            models[rep+'__'+schedule]={'w':w,'lambda':lam,'risk':xo[rep][te]@w,'optimizer':dg,'parameters':len(w)}
    target.parent.mkdir(exist_ok=True);torch.save({'source':vsource(),'fold':fold,'parts':parts,'states':{'inner':si,'outer':so},'features':{'inner':xi,'outer':xo},'models':models,'trials':trials},target);write(cell,{'status':'COMPLETE','source':vsource(),'sha256':sha(target),'models':18,'outer_test_selection':False});print('VISUAL_FOLD_DONE',fold,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);lock();identity={'source':vsource(),'protocol_sha256':sha(VC)};p=P/'VISUAL_LOCK.json'
    if p.exists() and json.loads(p.read_text())!=identity:raise ValueError('visual protocol/source changed')
    write(p,identity);write(R/'VISUAL_PROTOCOL_LOCK.json',identity);data=prepare()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lambda f:run(f,data),cfg()['folds']))
    write(P/'VISUAL_TRAINING_DONE.json',{'status':'PASS','unique_outer_fits':90,'arms':18,'seed_references':[31,47,83],'source':vsource()})
