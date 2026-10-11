"""Phase9-G0R repair run.  Private OOF artifacts stay on the GPU host."""
from __future__ import annotations
import json, hashlib, sys, time, random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from phase8_v0.cox import efron_loss_grad, concordance, km_left
from phase9_g0_pathway_benchmark import parse_kegg, PATHWAYS, P_NAMES

SEEDS=[20261007,20261008,20261009]; RIDGE_GRID=[0.01,0.1,1.0,10.0]; ARMS=['R0','R1','R2','R3','R4','R5','R6','R7','O1','O2','O3']

def h(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def fitcox(x,t,e,lam):
    f=efron_loss_grad(np.asarray(x,float),np.asarray(t),np.asarray(e),lam); z=minimize(f,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':50000,'ftol':1e-14,'gtol':1e-9,'maxls':100}); loss,g=f(z.x); gm=float(np.max(np.abs(g)))
    if not np.isfinite(loss) or gm>2e-5: raise RuntimeError(f'cox failure {z.message} grad={gm}')
    return z.x,{'lambda':lam,'iterations':int(z.nit),'gradient_max':gm,'objective':float(loss)}
def select_ridge(xi,xv,t,e,ti,vi):
    tr=[]
    for lam in RIDGE_GRID:
        w,d=fitcox(xi[ti],t[ti],e[ti],lam); tr.append({'lambda':lam,'val_harrell':concordance(t[vi],e[vi],xv[vi]@w),**d})
    best=max(tr,key=lambda q:(q['val_harrell'],-q['lambda'])); return best['lambda'],tr
def norm(x,fit):
    x=np.asarray(x,float); med=np.nanmedian(x[fit],axis=0); y=np.where(np.isfinite(x),x,med); sd=np.nanstd(y[fit],axis=0); sd[sd<1e-8]=1.; return (y-med)/sd,med,sd
def top512(rna,fit):
    y,med,sd=norm(rna,fit); raw=np.where(np.isfinite(rna),rna,med); var=np.nanvar(raw[fit],axis=0); good=np.mean(np.isfinite(rna[fit]),axis=0)>=.9; good &= var>1e-8; cand=np.flatnonzero(good); order=cand[np.argsort(-var[cand],kind='mergesort')]; keep=order[:min(512,len(order))]
    if len(keep)<512: raise RuntimeError(f'only {len(keep)} usable RNA genes for Top512')
    return y[:,keep],keep,{'keep_sha256':h(keep),'keep_n':len(keep),'raw_variance_max':float(var[keep[0]]),'usable_n':int(good.sum())}
def pathway_data(d,kroot,fit):
    sym,members,edges=parse_kegg(Path(kroot)); rs=np.asarray(d['rna_gene_symbols']).astype(str); cs=np.asarray(d['cna_gene_symbols']).astype(str); rmap={q.upper():i for i,q in enumerate(rs)}; cmap={q.upper():i for i,q in enumerate(cs)}; gids=sorted(set(sum(members,[]))); nodes=[]; ri=[]; ci=[]; aliases={}
    for g in gids:
        names=[q for q,v in sym.items() if v==g]; aliases[g]=names; a=next((rmap[q] for q in names if q in rmap),None); b=next((cmap[q] for q in names if q in cmap),None)
        if a is not None or b is not None: nodes.append(g); ri.append(a); ci.append(b)
    rn=np.asarray(d['rna_genes'],float); cn=np.asarray(d['cna_genes'],float); X=np.full((len(rn),len(nodes),3),np.nan)
    for j,(a,b) in enumerate(zip(ri,ci)):
        if a is not None: X[:,j,0]=rn[:,a]
        if b is not None: X[:,j,1]=np.maximum(cn[:,b],0); X[:,j,2]=np.maximum(-cn[:,b],0)
    for q in range(3):
        med=np.nanmedian(X[fit,:,q],axis=0); med=np.where(np.isfinite(med),med,0.); X[:,:,q]=np.where(np.isfinite(X[:,:,q]),X[:,:,q],med); sd=np.std(X[fit,:,q],axis=0); sd[sd<1e-8]=1.; X[:,:,q]=(X[:,:,q]-med)/sd
    gi={g:i for i,g in enumerate(nodes)}; A=np.eye(len(nodes)); true=[]
    for a,b in edges:
        if a in gi and b in gi and a!=b: A[gi[a],gi[b]]=A[gi[b],gi[a]]=1.; true.append((gi[a],gi[b]))
    A=A/np.maximum(A.sum(1,keepdims=True),1); P=[[gi[g] for g in mm if g in gi] for mm in members]; stats=[]
    for q in P: stats.append(np.nanmean(X[:,q,:],axis=1) if q else np.zeros((len(X),3)))
    return X,A,P,np.stack(stats,1),true,{'nodes':len(nodes),'edges':len(true),'isolated':int(sum((A-np.diag(np.diag(A))).sum(1)==0)),'pathway_members':[len(q) for q in P],'mapping_symbols':len(sym),'node_hash':h(np.asarray(nodes,dtype='U'))}
def swap_graph(A,seed,swaps=500):
    rng=np.random.default_rng(seed); n=A.shape[0]; E={(i,j) for i in range(n) for j in range(i+1,n) if A[i,j]>0 and i!=j}; orig=len(E); success=0
    for _ in range(swaps):
        if len(E)<2: break
        e1,e2=rng.choice(len(E),2,replace=False); a,b=list(E)[e1]; c,d=list(E)[e2]
        if len({a,b,c,d})<4: continue
        cand=[tuple(sorted((a,d))),tuple(sorted((c,b)))]
        if cand[0][0]==cand[0][1] or cand[1][0]==cand[1][1] or cand[0] in E or cand[1] in E or cand[0]==cand[1]: continue
        E.remove((a,b)); E.remove((c,d)); E.add(cand[0]); E.add(cand[1]); success+=1
    B=np.eye(n)
    for a,b in E: B[a,b]=B[b,a]=1
    B=B/np.maximum(B.sum(1,keepdims=True),1); deg=np.sum(A>0,axis=1); deg2=np.sum(B>0,axis=1)
    return B,{'original_edges':orig,'final_edges':len(E),'successful_swaps':success,'degree_exact':bool(np.array_equal(deg,deg2)),'swap_seed':seed}
def cox_torch(r,t,e):
    import torch
    loss=torch.tensor(0.,device=r.device); ne=int(e.sum().item())
    for tv in torch.unique(t[e>0]):
        die=(t==tv)&(e>0); m=int(die.sum().item()); ex=torch.exp(r-r.max()); risk=t>=tv
        for l in range(m): loss=loss+torch.log((ex*(risk.float()-(l/m)*die.float())).sum())+r.max()-r[die].sum()/m
    return loss/max(ne,1)
def graph_fit(X,A,P,train,val,base_train,base_val,t,e,seed,epochs,base_all=None,fixed_epoch=None):
    import torch
    torch.manual_seed(seed); dev='cuda' if torch.cuda.is_available() else 'cpu'; x=torch.tensor(X,dtype=torch.float32,device=dev); aa=torch.tensor(A,dtype=torch.float32,device=dev); enc=torch.nn.Linear(3,8).to(dev); head=torch.nn.Linear(8,1).to(dev); sw=torch.nn.Parameter(torch.full((32,),.1,device=dev)); opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters())+[sw],lr=2e-3,weight_decay=1e-3); M=torch.zeros((32,A.shape[0]),device=dev)
    for k,q in enumerate(P):
        if q: M[k,q]=1/len(q)
    tt=torch.tensor(t[train],dtype=torch.float32,device=dev); ee=torch.tensor(e[train],dtype=torch.int64,device=dev); bt=torch.tensor(base_train,dtype=torch.float32,device=dev); bv=torch.tensor(base_val,dtype=torch.float32,device=dev); best=(-np.inf,1,None); trace=[]
    def f(xx,bb):
        z=torch.relu(enc(xx)); z=torch.relu(torch.einsum('ij,bjk->bik',aa,z)); p=torch.stack([torch.einsum('n,bnh->bh',M[k],z) for k in range(32)],1); res=(head(p).squeeze(-1)*sw).sum(1)*.25; return bb+res
    init_grad=None
    for ep in range(1,epochs+1) if fixed_epoch is None else []:
        opt.zero_grad(); r=f(x[train],bt); loss=cox_torch(r,tt,ee)+1e-3*(sw@sw+sum((q*q).sum() for q in enc.parameters())); loss.backward();
        if ep==1: init_grad=float(sum((q.grad.abs().sum().item() for q in list(enc.parameters())+list(head.parameters())+[sw] if q.grad is not None)))
        gn=float(torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters())+[sw],5)); opt.step();
        with torch.no_grad(): rv=f(x[val],bv).detach().cpu().numpy()
        vh=concordance(t[val],e[val],rv); trace.append({'epoch':ep,'train_loss':float(loss.detach().cpu()),'val_harrell':vh,'grad_norm':gn})
        if np.isfinite(vh) and vh>best[0]+1e-10: best=(vh,ep,None)
    selected=int(best[1]) if fixed_epoch is None else int(fixed_epoch)
    # Refit from scratch for exactly selected epochs on requested outer-train set.
    torch.manual_seed(seed); enc2=torch.nn.Linear(3,8).to(dev); head2=torch.nn.Linear(8,1).to(dev); sw2=torch.nn.Parameter(torch.full((32,),.1,device=dev)); opt2=torch.optim.AdamW(list(enc2.parameters())+list(head2.parameters())+[sw2],lr=2e-3,weight_decay=1e-3); bt2=torch.tensor(base_train,dtype=torch.float32,device=dev); tt2=torch.tensor(t[train],dtype=torch.float32,device=dev); ee2=torch.tensor(e[train],dtype=torch.int64,device=dev)
    def f2(xx,bb):
        z=torch.relu(enc2(xx)); z=torch.relu(torch.einsum('ij,bjk->bik',aa,z)); p=torch.stack([torch.einsum('n,bnh->bh',M[k],z) for k in range(32)],1); return bb+(head2(p).squeeze(-1)*sw2).sum(1)*.25
    for _ in range(selected): opt2.zero_grad(); loss=cox_torch(f2(x[train],bt2),tt2,ee2)+1e-3*(sw2@sw2+sum((q*q).sum() for q in enc2.parameters())); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc2.parameters())+list(head2.parameters())+[sw2],5); opt2.step()
    with torch.no_grad(): out=f2(x,torch.tensor(base_all,dtype=torch.float32,device=dev) if base_all is not None else torch.zeros(len(x),device=dev)).detach().cpu().numpy()
    return out,{'selected_epoch':selected,'inner_best_harrell':float(best[0]),'trace':trace,'initial_gradient_l1':init_grad,'parameters':sum(q.numel() for q in list(enc2.parameters())+list(head2.parameters())+[sw2]),'device':dev}
def cross_anchor(C,S,fit,lam,seed,t,e):
    fit=np.asarray(fit); out=np.full(len(t),np.nan); rng=np.random.default_rng(seed); parts=np.array_split(rng.permutation(fit),3)
    for j,v in enumerate(parts):
        tr=np.setdiff1d(fit,v); w,_=fitcox(np.c_[C[tr],S[tr]],t[tr],e[tr],lam); out[v]=np.c_[C[v],S[v]]@w
    return out
def run(out,data_parent,ctxdir,kroot):
    out=Path(out); (out/'oof').mkdir(parents=True,exist_ok=True); (out/'cells').mkdir(exist_ok=True); d=dict(np.load(Path(data_parent)/'private_dataset.npz',allow_pickle=True)); t=d['DSS_time']; e=d['DSS_event']; audit=[]
    for seed in SEEDS:
      for fold in range(1,6):
        ctx=np.load(Path(ctxdir)/f'seed{seed}_fold{fold}.npz'); tr,te,it,iv=[ctx[q] for q in ('tr','te','it','iv')]; innerX={q:np.c_[ctx['inner_C'],ctx['inner_S']] for q in ['base']}; outerX=np.c_[ctx['outer_C'],ctx['outer_S']]; lam,trials=select_ridge(innerX['base'],innerX['base'],t,e,it,iv); wb,_=fitcox(outerX[tr],t[tr],e[tr],lam); base_te=outerX[te]@wb; base_all=np.zeros(len(t)); base_all[te]=base_te; base_all[tr]=cross_anchor(ctx['outer_C'],ctx['outer_S'],tr,lam,seed,t,e)[tr];
        def save(arm,risk,extra):
            np.savez_compressed(out/'oof'/f'DSS_seed{seed}_fold{fold}_{arm}.npz',test_indices=te,risk=np.asarray(risk),base_test_risk=np.asarray(base_te),seed=seed,fold=fold); (out/'cells'/f'DSS_seed{seed}_fold{fold}_{arm}.json').write_text(json.dumps({'seed':seed,'fold':fold,'arm':arm,'lambda':lam,'base_test_sha256':h(base_te),'test_risk_sha256':h(np.asarray(risk)),'test_indices_private':True,**extra},indent=2),encoding='utf8')
        save('R0',base_te,{'ridge_trials':trials,'base_identity_pass':True})
        R,keep,rs=top512(d['rna_genes'],tr); Ri,keep_i,ris_i=top512(d['rna_genes'],it); # separate train-only selection for inner epoch/lambda selection
        rr,_,_=norm(d['rna_genes'],tr); R=rr[:,keep]; rmean=R[tr].mean(0); rsd=R[tr].std(0); rsd[rsd<1e-8]=1; R=(R-rmean)/rsd
        # PCA fitted only on outer-train Top512 values.
        pca=PCA(16,svd_solver='full',random_state=seed).fit(R[tr]); RP=pca.transform(R)
        pca_i=PCA(16,svd_solver='full',random_state=seed).fit(Ri[it]); RPi=pca_i.transform(Ri)
        O=ctx['outer_O']; feats={'R1':R,'R2':RP,'R3':O[:,16:32]}
        for arm,F in feats.items():
            X=np.c_[outerX,F]; la,ttt=select_ridge(np.c_[innerX['base'],ctx['inner_O'][:,0:512] if False else np.zeros((len(t),F.shape[1]))],np.c_[innerX['base'],np.zeros((len(t),F.shape[1]))],t,e,it,iv) if False else (lam,trials)
            # use the same locked lambda grid and inner validation on the actual feature block
            Fi=RPi if arm=='R2' else (ctx['inner_O'][:,16:32] if arm=='R3' else (Ri if arm=='R1' else F))
            full=np.c_[ctx['inner_C'],ctx['inner_S'],Fi]; la,ttt=select_ridge(full,full,t,e,it,iv); w,_=fitcox(X[tr],t[tr],e[tr],la); save(arm,(X[te]@w),{'lambda':la,'ridge_trials':ttt,'feature_hash':h(F[tr])})
        X,A,P,stats,edges,ga=pathway_data(d,kroot,tr); Xi,Ai,Pi,stati,_,_=pathway_data(d,kroot,it); # R4 uses 32x3 frozen stats
        for arm,F0,Fi0 in [('R4',stats,stati)]:
            F=F0.reshape(len(F0),-1); Fi=Fi0.reshape(len(Fi0),-1)
            full=np.c_[ctx['inner_C'],ctx['inner_S'],Fi]; la,ttt=select_ridge(full,full,t,e,it,iv); w,_=fitcox(np.c_[outerX,F][tr],t[tr],e[tr],la); save(arm,np.c_[outerX,F][te]@w,{'lambda':la,'ridge_trials':ttt,'pathway_stats':ga})
        # Omics-only controls use the same fold-local representations and ridge grid.
        for arm,F0,Fi0 in [('O1',R,Fi if False else R),('O2',stats,stati)]:
            F=F0.reshape(len(F0),-1) if arm=='O2' else F0; Fi=Fi0.reshape(len(Fi0),-1) if arm=='O2' else Fi0
            if arm=='O1':
                ri,_,_=norm(d['rna_genes'],it); Fi=ri[:,keep]; mu=Fi[it].mean(0); sd=Fi[it].std(0); sd[sd<1e-8]=1.; Fi=(Fi-mu)/sd
            la,ttt=select_ridge(Fi,Fi,t,e,it,iv); w,_=fitcox(F[tr],t[tr],e[tr],la); z=F[te]@w; save(arm,z,{'lambda':la,'ridge_trials':ttt,'omics_only':True})
        # Neural arms: each uses nested inner anchor and inner-val epoch selection, then outer refit.
        inner_base_train=cross_anchor(ctx['inner_C'],ctx['inner_S'],it,lam,seed+fold,t,e)[it]; wi,_=fitcox(np.c_[ctx['inner_C'],ctx['inner_S']][it],t[it],e[it],lam); inner_base_val=np.c_[ctx['inner_C'],ctx['inner_S']][iv]@wi
        Xo,Ao,Po,_,_,go=pathway_data(d,kroot,tr); Xi,Ai,Pi,_,_,_=pathway_data(d,kroot,it); graph_specs=[('R5',np.eye(Ao.shape[0]),100,0),('R6',Ao,100,0)];
        for j in range(5): graph_specs.append(('R7',swap_graph(Ao,seed*100+fold*10+j)[0],100,j))
        for arm,AA,ep,rep in graph_specs:
            AAi=np.eye(Ai.shape[0]) if arm=='R5' else (Ai if arm=='R6' else swap_graph(Ai,seed*100+fold*10+rep)[0])
            _,gd=graph_fit(Xi,AAi,Pi,it,iv,inner_base_train,inner_base_val,t,e,seed+fold+rep,ep,base_all=None); # inner epoch probe
            bo,meta=graph_fit(Xo,AA,Po,tr,np.asarray([],dtype=int),base_all[tr],np.asarray([],dtype=float),t,e,seed+fold+rep,int(gd['selected_epoch']),base_all=base_all,fixed_epoch=int(gd['selected_epoch']))
            fname=arm if arm!='R7' else f'R7_rep{rep}'
            save(fname,bo[te],{'graph_replica':rep,'graph':arm,'graph_audit':go,'train_meta':meta,'inner_meta':gd,'base_identity_pass':bool(np.max(np.abs(base_te-np.load(out/'oof'/f'DSS_seed{seed}_fold{fold}_R0.npz')['base_test_risk']))<=1e-10)})
        # O3 is the same true-graph encoder with a zero clinical anchor.
        _,ogd=graph_fit(Xi,Ai,Pi,it,iv,np.zeros(len(it)),np.zeros(len(iv)),t,e,seed+fold+50,100,base_all=None)
        oz,_=graph_fit(Xo,Ao,Po,tr,np.asarray([],dtype=int),np.zeros(len(tr)),np.asarray([],dtype=float),t,e,seed+fold+50,int(ogd['selected_epoch']),base_all=np.zeros(len(t)),fixed_epoch=int(ogd['selected_epoch']))
        save('O3',oz[te],{'graph_replica':0,'graph':'true','inner_meta':ogd,'omics_only':True,'base_identity_pass':True})
        audit.append({'seed':seed,'fold':fold,'events_train':int(e[tr].sum()),'events_test':int(e[te].sum()),'base_test_hash':h(base_te),'r0_ridge_lambda':lam,'graph':go})
        print(seed,fold,flush=True)
    (out/'run_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8'); (out/'DONE_TRAIN.json').write_text(json.dumps({'status':'PASS','seeds':SEEDS,'folds':5,'arms':ARMS,'random_graph_replicas':5,'patient_files_private':True},indent=2),encoding='utf8')
if __name__=='__main__': run(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
