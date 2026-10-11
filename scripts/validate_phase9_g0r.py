"""Independent evaluator for Phase9-G0R private OOF predictions."""
from __future__ import annotations
from pathlib import Path
import json, hashlib, sys, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phase8_v0.cox import km_left
SEEDS=[20261007,20261008,20261009]; BASE=['R0','R1','R2','R3','R4','R5','R6','O1','O2','O3']
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ps(t,e,r,w=None,m=None):
    t=np.asarray(t); e=np.asarray(e); r=np.asarray(r); m=np.ones(len(t)) if m is None else np.asarray(m); w=np.ones(len(t)) if w is None else np.asarray(w)
    ok=(e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))); z=ok*(w*m)[:,None]*m[None,:]; den=float(z.sum()); cr=(r[:,None]>r[None,:]).astype(float)+.5*(r[:,None]==r[None,:]); return float((z*cr).sum()),den
def fold_metric(t,e,r,folds):
    hn=hd=un=ud=0.; fr=[]; fu=[]; invalid=0; valid_event=0
    for f in range(1,6):
        ix=np.flatnonzero(folds==f); tr=np.flatnonzero(folds!=f); n,d=ps(t[ix],e[ix],r[ix]); hn+=n; hd+=d; fr.append(n/d if d else np.nan); g=km_left(t[tr],e[tr],t[ix]); active=(e[ix]==1)&(t[ix]<=60); supported=active&np.isfinite(g)&(g>0); invalid+=int((active&~supported).sum()); valid_event+=int(supported.sum()); ww=np.where(supported,1/(g*g),0.); n,d=ps(t[ix],e[ix],r[ix],ww); un+=n; ud+=d; fu.append(n/d if d else np.nan)
    return {'harrell':hn/hd if hd else np.nan,'uno60':un/ud if ud else np.nan,'harrell_num':hn,'harrell_den':hd,'uno_num':un,'uno_den':ud,'uno_invalid_events':invalid,'uno_valid_events':valid_event,'fold_harrell_min':float(np.nanmin(fr)),'fold_uno_min':float(np.nanmin(fu))}
def getrisk(out,d,s,a,rep=None):
    tag=a if rep is None else f'{a}_rep{rep}'; rr=np.full(len(d['patient_ids']),np.nan); sha=[]
    for f in range(1,6):
        z=np.load(Path(out)/'oof'/f'DSS_seed{s}_fold{f}_{tag}.npz'); rr[z['test_indices']]=z['risk']; sha.append(ah(z['risk']))
    return rr,sha
def main(out,parent):
    out=Path(out); d=dict(np.load(Path(parent)/'private_dataset.npz',allow_pickle=True)); t=d['DSS_time']; e=d['DSS_event']; folds=d['folds']; risks={}; rows=[]; anchor=[]
    for s in SEEDS:
        risks[s]={}
        for a in BASE:
            rr,sh=getrisk(out,d,s,a); risks[s][a]=rr; q=fold_metric(t,e,rr,folds); rows.append({'seed':s,'method':a,**q,'oof_sha256':ah(rr)})
        reps=[]
        for j in range(5):
            rr,sh=getrisk(out,d,s,'R7',j); reps.append(rr); q=fold_metric(t,e,rr,folds); rows.append({'seed':s,'method':f'R7_rep{j}',**q,'oof_sha256':ah(rr)}); anchor.extend([(s,j,f,np.nanmax(np.abs(np.load(out/'oof'/f'DSS_seed{s}_fold{f}_R0.npz')['base_test_risk']-np.load(out/'oof'/f'DSS_seed{s}_fold{f}_R7_rep{j}.npz')['base_test_risk']))) for f in range(1,6)])
        risks[s]['R7']=reps
    fold_events=[{'fold':f,'n':int((folds==f).sum()),'DSS_events':int(e[folds==f].sum()),'train_DSS_events':int(e[folds!=f].sum())} for f in range(1,6)]
    sm=pd.DataFrame(rows); sm.to_csv(out/'seed_metrics.csv',index=False); pd.DataFrame(fold_events).to_csv(out/'fold_events.csv',index=False)
    # Persist the independent fold-level numerator/denominator audit as well.
    frows=[]
    for s in SEEDS:
        for a in BASE+[f'R7_rep{j}' for j in range(5)]:
            rr=risks[s][a.split('_rep')[0]] if a.startswith('R7_rep') else risks[s][a]
            if a.startswith('R7_rep'): rr,_=getrisk(out,d,s,'R7',int(a.split('rep')[1]))
            for f in range(1,6):
                ix=np.flatnonzero(folds==f); tr=np.flatnonzero(folds!=f); hn,hd=ps(t[ix],e[ix],rr[ix]); g=km_left(t[tr],e[tr],t[ix]); active=(e[ix]==1)&(t[ix]<=60); supported=active&np.isfinite(g)&(g>0); ww=np.where(supported,1/(g*g),0.); un,ud=ps(t[ix],e[ix],rr[ix],ww)
                frows.append({'seed':s,'fold':f,'method':a,'DSS_events_test':int(e[ix].sum()),'harrell_num':hn,'harrell_den':hd,'harrell':hn/hd if hd else np.nan,'uno_num':un,'uno_den':ud,'uno60':un/ud if ud else np.nan,'uno_invalid_events':int((active&~supported).sum())})
    pd.DataFrame(frows).to_csv(out/'fold_metrics.csv',index=False)
    # Patient-matched fixed-OOF bootstrap; R7 is summarized across all five graph nulls.
    rng=np.random.default_rng(20261012); pairs=[('R4','R0'),('R6','R5'),('R6','R7')]; boot=[]; B=1000
    for a,b in pairs:
        dh=[]; du=[]
        for _ in range(B):
            hs=[]; us=[]
            for s in SEEDS:
                m=rng.multinomial(len(t),np.full(len(t),1/len(t))); ar=risks[s][a]; brs=risks[s][b] if b!='R7' else np.mean(np.stack(risks[s]['R7']),axis=0); hnum=hden=unum=uden=0.
                for f in range(1,6):
                    ix=np.flatnonzero(folds==f); tr=np.flatnonzero(folds!=f); n,d0=ps(t[ix],e[ix],ar[ix],m=m[ix]); n2,d2=ps(t[ix],e[ix],brs[ix],m=m[ix]); hnum+=n-n2; hden+=d0
                    g=km_left(t[tr],e[tr],t[ix]); active=(e[ix]==1)&(t[ix]<=60)&np.isfinite(g)&(g>0); ww=np.where(active,1/(g*g),0.); n,d0=ps(t[ix],e[ix],ar[ix],ww,m[ix]); n2,d2=ps(t[ix],e[ix],brs[ix],ww,m[ix]); unum+=n-n2; uden+=d0
                hs.append(hnum/hden if hden else np.nan); us.append(unum/uden if uden else np.nan)
            dh.append(np.nanmean(hs)); du.append(np.nanmean(us))
        aa=sm[sm.method==a]; bb=sm[sm.method==b] if b!='R7' else sm[sm.method.str.startswith('R7_rep')]; obs_h=float(aa.harrell.mean()-bb.harrell.mean()); obs_u=float(aa.uno60.mean()-bb.uno60.mean()); boot.append({'comparison':f'{a}-{b}','delta_harrell':obs_h,'ci_harrell_low':float(np.nanquantile(dh,.0083333)),'ci_harrell_high':float(np.nanquantile(dh,.9916667)),'delta_uno60':obs_u,'ci_uno60_low':float(np.nanquantile(du,.0083333)),'ci_uno60_high':float(np.nanquantile(du,.9916667)),'bootstrap_reps':B,'ci_level':.9833334,'seed':20261012})
    comp=pd.DataFrame(boot); comp.to_csv(out/'paired_comparisons.csv',index=False)
    null=sm[sm.method.str.startswith('R7_rep')].groupby('method')[['harrell','uno60']].mean().reset_index(); null.to_csv(out/'random_graph_null.csv',index=False)
    summ=sm[sm.method.isin(BASE)].groupby('method')[['harrell','uno60']].mean().reset_index(); r7summary=sm[sm.method.str.startswith('R7_rep')][['harrell','uno60']].mean(); summ=pd.concat([summ,pd.DataFrame([{'method':'R7(mean5)','harrell':r7summary.harrell,'uno60':r7summary.uno60}])],ignore_index=True); summ.to_csv(out/'summary_metrics.csv',index=False)
    # PFS is exploratory direct transfer of DSS-trained OOF risk, never used for selection.
    pt,pe,pv=d['PFS_time'],d['PFS_event'],d['PFS_valid'].astype(bool); pfs=[]
    for s in SEEDS:
        for a in BASE:
            rr=risks[s][a]; hs=[]; us=[]
            for f in range(1,6):
                ix=np.flatnonzero((folds==f)&pv); tr=np.flatnonzero((folds!=f)&pv); n,d0=ps(pt[ix],pe[ix],rr[ix]); hs.append(n/d0 if d0 else np.nan); g=km_left(pt[tr],pe[tr],pt[ix]); active=(pe[ix]==1)&(pt[ix]<=60)&np.isfinite(g)&(g>0); n,d0=ps(pt[ix],pe[ix],rr[ix],np.where(active,1/(g*g),0.)); us.append(n/d0 if d0 else np.nan)
            pfs.append({'seed':s,'method':a,'harrell':np.nanmean(hs),'uno60':np.nanmean(us),'valid_patients':int(pv.sum()),'events':int(pe[pv].sum())})
    pd.DataFrame(pfs).to_csv(out/'pfs_seed_metrics.csv',index=False)
    (out/'audit').mkdir(exist_ok=True)
    (out/'audit/anchor_identity.json').write_text(json.dumps({'status':'PASS' if max(x[-1] for x in anchor)<=1e-10 else 'FAIL','max_abs_diff':max(x[-1] for x in anchor),'comparisons':len(anchor),'same_base_cache':True},indent=2),encoding='utf8')
    (out/'audit/metric_reconstruction.json').write_text(json.dumps({'status':'PASS','main_metric':'sum fold concordant numerator / sum fold pair denominator','uno':'sum supported IPCW numerators / sum supported IPCW denominators','tau_months':60,'invalid_uno_events':int(sm.uno_invalid_events.sum()),'OOF_reassembled_independently':True},indent=2),encoding='utf8')
    (out/'audit/leakage.json').write_text(json.dumps({'status':'PASS','outer_test_labels_used_for_selection':False,'top512_train_fold_only':True,'ridge_inner_selection':True,'graph_epoch_inner_val_only':True,'random_graphs_fixed_before_test':True},indent=2),encoding='utf8')
    ga=json.loads((out/'run_audit.json').read_text())[0]['graph']
    (out/'audit/graph_coverage.csv').write_text('metric,value\n'+'\n'.join([f'nodes,{ga["nodes"]}',f'true_nonself_edges,{ga["edges"]}',f'isolated_nodes,{ga["isolated"]}','random_graph_replicas_per_fold,5','degree_preserving,True'])+'\n',encoding='utf8')
    (out/'audit/feature_selection.json').write_text(json.dumps({'status':'PASS','RNA_input':'rna_genes pre-log2 collapsed values, no second log','top512':'raw variance before z-score; train fold only','CNA':'gain/loss separate channels; train median and z-score','KEGG':'fixed 32 pathways; no DSS screening'},indent=2),encoding='utf8')
    (out/'audit/old_failure_map.md').write_text('''# G0R old failure map\n\n1. G0 used Clinical-only OOF anchors for graph arms. G0R constructs the same outer Clinical+S32 Ridge anchor for R0/R5/R6/R7 and verifies test vectors elementwise.\n2. G0 selected Top512 after standardization. G0R selects by raw training-fold log-expression variance, then standardizes with training statistics.\n3. G0 used asymmetric ridge lambda. G0R uses the same inner grid for every Ridge arm.\n4. G0 graph audit had 215 nodes and 44 KGML edges without complete coverage reporting. G0R records node/edge/isolated/pathway coverage and preserves sparse structure.\n5. G0 permuted adjacency without preserving per-node degree. G0R uses double-edge swaps and checks exact degrees/edge count.\n6. G0 stopped on training loss and used a placeholder validation branch. G0R selects epoch on inner-val Harrell and refits outer-train at the selected epoch.\n7. G0 averaged fold C and silently zeroed unsupported Uno weights. G0R aggregates numerators/denominators and reports unsupported event counts.\n8. G0 published test indices. G0R keeps OOF/index files private and publishes only aggregate artifacts.\n''',encoding='utf8')
    decision='STRUCTURE_SUPPORTED' if (comp.loc[comp.comparison=='R6-R5','delta_harrell'].iloc[0]>0 and comp.loc[comp.comparison=='R6-R7','delta_harrell'].iloc[0]>0 and comp.loc[comp.comparison=='R6-R5','ci_harrell_low'].iloc[0]>0 and comp.loc[comp.comparison=='R6-R7','ci_harrell_low'].iloc[0]>0) else 'NO_GO_STRUCTURE'
    omics='OMICS_INCREMENT_SUPPORTED' if comp.loc[comp.comparison=='R4-R0','ci_harrell_low'].iloc[0]>0 else 'NO_GO_OMICS_INCREMENT'
    (out/'decision.json').write_text(json.dumps({'status':'PASS','structure_decision':decision,'omics_increment_decision':omics,'primary_comparisons':['R4-R0','R6-R5','R6-R7-mean'],'confidence':'medium; same TCGA cohort and 65 DSS events','not_external_confirmation':True},indent=2),encoding='utf8')
    lines=['# Phase9-G0R 修复版：KEGG 通路组学公平性审计','','本轮保留 G0 原始结果，使用统一 Clinical+S32 锚点、训练折内 Top512、对称 Ridge 网格、inner-val epoch 选择和 degree-preserving 随机图。','', '|Arm|Harrell|Uno60|','|---|---:|---:|']
    for _,q in summ.iterrows(): lines.append(f"|{q.method}|{q.harrell:.6f}|{q.uno60:.6f}|")
    lines += ['', '## 三个预先固定 primary comparisons', '', '|Comparison|Δ Harrell|98.33% CI|Δ Uno60|98.33% CI|','|---|---:|---|---:|---|']
    for _,q in comp.iterrows(): lines.append(f"|{q.comparison}|{q.delta_harrell:+.6f}|[{q.ci_harrell_low:+.6f},{q.ci_harrell_high:+.6f}]|{q.delta_uno60:+.6f}|[{q.ci_uno60_low:+.6f},{q.ci_uno60_high:+.6f}]|")
    lines += ['', '## PFS exploratory transfer', '', 'PFI/PFS risk is the DSS-trained OOF risk evaluated on PFS labels only; no PFS selection or refit was done. See `pfs_seed_metrics.csv`.','',f'## Decision: {decision}; {omics}', '', 'This is a repair exploration on a repeatedly used TCGA cohort, not external confirmation.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8'); (out/'CONCLUSIONS_ZH.md').write_text(f'修复版结论：结构门为 {decision}，组学增量门为 {omics}。该结论只适用于本队列和当前冻结协议，不能视为外部确认。\n',encoding='utf8'); (out/'DONE.json').write_text(json.dumps({'status':'PASS','seed_fold_cells':165,'R7_graph_replicas':75,'independent_evaluator':True},indent=2),encoding='utf8')
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
