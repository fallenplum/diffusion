from pathlib import Path
import json, hashlib, numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phase8_v0.cox import km_left

SEEDS=[20261007,20261008,20261009]; ARMS=['B0','B1','B2','B3','B4','B5','B6','O1','O2','O3','O4','O5','O6']
def ps(t,e,r,w=None,m=None):
    t=np.asarray(t); e=np.asarray(e); r=np.asarray(r); n=len(t); w=np.ones(n) if w is None else np.asarray(w); m=np.ones(n) if m is None else np.asarray(m)
    ok=(e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0)))
    z=ok*(w*m)[:,None]*m[None,:]; den=z.sum(); cr=(r[:,None]>r[None,:]).astype(float)+.5*(r[:,None]==r[None,:])
    return float((z*cr).sum()/den) if den>0 else np.nan
def main(out, parent):
    out=Path(out); d=dict(np.load(Path(parent)/'private_dataset.npz',allow_pickle=True)); t=d['DSS_time']; e=d['DSS_event']; folds=d['folds']; risks={}
    foldrows=[]; seedrows=[]
    for s in SEEDS:
      risks[s]={}
      for a in ARMS:
        rr=np.full(len(t),np.nan)
        for f in range(1,6):
          z=np.load(out/'oof'/f'DSS_seed{s}_fold{f}_{a}.npz'); rr[z['test_indices']]=z['risk']
        risks[s][a]=rr; hs=[]; us=[]
        for f in range(1,6):
          ix=np.flatnonzero(folds==f); train=np.flatnonzero(folds!=f); g=km_left(t[train],e[train],t[ix]); active=(e[ix]==1)&(t[ix]<=60)&np.isfinite(g)&(g>0); ww=np.where(active,1/(g*g),0.0)
          h=ps(t[ix],e[ix],rr[ix]); u=ps(t[ix],e[ix],rr[ix],ww); hs.append(h); us.append(u); foldrows.append({'seed':s,'fold':f,'method':a,'harrell':h,'uno60':u,'uno_valid_events':int((ww>0).sum())})
        seedrows.append({'seed':s,'method':a,'harrell':ps(t,e,rr, m=None),'within_fold_harrell':np.nanmean(hs),'uno60':np.nanmean(us),'fold_harrell_std':np.nanstd(hs,ddof=1),'fold_uno60_std':np.nanstd(us,ddof=1),'uno_valid_events':int(sum(x['uno_valid_events'] for x in foldrows if x['seed']==s and x['method']==a))})
    pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False); sm=pd.DataFrame(seedrows); sm.to_csv(out/'seed_metrics_repaired.csv',index=False)
    # Prespecified secondary endpoint: evaluate the DSS-trained OOF risk on PFS
    # without refitting or using PFS labels for model selection.
    pt,pe=d['PFS_time'],d['PFS_event']; pv=d['PFS_valid'].astype(bool); pfsrows=[]
    for s in SEEDS:
      for a in ARMS:
        rr=risks[s][a]; hs=[]; us=[]
        for f in range(1,6):
          ix=np.flatnonzero((folds==f)&pv); train=np.flatnonzero((folds!=f)&pv); g=km_left(pt[train],pe[train],pt[ix]); active=(pe[ix]==1)&(pt[ix]<=60)&np.isfinite(g)&(g>0); ww=np.where(active,1/(g*g),0.0); hs.append(ps(pt[ix],pe[ix],rr[ix])); us.append(ps(pt[ix],pe[ix],rr[ix],ww))
        pfsrows.append({'seed':s,'method':a,'within_fold_harrell':np.nanmean(hs),'uno60':np.nanmean(us),'valid_patients':int(pv.sum()),'events':int(pe[pv].sum())})
    pd.DataFrame(pfsrows).to_csv(out/'pfs_seed_metrics.csv',index=False)
    rng=np.random.default_rng(20261011); pairs=[('B4','B0'),('B5','B4'),('B5','B6')]; rows=[]; B=1000
    for a,b in pairs:
      dh=[]; du=[]
      for rep in range(B):
        vh=[]; vu=[]
        for s in SEEDS:
          m=rng.multinomial(len(t),np.ones(len(t))/len(t)); ha=[]; ua=[]
          for f in range(1,6):
            ix=np.flatnonzero(folds==f); train=np.flatnonzero(folds!=f); g=km_left(t[train],e[train],t[ix]); active=(e[ix]==1)&(t[ix]<=60)&np.isfinite(g)&(g>0); ww=np.where(active,1/(g*g),0.0)
            ha.append(ps(t[ix],e[ix],risks[s][a][ix],m=m[ix])-ps(t[ix],e[ix],risks[s][b][ix],m=m[ix]))
            ua.append(ps(t[ix],e[ix],risks[s][a][ix],ww,m=m[ix])-ps(t[ix],e[ix],risks[s][b][ix],ww,m=m[ix]))
          vh.append(np.nanmean(ha)); vu.append(np.nanmean(ua))
        dh.append(np.nanmean(vh)); du.append(np.nanmean(vu))
      obs_h=float(sm[sm.method==a].within_fold_harrell.mean()-sm[sm.method==b].within_fold_harrell.mean()); obs_u=float(sm[sm.method==a].uno60.mean()-sm[sm.method==b].uno60.mean())
      rows.append({'comparison':f'{a}-{b}','delta_harrell':obs_h,'ci_harrell_low':float(np.nanquantile(dh,.0083333)),'ci_harrell_high':float(np.nanquantile(dh,.9916667)),'delta_uno60':obs_u,'ci_uno60_low':float(np.nanquantile(du,.0083333)),'ci_uno60_high':float(np.nanquantile(du,.9916667)),'bootstrap_reps':B,'ci_level':.9833334,'bootstrap_seed':20261011})
    comp=pd.DataFrame(rows); comp.to_csv(out/'paired_bootstrap.csv',index=False)
    g=sm.groupby('method').agg(harrell=('within_fold_harrell','mean'),uno60=('uno60','mean'),harrell_seed_std=('within_fold_harrell','std'),uno60_seed_std=('uno60','std')).reset_index(); g.to_csv(out/'summary_metrics.csv',index=False)
    decision='GO' if (comp.loc[comp.comparison=='B4-B0','delta_harrell'].iloc[0]>0 and comp.loc[comp.comparison=='B5-B4','delta_harrell'].iloc[0]>0 and comp.loc[comp.comparison=='B5-B6','delta_harrell'].iloc[0]>0) else 'NO_GO'
    (out/'decision.json').write_text(json.dumps({'status':'PASS','decision':decision,'primary_comparisons':['B4-B0','B5-B4','B5-B6'],'uno_repair':'invalid G(t) weights excluded; no patient IDs written','confidence':'high for metric repair, medium for biological decision due 65 events'},indent=2),encoding='utf8')
    lines=['# Phase9-G0 Pathway-Guided Omics Representation Benchmark','','固定 3 seeds × 5 patient-level folds；所有 KEGG 成员和 KGML 基因边在结果前固定，未使用 DSS 进行通路筛选。CNA 使用 gain/loss 两个独立通道。','', '|Arm|Harrell|Uno60|seed SD|','|---|---:|---:|---:|']
    for _,r in g.iterrows(): lines.append(f"|{r.method}|{r.harrell:.6f}|{r.uno60:.6f}|{r.harrell_seed_std:.6f}|")
    lines += ['', '## 预先固定配对比较', '', '|比较|Δ Harrell|98.33% CI|Δ Uno60|98.33% CI|','|---|---:|---|---:|---|']
    for _,r in comp.iterrows(): lines.append(f"|{r.comparison}|{r.delta_harrell:+.6f}|[{r.ci_harrell_low:+.6f},{r.ci_harrell_high:+.6f}]|{r.delta_uno60:+.6f}|[{r.ci_uno60_low:+.6f},{r.ci_uno60_high:+.6f}]|")
    pg=pd.DataFrame(pfsrows).groupby('method')[['within_fold_harrell','uno60']].mean()
    lines += ['', '## 预先锁定的 PFS 次要终点（DSS 训练风险直接评估，未重新选择模型）', '', '|Arm|PFS Harrell|PFS Uno60|','|---|---:|---:|']
    for a,r in pg.iterrows(): lines.append(f'|{a}|{r.within_fold_harrell:.6f}|{r.uno60:.6f}|')
    lines += ['',f'## 决策：**{decision}**','', 'B4 是无消息的 pathway pooling；B5 是一层归一化 KEGG 消息传递；B6 使用同节点和同度数的随机拓扑。该实现将图编码器限制为 3→8→1 的低容量可审计版本。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (out/'manifest.json').write_text(json.dumps({'status':'PASS','arms':ARMS,'seeds':SEEDS,'folds':5,'bootstrap':1000,'protected_data':'not published','kegg_source':'https://rest.kegg.jp'},indent=2),encoding='utf8')
if __name__=='__main__': main(__import__('sys').argv[1],__import__('sys').argv[2])
