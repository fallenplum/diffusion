"""Standalone auditable scientific figures; Python backend, no synthetic data."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=ROOT/'reports/phase17_ssd_fusion';F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})
COL={'A2':'#A8A8A8','A3':'#484878','A7':'#7884B4','A8':'#B45E80'}

def save(fig,name):
    svg=F/(name+'.svg');fig.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(F/(name+'.png'),dpi=240,bbox_inches='tight');plt.close(fig)

def main():
    contract={'backend':'Python/matplotlib only','evidence':'actual saved inner curves,valid patch gates,training-only context perturbations,heldout frozen risk interventions','exports':['editable-text SVG','PNG240dpi'],'risk':'repeated cohort;seeds/folds not independent patients;no raw histology;no disease localization claims','figures':{'01_training':'training improvement does not imply validation improvement;preselected seed31fold1 illustration','02_gates':'write gate remains near high-retain initialization','03_propagation':'context changes are attenuated before final risk','04_state_gradient':'final-state sensitivity is concentrated late in coordinate sequence','05_order':'order perturbations change risks;no permutation invariance claim','06_primary_intervals':'primary corrected intervals do not establish positive increment','07_compute':'SSM branch has measured computational cost;FLOPs only supported profiler ops'}}
    (F/'FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    curves=pd.read_csv(R/'TRAINING_CURVES.csv');fig,axes=plt.subplots(1,3,figsize=(9,2.7),layout='constrained')
    for arm in ['A3','A7','A8']:
        q=curves[(curves.stage=='inner')&(curves.arm==arm)&(curves.seed==31)&(curves.fold==1)]
        for ax,col,label in zip(axes,['train_loss','validation_loss','validation_Harrell'],['Train Efron loss','Inner-validation Efron loss','Inner-validation Harrell']):ax.plot(q.epoch,q[col],label=arm,color=COL[arm],lw=1.3);ax.set(xlabel='Epoch',ylabel=label)
    axes[0].legend();fig.suptitle('Training diagnosis: seed 31, fold 1 (fixed illustration)');save(fig,'01_training')
    h=pd.read_csv(R/'GATE_HISTOGRAM.csv');cx=pd.read_csv(R/'CONTEXT_STATE_AUDIT.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained');agg=h.groupby('bin_left').tokens.sum();axes[0].bar(agg.index,agg.values/agg.sum(),width=.0095,align='edge',color=COL['A8']);axes[0].set(xlim=(.85,.96),xlabel='Valid patch write gate m',ylabel='Token fraction');axes[0].axvline(.9,color='#606060',ls='--',lw=.8)
    axes[1].errorbar(np.arange(len(cx)),cx.gate_valid_mean,yerr=cx.gate_valid_std,fmt='o',ms=3,color=COL['A8'],capsize=2);axes[1].axhline(.9,color='#606060',ls='--',lw=.8);axes[1].set(xlabel='Seed/fold checkpoint (15)',ylabel='Mean gate ± token SD');fig.suptitle('A8 gates: first 32 training patients per checkpoint');save(fig,'02_gates')
    cols=['g_matched_wrong_relative_L2','dt_matched_wrong_relative_L2','B_matched_wrong_relative_L2','h_matched_wrong_relative_L2'];fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained');axes[0].boxplot([cx[k] for k in cols],tick_labels=['g','Delta','B','Final h'],patch_artist=True,boxprops={'facecolor':'#B4C0E4'},medianprops={'color':'#484878'});axes[0].set(yscale='log',ylabel='Matched–wrong relative L2')
    axes[1].scatter(np.arange(len(cx)),cx.matched_wrong_risk_mean_abs,color=COL['A8'],s=15);axes[1].set(yscale='log',xlabel='Seed/fold checkpoint (15)',ylabel='Mean absolute risk difference');fig.suptitle('Context perturbation is attenuated in the learned operator');save(fig,'03_propagation')
    grad=pd.read_csv(R/'LONG_RANGE_STATE_GRADIENT.csv');fig,ax=plt.subplots(figsize=(6.1,2.8),layout='constrained')
    for _,g in grad.groupby(['seed','fold']):ax.plot(g.position_decile,g.gradient_fraction,color='#B4C0E4',lw=.7,alpha=.7)
    ax.plot(grad.groupby('position_decile').gradient_fraction.mean(),color='#484878',lw=1.8,label='Mean of 15 checkpoint probes');ax.set(xlabel='Position decile in coordinate order',ylabel='Final-state input gradient fraction',xticks=range(1,11));ax.legend(loc='upper left');ax.set_title('One label-blind training WSI per checkpoint; no biological time');save(fig,'04_state_gradient')
    order=pd.read_csv(R/'ORDER_SENSITIVITY.csv');arms=['A2','A3','A4','A5','A6','A7','A8','A9','A10'];fig,axes=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    for shift,tag,color in [(-.12,'reverse','#484878'),(.12,'random','#B45E80')]:
        q=order[order.ordering==tag].groupby('arm').mean(numeric_only=True).reindex(arms);axes[0].scatter(np.arange(len(arms))+shift,q.mean_absolute_risk_drift,s=15,label=tag,color=color);axes[1].scatter(np.arange(len(arms))+shift,q.risk_Spearman,s=15,color=color)
    for ax in axes:ax.set(xticks=range(len(arms)),xticklabels=arms,xlabel='Frozen arm')
    axes[0].set(ylabel='Mean absolute risk drift',yscale='log');axes[1].set(ylabel='Risk Spearman correlation');axes[0].legend();fig.suptitle('Heldout risk changes under within-WSI reordering');save(fig,'05_order')
    ci=pd.read_csv(R/'PAIRED_COMPARISONS.csv');q=ci[ci.family=='primary'].reset_index(drop=True);fig,ax=plt.subplots(figsize=(6.6,2.8),layout='constrained')
    for i,row in q.iterrows():
        ax.plot([row.Bonferroni_low,row.Bonferroni_high],[i,i],color='#B4C0E4',lw=6,solid_capstyle='butt');ax.plot([row.CI95_low,row.CI95_high],[i,i],color='#484878',lw=1.4);ax.plot(row.delta,i,'o',ms=4,color='#B45E80')
    ax.axvline(0,color='#606060',lw=.8,ls='--');ax.set(yticks=range(len(q)),yticklabels=[r.comparison+' / '+r.metric for r in q.itertuples()],xlabel='Paired difference (thin: 95%; thick: Bonferroni 98.75%)');ax.invert_yaxis();ax.set_title('20,000 within-fold patient bootstrap draws; shared across seeds');save(fig,'06_primary_intervals')
    cost=pd.read_csv(R/'COMPUTE_ABLATION.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
    for i,mode in enumerate(['full','SSM_actual_compute_removed']):
        q=cost[cost['mode']==mode];axes[0].scatter(np.full(len(q),i),q.seconds_per_forward*1000,color=['#7884B4','#B45E80'][i],s=14);axes[1].bar(i,q.profiled_matmul_conv_FLOPs_lower_bound.mean()/1e9,color=['#7884B4','#B45E80'][i])
    for ax in axes:ax.set(xticks=[0,1],xticklabels=['Full A8','Recurrent branch removed'])
    axes[0].set(ylabel='Forward time per 32 patients (ms)',yscale='log');axes[1].set(ylabel='Profiled matmul/conv GFLOPs');fig.suptitle('Actual inference removal; profiler excludes scan elementwise FLOPs');save(fig,'07_compute')
    (F/'CAPTIONS.md').write_text('# Figure notes\n\nAll figures derive from committed aggregate CSVs. Training curves use the fixed seed31/fold1 illustration. Gate and context diagnostics use the first32 training rows, independent of outcomes. State input gradients use one training WSI per checkpoint and target final-state squared norm, not survival risk. Seeds/folds reuse patients and are not independent cohorts. Reordering uses no outcome labels. Primary uncertainty is patient bootstrap; it does not capture external population shift or all training uncertainty. No figure is histological ground truth.\n')
    print('FIGURES_COMPLETE',len(list(F.glob('*.svg'))),flush=True)

if __name__=='__main__':main()
