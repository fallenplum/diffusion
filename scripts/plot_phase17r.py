import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.phase17r_ssd.data import R
F=R/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.linewidth':.7,'savefig.facecolor':'white'})
def save(fig,name):
    svg=F/(name+'.svg');fig.savefig(svg,bbox_inches='tight');svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n');fig.savefig(F/(name+'.png'),dpi=240,bbox_inches='tight');plt.close(fig)
def main():
    contract={'backend':'Python/matplotlib','exports':['editable SVG','240dpi PNG'],'figures':{'01_mechanism':'compare context-to-state response,isolated SSM gradients,and branch risk scale;none alone proves predictive value','02_primary_Harrell':'show all available preregistered Harrell differences and paired uncertainty without selecting positive contrasts','03_training':'fixed seed31fold1 R1 illustration separates training loss,validation loss,and context gradients'},'sources':['INNER_MECHANISM.csv','PAIRED_COMPARISONS.csv','TRAINING_CURVES.csv'],'risks':['overlapping inner partitions','fixed-model rather than retraining bootstrap','no histology ground truth','zeros floored solely for log plotting']}
    (F/'FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    q=pd.read_csv(R/'INNER_MECHANISM.csv');arms=list(dict.fromkeys(q.arm));labels=[a.replace('R1_','').replace('R2_','N_').replace('_noDirect','-ND') for a in arms]
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.0),layout='constrained')
    for ax,key,label in zip(axes,['h_relative_L2','isolated_SSM_g_gradient_L2','ssm_over_anchor_SD'],['Matched/wrong final-state relative L2','Isolated SSM context gradient L2','SSM risk SD / anchor risk SD']):
        for i,a in enumerate(arms):
            v=q.loc[q.arm==a,key].to_numpy();ax.scatter(np.full(len(v),i),np.maximum(v,1e-12),s=10,alpha=.55,color='#7884B4');ax.plot(i,max(np.median(v),1e-12),'_',ms=12,color='#484878')
        ax.set(yscale='log',xticks=range(len(arms)),xticklabels=labels,ylabel=label);ax.tick_params(axis='x',rotation=55)
    axes[0].axhline(.01,color='#B45E80',ls='--',lw=.8);fig.suptitle('Selected inner checkpoints: training-only mechanism probes (15 cells/arm)');save(fig,'01_mechanism')
    ci=pd.read_csv(R/'PAIRED_COMPARISONS.csv');ci=ci[(ci.family=='primary')&(ci.metric=='Harrell')].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.6,max(2.6,len(ci)*.42)),layout='constrained')
    for i,r in ci.iterrows():
        ax.plot([r.Bonferroni_low,r.Bonferroni_high],[i,i],color='#B4C0E4',lw=6);ax.plot([r.CI95_low,r.CI95_high],[i,i],color='#484878',lw=1.5);ax.plot(r.delta,i,'o',color='#B45E80',ms=4)
    ax.axvline(0,color='#606060',ls='--',lw=.8);ax.set(yticks=range(len(ci)),yticklabels=ci.comparison,xlabel='Harrell difference (thin:95%; thick:Bonferroni99.5833%)');ax.invert_yaxis();ax.set_title('Outer DSS: 20,000 within-fold paired patient bootstrap draws');save(fig,'02_primary_Harrell')
    curves=pd.read_csv(R/'TRAINING_CURVES.csv');q=curves[(curves.stage=='inner')&(curves.seed==31)&(curves.fold==1)]
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),layout='constrained')
    palette=['#484878','#B45E80','#3F8D89','#A08548']
    for color,arm in zip(palette,['R1_A8_full','R1_A8_noDirect','R1_A3_noDirect','R1_S6_noDirect']):
        z=q[q.arm==arm]
        for ax,key in zip(axes,['train_loss','validation_loss_EMA','g_attention_grad']):
            values=np.maximum(z[key],1e-12) if key=='g_attention_grad' else z[key]
            ax.plot(z.epoch,values,label=arm.replace('R1_',''),color=color,lw=1.1)
    for ax,key in zip(axes,['Full-risk train Efron','Inner-validation Efron EMA','Efron context gradient (floor1e-12)']):ax.set(xlabel='Epoch',ylabel=key)
    axes[2].set_yscale('log');axes[0].legend(fontsize=6);fig.suptitle('Fixed illustration: inner fold1 / seed31; selection uses loss, not displayed rank');save(fig,'03_training')
    (F/'CAPTIONS.md').write_text('# Figure scope\n\nPython/matplotlib; editable SVG and 240dpi PNG. No histological maps. Mechanism points reuse patients across folds and are not independent cohorts. Zero gradients plotted at1e-12 only for log visibility; source CSV keeps exact zero. Training illustration was fixed before outcome inspection. Uncertainty is paired patient bootstrap conditional on frozen fits, not external or retraining uncertainty.\n')
    print('FIGURES_CREATED',len(list(F.glob('*.png'))),flush=True)
if __name__=='__main__':main()
