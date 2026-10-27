"""Plain research diagnostics from anonymous BG-BEN aggregate CSVs only.

No private model, patient output or raw image is read. Missing or unfinished
comparisons are skipped and recorded; no placeholder scientific values appear.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white'})
COLORS=('#1764ab','#167c58','#ce6533','#7855a4')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    try:
        value=float(value)
        return value if np.isfinite(value) else None
    except (TypeError,ValueError):return None


def read_rows(path):
    if not path.is_file():return []
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


class Renderer:
    def __init__(self,report,output):
        self.report,self.output=Path(report),Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.plots=[];self.skipped=[]

    def save(self,figure,name,sources,rows,scope):
        files=[]
        for extension in ('svg','png'):
            path=self.output/(name+'.'+extension)
            metadata={'Creator':'BG-BEN anonymous aggregate diagnostics'}
            if extension=='svg':metadata['Date']=None
            figure.savefig(path,dpi=150,bbox_inches='tight',metadata=metadata)
            files.append(dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size))
        plt.close(figure)
        self.plots.append(dict(name=name,files=files,source_files=[dict(file=path.name,sha256=sha256(path)) for path in sources],
                               rows_drawn=int(rows),scientific_scope=scope))

    def skip(self,name,reason):self.skipped.append(dict(name=name,status='SKIPPED_NO_COMPLETE_DATA',reason=reason))

    def performance(self,stage,filename,baseline):
        path=self.report/filename
        rows=[row for row in read_rows(path) if row.get('status')=='COMPLETE'
              and any(number(row.get(metric)) is not None for metric in ('Harrell','Uno60'))]
        name=stage+'_PERFORMANCE_COST'
        if not rows:self.skip(name,'No completed arm summary');return
        fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
        costs=[('parameters_mean','Registered model parameters'),
               ('mean_unit_elapsed_seconds','Mean unit wall time (s)'),
               ('peak_GPU_bytes_max','Peak GPU allocated (MiB)')]
        for line,metric in enumerate(('Harrell','Uno60')):
            for col,(field,label) in enumerate(costs):
                ax=axes[line,col];points=[]
                for row in rows:
                    x,y=number(row.get(field)),number(row.get(metric))
                    if x is None or y is None:continue
                    if field=='peak_GPU_bytes_max':x/=1024**2
                    points.append((x,y,row['arm']))
                if not points:
                    ax.text(.5,.5,'No complete cost/metric pair',ha='center',va='center',transform=ax.transAxes)
                else:
                    for x,y,arm in points:
                        ax.scatter(x,y,s=27,color=COLORS[line],edgecolor='white',linewidth=.4,zorder=3)
                        ax.annotate(arm,(x,y),xytext=(3,3),textcoords='offset points',fontsize=7)
                    xs=[item[0] for item in points]
                    if min(xs)>0 and max(xs)/min(xs)>20:ax.set_xscale('log')
                    for row in rows:
                        value=number(row.get(metric))
                        if row['arm']==baseline and value is not None:
                            ax.axhline(value,color='.45',lw=.8,ls='--',label=baseline+' reference')
                ax.set_xlabel(label);ax.set_ylabel(metric);ax.grid(alpha=.18)
        fig.suptitle(stage+': completed-arm performance and measured cost\nNeural parameter counts exclude shared frozen anchors',fontsize=11)
        self.save(fig,name,[path],len(rows),'Inner development summaries; no independent cohorts or uncertainty intervals')

    def retention(self):
        path=self.report/'R2_RNA_RETENTION.csv';name='R2_RNA_RETENTION'
        rows=[row for row in read_rows(path) if row.get('status')=='COMPLETE']
        groups={}
        for row in rows:
            key=(row.get('arm',''),row.get('representation',''))
            group=groups.setdefault(key,{'Pearson':[],'R2':[]})
            for metric,column in (('Pearson','decoder_Pearson'),('R2','decoder_R2')):
                value=number(row.get(column))
                if value is not None:group[metric].append(value)
        keys=[key for key,value in groups.items() if value['Pearson'] or value['R2']]
        if not keys:self.skip(name,'No completed held-out decoder diagnostics');return
        fig,axes=plt.subplots(1,2,figsize=(max(10,len(keys)*.48),4.8),constrained_layout=True)
        labels=[arm if representation==arm else arm+'\n'+representation for arm,representation in keys]
        for ax,metric,color in zip(axes,('Pearson','R2'),COLORS):
            x=[index for index,key in enumerate(keys) if groups[key][metric]]
            y=[np.mean(groups[key][metric]) for key in keys if groups[key][metric]]
            ax.bar(x,y,color=color,width=.72);ax.axhline(0,color='.4',lw=.7)
            ax.set_xticks(range(len(keys)),labels,rotation=60,ha='right',fontsize=7)
            ax.set_ylabel('Mean held-out decoder '+metric);ax.grid(axis='y',alpha=.18)
        fig.suptitle('RNA reference-risk decodability: completed representations\nDescriptive means of fit-local decoder diagnostics',fontsize=11)
        self.save(fig,name,[path],len(rows),'Decoder fit uses training rows; folds/seeds are not independent cohorts')

    def mechanism(self):
        path=self.report/'R3_MECHANISM_ABLATIONS.csv';name='R3_MECHANISM_DELTAS'
        rows=[row for row in read_rows(path) if row.get('status')=='COMPLETE'
              and any(number(row.get(key)) is not None for key in ('paired_delta_H','paired_delta_U60'))]
        if not rows:self.skip(name,'No completed paired mechanism comparison');return
        fig,axes=plt.subplots(1,2,figsize=(12,max(4,len(rows)*.27)),constrained_layout=True)
        for ax,field,title,color in zip(axes,('paired_delta_H','paired_delta_U60'),('Delta Harrell','Delta Uno60'),COLORS):
            positions=[index for index,row in enumerate(rows) if number(row.get(field)) is not None]
            values=[number(row[field]) for row in rows if number(row.get(field)) is not None]
            ax.barh(positions,values,color=color,height=.7);ax.axvline(0,color='.25',lw=.8)
            ax.set_yticks(range(len(rows)),[row['comparison'] for row in rows],fontsize=8)
            ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.18)
            # Keep small delta labels readable without changing any bar value.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        fig.suptitle('Completed real-retraining mechanism contrasts\nEach bar keeps its named reference; contrasts are not accumulated',fontsize=11)
        self.save(fig,name,[path],len(rows),'Paired inner comparisons with the original denominator aggregation; no p-values')

    def repair(self):
        path=self.report/'R3_PAIRING_CONTROLS.csv';name='R3_PAIR_REPAIR_DAMAGE'
        rows=[row for row in read_rows(path) if row.get('direct_replay_status')=='COMPLETE'
              and all(number(row.get(key)) is not None for key in ('full_control_repair','full_control_damage','full_control_Uno_repair','full_control_Uno_damage'))]
        if not rows:self.skip(name,'No complete saved-risk pairwise replay; receipt C-index deltas cannot reconstruct repair/damage');return
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,len(rows)*.26)),constrained_layout=True)
        for ax,prefix,title in zip(axes,('','Uno_'),('Harrell pairs','IPCW-weighted Uno pairs')):
            repaired=np.array([number(row['full_control_'+prefix+'repair']) for row in rows])
            damaged=np.array([number(row['full_control_'+prefix+'damage']) for row in rows])
            at=np.arange(len(rows));ax.barh(at,repaired,color=COLORS[0],label='Repaired (+)',height=.7)
            ax.barh(at,-damaged,color=COLORS[2],label='Damaged (-)',height=.7)
            ax.scatter(repaired-damaged,at,s=15,color=COLORS[1],label='Net',zorder=3)
            ax.axvline(0,color='.25',lw=.8);ax.set_yticks(at,[row['comparison'] for row in rows],fontsize=7)
            ax.invert_yaxis();ax.set_xlabel(title+' fraction');ax.grid(axis='x',alpha=.18);ax.legend(fontsize=7)
        fig.suptitle('Trained full model versus frozen pairing intervention\nDependence of saved predictions; not a retraining-necessity test',fontsize=11)
        self.save(fig,name,[path],len(rows),'Exact direct saved-risk repair/damage, with each frozen control kept distinct')

    def risk_shift(self):
        path=self.report/'R3_MODEL_MATRIX.csv';name='R3_RISK_SHIFT_REPAIR_DAMAGE'
        rows=[row for row in read_rows(path) if row.get('status')=='COMPLETE' and row.get('arm') not in ('F0','F1')
              and number(row.get('delta_risk_sd_ratio_mean')) is not None
              and all(number(row.get(key)) is not None for key in ('repair','damage','Uno_repair','Uno_damage'))]
        if not rows:self.skip(name,'No complete common-F1 risk-shift and pair-repair diagnostics');return
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
        for ax,prefix,title,color in zip(axes,('','Uno_'),('Harrell','Uno60'),COLORS):
            for row in rows:
                x=number(row['delta_risk_sd_ratio_mean']);y=number(row[prefix+'repair'])-number(row[prefix+'damage'])
                ax.scatter(x,y,s=30,color=color);ax.annotate(row['arm'],(x,y),xytext=(3,3),textcoords='offset points',fontsize=8)
            ax.axhline(0,color='.3',lw=.8);ax.grid(alpha=.18)
            ax.set_xlabel('Residual risk SD / common F1 reference SD')
            ax.set_ylabel(title+' net repaired pair fraction')
        fig.suptitle('Residual magnitude and pair-order changes relative to common F1',fontsize=11)
        self.save(fig,name,[path],len(rows),'Same common-F1 pair reference; magnitude alone does not establish useful interaction')

    def throughput(self):
        path=self.report/'PATCH_THROUGHPUT_AUDIT.csv';name='PATCH_THROUGHPUT'
        rows=[];groups={}
        for row in read_rows(path):
            # SUPPORTED is emitted only after an actual COMPLETE receipt and
            # frozen forward contract have passed the independent audit.
            if (row.get('role')!='PRIMARY' or row.get('category')!='NEURAL'
                or row.get('throughput_status')!='SUPPORTED_CONTRACT_DERIVED_LOGICAL_VALID_NODES'):
                continue
            rate=number(row.get('logical_valid_node_admissions_per_second'))
            admissions=number(row.get('logical_valid_node_admissions'))
            budget,dim=number(row.get('budget_per_patient')),number(row.get('visual_feature_dim'))
            if (rate is None or rate<=0 or admissions is None or admissions<=0
                or budget is None or budget<=0 or dim is None or dim<=0
                or not all(row.get(key) for key in ('root','stage','arm','cache_selector'))):continue
            stage='R4' if row['root']=='outer' or row['stage'].startswith('R4') else row['stage']
            if stage not in ('R1','R3','R4'):continue
            key=(row['root'],row['stage'],row['arm'],int(budget),int(dim),row['cache_selector'])
            groups.setdefault(stage,{}).setdefault(key,[]).append(rate/1000.)
            rows.append(row)
        stages=[stage for stage in ('R1','R3','R4') if groups.get(stage)]
        if not stages:self.skip(name,'No actual completed PRIMARY positive visual admissions with audited READY metadata');return
        height=max(4.,max(len(groups[stage]) for stage in stages)*.30+2.2)
        fig,axes=plt.subplots(1,len(stages),figsize=(6.4*len(stages),height),constrained_layout=True,squeeze=False)
        for ax,stage,color in zip(axes[0],stages,COLORS):
            labels=[]
            for at,(key,values) in enumerate(sorted(groups[stage].items())):
                root,raw_stage,arm,budget,dim,selector=key
                offsets=np.linspace(-.11,.11,len(values)) if len(values)>1 else np.zeros(1)
                ax.scatter(values,at+offsets,s=17,color=color,alpha=.70,zorder=3,
                           label='Completed unit' if at==0 else None)
                median=float(np.median(values));ax.plot([median,median],[at-.20,at+.20],
                    color='.12',lw=1.8,zorder=4,label='Descriptive median' if at==0 else None)
                labels.append(f'{root}/{raw_stage}/{arm} | B{budget}, D{dim}, {selector} (n={len(values)})')
            ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.invert_yaxis()
            ax.set_xlabel('Logical valid-node admissions / unit wall second (×10³)')
            ax.set_title(stage+': actual completed visual units');ax.grid(axis='x',alpha=.18)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.legend(fontsize=7,loc='best')
        fig.suptitle('Measured-unit logical patch admissions\nRepeated valid nodes; not unique tissue coverage or hardware kernel throughput',fontsize=11)
        self.save(fig,name,[path],len(rows),'PRIMARY actual COMPLETE visual units only; group by root/stage/arm/READY budget/dimension/selector. Points are units and black marks descriptive medians, not confidence intervals; whole-unit wall timers include driver overhead and concurrent resource contention.')

    def run(self):
        for stage,filename,baseline in (('R1','R1_ARCH_ABLATION.csv','P0'),('R2','R2_RESULTS.csv','O0'),('R3','R3_MODEL_MATRIX.csv','F1')):
            self.performance(stage,filename,baseline)
        self.retention();self.mechanism();self.repair();self.risk_shift();self.throughput()
        audit=self.report/'INNER_REPORT_AUDIT.json'
        state=json.loads(audit.read_text()).get('status','UNKNOWN') if audit.is_file() else 'UNKNOWN'
        result=dict(schema='bgb_en_anonymous_plot_manifest_v1',status='COMPLETE_AVAILABLE_FIGURES_ONLY',
            source_inner_report_state=state,scientific_completion_inferred=False,
            private_assets_read=False,plots=self.plots,skipped=self.skipped,
            plotting_source_sha256=sha256(Path(__file__)))
        (self.output/'PLOT_MANIFEST.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output if args.output else args.reports/'figures'
    result=Renderer(args.reports,output).run()
    print(json.dumps(dict(status=result['status'],source_inner_report_state=result['source_inner_report_state'],
                         plots=len(result['plots']),skipped=len(result['skipped']))),flush=True)


if __name__=='__main__':main()
