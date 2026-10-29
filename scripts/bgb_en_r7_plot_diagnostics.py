"""Six fixed R7 diagnostic figures from anonymous aggregate reports only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ARMS=("ADD","RAW","NORM","INDEP","SLOT","GATE1","GATE2","SLOT_WRONG_RNA","GATE1_WRONG_RNA","GATE1_WRONG_PATH","GATE2_WRONG_RNA")
MATCHED=ARMS[:7]
STAGES=("B1",*MATCHED)
PARTITIONS,SEEDS,EPOCHS=(0,1,2),(31,47,83),(0,20,40,80)
PAIRS=(("SLOT","SLOT_WRONG_RNA"),("GATE1","GATE1_WRONG_RNA"),("GATE1","GATE1_WRONG_PATH"),("GATE2","GATE2_WRONG_RNA"))
SERIES=("Base_query","Query_change","Ebase","delta_E","delta_G","cross_risk_SD")
BASENAMES=("interface_trajectory","stage_Harrell","stage_Uno60","query_pairing_deltas","interaction_risk_SD","program_cancellation")
INTERFACE="R7_0_R6_INTERFACE_AUDIT.json"
RESULTS="R7_RESULTS.json"
SHARDS=tuple(f"R7_DIAGNOSTICS_{arm}_p{part}.json" for arm in ARMS for part in PARTITIONS)
INPUTS=frozenset((INTERFACE,RESULTS,*SHARDS))
COLORS=("#3B6EA8","#CB6D34","#4C916F","#9465A7","#A59439","#70858D")
HASHSALT="bgb_en_r7_anonymous_diagnostics_v1"
BINDING,README="R7_FIGURE_BINDINGS.json","FIGURE_README.md"


def require(value,code):
    if not value:raise ValueError(code)


def sha(content):return hashlib.sha256(content).hexdigest()


def number(value,code,*,low=None,high=None,nullable=False):
    if nullable and value is None:return None
    require(type(value) in (int,float) and math.isfinite(value) and (low is None or value>=low)
        and (high is None or value<=high),code)
    return float(value)


def equal(left,right,code):
    require(left is None and right is None or type(left) in (float,int) and type(right) in (float,int)
        and math.isfinite(left) and math.isfinite(right) and abs(left-right)<=1e-10,code)


def exact(rows,fields,expected,code):
    require(isinstance(rows,list) and all(isinstance(r,dict) for r in rows),code)
    keys=[tuple(r.get(k) for k in fields) for r in rows]
    require(len(keys)==len(expected) and len(set(keys))==len(keys) and set(keys)==set(expected),code)
    return dict(zip(keys,rows))


class Reports:
    def __init__(self,path):self.root=Path(path).resolve();self.sources={}
    def read(self,name):
        require(name in INPUTS,"ONLY_FIXED_35_ANONYMOUS_INPUTS_ALLOWED")
        p=self.root/name
        require(p.is_file() and not p.is_symlink() and p.resolve().parent==self.root,"ANONYMOUS_REPORT_FILE_MISSING_OR_UNSAFE")
        before=p.stat();content=p.read_bytes();after=p.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and len(content)<=2*1024**2,"REPORT_CHANGED_OR_EXCEEDS_2MIB")
        record=dict(file=name,bytes=len(content),sha256=sha(content))
        require(name not in self.sources or self.sources[name]==record,"REPORT_BYTES_CHANGED_BETWEEN_READS")
        self.sources[name]=record
        def reject(_):raise ValueError("NONFINITE_REPORT_JSON")
        value=json.loads(content.decode("utf-8-sig"),parse_constant=reject)
        require(isinstance(value,dict),"ANONYMOUS_JSON_OBJECT_REQUIRED")
        return value
    def unchanged(self):
        for name in list(self.sources):self.read(name)


def mean(values):return statistics.fmean(values) if values and all(v is not None for v in values) else None


def group(label,color,values,*,marker="o",means=None):
    return dict(label=label,color=color,values=values,marker=marker,means=means)


def plan(name,title,xlabel,labels,groups,means,note,sources,table,aggregation,unit,*,zero=False):
    return dict(basename=name,title=title,xlabel=xlabel,labels=labels,groups=groups,means=means,note=note,
        sources=sorted(sources),source_table=table,aggregation=aggregation,axis_unit=unit,zero_line=zero,kind="scatter")


def interface_plan(reports):
    audit=reports.read(INTERFACE)
    require(audit.get("schema")=="bgb_en_r7_readonly_r6_interface_audit_v1"
        and audit.get("status")=="PASS_9_R6_CELLS_36_STATE_FORWARDS" and audit.get("rows_count")==36
        and audit.get("actual_model_forwards")==36 and audit.get("repeated_probes_not_independent_cohorts") is True
        and audit.get("validation_scores_computed") is False and audit.get("labels_used_for_probe_selection") is False,
        "READONLY_R6_INTERFACE_AUDIT_REQUIRED")
    rows=exact(audit.get("rows"),("partition","seed","epoch"),{(p,s,e) for p in PARTITIONS for s in SEEDS for e in EPOCHS},"EXACT_36_INTERFACE_STATES_REQUIRED")
    series=[]
    for j,name in enumerate(SERIES):
        values=[]
        for epoch in EPOCHS:
            cells=[]
            for part in PARTITIONS:
                for seed in SEEDS:
                    raw=rows[(part,seed,epoch)].get("quantities",{}).get(name)
                    value=raw if name=="cross_risk_SD" else raw.get("coordinate_RMS") if isinstance(raw,dict) else None
                    value=number(value,"POSITIVE_INTERFACE_VALUE_REQUIRED",low=0)
                    require(value>0,"LOG_INTERFACE_ZERO_CANNOT_BE_REPLACED")
                    cells.append(value)
            values.append(statistics.fmean(cells))
        series.append(dict(label=name,color=COLORS[j],raw_equal_cell_means=values,values=[v/values[0] for v in values]))
    return dict(basename="interface_trajectory",kind="trajectory",title="R6 interface scale trajectories",
        xlabel="Saved state epoch",ylabel="Mean / own epoch-0 mean (log scale)",series=series,
        sources=[INTERFACE],source_table="rows.quantities coordinate_RMS; cross_risk_SD scalar",
        aggregation="equal nine-cell mean at each epoch, divided by that series' equal nine-cell epoch-zero mean",
        axis_unit="dimensionless ratio to own epoch-zero mean; logarithmic Y",
        note="First average the same 9 cells equally at each epoch, then normalize each series by its own epoch-0 mean. Reused fixed training probes; these descriptive R6 trajectories are not independent patients or a new R7 performance estimate.")


def metric(row,label):
    numerator,denominator=("Harrell_num","Harrell_den") if label=="Harrell" else ("Uno_num","Uno_den")
    n=number(row.get(numerator),"CONCORDANCE_COMPONENT_INVALID",low=0)
    d=number(row.get(denominator),"CONCORDANCE_COMPONENT_INVALID",low=0)
    require(n<=d+1e-10,"CONCORDANCE_NUMERATOR_EXCEEDS_DENOMINATOR")
    value=n/d if d else None
    equal(row.get(label),value,"REPORTED_COMPONENT_RATIO_CHANGED")
    return value


def result_plans(reports):
    result=reports.read(RESULTS)
    require(result.get("schema")=="bgb_en_r7_complete_report_v1" and result.get("status")=="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
        and result.get("budget",{}).get("formal_neural_fits")==99 and result["budget"].get("formal_optimizer_updates")==7920
        and result.get("decision",{}).get("complete_actual_checkpoint_replays")==297
        and result.get("no_cross_partition_risk_ranking") is True,"COMPLETE_R7_ANONYMOUS_RESULTS_REQUIRED")
    summaries=result.get("primary_DSS",{})
    require(set(summaries)==set(("B0","B1",*ARMS)),"EXACT_13_PRIMARY_ARM_SUMMARIES_REQUIRED")
    points={}
    for arm,row in summaries.items():
        require(row.get("complete_cells")==9 and row.get("status")=="COMPLETE_NINE_CELLS","PRIMARY_ARM_INCOMPLETE")
        seeds=exact(row.get("per_seed"),("seed",),{(s,) for s in SEEDS},"THREE_SEED_COMPONENT_SUMMARIES_REQUIRED")
        points[arm]={label:[metric(seeds[(s,)],label) for s in SEEDS] for label in ("Harrell","Uno60")}
        for label in ("Harrell","Uno60"):equal(row.get(label),mean(points[arm][label]),"PRIMARY_SEED_MEAN_CHANGED")
    plans=[]
    for label in ("Harrell","Uno60"):
        values=[points[a][label] for a in STAGES]
        groups=[group("Seed "+str(s),COLORS[j],[[r[j]] for r in values]) for j,s in enumerate(SEEDS)]
        plans.append(plan("stage_"+label,"Fixed-stage DSS comparison",label+" concordance",list(STAGES),groups,[mean(r) for r in values],
            "Colored points: each seed's reported sum of 3 within-partition component counts. Black diamond: equal-three-seed mean. Fixed stage order; no cross-partition risk ranking, CI, p-values or winner selection.",
            [RESULTS],"primary_DSS.per_seed concordance components","component-pooled per-seed values followed by equal three-seed mean","dimensionless concordance"))
    paired=exact([r for r in result.get("paired_comparisons",[]) if (r.get("candidate"),r.get("reference")) in PAIRS],
        ("candidate","reference"),set(PAIRS),"FOUR_FIXED_QUERY_PAIR_COMPARISONS_REQUIRED")
    groups=[]
    for j,label in enumerate(("Harrell","Uno60")):
        values=[];averages=[]
        for candidate,reference in PAIRS:
            row=paired[(candidate,reference)]
            values.append([a-b if a is not None and b is not None else None for a,b in zip(points[candidate][label],points[reference][label])])
            averages.append(mean(values[-1]));equal(row.get("delta_"+label),averages[-1],"QUERY_PAIRED_DELTA_CHANGED")
        groups.append(group("Delta "+label+" (3 seeds + mean)",COLORS[j],values,marker="o" if j==0 else "s",means=averages))
    plans.append(plan("query_pairing_deltas","Patient-context pairing controls","Delta concordance: matched minus retrained wrong context",
        [a+" - "+b for a,b in PAIRS],groups,None,
        "Circles: Harrell; squares: Uno60. Small marks: same-seed comparisons; outlined marks: equal-three-seed means. The two metrics remain separate and do not form a composite ranking. Controls are retrained models, not frozen interventions.",
        [RESULTS],"primary_DSS per_seed and four fixed paired_comparisons","same-seed matched-minus-control after reported component pooling; equal seed mean","dimensionless delta concordance",zero=True))
    return result,plans


def diagnostic_plans(reports,result):
    index=result.get("diagnostic_index",{})
    require(index.get("logical_cells")==99 and index.get("epochs_per_cell")==[20,40,60,80],"DIAGNOSTIC_SCOPE_CHANGED")
    desc=exact(index.get("parts"),("file",),{(s,) for s in SHARDS},"EXACT_33_DIAGNOSTIC_SHARDS_REQUIRED")
    rows=[]
    for arm in ARMS:
        for part in PARTITIONS:
            name=f"R7_DIAGNOSTICS_{arm}_p{part}.json";payload=reports.read(name);bound=desc[(name,)];source=reports.sources[name]
            require(payload.get("schema")=="bgb_en_r7_diagnostic_shard_v1" and bound.get("rows")==3
                and bound.get("sha256")==source["sha256"] and bound.get("bytes")==source["bytes"],"DIAGNOSTIC_SHARD_BYTES_OR_SHA_CHANGED")
            exact(payload.get("rows"),("arm","partition","seed"),{(arm,part,s) for s in SEEDS},"DIAGNOSTIC_SHARD_IDENTITY_CHANGED")
            rows.extend(payload["rows"])
    cells=exact(rows,("arm","partition","seed"),{(a,p,s) for a in ARMS for p in PARTITIONS for s in SEEDS},"EXACT_99_DIAGNOSTIC_MODEL_ROWS_REQUIRED")
    values=[]
    for arm in ARMS:
        scores=[]
        for part in PARTITIONS:
            for seed in SEEDS:
                validation=cells[(arm,part,seed)].get("final_risk_diagnostics",{}).get("validation",{})
                value=number(validation.get("SD_cross_delta_over_offset"),"VALIDATION_CROSS_ANCHOR_SD_RATIO_REQUIRED",low=0)
                require(arm!="ADD" or value==0,"ADD_INTERACTION_STRUCTURAL_ZERO_CHANGED")
                scores.append(value)
        values.append(scores)
    risk=plan("interaction_risk_SD","Validation interaction-risk amplitude","Cross-risk SD / frozen anchor-risk SD (linear scale)",
        [a+(" [cross off]" if a=="ADD" else "") for a in ARMS],
        [group("Model cells",COLORS[0],values)],[statistics.fmean(v) for v in values],
        "Small points: nine partition-seed validation SD ratios; black diamond: equal-nine-cell mean. ADD is structurally zero. This is a ratio of standard deviations, not variance explained or a causal contribution; zeros stay zero.",
        [RESULTS,*SHARDS],"rows.final_risk_diagnostics.validation.SD_cross_delta_over_offset","saved per-cell ratio then equal nine-cell mean","dimensionless SD ratio; linear scale",zero=True)
    groups=[];counts=[]
    for j,(key,label) in enumerate((("slot_path_contributions","Path"),("slot_gene_contributions","Gene"))):
        series=[];means=[];nvalid=[]
        for arm in MATCHED:
            scores=[]
            for part in PARTITIONS:
                for seed in SEEDS:
                    slot=cells[(arm,part,seed)].get("final_risk_diagnostics",{}).get("validation",{}).get("slot_contributions",{}).get(key)
                    if slot is None:value=None
                    else:
                        value=number(slot.get("absolute_sum_over_sum_absolute_mean"),"PROGRAM_CANCELLATION_RATIO_INVALID",low=0,high=1,nullable=True)
                        support=slot.get("nonzero_absolute_sum_patients")
                        require(type(support) is int and support>=0 and (value is None)==(support==0),"PROGRAM_CANCELLATION_SUPPORT_CHANGED")
                    scores.append(value)
            valid=[v for v in scores if v is not None];series.append(scores);nvalid.append(len(valid));means.append(statistics.fmean(valid) if valid else None)
        groups.append(group(label+" (available cells + mean)",COLORS[j],series,marker="o" if j==0 else "s",means=means));counts.append(nvalid)
    cancellation=plan("program_cancellation","Program contribution cancellation", "Mean patient ratio: |sum(contributions)| / sum(|contributions|)",
        [f"{a} (P {counts[0][i]}/9; G {counts[1][i]}/9)" for i,a in enumerate(MATCHED)],groups,None,
        "Each saved cell value averages the patient-level ratio only where the absolute-contribution denominator is positive. P/G counts label available model cells out of 9. Missing values are NA, not zero. A higher ratio is not automatically better prediction.",
        [RESULTS,*SHARDS],"rows.final_risk_diagnostics.validation.slot_contributions.*.absolute_sum_over_sum_absolute_mean",
        "saved mean of patient-level ratios; displayed available-cell values and their equal mean","dimensionless cancellation ratio; range 0 to 1")
    cancellation["ratio_range"]=(0,1)
    return risk,cancellation


def plans_from(reports):
    first=interface_plan(reports);result,stages=result_plans(reports)
    plans=[first,*stages,*diagnostic_plans(reports,result)]
    require(tuple(p["basename"] for p in plans)==BASENAMES and set(reports.sources)==INPUTS,"FIXED_SIX_FIGURES_OR_35_SOURCES_CHANGED")
    reports.unchanged();return plans


def draw(plan):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":15,"axes.labelsize":11,
        "svg.fonttype":"path","svg.hashsalt":HASHSALT,"figure.facecolor":"white"})
    fig=plt.figure(figsize=(11.5,7.4));trajectory=plan["kind"]=="trajectory"
    ax=fig.add_axes((.12,.27,.83,.61) if trajectory else (.40,.27,.57,.61));handles=[]
    if trajectory:
        for row in plan["series"]:
            ax.plot(EPOCHS,row["values"],marker="o",lw=1.4,ms=4,color=row["color"],label=row["label"])
        ax.set_yscale("log");ax.set_xticks(EPOCHS);ax.set_ylabel(plan["ylabel"])
        handles,_=ax.get_legend_handles_labels()
    else:
        count=len(plan["labels"])
        for gi,row in enumerate(plan["groups"]):
            offset=(gi-(len(plan["groups"])-1)/2)*.16
            for y,values in enumerate(row["values"]):
                for j,value in enumerate(values):
                    if value is None:continue
                    jitter=(j-(len(values)-1)/2)*.025
                    ax.scatter([value],[y+offset+jitter],s=23,marker=row["marker"],c=row["color"],alpha=.8,edgecolors="white",linewidths=.4,zorder=3)
                if row["means"] is not None:
                    value=row["means"][y]
                    if value is None:ax.text(.985,y+offset,"NA",transform=ax.get_yaxis_transform(),ha="right",va="center",color=row["color"],fontsize=8)
                    else:ax.scatter([value],[y+offset+.11],s=45,marker=row["marker"],c=row["color"],edgecolors="black",linewidths=.7,zorder=5)
            handles.append(Line2D([],[],marker=row["marker"],color="none",markerfacecolor=row["color"],markeredgecolor=row["color"],markersize=5,label=row["label"]))
        if plan["means"] is not None:
            for y,value in enumerate(plan["means"]):
                if value is None:ax.text(.985,y+.29,"NA",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=8,color="#777777")
                else:ax.scatter([value],[y+.29],marker="D",c="black",s=42,zorder=6)
            handles.append(Line2D([],[],marker="D",color="none",markerfacecolor="black",markersize=5,label="Aggregate"))
        ax.set_yticks(range(count),plan["labels"]);ax.set_ylim(count-.5,-.5);ax.tick_params(axis="y",length=0,pad=8)
        if plan["zero_line"]:ax.axvline(0,color="#777777",lw=.8,ls="--",zorder=1)
        if "ratio_range" in plan:ax.set_xlim(-.02,1.02)
    ax.set_xlabel(plan["xlabel"]);ax.set_title(plan["title"],loc="left",pad=14)
    ax.grid(axis="both" if trajectory else "x",color="#DFE3E8",lw=.6);ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    if not trajectory:ax.spines["left"].set_visible(False)
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.115 if trajectory else .395,.12),frameon=False,ncol=min(3,len(handles)),fontsize=9,handletextpad=.4,columnspacing=1.1)
    fig.text(.035,.038,"\n".join(textwrap.wrap(plan["note"],143)),fontsize=9,color="#47515D",va="bottom",linespacing=1.35)
    return fig


def validate_svg(content):
    root=ET.fromstring(content)
    require(root.tag.rsplit("}",1)[-1]=="svg","STATIC_SVG_ROOT_REQUIRED")
    for node in root.iter():
        require(node.tag.rsplit("}",1)[-1].lower() not in ("image","script","foreignobject","animate"),"SVG_RASTER_OR_ACTIVE_CONTENT_FORBIDDEN")
        require(all(v.startswith("#") for k,v in node.attrib.items() if k.rsplit("}",1)[-1].lower()=="href"),"SVG_EXTERNAL_HREF_FORBIDDEN")


def file_record(path,format):
    content=path.read_bytes();return dict(file=path.name,format=format,bytes=len(content),sha256=sha(content))


def render(report_dir,output):
    reports=Reports(report_dir);plans=plans_from(reports);output=Path(output).resolve()
    require(not any(output==p.resolve() or output.is_relative_to(p.resolve()) for p in
        (ROOT/"reports/bgb_en",ROOT/"reports/bgb_en_r5",ROOT/"reports/bgb_en_r6")),"HISTORICAL_REPORT_NAMESPACES_PROTECTED")
    names=[n+"."+e for n in BASENAMES for e in ("svg","png")]+[BINDING,README]
    require(not any((output/n).exists() for n in names),"PLOT_OUTPUT_EXISTS_NO_OVERWRITE")
    output.mkdir(parents=True,exist_ok=True);figures=[]
    generator=dict(file="scripts/bgb_en_r7_plot_diagnostics.py",sha256=sha(Path(__file__).read_bytes()))
    for plan in plans:
        fig=draw(plan)
        try:
            svg=output/(plan["basename"]+".svg");png=output/(plan["basename"]+".png")
            with svg.open("xb") as stream:fig.savefig(stream,format="svg",metadata={"Date":None,"Creator":"BG-BEN R7 diagnostic plotter"})
            validate_svg(svg.read_bytes())
            with png.open("xb") as stream:fig.savefig(stream,format="png",dpi=160,metadata={"Software":"BG-BEN R7 diagnostic plotter"})
        finally:plt.close(fig)
        figures.append(dict(basename=plan["basename"],source_files=plan["sources"],source_table=plan["source_table"],aggregation=plan["aggregation"],
            axes=dict(x=plan["xlabel"],y=plan.get("ylabel","anonymous model or comparison code"),unit=plan["axis_unit"],
                x_scale="linear",y_scale="log" if plan["kind"]=="trajectory" else "linear"),
            one_axes=True,uncertainty_intervals=False,pvalues=False,outputs=[file_record(svg,"SVG"),file_record(png,"PNG")]))
    reports.unchanged();require(generator["sha256"]==sha(Path(__file__).read_bytes()),"PLOT_GENERATOR_CHANGED_DURING_RENDER")
    readme="""# R7 普通科研诊断图

每个PNG/SVG配对来自同一个matplotlib figure；每图只有一个坐标轴，没有CI/p值、患者图像、热图或模型调用。

- interface_trajectory：只读旧R6的36个保存状态。每epoch先等权平均9个模型cell，再除以该series自己的epoch0均值；只有这张图使用log Y。所有原值和均值必须严格大于0，不加伪计数，不替换零值。训练探针重复使用，不是独立患者证据。
- stage_Harrell / stage_Uno60：固定B1、ADD、RAW、NORM、INDEP、SLOT、GATE1、GATE2顺序。3个点为已报告的每seed分区组件汇总值，黑菱形为3seed等权均值，不在图中跨分区拼风险。
- query_pairing_deltas：四个固定正确/错患者重训比较；Harrell圆点和Uno60方点分别表示各seed差值及其均值，不混成总体排名。
- interaction_risk_SD：11个模型在第80轮validation的cross SD / anchor SD，9个cell点和等权均值，linear坐标，保留ADD结构零；不称为variance explained。
- program_cancellation：7个matched模型的path/gene读出，直接使用已保存的病人级比值平均：mean_i[abs(sum_k contribution_ik) / sum_k abs(contribution_ik)]，只纳入分母非零的病例。图中P/G n/9是该统计可用的模型cell数；缺失为NA而非0。较高比值不自动代表更好预测。

`R7_FIGURE_BINDINGS.json`绑定严格35份匿名输入（接口audit、R7_RESULTS、33份诊断分片）的字节数/SHA、绘图器SHA与12张图文件的字节数/SHA。SVG为静态向量，禁止image或外部href。已有同名产物拒绝覆盖。本脚本不训练、不前向、不读取数组、标签或权重，也不选择模型、epoch或预算。
"""
    with (output/README).open("x",encoding="utf-8") as stream:stream.write(readme)
    binding=dict(schema="bgb_en_r7_figure_bindings_v1",status="COMPLETE_6_ANONYMOUS_DIAGNOSTIC_FIGURES",figure_count=6,
        generator=generator,source_files=[reports.sources[n] for n in sorted(reports.sources)],figures=figures,readme=file_record(output/README,"Markdown"),
        rendering=dict(backend="Agg",matplotlib_version=matplotlib.__version__,svg_hashsalt=HASHSALT,svg_fonttype="path",svg_metadata_Date=None,pair_saved_from_same_figure=True),
        runtime=dict(optimizer_fits=0,optimizer_updates=0,model_forwards=0,backward_calls=0,model_patient_array_label_or_weight_files_read=False),
        limits=dict(no_CI_pvalues_or_model_selection=True,reused_training_probes_not_independent_patients=True))
    with (output/BINDING).open("x",encoding="utf-8") as stream:stream.write(json.dumps(binding,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
    return binding


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--report-dir",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();value=render(args.report_dir,args.output)
    print(json.dumps(dict(status=value["status"],figures=6,optimizer_fits=0,model_forwards=0)))


if __name__=="__main__":main()
