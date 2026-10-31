"""Append-only compact delivery views of already public, actual P2 receipts.

No private source, label, prediction vector, checkpoint or model is opened.
Original generated files and their index remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

SEEDS=(31,47,83)
MAX_BYTES=2*1024**2
OUTPUTS=("04_INNER_SELECTION_LEDGER.csv","05_DSS_B0_B7_FOLD_SEED_METRICS.csv",
         "06_PAIRED_CONTRASTS.csv","07_CAPACITY_TEXT_MAP_PAIRING_CONTROLS.csv",
         "08_RISK_REPAIR_DAMAGE_AND_CALIBRATION.md","09_TRAINING_CURVES_AND_COMPUTE.csv",
         "DELIVERY_FILE_MAP.json")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def near(a,b):return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-12)


def csv_bytes(rows):
    require(bool(rows),"EMPTY_DELIVERY_VIEW")
    fields=list(dict.fromkeys(key for row in rows for key in row))
    require(len(fields)<=150 and len(rows)+1<=20000,"CSV_SHARD_LIMIT")
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for row in rows:
        require(all(v is None or type(v) in (str,int,float,bool) for v in row.values()),"NONSCALAR_CSV")
        require(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()),"NONFINITE_CSV")
        writer.writerow({k:"" if v is None else v for k,v in row.items()})
    raw=stream.getvalue().encode("utf-8")
    require(len(raw)<=MAX_BYTES,"DELIVERY_FILE_TOO_LARGE")
    return raw


def seed_contrasts(comparisons,aggregates):
    by_arm={row["arm"]:row for row in aggregates}
    require(len(by_arm)==len(aggregates),"DUPLICATE_ARM")
    require(by_arm["B0"]["required_seeds"]==[31] and len(by_arm["B0"]["seed_results"])==1,
            "B0_DETERMINISTIC_REFERENCE_CHANGED")
    out=[]
    for comparison in comparisons:
        left,right=comparison["left"],comparison["reference"]
        for seed in SEEDS:
            a=next(x for x in by_arm[left]["seed_results"] if x["seed"]==seed)
            b=next(x for x in by_arm[right]["seed_results"] if x["seed"]==(31 if right=="B0" else seed))
            row={"comparison":comparison["comparison"],"left":left,"reference":right,"seed":seed,
                 "reference_seed":b["seed"],"reference_B0_is_one_shared_deterministic_fit":right=="B0",
                 "scope":"five_within_fold_micro_per_seed_then_equal_three_seed_final_delta",
                 "source_fold_seed_detail":"06_fold_seed_paired_deltas.json","new_fits":0}
            for metric,nk,dk in (("Harrell","Harrell_num","Harrell_den"),("Uno60","Uno_num","Uno_den")):
                delta=a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None
                for label,values in (("left",a),("reference",b)):
                    for key in (metric,nk,dk):row[label+"_"+key]=values.get(key)
                final=(by_arm[left][metric]-by_arm[right][metric]
                       if by_arm[left][metric] is not None and by_arm[right][metric] is not None else None)
                require(near(final,comparison[metric+"_delta"]),"ACTUAL_FINAL_CONTRAST_CHANGED")
                row[metric+"_seed_micro_delta"]=delta
                row[metric+"_final_equal_seed_delta"]=final
                row[metric+"_fold_seed_positive"]=comparison[metric+"_positive_fold_seed_pairs"]
                row[metric+"_fold_seed_negative"]=comparison[metric+"_negative_fold_seed_pairs"]
                row[metric+"_fold_seed_zero"]=comparison[metric+"_zero_fold_seed_pairs"]
            out.append(row)
    require(len(out)==3*len(comparisons),"INCOMPLETE_SEED_CONTRAST")
    return out


def repair_micro(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row["comparison"],row["seed"])].append(row)
    out={}
    for (name,seed),items in groups.items():
        require(len(items)==5 and {r["fold"] for r in items}==set(range(1,6)),"REPAIR_FIVE_FOLDS_REQUIRED")
        good=math.fsum(r["repair_credit"] for r in items)
        bad=math.fsum(r["damage_credit"] for r in items)
        den=sum(r["comparable_pairs"] for r in items)
        supported=all(r["Uno_repair_credit"] is not None and r["Uno_damage_credit"] is not None and
                      r["Uno_pair_weight_denominator"] is not None and r["Uno_pair_weight_denominator"]>0 for r in items)
        un=math.fsum(r["Uno_repair_credit"]-r["Uno_damage_credit"] for r in items) if supported else None
        ud=math.fsum(r["Uno_pair_weight_denominator"] for r in items) if supported else None
        out[(name,seed)]={"repair_credit":good,"damage_credit":bad,"comparable_pairs":den,
            "Harrell_delta":(good-bad)/den if den else None,"Uno60_delta":un/ud if supported else None}
    return out


def curve_links(index,phase):
    coverage={(r["split"],r["family"],r["seed"]):r for r in index["source_coverage"]}
    require(len(coverage)==len(index["source_coverage"])==len(phase),"CURVE_SOURCE_COVERAGE_CHANGED")
    spans=[];offset=0
    for shard in index["curve_shards"]:
        spans.append((offset,offset+shard["data_records"],shard["file"]))
        offset+=shard["data_records"]
    require(offset==index["actual_epoch_records"]==sum(p["optimizer_steps"] for p in phase),"CURVE_UPDATES_DISAGREE")
    out=[]
    for p in phase:
        c=coverage[(p["split"],p["family"],p["seed"])]
        require(c["source_receipt_sha256"]==p["receipt_sha256"] and c["curve_records"]==p["optimizer_steps"],
                "CURVE_RECEIPT_BINDING_CHANGED")
        start=c["curve_start_offset"];end=start+c["curve_records"]
        shards=[name for a,b,name in spans if a<end and b>start]
        require(bool(shards),"CURVE_RANGE_WITHOUT_SHARD")
        out.append({"family":p["family"],"split":p["split"],"role":"inner" if p["evaluate_valid"] else "outer_final",
            "seed":p["seed"],"source_fits":p["actual_neural_fits"],"optimizer_steps":p["optimizer_steps"],
            "selected_epoch":p["selected_epoch"],"actual_saved_reload_replays":p["actual_checkpoint_replays"],
            "elapsed_seconds_not_wallclock":p["elapsed_seconds"],"peak_cuda_allocated_bytes":p["peak_cuda_allocated_bytes"],
            "pretraining_parameters":p["parameters"]["pretraining_trainable"],
            "selected_program_Huber":p.get("selected_program_huber"),
            "selected_hard_gap_image_to_gene":p.get("selected_hard_gap_image_to_gene"),
            "selected_hard_gap_gene_to_image":p.get("selected_hard_gap_gene_to_image"),
            "curve_global_start_offset":start,"curve_epoch_records":c["curve_records"],
            "curve_CSV_shards":";".join(shards),"full_curve_index":"09_TRAINING_BUDGET_CURVE_INDEX.json",
            "source_receipt_sha256":p["receipt_sha256"],"selected_checkpoint_sha256":p["source_checkpoint_sha256"],
            "shared_teacher_or_risk_head_fits_assigned_to_this_source":0,
            "separate_shared_compute_ledger":"09_cost_ledger.json","new_fits_by_view_generator":0})
    return out


def risk_markdown(repairs,scales,comparisons):
    micros=repair_micro(repairs)
    final=[r for r in scales if r["role"]=="outer_final"]
    require(len(final)==180 and len({(r["arm"],r["split"],r["seed"]) for r in final})==180,
            "FINAL_SCALE_CELL_SET_CHANGED")
    lines=["# 风险排序修复、损伤与尺度诊断","",
        "本附录只汇总已经拟合的风险向量及匿名统计，没有新拟合、重选缩放系数或使用外折结果修改模型。",
        "`修复/损伤 credit` 允许风险相等时的半分；先在每个 seed 内合并五个折的可比对，再对三个 seed 等权平均，不能将15 cells当15个独立重训基线。B0每折只有一个确定性seed31教师，被各模型seed共同引用。","",
        "## 配对排序修复与损伤","",
        "下表的credit是每个seed五折累计credit的三seed均值，仅作为量级诊断；Δ是相应分子/分母的seed微汇总差值。不同折风险不互相比较。","",
        "| 核心比较 | 修复credit均值 | 损伤credit均值 | ΔHarrell | ΔUno60 |",
        "|---|---:|---:|---:|---:|"]
    for c in comparisons:
        group=[micros[(c["comparison"],seed)] for seed in SEEDS]
        values={metric:statistics.mean(x[metric+"_delta"] for x in group)
                if all(x[metric+"_delta"] is not None for x in group) else None for metric in ("Harrell","Uno60")}
        require(all(near(values[m],c[m+"_delta"]) for m in values),"REPAIR_AND_ACTUAL_CORE_DIFFER")
        show=lambda value:"NA" if value is None else f"{value:.9f}"
        lines.append(f"| {c['comparison']} | {statistics.mean(x['repair_credit'] for x in group):.3f} | "
            f"{statistics.mean(x['damage_credit'] for x in group):.3f} | {show(values['Harrell'])} | {show(values['Uno60'])} |")
    lines += ["","任何无Uno支持的折都保持NA，不把零分母解释成零增量。全部逐fold/seed原始汇总继续保留在 `08_pair_repair_damage.json/.csv`。",
        "","## 风险修正的训练与外折尺度","",
        "下表只使用180个最终已锁λ的读出头，每个模型15 cells；中位数是描述性尺度，不代替主C-index汇总。σ(Δrisk)/σ(r0)不等于概率校准。",
        "训练列来自**在完整交叉拟合教师offset上拟合的读出头的样本内指标**。虽然offset对该训练患者排除了教师和β自身标签，读出头仍使用这些患者的DSS标签；因此训练risk不是完整pipeline的OOF。",
        "","| 模型 | cells | 训练σ比中位数 | 外折σ比中位数 | 训练ΔHarrell中位数 | 外折ΔHarrell中位数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in sorted({r["arm"] for r in final}):
        rows=[r for r in final if r["arm"]==arm]
        require(len(rows)==15,"MODEL_SCALE_SUPPORT_CHANGED")
        fields=("train_delta_sd_over_anchor_sd","evaluation_delta_sd_over_anchor_sd",
                "train_Harrell_delta_vs_r0","evaluation_Harrell_delta_vs_r0")
        values=[statistics.median(r[k] for r in rows if r[k] is not None) for k in fields]
        lines.append(f"| {'B3_TEXTLESS_INCREMENT' if arm=='B3_TEXTLESS' else arm} | {len(rows)} | "+
                     " | ".join(f"{v:.9f}" for v in values)+" |")
    lines += ["","## 适用边界","",
        "- 此处没有估计基线生存函数、绝对生存概率、calibration slope或Brier score，所以不能宣布已经完成概率校准验证。文件名中的calibration要求在此用明确的未实施边界回答；已做的是风险尺度与排序诊断。",
        "- 26系数头总增量包含压缩后的P/G/C主特征和两个机制特征；上表不能把全部Δrisk归因于纯交互分支。",
        "- 训练/外折σ差异只能帮助定位尺度问题，不能在已观察的验证或外折上挑选新的缩放系数并当作独立收益。",
        "- TEXTLESS只关闭增量编码路径中的显式文本，共同r0仍包含TITAN S32；B2是共享8视觉reader×50程序token的普通attention适配，不是外部论文官方native复现。",
        "- 全部核心对照、包括负向结果均保留；没有置信区间、bootstrap或p值，也没有因本附录触发新的训练或扩大模型。",""]
    return "\n".join(lines)


def generate(root):
    root=Path(root).resolve()
    require(all(not (root/n).exists() for n in OUTPUTS),"APPEND_ONLY_OUTPUT_EXISTS")
    names={"04_inner_phase_selection.json","04_inner_ridge_selection.json","05_fold_seed_metrics.json",
           "05_seed_aggregated_metrics.json","06_paired_comparisons.json","06_fold_seed_paired_deltas.json",
           "07_controls.json","08_pair_repair_damage.json","09_phase_a_actual_fits.json","09_cost_ledger.json",
           "09_resource_profile.json","09_TRAINING_BUDGET_CURVE_INDEX.json","00_report_index.json"}
    names.update(p.name for p in root.glob("08_risk_scale_and_fit_metrics*.json"))
    require(len([n for n in names if n.startswith("08_risk_scale")])>0,"RISK_SCALE_SOURCE_MISSING")
    sources={n:sha(root/n) for n in names}
    phase=load(root/"04_inner_phase_selection.json");ridge=load(root/"04_inner_ridge_selection.json")
    require(len(phase)==360 and len(ridge)==180,"INNER_SOURCE_CELL_COUNT_CHANGED")
    ledger=[]
    for r in phase:
        ledger.append({"row_type":"OUTCOME_FREE_PHASE_A_CHECKPOINT_SELECTION","family":r["family"],
            "split":r["split"],"seed":r["seed"],"selected_epoch":r["selected_epoch"],
            "selected_program_Huber":r["selected_program_huber"],
            "selected_hard_gap_image_to_gene":r["selected_hard_gap_image_to_gene"],
            "selected_hard_gap_gene_to_image":r["selected_hard_gap_gene_to_image"],
            "DSS_used_for_this_selection":False,"uses_outer_test":False,
            "source_receipt_sha256":r["receipt_sha256"],"new_fits":0})
    for r in ridge:
        for c in r["candidate_summaries"]:
            ledger.append({"row_type":"DSS_RIDGE_CANDIDATE_THREE_INNER_FOLD_MICRO","outer":r["outer"],
                "arm":r["arm"],"seed":r["seed"],"candidate_lambda":c["ridge_lambda"],
                "selected_lambda":r["ridge_lambda"],"candidate_is_selected":r["ridge_lambda"]==c["ridge_lambda"],
                **{k:c[k] for k in ("Harrell","Harrell_num","Harrell_den","Uno60","Uno_num","Uno_den","Uno60_complete_support")},
                "DSS_used_for_this_selection":True,"uses_outer_test":r["uses_outer_test"],
                "selection_rule":r["selection_rule"],"new_fits":0})
    folds=load(root/"05_fold_seed_metrics.json");require(len(folds)==185,"ACTUAL185_FOLD_ROWS_REQUIRED")
    fields=("arm","reported_arm","fold","seed","status","Harrell","Harrell_num","Harrell_den","Uno60",
            "Uno_num","Uno_den","Uno60_valid","Uno60_NA_reason","ridge_lambda","selected_phase_a_epoch",
            "receipt_sha256","state_sha256","source_anchor_sha256","deterministic_single_refit")
    fold_view=[{k:r.get(k) for k in fields} for r in folds]
    require(sum(r["arm"]=="B0" for r in folds)==5,"B0_FITS_MISCOUNTED")
    comparisons=load(root/"06_paired_comparisons.json")
    require(len(comparisons)==12 and len(load(root/"06_fold_seed_paired_deltas.json"))==180,"CORE_COMPARISON_COUNT_CHANGED")
    pairs=seed_contrasts(comparisons,load(root/"05_seed_aggregated_metrics.json"))
    controls=load(root/"07_controls.json");control_rows=[]
    for r in controls["results"]:
        control_rows.append({"row_type":"SAME_LOCKED_MODEL_PERFORMANCE_CONTROL","comparison":r["comparison"],
            "left":r["left"],"reference":r["reference"],"Harrell_delta":r["Harrell_delta"],"Uno60_delta":r["Uno60_delta"],
            "Harrell_positive_fold_seed_pairs":r["Harrell_positive_fold_seed_pairs"],
            "Uno60_positive_fold_seed_pairs":r["Uno60_positive_fold_seed_pairs"],
            "TEXTLESS_entire_risk_text_free":controls["TEXTLESS"]["whole_risk_text_free"],
            "shared_r0_contains_TITAN_S32":controls["TEXTLESS"]["common_r0_contains_TITAN_S32"],
            "B2_NATIVE_is_official_paper_reproduction":controls["B2"]["B2_NATIVE_is_official_paper_reproduction"],
            "CI_or_p_value_computed":False,"new_fits":0})
    for r in load(root/"09_resource_profile.json")["comparisons"]:
        control_rows.append({"row_type":"ACTUAL_ZERO_UPDATE_RESOURCE_PROFILE",**r,"new_fits":0})
    scales=[]
    for n in sorted(names):
        if n.startswith("08_risk_scale"):scales.extend(load(root/n))
    require(len(scales)==1800,"ACTUAL_RISK_SCALE_FIT_COUNT_CHANGED")
    md=risk_markdown(load(root/"08_pair_repair_damage.json"),scales,comparisons)
    allphase=load(root/"09_phase_a_actual_fits.json");curves=load(root/"09_TRAINING_BUDGET_CURVE_INDEX.json")
    require(len(allphase)==480,"ACTUAL480_SOURCE_FITS_REQUIRED")
    curve_rows=curve_links(curves,allphase)
    for shard in curves["curve_shards"]:
        name=shard["file"];require(Path(name).name==name and sha(root/name)==shard["sha256"],"ORIGINAL_FULL_CURVE_BYTES_CHANGED")
        sources[name]=shard["sha256"]
    outputs={OUTPUTS[0]:csv_bytes(ledger),OUTPUTS[1]:csv_bytes(fold_view),OUTPUTS[2]:csv_bytes(pairs),
             OUTPUTS[3]:csv_bytes(control_rows),OUTPUTS[4]:md.encode("utf-8"),OUTPUTS[5]:csv_bytes(curve_rows)}
    for name,raw in outputs.items():
        require(len(raw)<=MAX_BYTES,"VIEW_OVER2MIB")
        with (root/name).open("xb") as f:f.write(raw)
    counts={OUTPUTS[0]:len(ledger),OUTPUTS[1]:len(fold_view),OUTPUTS[2]:len(pairs),OUTPUTS[3]:len(control_rows),OUTPUTS[5]:len(curve_rows)}
    mapping={"schema":"semantic_p2_actual_delivery_file_map_v1","status":"COMPLETE_APPEND_ONLY_DERIVED_VIEWS",
        "generator_source_sha256":sha(__file__),"sources":[{"file":n,"sha256":sources[n]} for n in sorted(sources)],
        "views":[{"file":n,"sha256":sha(root/n),"bytes":len(raw),"data_rows":counts.get(n)} for n,raw in outputs.items()],
        "original_generated_files_and_index_modified":False,"original_180_fold_seed_pairs_preserved":True,
        "formal_source_fits":480,"source_history_epoch_records_indexed":curves["actual_epoch_records"],
        "original_curve_shards_indexed":len(curves["curve_shards"]),"curve_rows_copied_or_resampled":0,
        "shared_deterministic_B0_fits":5,"CORE_comparisons":12,"CORE_seed_micro_rows":len(pairs),
        "training_metrics_scope":"head_in_sample_over_crossfit_teacher_offsets_not_end_to_end_OOF",
        "absolute_survival_probability_calibration_performed":False,"new_fits":0,"new_model_forwards":0,
        "new_optimizer_updates":0,"confidence_intervals_computed":False,"p_values_computed":False,
        "final_publication_original_private_identifier_scan_pending":True,
        "self_hash_bound_externally_by_final_publication_manifest_not_self_referential":True}
    raw=(json.dumps(mapping,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
    require(len(raw)<=MAX_BYTES,"VIEW_MAP_TOO_LARGE")
    with (root/OUTPUTS[6]).open("xb") as f:f.write(raw)
    require(all(sha(root/n)==digest for n,digest in sources.items()),"SOURCE_CHANGED_DURING_VIEW_GENERATION")
    return {"status":mapping["status"],"new_files":len(OUTPUTS),"CSV_data_rows":counts,
            "delivery_map_sha256":sha(root/OUTPUTS[6]),"new_fits":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--public-reports",required=True)
    print(json.dumps(generate(parser.parse_args().public_reports),sort_keys=True))
