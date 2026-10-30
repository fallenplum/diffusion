"""Lossless anonymous R0 state projections and T2 summaries; zero model I/O.

Consumes JSON aggregates only. It never loads checkpoints, patient tensors,
coefficients or predictions. Repeated historical training curves are referenced
by their immutable parent report rather than copied into each new report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n")


def table(path,rows):
    fields=list(rows[0])
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def build(r0_root,t2_root,lock_path,output):
    r0_root,t2_root,output=map(Path,(r0_root,t2_root,output))
    r0_path=r0_root/"01_SAVED_STATE_DIAGNOSTICS.json"
    t2_path=t2_root/"03_SLOT_READOUT_COMPARISON.json"
    r0,t2,lock=read(r0_path),read(t2_path),read(lock_path)
    assert r0["status"]=="COMPLETE_SAVED_STATE_ATTRIBUTION_ONLY"
    assert t2["status"]=="COMPLETE_REFIT_HEAD_ONLY"
    assert r0["input_sha256"]==t2["input_sha256"]==lock["original_input_sha256"]
    assert r0["lock_sha256"]==t2["lock_sha256"]==sha(lock_path)
    assert r0["ledger"]["actual_saved_state_replays"]==48 and r0["ledger"]["separate_loss_backward_calls"]==96
    assert t2["ledger"]["final_refit_head_families"]==18 and t2["ledger"]["CV_head_fit_families"]==270
    output.mkdir(parents=True,exist_ok=False)
    shards,selected,gradients,relations=[],[],[],[]
    for row in r0["cells"]:
        private=r0_root/row["file"];assert sha(private)==row["sha256"]
        cell=read(private)
        public={k:v for k,v in cell.items() if k!="training_curves"}
        public.update({"schema":"semantic_b0b7_p1r_r0_anonymous_state_cell_v1",
            "private_producer_json_sha256":row["sha256"],
            "unchanged_training_curves_reference_sha256":lock["original_aggregate_sha256"],
            "unchanged_training_curves_reference":"parent ordinary lossless12 JSON cells; no retraining here"})
        filename="01_R0_CELL_"+cell["arm"]+"_seed"+str(cell["seed"])+".json"
        write(output/filename,public)
        shards.append({"file":filename,"bytes":(output/filename).stat().st_size,"sha256":sha(output/filename),
                       "arm":cell["arm"],"seed":cell["seed"],"actual_states":len(cell["states"])})
        for state in cell["states"]:
            for parameter,g in state["gradients"]["parameter_groups"].items():
                gradients.append({"arm":cell["arm"],"seed":cell["seed"],"epoch":state["epoch"],
                                  "parameter":parameter,**g})
            for side in ("relations_validation_own_pair","relations_training_assigned_positive"):
                r=state[side]
                for program in r["per_program"]:
                    flat={"arm":cell["arm"],"seed":cell["seed"],"epoch":state["epoch"],"scope":side}
                    for name,value in program.items():
                        if isinstance(value,dict):
                            for key,number in value.items():flat[name+"_"+key]=number
                        else:flat[name]=value
                    relations.append(flat)
            if state["epoch"]==cell["historical_selected_epoch_unchanged"]:
                selected.append({"arm":cell["arm"],"seed":cell["seed"],"epoch":state["epoch"],
                    "checkpoint_sha256":state["checkpoint_sha256"],"gradients":state["gradients"],
                    "text_routes":state["text_routes"],"interventions":state["interventions"],
                    "known_supported_programs":state["relations_validation_own_pair"]["known_supported_programs"],
                    "normalized_unknown_mass":state["relations_validation_own_pair"]["normalized_unknown_mass"],
                    "validation_program":state["validation_program"],"slots_validation":state["slots_validation"]})
    index={**r0,"schema":"semantic_b0b7_p1r_r0_public_index_v1","cells":shards,
           "private_producer_index_sha256":sha(r0_path),"reporter_source_sha256":sha(__file__),
           "state_fields_preserved_exactly":True,"historical_training_curves_not_duplicated":True}
    write(output/"01_SAVED_STATE_DIAGNOSTICS.json",index)
    write(output/"02_TEXT_ROUTE_AND_INPUT_FAIRNESS.json",{
        "schema":"semantic_b0b7_p1r_r0_selected_attribution_v1","status":"COMPLETE_FROZEN_ATTRIBUTION_ONLY",
        "producer_index_sha256":sha(r0_path),"original_frozen_sources":r0["original_frozen_sources"],
        "input_sha256":r0["input_sha256"],"initialization":r0["initialization"],"rows":selected,
        "interventions_are_retraining":False,"CI_computed":False,"DSS_conclusion":False})
    table(output/"05_AUXILIARY_LOSS_GRADIENT_AND_WEIGHT.csv",gradients)
    table(output/"04_R0_RELATION_MASS_AND_CONTRIBUTION.csv",relations)
    points,programs=[],[]
    for row in t2["rows"]:
        metric=row["heldout_program"]
        points.append({"arm":row["arm"],"seed":row["seed"],"head":row["head"],
            "historical_selected_epoch":row["selected_epoch"],"alpha_training_CV":row["selected_alpha_training_only"],
            "features":row["ridge_input_features"],"nominal_coefficients_including_intercepts":row["nominal_coefficients_including_50_intercepts"],
            "training_numerical_rank":row["standardized_training_feature_rank"]["numerical_rank"],
            "training_energy_entropy_effective_rank":row["standardized_training_feature_rank"]["energy_entropy_effective_rank"],
            "macro_program_Pearson":metric["macro_program_pearson"],"macro_program_Spearman":metric["macro_program_spearman"],
            "patient_equal_masked_Huber":metric["patient_equal_masked_huber"],
            "alpha_is_upper_grid_boundary":row["selected_alpha_training_only"]==max(lock["ridge"]["alphas"])})
        for p,a in zip(metric["per_program"],row["prediction_amplitude"]):
            assert p["program_index"]==a["program_index"] and p["patients"]==a["patients"]
            programs.append({"arm":row["arm"],"seed":row["seed"],"head":row["head"],**p,
                             "prediction_SD":a["prediction_sd"],"target_SD":a["target_sd"],"SD_ratio":a["SD_ratio"]})
    table(output/"03_SLOT_READOUT_COMPARISON.csv",points)
    table(output/"03_SLOT_READOUT_PROGRAM_METRICS.csv",programs)
    write(output/"03_SLOT_READOUT_COMPARISON.json",{**t2,"producer_json_sha256":sha(t2_path),
        "reporter_source_sha256":sha(__file__),"all_alpha_choices_at_upper_prelocked_boundary":all(r["alpha_is_upper_grid_boundary"] for r in points)})
    ledger=[]
    for scope,values,seconds in (("R0_REPLAY_ONLY",r0["ledger"],r0["elapsed_seconds"]),
                                ("T2_REFIT_HEAD_ONLY",t2["ledger"],t2["elapsed_seconds"])):
        for name,value in values.items():ledger.append({"scope":scope,"quantity":name,"value":value})
        ledger.append({"scope":scope,"quantity":"elapsed_wall_seconds_excluding_input_qualification","value":seconds})
    table(output/"07_R0_T2_COMPUTE_LEDGER.csv",ledger)
    means=[]
    for arm in ("B3","B4"):
        for head in ("T2_mean64","T2_keep_slots","T2_capacity_control"):
            use=[x for x in points if x["arm"]==arm and x["head"]==head]
            means.append((arm,head,statistics.mean(x["macro_program_Pearson"] for x in use),
                          statistics.mean(x["macro_program_Spearman"] for x in use),
                          statistics.mean(x["patient_equal_masked_Huber"] for x in use)))
    lines=["# R0 保存状态归因与 T2 冻结读出", "",
        "R0 完成48份旧状态的精确前向回放及96次分项反传；新增神经拟合和优化器更新均为0。T2 完成270个训练内CV头、18个最终Ridge头及18次保存头回放。", "",
        "## 同一已训练表示下的读出比较", "",
        "以下为同一137名开发验证患者上的三seed描述性均值，不计算置信区间，不视为独立队列。", "",
        "| 编码器 | 读出 | Pearson | Spearman | Huber（低优） |", "|---|---|---:|---:|---:|"]
    lines += [f"| {a} | {h} | {p:.6f} | {s:.6f} | {e:.6f} |" for a,h,p,s,e in means]
    lines += ["", "所有18个Ridge头均在预锁网格上选择alpha=100的上界；本轮不据137人结果扩展搜索。mean64、slot512和二阶控制分别共同附加同checkpoint的clinical64，因此输入维数为128/576/576。", "",
        "## 机制解释边界", "",
        "- 诊断中的梯度均在checkpoint的eval模式计算，不重现历史dropout；分项范数和夹角必须分开判断，损失值比例不是梯度贡献。",
        "- B3仍收到真实TITAN32得分，因此B3不是无显式文本模型。置零或训练均值属于冻结通路干预，不能替代匹配重训。",
        "- known关系是历史候选边，unknown不是负边；关系权重质量不等于预测贡献，逐程序实际贡献另表保留。",
        "- 二阶控制只包含mean64的信息，匹配512维读出；它不能排除所有可能的MLP或正则解释。名义系数数不等于有效自由度。",
        "- 训练内3折仅重拟合读出头，encoder已在全部272名训练患者上进行过辅助训练，因此不能称完整pipeline的OOF。137人未参与新lambda选择。",
        "- 本轮没有读取或评估DSS/OS/PFI，也没有医学形态真值，不能将程序可解码性直接称为预后或语义机制成功。", ""]
    with (output/"01_SAVED_STATE_DIAGNOSTICS.md").open("x",encoding="utf-8") as f:f.write("\n".join(lines))
    files=[{"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    audit={"schema":"semantic_b0b7_p1r_r0_t2_report_build_v1","status":"COMPLETE_METADATA_PROJECTION_ONLY",
           "reporter_source_sha256":sha(__file__),"source_R0_index_sha256":sha(r0_path),
           "source_T2_sha256":sha(t2_path),"T2_lock_sha256":sha(lock_path),"files":files,
           "model_forwards":0,"optimizer_updates":0,"raw_patient_arrays_or_weights_loaded":False,
           "historical_state_fields_changed":False}
    write(output/"R0_T2_REPORT_BUILD_AUDIT.json",audit)
    return audit


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for key in ("r0-root","t2-root","lock","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();r=build(a.r0_root,a.t2_root,a.lock,a.output)
    print(json.dumps({"status":r["status"],"public_report_files":len(r["files"])+1}))
