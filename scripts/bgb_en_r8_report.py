"""R8 complete anonymous report; metadata only, no models, arrays or refitting."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_r6_report_target import (
    aggregate_arm, canonical_hash, read, require, sha256, validate_metrics,
    write_csv, write_json,
)

ARMS = ("R8-2", "R8-3", "R8-4", "R8-5", "R8-6", "R8-8a_RANDOM_SOURCE",
        "R8-8b_WRONG_PROGRAM", "R8-9_WRONG_PATIENT", "R8-10_NO_INTERACTION")
PARAMETERS = dict(zip(ARMS, (4, 8, 16, 8, 32, 32, 32, 32, 32)))
PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
REFS = ("R8-0", "R8-1")
FIELDS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
COMPARISONS = (("R8-2", "R8-1"), ("R8-3", "R8-2"), ("R8-4", "R8-1"),
    ("R8-5", "R8-3"), ("R8-6", "R8-1"), ("R8-6", "R8-4"),
    ("R8-6", "R8-8a_RANDOM_SOURCE"), ("R8-6", "R8-8b_WRONG_PROGRAM"),
    ("R8-6", "R8-9_WRONG_PATIENT"), ("R8-6", "R8-10_NO_INTERACTION"))
CREDITS = (("repair", "Harrell_den"), ("damage", "Harrell_den"),
           ("Uno_repair", "Uno_den"), ("Uno_damage", "Uno_den"))


def exact(rows, keys, expected, code):
    require(isinstance(rows, list) and len(rows) == len(expected), code)
    index = {tuple(r[k] for k in keys): r for r in rows}
    require(set(index) == set(expected) and len(index) == len(rows), code)
    return index


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def difference(a, b):
    return None if a is None or b is None else a - b


def paired(candidate, reference, summaries, rows, *, secondary=False):
    selected = [r for r in rows if r["candidate"] == candidate and r["reference"] == reference]
    exact(selected, ("partition", "seed"), {(p, s) for p in PARTITIONS for s in SEEDS},
          "R8_NINE_ACTUAL_PAIRED_CREDIT_ROWS_REQUIRED")
    out = dict(candidate=candidate, reference=reference,
        delta_Harrell=difference(summaries[candidate]["Harrell"], summaries[reference]["Harrell"]),
        delta_Uno60=difference(summaries[candidate]["Uno60"], summaries[reference]["Uno60"]), per_seed=[])
    field = "PFI_frozen_DSS_risk" if secondary else "candidate_vs_reference"
    for seed in SEEDS:
        source = [validate_metrics(r[field]) for r in selected if r["seed"] == seed]
        sa = next(r for r in summaries[candidate]["per_seed"] if r["seed"] == seed)
        sb = next(r for r in summaries[reference]["per_seed"] if r["seed"] == seed)
        r = dict(seed=seed, delta_Harrell=difference(sa["Harrell"], sb["Harrell"]),
                 delta_Uno60=difference(sa["Uno60"], sb["Uno60"]))
        for credit, den in CREDITS:
            require(all(m.get(credit) is None if m[den] == 0 else
                finite(m.get(credit)) and 0 <= m[credit] <= 1 for m in source), "R8_INVALID_PAIR_CREDIT")
            denominator = math.fsum(m[den] for m in source)
            r[credit] = math.fsum(m[credit] * m[den] for m in source if m[den] > 0) / denominator if denominator else None
        for name, repair, damage in (("Harrell", "repair", "damage"), ("Uno60", "Uno_repair", "Uno_damage")):
            d = r["delta_" + name]
            require(d is None if r[repair] is None else d is not None and
                abs(r[repair] - r[damage] - d) <= 1e-10, "R8_DIRECT_PAIR_CREDIT_AND_METRIC_DISAGREE")
        out["per_seed"].append(r)
    for credit, _ in CREDITS:
        v = [r[credit] for r in out["per_seed"]]
        out[credit] = statistics.mean(v) if all(x is not None for x in v) else None
    h, u = out["delta_Harrell"], out["delta_Uno60"]
    out.update(joint_directional_support=h is not None and u is not None and h > 0 and u > 0,
        undefined_metric=h is None or u is None,
        small_delta_is_weak_evidence=h is not None and u is not None and (abs(h) < .003 or abs(u) < .003),
        positive_nominal_seed_repetitions=sum(r["delta_Harrell"] is not None and r["delta_Uno60"] is not None
            and r["delta_Harrell"] > 0 and r["delta_Uno60"] > 0 for r in out["per_seed"]),
        nominal_seed_repetitions_are_independent_random_evidence=False,
        repair_damage_are_normalized_concordance_credits_not_patient_counts=True)
    return out


def build(private, config_path, output):
    private, config_path, output = Path(private), Path(config_path), Path(output)
    config, lock = read(config_path), read(private / "R8_PROTOCOL_LOCK.json")
    require(tuple(config["trainable_arms"]) == ARMS and config["primary_candidate"] == "R8-6"
        and config["gamma"] == .1 and config["ridge_lambda"] == 1., "R8_FIXED_CONFIG_CHANGED")
    require(lock["schema"] == "bgb_en_r8_frozen_evidence_protocol_lock_v1"
        and lock["status"] == "LOCKED_BEFORE_R8_TARGET_FITS"
        and lock["contract_sha256"] == canonical_hash({k:v for k,v in lock.items() if k != "contract_sha256"})
        and lock["config_sha256"] == sha256(config_path) and lock["planned_convex_fits"] == 81
        and lock["planned_neural_fits"] == lock["planned_neural_updates"] == lock["source_or_teacher_refits"] == 0
        and lock["primary_candidate"] == "R8-6", "R8_IMMUTABLE_CONTRACT_CHANGED")
    require(bool(lock["sources"]), "R8_SOURCE_BINDINGS_REQUIRED")
    for name, digest in lock["sources"].items():
        p = ROOT / name
        require(not p.is_symlink() and p.resolve().is_relative_to(ROOT) and sha256(p) == digest,
                "R8_LOCKED_SOURCE_BYTES_CHANGED")
    qualification = read(private / "R8_FEATURE_QUALIFICATION.json")
    require(sha256(private / "R8_FEATURE_QUALIFICATION.json") == lock["qualification_sha256"]
        and qualification["status"] == "PASS_R8_FROZEN_SOURCE_SCOPED_409_FEATURE_QUALIFICATION"
        and qualification["config_sha256"] == sha256(config_path)
        and qualification["feature_families"] == 6
        and qualification["source_neural_fits"] == qualification["target_neural_fits"] == qualification["target_convex_fits"] == 0,
        "R8_REAL_FEATURE_QUALIFICATION_REQUIRED")
    feature_rows = exact(qualification["rows"], ("partition", "source_kind"),
        {(p,k) for p in PARTITIONS for k in ("MATCHED","RANDOM_NO_ST")}, "R8_EXACT_SIX_QUALIFIED_FEATURE_FAMILIES_REQUIRED")
    for (part,kind), original in feature_rows.items():
        name = f"p{part}_{kind}"; folder = private / "features" / name
        ready = read(folder / "READY.json")
        require(original["receipt_sha256"] == sha256(folder / "READY.json") == lock["feature_receipt_sha256"][name]
            and ready["status"] == "COMPLETE_R8_SCOPED_FROZEN_FEATURES"
            and ready["artifact_sha256"] == sha256(folder / "features.private.pt")
            and ready["config_sha256"] == sha256(config_path)
            and ready["source_state_sha256"] == lock["source_full_state_sha256"][kind]
            and ready["scoped_rows"] == 409 and ready["outer_test_rows_consumed"] == 0
            and ready["old_container_rows"] == 512 and ready["old_container_physically_loaded"] is True
            and ready["source_visual_forward_uses_RNA"] is False
            and ready["source_spot_scaler_applied_to_bulk"] is False
            and ready["source_fit_count"] == ready["teacher_fit_count"] == ready["target_supervised_fit_count"] == 0,
            "R8_QUALIFIED_FEATURE_ARTIFACT_OR_SCOPE_CHANGED")
    replay_path = private / "REPLAY_AUDIT.json"; replay = read(replay_path)
    require(replay["status"] == "PASS_COMPLETE_R8_81_SAVED_RISK_REPLAYS"
        and replay["contract_sha256"] == lock["contract_sha256"]
        and replay["actual_saved_models"] == replay["passed_units"] == 81 and replay["failed_units"] == 0
        and replay["new_optimizer_fits"] == replay["source_or_teacher_refits"] == 0,
        "R8_COMPLETE_81_ACTUAL_REPLAYS_REQUIRED")
    units = exact(replay["units"], ("arm", "partition", "seed"),
        {(a,p,s) for a in ARMS for p in PARTITIONS for s in SEEDS}, "R8_EXACT_81_REPLAY_UNITS_REQUIRED")
    refs = exact(replay["zero_fit_reference_replays"], ("arm", "partition", "seed"),
        {(a,p,s) for a in REFS for p in PARTITIONS for s in SEEDS}, "R8_EXACT_18_REFERENCE_REPLAYS_REQUIRED")
    seed_checks = exact(replay["deterministic_seed_checks"], ("arm", "partition"),
        {(a,p) for a in ARMS for p in PARTITIONS}, "R8_EXACT_27_DETERMINISM_CHECKS_REQUIRED")
    direct = replay["paired_retrained_controls"]
    exact(direct, ("candidate", "reference", "partition", "seed"),
        {(a,b,p,s) for a,b in COMPARISONS for p in PARTITIONS for s in SEEDS}, "R8_EXACT_90_DIRECT_PAIRS_REQUIRED")
    require(all(r["direct_actual_saved_risk_pair_credits"] is True and
                r["bootstrap_CI_or_pvalue_computed"] is False for r in direct), "R8_ACTUAL_PAIRS_NOT_DECLARED")
    metrics, secondary, fold_rows, cost_rows, diagnostic_parts = {}, {}, [], [], []
    diagnostic_payloads = {}
    expected_names = {f"{a}_p{p}_seed{s}" for a,p,s in units}
    require({p.parent.name for p in (private / "cells").glob("*/COMPLETE.json")} == expected_names
        and not list((private / "cells").glob("*/ERROR_RECEIPT.json")), "R8_EXACT_COMPLETE_FIT_MATRIX_REQUIRED")
    output.mkdir(parents=True, exist_ok=True)
    outputs = ["R8_RESULTS.json", "R8_RESULTS.md", "R8_DECISION.json", "R8_DECISION.md", "R8_REAL_TRAINING_LEDGER.csv",
               "R8_FOLD_METRICS.csv", "R8_PFI_FOLD_METRICS.csv", "R8_PAIRED_COMPARISONS.csv"]
    outputs += [f"R8_DIAGNOSTICS_{a}.json" for a in ARMS]
    require(not any((output / n).exists() for n in outputs), "R8_REPORT_OUTPUT_EXISTS_NO_OVERWRITE")
    for arm in ARMS:
        diagnostics = []
        for part in PARTITIONS:
            for seed in SEEDS:
                key = (arm,part,seed); row = units[key]; directory = private / "cells" / f"{arm}_p{part}_seed{seed}"
                receipt_path = directory / "COMPLETE.json"; cell = read(receipt_path)
                require(row["status"] == "PASS_ACTUAL_R8_SAVED_COEFFICIENT_AND_FEATURE_REPLAY"
                    and row["actual_saved_risk_forward_replayed"] is True
                    and finite(row["maximum_risk_error"]) and 0 <= row["maximum_risk_error"] <= 2e-10
                    and finite(row["maximum_metric_or_control_error"]) and 0 <= row["maximum_metric_or_control_error"] <= 2e-10
                    and row["changed_pair_credits"] == row["new_optimizer_fits"] == row["new_neural_updates"] == 0
                    and row["completion_sha256"] == sha256(receipt_path)
                    and row["artifact_sha256"] == cell["artifact_sha256"] == sha256(directory / "model.private.pt"),
                    "R8_ACTUAL_REPLAY_OR_ARTIFACT_BINDING_CHANGED")
                require(cell["status"] == "COMPLETE_R8_CONVEX_FIT"
                    and cell["contract"]["contract_sha256"] == lock["contract_sha256"]
                    and (cell["arm"],cell["partition"],cell["seed"]) == key
                    and cell["actual_convex_fits"] == 1 and cell["actual_neural_fits"] == cell["neural_updates"] == 0
                    and cell["source_refits"] == cell["teacher_refits"] == 0
                    and cell["registered_coefficients"] == PARAMETERS[arm], "R8_MODEL_BUDGET_OR_CONTRACT_CHANGED")
                feature = f"p{part}_{'RANDOM_NO_ST' if arm=='R8-8a_RANDOM_SOURCE' else 'MATCHED'}"
                require(cell["feature_receipt_sha256"] == row["feature_receipt_sha256"] == lock["feature_receipt_sha256"][feature]
                    == sha256(private / "features" / feature / "READY.json"), "R8_FROZEN_FEATURE_BINDING_CHANGED")
                require(cell["contract"]["source_state_sha256"] == lock["source_full_state_sha256"][
                    "RANDOM_NO_ST" if arm=="R8-8a_RANDOM_SOURCE" else "MATCHED"], "R8_FIT_SOURCE_VERSION_CHANGED")
                fit = cell["actual_fit"]; cert = row["convex_KKT"]
                require(fit["status"] == "CONVERGED_KKT_AND_TRAIN_SD_PROTECTION"
                    and fit["registered_coefficients"] == PARAMETERS[arm] and fit["learned_intercepts"] == 0
                    and fit["gamma"] == cert["gamma"] == .1 and fit["ridge_lambda"] == cert["ridge_lambda"] == 1.
                    and fit["training_riskset_complete"] is True and fit["validation_or_test_labels_used"] is False
                    and fit["hyperparameters_selected_from_metrics"] is False
                    and cert["status"] == "PASS_ACTUAL_SAVED_COEFFICIENT_KKT_AND_SD_REPLAY"
                    and cert["actual_saved_coefficient_forward_replayed"] is True and cert["new_optimizer_fits"] == 0
                    and finite(cert["stationarity_max"]) and 0 <= cert["stationarity_max"] <= 5e-6
                    and finite(cert["complementarity"]) and 0 <= cert["complementarity"] <= 5e-6
                    and finite(cert["constraint_slack"]) and cert["constraint_slack"] >= -1e-8
                    and finite(cert["train_residual_SD_ratio"]) and 0 <= cert["train_residual_SD_ratio"] <= .1+1e-8,
                    "R8_ACTUAL_CONVEX_KKT_OR_TRAIN_SD_BOUND_INVALID")
                metrics[key] = validate_metrics(cell["metrics"])
                secondary[key] = validate_metrics(cell["PFI_frozen_DSS_risk"])
                validate_metrics(cell["train_metrics"])
                for control in cell["frozen_controls"].values():
                    require(control["frozen_not_retrained"] is True
                        and control["recipient_gene_values_masks_RNA512_offset_and_images_unchanged"] is True,
                        "R8_FROZEN_CONTROL_NOT_ISOLATED")
                    validate_metrics(control["metrics_vs_anchor"])
                    validate_metrics(control["full_relative_to_control"])
                cost_rows.append(dict(arm=arm,partition=part,seed=seed,convex_fits=1,neural_fits=0,neural_updates=0,
                    coefficients=PARAMETERS[arm],iterations=fit["iterations"],function_evaluations=fit["function_evaluations"],
                    elapsed_seconds=cell["elapsed_seconds"],KKT_stationarity=cert["stationarity_max"],
                    train_residual_SD_ratio=cert["train_residual_SD_ratio"],feature_design_rank=fit["train_design_rank"]))
                diagnostics.append(dict(arm=arm,partition=part,seed=seed,
                    completion_sha256=sha256(receipt_path),artifact_sha256=cell["artifact_sha256"],
                    train_metrics=cell["train_metrics"],risk_diagnostics=cell["risk_diagnostics"],
                    frozen_controls=cell["frozen_controls"],convex_certificate=cert,
                    source_version=cell["source_version"],source_state_sha256=cell["contract"]["source_state_sha256"],
                    feature_receipt_sha256=cell["feature_receipt_sha256"]))
        path = output / f"R8_DIAGNOSTICS_{arm}.json"
        import json
        blob = json.dumps(dict(schema="r8_anonymous_diagnostic_shard_v1",rows=diagnostics),separators=(",",":"),allow_nan=False).encode()
        require(len(blob) <= 2*1024**2,"R8_ANONYMOUS_DIAGNOSTIC_SHARD_TOO_LARGE")
        diagnostic_payloads[path.name] = blob
        diagnostic_parts.append(dict(file=path.name,rows=9,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    for key, replayed in refs.items():
        arm,part,seed = key; path = private / "references" / f"{arm}_p{part}_seed{seed}.json"; v = read(path)
        require(v["status"] == "COMPLETE_R8_ZERO_FIT_REFERENCE" and v["contract_sha256"] == lock["contract_sha256"]
            and v["actual_convex_fits"] == v["actual_neural_fits"] == v["actual_new_teacher_fits"] == 0
            and replayed["actual_baseline_values_replayed"] is True and replayed["reference_sha256"] == sha256(path)
            and finite(replayed["maximum_metric_error"]) and 0 <= replayed["maximum_metric_error"] <= 2e-10
            and replayed["new_optimizer_fits"] == 0,
            "R8_ZERO_FIT_BASELINE_REPLAY_CHANGED")
        metrics[key], secondary[key] = validate_metrics(v["metrics"]), validate_metrics(v["PFI_frozen_DSS_risk"])
    for key in metrics:
        a,p,s = key
        fold_rows.append(dict(arm=a,partition=p,seed=s,**{k:metrics[key][k] for k in FIELDS}))
    summaries = {a:aggregate_arm(a,metrics) for a in REFS+ARMS}
    secondary_summaries = {a:aggregate_arm(a,secondary) for a in REFS+ARMS}
    for record in direct:
        for field, table in (("candidate_vs_reference",metrics),("PFI_frozen_DSS_risk",secondary)):
            a = table[(record["candidate"],record["partition"],record["seed"])]
            b = table[(record["reference"],record["partition"],record["seed"])]
            m = validate_metrics(record[field])
            for key in FIELDS:
                require(m[key] is None and a[key] is None or m[key] is not None and a[key] is not None
                    and abs(m[key]-a[key]) <= 1e-10, "R8_DIRECT_CANDIDATE_PARTITION_METRIC_CHANGED")
            for metric,den,repair,damage in (("Harrell","Harrell_den","repair","damage"),
                                            ("Uno60","Uno_den","Uno_repair","Uno_damage")):
                require(abs(a[den]-b[den]) <= 1e-10, "R8_PAIRED_PARTITION_DENOMINATORS_DIFFER")
                d = difference(a[metric],b[metric])
                require(m.get(repair) is None and m.get(damage) is None if d is None else
                    finite(m.get(repair)) and finite(m.get(damage)) and 0 <= m[repair] <= 1
                    and 0 <= m[damage] <= 1 and abs(m[repair]-m[damage]-d) <= 1e-10,
                    "R8_DIRECT_PARTITION_REPAIR_DAMAGE_CHANGED")
    paired_rows = [paired(a,b,summaries,direct) for a,b in COMPARISONS]
    pfi_pairs = [paired(a,b,secondary_summaries,direct,secondary=True) for a,b in COMPARISONS]
    gates = [r for r in paired_rows if r["candidate"] == "R8-6"]
    require(len(gates) == 6,"R8_SIX_FIXED_MAIN_CANDIDATE_GATES_REQUIRED")
    defined = not any(r["undefined_metric"] for r in gates)
    positive = defined and all(r["joint_directional_support"] for r in gates)
    identical = all(v["bitwise_identical_predictions"] is True and v["maximum_absolute_difference"] == 0 for v in seed_checks.values())
    decision = dict(status="INCONCLUSIVE_UNDEFINED_METRIC" if not defined else
        "DIRECTIONAL_DEVELOPMENT_INCREMENT_DETERMINISTIC_REPETITIONS" if positive else "CURRENT_R8_6_NOT_SUPPORTED",
        primary_candidate="R8-6", all_fixed_primary_comparisons=gates,
        numerical_training_problems=27,actual_convex_fits=81,nominal_seed_repetitions=3,
        all_nominal_seed_predictions_bitwise_identical=identical,independent_random_seed_evidence=False,
        PFI_used_for_selection=False,new_outer_results="NOT_RUN",new_SPGEL_source_used=False,
        R8_DSS_selects_source_model=False,CI_or_pvalues_computed=False,
        R8_9_bilinear_control_confirms_R8_5_patch_routing=False,
        all_medical_ideas_rejected=False,causal_biology_or_external_generalization_proven=False)
    result = dict(schema="bgb_en_r8_complete_report_v1",status="COMPLETE_81_CONVEX_FITS_AND_ACTUAL_SAVED_MODEL_REPLAYS",
        contract_sha256=lock["contract_sha256"],config_sha256=sha256(config_path),qualification_sha256=lock["qualification_sha256"],
        replay_sha256=sha256(replay_path),primary_DSS=summaries,secondary_PFI_frozen_DSS_risk=secondary_summaries,
        paired_comparisons=paired_rows,secondary_paired_comparisons=pfi_pairs,decision=decision,
        budget=dict(actual_convex_fits=81,neural_fits=0,neural_updates=0,source_teacher_beta_refits=0,
            deterministic_unique_training_problems=27,actual_saved_model_replays=81,zero_fit_references=18,unsupervised_PCA_fits=60),
        diagnostic_index=diagnostic_parts,legacy_global_container_loaded=True,
        new_features_fit_metrics_only_scoped_outer1_train409=True,outer_performance_evaluation=False,
        SD_bound_applies_to_training_only=True,rank2_is_2_by_2_dimension_bound_not_extra_low_rank_regularization=True)
    for name, blob in diagnostic_payloads.items():
        with (output/name).open("xb") as stream: stream.write(blob)
    write_csv(output/"R8_REAL_TRAINING_LEDGER.csv",cost_rows,list(cost_rows[0]))
    write_csv(output/"R8_FOLD_METRICS.csv",fold_rows,["arm","partition","seed",*FIELDS])
    pfi_fold=[dict(arm=a,partition=p,seed=s,**{k:m[k] for k in FIELDS}) for (a,p,s),m in secondary.items()]
    write_csv(output/"R8_PFI_FOLD_METRICS.csv",pfi_fold,["arm","partition","seed",*FIELDS])
    flat=[{k:v for k,v in r.items() if k!="per_seed"} for r in paired_rows]
    write_csv(output/"R8_PAIRED_COMPARISONS.csv",flat,list(flat[0]))
    fmt=lambda v: "NA" if v is None else f"{v:.9f}"
    text=["# R8 冻结R6证据的低容量迁移结果","", "状态：81次凸拟合及81次保存模型实际回放完成；无新神经/源/教师拟合。",
        "27个数值训练问题各按三名义seed重复；确定性重复不作为独立随机稳健性证据。", "",
        "| 模型 | DSS Harrell | DSS Uno60 | 系数 |", "|---|---:|---:|---:|"]
    text += [f"| {a} | {fmt(summaries[a]['Harrell'])} | {fmt(summaries[a]['Uno60'])} | {PARAMETERS.get(a,'旧引用')} |" for a in REFS+ARMS]
    text += ["", "## 固定比较", "", "| 候选−对照 | ΔHarrell | ΔUno60 |", "|---|---:|---:|"]
    text += [f"| {r['candidate']}−{r['reference']} | {fmt(r['delta_Harrell'])} | {fmt(r['delta_Uno60'])} |" for r in paired_rows]
    text += ["",f"固定主候选R8-6：`{decision['status']}`。不依此自动启动新外层。",
        "保护仅为训练残差SD上界，不保证验证SD或患者对排序；共同λ/参数预算不等于同函数先验。",
        "R8-6是每程序PCA2空间完整双线性；R8-5才是冻结RNA条件Patch读取，其wrong query仅冻结诊断，非真实重训。",
        "源采用现成R6_S7_locked，spot与bulk共同Z形式不证明域对齐。CNA/text/新WSI特征及新SPGEL源均未启用。",
        "旧512行容器和全局终点QC曾物理加载；新特征、拟合、选择与评价仅消费outer1训练侧409人，不声称全局容器未读。",
        "Harrell/Uno分区内累加num/den后按名义seed平均，PFI只评估冻结DSS风险；无跨分区风险排序或CI/p。",
        "优势是冻结强主效应、低自由度、实际KKT/SD复核及真实错患者/错程序控制；风险是旧开发队列复用、有限事件和源—目标域差。",
        "本结果不是独立外部验证、组织学真值或医学因果证明；负结果只限定此预锁配方。"]
    (output/"R8_RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    (output/"R8_DECISION.md").write_text("# R8决策\n\n"+decision["status"]+"\n\n固定主候选为R8-6；六项关键比较须Harrell和Uno60均正。\n"
        "名义seed重复不作为独立随机证据；新外层NOT_RUN，DSS不会回传选择SPGEL源模型。\n",encoding="utf-8")
    write_json(output/"R8_DECISION.json",decision)
    write_json(output/"R8_RESULTS.json",result)  # Complete marker is written after every other report artifact.
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--private",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();r=build(args.private,args.config,args.output)
    print(__import__("json").dumps({"status":r["status"],"decision":r["decision"]["status"],"new_model_calls":0}))


if __name__ == "__main__": main()
