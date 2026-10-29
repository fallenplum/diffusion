"""Independent arithmetic and artifact review of the four fixed HEST cells.

No model is deserialized, no image/expression/prediction array is read, and no
optimizer is called. Actual saved-model replay is a separate upstream receipt.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def average(values):
    return math.fsum(values)/len(values) if all(v is not None for v in values) else None


def deviation(values):
    mean = average(values)
    return math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)) if mean is not None else None


def rounded(value):
    return round(value, 4) if value is not None else None


def review(private, report, output):
    private, report, output = Path(private), Path(report), Path(output)
    result, lock = read(report), read(private/"HEST_RUN_LOCK.json")
    replay_path = private/"REPLAY_AUDIT.json"
    replay = read(replay_path)
    require(result["status"] == "COMPLETE_FOUR_OFFICIAL_RIDGE_FITS_AND_ACTUAL_REPLAYS"
            and replay["status"] == "PASS_FOUR_ACTUAL_OFFICIAL_RIDGE_MODEL_REPLAYS"
            and replay["passed_models"] == 4 and replay["failed_models"] == 0,
            "ALL_FOUR_ACTUAL_REPLAYS_REQUIRED")
    contract = lock["internal_contract_sha256"]
    require(result["internal_contract_sha256"] == replay["internal_contract_sha256"] == contract
            and result["source_run_lock_file_sha256"] == sha(private/"HEST_RUN_LOCK.json"),
            "LOCK_OR_REPORT_BINDING_CHANGED")
    evidence = {"public_report": sha(report), "run_lock": sha(private/"HEST_RUN_LOCK.json"),
                "actual_model_replay": sha(replay_path)}
    comparisons, errors, fold_rows, gene_rows, task_summary = 0, [], [], [], {}

    def same(a, b):
        nonlocal comparisons
        comparisons += 1
        require((a is None) == (b is None), "UNDEFINED_METRIC_HANDLING_CHANGED")
        if a is not None:
            require(math.isfinite(a) and math.isfinite(b), "NONFINITE_PUBLIC_METRIC")
            errors.append(abs(a-b))

    require({(r["dataset"], r["fold"], r["seed"]) for r in replay["units"]}
            == {(d, f, 1) for d in ("COAD", "READ") for f in (0, 1)}, "REPLAY_IDENTITIES_CHANGED")
    for dataset in ("COAD", "READ"):
        cells, raw_means, rounded_means = [], [], []
        for fold in (0, 1):
            folder = private/"cells"/f"{dataset}_fold{fold}_seed1"
            completion = folder/"COMPLETE.json"
            cell = read(completion)
            row = next(r for r in replay["units"] if r["dataset"] == dataset and r["fold"] == fold)
            require(cell["status"] == "COMPLETE_ONE_OFFICIAL_RIDGE_FIT"
                    and cell["internal_contract_sha256"] == contract
                    and row["status"] == "PASS" and row["actual_saved_model_loaded"] is True
                    and row["actual_forward_replayed"] is True
                    and row["new_Ridge_fits"] == row["new_scaler_fits"] == row["new_PCA_fits"] == 0
                    and row["maximum_prediction_error"] <= 1e-6
                    and row["COMPLETE_sha256"] == sha(completion)
                    and cell["model_sha256"] == row["model_sha256"] == sha(folder/"model.private.joblib")
                    and cell["predictions_sha256"] == sha(folder/"predictions.private.npz"),
                    "ACTUAL_SAVED_MODEL_OR_PREDICTION_BINDING_CHANGED")
            require(cell["new_Ridge_fits"] == cell["new_scaler_fits"] == cell["new_PCA_fits"] == 1
                    and cell["neural_fits"] == 0 and cell["registered_Ridge_coefficients"] == 12800
                    and cell["test_RNA_in_predict"] is False and cell["source_pretrained_weights_loaded"] is False,
                    "OFFICIAL_FIT_OR_INPUT_BUDGET_CHANGED")
            full, split = cell["full_precision_metrics"], cell["official_rounded4_split_metrics"]
            values = full["pearson_corrs"]
            require(len(values) == len(split["pearson_corrs"]) == 50
                    and [r["name"] for r in values] == cell["genes"]
                    and len(set(cell["genes"])) == 50, "EXACT_GENE_ORDER_REQUIRED")
            mean = average([r["pearson_corr"] for r in values])
            same(mean, full["pearson_mean"])
            same(deviation([r["pearson_corr"] for r in values]), full["pearson_std"])
            same(rounded(full["pearson_mean"]), split["pearson_mean"])
            raw_means.append(full["pearson_mean"])
            rounded_means.append(split["pearson_mean"])
            for g, item in enumerate(values):
                require(split["pearson_corrs"][g]["name"] == item["name"], "ROUNDED_GENE_ORDER_CHANGED")
                same(rounded(item["pearson_corr"]), split["pearson_corrs"][g]["pearson_corr"])
                gene_rows.append(dict(dataset=dataset, fold=fold, seed=1, gene=item["name"],
                    Pearson_full_precision=item["pearson_corr"],
                    Pearson_official_round4=split["pearson_corrs"][g]["pearson_corr"]))
            fold_rows.append(dict(dataset=dataset, fold=fold, seed=1, train_spots=cell["train_rows"],
                test_spots=cell["test_rows"], Pearson_mean_full_precision=full["pearson_mean"],
                Pearson_mean_official_round4=split["pearson_mean"],
                defined_genes=sum(v["pearson_corr"] is not None for v in values),
                total_genes=50, coefficients=cell["registered_Ridge_coefficients"],
                actual_PCA_solver=cell["actual_PCA_solver"], elapsed_seconds=cell["elapsed_seconds"]))
            evidence[f"{dataset}_fold{fold}_COMPLETE"] = sha(completion)
            cells.append(cell)
        public = result["tasks"][dataset]["official_rounded4_fold_aggregate"]
        same(rounded(average(rounded_means)), public["pearson_mean"])
        same(rounded(deviation(rounded_means)), public["pearson_std"])
        require(public["mean_per_split"] == rounded_means, "OFFICIAL_SPLIT_RETURN_VALUES_CHANGED")
        public_genes = {r["name"]:r for r in public["pearson_corrs"]}
        require(set(public_genes) == set(cells[0]["genes"]), "GENE_AGGREGATION_SUPPORT_CHANGED")
        for g, gene in enumerate(cells[0]["genes"]):
            values = [c["official_rounded4_split_metrics"]["pearson_corrs"][g]["pearson_corr"] for c in cells]
            same(rounded(average(values)), public_genes[gene]["mean"])
            same(rounded(deviation(values)), public_genes[gene]["std"])
        task_summary[dataset] = dict(official_round4_Pearson=public["pearson_mean"],
            full_precision_fold_means=raw_means, identity_status=result["tasks"][dataset]["donor_separation_status"])
    require(max(errors, default=0) <= 2e-10, "INDEPENDENT_OFFICIAL_ARITHMETIC_MISMATCH")
    audit = dict(schema="hest_four_ridge_independent_arithmetic_v1",
        status="PASS_FOUR_OFFICIAL_CELLS_GENE_AND_ROUND4_AGGREGATION_AND_ARTIFACT_BINDINGS",
        internal_contract_sha256=contract, actual_Ridge_fits_in_research=4,
        actual_model_replays_in_research=4, this_review_is_not_an_additional_model_replay=True,
        reviewer_model_loads=0, reviewer_new_fits=0, metadata_scalar_checks=comparisons,
        maximum_absolute_arithmetic_error=max(errors, default=0), fold_rows=4, gene_rows=200,
        original_official_rounding_order_preserved=True, tasks=task_summary, evidence_sha256=evidence,
        script_sha256=sha(__file__))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("HEST_OFFICIAL_RIDGE_FOLD_METRICS_20261029.csv",fold_rows),
                       ("HEST_OFFICIAL_RIDGE_GENE_METRICS_20261029.csv",gene_rows)):
        with (output/name).open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/"HEST_OFFICIAL_RIDGE_INDEPENDENT_REVIEW_20261029.json").open("x",encoding="utf-8") as f:
        json.dump(audit,f,indent=2,allow_nan=False);f.write("\n")
    return audit


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--private",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=review(a.private,a.report,a.output)
    print(json.dumps({k:r[k] for k in ("status","metadata_scalar_checks","maximum_absolute_arithmetic_error")}))


if __name__=="__main__":main()
