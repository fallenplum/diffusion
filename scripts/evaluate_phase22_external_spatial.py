"""Validation-only A selection and frozen external-ST evaluation.

The test entry point requires the locked validation selection receipt and only
evaluates its selected arm. It cannot rank all candidate arms on ST-test.
Spot predictions, source identifiers and dictionary-oracle coefficients remain
private. The aggregate writer contains no identifiers or patient/spot arrays.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_phase22_external_spatial import (
    ARMS, SEEDS, SCHEDULE, align_genes, file_sha256, infer_arrays,
    load_dictionary, load_frozen_model, oracle_coefficients, private_output,
    qualification_rows, read_json, source_hashes, write_json,
)


def fractional_top_mass(values, fraction=.2):
    """Tie-aware top fraction, with exactly ceil(fraction*N) selected mass."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    k = max(1, int(np.ceil(n * fraction)))
    cutoff = np.partition(values, n - k)[n - k]
    above, equal = values > cutoff, values == cutoff
    result = above.astype(np.float64)
    result[equal] = (k - above.sum()) / equal.sum()
    return result, k


def section_metrics(mu, counts, measured):
    """Gene-wise within-section metrics on measured-gene composition.

    Genes with constant truth have undefined spatial correlation and are
    counted/excluded. Constant prediction on a variable gene scores zero,
    rather than disappearing from the average. Top-20% ties are fractional.
    """
    mu, counts, measured = np.asarray(mu, dtype=np.float64), np.asarray(counts, dtype=np.float64), np.asarray(measured, dtype=bool)
    if mu.shape != counts.shape or measured.shape != (counts.shape[1],):
        raise ValueError("metric composition/count/mask dimensions disagree")
    observed = counts[:, measured]
    positive = observed.sum(1) > 0
    if positive.sum() < 3:
        raise ValueError("section has fewer than three count-observed spots")
    truth = observed[positive]
    truth = truth / truth.sum(1, keepdims=True)
    prediction = mu[positive][:, measured]
    if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction.sum(1) <= 0).any():
        raise ValueError("nonfinite or nonpositive measured prediction")
    prediction = prediction / prediction.sum(1, keepdims=True)
    corrs, precision = [], []
    constant_prediction = 0
    for gene in range(truth.shape[1]):
        y, p = truth[:, gene], prediction[:, gene]
        if np.ptp(y) == 0:
            continue
        if np.ptp(p) == 0:
            corrs.append(0.)
            constant_prediction += 1
        else:
            rho = float(spearmanr(y, p).statistic)
            if not np.isfinite(rho):
                raise ValueError("undefined correlation for a nonconstant spatial gene")
            corrs.append(rho)
        yt, k = fractional_top_mass(y)
        pt, _ = fractional_top_mass(p)
        precision.append(float(np.dot(yt, pt) / k))
    if not corrs:
        raise ValueError("section has no variable measured spatial genes")
    centered_truth = truth - truth.mean(0, keepdims=True)
    centered_prediction = prediction - prediction.mean(0, keepdims=True)
    point = -np.sum(truth * np.log(np.maximum(prediction, np.finfo(float).tiny)), axis=1).mean()
    return dict(gene_spearman=float(np.mean(corrs)),
        spatially_centered_MSE=float(np.square(centered_prediction - centered_truth).mean()),
        top20_precision=float(np.mean(precision)), mean_composition_NLL=float(point),
        eligible_spatial_genes=int(len(corrs)), excluded_constant_truth_genes=int(truth.shape[1] - len(corrs)),
        constant_prediction_genes=int(constant_prediction), positive_count_spots=int(positive.sum()))


def candidate_selection(output_root, split_lock, qualification):
    out = private_output(output_root)
    lock_sha = file_sha256(split_lock)
    marker_sha = file_sha256(qualification)
    _, _, dictionary_receipt, dictionary_sha = load_dictionary(out)
    scores, models, receipts = {}, {}, {}
    for arm in ARMS:
        scores[arm] = []
        for seed in SEEDS:
            folder = out / "fits" / arm / ("seed%03d" % seed)
            done_path, model_path = folder / "DONE_PRIVATE.json", folder / "model.pt"
            done = read_json(done_path)
            if done.get("status") != "COMPLETE" or done.get("model_sha256") != file_sha256(model_path):
                raise ValueError("arm selection requires nine completed, hash-verified cells")
            identity = done["identity"]
            if identity["split_lock_sha256"] != lock_sha or identity["data_qualification_sha256"] != marker_sha or identity["dictionary_npz_sha256"] != dictionary_sha or identity["source_hashes"] != source_hashes():
                raise ValueError("formal cells do not share the locked source/data/dictionary protocol")
            if done["schedule"] != SCHEDULE or done["test_labels_or_counts_read"]:
                raise ValueError("selection requires unchanged budget and no test access")
            scores[arm].append(float(done["validation"]["gene_spearman"]))
            key = "%s_seed%03d" % (arm, seed)
            models[key] = {"model_path": str(model_path), "model_sha256": done["model_sha256"],
                           "done_sha256": file_sha256(done_path), "selected_epoch": done["selected_epoch"]}
            receipts[key] = done["validation"]
    delta10 = np.asarray(scores["A1"]) - np.asarray(scores["A0"])
    delta21 = np.asarray(scores["A2"]) - np.asarray(scores["A1"])
    retain1, retain2 = bool((delta10 > 0).all()), bool((delta21 > 0).all())
    selected = "A2" if retain1 and retain2 else "A1" if retain1 else "A0"
    # A finite candidate gate, not a significance claim. Only two validation
    # donors are available in CRC16, and foundation pretraining overlap is unknown.
    gate = bool((np.asarray(scores[selected]) > 0).all())
    record = dict(schema_version="phase22_validation_selection_v1", status="SELECTED_ON_VALIDATION_ONLY",
        selected_arm=selected, selected_seeds=list(SEEDS), A_gate="PASS_CANDIDATE" if gate else "NOT_SUPPORTED_ON_VALIDATION",
        validation_candidate_gate=bool(gate), candidate_only=True, authorizes_TCGA_survival=False,
        gate_metric="donor_equal_within_section_gene_Spearman", gate_rule="selected_arm_all_three_initialization_seeds_positive",
        gate_is_not_statistical_significance=True, independent_external_confirmation=False,
        complexity_rule="retain_A1_if_each_seed_A1_minus_A0_positive; retain_A2_only_after_A1_and_each_seed_A2_minus_A1_positive",
        validation_scores_by_seed=scores, delta_A1_minus_A0=delta10.tolist(), delta_A2_minus_A1=delta21.tolist(),
        retained_A1=retain1, retained_A2=retain1 and retain2,
        split_lock_sha256=lock_sha, data_qualification_sha256=marker_sha,
        dictionary_npz_sha256=dictionary_sha, dictionary_receipt_sha256=file_sha256(out / "dictionary/DICT_RECEIPT_PRIVATE.json"),
        source_hashes=source_hashes(), schedule=SCHEDULE, models=models, validation_aggregate=receipts,
        test_used_for_selection=False, DSS_or_PFI_used=False,
        limitations=["two_validation_donors", "two_studies_total", "unknown_UNI2_pretraining_overlap", "fixed_model_selection_not_retraining_uncertainty"])
    path = out / "SELECTION_PRIVATE.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("locked validation selection exists with different contents; cannot overwrite")
    else:
        write_json(path, record)
    print("VALIDATION_SELECTION_LOCKED", selected, record["A_gate"], flush=True)
    return record


def bootstrap_donors(donor_values, *, seed=220122, repeats=2000):
    values = np.asarray(donor_values, dtype=np.float64)
    if len(values) < 2:
        return dict(mean=float(values.mean()), lower95=None, upper95=None, donors=len(values),
                    bootstrap_unit="donor", inference="INSUFFICIENT_DONORS")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    samples = values[draws].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(mean=float(values.mean()), lower95=float(lo), upper95=float(hi), donors=len(values),
                bootstrap_unit="donor", bootstrap_repeats=repeats,
                inference="DIRECTIONAL_SMALL_DONOR_SAMPLE_NOT_CONFIRMATORY")


def evaluate(args):
    out = private_output(args.output_root)
    d, genes, _, d_sha = load_dictionary(out)
    if args.split == "test":
        if args.selection_receipt is None:
            raise ValueError("test evaluation requires an immutable validation selection receipt")
        selection = read_json(args.selection_receipt)
        if selection.get("status") != "SELECTED_ON_VALIDATION_ONLY" or selection.get("test_used_for_selection") is not False:
            raise ValueError("invalid validation-only selection receipt")
        if selection.get("split_lock_sha256") != file_sha256(args.split_lock) or selection.get("dictionary_npz_sha256") != d_sha or selection.get("source_hashes") != source_hashes() or selection.get("data_qualification_sha256") != file_sha256(args.qualification):
            raise ValueError("test/source/dictionary selection identity drift")
        if args.arm is not None and args.arm != selection["selected_arm"]:
            raise ValueError("candidate arms cannot be ranked on ST-test")
        arms = [selection["selected_arm"]]
    else:
        selection = None
        arms = [args.arm] if args.arm is not None else list(ARMS)
    sections, lock_sha, _ = qualification_rows(args.data_root, args.split_lock, args.qualification,
        roles=(args.split,), with_features=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    records, prediction_receipts = [], []
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for arm in arms:
        for seed in seeds:
            model_path = out / "fits" / arm / ("seed%03d" % seed) / "model.pt"
            done_path = model_path.with_name("DONE_PRIVATE.json")
            done = read_json(done_path)
            if done.get("status") != "COMPLETE" or done.get("model_sha256") != file_sha256(model_path):
                raise ValueError("evaluation requires a completed exact selected checkpoint")
            if done["identity"]["data_qualification_sha256"] != file_sha256(args.qualification):
                raise ValueError("evaluation count/data-qualification lineage drift")
            if selection is not None and selection["models"]["%s_seed%03d" % (arm, seed)]["model_sha256"] != file_sha256(model_path):
                raise ValueError("test checkpoint differs from the validation-selected model")
            model, checkpoint = load_frozen_model(model_path, device)
            if checkpoint["gene_names"] != list(genes) or checkpoint["identity"]["split_lock_sha256"] != lock_sha or checkpoint["identity"]["dictionary_npz_sha256"] != d_sha:
                raise ValueError("evaluation model gene/data identity mismatch")
            for section in sections:
                count, measured = align_genes(section, genes)
                prediction = infer_arrays(model, section.features, section.coords_um, section.section_id, device=device)
                metrics = section_metrics(prediction["mu"], count, measured)
                # Fixed-D reconstruction oracle on observed genes only. Its
                # true-expression coefficients are never saved into pi0.
                dd = d[:, measured]
                dd = dd / dd.sum(1, keepdims=True)
                oracle_pi, oracle_audit = oracle_coefficients(count[:, measured], dd)
                oracle_metrics = section_metrics(oracle_pi @ dd, count[:, measured], np.ones(measured.sum(), bool))
                row = dict(arm=arm, seed=seed, split=args.split, section_id_private=section.section_id,
                    donor_id_private=section.donor_id, study_id=section.study_id, assay=section.assay,
                    metrics=metrics, fixed_D_oracle_metrics=oracle_metrics, oracle_audit=oracle_audit,
                    context_scope=prediction["context_scope"], source_counts_npz_sha256=section.evidence["npz_sha256"],
                    features_sha256=section.evidence["features_sha256"], source_model_sha256=file_sha256(model_path))
                records.append(row)
                pred_path = out / "predictions" / args.split / arm / ("seed%03d" % seed) / (section.section_id + ".npz")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(pi0=prediction["pi0"], mu=prediction["mu"], gene_names=genes,
                    coords_um=section.coords_um, barcodes=section.barcodes, measured_mask=measured,
                    source_counts_npz_sha256=np.asarray(section.evidence["npz_sha256"]),
                    source_model_sha256=np.asarray(file_sha256(model_path)),
                    context_scope=np.asarray("FULL_SECTION_GRAPH"))
                if pred_path.exists():
                    with np.load(pred_path, allow_pickle=False) as old:
                        if set(old.files) != set(payload) or any(not np.array_equal(old[k], v) for k, v in payload.items()):
                            raise ValueError("frozen prediction exists with different elements; cannot overwrite")
                else:
                    temporary = pred_path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, **payload)
                    temporary.replace(pred_path)
                prediction_receipts.append(dict(path_private=str(pred_path), sha256=file_sha256(pred_path), **row))
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["arm"], row["seed"], row["donor_id_private"])].append(row)
    donor_rows = []
    for (arm, seed, donor), rows in grouped.items():
        donor_rows.append(dict(arm=arm, seed=seed, donor_id_private=donor,
            metrics={key: float(np.mean([r["metrics"][key] for r in rows])) for key in rows[0]["metrics"]},
            oracle_metrics={key: float(np.mean([r["fixed_D_oracle_metrics"][key] for r in rows])) for key in rows[0]["metrics"]}))
    aggregate = []
    for arm in arms:
        for seed in seeds:
            rows = [r for r in donor_rows if r["arm"] == arm and r["seed"] == seed]
            aggregate.append(dict(arm=arm, seed=seed, split=args.split,
                metrics={key: bootstrap_donors([r["metrics"][key] for r in rows]) for key in rows[0]["metrics"]},
                oracle_metrics={key: bootstrap_donors([r["oracle_metrics"][key] for r in rows]) for key in rows[0]["oracle_metrics"]}))
    folder = out / "evaluation" / args.split
    write_json(folder / "EVALUATION_PRIVATE.json", dict(status="COMPLETE", split_lock_sha256=lock_sha,
        dictionary_npz_sha256=d_sha, source_hashes=source_hashes(), rows_private=prediction_receipts, donor_rows_private=donor_rows,
        selection_receipt_sha256=file_sha256(args.selection_receipt) if selection is not None else None,
        no_ST_test_model_selection=True, no_DSS_or_PFI=True))
    write_json(folder / "AGGREGATE_ANONYMOUS.json", dict(status="COMPLETE", split=args.split,
        aggregate=aggregate, context_scope="FULL_SECTION_GRAPH", gene_metric_unit="composition_on_measured_dictionary_genes",
        zero_count_spots_excluded_from_spatial_metrics=True, initialization_seeds_not_independent_donors=True,
        oracle_role="fixed_D_expression_reconstruction_reference_not_image_input_nor_ranking_upper_bound",
        no_ST_test_model_selection=True, no_DSS_or_PFI=True))
    print("FROZEN_EVALUATION_COMPLETE", args.split, len(arms), len(seeds), len(sections), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--select-on-validation", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--selection-receipt", type=Path)
    args = parser.parse_args()
    if args.select_on_validation:
        if args.split or args.arm or args.seed or args.selection_receipt:
            parser.error("selection is one locked nine-cell validation-only operation")
        candidate_selection(args.output_root, args.split_lock, args.qualification)
    else:
        if args.data_root is None or args.split is None:
            parser.error("frozen evaluation requires data-root and validation/test role")
        evaluate(args)


if __name__ == "__main__":
    main()
