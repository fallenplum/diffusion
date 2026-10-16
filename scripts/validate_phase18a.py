"""Phase18A independent, patient-paired evaluation and historical eligibility.

The statistical functions are deliberately independent of the training code.
Each seed's C is a ratio of summed within-fold numerators/denominators;
neither risks nor C values across folds are averaged to make that ratio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORICAL_ARMS = {"F0": "R1_S6_noDirect", "F1": "R1_A8_noDirect"}
COMPARISONS = {
    "L0-F0": {"L0": 1.0, "F0": -1.0},
    "L1-F1": {"L1": 1.0, "F1": -1.0},
    "L1-L0": {"L1": 1.0, "L0": -1.0},
    "DID=(L1-L0)-(F1-F0)": {"L1": 1.0, "L0": -1.0, "F1": -1.0, "F0": 1.0},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def cast(a):
        if isinstance(a, np.ndarray):
            return a.tolist()
        if isinstance(a, np.generic):
            return a.item()
        raise TypeError(type(a).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=cast, allow_nan=False) + "\n")
    temporary.replace(path)


def comparable_pairs(time, event):
    """Harrell pairs: event before later time, or event/censor at same time."""
    time = np.asarray(time)
    event = np.asarray(event, bool)
    valid = event[:, None] & ((time[:, None] < time[None, :]) |
                             ((time[:, None] == time[None, :]) & ~event[None, :]))
    return np.nonzero(valid)


def reverse_km_ipcw(train_time, train_event, test_time, test_event, tau=60.0):
    """Reverse KM tie handling independently implemented, with frozen train G."""
    train_time = np.asarray(train_time)
    train_event = np.asarray(train_event, bool)
    grid = np.unique(train_time)
    survival = []
    current = 1.0
    for value in grid:
        deaths = np.sum((train_time == value) & ~train_event)
        at_risk = np.sum(train_time >= value) - np.sum((train_time == value) & train_event)
        if deaths:
            if at_risk <= 0:
                raise ValueError("invalid reverse-KM risk set")
            current *= 1.0 - deaths / at_risk
        survival.append(current)
    result = np.zeros(len(test_time), float)
    for i, (value, is_event) in enumerate(zip(test_time, test_event)):
        if is_event and value < tau:
            index = np.searchsorted(grid, value, side="right") - 1
            g = 1.0 if index < 0 else survival[index]
            if g <= 0:
                raise ValueError("test event outside positive censoring support")
            result[i] = g ** -2
    return result


def components(time, event, risk, ipcw):
    i, j = comparable_pairs(time, event)
    delta = np.asarray(risk)[i] - np.asarray(risk)[j]
    credit = np.where(np.abs(delta) <= 1e-8, 0.5, delta > 0)
    weights = np.stack([np.ones(len(time)), np.asarray(ipcw)])[:, i]
    return (weights * credit[None, :]).sum(1), weights.sum(1)


def resampled_c_indices(time, event, valid, folds, risks, ipcw, *, draws, seed,
                        batch_size=256, forced_counts=None, count_audit=None):
    """Shared within-fold patient bootstrap, retaining seed/model coordinates.

    Counts are shared for every model and seed. Duplicate original patients
    reweight comparable pairs by count_i * count_j, with no self-pair invention.
    forced_counts is for an independent explicit-replication numerical oracle.
    """
    time, event, valid, folds = map(np.asarray, (time, event, valid, folds))
    keys = sorted(risks)
    num = np.zeros((draws, len(keys), 2), float)
    den = np.zeros((draws, 2), float)
    rng = np.random.default_rng(seed)
    for fold in sorted(set(folds[valid])):
        ix = np.flatnonzero((folds == fold) & valid)
        n = len(ix)
        counts = (rng.multinomial(n, np.full(n, 1.0 / n), size=draws)
                  if forced_counts is None else np.asarray(forced_counts[int(fold)]))
        if counts.shape != (draws, n) or (counts < 0).any() or not np.all(counts.sum(1) == n):
            raise ValueError("bootstrap counts violate fixed fold patient size")
        if count_audit is not None:
            count_audit.append({"fold": int(fold), "patients": n, "draws": draws,
                                "counts_sha256": hashlib.sha256(np.ascontiguousarray(counts, dtype=np.int64).tobytes()).hexdigest(),
                                "one_shared_count_matrix_for_model_seed_vectors": len(keys)})
        i, j = comparable_pairs(time[ix], event[ix])
        weights = np.stack([np.ones(n), ipcw[ix]])[:, i]
        den_matrix = np.zeros((2, n, n), float)
        den_matrix[:, i, j] = weights
        matrices = np.zeros((len(keys), 2, n, n), float)
        for k, key in enumerate(keys):
            risk = np.asarray(risks[key])[ix]
            if not np.isfinite(risk).all():
                raise ValueError("nonfinite risk in requested endpoint partition")
            delta = risk[i] - risk[j]
            credit = np.where(abs(delta) <= 1e-8, 0.5, delta > 0)
            matrices[k, :, i, j] = (weights * credit).T
        for start in range(0, draws, batch_size):
            cc = counts[start:start + batch_size].astype(float)
            products = cc[:, :, None] * cc[:, None, :]
            num[start:start + len(cc)] += np.einsum("bij,kmij->bkm", products, matrices, optimize=True)
            den[start:start + len(cc)] += np.einsum("bij,mij->bm", products, den_matrix, optimize=True)
    good = (den > 0).all(1)
    return keys, num[good] / den[good, None, :], den, good


def linear_c_contrast(keys, values, coefficients, seeds):
    """Compute each complete seed C first; DID is an exact four-C contrast."""
    index = {key: i for i, key in enumerate(keys)}
    terms = []
    for arm, coefficient in coefficients.items():
        selected = [index[arm, seed] for seed in seeds]
        terms.append(float(coefficient) * values[:, selected].mean(1))
    return np.sum(terms, axis=0)


def metrics(d, risks, endpoint, report_root):
    """Independent fold components checked against scikit-survival, then pooled."""
    import pandas as pd
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
    from sksurv.util import Surv
    from src.phase13_rna.data import endpoints
    ev = endpoints(d)[endpoint]
    time, event, valid = ev["time"], ev["event"].astype(bool), ev["valid"].astype(bool)
    folds = d["folds"]
    ipcw = np.zeros(len(time), float)
    fold_rows, seed_rows = [], []
    for fold in sorted(set(folds)):
        train = np.flatnonzero((folds != fold) & valid)
        test = np.flatnonzero((folds == fold) & valid)
        ipcw[test] = reverse_km_ipcw(time[train], event[train], time[test], event[test])
        for (arm, seed), risk in sorted(risks.items()):
            numerator, denominator = components(time[test], event[test], risk[test], ipcw[test])
            h, u = numerator / denominator
            hh = concordance_index_censored(event[test], time[test], risk[test])[0]
            uu = concordance_index_ipcw(Surv.from_arrays(event[train], time[train]),
                                       Surv.from_arrays(event[test], time[test]), risk[test], tau=60)[0]
            if abs(h - hh) > 1e-10 or abs(u - uu) > 1e-10:
                raise ValueError(("independent metric disagreement", endpoint, arm, seed, fold, h, hh, u, uu))
            fold_rows.append({"endpoint": endpoint, "arm": arm, "seed": seed, "fold": fold,
                              "patients": len(test), "events": int(event[test].sum()),
                              "Harrell": h, "Uno60": u, "H_num": numerator[0], "H_den": denominator[0],
                              "U_num": numerator[1], "U_den": denominator[1]})
    for arm, seed in sorted(risks):
        rows = [q for q in fold_rows if q["arm"] == arm and q["seed"] == seed]
        sums = {key: sum(q[key] for q in rows) for key in ["H_num", "H_den", "U_num", "U_den"]}
        seed_rows.append({"endpoint": endpoint, "arm": arm, "seed": seed,
                          "Harrell": sums["H_num"] / sums["H_den"],
                          "Uno60": sums["U_num"] / sums["U_den"], **sums})
    prefix = "" if endpoint == "DSS" else "PFI_"
    pd.DataFrame(fold_rows).to_csv(Path(report_root) / (prefix + "FOLD_METRICS.csv"), index=False)
    pd.DataFrame(seed_rows).to_csv(Path(report_root) / (prefix + "SEED_METRICS.csv"), index=False)
    pd.DataFrame(seed_rows).groupby("arm")[["Harrell", "Uno60"]].agg(["mean", "std"]).to_csv(
        Path(report_root) / (prefix + "SUMMARY.csv"))
    return time, event, valid, ipcw, seed_rows


def bootstrap(d, risks, endpoint_data, seed_rows, config, report_root):
    import pandas as pd
    time, event, valid, ipcw = endpoint_data
    bc = config["bootstrap"]
    count_audit = []
    keys, values, den, good = resampled_c_indices(time, event, valid, d["folds"], risks, ipcw,
                                                draws=bc["draws"], seed=bc["seed"], count_audit=count_audit)
    if int(good.sum()) < 0.99 * bc["draws"]:
        raise ValueError("too many zero-denominator resamples")
    coefficients = dict(COMPARISONS)
    secondary = ["ANCHOR_CS", "HIST_CS_S32_CONCH32", "R1_A3_noDirect"]
    for arm in secondary:
        if (arm, config["seeds"][0]) in risks:
            for candidate in ["F0", "F1", "L0", "L1"]:
                coefficients[candidate + "-" + arm] = {candidate: 1.0, arm: -1.0}
    for arm in ["F1", "L1"]:
        if (arm + "__WRONG", config["seeds"][0]) in risks:
            coefficients[arm + "-" + arm + "__WRONG"] = {arm: 1.0, arm + "__WRONG": -1.0}
    rows = []
    point_values = np.array([[[q[metric] for metric in ["Harrell", "Uno60"]]
                             for arm, seed in keys
                             for q in seed_rows if q["arm"] == arm and q["seed"] == seed]])
    if point_values.shape != (1, len(keys), 2):
        raise ValueError("missing or repeated point-estimate cell")
    for name, coeff in coefficients.items():
        delta = linear_c_contrast(keys, values, coeff, config["seeds"])
        point = linear_c_contrast(keys, point_values, coeff, config["seeds"])[0]
        primary = name in COMPARISONS
        family_tests = 8 if primary else 2 * (len(coefficients) - len(COMPARISONS))
        alpha = 0.05 / family_tests
        for m, metric in enumerate(["Harrell", "Uno60"]):
            lo, hi = np.quantile(delta[:, m], [.025, .975])
            cl, ch = np.quantile(delta[:, m], [alpha / 2, 1 - alpha / 2])
            rows.append({"comparison": name, "metric": metric, "family": "primary" if primary else "secondary",
                         "delta": point[m], "CI95_low": lo, "CI95_high": hi,
                         "Bonferroni_low": cl, "Bonferroni_high": ch, "family_tests": family_tests,
                         "draws_requested": bc["draws"], "draws_valid": int(good.sum()),
                         "point_definition": "mean seed C after within-fold num/den sums; never mean risk"})
    pd.DataFrame(rows).to_csv(Path(report_root) / "PAIRED_COMPARISONS.csv", index=False)
    write(Path(report_root) / "BOOTSTRAP_AUDIT.json", {
        "status": "PASS", "draws": bc["draws"], "valid": int(good.sum()),
        "zero_denominator_draws": int((~good).sum()), "primary_family_reserved_tests": 8,
        "minimum_H_denominator": float(den[:, 0].min()), "minimum_U_denominator": float(den[:, 1].min()),
        "sampling": "patient multinomial within fixed outer folds; shared across all models and seeds",
        "shared_patient_count_matrices": count_audit,
        "IPCW": "fixed original outer-train reverse KM", "pair_weight": "count_i*count_j",
        "DID": "each draw recomputes four complete C values, then (L1-L0)-(F1-F0)",
        "uncertainty_scope": "conditional on fitted models; excludes complete retraining and researcher selection",
        "risk_ensembling": False})


def assert_equal_structure(left, right, label):
    """Exact feature/provenance equality, including numpy-bearing audit dicts."""
    if isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError((label, "dictionary keys changed"))
        for key in left:
            assert_equal_structure(left[key], right[key], label + "." + str(key))
    elif isinstance(left, (np.ndarray, list, tuple)):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError((label, "array changed"))
    elif left != right:
        raise ValueError((label, left, right))


def software_eligibility():
    import importlib.metadata
    import platform
    import torch
    prior = json.loads((ROOT / "reports/phase17r_conditional_repair/ENVIRONMENT.json").read_text())
    packages = {name: importlib.metadata.version(name) for name in prior["packages"]}
    for name, value in packages.items():
        if value != prior["packages"][name]:
            raise ValueError(("historical software package mismatch", name, value, prior["packages"][name]))
    if platform.python_version() != prior["python"].split()[0]:
        raise ValueError("historical Python version mismatch")
    if torch.version.cuda != prior["torch_CUDA"]:
        raise ValueError("historical torch CUDA mismatch")
    return {"status": "PASS", "python": platform.python_version(), "packages": packages,
            "torch_CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
            "note": "package/Python/CUDA versions match saved Phase17R environment; replay separately checks arithmetic"}


def historical_jobs(config):
    from src.phase17r_ssd.data import P
    from scripts.run_phase17r import stem
    jobs = []
    for arm, parent_arm in HISTORICAL_ARMS.items():
        for role in ["inner", "outer"]:
            for seed in config["seeds"]:
                for fold in config["folds"]:
                    jobs.append((arm, parent_arm, role, seed, fold, P / role / (stem(parent_arm, seed, fold) + ".pt")))
    for seed in config["seeds"]:
        for fold in config["folds"]:
            arm = "R1_A3_noDirect"
            jobs.append((arm, arm, "outer", seed, fold, P / "outer" / (stem(arm, seed, fold) + ".pt")))
    return jobs


def verify_checkpoint(path, arm, base_arm, role, seed, fold, data, expected_source, cache):
    """Source/SHA/partitions/schedule/anchor/preprocessing, before any reference use."""
    import torch
    from scripts import validate_phase17_independent as anchor_oracle
    from src.phase17r_ssd import data as prior
    path = Path(path)
    if not path.exists() or not path.with_suffix(".json").exists():
        raise FileNotFoundError("checkpoint and sidecar required: " + str(path))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != sidecar["checkpoint_sha256"] or ck["source"] != expected_source or sidecar["source"] != expected_source:
        raise ValueError(("source or checkpoint SHA mismatch", path.name))
    for field, value in [("arm", base_arm if arm in HISTORICAL_ARMS or arm == "R1_A3_noDirect" else arm),
                         ("seed", seed), ("fold", fold), ("role", role), ("budget", 512)]:
        if ck[field] != value:
            raise ValueError(("checkpoint identity mismatch", path.name, field))
    if ck.get("direction", "forward") != "forward" or ck.get("scale", 1.) != 1.:
        raise ValueError("only unscaled forward R1 lineage eligible")
    if arm in ["L0", "L1"] and (ck.get("base_arm") != base_arm or ck.get("readout_arm") != arm):
        raise ValueError("local readout checkpoint lineage mismatch")
    split = prior.old.parent.splits(data, "DSS", fold)
    fit = split["inner_train" if role == "inner" else "outer_train"]
    valid = split["inner_val" if role == "inner" else "outer_test"]
    if not np.array_equal(ck["fit"], fit) or not np.array_equal(ck["validation_or_test"], valid):
        raise ValueError("patient partition mismatch")
    if set(fit) & set(valid):
        raise ValueError("patient leakage")
    if role == "inner" and set(np.r_[fit, valid]) & set(split["outer_test"]):
        raise ValueError("outer test entered inner fitting/validation")
    key = (role, fold)
    if key not in cache:
        _, preprocess = prior.old.molecular_features(data, fold, role)
        c, s = anchor_oracle.independent_cs(data, fit)
        cache[key] = (preprocess, np.c_[c, s])
    preprocess, x = cache[key]
    assert_equal_structure(ck["preprocess_audit"], preprocess, "preprocess")
    expected_anchor = (x @ ck["anchor_w"]).astype(np.float32)
    anchor_error = float(abs(expected_anchor - ck["anchor_risk"]).max())
    gradient = float(abs(anchor_oracle.anchor_gradient(x[fit].astype(float), data["DSS_time"][fit],
                                                      data["DSS_event"][fit], ck["anchor_w"])).max())
    if anchor_error >= 1e-6 or gradient >= 2e-5:
        raise ValueError(("independent anchor failure", path.name, anchor_error, gradient))
    history = ck["history"]
    if [q["epoch"] for q in history] != list(range(1, len(history) + 1)):
        raise ValueError("noncontiguous epoch history")
    selected = ck["selected_epoch"]
    if selected != sidecar["selected_epoch"] or selected < 1 or selected > len(history):
        raise ValueError("selected epoch mismatch")
    if role == "inner":
        best = float("inf"); expected_epoch = 0; ema = None
        for row in history:
            ema = row["validation_loss"] if ema is None else .2 * row["validation_loss"] + .8 * ema
            if abs(ema - row["validation_loss_EMA"]) > 1e-10:
                raise ValueError("inner EMA recurrence changed")
            if ema < best - 1e-8:
                best, expected_epoch = ema, row["epoch"]
        if expected_epoch != selected:
            raise ValueError("selected epoch does not follow locked inner-only EMA rule")
    else:
        if len(history) != selected:
            raise ValueError("outer refit did not use exact selected epochs")
        inner_path = path.parent.parent / "inner" / path.name
        inner = torch.load(inner_path, map_location="cpu", weights_only=False)
        if inner["source"] != expected_source or inner["selected_epoch"] != selected:
            raise ValueError("outer epoch not locked from same-source inner checkpoint")
        if not np.array_equal(inner["fit"], split["inner_train"]) or not np.array_equal(inner["validation_or_test"], split["inner_val"]):
            raise ValueError("inner selector for outer refit uses wrong partitions")
        if sha(inner_path) != json.loads(inner_path.with_suffix(".json").read_text())["checkpoint_sha256"]:
            raise ValueError("inner selection checkpoint SHA mismatch")
    for key in ["risk", "wrong_risk", "SSM_off_risk", "gamma_one_risk"]:
        if np.asarray(ck[key]).shape != (len(valid),) or not np.isfinite(ck[key]).all():
            raise ValueError("missing/nonfinite persisted prediction " + key)
    for partition in [fit, valid]:
        wrong = np.asarray(ck["wrong_map"])[partition]
        if set(wrong) != set(partition) or np.any(wrong == partition):
            raise ValueError("wrong-patient context not a partition-local derangement")
    if any(name.startswith("direct.") for name in ck["state"]):
        raise ValueError("noDirect historical checkpoint contains Direct head")
    row = {"arm": arm, "parent_arm": base_arm, "role": role, "seed": seed, "fold": fold,
           "source": expected_source, "checkpoint_sha256": sidecar["checkpoint_sha256"],
           "selected_epoch": selected, "trained_epochs": len(history),
           "fit_patients": len(fit), "heldout_patients": len(valid),
           "anchor_max_error": anchor_error, "anchor_stationarity_error": gradient,
           "status": "PASS"}
    return ck, row


def replay_checkpoint(ck, label):
    import torch
    from src.phase17r_ssd import data as prior
    from src.phase17r_ssd.model import RepairS6
    from src.phase17_ssd.train import seed_all
    seed_all(ck["seed"] + 100 * ck["fold"])
    ds = prior.Data(ck["fold"], ck["role"], ck["seed"], "forward")
    if label in ["L0", "L1"]:
        from src.phase18a_readout.model import ReadoutS6
        from src.phase18a_readout.data import cfg
        model = ReadoutS6(ds.f["C"].shape[1], ds.f["membership"], label,
                         cfg()["parent_training"]["dropout"], cfg()["scan"]["chunk_size"]).cuda()
    else:
        model = RepairS6(ds.f["C"].shape[1], ds.f["membership"], ck["arm"], 1.,
                         prior.cfg()["training"]["dropout"]).cuda()
    model.load_state_dict(ck["state"], strict=True)
    model.eval()
    if not np.array_equal(ds.wrong, ck["wrong_map"]):
        raise ValueError("wrong-patient matching changed")
    if not np.array_equal(ds.anchor.cpu().numpy(), ck["anchor_risk"]):
        raise ValueError("anchor no longer bitwise matches original training")
    maximum = 0.
    with torch.no_grad():
        for key, kwargs in [("risk", {}), ("wrong_risk", {"wrong_inference": True}),
                            ("SSM_off_risk", {"module_off": True})]:
            risk = ds.forward(model, ds.valid, **kwargs).cpu().numpy()
            maximum = max(maximum, float(abs(risk - ck[key]).max()))
        model.gamma.fill_(1.)
        risk = ds.forward(model, ds.valid).cpu().numpy()
        maximum = max(maximum, float(abs(risk - ck["gamma_one_risk"]).max()))
    if maximum >= 1e-6:
        raise ValueError(("checkpoint replay failure", label, ck["seed"], ck["fold"], ck["role"], maximum))
    del model, ds
    torch.cuda.empty_cache()
    return maximum


def historical_audit(shard=0, shards=1, replay=False):
    import pandas as pd
    from src.phase18a_readout.data import cfg, R, P
    from src.phase17r_ssd import data as prior
    config = cfg()
    if prior.source() != config["parent_training_source"]:
        raise ValueError("historical training code drift")
    for key, value in config["parent_training"].items():
        if prior.cfg()["training"][key] != value:
            raise ValueError(("training schedule drift", key))
    visual = prior.old.prepare_visual()
    if visual["dataset_sha256"] != sha(prior.old.parent.DATA):
        raise ValueError("dataset SHA drift")
    software = software_eligibility()
    data = prior.old.parent.load()
    rows = []; cache = {}
    jobs = historical_jobs(config)
    for arm, parent_arm, role, seed, fold, path in jobs[shard::shards]:
        ck, row = verify_checkpoint(path, arm, parent_arm, role, seed, fold, data, prior.source(), cache)
        row["fit_classification"] = "historical_reference" if arm in HISTORICAL_ARMS else "historical_secondary_reference"
        row["new_optimization"] = False
        if replay:
            row["maximum_replay_error"] = replay_checkpoint(ck, arm)
        rows.append(row)
        print("HISTORICAL_ELIGIBLE", arm, role, seed, fold, row.get("maximum_replay_error", "NOT_REPLAYED"), flush=True)
    R.mkdir(parents=True, exist_ok=True); P.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(R / f"HISTORICAL_REUSE_AUDIT_{shard}.csv", index=False)
    write(P / f"HISTORICAL_REUSE_AUDIT_{shard}.json", {
        "status": "PASS", "source": prior.source(), "shard": shard, "shards": shards,
        "rows": len(rows), "replay": replay, "software": software,
        "visual_hashes": visual["hashes"], "dataset_sha256": visual["dataset_sha256"]})
    if shards == 1:
        write(R / "HISTORICAL_REUSE_AUDIT.json", {
            "status": "PASS" if replay else "REPLAY_REQUIRED", "source": prior.source(),
            "primary_reference_cells": 60, "primary_outer_references": 30,
            "secondary_outer_references": 15, "total_audited_checkpoints": len(rows),
            "new_model_fits": 0, "replayed": replay, "software": software,
            "maximum_replay_error": max((row.get("maximum_replay_error", 0.) for row in rows), default=0.)})
    return rows


def aggregate_historical(shards):
    import pandas as pd
    from src.phase18a_readout.data import cfg, R, P
    from src.phase17r_ssd.data import source
    rows = []
    for shard in range(shards):
        saved = json.loads((P / f"HISTORICAL_REUSE_AUDIT_{shard}.json").read_text())
        if saved["status"] != "PASS" or not saved["replay"] or saved["source"] != source() or saved["shards"] != shards:
            raise ValueError("historical replay shard incomplete")
        rows.extend(pd.read_csv(R / f"HISTORICAL_REUSE_AUDIT_{shard}.csv").to_dict("records"))
    expected = {(a, r, s, f) for a, _, r, s, f, _ in historical_jobs(cfg())}
    actual = {(q["arm"], q["role"], q["seed"], q["fold"]) for q in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("historical eligible checkpoint coverage incomplete/duplicated")
    pd.DataFrame(rows).to_csv(R / "HISTORICAL_REUSE_AUDIT.csv", index=False)
    write(R / "HISTORICAL_REUSE_AUDIT.json", {
        "status": "PASS", "source": source(), "primary_reference_cells": 60,
        "primary_outer_references": 30, "secondary_outer_references": 15,
        "total_audited_checkpoints": len(rows), "new_model_fits": 0, "replayed": True,
        "maximum_replay_error": max(q["maximum_replay_error"] for q in rows),
        "software": saved["software"], "dataset_sha256": saved["dataset_sha256"],
        "visual_hashes": saved["visual_hashes"]})
    return rows


def historical_readout_probes(shard=0, shards=1):
    """New label-blind diagnostics of frozen old weights; no old file is written."""
    import torch
    import pandas as pd
    from src.phase18a_readout.data import cfg, R, P
    from src.phase18a_readout.diagnostics import diagnose
    from src.phase17r_ssd import data as prior
    from src.phase17r_ssd.model import RepairS6
    from src.phase17_ssd.train import seed_all
    rows = []
    jobs = [job for job in historical_jobs(cfg()) if job[0] in HISTORICAL_ARMS]
    for arm, parent_arm, role, seed, fold, path in jobs[shard::shards]:
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if sha(path) != sidecar["checkpoint_sha256"]:
            raise ValueError("frozen historical weights changed before probe")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["source"] != prior.source():
            raise ValueError("historical source changed before probe")
        seed_all(seed + 100 * fold)
        ds = prior.Data(fold, role, seed)
        model = RepairS6(ds.f["C"].shape[1], ds.f["membership"], parent_arm, 1., prior.cfg()["training"]["dropout"]).cuda()
        model.load_state_dict(ck["state"], strict=True); model.eval()
        if not np.array_equal(ds.fit, ck["fit"]) or not np.array_equal(ds.wrong, ck["wrong_map"]):
            raise ValueError("frozen historical probe inputs changed")
        diagnostic = diagnose(model, ds)
        rows.append({"arm": arm, "parent_arm": parent_arm, "role": role, "seed": seed, "fold": fold,
                     "selected_epoch": ck["selected_epoch"], "checkpoint_sha256": sidecar["checkpoint_sha256"],
                     "frozen_historical_weights": True, "new_model_fit": False, **diagnostic})
        print("HISTORICAL_PROBE", arm, role, seed, fold, flush=True)
        del model, ds; torch.cuda.empty_cache()
    write(P / f"HISTORICAL_READOUT_PROBES_{shard}.json", {"status": "PASS", "source": prior.source(),
          "diagnostics_sha256": sha(ROOT / "src/phase18a_readout/diagnostics.py"),
          "shards": shards, "rows": rows})
    return rows


def flatten_mechanisms(rows):
    flattened = []
    for row in rows:
        simple = {key: value for key, value in row.items()
                  if not isinstance(value, (dict, list, tuple, np.ndarray)) and key not in ["target_value", "risk", "anchor_risk"]}
        probes = row.get("position_gradient_probes", [])
        if probes:
            for probe in probes:
                flattened.append({**simple, **{"position_probe_" + key: value for key, value in probe.items() if key != "target_value"}})
        else:
            flattened.append({**simple, "position_probe_target": "not_available"})
    return flattened


def aggregate_historical_probes(shards):
    import pandas as pd
    from src.phase18a_readout.data import P, R, cfg
    from src.phase17r_ssd.data import source
    rows = []
    for shard in range(shards):
        z = json.loads((P / f"HISTORICAL_READOUT_PROBES_{shard}.json").read_text())
        if z["status"] != "PASS" or z["source"] != source() or z["shards"] != shards or z["diagnostics_sha256"] != sha(ROOT / "src/phase18a_readout/diagnostics.py"):
            raise ValueError("historical readout probe incomplete")
        rows.extend(z["rows"])
    expected = {(a, r, s, f) for a, _, r, s, f, _ in historical_jobs(cfg()) if a in HISTORICAL_ARMS}
    if len(rows) != len(expected) or {(q["arm"], q["role"], q["seed"], q["fold"]) for q in rows} != expected:
        raise ValueError("incomplete/duplicate historical readout probe coverage")
    write(P / "HISTORICAL_READOUT_PROBES.json", {"status": "PASS", "source": source(),
          "diagnostics_sha256": sha(ROOT / "src/phase18a_readout/diagnostics.py"), "rows": rows})
    pd.DataFrame(flatten_mechanisms(rows)).to_csv(R / "HISTORICAL_READOUT_MECHANISM.csv", index=False)
    write(R / "HISTORICAL_READOUT_PROBE_AUDIT.json", {"status": "PASS", "selected_checkpoints": 60,
          "state_targets": 60, "patient_risk_targets": 60, "first_training_patients_per_checkpoint": 1,
          "label_blind_z_variance_patients_per_checkpoint": 32,
          "historical_weights_unchanged": True, "new_model_fits": 0})
    return rows


def new_checkpoint_replay(shard=0, shards=1):
    import pandas as pd
    from src.phase18a_readout.data import P, R, cfg, source
    from src.phase17r_ssd import data as prior
    data = prior.old.parent.load(); rows = []; cache = {}; config = cfg()
    jobs = [(arm, role, seed, fold) for arm in config["new_arms"] for role in ["inner", "outer"]
            for seed in config["seeds"] for fold in config["folds"]]
    for arm, role, seed, fold in jobs[shard::shards]:
        path = P / role / f"{arm}_seed{seed}_fold{fold}.pt"
        ck, row = verify_checkpoint(path, arm, config["new_arms"][arm], role, seed, fold, data, source(), cache)
        row["maximum_replay_error"] = replay_checkpoint(ck, arm)
        row["fit_classification"] = "new_fit"
        rows.append(row)
        print("NEW_CHECKPOINT_REPLAY", arm, role, seed, fold, row["maximum_replay_error"], flush=True)
    pd.DataFrame(rows).to_csv(R / f"NEW_CHECKPOINT_REPLAY_{shard}.csv", index=False)
    write(P / f"NEW_REPLAY_SHARD_{shard}.json", {"status": "PASS", "source": source(), "shards": shards,
                                                  "rows": len(rows), "replayed": True})


def collect():
    import torch
    import pandas as pd
    from src.phase18a_readout.data import P, R, cfg, source
    from src.phase17r_ssd import data as prior
    config = cfg(); new_source = source(); data = prior.old.parent.load()
    historical_gate = json.loads((R / "HISTORICAL_REUSE_AUDIT.json").read_text())
    if historical_gate["status"] != "PASS" or historical_gate["source"] != prior.source() or not historical_gate["replayed"]:
        raise ValueError("historical outputs are ineligible for reference")
    if len(data["patient_ids"]) != 512 or int(data["DSS_event"].sum()) != 65:
        raise ValueError("frozen DSS cohort changed")
    lock = json.loads((R / "PROTOCOL_LOCK.json").read_text())
    if lock["source"] != new_source:
        raise ValueError("new training source differs from lock")
    rows = []; cache = {}; risks = {}; curves = []; mechanisms = []; summaries = []; anchors = {}; replay_rows = []
    probe_coverage = {}; visual_owner = np.load(prior.old.P / "visual/owner.npy")
    historical_probes = json.loads((P / "HISTORICAL_READOUT_PROBES.json").read_text())
    if historical_probes["status"] != "PASS" or historical_probes["source"] != prior.source() or historical_probes["diagnostics_sha256"] != sha(ROOT / "src/phase18a_readout/diagnostics.py"):
        raise ValueError("historical readout diagnostics missing")
    probe_index = {(q["arm"], q["role"], q["seed"], q["fold"]): q for q in historical_probes["rows"]}
    jobs = historical_jobs(config)
    jobs += [(arm, config["new_arms"][arm], role, seed, fold, P / role / f"{arm}_seed{seed}_fold{fold}.pt")
             for arm in config["new_arms"] for role in ["inner", "outer"] for seed in config["seeds"] for fold in config["folds"]]
    for arm, base_arm, role, seed, fold, path in jobs:
        new = arm in config["new_arms"]
        expected_source = new_source if new else prior.source()
        ck, row = verify_checkpoint(path, arm, base_arm, role, seed, fold, data, expected_source, cache)
        row["fit_classification"] = "new_fit" if new else "historical_reference" if arm in HISTORICAL_ARMS else "historical_secondary_reference"
        row["new_optimization"] = new
        rows.append(row)
        if arm in ["F0", "F1", "L0", "L1"]:
            coverage = probe_coverage.setdefault((arm, role), {"checkpoints": 0, "gradient_patients": set(),
                "gradient_WSIs": set(), "variance_patients": set(), "variance_WSIs": set(),
                "gradient_WSI_counts": [], "variance_WSI_counts": []})
            first = int(ck["fit"][0]); variance_patients = np.asarray(ck["fit"][:32])
            first_slides = np.flatnonzero(visual_owner == first)
            variance_slides = np.flatnonzero(np.isin(visual_owner, variance_patients))
            coverage["checkpoints"] += 1
            coverage["gradient_patients"].add(first); coverage["gradient_WSIs"].update(first_slides.tolist())
            coverage["variance_patients"].update(variance_patients.tolist()); coverage["variance_WSIs"].update(variance_slides.tolist())
            coverage["gradient_WSI_counts"].append(len(first_slides)); coverage["variance_WSI_counts"].append(len(variance_slides))
        for value in ck["history"]:
            curves.append({"arm": arm, "role": role, "seed": seed, "fold": fold,
                           "selected_epoch": ck["selected_epoch"], "fit_classification": row["fit_classification"], **value})
        scalar_mechanism = {key: value for key, value in ck["mechanism"].items()}
        if arm in HISTORICAL_ARMS:
            scalar_mechanism = probe_index[arm, role, seed, fold]
        mechanisms.append({"arm": arm, "role": role, "seed": seed, "fold": fold,
                           "selected_epoch": ck["selected_epoch"], **scalar_mechanism})
        summaries.append({"arm": arm, "role": role, "seed": seed, "fold": fold,
                          "selected_epoch": ck["selected_epoch"], "epochs_ran": len(ck["history"]),
                          "fit_classification": row["fit_classification"],
                          **{key: value for key, value in ck["summary"].items() if not isinstance(value, dict)}})
        # Anchors are bitwise identical across the 2x2 arms and seeds for each role/fold.
        anchor_key = role, fold
        if anchor_key in anchors:
            if not np.array_equal(anchors[anchor_key][0], ck["anchor_w"]) or not np.array_equal(anchors[anchor_key][1], ck["anchor_risk"]):
                raise ValueError(("anchor differs across readout/condition/seed", arm, role, seed, fold))
        else:
            anchors[anchor_key] = ck["anchor_w"].copy(), ck["anchor_risk"].copy()
        if role == "outer":
            test = np.asarray(ck["validation_or_test"])
            for label, values in [(arm, ck["risk"]), ("ANCHOR_CS", ck["anchor_risk"][test])]:
                vector = risks.setdefault((label, seed), np.full(512, np.nan))
                if np.isfinite(vector[test]).any() and not np.array_equal(vector[test], values):
                    raise ValueError("duplicate OOF assignment disagrees")
                vector[test] = values
            if arm in ["F1", "L1"]:
                risks.setdefault((arm + "__WRONG", seed), np.full(512, np.nan))[test] = ck["wrong_risk"]
    if sum(q["new_optimization"] for q in rows) != 60:
        raise ValueError("new fit count does not equal 30 inner plus 30 outer")
    # Compare saved historical simple baseline cell predictions, without refitting or risk ensembling.
    old_oof = np.load("/root/phase16r_light_rna_private/OOF_PRIVATE.npz")
    historical_simple_rows = []
    for seed in config["seeds"]:
        vector = old_oof["LOCAL__CV3__seed" + str(seed)]
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("historical CONCH32 reference incomplete")
        for fold in config["folds"]:
            path = Path("/root/phase16r_light_rna_private/artifacts") / f"seed{seed}_fold{fold}.pt"
            receipt = json.loads((Path("/root/phase16r_light_rna_private/cells") / f"seed{seed}_fold{fold}.json").read_text())
            if sha(path) != receipt["sha256"]:
                raise ValueError("historical simple baseline SHA mismatch")
            frozen = torch.load(path, map_location="cpu", weights_only=False)
            test = np.flatnonzero(data["folds"] == fold)
            if not np.array_equal(frozen["part"]["outer_test"], test):
                raise ValueError("historical simple baseline patient fold mismatch")
            error = float(abs(frozen["models"]["LOCAL__CV3"]["risk"] - vector[test]).max())
            if error > 1e-9:
                raise ValueError("historical simple baseline OOF/cell disagreement")
            historical_simple_rows.append({"arm": "HIST_CS_S32_CONCH32", "seed": seed, "fold": fold,
                                           "source_artifact_sha256": receipt["sha256"], "maximum_error": error,
                                           "reference_status": "different visual representation/training protocol; not same-input ablation"})
        risks["HIST_CS_S32_CONCH32", seed] = vector.copy()
    for key, vector in risks.items():
        if not np.isfinite(vector).all():
            raise ValueError(("OOF predictions missing", key))
    # Every new replay record must identify current weights; no stale replay receipts accepted.
    for path in sorted(R.glob("NEW_CHECKPOINT_REPLAY_*.csv")):
        replay_rows.extend(pd.read_csv(path).to_dict("records"))
    expected_new = {(q["arm"], q["role"], q["seed"], q["fold"]): q for q in rows if q["new_optimization"]}
    actual_new = {(q["arm"], q["role"], q["seed"], q["fold"]): q for q in replay_rows}
    if len(replay_rows) != 60 or set(actual_new) != set(expected_new):
        raise ValueError("new checkpoint replay coverage incomplete/duplicated")
    for key, row in actual_new.items():
        if row["source"] != new_source or row["checkpoint_sha256"] != expected_new[key]["checkpoint_sha256"] or row["maximum_replay_error"] >= 1e-6:
            raise ValueError("new checkpoint replay provenance mismatch")
    old_replay = pd.read_csv(R / "HISTORICAL_REUSE_AUDIT.csv").to_dict("records")
    expected_old = {(q["arm"], q["role"], q["seed"], q["fold"]): q for q in rows if not q["new_optimization"]}
    if len(old_replay) != len(expected_old):
        raise ValueError("historical replay coverage changed")
    for row in old_replay:
        key = row["arm"], row["role"], row["seed"], row["fold"]
        if key not in expected_old or row["checkpoint_sha256"] != expected_old[key]["checkpoint_sha256"] or row["maximum_replay_error"] >= 1e-6:
            raise ValueError("historical replay receipt does not identify currently qualified checkpoint")
    pd.DataFrame(old_replay + replay_rows).to_csv(R / "CHECKPOINT_REPLAY.csv", index=False)
    write(R / "CHECKPOINT_REPLAY.json", {"status": "PASS", "historical_replayed": len(old_replay), "new_replayed": 60,
          "maximum_error": max(q["maximum_replay_error"] for q in old_replay + replay_rows), "conditions_per_checkpoint": 4})
    pd.DataFrame(rows).to_csv(R / "CELL_MANIFEST.csv", index=False)
    pd.DataFrame(curves).to_csv(R / "TRAINING_CURVES.csv", index=False)
    pd.DataFrame(flatten_mechanisms(mechanisms)).to_csv(R / "READOUT_MECHANISM.csv", index=False)
    pd.DataFrame(summaries).to_csv(R / "COMPUTE_PROFILE.csv", index=False)
    pd.DataFrame(historical_simple_rows).to_csv(R / "HISTORICAL_CONCH32_REFERENCE_AUDIT.csv", index=False)
    coverage_rows = []
    for (arm, role), coverage in sorted(probe_coverage.items()):
        if coverage["checkpoints"] != 15:
            raise ValueError("incomplete 2x2 readout probe coverage")
        coverage_rows.append({"arm": arm, "role": role, "checkpoint_count": coverage["checkpoints"],
            "gradient_probe_targets_per_checkpoint": 2, "gradient_probe_patients_per_checkpoint": 1,
            "gradient_distinct_patient_count": len(coverage["gradient_patients"]),
            "gradient_distinct_WSI_count": len(coverage["gradient_WSIs"]),
            "gradient_WSIs_per_checkpoint_min": min(coverage["gradient_WSI_counts"]),
            "gradient_WSIs_per_checkpoint_max": max(coverage["gradient_WSI_counts"]),
            "variance_probe_patients_per_checkpoint": 32,
            "variance_distinct_patient_count": len(coverage["variance_patients"]),
            "variance_distinct_WSI_count": len(coverage["variance_WSIs"]),
            "variance_WSIs_per_checkpoint_min": min(coverage["variance_WSI_counts"]),
            "variance_WSIs_per_checkpoint_max": max(coverage["variance_WSI_counts"])})
    if len(coverage_rows) != 8:
        raise ValueError("missing arm/role readout probe group")
    write(R / "PROBE_COVERAGE_AUDIT.json", {"status": "PASS", "groups": coverage_rows,
          "gradient_selection": "same first training patient fit[0] in each selected checkpoint; all its WSIs",
          "variance_selection": "first32 training patients, all their WSIs; label blind",
          "limitation": "checkpoint_count is not independent patient count; seeds and folds repeatedly probe the same patients; no original histology QA",
          "privacy": "only union counts and per-checkpoint count ranges; no patient/slide indices or raw target values"})
    np.savez_compressed(P / "OOF_PRIVATE.npz", **{arm + "__seed" + str(seed): value for (arm, seed), value in risks.items()})
    write(R / "FIT_AND_REFERENCE_COUNTS.json", {"status": "PASS", "new_inner_fits": 30, "new_outer_fits": 30,
          "historical_primary_inner_references": 30, "historical_primary_outer_references": 30,
          "historical_A3_outer_references": 15, "historical_CONCH32_frozen_outer_references": 15,
          "evaluation_reference_count": 90, "new_model_fits": 60,
          "not_counted_as_new_training": "historical references, frozen wrong/SSMoff/gamma1, label-blind diagnostics and replay"})
    return data, risks, rows


def formal_evaluation():
    import pandas as pd
    from src.phase18a_readout.data import P, R, cfg, source
    data, risks, rows = collect()
    time, event, valid, ipcw, seed_rows = metrics(data, risks, "DSS", R)
    parity_rows = parent_metric_parity(seed_rows, "DSS")
    bootstrap(data, risks, (time, event, valid, ipcw), seed_rows, cfg(), R)
    pfi = metrics(data, risks, "PFI", R)
    parity_rows.extend(parent_metric_parity(pfi[-1], "PFI"))
    write(R / "PARENT_PUBLISHED_METRIC_PARITY.json", {"status": "PASS", "rows": parity_rows,
          "maximum_error": max(row["absolute_error"] for row in parity_rows),
          "note": "exact seed-level and cross-seed means, independently recomputed; no scores copied into new tables"})
    comparisons = pd.read_csv(R / "PAIRED_COMPARISONS.csv")
    point_audit = []
    seed_contrasts = []
    for name, coefficient in COMPARISONS.items():
        for metric in ["Harrell", "Uno60"]:
            per_seed = []
            for seed in cfg()["seeds"]:
                lookup = {q["arm"]: q[metric] for q in seed_rows if q["seed"] == seed}
                value = sum(lookup[arm] * scale for arm, scale in coefficient.items())
                per_seed.append(value)
                seed_contrasts.append({"comparison": name, "metric": metric, "seed": seed, "delta": value})
            row = comparisons[(comparisons["comparison"] == name) & (comparisons["metric"] == metric)]
            if len(row) != 1 or abs(float(row.iloc[0]["delta"]) - float(np.mean(per_seed))) > 1e-12:
                raise ValueError("reported contrast is not exact mean seed four-model C contrast")
            point_audit.append({"comparison": name, "metric": metric,
                                "from_seed_C": float(np.mean(per_seed)), "reported": float(row.iloc[0]["delta"]),
                                "positive_seeds": sum(value > 0 for value in per_seed)})
    pd.DataFrame(seed_contrasts).to_csv(R / "PRIMARY_SEED_CONTRASTS.csv", index=False)
    write(R / "CONTRAST_POINT_AUDIT.json", {"status": "PASS", "rows": point_audit,
          "DID": "exactly mean_s [(C_L1-C_L0)-(C_F1-C_F0)], both H and Uno"})
    write(R / "INDEPENDENT_VALIDATION.json", {"status": "PASS", "source": source(),
          "new_model_fits": 60, "primary_historical_outer_references": 30, "model_seed_vectors": len(risks),
          "fold_metric_rows": len(risks) * 5, "independently_checked_against_sksurv": True,
          "Clinical_S32_anchor": "independent fold-local reconstruction/stationarity and bitwise equality across all arms/seeds",
          "PFI": "same frozen DSS risks; no PFI optimization or selection", "risk_ensembling": False,
          "DID": "four complete C values recomputed in each of 20000 shared patient resamples"})
    print("PHASE18A_INDEPENDENT_EVALUATION_PASS", len(risks) * 5, flush=True)


def parent_metric_parity(seed_rows, endpoint):
    """Check recomputed historical predictions against published parent values."""
    import pandas as pd
    reference = ROOT / "reports/phase17r_conditional_repair"
    prefix = "" if endpoint == "DSS" else "PFI_"
    published = pd.read_csv(reference / (prefix + "SEED_OOF_METRICS.csv"))
    summary = pd.read_csv(reference / (prefix + "SUMMARY.csv"), header=[0, 1], index_col=0)
    aliases = {**HISTORICAL_ARMS, "R1_A3_noDirect": "R1_A3_noDirect", "ANCHOR_CS": "ANCHOR_CS"}
    rows = []
    for arm, parent_arm in aliases.items():
        current = [q for q in seed_rows if q["arm"] == arm]
        for q in current:
            old = published[(published["arm"] == parent_arm) & (published["seed"] == q["seed"])]
            if len(old) != 1:
                raise ValueError("parent published seed metric missing/duplicated")
            for metric in ["Harrell", "Uno60"]:
                error = abs(q[metric] - float(old.iloc[0][metric]))
                if error > 1e-12:
                    raise ValueError(("historical published metric drift", endpoint, arm, q["seed"], metric, error))
                rows.append({"endpoint": endpoint, "arm": arm, "seed": q["seed"], "metric": metric,
                             "absolute_error": error, "scope": "seed within-fold num/den ratio"})
        for metric in ["Harrell", "Uno60"]:
            error = abs(np.mean([q[metric] for q in current]) - float(summary.loc[parent_arm, (metric, "mean")]))
            if error > 1e-12:
                raise ValueError(("parent summary metric drift", endpoint, arm, metric, error))
            rows.append({"endpoint": endpoint, "arm": arm, "seed": "mean_of_three_seed_C", "metric": metric,
                         "absolute_error": float(error), "scope": "published cross-seed mean"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--aggregate-historical", action="store_true")
    parser.add_argument("--historical-probes-only", action="store_true")
    parser.add_argument("--aggregate-historical-probes", action="store_true")
    parser.add_argument("--new-replay", action="store_true")
    args = parser.parse_args()
    if args.aggregate_historical:
        aggregate_historical(args.shards)
    elif args.aggregate_historical_probes:
        aggregate_historical_probes(args.shards)
    elif args.historical_probes_only:
        historical_readout_probes(args.shard, args.shards)
    elif args.new_replay:
        new_checkpoint_replay(args.shard, args.shards)
    elif args.historical_only:
        historical_audit(args.shard, args.shards, args.replay)
    else:
        formal_evaluation()


if __name__ == "__main__":
    main()
