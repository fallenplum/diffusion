"""Phase6B: RNA-conditioned adaptation of frozen TITAN CRC concepts.

This audit deliberately operates on the *frozen TITAN output space* produced in
Phase6A.  The TITAN image and text encoders are not trained here; the 32 text
prototypes and 768-d patient image representations are loaded from the
official Phase6A cache.  A small RNA adapter generates patient-specific
output-space prompt offsets.  This is the deployable low-capacity analogue of
continuous prompt conditioning, and is labelled as such in all artifacts.

The only endpoint inputs are T/N/M/Stage labels.  Survival fields are never
read.  Every RNA transform, adapter, prompt basis, gate and classifier is fit
inside each inner-training split; an outer test patient is evaluated once.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.I)
SAMPLE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})", re.I)
ENDPOINTS = ("T", "N", "M", "Stage")
METHODS = ("static_concepts", "rna_conditioned", "static_concepts_plus_rna")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_label(value: object, endpoint: str) -> str | None:
    if pd.isna(value):
        return None
    value = re.sub(r"\s+", "", str(value).upper())
    if endpoint == "Stage":
        match = re.fullmatch(r"(?:STAGE)?(IV|III|II|I)[ABC]?", value)
        return match.group(1) if match else None
    allowed = {"T": "1234", "N": "012", "M": "01"}[endpoint]
    match = re.fullmatch(endpoint + "([" + allowed + r"])[ABC]?", value)
    return endpoint + match.group(1) if match else None


def load_rna_only(path: Path, patient_ids: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only the RNA matrix.  No clinical or survival columns are read."""
    table = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if table.shape[1] < 3:
        raise ValueError("RNA table does not contain sample columns")
    sample_columns: dict[str, list[str]] = {}
    for name in table.columns[2:]:
        match = PATIENT_RE.match(str(name).strip().upper())
        if match and match.group(1) in patient_ids:
            sample_columns.setdefault(match.group(1), []).append(str(name))
    missing = [p for p in patient_ids if p not in sample_columns]
    if missing:
        raise ValueError(f"RNA table misses {len(missing)} patients; first={missing[:5]}")
    numeric = table.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    columns = {str(name): i for i, name in enumerate(table.columns[2:])}

    def priority(name: str) -> tuple[int, str]:
        m = SAMPLE_RE.match(name)
        if not m:
            return 99, name
        return {"01": 0, "02": 1, "06": 2, "11": 3}.get(m.group(2), 10), name

    output = np.empty((len(patient_ids), numeric.shape[0]), dtype=np.float32)
    for row, patient in enumerate(patient_ids):
        names = sorted(sample_columns[patient], key=priority)
        primary = [name for name in names if priority(name)[0] == 0]
        selected = primary or names
        output[row] = np.nanmean(numeric[:, [columns[name] for name in selected]], axis=1).astype(np.float32)
    if not np.isfinite(output).any():
        raise ValueError("RNA matrix has no finite values")
    return output, {
        "raw_shape": list(output.shape),
        "input_sha256": sha256(path),
        "patient_count": len(patient_ids),
        "survival_columns_read": [],
        "clinical_columns_read": [],
    }


def fit_numeric(raw: np.ndarray, fit_idx: np.ndarray, output_dim: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    """Outcome-blind fold-local RNA transform copied from the M2 audited rule."""
    raw = np.asarray(raw, dtype=np.float32)
    train = raw[fit_idx]
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0.0
    clean = np.where(np.isfinite(raw), raw, median).astype(np.float32)
    if np.any(clean <= -1.0):
        raise ValueError("RNA log2(x+1) domain violation: finite input <= -1")
    clean = np.log2(clean + 1.0).astype(np.float32)
    train_clean = clean[fit_idx]
    variance = np.var(train_clean, axis=0)
    candidate = np.isfinite(variance) & (variance > 1e-12)
    ranked = np.flatnonzero(candidate)
    ranked = ranked[np.argsort(-variance[ranked], kind="stable")]
    keep = ranked[: min(output_dim, len(ranked))]
    if len(keep) < min(output_dim, clean.shape[1]):
        rest = np.setdiff1d(np.arange(clean.shape[1]), keep, assume_unique=False)
        keep = np.concatenate([keep, rest[: min(output_dim, clean.shape[1]) - len(keep)]])
    selected = clean[:, keep]
    mean = selected[fit_idx].mean(axis=0)
    std = selected[fit_idx].std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    result = ((selected - mean) / std).astype(np.float32)
    return result, {
        "fit_n": int(len(fit_idx)),
        "selected_dim": int(len(keep)),
        "selected_indices_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
        "transform": "log2(x+1), median imputation, variance top-512, fit-index mean/std",
        "fit_indices_sha256": hashlib.sha256(np.asarray(fit_idx, dtype=np.int64).tobytes()).hexdigest(),
    }


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError("nonfinite or zero vector")
    return x / norm


@dataclass
class FitResult:
    model: Any
    selected_epoch: int
    selected_lambda: float
    inner_score: float


def seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallHead:
    """Torch model wrapper built lazily to keep CPU import and audit simple."""

    def __init__(self, method: str, n_classes: int, lambda_prompt: float, seed: int, device: str):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.method = method
                self.lambda_prompt = float(lambda_prompt)
                if method in {"rna_conditioned", "static_concepts_plus_rna"}:
                    self.adapter = nn.Linear(512, 16)
                if method == "rna_conditioned":
                    self.basis = nn.Parameter(torch.randn(16, 768) * 0.005)
                    self.gate = nn.Linear(16, 32 * 16)
                # Dynamic prompting still feeds the classifier one score per
                # concept (32).  Only the additive control concatenates the
                # 16-dimensional RNA adapter with the 32 static scores.
                dim = 48 if method == "static_concepts_plus_rna" else 32
                self.head = nn.Linear(dim, n_classes)

            def features(self, image, static, rna, text):
                if self.method == "static_concepts":
                    return static
                c = self.adapter(rna)
                if self.method == "static_concepts_plus_rna":
                    return torch.cat([static, c], dim=1)
                gate = self.gate(c).view(-1, 32, 16)
                alpha = torch.softmax(gate, dim=-1)
                delta = torch.einsum("nkm,md->nkd", alpha, self.basis)
                dynamic_text = torch.nn.functional.normalize(text.unsqueeze(0) + self.lambda_prompt * delta, dim=-1)
                scores = torch.einsum("nd,nkd->nk", image, dynamic_text)
                return scores

            def forward(self, image, static, rna, text):
                return self.head(self.features(image, static, rna, text))

        seed_everything(seed)
        self.net = Net().to(device)
        self.device = device

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def class_weights(y: np.ndarray, n_classes: int, device: str):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict(model: SmallHead, image, static, rna, text) -> np.ndarray:
    import torch

    model.net.eval()
    with torch.no_grad():
        logits = model.net(image, static, rna, text)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()


def train_to_epochs(method: str, image_np: np.ndarray, static_np: np.ndarray, rna_np: np.ndarray,
                    text_np: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                    n_classes: int, seed: int, lambda_prompt: float, max_epoch: int,
                    checkpoints: set[int], device: str) -> tuple[SmallHead, dict[int, np.ndarray]]:
    import torch
    import torch.nn.functional as F

    model = SmallHead(method, n_classes, lambda_prompt, seed, device)
    model.net.train()
    image = torch.tensor(image_np, dtype=torch.float32, device=device)
    static = torch.tensor(static_np, dtype=torch.float32, device=device)
    rna = torch.tensor(rna_np, dtype=torch.float32, device=device)
    text = torch.tensor(text_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=0.01, weight_decay=1e-3)
    weight = class_weights(y[train_idx], n_classes, device)
    outputs: dict[int, np.ndarray] = {}
    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model.net(image[idx], static[idx], rna[idx], text)
        loss = F.cross_entropy(logits, y_t[idx], weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.0)
        optimizer.step()
        if epoch in checkpoints:
            # A copy of the trained weights is not needed: selection happens in
            # the caller immediately and the next candidate starts from seed.
            outputs[epoch] = predict(model, image, static, rna, text)
    return model, outputs


def choose_inner(method: str, image: np.ndarray, static: np.ndarray, rna_raw: np.ndarray,
                 text: np.ndarray, y: np.ndarray, outer_train: np.ndarray, n_classes: int,
                 seed: int, epochs: list[int], lambdas: list[float], inner_folds: int,
                 device: str) -> tuple[int, float, float, list[dict[str, Any]]]:
    from sklearn.model_selection import StratifiedKFold

    labels = y[outer_train]
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(outer_train, labels))
    candidates = lambdas if method == "rna_conditioned" else [0.0]
    scores: list[dict[str, Any]] = []
    for lam in candidates:
        # One max-epoch training per inner split and lambda.  The checkpoint
        # scores at 40/80/120 are read from the same trajectory, so schedule
        # selection is consistent and the nested run remains tractable.
        per_epoch: dict[int, list[float]] = {int(epoch): [] for epoch in epochs}
        for inner_id, (tr_rel, va_rel) in enumerate(splits):
            inner_tr = outer_train[tr_rel]
            inner_va = outer_train[va_rel]
            # Fit RNA standardisation and feature ranking on inner-train
            # only.  This is the key leakage guard for the upstream main
            # effects.
            rna_all, _ = fit_numeric(rna_raw, inner_tr)
            _, outputs = train_to_epochs(method, image, static, rna_all, text, y,
                                          inner_tr, n_classes, seed + inner_id * 1009 + int(lam * 1000),
                                          lam, max(epochs), set(epochs), device)
            for epoch in epochs:
                pred = outputs[epoch][inner_va]
                per_epoch[int(epoch)].append(float(balanced_accuracy_score(y[inner_va], pred.argmax(1))))
        for epoch in epochs:
            vals = per_epoch[int(epoch)]
            scores.append({"method": method, "lambda_prompt": float(lam), "epoch": int(epoch),
                           "inner_balanced_accuracy": float(np.mean(vals)),
                           "inner_fold_scores": vals,
                           "preprocess_fit_scope": "inner-train only"})
    # Deterministic tie breaking: higher mean, smaller lambda, earlier epoch.
    winner = sorted(scores, key=lambda x: (-x["inner_balanced_accuracy"], x["lambda_prompt"], x["epoch"]))[0]
    return int(winner["epoch"]), float(winner["lambda_prompt"]), float(winner["inner_balanced_accuracy"]), scores


def fit_outer(method: str, image: np.ndarray, static: np.ndarray, rna_raw: np.ndarray, text: np.ndarray,
              y: np.ndarray, outer_train: np.ndarray, outer_test: np.ndarray, n_classes: int,
              seed: int, epoch: int, lam: float, device: str) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    rna_all, transform_audit = fit_numeric(rna_raw, outer_train)
    model, _ = train_to_epochs(method, image, static, rna_all, text, y, outer_train, n_classes,
                               seed, lam, epoch, {epoch}, device)
    matched = predict(model, _tensor(image[outer_test], device), _tensor(static[outer_test], device),
                      _tensor(rna_all[outer_test], device), _tensor(text, device))
    wrong_order = np.roll(np.arange(len(outer_test)), 1 + (seed + int(outer_test[0])) % max(1, len(outer_test) - 1))
    wrong = predict(model, _tensor(image[outer_test], device), _tensor(static[outer_test], device),
                    _tensor(rna_all[outer_test][wrong_order], device), _tensor(text, device))
    return matched, wrong, model.parameter_count(), transform_audit


def _tensor(x: np.ndarray, device: str):
    import torch
    return torch.tensor(x, dtype=torch.float32, device=device)


def metrics(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = probs.argmax(1)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
    }
    try:
        result["macro_ovr_auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=np.arange(n_classes)))
    except ValueError:
        result["macro_ovr_auroc"] = float("nan")
    return result


def bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_classes: int,
                    reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        values[i] = balanced_accuracy_score(y[idx], p_a[idx].argmax(1)) - balanced_accuracy_score(y[idx], p_b[idx].argmax(1))
    lo, hi = np.quantile(values, [0.00625, 0.99375])
    return {"delta_balanced_accuracy": float(np.mean(values)), "ci98_75_low": float(lo), "ci98_75_high": float(hi),
            "bootstrap_replicates": int(reps), "bootstrap_seed": int(seed)}


def build_report(out: Path, validation: dict[str, Any], metric_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> None:
    lines = ["# Phase6B RNA-conditioned TITAN concept audit", "",
             "本实验使用 Phase6A 官方 TITAN 输出空间的冻结患者图像表示和 32 个冻结 CRC 形态概念原型。RNA 只通过 512→16 的小适配器进入输出空间 prompt basis；TITAN 图像/文本编码器均未训练。", "",
             "## Protocol", "",
             "- 3 seeds × 5 outer folds；每个 outer-train 内再做 3-fold inner selection。",
             "- RNA 特征选择、填补和标准化均只在 inner-train 或最终 outer-train 拟合。",
             "- 只读取 T/N/M/Stage；未读取 DSS/OS 生存字段。",
             "- 四个机制对照：static concepts、RNA-conditioned concepts、static concepts + RNA additive、matched model 的 wrong-patient RNA。", "",
             "## Interpretation rule", "",
             "如果 RNA-conditioned 同时优于 static+RNA additive，且 matched 明显优于 wrong-patient，才支持 RNA 改变 pathology concept interpretation。若只优于 static 而不优于 additive，则只能归因于 RNA 提供额外信息。", "",
             "## Validation", "", f"```json\n{json.dumps(validation, indent=2, ensure_ascii=False)}\n```", "",
             "## Aggregate results", "", "结果明细见 `metrics.csv` 和 `paired_comparisons.csv`。"]
    if metric_rows:
        lines += ["", "| Endpoint | Method | Seed mean BA | Seed BA values |", "|---|---|---:|---|"]
        for endpoint in ENDPOINTS:
            for method in ("static_concepts", "rna_conditioned", "static_concepts_plus_rna", "wrong_patient_rna_conditioned"):
                rows = [r for r in metric_rows if r["endpoint"] == endpoint and r["method"] == method]
                if not rows:
                    continue
                vals = [r["balanced_accuracy"] for r in rows]
                lines.append(f"| {endpoint} | {method} | {np.mean(vals):.4f} | " + ", ".join(f"{v:.4f}" for v in vals) + " |")
    if paired_rows:
        lines += ["", "## Paired mechanism comparisons", "", "| Endpoint | Comparison | Mean ΔBA | 98.75% CI |", "|---|---|---:|---|"]
        for r in paired_rows:
            lines.append(f"| {r['endpoint']} | {r['comparison']} | {r['delta_balanced_accuracy']:.4f} | [{r['ci98_75_low']:.4f}, {r['ci98_75_high']:.4f}] |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clinical-targets", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oof").mkdir(exist_ok=True)
    started = time.time()

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Phase6B is required to run on the server GPU")
    torch.set_float32_matmul_precision("high")
    device = args.device
    rep = np.load(args.cache / "patient_representations.npz", allow_pickle=False)
    required = {"patient_ids", "semantic", "concept_cosine", "text_prototypes", "concept_ids"}
    if not required.issubset(rep.files):
        raise ValueError(f"missing representation keys: {required - set(rep.files)}")
    ids = rep["patient_ids"].astype(str)
    image = unit(rep["semantic"].astype(np.float32))
    static = rep["concept_cosine"].astype(np.float32)
    text = unit(rep["text_prototypes"].astype(np.float32))
    concepts = rep["concept_ids"].astype(str)
    if image.shape != (len(ids), 768) or static.shape != (len(ids), 32) or text.shape != (32, 768):
        raise ValueError("Phase6A representation shapes do not match locked 32×768 protocol")
    if not np.isfinite(image).all() or not np.isfinite(static).all() or not np.isfinite(text).all():
        raise ValueError("nonfinite Phase6A representation")

    folds = pd.read_csv(args.folds, dtype={"patient_id": str, "fold": int})
    if set(folds.columns) != {"patient_id", "fold"}:
        raise ValueError("fold file must contain only patient_id,fold")
    folds["patient_id"] = folds["patient_id"].str.upper()
    if set(folds.patient_id) != set(ids) or folds.patient_id.duplicated().any():
        raise ValueError("fixed fold patients do not exactly match Phase6A representations")
    fold_map = dict(zip(folds.patient_id, folds.fold))
    fold_ids = np.array([fold_map[x] for x in ids], dtype=int)
    if set(fold_ids) != {1, 2, 3, 4, 5}:
        raise ValueError("fixed split must contain five folds numbered 1..5")

    targets_path = args.clinical_targets or (args.cache / "clinical_targets.csv")
    targets = pd.read_csv(targets_path, dtype=str)
    targets["patient_id"] = targets["patient_id"].str.upper()
    if set(targets.patient_id) != set(ids):
        raise ValueError("target IDs do not match Phase6A representations")
    targets = targets.set_index("patient_id").loc[ids]
    raw_rna, rna_audit = load_rna_only(args.rna, list(ids))

    config = {
        "version": "phase6b_rna_conditioned_titan_concepts_v1",
        "seeds": [20261007, 20261008, 20261009],
        "outer_folds": 5,
        "inner_folds": 3,
        "targets": list(ENDPOINTS),
        "methods": list(METHODS) + ["wrong_patient_rna_conditioned"],
        "schedule": {"max_epochs": 120, "checkpoint_epochs": [40, 80, 120], "optimizer": "AdamW(lr=0.01, weight_decay=0.001)", "selection": "inner-train fit, inner-val balanced accuracy"},
        "rna_adapter": "fold-local log2(x+1) variance-top-512 -> Linear(512,16)",
        "prompt_adapter": "output-space basis 16x768 plus 32x16x16 gate; lambda candidates [0.15,0.30]",
        "frozen": ["TITAN image encoder", "TITAN text encoder", "Phase6A image representations", "Phase6A text prototypes"],
        "concept_vocabulary": "Phase6A outcome-blind 32 CRC morphology concepts; no Hallmark or survival terms",
        "wrong_patient": "test-only cyclic derangement of RNA within each outer-test fold; matched model unchanged",
        "survival_columns_read": [],
        "clinical_columns_used": ["T", "N", "M", "Stage"],
    }
    write_json(args.out / "config.json", config)
    write_json(args.out / "input_audit.json", {"representation_cache_sha256": sha256(args.cache / "patient_representations.npz"),
                                                  "rna": rna_audit, "fold_file_sha256": sha256(args.folds),
                                                  "patient_count": len(ids), "concept_ids": concepts.tolist()})

    all_metric_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    parameter_counts: dict[str, int] = {}
    max_epochs = [40, 80, 120]
    seeds = config["seeds"]

    for seed in seeds:
        seed_metric_store: dict[tuple[str, str], list[np.ndarray]] = {}
        seed_y_store: dict[str, list[np.ndarray]] = {}
        for endpoint in ENDPOINTS:
            labels = [normalize_label(v, endpoint) for v in targets[endpoint]]
            valid = np.array([v is not None for v in labels])
            classes = sorted({v for v in labels if v is not None})
            class_to_int = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_int[v] if v is not None else -1 for v in labels], dtype=int)
            if valid.sum() < 30 or min(Counter(y[valid]).values()) < 5:
                raise ValueError(f"{endpoint} target has too few valid samples/classes")
            outer_predictions = {method: np.full((len(ids), len(classes)), np.nan, dtype=np.float32) for method in ("static_concepts", "rna_conditioned", "static_concepts_plus_rna", "wrong_patient_rna_conditioned")}
            y_oof = np.full(len(ids), -1, dtype=int)
            for fold in range(1, 6):
                outer_test = np.flatnonzero((fold_ids == fold) & valid)
                outer_train = np.flatnonzero((fold_ids != fold) & valid)
                if len(outer_test) == 0:
                    continue
                for method in METHODS:
                    epoch, lam, inner_score, selection_detail = choose_inner(method, image, static, raw_rna, text, y, outer_train,
                                                                              len(classes), seed + fold * 10000, max_epochs,
                                                                              [0.15, 0.30], 3, device)
                    selection_rows.append({"seed": seed, "endpoint": endpoint, "outer_fold": fold, "method": method,
                                           "selected_epoch": epoch, "selected_lambda": lam, "inner_balanced_accuracy": inner_score,
                                           "selection_detail": json.dumps(selection_detail, ensure_ascii=False)})
                    matched, wrong, count, transform_audit = fit_outer(method, image, static, raw_rna, text, y, outer_train, outer_test,
                                                                          len(classes), seed + fold * 10000, epoch, lam, device)
                    parameter_counts[method] = count
                    outer_predictions[method][outer_test] = matched
                    if method == "rna_conditioned":
                        outer_predictions["wrong_patient_rna_conditioned"][outer_test] = wrong
                    if method != "rna_conditioned":
                        # A wrong-patient dynamic prediction is defined only by
                        # the matched dynamic model; these slots remain empty.
                        pass
                    np.savez_compressed(args.out / "oof" / f"{endpoint}__{method}__seed{seed}__fold{fold}.npz",
                                        patient_ids=ids[outer_test], y_true=y[outer_test],
                                        probabilities=matched if method != "rna_conditioned" else matched,
                                        wrong_probabilities=wrong if method == "rna_conditioned" else np.empty((0, len(classes))),
                                        classes=np.asarray(classes, dtype=str), selected_epoch=np.asarray(epoch), selected_lambda=np.asarray(lam))
                y_oof[outer_test] = y[outer_test]
            valid_oof = y_oof >= 0
            for method, probs in outer_predictions.items():
                p = probs[valid_oof]
                if not np.isfinite(p).all():
                    raise RuntimeError(f"missing OOF predictions for {endpoint}/{method}/seed={seed}")
                row = {"seed": seed, "endpoint": endpoint, "method": method, "n": int(valid_oof.sum()), **metrics(y_oof[valid_oof], p, len(classes))}
                all_metric_rows.append(row)
                seed_metric_store[(endpoint, method)] = seed_metric_store.get((endpoint, method), []) + [p]
                seed_y_store[endpoint] = seed_y_store.get(endpoint, []) + [y_oof[valid_oof]]
            # Explicit paired controls for this seed are evaluated on identical
            # patients, then pooled as seed-level OOF metrics below.
            for comparison, a, b in [("rna_conditioned - static_concepts", "rna_conditioned", "static_concepts"),
                                      ("rna_conditioned - static_concepts_plus_rna", "rna_conditioned", "static_concepts_plus_rna"),
                                      ("matched - wrong_patient", "rna_conditioned", "wrong_patient_rna_conditioned")]:
                pair_seed = int.from_bytes(hashlib.sha256(f"{endpoint}|{comparison}".encode()).digest()[:4], "little")
                result = bootstrap_delta(y_oof[valid_oof], outer_predictions[a][valid_oof], outer_predictions[b][valid_oof], len(classes),
                                         2000, seed + pair_seed % 100000)
                all_pair_rows.append({"seed": seed, "endpoint": endpoint, "comparison": comparison, **result})

    pd.DataFrame(all_metric_rows).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(all_pair_rows).to_csv(args.out / "paired_comparisons_by_seed.csv", index=False)
    pd.DataFrame(selection_rows).drop(columns=["selection_detail"]).to_csv(args.out / "selection_summary.csv", index=False)
    write_json(args.out / "parameter_counts.json", parameter_counts)
    validation = {"status": "PASS", "patients": len(ids), "seeds": len(seeds), "outer_folds": 5,
                  "inner_folds": 3, "metrics_rows": len(all_metric_rows), "pair_rows": len(all_pair_rows),
                  "oof_files": len(list((args.out / "oof").glob("*.npz"))), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  "elapsed_seconds": time.time() - started, "survival_columns_read": [],
                  "server_weight_copy_used": False, "patient_level_outputs_published": False}
    write_json(args.out / "validation.json", validation)
    build_report(args.out, validation, all_metric_rows, all_pair_rows)
    write_json(args.out / "DONE.json", {"status": "PASS", "validation": validation, "config_sha256": sha256(args.out / "config.json")})
    print(json.dumps({"status": "PASS", "metrics_rows": len(all_metric_rows), "pair_rows": len(all_pair_rows), "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
