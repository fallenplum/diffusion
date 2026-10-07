"""Validate actual Phase6A completion and seed OOF caches before publication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(r"E:\diffusion_phase6a_titan\server_cache_v1"))
    parser.add_argument("--report", type=Path, default=ROOT/"reports/phase6a_titan_semantic_preflight")
    args = parser.parse_args()
    done = json.loads((args.report/"DONE.json").read_text())
    lock = json.loads((args.cache/"protocol_lock.json").read_text())
    config = json.loads((ROOT/"configs/phase6a_titan_semantic_preflight.json").read_text())
    assert lock["script_sha256"] == digest(ROOT/"scripts/phase6a_titan_semantic_preflight.py")
    assert lock["memory_helper_sha256"] == digest(ROOT/"scripts/phase6a_titan_memory.py")
    assert lock["protocol_sha256"] == digest(ROOT/"configs/phase6a_titan_semantic_preflight.json")
    assert lock["vocabulary_sha256"] == digest(ROOT/"configs/phase6a_titan_crc_morphology_concepts.json")
    assert done["protocol_lock"] == lock
    expected = pd.read_csv(ROOT/"reports/phase6_conch_v15_download_audit/cohort_manifest.csv", dtype=str)
    matched = expected.loc[expected.status=="EXACT_SLIDE_MATCH"]
    bundle = np.load(args.cache/"patient_representations.npz", allow_pickle=False)
    ids = bundle["patient_ids"].tolist()
    assert len(ids) == len(set(ids)) == 512
    assert ids == sorted(expected.patient_id.unique())
    for key in ["native", "semantic", "concept_cosine", "text_prototypes"]:
        assert np.isfinite(bundle[key]).all()
    assert bundle["native"].shape == bundle["semantic"].shape == (512, 768)
    assert bundle["concept_cosine"].shape == (512, 32)
    assert np.allclose(np.linalg.norm(bundle["native"], axis=1), 1, atol=2e-6)
    assert np.allclose(np.linalg.norm(bundle["semantic"], axis=1), 1, atol=2e-6)
    assert np.allclose(bundle["concept_cosine"], bundle["semantic"]@bundle["text_prototypes"].T, atol=2e-6)
    for row in matched.itertuples(index=False):
        cache = np.load(args.cache/"slides"/(Path(row.filename).stem+".npz"), allow_pickle=False)
        assert str(cache["source_sha256"].item()) == row.conch_sha256
        assert cache["native"].shape == (768,) and np.isfinite(cache["native"]).all()
    checks = {"status": "PASS", "protocol_and_code_hashes_match": True,
              "verified_patient_embeddings": len(ids), "verified_slide_caches": len(matched),
              "patient_cosine_recomputed": True, "duplicate_patients": 0,
              "patient_representations_sha256": digest(args.cache/"patient_representations.npz")}
    if done["clinical_evaluation_performed"]:
        frame = pd.read_csv(args.report/"seed_oof_metrics.csv")
        fit = pd.read_csv(args.report/"fit_manifest.csv")
        assert len(frame) == 4*3*4
        assert len(fit) == 4*3*5*4
        comparisons = json.loads((args.report/"paired_comparisons.json").read_text())
        assert len(comparisons) == 4*3
        verified = 0
        for endpoint in config["evaluation"]["targets"]:
            for seed in config["evaluation"]["seeds"]:
                reference_folds = None
                reference_ids = None
                for model in config["evaluation"]["models"]:
                    cache = np.load(args.cache/"oof"/f"{endpoint}_{seed}_{model}.npz", allow_pickle=False)
                    probability, y, fold = cache["probabilities"], cache["target"], cache["outer_fold"]
                    assert len(cache["patient_ids"]) == len(set(cache["patient_ids"].tolist()))
                    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
                    assert set(fold) == set(range(5))
                    if reference_folds is None:
                        reference_folds, reference_ids = fold, cache["patient_ids"]
                    assert np.array_equal(reference_folds, fold)
                    assert np.array_equal(reference_ids, cache["patient_ids"])
                    pred = probability.argmax(axis=1)
                    observed = frame.loc[(frame.endpoint==endpoint)&(frame.seed==seed)&(frame.model==model)].iloc[0]
                    recomputed = {"balanced_accuracy": balanced_accuracy_score(y,pred), "accuracy": accuracy_score(y,pred),
                                  "macro_f1": f1_score(y,pred,average="macro",zero_division=0),
                                  "macro_ovr_auroc": roc_auc_score(y,probability[:,1]) if probability.shape[1]==2 else roc_auc_score(y,probability,multi_class="ovr",average="macro")}
                    assert all(abs(float(observed[key])-value)<1e-10 for key,value in recomputed.items())
                    verified += 1
        checks.update({"verified_seed_oof_caches": verified, "seed_metrics_recomputed_from_predictions": True,
                       "identical_patient_folds_across_models": True, "fit_manifest_rows": len(fit)})
    else:
        assert done["status"] == "STOPPED_AT_COLLAPSE_GATE"
        assert not list((args.cache/"oof").glob("*.npz"))
        checks["clinical_omitted_due_to_registered_gate"] = True
    transfer = json.loads((args.report/"transfer_audit.json").read_text())
    assert transfer["server_weight_copy_removed"] and not transfer["clinical_labels_uploaded"]
    checks["temporary_server_weight_removed"] = True
    (args.report/"validation.json").write_text(json.dumps(checks, indent=2)+"\n", newline="\n")
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
