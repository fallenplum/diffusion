"""Read-only, preprocessing-only reproduction of P2 input hash sensitivity.

This parameterized public reproducer was not the source piped to the four
original diagnostic processes. Their separate source hashes and actual results
are recorded in the accompanying diagnostic receipt. Set BLAS environment
variables before starting Python; this program does not repair its environment.
It never loads a trained checkpoint, endpoint, risk vector, or predictive model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal2_torch2", "formal2_torch1",
        "ambient_numpyfirst_torch1", "formal1_torch1"), required=True)
    for name in ("repository", "private", "lock", "expected-lock-sha256", "assets",
                 "legacy-root", "frozen-images", "failed-private-output", "failed-public-output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--compare-blas-threads", type=int)
    args = parser.parse_args()
    before = {k: os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    numpy_preloaded = "numpy" in sys.modules
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    repo = Path(args.repository).resolve()
    sys.path.insert(0, str(repo))
    from scripts import semantic_b0b7_p2_matrix as matrix
    torch_threads = 2 if args.mode == "formal2_torch2" else 1
    torch.set_num_threads(torch_threads)
    counts = {name: 0 for name in ("neural_fit", "teacher_Cox_fit", "risk_Cox_fit",
        "model_load", "neural_feature_forward", "DSS_read", "risk_scoring", "PCA_fit")}

    def forbidden(name):
        def blocked(*unused_args, **unused_kwargs):
            counts[name] += 1
            raise RuntimeError("FORBIDDEN_IN_PREPROCESSING_PROBE:" + name)
        return blocked

    for name, attribute in (("neural_fit", "run_cell"), ("teacher_Cox_fit", "fit_r4_anchor"),
            ("risk_Cox_fit", "fit_risk_cell"), ("model_load", "load_checkpoint_model"),
            ("neural_feature_forward", "extract_frozen_features"), ("DSS_read", "read_dss_only"),
            ("risk_scoring", "evaluate_safe")):
        setattr(matrix, attribute, forbidden(name))
    from sklearn.decomposition import PCA
    PCA.fit = forbidden("PCA_fit")
    PCA.fit_transform = forbidden("PCA_fit")
    private = Path(args.private)
    a = SimpleNamespace(lock=args.lock, expected_lock_sha256=args.expected_lock_sha256,
        assets=args.assets, legacy_root=args.legacy_root, frozen_images=args.frozen_images,
        private=str(private), split="outer1_final", family="MAIN", seed=31, threads=torch_threads)
    original_load = torch.load

    def feature_only_load(path, *load_args, **load_kwargs):
        if Path(path).resolve() != Path(a.frozen_images).resolve():
            raise RuntimeError("MODEL_OR_UNKNOWN_TORCH_FILE_FORBIDDEN")
        return original_load(path, *load_args, **load_kwargs)

    torch.load = feature_only_load
    master = matrix.verify_lock(a)
    phase_dir = private / "phase_a/outer1_final/MAIN_seed31"
    phase = json.loads((phase_dir / "cell.private.json").read_text())
    binding = json.loads((phase_dir / "MATRIX_BINDING.private.json").read_text())
    if phase["input_sha256"] != binding["training_only_input_sha256"]:
        raise RuntimeError("SAVED_FORMAL_INPUT_BINDING_DISAGREES")
    start = time.monotonic()
    data, preprocessing, split, input_sha = matrix.build_inputs(a, master)
    train = data["train_indices"]
    components = {section + "." + key: matrix.tensor_hash(value[train])
        for section in ("image", "omics") for key, value in data[section].items() if value is not None}
    for key in ("target", "target_mask"):
        components[key] = matrix.tensor_hash(data[key][train])
    for key, value in data["metadata"].items():
        components["metadata." + key] = matrix.tensor_hash(np.asarray(value)[train.numpy()])
    channels = {name: matrix.tensor_hash(data["omics"]["values"][train, :, j])
        for j, name in enumerate(("RNA", "CNA_gain", "CNA_loss"))}
    transform = preprocessing["program_transform"]
    scales = {name + "." + field: matrix.tensor_hash(np.asarray(getattr(scale, field)))
        for name, scale in (("RNA", transform.rna_scale), ("gain", transform.gain_scale),
                            ("loss", transform.loss_scale))
        for field in ("mean", "sd", "measured")}

    def pools():
        return [{k: pool.get(k) for k in ("user_api", "internal_api", "prefix", "version",
                                         "num_threads", "architecture")}
                for pool in threadpool_info()]

    result = {"mode": args.mode, "process_environment_before_numpy": before,
        "numpy_preloaded_by_python_startup": numpy_preloaded,
        "process_environment_after_matrix_import": {k: os.environ.get(k) for k in before},
        "torch_cpu_threads": torch.get_num_threads(), "threadpools": pools(),
        "actual_input_sha256": input_sha, "saved_formal_input_sha256": phase["input_sha256"],
        "exact_saved_input_match": input_sha == phase["input_sha256"],
        "training_tensor_component_sha256": components, "program_value_channel_sha256": channels,
        "program_scale_component_sha256": scales,
        "fit_rows_hash": data["provenance"]["fit_rows_sha256"],
        "molecular_fit_hash": data["provenance"]["molecular_fit_hash"],
        "clinical_fit_hash": data["provenance"]["clinical_fit_hash"],
        "cohort_input_hash": data["provenance"]["raw_container_sha256"],
        "fit_n": len(train), "evaluation_n": len(data["valid_indices"]),
        "outcomes_loaded": data["provenance"]["outcomes_loaded"],
        "forbidden_call_counts": counts, "elapsed_seconds": time.monotonic() - start,
        "failed_original_private_output_exists": Path(args.failed_private_output).exists(),
        "failed_original_public_output_exists": Path(args.failed_public_output).exists(),
        "weights_or_predictions_read": False, "source_or_data_files_changed": False,
        "public_reproducer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.compare_blas_threads is not None:
        if args.compare_blas_threads < 1:
            raise ValueError("positive BLAS thread count required")
        with threadpool_limits(limits=args.compare_blas_threads, user_api="blas"):
            changed, _, _, changed_sha = matrix.build_inputs(a, master)
        deltas = {}
        for section, key in (("omics", "values"), ("root", "target")):
            first = (data[section][key] if section != "root" else data[key])[train].numpy()
            second = (changed[section][key] if section != "root" else changed[key])[train].numpy()
            d = np.abs(first.astype(np.float64) - second.astype(np.float64))
            deltas[section + "." + key] = {"shape": list(first.shape),
                "different_values": int(np.count_nonzero(first != second)), "total_values": int(first.size),
                "max_absolute_difference": float(d.max()), "mean_absolute_difference": float(d.mean()),
                "finite_both": bool(np.isfinite(first).all() and np.isfinite(second).all())}
        result["same_process_changed_blas_threads"] = args.compare_blas_threads
        result["same_process_changed_blas_input_sha256"] = changed_sha
        result["same_process_changed_blas_value_differences"] = deltas
        result["same_process_changed_blas_only_training_components_changed"] = [section + "." + key
            for section in ("image", "omics") for key in data[section]
            if matrix.tensor_hash(data[section][key][train]) != matrix.tensor_hash(changed[section][key][train])]
        result["threadpools_restored"] = pools()
    if any(counts.values()) or result["outcomes_loaded"]:
        raise RuntimeError("PREPROCESSING_ONLY_PROBE_SCOPE_VIOLATED")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
