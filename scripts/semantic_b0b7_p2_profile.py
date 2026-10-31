"""Real zero-update Phase-A resource qualification, never reading endpoints.

The private scope file contains only previously locked patient row indices.
FLOPs are PyTorch profiler's supported matrix operators, reported separately
for forward and forward-plus-backward (including checkpoint recomputation).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import legacy_data_module, file_sha, read_json
from scripts.semantic_b0b7_p1r import make_model, data_batch, canonical_hash, tensor_hash
from scripts.semantic_b0b7_p1 import configure_runtime, train_common_support, preserve_replay_rng
from src.semantic_b0b7.p1r_data import build_partition_inputs, molecular_memberships
from src.semantic_b0b7.p1r_models import RepairSpec
from src.semantic_b0b7.pretext import auxiliary_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "legacy-root", "frozen-images", "scope", "output-newdir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-frozen-sha256", required=True)
    p.add_argument("--expected-scope-sha256", required=True)
    a = p.parse_args()
    if a.output_newdir.exists():
        raise ValueError("REFUSE_PROFILE_OVERWRITE")
    if file_sha(a.frozen_images) != a.expected_frozen_sha256 or file_sha(a.scope) != a.expected_scope_sha256:
        raise ValueError("PROFILE_INPUT_OR_SCOPE_HASH_CHANGED")
    torch.set_num_threads(2)
    runtime = configure_runtime()
    assets = read_json(a.assets)
    old, _ = legacy_data_module(a.legacy_root)
    raw = old.load_raw(assets, include_endpoints=False)
    memberships = molecular_memberships(raw, assets["gmt"])
    frozen = torch.load(a.frozen_images, map_location="cpu", weights_only=False)
    frozen["frozen_input_sha256"] = a.expected_frozen_sha256
    scopes = read_json(a.scope)
    if set(scopes) != {"inner", "outer_final"}:
        raise ValueError("EXACT_TWO_PRELOCKED_TRAIN_SCOPES_REQUIRED")
    specs = {
        "B4_DUAL": RepairSpec(relation_mode="dual", known_fraction=.5),
        "B2_MATCHED": RepairSpec(source_arm="B2_MATCHED", query_mode="visual_random"),
        "B2_NATIVE": RepairSpec(source_arm="B2_NATIVE", query_mode="visual_random")}
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha(path) for path in (
        Path(__file__), ROOT / "src/semantic_b0b7/p1r_models.py", ROOT / "src/semantic_b0b7/p2_models.py",
        ROOT / "scripts/semantic_b0b7_p1r.py", ROOT / "src/semantic_b0b7/p1r_data.py")}
    a.output_newdir.mkdir(parents=True)
    (a.output_newdir / "LOCK_BEFORE_PROFILE.json").write_text(json.dumps({
        "sources": sources, "scope_sha256": a.expected_scope_sha256,
        "frozen_input_sha256": a.expected_frozen_sha256, "specs": {k: asdict(v) for k, v in specs.items()},
        "optimizer_updates": 0, "endpoint_access": False, "runtime": runtime}, indent=2), encoding="utf-8")
    results = []
    for role, scope in scopes.items():
        data, _ = build_partition_inputs(raw, frozen, scope["fit"], scope["evaluation"], memberships,
                                         split_name="zero_step_profile_" + role)
        tr = data["train_indices"]
        binding = canonical_hash({k: tensor_hash(data[k][tr]) for k in ("target", "target_mask")})
        for name, spec in specs.items():
            torch.manual_seed(31)
            torch.cuda.manual_seed_all(31)
            model = make_model(data, spec, 31, tr, torch.device("cuda"))
            train, wrong, support = train_common_support(data_batch(data, tr, "cuda"), 31)
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            # CPU dispatcher events retain CUDA tensor shapes and supported
            # matrix FLOPs without requiring privileged CUPTI on a vGPU.
            # CUDA execution is timed by synchronized wall time, not CPU time.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                out = model(train[0], train[1])
                loss = auxiliary_loss(out, train[2], train[3], train[5], lambda_pair=.1)["loss"]
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - started
                loss.backward()
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            gradients = {k: float(v.grad.norm().cpu()) if v.grad is not None else None
                         for k, v in model.named_parameters()}
            if not torch.isfinite(loss) or any(v is None or not np.isfinite(v) or v == 0 for v in gradients.values()):
                raise ValueError("REAL_PROFILE_INACTIVE_OR_NONFINITE_PARAMETER")
            peak = torch.cuda.max_memory_allocated()
            total_flops = int(sum(x.flops or 0 for x in prof.key_averages()))
            model.eval()
            with torch.no_grad():
                # Profile forward separately: no backward recomputation counted.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as inference_prof:
                    expected = model(train[0], train[1])
                expected = {k: v.detach().cpu() for k, v in {
                    "program_prediction": expected["program_prediction"], "scores": expected["pair"]["scores"]}.items()}
            path = a.output_newdir / (role + "_" + name + ".private.pt")
            torch.save({"model": model.state_dict(), "spec": asdict(spec), "input_binding": binding}, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            with preserve_replay_rng("cuda"):
                replay = make_model(data, spec, 31, tr, torch.device("cuda"))
                replay.load_state_dict(saved["model"], strict=True)
                replay.eval()
                with torch.no_grad():
                    actual = replay(train[0], train[1])
                if not torch.equal(expected["program_prediction"], actual["program_prediction"].cpu()) or not torch.equal(expected["scores"], actual["pair"]["scores"].cpu()):
                    raise ValueError("REAL_PROFILE_ACTUAL_STATE_RELOAD_MISMATCH")
                del actual, replay
            results.append({"role": role, "arm": name, "train_patients": len(tr), "parameters": sum(v.numel() for v in model.parameters()),
                "gradient_norms": gradients, "forward_profile_seconds": forward_seconds,
                "forward_backward_profile_seconds": elapsed, "peak_cuda_allocated_bytes": peak,
                "forward_supported_matrix_FLOPs": int(sum(x.flops or 0 for x in inference_prof.key_averages())),
                "forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs": total_flops,
                "FLOPs_exclude_unsupported_elementwise_operators": True,
                "profiler_event_collection": "CPU_dispatcher_shapes_for_real_CUDA_execution_no_CUPTI_trace",
                "checkpoint_sha256": file_sha(path), "actual_saved_reload_forward_replay": True,
                "optimizer_steps": 0, "new_neural_fits": 0,
                "negative_support": support, "architecture": model.architecture_descriptor() if hasattr(model, "architecture_descriptor") else None})
            print(json.dumps({k: results[-1][k] for k in ("role", "arm", "parameters", "forward_backward_profile_seconds", "peak_cuda_allocated_bytes")}), flush=True)
            del model, out, loss, train, saved, expected
            torch.cuda.empty_cache()
        del data
    report = {"status": "PASS_REAL_SIX_ZERO_UPDATE_MODEL_PROFILES", "sources": sources,
              "frozen_input_sha256": a.expected_frozen_sha256, "scope_sha256": a.expected_scope_sha256,
              "optimizer_updates": 0, "new_neural_fits": 0, "endpoints_read": False,
              "actual_saved_state_replays": len(results), "profiles": results}
    (a.output_newdir / "PROFILE.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
