"""P1-R saved-state attribution and locked frozen-encoder Ridge refits.

Only the already qualified outcome-free P1 cache is accepted. R0 performs
actual inference and separate auxiliary-loss backpropagation, with no optimizer.
T2 keeps historical encoder selection fixed and fits training-only Ridge heads.
Private arrays, coefficients and per-patient predictions remain at --output.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p1 as original
from src.semantic_b0b7.models import OmicsInputs, masked_softmax
from src.semantic_b0b7.pretext import auxiliary_loss

ARMS = ("B3", "B4", "B6a", "B4_MAP_SHUFFLE")
SEEDS = (31, 47, 83)
EPOCHS = (50, 100, 200, 400)
HEADS = ("T2_mean64", "T2_keep_slots", "T2_capacity_control")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    if isinstance(value, (str, Path)):
        return original.sha_file(value)
    raw = np.ascontiguousarray(value)
    return hashlib.sha256(raw.tobytes()).hexdigest()


def numbers(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    return {"n": int(len(a)), "mean": float(a.mean()) if len(a) else None,
            "std": float(a.std()) if len(a) else None,
            "min": float(a.min()) if len(a) else None,
            "max": float(a.max()) if len(a) else None}


def array(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)


def require(test, message):
    if not test:
        raise ValueError(message)


def write(path, value):
    original.write_json(path, value)


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(array(value).tobytes())
    return h.hexdigest()


def qualify(args):
    lock = read(args.lock)
    require(lock["schema"] == "semantic_b0b7_p1r_t2_head_refit_lock_v1", "BAD_T2_LOCK_SCHEMA")
    require(lock["ridge"]["alphas"] == [.01, .1, 1., 10., 100.], "UNLOCKED_ALPHA_GRID")
    require(lock["ridge"]["folds"] == 3 and lock["no_survival_data_reads"] is True,
            "UNLOCKED_FOLDS_OR_OUTCOME_SCOPE")
    require(args.expected_input_sha256 == lock["original_input_sha256"], "INPUT_NOT_LOCKED")
    require(digest(args.gate) == lock["original_gate_sha256"], "ORIGINAL_GATE_CHANGED")
    require(digest(args.aggregate) == lock["original_aggregate_sha256"], "ORIGINAL_AGGREGATE_CHANGED")
    data, binding = original.load_gate_and_inputs(args)
    require(binding["frozen_sources"] == lock["original_frozen_sources"], "OLD_SCIENCE_CHANGED")
    report = read(args.aggregate)
    require(report["input_sha256"] == binding["input_sha256"], "REPORT_INPUT_CHANGED")
    require({(x["arm"], x["seed"]) for x in report["cells"]}
            == {(a, s) for a in ARMS for s in SEEDS}, "MISSING_OR_EXTRA_HISTORICAL_CELLS")
    cells, checked_states = [], []
    selected = {(x["arm"], x["seed"]): x for x in lock["selected_frozen_encoders"]}
    for public in report["cells"]:
        directory = Path(args.fits_root) / "cells" / (public["arm"] + "_seed" + str(public["seed"]))
        receipt_path = directory / "cell.private.json"
        require(digest(receipt_path) == public["private_receipt_sha256"], "RECEIPT_CHANGED")
        receipt = read(receipt_path)
        require(receipt["frozen_sources"] == binding["frozen_sources"]
                and receipt["input_sha256"] == binding["input_sha256"], "RECEIPT_BINDING_CHANGED")
        require(len(receipt["states"]) == 4 and {s["epoch"] for s in receipt["states"]} == set(EPOCHS),
                "EXACT_FOUR_STATES_REQUIRED")
        expected_epoch = min((r for r in receipt["history"] if "validation" in r),
                             key=lambda r: (r["validation"]["auxiliary_loss"], r["epoch"]))["epoch"]
        require(public["selected_epoch"] == expected_epoch, "OLD_SELECTION_NOT_REPRODUCED")
        for state in receipt["states"]:
            path = directory / state["file"]
            require(path.resolve().parent == directory.resolve() and digest(path) == state["sha256"],
                    "CHECKPOINT_PATH_OR_SHA_CHANGED")
            checked_states.append({"arm": public["arm"], "seed": public["seed"],
                                   "epoch": state["epoch"], "sha256": state["sha256"]})
        if public["arm"] in ("B3", "B4"):
            locked = selected[(public["arm"], public["seed"])]
            info = next(s for s in receipt["states"] if s["epoch"] == expected_epoch)
            require(locked["epoch"] == expected_epoch and locked["sha256"] == info["sha256"],
                    "T2_ENCODER_RESELECTION_OR_SHA_CHANGE")
        cells.append((public, receipt, directory))
    require(len(checked_states) == 48, "EXACT_48_STATE_ASSETS_REQUIRED")
    return data, binding, lock, cells, checked_states


def rank_description(x):
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(len(x), -1)
    x = x - x.mean(0)
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    eig = np.maximum(np.linalg.eigvalsh(gram), 0.)
    threshold = eig.max(initial=0.) * max(x.shape) * np.finfo(float).eps
    use = eig[eig > threshold]
    p = use / use.sum() if use.sum() else np.empty(0)
    return {"rows": x.shape[0], "columns": x.shape[1], "numerical_rank": len(use),
            "energy_entropy_effective_rank": float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,
            "total_centered_squared_energy": float(eig.sum())}


def slot_diagnostics(tokens):
    x = array(tokens).astype(float)
    vectors = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    corr = np.corrcoef(vectors)
    corr = [[float(v) if np.isfinite(v) else None for v in row] for row in corr]
    per_slot = [{"slot": k, "patient_variance_mean": float(x[:, k].var(0).mean()),
                 "across_patient_rank": rank_description(x[:, k])} for k in range(x.shape[1])]
    per_patient_rank = [rank_description(row)["energy_entropy_effective_rank"] for row in x]
    return {"per_slot": per_slot, "centered_slot_correlation": corr,
            "per_patient_across_slot_effective_rank": numbers(per_patient_rank),
            "mean64": rank_description(x.mean(1)), "flatten512": rank_description(x)}


def gradient_diagnostics(model, output, target, measured, negatives, positive):
    objective = auxiliary_loss(output, target, measured, negatives, positive=positive, lambda_pair=.1)
    gradients = []
    for name in ("program", "pair"):
        model.zero_grad(set_to_none=True)
        objective[name]["loss"].backward(retain_graph=name == "program")
        gradients.append({n: None if p.grad is None else p.grad.detach().cpu().double().clone()
                          for n, p in model.named_parameters()})
    result, dot, norm_a2, norm_b2 = {}, 0., 0., 0.
    for name in gradients[0]:
        a, b = gradients[0][name], gradients[1][name]
        na = 0. if a is None else float(a.norm()); nb = 0. if b is None else float(b.norm())
        cosine = float((a*b).sum() / (na*nb)) if na > 0 and nb > 0 else None
        if cosine is not None:
            cosine = float(np.clip(cosine, -1, 1)); dot += float((a*b).sum())
            norm_a2 += na*na; norm_b2 += nb*nb
        require(all(v is None or math.isfinite(v) for v in (na, nb, cosine)), "NONFINITE_GRADIENT")
        result[name] = {"program_connected": a is not None, "pair_connected": b is not None,
                        "program_norm": na, "pair_norm": nb, "weighted_pair_norm": .1*nb,
                        "cosine": cosine,
                        "angle_degrees": float(np.degrees(np.arccos(cosine))) if cosine is not None else None}
    model.zero_grad(set_to_none=True)
    return {"parameter_groups": result, "shared_nonzero_cosine":
            dot / math.sqrt(norm_a2*norm_b2) if norm_a2 and norm_b2 else None,
            "eval_mode_post_checkpoint_no_dropout": True, "backward_calls": 2,
            "program_loss": float(objective["program"]["loss"].detach()),
            "pair_loss": float(objective["pair"]["loss"].detach()),
            "lambda_pair": .1, "loss_value_ratio_used_to_infer_dominance": False}


def relation_diagnostics(model, output, *, assignment=None):
    visual, gene = output["visual_tokens"], output["gene_tokens"]
    mask = output["gene_mask"]
    assignment = torch.arange(len(visual), device=visual.device) if assignment is None else assignment
    uu, vv = F.normalize(model.u(visual), dim=-1), F.normalize(model.v(gene[assignment]), dim=-1)
    valid = mask[assignment]
    relation = model._relations()
    known = model.relation_prior > 0
    denominator = torch.einsum("sk,bk->b", relation, valid.to(relation.dtype)).clamp_min(1e-12)
    weights = relation[None] * valid[:, None, :] / denominator[:, None, None]
    contribution = torch.einsum("bsr,bkr->bsk", uu, vv) * weights / model.config.temperature
    expected = output["pair"]["scores"][torch.arange(len(visual), device=visual.device), assignment]
    error = float((contribution.sum((1, 2)) - expected).abs().max())
    require(error <= 2e-6, "RELATION_CONTRIBUTION_DECOMPOSITION_MISMATCH")
    weighted_known = weights * known[None]
    known_contribution = contribution * known[None]
    unknown_contribution = contribution * (~known)[None]
    per_program = []
    for k in range(gene.shape[1]):
        use = valid[:, k]
        per_program.append({"program_index": k, "measured_patients": int(use.sum()),
            "known_edges": int(known[:, k].sum()), "unknown_edges": int((~known[:, k]).sum()),
            "known_raw_mass": float(model.relation_prior[:, k].sum()),
            "unknown_raw_mass": float((~known[:, k]).sum()*torch.sigmoid(model.unknown_logit)),
            "normalized_known_mass": numbers(array(weighted_known[use, :, k].sum(1))),
            "normalized_unknown_mass": numbers(array((weights-weighted_known)[use, :, k].sum(1))),
            "actual_known_score_contribution": numbers(array(known_contribution[use, :, k].sum(1))),
            "actual_unknown_score_contribution": numbers(array(unknown_contribution[use, :, k].sum(1)))})
    per_slot = [{"slot": s, "known_edges": int(known[s].sum()),
                 "normalized_total_mass": numbers(array(weights[:, s].sum(1))),
                 "known_score_contribution": numbers(array(known_contribution[:, s].sum(1))),
                 "unknown_score_contribution": numbers(array(unknown_contribution[:, s].sum(1)))}
                for s in range(visual.shape[1])]
    return {"unknown_logit": float(model.unknown_logit.detach()),
            "unknown_edge_weight": float(torch.sigmoid(model.unknown_logit).detach()),
            "known_edges": int(known.sum()), "unknown_edges": int((~known).sum()),
            "known_supported_programs": int(known.any(0).sum()),
            "normalized_known_mass": numbers(array(weighted_known.sum((1, 2)))),
            "normalized_unknown_mass": numbers(array((weights-weighted_known).sum((1, 2)))),
            "decomposition_max_abs_error": error, "per_program": per_program, "per_slot": per_slot,
            "unknown_is_negative_relation": False, "historical_candidate_edges_are_ground_truth": False}


def text_route(model, image, output, data):
    c = model.config
    aligned = F.normalize(torch.where(image.patch_mask[..., None], image.patch_aligned, 0.), dim=-1)
    fixed = aligned @ model.conch_prototypes.T / c.temperature
    q = model.query_projection(model.conch_prototypes) + model.query_residual
    learned = model.aligned_key(aligned) @ q.T / math.sqrt(c.width)
    raw = torch.where(image.patch_mask[..., None], image.patch_values, 0.)
    local = torch.einsum("bsn,bnd->bsd", output["patch_attention"], model.raw_value(raw))
    evidence = output["titan_query_evidence"]
    native = evidence[..., None] * model.native_projection(image.titan_native)[:, None]
    score = evidence[..., None] * model.score_projection(image.titan_scores)[:, None]
    pieces = {"local_patch": local.mean(1), "titan_native": native.mean(1), "titan_scores": score.mean(1)}
    reconstructed = sum(pieces.values())
    error = float((reconstructed-output["image_embedding"]).abs().max())
    require(error <= 2e-6, "IMAGE_ROUTE_DECOMPOSITION_MISMATCH")
    w = model.program_head.weight
    predictions = {k: v @ w[:, :c.width].T for k, v in pieces.items()}
    predictions["clinical"] = output["clinical_embedding"] @ w[:, c.width:].T
    predictions["intercept"] = model.program_head.bias.expand_as(output["program_prediction"])
    pred_error = float((sum(predictions.values())-output["program_prediction"]).abs().max())
    require(pred_error <= 2e-6, "PROGRAM_HEAD_ROUTE_DECOMPOSITION_MISMATCH")
    entropy = -(output["patch_attention"].clamp_min(1e-30).log()*output["patch_attention"]).sum(-1)
    cp, tp = model.conch_prototypes, model.titan_prototypes
    true_c, true_t = F.normalize(data["conch_prototypes"].to(cp), dim=-1), F.normalize(data["titan_prototypes"].to(tp), dim=-1)
    return {"conch_Gram": array(cp@cp.T).tolist(), "titan_Gram": array(tp@tp.T).tolist(),
        "conch_same_Gram_max_abs_error": float((cp@cp.T-true_c@true_c.T).abs().max()),
        "titan_same_Gram_max_abs_error": float((tp@tp.T-true_t@true_t.T).abs().max()),
        "conch_prototype_norm": numbers(array(cp.norm(dim=-1))),
        "titan_prototype_norm": numbers(array(tp.norm(dim=-1))),
        "projected_query_norm": numbers(array(q.norm(dim=-1))),
        "fixed_cosine_logit": numbers(array(fixed[image.patch_mask])),
        "learned_query_logit": numbers(array(learned[image.patch_mask])),
        "patch_attention_effective_patches": numbers(array(entropy.exp())),
        "image_route_decomposition_max_abs_error": error,
        "program_head_decomposition_max_abs_error": pred_error,
        "branch_prediction_values": {k: numbers(array(v)) for k, v in predictions.items()},
        "branch_prediction_patient_centered_sd": {k: numbers(array(v.std(0, unbiased=False))) for k, v in predictions.items()},
        "B3_still_receives_real_TITAN32_scores": model.arm == "B3"}


def amplitude(prediction, target, measured):
    rows = []
    for k in range(target.shape[1]):
        use = measured[:, k]
        td = target[use, k].std() if use.any() else 0.
        rows.append({"program_index": k, "patients": int(use.sum()),
                     "prediction_sd": float(prediction[use, k].std()) if use.any() else None,
                     "target_sd": float(td) if use.any() else None,
                     "SD_ratio": float(prediction[use, k].std()/td) if use.sum() >= 3 and td > 0 else None})
    return rows


def frozen_route_interventions(model, valid, baseline, train_mean_scores):
    image, genes, target, measured, *_ = valid
    permutation = torch.arange(len(target)-1, -1, -1, device=target.device)
    changed = OmicsInputs(genes.values[permutation], genes.measured[permutation])
    after = model(image, changed)["program_prediction"]
    require(torch.equal(after, baseline["program_prediction"]), "VALIDATION_RNA_ENTERED_IMAGE_ONLY")
    result = {"RNA_permutation_prediction_bytes_identical": True,
              "validation_RNA_permutation_sha256": digest(array(permutation)),
              "interventions_are_frozen_not_retrained": True}
    if model.arm == "B3":
        result["B3_score_branch"] = {}
        for name, scores in (("zero", torch.zeros_like(image.titan_scores)),
                             ("training_mean", train_mean_scores.expand_as(image.titan_scores))):
            prediction = model.image_only(replace(image, titan_scores=scores))["program_prediction"]
            result["B3_score_branch"][name] = {"prediction_change": numbers(array(prediction-baseline["program_prediction"])),
                "program": original.program_metrics(array(prediction), array(target), array(measured))}
    return result


def identity_initialization(data, device):
    result = []
    for seed in SEEDS:
        a = original.build_model(data, "B3", seed, device); b = original.build_model(data, "B4", seed, device)
        ap, bp = dict(a.named_parameters()), dict(b.named_parameters())
        require(set(ap) == set(bp) and all(torch.equal(ap[k], bp[k]) for k in ap), "B3_B4_INITIAL_PARAMETERS_DIFFER")
        aa, bb = dict(a.named_buffers()), dict(b.named_buffers())
        other = set(aa)-{"conch_prototypes", "titan_prototypes"}
        require(other == set(bb)-{"conch_prototypes", "titan_prototypes"}
                and all(torch.equal(aa[k], bb[k]) for k in other), "B3_B4_NONQUERY_BUFFERS_DIFFER")
        result.append({"seed": seed, "all_initial_parameter_tensors_identical": True,
                       "parameter_tensors": len(ap), "nonprototype_buffers_identical": True,
                       "initialization_rotation_rng_isolated": True})
        del a, b
    return result


def run_r0(args, data, binding, lock, cells, states):
    start = time.perf_counter(); device = torch.device(args.device)
    train0 = original.batch(data, "train", device); valid = original.batch(data, "valid", device)
    init = identity_initialization(data, device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    ledger = {"actual_saved_state_replays": 0, "separate_loss_backward_calls": 0,
              "new_neural_fits": 0, "optimizer_updates": 0, "replay_error_max": 0.}
    cell_reports = []
    for public, receipt, directory in cells:
        arm, seed = public["arm"], public["seed"]
        train, wrong, _ = original.train_common_support(train0, seed)
        train_image, train_genes, target, measured, _, negatives, *_ = train
        positive = torch.as_tensor(wrong, device=device) if arm == "B6a" else None
        rows = []
        for info in sorted(receipt["states"], key=lambda r:r["epoch"]):
            model = original.build_model(data, arm, seed, device)
            snapshot = torch.load(directory/info["file"], map_location=device, weights_only=False)
            require(snapshot["frozen_sources"] == binding["frozen_sources"]
                    and snapshot["input_sha256"] == binding["input_sha256"], "STATE_LINEAGE_CHANGED")
            model.load_state_dict(snapshot["model"], strict=True); model.eval(); before = state_digest(model)
            metrics, replay = original.evaluate(model, valid)
            for key, expected in snapshot["expected_validation"].items():
                require(torch.equal(replay[key], expected.cpu()), "SAVED_STATE_REPLAY_NOT_EXACT:"+key)
            require(json.dumps(metrics, sort_keys=True, allow_nan=False)
                    == json.dumps(snapshot["validation"], sort_keys=True, allow_nan=False), "SAVED_METRIC_REPLAY_CHANGED")
            ledger["actual_saved_state_replays"] += 1
            with torch.no_grad():
                valid_out = model(valid[0], valid[1])
                interventions = frozen_route_interventions(model, valid, valid_out, train_image.titan_scores.mean(0, keepdim=True))
                relation_valid = relation_diagnostics(model, valid_out)
                routes = text_route(model, valid[0], valid_out, data)
                slots = slot_diagnostics(valid_out["visual_tokens"])
                prior = model.relation_prior.clone()
                alternate = data["relation_prior"] if arm == "B4_MAP_SHUFFLE" else data["shuffled_relation_prior"]
                model.relation_prior.copy_(alternate.to(device))
                changed = model.pair_scores(valid_out, valid_out)["scores"]
                model.relation_prior.copy_(prior)
                mapping = numbers(array(changed-valid_out["pair"]["scores"]))
            train_out = model(train_image, train_genes)
            with torch.no_grad():
                relation_train = relation_diagnostics(model, train_out, assignment=positive)
                train_program = original.program_metrics(array(train_out["program_prediction"]), array(target), array(measured))
            gradients = gradient_diagnostics(model, train_out, target, measured, negatives, positive)
            ledger["separate_loss_backward_calls"] += gradients["backward_calls"]
            require(state_digest(model) == before, "DIAGNOSTIC_MODIFIED_SAVED_PARAMETERS_OR_BUFFERS")
            rows.append({"epoch": info["epoch"], "checkpoint_sha256": info["sha256"],
                "exact_saved_forward_replay": True, "validation_program": metrics["program"],
                "train_eval_program": train_program, "gradients": gradients,
                "relations_validation_own_pair": relation_valid, "relations_training_assigned_positive": relation_train,
                "frozen_mapping_swap_score_change": mapping, "text_routes": routes,
                "slots_validation": slots, "interventions": interventions,
                "prediction_amplitude": amplitude(array(valid_out["program_prediction"]), array(valid[2]), array(valid[3]))})
            del model, snapshot, valid_out, train_out, replay
            original.device_sync(device)
        item = {"arm":arm,"seed":seed,"historical_selected_epoch_unchanged":public["selected_epoch"],
                "states":rows,"training_curves":receipt["history"]}
        filename = "R0_"+arm+"_seed"+str(seed)+".private.json"
        write(output/filename,item)
        cell_reports.append({"arm":arm,"seed":seed,"file":filename,"sha256":digest(output/filename)})
        print(json.dumps({"stage":"R0","arm":arm,"seed":seed,"replays_complete":ledger["actual_saved_state_replays"]}),flush=True)
    require(ledger["actual_saved_state_replays"] == 48 and ledger["separate_loss_backward_calls"] == 96,
            "INCOMPLETE_R0_EXECUTION")
    result = {"schema":"semantic_b0b7_p1r_r0_diagnostics_v1","status":"COMPLETE_SAVED_STATE_ATTRIBUTION_ONLY",
        "input_sha256":binding["input_sha256"],"gate_sha256":binding["gate_sha256"],
        "original_frozen_sources":binding["frozen_sources"],"source_sha256":digest(__file__),
        "lock_sha256":digest(args.lock),"checked_checkpoint_sha256":states,"initialization":init,
        "cells":cell_reports,"ledger":ledger,"elapsed_seconds":time.perf_counter()-start,
        "DSS_OS_PFI_or_survival_time_events_loaded":False,"CI_computed":False,
        "limitations":["frozen interventions do not replace retraining", "known edges are historical allowed candidates, not local labels",
                        "gradient snapshot uses eval mode without historical dropout", "linear decodability tested by separately locked T2 selected states",
                        "validation already used for development", "no survival or external generalization claim"]}
    write(output/"01_SAVED_STATE_DIAGNOSTICS.json",result)
    return result


def cross_pairs():
    all_pairs = np.asarray([(i,j) for i in range(64) for j in range(i+1,64)],dtype=np.int64)
    selected = np.sort(np.random.default_rng(103031).choice(len(all_pairs),384,replace=False))
    return all_pairs[selected]


def feature_pair(kind, train_slots, train_clinical, test_slots, test_clinical):
    train_slots, test_slots = np.asarray(train_slots,float), np.asarray(test_slots,float)
    state = {"kind":kind}
    if kind == "T2_keep_slots":
        a, b = train_slots.reshape(len(train_slots),-1), test_slots.reshape(len(test_slots),-1)
    else:
        a,b = train_slots.mean(1),test_slots.mean(1)
        if kind == "T2_capacity_control":
            center,scale=a.mean(0),a.std(0);scale[scale<1e-8]=1.
            a,b=(a-center)/scale,(b-center)/scale
            pairs=cross_pairs()
            def expand(x):return np.concatenate([x,x*x,x[:,pairs[:,0]]*x[:,pairs[:,1]]],axis=1)
            a,b=expand(a),expand(b)
            state.update({"mean64_center":center,"mean64_scale":scale,"cross_pairs_sha256":digest(pairs)})
        elif kind != "T2_mean64":
            raise ValueError("UNKNOWN_T2_FEATURES")
    return np.concatenate([a,train_clinical],axis=1),np.concatenate([b,test_clinical],axis=1),state


def alpha_choice(scores, tolerance=1e-12):
    require(scores and all(math.isfinite(v) for v in scores.values()),"INVALID_CV_ALPHA_METRIC")
    best=min(scores.values())
    return max(a for a,v in scores.items() if abs(v-best)<=tolerance)


def ridge_predict(state, x):
    z=(np.asarray(x,float)-state["standardize_center"])/state["standardize_scale"]
    return np.column_stack([(z-fit["feature_local_center"])@fit["coefficient"]+fit["target_mean"]
                            for fit in state["program_fits"]])


def run_t2(args,data,binding,lock,cells,states):
    from sklearn.model_selection import KFold
    start=time.perf_counter();device=torch.device(args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    train=original.batch(data,"train",device);valid=original.batch(data,"valid",device)
    y,mask=array(train[2]),array(train[3]);yv,mv=array(valid[2]),array(valid[3])
    folds=list(KFold(n_splits=3,shuffle=True,random_state=103031).split(np.arange(len(y))))
    fold_hashes=[{"train_rows_sha256":digest(a.astype('<i8')),"validation_rows_sha256":digest(b.astype('<i8')),
                  "train_patients":len(a),"validation_patients":len(b)} for a,b in folds]
    torch.save({"folds":folds,"input_sha256":binding["input_sha256"]},output/"T2_training_cv_folds.private.pt")
    rows=[];ledger={"new_neural_fits":0,"final_refit_head_families":0,"CV_head_fit_families":0,
                   "CV_scalar_linear_system_rhs":0,"final_scalar_linear_system_rhs":0,
                   "CV_factorizations":0,"final_factorizations":0,"saved_Ridge_head_replays":0,
                   "encoder_feature_forward_calls":0,"optimizer_updates":0}
    for public,receipt,directory in cells:
        if public["arm"] not in ("B3","B4"):continue
        arm,seed=public["arm"],public["seed"]
        info=next(x for x in receipt["states"] if x["epoch"]==public["selected_epoch"])
        model=original.build_model(data,arm,seed,device)
        snapshot=torch.load(directory/info["file"],map_location=device,weights_only=False)
        model.load_state_dict(snapshot["model"],strict=True);model.eval()
        with torch.no_grad():a=model.image_only(train[0]);b=model.image_only(valid[0])
        ledger["encoder_feature_forward_calls"]+=2
        require(torch.equal(b["program_prediction"].cpu(),snapshot["expected_validation"]["prediction"].cpu()),
                "T2_HISTORICAL_PROGRAM_HEAD_REPLAY_CHANGED")
        x,xv=array(a["visual_tokens"]),array(b["visual_tokens"])
        c,cv=array(a["clinical_embedding"]),array(b["clinical_embedding"])
        features={"train_slots":x,"valid_slots":xv,"train_clinical":c,"valid_clinical":cv,
                  "input_sha256":binding["input_sha256"],"checkpoint_sha256":info["sha256"]}
        torch.save(features,output/("T2_features_"+arm+"_seed"+str(seed)+".private.pt"))
        historical=original.program_metrics(array(b["program_prediction"]),yv,mv)
        del model,snapshot,a,b
        for kind in HEADS:
            cv_scores={};oofs={}
            for alpha in lock["ridge"]["alphas"]:
                oof=np.empty_like(y,dtype=float)
                for fit_idx,test_idx in folds:
                    xa,xb,transform=feature_pair(kind,x[fit_idx],c[fit_idx],x[test_idx],c[test_idx])
                    pred,fit=original.fit_masked_ridge(xa,xb,y[fit_idx],mask[fit_idx],alpha=alpha)
                    oof[test_idx]=pred;ledger["CV_head_fit_families"]+=1
                    ledger["CV_scalar_linear_system_rhs"]+=fit["linear_system_scalar_rhs"]
                    ledger["CV_factorizations"]+=fit["linear_solver_calls_factoring_shared_masks"]
                cv_scores[alpha]=original.program_metrics(oof,y,mask)["patient_equal_masked_huber"]
                oofs[str(alpha)]=oof
            selected=alpha_choice(cv_scores,lock["ridge"]["tie_absolute_tolerance"])
            xa,xb,transform=feature_pair(kind,x,c,xv,cv)
            pred,fit=original.fit_masked_ridge(xa,xb,y,mask,alpha=selected)
            state={"schema":"semantic_b0b7_p1r_t2_ridge_state_v1","arm":arm,"seed":seed,"head":kind,
                   "checkpoint_sha256":info["sha256"],"selected_epoch":info["epoch"],"selected_alpha":selected,
                   "input_sha256":binding["input_sha256"],"lock_sha256":digest(args.lock),
                   "fit":fit,"transform":transform,"validation_prediction":pred,"training_oof_predictions":oofs,
                   "source_sha256":digest(__file__)}
            filename="T2_"+arm+"_seed"+str(seed)+"_"+kind+".private.pt"
            torch.save(state,output/filename)
            loaded=torch.load(output/filename,map_location="cpu",weights_only=False)
            replay=ridge_predict(loaded["fit"],xb);error=float(np.abs(replay-pred).max())
            require(error<=1e-12,"T2_SAVED_HEAD_REPLAY_CHANGED")
            ledger["saved_Ridge_head_replays"]+=1;ledger["final_refit_head_families"]+=1
            ledger["final_scalar_linear_system_rhs"]+=fit["linear_system_scalar_rhs"]
            ledger["final_factorizations"]+=fit["linear_solver_calls_factoring_shared_masks"]
            standardized=(xa-fit["standardize_center"])/fit["standardize_scale"]
            rows.append({"arm":arm,"seed":seed,"head":kind,"selected_epoch":info["epoch"],
                "checkpoint_sha256":info["sha256"],"selected_alpha_training_only":selected,
                "training_CV_patient_equal_Huber_by_alpha":{str(k):v for k,v in cv_scores.items()},
                "historical_neural_head_REPLAY_ONLY":historical,"heldout_program":original.program_metrics(pred,yv,mv),
                "prediction_amplitude":amplitude(pred,yv,mv),"ridge_input_features":xa.shape[1],
                "nominal_coefficients_including_50_intercepts":(xa.shape[1]+1)*50,
                "standardized_training_feature_rank":rank_description(standardized),
                "saved_private_head_sha256":digest(output/filename),"saved_head_replay_max_error":error})
            print(json.dumps({"stage":"T2","arm":arm,"seed":seed,"head":kind,"final_heads":len(rows)}),flush=True)
    require(ledger["final_refit_head_families"]==18 and ledger["CV_head_fit_families"]==270,"INCOMPLETE_T2_FITS")
    result={"schema":"semantic_b0b7_p1r_t2_frozen_head_comparison_v1","status":"COMPLETE_REFIT_HEAD_ONLY",
        "input_sha256":binding["input_sha256"],"gate_sha256":binding["gate_sha256"],"lock_sha256":digest(args.lock),
        "source_sha256":digest(__file__),"original_frozen_sources":binding["frozen_sources"],
        "selected_frozen_encoders":lock["selected_frozen_encoders"],"training_CV_folds":fold_hashes,
        "cross_product_indices_sha256":digest(cross_pairs()),"rows":rows,"ledger":ledger,
        "elapsed_seconds":time.perf_counter()-start,"DSS_OS_PFI_or_survival_time_events_loaded":False,
        "CI_computed":False,"validation_previously_reused_for_development":True,
        "limitations":["fixed historical selected encoders, not independently selected new representations",
                       "training CV refits heads only; encoders previously used all272 training patients, not full-pipeline OOF",
                       "same patients across seeds", "coefficient count is not effective degrees of freedom",
                       "polynomial control holds mean information, not operator class", "no DSS conclusion"]}
    write(output/"03_SLOT_READOUT_COMPARISON.json",result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("mode",choices=("assets","r0","t2"))
    for key in ("inputs","gate","aggregate","fits-root","lock","output"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--expected-input-sha256",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--torch-threads",type=int,default=2);args=p.parse_args()
    torch.set_num_threads(args.torch_threads);original.configure_runtime()
    data,binding,lock,cells,states=qualify(args)
    if args.mode=="assets":
        result={"schema":"semantic_b0b7_p1r_assets_v1","status":"EXACT_INPUT_SOURCE_48_STATE_ASSETS_VERIFIED",
                "input_sha256":binding["input_sha256"],"gate_sha256":binding["gate_sha256"],
                "source_sha256":digest(__file__),"lock_sha256":digest(args.lock),"states":states,
                "train_patients":len(data["train_indices"]),"validation_patients":len(data["valid_indices"]),
                "model_forwards":0,"backward_calls":0,"new_fits":0,"outcome_arrays_loaded":False}
        args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,result)
    elif args.mode=="r0":result=run_r0(args,data,binding,lock,cells,states)
    else:result=run_t2(args,data,binding,lock,cells,states)
    print(json.dumps({"status":result["status"],"ledger":result.get("ledger",{})}),flush=True)


if __name__=="__main__":main()
