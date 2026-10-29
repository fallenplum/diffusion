"""R8 private frozen interfaces; R6 inputs remain read-only and no source fits occur."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_r6_common as old
from src.bgb_en.protocol import array_hash
from src.bgb_en.r6_omics import GeneValueBank
from src.bgb_en_r8.cox import ProtectedOffsetCox
from src.bgb_en_r8.evidence import decode_frozen_source_cache, pool_frozen_evidence
from src.bgb_en_r8.pipeline import (ARMS, PARAMETERS, WRONG_PROGRAM, FrozenPatientEvidence,
    FrozenPartitionTransform, encode_frozen_bulk_programs, validate_query_partition_mapping,
    signed_mean_dispersion_audit)
from src.bgb_en.survival import evaluate

PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
SOURCE_KINDS = ("MATCHED", "RANDOM_NO_ST")
SOURCE_STATES = dict(MATCHED="399ff5c38e1499dd3751d45c25d2a37a8c3941314d7864522077dddc3c6f0c5e",
    RANDOM_NO_ST="fc5c05911467dfb690b60ac32dce8f47e9e974c65e367b52dfb2fe8606c3ff7f")
PARENT_COMMIT = "1e0cc1e7d947263699885870317874a27b8d6294"
UPSTREAM_R6_LOCK_SHA="ea1f240998256fe1aa671a0928f29dfa0989e9a9adc0366d43143463456560c9"
UPSTREAM_SOURCE_CONFIG_SHA="49fa74f691027facbe2d980c2d813534bd7d12e73dd9541cae8b0d9841682597"
FEATURE_SOURCE_FILES=("scripts/bgb_en_r8_common.py","src/bgb_en_r8/evidence.py",
    "src/bgb_en_r8/features.py","src/bgb_en_r8/pipeline.py")


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    for name in ("assets", "private", "primary-private", "source-private", "config", "source-config", "r6-private", "r6-config"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p


def validate_config(c):
    expected = dict(source_version="R6_S7_locked", gamma=.1, ridge_lambda=1.,
        source_prior_scale=.1, cosine_temperature=1., partitions=list(PARTITIONS), seeds=list(SEEDS),
        primary_candidate="R8-6", trainable_arms=list(ARMS), wrong_program_permutation=list(WRONG_PROGRAM),
        global_pca_components=8, program_pca_components=2)
    require(all(c.get(k) == v for k, v in expected.items()), "R8_FIXED_FINITE_RECIPE_CHANGED")
    return c


def legacy_args(args):
    return argparse.Namespace(assets=args.assets, private=args.r6_private,
        primary_private=args.primary_private, source_private=args.source_private,
        config=args.r6_config, source_config=args.source_config, device=args.device,
        resume=False, launch_gate=None, launch_gate_sha256=None)


def source_context(args):
    require(not args.private.resolve().is_relative_to(ROOT)
            and args.private.resolve() != args.r6_private.resolve(), "R8_NEW_PRIVATE_NAMESPACE_REQUIRED")
    c = validate_config(read(args.config));old.configure_deterministic_runtime();torch.set_num_threads(2)
    require(sha(args.r6_private/"targetLOCK.json")==UPSTREAM_R6_LOCK_SHA
            and sha(args.source_config)==UPSTREAM_SOURCE_CONFIG_SHA,
            "R8_EXACT_REUSED_R6_AND_SOURCE_CONFIG_CHANGED")
    a = legacy_args(args);ctx = old.context(a)
    qualification_sha = old.validate_qualification(a, ctx)
    require(all(ctx["source_audits"][k]["complete_source_state_sha256"] == SOURCE_STATES[k]
            for k in SOURCE_KINDS), "R8_FROZEN_SOURCE_VERSION_CHANGED")
    allowed = np.flatnonzero(np.asarray(ctx["raw"]["folds"]) != 1)
    require(len(allowed) == 409 and len(ctx["raw"]["t"]) == 512,
            "R8_CANONICAL_OUTER1_TRAIN_SCOPE_CHANGED")
    return dict(config=c, old_args=a, old_context=ctx, allowed=allowed,
                old_qualification_sha256=qualification_sha)


def _scoped_partition(ctx, prepared):
    oldctx, allowed = ctx["old_context"], ctx["allowed"]
    raw = oldctx["raw"];lookup = np.full(len(raw["t"]), -1, np.int64)
    lookup[allowed] = np.arange(len(allowed))
    tr, va = lookup[prepared["train"]], lookup[prepared["valid"]]
    require((tr >= 0).all() and (va >= 0).all() and set(tr)|set(va) == set(range(409)),
            "R8_OUTER_TEST_ROW_ENTERED_SCOPED_PARTITION")
    scope = dict(train=tr, valid=va,
        t=np.asarray(raw["t"])[allowed].copy(), e=np.asarray(raw["e"])[allowed].copy(),
        pfi_t=np.asarray(raw["pfi_t"])[allowed].copy(), pfi_e=np.asarray(raw["pfi_e"])[allowed].copy(),
        pfi_valid=np.asarray(raw["pfi_valid"])[allowed].copy(),
        CP=np.asarray(prepared["CP"])[allowed].copy(),
        RNA=np.asarray(prepared["RNA"])[allowed].copy(), beta=float(prepared["beta"]),
        base=np.asarray(prepared["base"])[allowed].copy(),
        original_train_identity_sha256=prepared["fit_hash"], original_valid_identity_sha256=prepared["valid_hash"],
        development_patient_indices_sha256=array_hash(allowed))
    require(np.array_equal(scope["base"], scope["CP"]+scope["beta"]*scope["RNA"]),
            "R8_CP_SCALAR_OFFSET_ELEMENTWISE_CHANGED")
    return scope, lookup


def feature_name(partition, source_kind):
    require(partition in PARTITIONS and source_kind in SOURCE_KINDS, "R8_FEATURE_FAMILY_OUTSIDE_MATRIX")
    return f"p{partition}_{source_kind}"


def prepare_features(args):
    """One outcome-blind feature qualification; no targets are optimized here."""
    ctx = source_context(args);a, oldctx = ctx["old_args"], ctx["old_context"]
    feature_sources={name:sha(ROOT/name) for name in FEATURE_SOURCE_FILES}
    directory = args.private/"features"
    require(not directory.exists(), "R8_EXISTING_FEATURE_NAMESPACE_NO_AUTOMATIC_REBUILD")
    directory.mkdir(parents=True)
    write(directory/"BUILD_CLAIM.json", dict(status="STARTED_R8_FROZEN_FEATURE_QUALIFICATION",
        source_version="R6_S7_locked", target_optimizer_fits=0, source_optimizer_fits=0))
    allowed = ctx["allowed"];rows = [];unconditional_hashes={}
    banks = {k: old.patch_bank(a, oldctx, k, create=False) for k in SOURCE_KINDS}
    for part in PARTITIONS:
        prepared, origin = old.qualified_prepared(a, oldctx, part)
        scope, lookup = _scoped_partition(ctx, prepared)
        global_map, pairing_audit = old.pairing(oldctx, prepared, part)
        query_map = lookup[global_map[allowed]]
        mapping_audit = validate_query_partition_mapping(query_map, scope["train"], scope["valid"], len(allowed))
        for kind in SOURCE_KINDS:
            model = oldctx["sources"][kind];bank, cache_receipt_sha = banks[kind]
            gene_bank = GeneValueBank(oldctx["raw"]["rna_symbols"], model.gene_reader.mapping.gene_symbols).to(args.device)
            pieces = {name: [] for name in ("scores", "image", "conditioned", "wrong_conditioned", "genes", "gene_mask", "coverage",
                                           "signed_audit", "signed_audit_mask")}
            decode_errors = [];bulk_count=0;bulk_squares=0.;source_forward_batches=0
            for start in range(0, len(allowed), 8):
                ids = allowed[start:start+8];index = torch.as_tensor(ids, device=args.device)
                encoded, _ = bank.select(index)
                ev, audit = decode_frozen_source_cache(model, encoded, expected_source_state_sha256=SOURCE_STATES[kind])
                decode_errors.append(audit["score_max_abs_error"])
                rna = torch.as_tensor(prepared["features"]["rna"][ids], device=args.device)
                mask = torch.as_tensor(prepared["features"]["rna_mask"][ids], device=args.device)
                measured = gene_bank(rna, mask)
                molecular, _ = encode_frozen_bulk_programs(model, measured,
                    expected_source_state_sha256=SOURCE_STATES[kind])
                signed_proxy=signed_mean_dispersion_audit(measured,model.gene_reader.mapping)
                observed = measured.values[measured.measured_mask].double()
                bulk_count += observed.numel();bulk_squares += float(observed.square().sum())
                normal = pool_frozen_evidence(ev, source_prior_scale=.1, cosine_temperature=1.)
                conditional = pool_frozen_evidence(ev, query=molecular["tokens"], query_mask=molecular["program_mask"],
                    source_prior_scale=.1, cosine_temperature=1.)
                foreign_ids=global_map[ids]
                foreign_values=gene_bank(torch.as_tensor(prepared["features"]["rna"][foreign_ids],device=args.device),
                    torch.as_tensor(prepared["features"]["rna_mask"][foreign_ids],device=args.device))
                foreign_molecular,_=encode_frozen_bulk_programs(model,foreign_values,
                    expected_source_state_sha256=SOURCE_STATES[kind])
                wrong_conditional=pool_frozen_evidence(ev,query=foreign_molecular["tokens"],
                    query_mask=foreign_molecular["program_mask"],source_prior_scale=.1,cosine_temperature=1.)
                for name, value in (("scores", normal["image_program_scores"]),
                        ("image", normal["image_program_embeddings"]),
                        ("conditioned", conditional["image_program_embeddings"]),
                        ("wrong_conditioned", wrong_conditional["image_program_embeddings"]),
                        ("genes", molecular["tokens"]), ("gene_mask", molecular["program_mask"]),
                        ("coverage", molecular["coverage"]), ("signed_audit", signed_proxy["values"]),
                        ("signed_audit_mask", signed_proxy["program_mask"])):
                    pieces[name].append(value.detach().cpu().numpy())
                source_forward_batches += 1
            z = {k: np.concatenate(v) for k, v in pieces.items()}
            data = FrozenPatientEvidence(z["scores"], z["image"], z["conditioned"], z["genes"],
                z["gene_mask"], SOURCE_STATES[kind], "R6_S7_locked" if kind == "MATCHED" else "R6_S7_RANDOM_NO_ST_locked",
                z["wrong_conditioned"]).validate()
            transform = FrozenPartitionTransform.fit(data, scope["train"], fit_identity_sha256=prepared["fit_hash"])
            image_hashes = dict(scores=array_hash(data.image_scores), image=array_hash(data.image_tokens))
            if kind in unconditional_hashes:
                require(unconditional_hashes[kind] == image_hashes, "R8_IMAGE_ONLY_EVIDENCE_CHANGED_WITH_RNA_FOLD")
            else:unconditional_hashes[kind] = image_hashes
            if kind == "MATCHED":
                additive = transform.features(data, "R8-4")
                for arm in ("R8-6", "R8-8b_WRONG_PROGRAM", "R8-9_WRONG_PATIENT", "R8-10_NO_INTERACTION"):
                    built = transform.features(data, arm, query_mapping=query_map if arm == "R8-9_WRONG_PATIENT" else None)
                    require(np.array_equal(built[:, :16], additive), "R8_RECIPIENT_ADDITIVE_MAIN_EFFECT_CHANGED")
            name = feature_name(part, kind);folder = directory/name;folder.mkdir()
            payload = folder/"features.private.pt"
            save(payload, dict(data=data, transform=transform, scoped=scope, query_mapping=query_map,
                gene_coverage=z["coverage"], original_global_development_indices=allowed.copy(),
                unlabelled_signed_proxy=z["signed_audit"], unlabelled_signed_proxy_mask=z["signed_audit_mask"],
                original_raw_container_rows=512))
            receipt = dict(status="COMPLETE_R8_SCOPED_FROZEN_FEATURES", partition=part, source_kind=kind,
                source_version=data.source_version, source_state_sha256=SOURCE_STATES[kind],
                config_sha256=sha(args.config), source_config_sha256=sha(args.source_config),
                source_patch_cache_receipt_sha256=cache_receipt_sha, prepared_origin=origin,
                source_origin=oldctx["source_audits"][kind], source_program_order=list(model.gene_reader.mapping.program_names),
                source_program_definition_sha256=model.gene_reader.mapping.definition_sha256,
                source_key_reconstruction_max_error=max(decode_errors), source_key_replay_tolerance=2e-6,
                frozen_bulk_forward_batches=2*source_forward_batches,
                own_bulk_forward_batches=source_forward_batches,
                wrong_context_bulk_forward_batches=source_forward_batches,
                cached_source_head_replay_batches=source_forward_batches,
                frozen_WSI_program_reads=3*source_forward_batches,
                wrong_RNA_query_is_frozen_diagnostic_not_R8_5_retraining=True,
                wrong_query_source_program_mask="foreign_query_availability_not_recipient_gene_measurement_mask",
                bulk_input_coordinate_RMS=(bulk_squares/bulk_count)**.5, measured_bulk_scalars=bulk_count,
                original_gene_bank=gene_bank.public_audit(), scoped_rows=409, outer_test_rows_consumed=0,
                old_container_rows=512, old_container_physically_loaded=True,
                scope_contract="only_outer1_train_indices_consumed_by_new_feature_generation_fit_and_metrics",
                source_spot_scaler_applied_to_bulk=False, source_visual_forward_uses_RNA=False,
                source_gene_encoder_training_unit="source_train_observed_only_gene_Z_of_log1p_CP10k_full_measured_library",
                target_gene_encoder_input_unit="inner_train_median_imputed_moments_gene_Z_of_TCGA_platform_expression_then_mask",
                source_dataset_raw_helper_is_not_actual_alignment_training_input=True,
                common_Z_form_proves_cross_domain_biological_alignment=False,
                locked_G_bulk_representation="frozen_source_gene64_then_train_program_PCA2",
                signed_mean_dispersion_is_unlabelled_audit_only=True,
                multi_slide_pooling="equal_nonempty_WSI_after_within_WSI_softmax",
                frozen_source_uses_spatial_graph=bool(model.use_spatial),
                PCA=transform.audit(), query_pairing=pairing_audit, query_mapping=mapping_audit,
                feature_generation_source_sha256=feature_sources,
                unchanged_raw_RNA512_teacher_and_CP_offset=True, source_fit_count=0,
                teacher_fit_count=0, target_supervised_fit_count=0, target_neural_updates=0,
                artifact_sha256=sha(payload), original_development_identity_sha256=array_hash(allowed),
                offset_sha256=array_hash(scope["base"]), CP_sha256=array_hash(scope["CP"]),
                RNA_teacher_sha256=array_hash(scope["RNA"]), frozen_feature_arrays_sha256={k:array_hash(v) for k,v in z.items()})
            write(folder/"READY.json", receipt)
            rows.append(dict(name=name, receipt_sha256=sha(folder/"READY.json"), **receipt))
    result = dict(status="PASS_R8_FROZEN_SOURCE_SCOPED_409_FEATURE_QUALIFICATION", config_sha256=sha(args.config),
        source_config_sha256=sha(args.source_config),R6_target_lock_sha256=sha(args.r6_private/"targetLOCK.json"),
        source_version="R6_S7_locked", source_family_selected_by_DSS=False, rows=rows,
        feature_families=6, source_neural_fits=0, target_neural_fits=0, target_convex_fits=0,
        unsupervised_PCA_fits=60, old_R6_qualification_sha256=ctx["old_qualification_sha256"],
        feature_generation_source_sha256=feature_sources,
        endpoint_or_validation_scores_used_for_feature_selection=False, outer_test_evaluation=False)
    require(all(sha(ROOT/name)==digest for name,digest in feature_sources.items()),
            "R8_FEATURE_GENERATOR_CHANGED_DURING_EXTRACTION")
    write(args.private/"R8_FEATURE_QUALIFICATION.json", result)
    return result


def scientific_paths(args):
    files = [p for folder, pattern in ((ROOT/"src/bgb_en_r8", "*.py"), (ROOT/"scripts", "bgb_en_r8_*.py"),
             (ROOT/"tests", "test_bgb_en_r8_*.py")) for p in folder.glob(pattern)]
    files.append(args.config.resolve())
    return sorted({str(p.resolve().relative_to(ROOT)).replace("\\", "/") for p in files})


def create_lock(args):
    c=validate_config(read(args.config));q=read(args.private/"R8_FEATURE_QUALIFICATION.json")
    require(q["status"] == "PASS_R8_FROZEN_SOURCE_SCOPED_409_FEATURE_QUALIFICATION"
            and q["config_sha256"] == sha(args.config) and q["feature_families"] == 6
            and q["source_neural_fits"] == q["target_neural_fits"] == q["target_convex_fits"] == 0,
            "R8_ALL_REAL_FEATURES_MUST_BE_QUALIFIED_BEFORE_LOCK")
    require(set(q["feature_generation_source_sha256"])==set(FEATURE_SOURCE_FILES)
            and all(sha(ROOT/name)==digest for name,digest in q["feature_generation_source_sha256"].items()),
            "R8_FEATURES_MUST_MATCH_CURRENT_FROZEN_GENERATOR_CODE")
    upstream=read(args.r6_private/"targetLOCK.json")
    require(q["R6_target_lock_sha256"]==sha(args.r6_private/"targetLOCK.json")==UPSTREAM_R6_LOCK_SHA
            and q["source_config_sha256"]==sha(args.source_config)==UPSTREAM_SOURCE_CONFIG_SHA
            and all(sha(ROOT/name)==digest for name,digest in upstream["sources"].items()),
            "R8_UPSTREAM_SOURCE_BYTES_CHANGED_AFTER_FEATURE_QUALIFICATION")
    paths=sorted(set(upstream["sources"])|set(scientific_paths(args)))
    required=set(FEATURE_SOURCE_FILES)|{"src/bgb_en_r8/__init__.py","src/bgb_en_r8/cox.py",
        "scripts/bgb_en_r8_run.py","tests/test_bgb_en_r8_frozen_core.py",
        "tests/test_bgb_en_r8_pipeline.py","tests/test_bgb_en_r8_runner.py"}
    require(required<=set(paths),"R8_ALL_EXECUTABLE_SOURCE_AND_VERIFICATION_FILES_REQUIRED")
    receipts={feature_name(p,k):sha(args.private/"features"/feature_name(p,k)/"READY.json")
              for p in PARTITIONS for k in SOURCE_KINDS}
    qualified={r["name"]:r for r in q["rows"]}
    require(len(q["rows"])==len(qualified)==6 and set(qualified)==set(receipts),
            "R8_QUALIFICATION_FEATURE_FAMILY_INVENTORY_CHANGED")
    for name,digest in receipts.items():
        folder=args.private/"features"/name;ready=read(folder/"READY.json")
        require(qualified[name]["receipt_sha256"]==digest
                and ready["artifact_sha256"]==sha(folder/"features.private.pt")
                and ready["feature_generation_source_sha256"]==q["feature_generation_source_sha256"],
                "R8_QUALIFIED_FEATURE_PAYLOAD_OR_RECEIPT_CHANGED_BEFORE_LOCK")
    lock=dict(schema="bgb_en_r8_frozen_evidence_protocol_lock_v1", status="LOCKED_BEFORE_R8_TARGET_FITS",
        parent_commit=PARENT_COMMIT, source_version="R6_S7_locked", source_full_state_sha256=SOURCE_STATES,
        source_config_sha256=sha(args.source_config), source_R6_target_lock_sha256=sha(args.r6_private/"targetLOCK.json"),
        config_sha256=sha(args.config), qualification_sha256=sha(args.private/"R8_FEATURE_QUALIFICATION.json"),
        sources={name:sha(ROOT/name) for name in paths}, feature_receipt_sha256=receipts,
        partitions=list(PARTITIONS), seeds=list(SEEDS), arms=list(ARMS), primary_candidate="R8-6",
        planned_convex_fits=81, planned_neural_fits=0, planned_neural_updates=0, reused_reference_rows=18,
        gamma=c["gamma"], ridge_lambda=c["ridge_lambda"], learned_intercepts=0,
        source_or_teacher_refits=0, raw_RNA512_risk_preserved=True,
        legacy_512_container_physically_loaded=True,
        legacy_global_endpoint_count_validation_performed=True,
        outer_rows_consumed_by_new_features_fits_or_metrics=0,
        outer_test_used_for_model_selection=False, outer_performance_evaluation=False,
        formal_fit_and_replay_read_only_scoped_409_inputs=True,
        PFI_used_for_selection=False, DSS_feedback_to_source_allowed=False,
        bilinear_rank_two_is_2x2_dimension_upper_bound_not_additional_low_rank_constraint=True,
        risk_SD_protection_applies_only_to_training_distribution_not_validation_or_each_pair=True,
        deterministic_convex_unique_problems=27,nominal_seed_repeat_fits=3,
        R8_9_does_not_prove_R8_5_conditional_attention_retraining=True,
        primary_metrics=["within_partition_Harrell", "within_partition_Uno60"],
        aggregation="sum_within_partition_concordance_components_per_seed_then_equal_three_seed_mean")
    lock["contract_sha256"]=canonical(lock)
    write(args.private/"R8_PROTOCOL_LOCK.json",lock)
    return lock


def target_context(args):
    c=validate_config(read(args.config));lock=read(args.private/"R8_PROTOCOL_LOCK.json")
    require(lock["status"] == "LOCKED_BEFORE_R8_TARGET_FITS" and lock["config_sha256"] == sha(args.config)
            and canonical({k:v for k,v in lock.items() if k!="contract_sha256"}) == lock["contract_sha256"]
            and lock["qualification_sha256"] == sha(args.private/"R8_FEATURE_QUALIFICATION.json")
            and all(sha(ROOT/name)==digest for name,digest in lock["sources"].items()),
            "R8_LOCKED_SOURCE_CONFIG_OR_QUALIFICATION_CHANGED")
    return dict(config=c,lock=lock)


def load_features(args,ctx,partition,source_kind):
    name=feature_name(partition,source_kind);folder=args.private/"features"/name
    require(sha(folder/"READY.json")==ctx["lock"]["feature_receipt_sha256"][name],
            "R8_FROZEN_FEATURE_RECEIPT_CHANGED")
    meta=read(folder/"READY.json");p=folder/"features.private.pt"
    require(meta["artifact_sha256"]==sha(p) and meta["source_state_sha256"]==SOURCE_STATES[source_kind]
            and meta["config_sha256"]==sha(args.config) and meta["scoped_rows"]==409,
            "R8_PRIVATE_SCOPED_FEATURE_PAYLOAD_CHANGED")
    payload=torch.load(p,map_location="cpu",weights_only=False)
    payload["data"].validate();scope=payload["scoped"]
    require(len(scope["t"])==payload["data"].rows==409
            and np.array_equal(scope["base"],scope["CP"]+scope["beta"]*scope["RNA"]),
            "R8_SCOPED_OFFSET_OR_ROWCOUNT_CHANGED")
    validate_query_partition_mapping(payload["query_mapping"],scope["train"],scope["valid"],409)
    return payload,meta


def metrics(scope,indices,risk,reference=None,*,pfi=False):
    tr=np.asarray(scope["train"]);idx=np.asarray(indices)
    if pfi:
        usable=scope["pfi_valid"][idx];fit=tr[scope["pfi_valid"][tr]]
        return evaluate(scope["pfi_t"][fit],scope["pfi_e"][fit],scope["pfi_t"][idx][usable],
            scope["pfi_e"][idx][usable],np.asarray(risk)[usable],None if reference is None else np.asarray(reference)[usable])
    return evaluate(scope["t"][tr],scope["e"][tr],scope["t"][idx],scope["e"][idx],risk,reference)


def risk_summary(delta,base):
    x=np.asarray(delta,float);s=float(np.std(base))
    return dict(rows=len(x),mean=float(x.mean()),SD=float(x.std()),minimum=float(x.min()),
                maximum=float(x.max()),SD_ratio_to_fixed_offset=float(x.std()/max(s,1e-12)))
