"""Small independent-review tests: no real data, training, or model imports."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_s5b1", ROOT / "scripts/verify_spgel_s5b1_final.py")
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


class IndependentReviewTests(unittest.TestCase):
    def test_incomplete_gate_reads_only_state_never_private_or_producer(self):
        seen = []

        def only_state(path):
            seen.append(path)
            if path != Path("state.json"):
                self.fail("Private/result metadata accessed before COMPLETE")
            return {"status": "RUNNING_20_OWN_FOLD_DEVELOPMENT_FITS"}

        with patch.object(REVIEW, "read_json", side_effect=only_state):
            with self.assertRaisesRegex(REVIEW.ReviewError, "NOT_COMPLETE_NO_RESULT"):
                REVIEW.audit_all(argparse.Namespace(state=Path("state.json")))
        self.assertEqual(seen, [Path("state.json")])

    def test_admin_status_without_exact_counts_cannot_open_gate(self):
        state = {"status": "COMPLETE_20_DEV_20_FINAL_120_ACTUAL_REPLAYS",
                 "development_completed": 20, "final_completed": 19, "active": [], "failed": []}
        with patch.object(REVIEW, "read_json", return_value=state):
            with self.assertRaisesRegex(REVIEW.ReviewError, "ADMIN_COMPLETE_COUNTS"):
                REVIEW.require_complete_gate(Path("state.json"))

    def test_exact_replay_set_rejects_duplicate_even_when_count_matches(self):
        rows = [{"stage": "DEV", "epoch": 80}, {"stage": "DEV", "epoch": 80}]
        with self.assertRaisesRegex(REVIEW.ReviewError, "BAD_REPLAY_SET"):
            REVIEW.unique_exact(rows, lambda r: (r["stage"], r["epoch"]),
                                {("DEV", 80), ("DEV", 160)}, "BAD_REPLAY_SET")

    def test_own_fold_selection_uses_five_arms_and_tie_smaller_epoch(self):
        rows = {a: {"snapshots": [{"epoch": e, "validation_MSE": 1.0 if e in (160, 320) else 2.0}
                                for e in REVIEW.EPOCHS]} for a in REVIEW.ARMS}
        scores, chosen = REVIEW.select_own_fold(rows)
        self.assertEqual(chosen, 160)
        self.assertEqual(set(scores), {"80", "160", "320", "640", "1280"})
        # The fifth arm actually participates rather than being silently omitted.
        rows["SPGEL50_ADAPT"]["snapshots"][1]["validation_MSE"] = 3.0
        self.assertEqual(REVIEW.select_own_fold(rows)[1], 320)

    def test_round_each_fold_before_aggregate_not_full_precision_average(self):
        def split(value):
            return {"pearson_corrs": [{"name": f"GENE{i}", "pearson_corr": value} for i in range(50)],
                    "pearson_mean": value}
        raw0, raw1 = split(0.10011), split(0.10021)
        aggregate = REVIEW.independently_aggregate(REVIEW.rounded4(raw0), REVIEW.rounded4(raw1))
        self.assertEqual(aggregate["mean_per_split"], [0.1001, 0.1002])
        self.assertEqual(aggregate["pearson_mean"], round(float(np.mean([0.1001, 0.1002])), 4))
        self.assertNotEqual(aggregate["pearson_mean"], round(float(np.mean([0.10011, 0.10021])), 4))

    def test_undefined_gene_is_not_imputed_or_dropped(self):
        left = {"pearson_corrs": [{"name": "A", "pearson_corr": None}, {"name": "B", "pearson_corr": 0.2}],
                "pearson_mean": None}
        right = {"pearson_corrs": [{"name": "A", "pearson_corr": 0.6}, {"name": "B", "pearson_corr": 0.4}],
                 "pearson_mean": 0.5}
        out = REVIEW.independently_aggregate(left, right)
        self.assertIsNone(out["pearson_mean"])
        self.assertEqual(len(out["pearson_corrs"]), 2)
        self.assertEqual(out["pearson_corrs"][-1]["name"], "A")
        self.assertIsNone(out["pearson_corrs"][-1]["mean"])
        self.assertEqual(out["pearson_corrs"][-1]["pearson_corrs"], [None, 0.6])

    def test_duplicate_json_keys_and_nonstandard_nan_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "metadata.json"
            path.write_text('{"count":120,"count":119}', encoding="utf-8")
            with self.assertRaisesRegex(REVIEW.ReviewError, "DUPLICATE_JSON_KEY"):
                REVIEW.read_json(path)
            path.write_text('{"metric":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(REVIEW.ReviewError, "NONSTANDARD_NONFINITE_JSON"):
                REVIEW.read_json(path)

    def test_stream_hash_tracks_current_bytes_and_mutations(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "fake.private.pt"
            path.write_bytes(b"opaque model bytes, not deserialized")
            hashes = REVIEW.FileHashes()
            digest = hashes.get(path, "logical_state")
            self.assertEqual(digest, REVIEW.hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotIn(str(path), json.dumps(hashes.inventory))
            path.write_bytes(b"changed bytes with different size")
            with self.assertRaisesRegex(REVIEW.ReviewError, "ARTIFACT_CHANGED"):
                hashes.unchanged()

    def test_complete_synthetic_metadata_chain_without_model_or_array_loading(self):
        """All 40/120/20/1000 contracts, with opaque bytes instead of real arrays.

        The only real inputs are already-public static source/config/cost receipts.
        No running experiment state, curves, selection or test result is opened.
        """
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            base = Path(directory)
            private = base / "private"
            private.mkdir()

            def dump(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(REVIEW.json_text(value), encoding="utf-8")
                return sha(path)

            def sha(path):
                return REVIEW.hashlib.sha256(path.read_bytes()).hexdigest()

            def sealed(value):
                return dict(value, internal_contract_sha256=REVIEW.canonical(value))

            def opaque(path, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode())
                return sha(path)

            freeze_path = ROOT / "tmp/SPGEL_S5B1_CODE_FREEZE_20261029_v3.json"
            freeze = REVIEW.read_json(freeze_path)
            cfg_path = ROOT / "configs/spgel_s5_hest/shared_feature_v1.json"
            genes = [f"SYNTHETIC_GENE_{i}" for i in range(50)]
            gene_file = base / "genes.json"
            gene_sha = dump(gene_file, {"genes": genes})
            assets_path = base / "assets.json"
            assets_sha = dump(assets_path, {"tasks": {d: {"gene_file": {"path": str(gene_file), "sha256": gene_sha}}
                                                       for d in ("COAD", "READ")}})
            cache_shas, dev_parts, final_parts = {}, [], []
            for stage, folder, scope in (("DEV", "dev_inputs", "INNER_TRAIN_ONLY"),
                                         ("FINAL", "final_inputs", "FULL_OFFICIAL_OUTER_TRAIN_ONLY"),
                                         ("TEST", "test_inputs", "NO_TEST_FITS_TRANSFORM_ONLY")):
                for d, f in REVIEW.PARTITIONS:
                    cache_folder = private / folder / f"{d}_fold{f}"
                    array_sha = opaque(cache_folder / "inputs.private.npz", "opaque arrays " + stage + d + str(f))
                    pipeline_sha = opaque(cache_folder / "pipeline.private.joblib", "opaque pipeline " + stage + d + str(f))
                    meta = {"dataset": d, "fold": f, "genes": genes, "status": "PASS",
                            "preprocessing_fit_scope": scope, "new_scaler_fits": int(stage != "TEST"),
                            "new_PCA_fits": int(stage != "TEST"), "arrays_sha256": {"x": "a" * 64},
                            "input_file_sha256": array_sha, "pipeline_sha256": pipeline_sha}
                    if stage != "TEST":
                        meta["official_test_expression_loaded"] = False
                    split = {"uses_expression_or_outcomes": False, "graph_cross_inner_partition_edges": 0,
                             "donor_heldout_validation_claimed": False,
                             "spatial_buffer_between_inner_train_and_validation": False}
                    if stage == "DEV":
                        meta["inner_split"] = split
                    cache_shas[stage, d, f] = dump(cache_folder / "CACHE.private.json", meta)
                    partition = {"dataset": d, "fold": f, "train_rows": 3,
                                 "cache_metadata_sha256": cache_shas[stage, d, f]}
                    if stage == "DEV":
                        dev_parts.append(dict(partition, validation_rows=2, split=split))
                    if stage == "FINAL":
                        final_parts.append(partition)
            initial = {"source_files": freeze["sources"], "runtime": freeze["runtime"],
                       "config_sha256": sha(cfg_path), "assets_sha256": assets_sha}
            dump(private / "S5B_PREPARATION_STARTED.private.json", initial)
            profile_sha = dump(private / "ZERO_OPTIMIZER_PROFILE.private.json", {
                "status": "PASS_FIVE_REAL_INNER_TRAIN_FORWARD_BACKWARD_ZERO_OPTIMIZER",
                "optimizer_steps": 0, "new_fits": 0,
                "units": [{"arm": a, "optimizer_steps": 0, "model_state_unchanged": True,
                           "test_expression_used": False} for a in REVIEW.ARMS]})
            dev = sealed(dict(initial, status="LOCKED_BEFORE_TWENTY_DEVELOPMENT_FITS", partitions=dev_parts,
                new_scaler_fits=4, new_PCA_fits=4, new_model_fits=0,
                official_test_expression_loaded=False, zero_optimizer_profile_sha256=profile_sha))
            dev_sha = dump(private / "S5B_DEV_RUN_LOCK.json", dev)
            complete_shas, producer_cells, replays = {}, [], []
            graph = {"cross_specimen_edges": 0, "physical_radius_used": False,
                     "graphsage_messages": {"self_edges": 0, "maximum_real_neighbors": 2,
                                            "mean_uses_actual_nonself_neighbor_count": True}}

            def make_cells(stage, contract):
                for d, f in REVIEW.PARTITIONS:
                    for a in REVIEW.ARMS:
                        epochs = 1280 if stage == "DEV" else 80
                        name = f"{d}_fold{f}_{a}"
                        folder = private / ("dev_cells" if stage == "DEV" else "final_cells") / name
                        snapshots = []
                        for epoch in REVIEW.EPOCHS if stage == "DEV" else [80]:
                            state_sha = opaque(folder / f"epoch{epoch:03d}.private.pt", "opaque state " + stage + name + str(epoch))
                            pred_sha = opaque(folder / f"epoch{epoch:03d}.private.npz", "opaque pred " + stage + name + str(epoch))
                            snapshots.append({"epoch": epoch, "state_sha256": state_sha, "predictions_sha256": pred_sha,
                                              "post_update_train_MSE": 1.0,
                                              "validation_MSE": (1.0 if epoch == 80 else 2.0) if stage == "DEV" else None})
                        pairing = {"train_rows": 3, "anchor_count": 3, "negative_count": 2,
                                   "pairings_per_direction": 9, "temperature": 0.1,
                                   "candidate_condition_recomputed_both_directions": True,
                                   "positive_conditioned_embeddings_reused": False,
                                   "image_anchor_CE": 2.0, "gene_anchor_CE": 4.0}
                        aux = 3.0 if a == "SPGEL50_ADAPT" else 0.0
                        total = float(np.float32(np.float32(1) + np.float32(0.1) * np.float32(aux)))
                        curve = [{"epoch": e, "optimizer_updates": e, "pre_update_main_MSE": 1.0,
                                  "pre_update_auxiliary_loss": aux, "pre_update_total_loss": total,
                                  "gradient_norms": {"synthetic": 0.1},
                                  "paired_ITC_audit": pairing if a == "SPGEL50_ADAPT" else None}
                                 for e in range(1, epochs + 1)]
                        params = dict(zip(("total", "image_prediction", "training_only_gene_auxiliary"), REVIEW.PARAMETERS[a]))
                        row = {"dataset": d, "fold": f, "arm": a, "phase": stage, "seed": 1,
                               "status": "COMPLETE_ONE_DEVELOPMENT_FIT" if stage == "DEV" else "COMPLETE_ONE_FINAL_FIT",
                               "epochs": epochs, "optimizer_updates": epochs, "learning_curve": curve,
                               "new_model_fits": 1, "internal_contract_sha256": contract,
                               "input_cache_metadata_sha256": cache_shas[stage, d, f],
                               "official_test_expression_loaded": False, "inference_reads_RNA": False,
                               "auxiliary_supervision_only_train_50genes": True,
                               "parameters": params, "snapshots": snapshots,
                               "train_graph": graph, "validation_graph": graph if stage == "DEV" else None,
                               "initial_state_sha256": {"synthetic": "a" * 64}, "elapsed_seconds": 0.1}
                        complete_sha = dump(folder / "COMPLETE.private.json", row)
                        complete_shas[stage, d, f, a] = complete_sha
                        producer_cells.append({"stage": stage, "dataset": d, "fold": f, "arm": a,
                            "epochs": epochs, "optimizer_updates": epochs, "parameters": params,
                            "snapshots": snapshots, "elapsed_seconds": 0.1, "COMPLETE_sha256": complete_sha,
                            "diagnostic_curve_stride20": [r for r in curve if r["epoch"] == 1 or r["epoch"] % 20 == 0 or r["epoch"] == epochs]})
                        for snap in snapshots:
                            replays.append({"stage": stage, "dataset": d, "fold": f, "arm": a,
                                "epoch": snap["epoch"], "state_sha256": snap["state_sha256"],
                                "COMPLETE_sha256": complete_sha, "status": "PASS", "actual_state_loaded": True,
                                "actual_forward_replayed": True, "new_model_fits": 0, "optimizer_steps": 0,
                                "maximum_prediction_error": 0.0})

            make_cells("DEV", dev["internal_contract_sha256"])
            selections = [{"dataset": d, "fold": f, "selected_epoch": 80,
                "inner_mean_MSE": {str(e): 1.0 if e == 80 else 2.0 for e in REVIEW.EPOCHS},
                "uses_only_this_outer_fold_training_partition": True,
                "budget_assessment": "INNER_SELECTED_NOT_CONVERGENCE_PROOF",
                "development_completion_sha256": {a: complete_shas["DEV", d, f, a] for a in REVIEW.ARMS}}
                for d, f in REVIEW.PARTITIONS]
            schedule = sealed({"source_files": freeze["sources"], "official_test_expression_loaded": False,
                "new_model_fits": 0, "status": "LOCKED_FROM_20_DEVELOPMENT_FITS_NO_OFFICIAL_TEST",
                "development_contract": dev["internal_contract_sha256"], "selections": selections})
            schedule_sha = dump(private / "S5B_FINAL_SCHEDULE_LOCK.json", schedule)
            final = sealed({"source_files": freeze["sources"], "official_test_expression_loaded": False,
                "new_model_fits": 0, "status": "LOCKED_FULL_TRAIN_INPUTS_BEFORE_TWENTY_FINAL_FITS",
                "development_contract": dev["internal_contract_sha256"], "schedule_sha256": schedule_sha,
                "partitions": final_parts, "new_scaler_fits": 4, "new_PCA_fits": 4})
            final_sha = dump(private / "S5B_FINAL_INPUT_LOCK.json", final)
            make_cells("FINAL", final["internal_contract_sha256"])
            gate = {"status": "OPENED_AFTER_ALL_20_FINAL_STATES_FROZEN", "new_model_fits": 0,
                    "final_completion_sha256": {f"{d}_fold{f}_{a}": complete_shas["FINAL", d, f, a]
                                                for d, f in REVIEW.PARTITIONS for a in REVIEW.ARMS}}
            gate_sha = dump(private / "OFFICIAL_TEST_OPEN_GATE.private.json", gate)
            for d, f in REVIEW.PARTITIONS:
                path = private / "test_inputs" / f"{d}_fold{f}" / "CACHE.private.json"
                value = REVIEW.read_json(path)
                value["official_test_open_gate_sha256"] = gate_sha
                cache_shas["TEST", d, f] = dump(path, value)
            units, tasks = [], {d: {} for d in ("COAD", "READ")}
            for d, f in REVIEW.PARTITIONS:
                for a in REVIEW.ARMS:
                    folder = private / "final_cells" / f"{d}_fold{f}_{a}"
                    prediction_sha = opaque(folder / "official_test_prediction.private.npz", "opaque test " + d + str(f) + a)
                    corr = [0.1 + i * 0.001 for i in range(50)]
                    errors = [0.05 + i * 0.001 for i in range(50)]
                    raw = {"l2_errors": errors, "r2_scores": [0.0] * 50,
                           "pearson_corrs": [{"name": gene, "pearson_corr": value} for gene, value in zip(genes, corr)],
                           "pearson_mean": float(np.mean(corr)), "pearson_std": float(np.std(corr))}
                    for prefix, values in (("l2_error", errors), ("r2_score", [0.0] * 50)):
                        for key, quantile in (("q1", 25), ("q2", 50), ("q3", 75)):
                            raw[prefix + "_" + key] = float(np.percentile(values, quantile))
                    row = {"dataset": d, "fold": f, "arm": a, "epoch": 80,
                           "state_sha256": sha(folder / "epoch080.private.pt"),
                           "COMPLETE_sha256": complete_shas["FINAL", d, f, a],
                           "test_cache_metadata_sha256": cache_shas["TEST", d, f],
                           "test_prediction_sha256": prediction_sha, "test_rows": 2, "test_graph": graph,
                           "first_test_evaluation_after_all_twenty_final_fits": True, "new_model_fits": 0,
                           "full_precision_metrics": raw, "official_round4": REVIEW.rounded4(raw)}
                    dump(folder / "EVALUATION.private.json", row)
                    units.append(row)
            for d in ("COAD", "READ"):
                for a in REVIEW.ARMS:
                    parts = [u["official_round4"] for u in units if u["dataset"] == d and u["arm"] == a]
                    tasks[d][a] = REVIEW.independently_aggregate(*parts)
            evaluation_sha = dump(private / "S5B_OFFICIAL_EVALUATION.private.json", {
                "status": "COMPLETE_TWENTY_FROZEN_FINAL_TEST_EVALUATIONS", "units": units,
                "new_model_fits": 0, "test_gate_sha256": gate_sha,
                "final_input_contract": final["internal_contract_sha256"]})
            replay_sha = dump(private / "S5B_REPLAY.private.json", {
                "status": "PASS_120_ACTUAL_SAVED_STATE_REPLAYS", "passed_states": 120,
                "new_model_fits": 0, "optimizer_steps": 0, "official_evaluation_sha256": evaluation_sha,
                "units": replays})
            state = {"status": "COMPLETE_20_DEV_20_FINAL_120_ACTUAL_REPLAYS", "development_completed": 20,
                     "final_completed": 20, "active": [], "failed": [],
                     "source_canonical_sha256": REVIEW.SOURCE_CANONICAL,
                     "expression_transferred_to_server": False, "GPU_used": False,
                     "workers": 2, "threads_per_worker": 2, "elapsed_seconds": 5.0}
            state_path = base / "state.json"
            dump(state_path, state)
            steps = [{"operation": op, "unit": f"{op}_{d}_fold{f}_{a}", "exit_code": 0, "elapsed_seconds": 0.1}
                     for op in ("fit-dev", "fit-final") for d, f in REVIEW.PARTITIONS for a in REVIEW.ARMS]
            steps.extend({"operation": op, "unit": op, "exit_code": 0, "elapsed_seconds": 0.1}
                         for op in ("prepare-dev", "select", "prepare-final", "evaluate", "replay", "report"))
            dump(private / "admin/EXECUTION_PRIVATE.json", {"state": state, "steps": steps,
                "telemetry": [{"process_RSS_bytes": 1, "available_RAM_bytes": 10, "disk_free_bytes": 20}],
                "new_fits_during_replay": 0, "official_test_selects_epoch_or_model": False})
            producer_path = base / "producer.json"
            dump(producer_path, {"status": "COMPLETE_TWENTY_DEVELOPMENT_AND_TWENTY_FINAL_FITS_AND_120_REPLAYS",
                "source_files": freeze["sources"], "config_sha256": sha(cfg_path),
                "actual_development_fits": 20, "actual_final_fits": 20, "actual_saved_state_replays": 120,
                "new_scaler_fits": 8, "new_PCA_fits": 8, "development_lock_sha256": dev_sha,
                "final_schedule_sha256": schedule_sha, "final_inputs_sha256": final_sha,
                "cells": producer_cells, "selections": selections, "evaluation_sha256": evaluation_sha,
                "replay_sha256": replay_sha, "tasks": tasks, "maximum_replay_error": 0.0,
                "actual_optimizer_updates": 27200,
                "boundaries": {"official_test_not_used_for_selection": True, "tissue_truth_or_teacher_used": False,
                    "extra_gene_membership_used": False, "queries_are_anonymous_not_medical_programs": True,
                    "SPGEL50_is_not_original_A5": True, "BLEEP_or_TRIPLEX_reproduced": False,
                    "optimizer_convergence_proved": False, "new_independent_cohort_claimed": False,
                    "DSS_PFI_or_R8_read": False, "own_fold_test_excluded_from_training_and_selection": True,
                    "reverse_CV_train_access_to_same_source_is_permitted": True,
                    "single_slide_inner_block_has_spatial_exclusion_buffer": False}})
            args = argparse.Namespace(state=state_path, root=ROOT, freeze=freeze_path, config=cfg_path,
                private=private, producer=producer_path, assets=assets_path,
                superseded_cost=ROOT / "reports/spgel_s5/S5B1_SUPERSEDED_PRETEST_COST.json")
            result = REVIEW.audit_all(args)
            self.assertEqual(len(result["ledger"]["cells"]), 40)
            self.assertEqual(len(result["fold_rows"]), 20)
            self.assertEqual(len(result["gene_rows"]), 1000)
            self.assertEqual(result["ledger"]["optimizer_updates"]["total"], 27200)
            self.assertEqual(result["arithmetic"].maximum_error, 0)
            self.assertNotIn(str(base), json.dumps(result["hashes"].inventory))


if __name__ == "__main__":
    unittest.main()
