"""Synthetic target input/source locks; never reads real patient artifacts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import bgb_en_r6_common as c
from src.bgb_en.protocol import fixed_inner_splits, array_hash


def prepared_fixture(tmp_path):
    n = 21
    raw = dict(t=np.arange(1., n+1), e=np.tile([0, 1, 1], 7), folds=np.r_[np.ones(3, int), np.full(18, 2)],
               rna=np.random.default_rng(1).normal(size=(n, 4)))
    raw["rna"][5, 1] = np.nan
    train, valid = fixed_inner_splits(raw["folds"], raw["e"], outer=1)[0]
    blocks = np.array_split(train, 3)
    groups = np.full(n, -1, int)
    teachers = []
    for k, held in enumerate(blocks):
        groups[held] = k
        fit = np.setdiff1d(train, held)
        teachers.append(dict(train=fit, held=held, preprocessing=dict(fit_hash=array_hash(fit))))
    measured = np.isfinite(raw["rna"][train]).sum(0)
    mask = np.isfinite(raw["rna"]) & (measured[None] > 0)
    prepared = dict(train=train, valid=valid, teachers=teachers, teacher_groups=groups,
        fit_hash=array_hash(train), valid_hash=array_hash(valid),
        features=dict(rna=np.where(mask, raw["rna"], 0).astype(np.float32), rna_mask=mask,
            state=dict(fit_hash=array_hash(train), molecular=dict(fit_hash=array_hash(train), measured=measured))),
        pair_teacher_rule="both_pair_members_in_one_excluded_teacher_block", RNA=np.zeros(n),
        CP=np.linspace(-1., 1., n), beta=0., scalar_audit=dict(gradient_max=0.))
    prepared["base"] = prepared["CP"].copy()
    args = SimpleNamespace(primary_private=tmp_path)
    ctx = dict(raw=raw, lineage=dict(schema="synthetic"), lock=None)
    payload, receipt = tmp_path/"prepared/outer1_inner0.pt", tmp_path/"prepared/outer1_inner0.json"
    payload.parent.mkdir(parents=True)

    def save(value=prepared):
        torch.save(value, payload)
        c.write(receipt, dict(status="COMPLETE", lineage=ctx["lineage"], sha256=c.sha256(payload), risk_distribution={}))
    save()
    return args, ctx, prepared, save


def test_exact_prepared_reuse_replays_beta_without_new_teachers(tmp_path):
    args, ctx, original, _ = prepared_fixture(tmp_path)
    saved, audit = c.qualified_prepared(args, ctx, 0)
    assert np.array_equal(saved["base"], original["base"])
    assert audit["teacher_refits"] == audit["scalar_refits"] == 0
    assert audit["validation_labels_excluded_from_beta_and_teachers"]
    assert audit["scalar_gradient_replayed"] == 0.


@pytest.mark.parametrize("kind, message", [("teacher_overlap", "TEACHER_PATIENT"),
    ("mask", "MEASUREMENT_MASK"), ("scale", "SCALE_NOT_TRAIN_ONLY"),
    ("offset", "OFFSET_ELEMENTWISE"), ("payload", "PAYLOAD_CHANGED")])
def test_prepared_scope_mask_offset_or_payload_corruption_is_rejected(tmp_path, kind, message):
    args, ctx, original, save = prepared_fixture(tmp_path)
    modified = deepcopy(original)
    if kind == "teacher_overlap":
        modified["teachers"][0]["held"] = modified["teachers"][0]["train"][:1]
    elif kind == "mask":
        modified["features"]["rna_mask"][5, 1] = True
    elif kind == "scale":
        modified["features"]["state"]["molecular"]["fit_hash"] = "a"*64
    elif kind == "offset":
        modified["base"][modified["valid"][0]] += .1
    if kind == "payload":
        with (tmp_path/"prepared/outer1_inner0.pt").open("ab") as handle:
            handle.write(b"changed")
    else:
        save(modified)
    with pytest.raises(ValueError, match=message):
        c.qualified_prepared(args, ctx, 0)


def lock_fixture(tmp_path, monkeypatch):
    c.configure_deterministic_runtime()
    root = tmp_path/"repo"
    root.mkdir()
    monkeypatch.setattr(c, "ROOT", root)
    source = root/"frozen.py"
    source.write_text("frozen=True\n")
    config, source_config = root/"target.json", root/"source.json"
    config.write_text("{}")
    source_config.write_text("{}")
    ready = tmp_path/"cache/READY.json"
    ready.parent.mkdir()
    ready.write_text("{}")
    source_private = tmp_path/"source_private"
    source_audits = {}
    for control in c.FULL_SOURCE_CONTROLS:
        directory = source_private/"source_cells"/f"S7_{control}_h-1_seed31"
        directory.mkdir(parents=True)
        origin = {}
        for name, field in (("state_epoch080.private.pt", "checkpoint_sha256"),
                            ("COMPLETE.json", "completion_sha256"), ("REPLAY.json", "replay_sha256")):
            path = directory/name
            path.write_text(control+name)
            origin[field] = c.sha256(path)
        source_audits[control] = origin
    args = SimpleNamespace(config=config, source_config=source_config, source_private=source_private)
    ctx = dict(raw=dict(audit=dict(cohort_sha256="a"*64, endpoints_sha256="b"*64)),
        source_audits=source_audits, assets=dict(patch_uni2=ready.parent), mask_sha256="c"*64)
    lock = dict(schema="bgb_en_r6_target_protocol_lock_v1", status="LOCKED_BEFORE_TARGET_FITS",
        sources={"frozen.py": c.sha256(source)}, config_sha256=c.sha256(config), source_config_sha256=c.sha256(source_config),
        source_full_states=source_audits, data_fingerprints=ctx["raw"]["audit"],
        RNA_finite_mask_sha256="c"*64, UNI2_regions_READY_sha256=c.sha256(ready),
        fixed_deterministic_runtime=dict(c.FIXED_DETERMINISTIC_RUNTIME))
    lock["contract_sha256"] = c.canonical_hash(lock)
    return args, ctx, lock, source, ready


@pytest.mark.parametrize("kind, message", [("source", "LOCKED_SOURCE"), ("cache", "PATCH_RECEIPT"),
    ("source_fit", "LOCKED_FULL_SOURCE_FILE"), ("runtime", "DETERMINISTIC_RUNTIME")])
def test_target_lock_pins_source_cache_and_deterministic_runtime(tmp_path, monkeypatch, kind, message):
    args, ctx, lock, source, ready = lock_fixture(tmp_path, monkeypatch)
    c.verify_target_lock(lock, ctx, args)
    if kind == "source":
        source.write_text("changed=True\n")
    elif kind == "cache":
        ready.write_text("changed")
    elif kind == "source_fit":
        (args.source_private/"source_cells/S7_MATCHED_h-1_seed31/state_epoch080.private.pt").write_text("changed")
    else:
        torch.use_deterministic_algorithms(False)
    try:
        with pytest.raises(ValueError, match=message):
            c.verify_target_lock(lock, ctx, args)
    finally:
        c.configure_deterministic_runtime()


def test_no_st_random_foundation_uses_uniform_prior_and_ignores_true_source_label_frequencies(tmp_path, monkeypatch):
    """Synthetic source receipts only: no optimizer, real data or source fit."""
    from src.bgb_en.r6_omics import mapping_from_programs, SparseProgramEncoder, PROGRAM_ORDER
    from src.bgb_en.r6_source_model import SourceProgramModel
    from src.bgb_en.r6_pathology import source_model_state_sha256

    config = tmp_path/"source_config.json"
    config.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(source_config=config, source_private=tmp_path/"synthetic_source_private", device="cpu")
    hashes = {"SYNTHETIC_SOURCE_SCHEMA_ONLY": "a"*64}
    monkeypatch.setattr(c, "validate_source_lock", lambda _: None)
    monkeypatch.setattr(c, "current_code_hashes", lambda: hashes)
    mapping = mapping_from_programs([dict(id=name, up_genes=[str(k)], down_genes=[str(k+4)])
                                     for k, name in enumerate(PROGRAM_ORDER)])

    def source_fixture(prior):
        for index, control in enumerate(c.FULL_SOURCE_CONTROLS):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(100+index)
                source = SourceProgramModel("S7", tissue_prior=prior,
                    gene_reader=SparseProgramEncoder(mapping), init_seed=31)
            cell = dict(arm="S7", control=control, heldout_donor=-1, seed=31)
            directory = args.source_private/"source_cells"/f"S7_{control}_h-1_seed31"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint, completion, replay = (directory/"state_epoch080.private.pt",
                                               directory/"COMPLETE.json", directory/"REPLAY.json")
            torch.save(dict(schema="bgb_en_r6_source_checkpoint_v1", cell=cell, epochs=80,
                optimizer_updates=80, config_sha256=c.sha256(config), source_code_sha256=hashes,
                statistics=dict(tissue_prior=list(prior)), transfer=source.transfer_state()), checkpoint)
            c.write(completion, dict(status="COMPLETE_R6_SOURCE_FIT", cell=cell, epochs=80,
                optimizer_updates=80, actual_new_optimizer_fits=1,
                config_sha256=c.sha256(config), source_code_sha256=hashes,
                artifact_sha256={checkpoint.name: c.sha256(checkpoint)}))
            c.write(replay, dict(status="PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY", cell=cell,
                optimizer_updates=0, new_optimizer_fits=0, config_sha256=c.sha256(config),
                source_code_sha256=hashes, completion_sha256=c.sha256(completion)))

    first_prior, second_prior = [.8, .15, .05], [.05, .2, .75]
    source_fixture(first_prior)
    first, audit_first = c.source_foundations(args, mapping)
    first_state = {name: value.detach().clone() for name, value in first["RANDOM_NO_ST"].state_dict().items()}
    source_fixture(second_prior)
    second, audit_second = c.source_foundations(args, mapping)
    uniform = torch.full((3,), 1/3, dtype=torch.float32)
    assert torch.equal(first["RANDOM_NO_ST"].train_tissue_prior, uniform)
    assert torch.equal(second["RANDOM_NO_ST"].train_tissue_prior, uniform)
    assert all(torch.equal(value, second["RANDOM_NO_ST"].state_dict()[name]) for name, value in first_state.items())
    assert audit_first["RANDOM_NO_ST"]["complete_source_state_sha256"] == \
        audit_second["RANDOM_NO_ST"]["complete_source_state_sha256"]
    assert source_model_state_sha256(first["MATCHED"]) != source_model_state_sha256(second["MATCHED"])
    for control in c.FULL_SOURCE_CONTROLS:
        assert torch.equal(first[control].train_tissue_prior, torch.tensor(first_prior))
        assert torch.equal(second[control].train_tissue_prior, torch.tensor(second_prior))
    for audit in (audit_first["RANDOM_NO_ST"], audit_second["RANDOM_NO_ST"]):
        assert audit["only_external_training_tissue_prior_reused"] is False
        assert audit["source_tissue_labels_reused"] is False
        assert audit["tissue_prior_source"] == "fixed_uniform_three_tissue_categories"
        assert "prior_origin_completion_sha256" not in audit
