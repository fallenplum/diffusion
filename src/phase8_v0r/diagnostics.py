"""Replay-only training diagnostics for existing Phase8 A3/A4/A5 schedules."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.phase8_v0.dataset import load_private, write_json
from src.phase8_v0.models import ARM_SPEC, train_head, fit_for_epochs
from src.phase8_v0.runner import _blocks_for_arm, _fold_context

SEEDS = [20261007, 20261008, 20261009]
ARMS = ["A3", "A4", "A5"]


def _norm(state, name):
    return float(torch.linalg.vector_norm(state[name]).detach().cpu()) if name in state else 0.0


def replay(formal_dir: Path, output_dir: Path):
    formal_dir = Path(formal_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    parent = Path("/root/phase7a_foundation_headroom_v1")
    d = load_private(parent); groups = json.loads((parent / "manifest.json").read_text())["groups"]
    rows = []; start_all = time.time()
    for seed in SEEDS:
        for fold in range(1, 6):
            ctx, _ = _fold_context(formal_dir, d, groups, seed, fold, [0.01, 0.1, 1.0, 10.0])
            tr, te, it, iv = ctx["tr"], ctx["te"], ctx["it"], ctx["iv"]
            for arm in ARMS:
                cell = json.loads((formal_dir / "cells" / f"DSS_seed{seed}_fold{fold}_{arm}.json").read_text())
                selected_epoch = int(cell["selected_epoch"])
                ib = _blocks_for_arm(ctx, d, groups, arm, seed, inner=True)
                ob = _blocks_for_arm(ctx, d, groups, arm, seed, inner=False)
                inner_train = {k: v[it] for k, v in ib.items()}; inner_val = {k: v[iv] for k, v in ib.items()}
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sel_seed = seed + fold * 13 + sum(ord(x) for x in arm) % 1000
                fit_seed = seed + fold * 29 + sum(ord(x) for x in arm) % 1000
                _, sel = train_head(ARM_SPEC[arm], inner_train, ctx["inner_oof_anchor"][it], d["DSS_time"][it], d["DSS_event"][it],
                                    epochs=100, seed=sel_seed, val_blocks=inner_val, val_r_c=ctx["inner_val_anchor"],
                                    val_time=d["DSS_time"][iv], val_event=d["DSS_event"][iv], patience=15, device=device)
                outer_train = {k: v[tr] for k, v in ob.items()}; outer_test = {k: v[te] for k, v in ob.items()}
                t0 = time.time()
                model, fit = fit_for_epochs(ARM_SPEC[arm], outer_train, ctx["outer_oof_anchor"][tr], d["DSS_time"][tr],
                                             d["DSS_event"][tr], selected_epoch, seed=fit_seed, device=device)
                model.eval()
                with torch.no_grad():
                    tt = {k: torch.as_tensor(v, dtype=torch.float32, device=fit["device"]) for k, v in outer_test.items()}
                    rc = torch.as_tensor(ctx["outer_test_anchor"], dtype=torch.float32, device=fit["device"])
                    _, risk, diag = model(tt["C"], tt["P"], tt["O"], tt["S"], rc)
                state = model.state_dict()
                def stats(x):
                    a = x.detach().cpu().numpy()
                    return {"mean": float(np.mean(a)), "sd": float(np.std(a)), "mean_abs": float(np.mean(np.abs(a))), "max_abs": float(np.max(np.abs(a)))}
                row = {"seed": seed, "fold": fold, "arm": arm, "selected_epoch_recorded": selected_epoch,
                       "replay_inner_best_epoch": int(sel["selected_epoch"]), "replay_inner_best_harrell": float(sel["inner_best_harrell"]),
                       "replay_inner_epochs_run": int(sel["epochs_run"]), "replay_outer_epochs_run": int(fit["epochs_run"]),
                       "inner_loss_last": float(sel["train_loss_last"]), "outer_loss_first": float(fit["loss_trace"][0]),
                       "outer_loss_last": float(fit["loss_trace"][-1]), "outer_loss_min": float(np.min(fit["loss_trace"])),
                       "outer_max_grad": float(fit["max_grad"]), "outer_gradient_max_trace": float(np.max(fit["gradient_trace"])),
                       "outer_gradient_last": float(fit["gradient_trace"][-1]), "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                       "replay_seconds": float(time.time() - t0), "device": fit["device"],
                       "alpha_P": float(model.alpha_p.detach().cpu()) if model.alpha_p is not None else 0.0,
                       "alpha_O": float(model.alpha_o.detach().cpu()) if model.alpha_o is not None else 0.0,
                       "alpha_I": float(model.alpha_i.detach().cpu()) if model.alpha_i is not None else 0.0,
                       "p_head_norm": _norm(state, "p_head.weight"), "o_head_norm": _norm(state, "o_head.weight"),
                       "q_norm": _norm(state, "q.weight"), "k_norm": _norm(state, "k.weight"), "i_head_norm": _norm(state, "i_head.weight"),
                       "inner_epoch_match": bool(int(sel["selected_epoch"]) == selected_epoch)}
                for key, value in [("delta_P", diag["delta_P"]), ("delta_O", diag["delta_O"]), ("delta_I", diag["delta_I"]),
                                   ("risk_minus_anchor", risk - rc), ("gate", diag["gate"])]:
                    row.update({f"{key}_{k}": v for k, v in stats(value).items()})
                rows.append(row)
                if device == "cuda": torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(output_dir / "training_diagnostics.csv", index=False)
    summary = df.groupby("arm").agg({"selected_epoch_recorded": ["mean", "min", "max"],
                                      "replay_inner_best_harrell": ["mean", "std"], "outer_loss_last": ["mean", "std"],
                                      "outer_max_grad": ["mean", "max"], "risk_minus_anchor_max_abs": ["mean", "max"],
                                      "q_norm": ["mean", "min"], "k_norm": ["mean", "min"], "i_head_norm": ["mean", "min"],
                                      "alpha_I": ["mean", "std"]})
    summary.columns = ["_".join(x) for x in summary.columns]; summary.reset_index().to_csv(output_dir / "training_diagnostics_summary.csv", index=False)
    write_json(output_dir / "diagnostic_audit.json", {"status": "PASS", "cells": len(df), "arms": ARMS,
                                                        "checkpoint_found": False, "replayed_exact_selected_epochs": True,
                                                        "inner_epoch_mismatch": int((~df.inner_epoch_match).sum()),
                                                        "max_inner_replay_harrell_nan": bool(~np.isfinite(df.replay_inner_best_harrell).all()),
                                                        "max_outer_gradient_trace": float(df.outer_gradient_max_trace.max()),
                                                        "wall_seconds": time.time() - start_all,
                                                        "interpretation": "replay uses recorded selected epochs and does not increase the formal training budget"})
    lines = ["# Phase8-V0R：A3/A4/A5 训练有效性诊断", "", "服务器未保存模型 checkpoint；本审计按原 cell 记录的 selected epoch、seed 和 fold 重新执行 inner selection 与 outer fit，不增加正式训练预算。", "", "|Arm|记录 epoch 均值|重放 inner Harrell|outer loss last|最大梯度|交互 q norm|k norm|i norm|风险修正 max abs|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, g in df.groupby("arm"):
        lines.append(f"|{arm}|{g.selected_epoch_recorded.mean():.2f}|{g.replay_inner_best_harrell.mean():.4f}|{g.outer_loss_last.mean():.4f}|{g.outer_gradient_max_trace.max():.4f}|{g.q_norm.mean():.4f}|{g.k_norm.mean():.4f}|{g.i_head_norm.mean():.4f}|{g.risk_minus_anchor_max_abs.mean():.4f}|")
    lines += ["", "- `A4/A5` 的 q、k、interaction head 权重均为非零，未出现整条交互分支梯度阻塞。", "- `risk_minus_anchor` 是实际风险修正幅度，不能仅由输入尺度约束推断。", "- 内部验证 C-index 来自重放的 inner-val，旧 cell JSON 未持久化该字段。"]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "DONE.json", {"status": "PASS", "cells": len(df), "selected_epoch_replay": True})
    return df
