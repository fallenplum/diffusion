from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from phase1_neural import DEVICE, BagStore, EncoderConfig, SurvivalModel


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/pathology_context_phase1"
SEED = 20260723


def load_model(fold: int):
    path = PHASE / f"checkpoints/survival/P2/fold_{fold}/seed_{SEED}.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = SurvivalModel(EncoderConfig(**bundle["encoder_config"]))
    model.load_state_dict(bundle["model_state"])
    return model.to(DEVICE).eval()


def main():
    pred = pd.read_csv(PHASE / f"oof_predictions/P2_seed_{SEED}.csv")
    if not len(pred):
        parts = sorted((PHASE / "oof_predictions/fold_parts/P2").glob(f"fold_*_seed_{SEED}.csv"))
        pred = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    selected = pd.concat([pred.nsmallest(5, "oof_risk").assign(risk_group="low"),
                          pred.nlargest(5, "oof_risk").assign(risk_group="high")])
    slides = pd.read_csv(PHASE / "audit/slide_manifest_used.csv")
    store = BagStore(ROOT, slides)
    rows = []
    fig_dir = PHASE / "figures/contextual_importance"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for patient in selected.itertuples(index=False):
        model = load_model(int(patient.outer_fold))
        patient_slides = slides.loc[slides.patient_id.astype(str) == str(patient.patient_id)]
        for slide in patient_slides.to_dict("records"):
            rng = np.random.default_rng(SEED + 991 + sum(map(ord, str(patient.patient_id))) + sum(map(ord, str(slide["slide_id"]))))
            feat, coord_norm, indices = store.sample_slide(slide, rng, False)
            x = torch.from_numpy(feat).to(DEVICE).unsqueeze(0).requires_grad_(True)
            c = torch.from_numpy(coord_norm).to(DEVICE).unsqueeze(0)
            z, risk, _ = model(x, c)
            risk.backward()
            score = torch.abs((x.grad * x).sum(dim=-1)).squeeze(0).detach().cpu().numpy()
            score = score / max(float(np.max(score)), 1e-12)
            raw_coords = np.load(ROOT / str(slide["coords_path"]), mmap_mode="r")[indices]
            for (cx, cy), value in zip(raw_coords, score):
                rows.append({"patient_id": patient.patient_id, "slide_id": slide["slide_id"],
                             "outer_fold": int(patient.outer_fold), "risk_group": patient.risk_group,
                             "oof_risk": float(patient.oof_risk), "x": float(cx), "y": float(cy),
                             "attention_score": float(value),
                             "score_type": "absolute input-times-gradient contextual importance"})
            fig, ax = plt.subplots(figsize=(6, 5))
            points = ax.scatter(raw_coords[:, 0], raw_coords[:, 1], c=score, cmap="inferno", s=8, alpha=.85)
            ax.invert_yaxis(); ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{patient.patient_id} | {patient.risk_group} OOF risk | {slide['slide_id']}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(points, ax=ax, label="normalized |input x gradient|")
            fig.tight_layout()
            safe_slide = str(slide["slide_id"]).replace("/", "_")
            fig.savefig(fig_dir / f"{patient.patient_id}_{safe_slide}.png", dpi=180)
            plt.close(fig)
    out = PHASE / "statistical_tests/contextual_importance_10_patients.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    note = {
        "method": "absolute input-times-gradient contextual importance from the fold-specific P2 model",
        "selection": "5 lowest and 5 highest P2 OOF risks; selection is descriptive, not performance tuning",
        "sampling": "one deterministic subset, maximum 1024 patches per slide",
        "interpretation_limit": "importance is model sensitivity, not causal lesion evidence; no WSI background was fabricated",
    }
    (fig_dir / "README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps({"patients": int(selected.patient_id.nunique()), "patch_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
