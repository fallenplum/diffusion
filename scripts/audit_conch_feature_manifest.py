"""Audit exact slide-level overlap between UNI2/CONCH H5 inventories.

This is a post-download gate. It does not download data and it never reads
survival labels. Slide identifiers are compared by the full filename stem,
including the TCGA UUID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def stems(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob("*.h5")}


def inspect_h5(path: Path) -> dict[str, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("h5py is required for CONCH manifest audit") from exc

    with h5py.File(path, "r") as h5:
        out: dict[str, object] = {
            "path": str(path),
            "keys": sorted(h5.keys()),
            "attrs": {str(k): str(v) for k, v in h5.attrs.items()},
        }
        for key in ("features", "coords"):
            if key in h5:
                out[f"{key}_shape"] = list(h5[key].shape)
                out[f"{key}_dtype"] = str(h5[key].dtype)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conch-root", type=Path, required=True)
    ap.add_argument("--uni-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    conch = stems(args.conch_root)
    uni = stems(args.uni_root)
    overlap = sorted(set(conch) & set(uni))
    conch_only = sorted(set(conch) - set(uni))
    uni_only = sorted(set(uni) - set(conch))
    sample = inspect_h5(conch[overlap[0]]) if overlap else None

    result = {
        "conch_root": str(args.conch_root),
        "uni_root": str(args.uni_root),
        "conch_slide_count": len(conch),
        "uni_slide_count": len(uni),
        "exact_stem_overlap": len(overlap),
        "overlap_fraction_of_uni": len(overlap) / len(uni) if uni else None,
        "missing_conch_for_uni": uni_only,
        "extra_conch": conch_only,
        "sample_overlap_h5": sample,
        "gate": "PASS" if len(overlap) == len(uni) else "REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "conch_slide_count", "uni_slide_count", "exact_stem_overlap",
        "overlap_fraction_of_uni", "gate",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
