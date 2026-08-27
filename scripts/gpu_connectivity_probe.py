from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import torch


def main() -> int:
    run_dir = Path(os.environ["DIFFUSION_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(os.environ["DIFFUSION_DATA_ROOT"])
    tcga_dir = data_root / "TCGA"

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NONE"
    cuda_version = torch.version.cuda or "NONE"

    if not cuda_available:
        raise RuntimeError("CUDA is not available in the runner Python environment")
    if not tcga_dir.is_dir():
        raise RuntimeError("DIFFUSION_DATA_ROOT/TCGA is not accessible")

    # Minimal GPU compute probe only. No patient file is opened or enumerated.
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    y = x @ x.T
    checksum = float(y.sum().item())

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["cuda_available", int(cuda_available)])
        writer.writerow(["gpu_name", gpu_name])
        writer.writerow(["cuda_version", cuda_version])
        writer.writerow(["tcga_root_accessible", int(tcga_dir.is_dir())])
        writer.writerow(["gpu_compute_checksum", checksum])

    probe = {
        "status": "PASS",
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "tcga_root_accessible": tcga_dir.is_dir(),
        "gpu_compute_checksum": checksum,
        "patient_files_opened": 0,
        "patient_files_enumerated": 0,
    }
    (run_dir / "gpu_connectivity_probe.json").write_text(
        json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(probe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
