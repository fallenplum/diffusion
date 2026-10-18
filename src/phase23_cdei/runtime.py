"""Finite Phase23 run identity and private artifact I/O; no training logic."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path("/root/autodl-tmp/phase23_cdei_private")
CFG = ROOT / "configs/phase23_cdei_g0.json"
INHERITED_PHASE19 = "ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5"
INHERITED_PHASE20 = "e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548"
TRAINING_MODULES = ("data", "model", "operators", "reference", "readout", "features", "warmup", "runtime", "train")


def cfg():
    return json.loads(CFG.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_inherited_sources():
    from src.phase19_attention.data import source as phase19_source
    from src.phase20_selective.data import source as phase20_source
    actual = {"phase19": phase19_source(), "phase20": phase20_source()}
    expected = {"phase19": INHERITED_PHASE19, "phase20": INHERITED_PHASE20}
    if actual != expected:
        raise ValueError("inherited Phase19/Phase20 source identity changed")
    return actual


def source_files():
    paths = [ROOT / f"src/phase23_cdei/{name}.py" for name in TRAINING_MODULES]
    paths += [ROOT / "scripts/run_phase23_cdei.py", CFG]
    if any(not path.is_file() for path in paths):
        raise ValueError("incomplete explicit Phase23 training-source manifest")
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def source():
    identity = {"inherited": verify_inherited_sources(), "files": source_files()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cell_name(seed, fold):
    return f"seed{int(seed)}_fold{int(fold)}"


def cell_directory(role, seed, fold):
    if role not in ("inner", "outer"):
        raise ValueError("role must be inner or outer")
    return PRIVATE / role / cell_name(seed, fold)


def require_run_authorization(role, identity):
    qualification_path = PRIVATE / "REAL_DATA_QUALIFICATION.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("status") != "PASS" or qualification.get("source") != identity:
        raise ValueError("formal run requires source-matched PASS real-data qualification")
    if role == "outer":
        gate = json.loads((PRIVATE / "INNER_GATE.json").read_text(encoding="utf-8"))
        if gate.get("decision") != "GO" or gate.get("source") != identity:
            raise ValueError("outer run requires source-matched GO inner gate")
    elif role != "inner":
        raise ValueError("role must be inner or outer")
    return qualification


def verify_done(directory, *, role, seed, fold, identity):
    directory = Path(directory)
    receipt = json.loads((directory / "DONE_PRIVATE.json").read_text(encoding="utf-8"))
    expected = {"status": "COMPLETE", "source": identity, "role": role,
                "seed": int(seed), "fold": int(fold)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("cell DONE status, identity or source mismatch")
    artifacts = receipt.get("artifacts", {})
    for name in ("kernel.pt", "readouts.pt"):
        path = directory / name
        if not path.is_file() or path.stat().st_size < 1 or artifacts.get(name) != sha(path):
            raise ValueError("cell checkpoint missing, empty or different from DONE hash")
    return receipt
