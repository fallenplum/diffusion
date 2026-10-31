"""Low-level hash-gated P2 anchor, Cox cell and inner-selection commands.

The matrix scheduler owns the finite full study. This CLI does not start a
search, load old predictions as training input, or select on outer outcomes.
All pickle inputs/outputs are private, user-owned, byte-hash-bound artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.semantic_b0b7.p2_anchor import fit_r4_anchor
from src.semantic_b0b7.p2_survival import fit_risk_cell,replay_risk_cell,select_lambda


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path,value,*,binary=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb" if binary else "x",**({} if binary else {"encoding":"utf-8","newline":"\n"})) as stream:
        if binary:
            pickle.dump(value,stream,protocol=5)
        else:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def locked(args):
    if sha_file(args.lock)!=args.expected_lock_sha256:
        raise ValueError("P2_LOCK_BYTE_HASH_CHANGED")
    lock=json.loads(Path(args.lock).read_text(encoding="utf-8-sig"))
    if lock.get("status") not in ("LOCKED_BEFORE_OUTER_EVALUATION","LOCKED"):
        raise ValueError("P2_STUDY_LOCK_NOT_FROZEN")
    for rel,expected in lock.get("frozen_sources",{}).items():
        path=ROOT/rel
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha_file(path)!=expected:
            raise ValueError("P2_LOCKED_SOURCE_CHANGED")
    return lock


def load_private(args):
    if sha_file(args.input)!=args.expected_input_sha256:
        raise ValueError("P2_PRIVATE_INPUT_HASH_CHANGED")
    with Path(args.input).open("rb") as stream:
        value=pickle.load(stream)
    if not isinstance(value,dict):
        raise ValueError("P2_PRIVATE_INPUT_MAPPING_REQUIRED")
    return value


def main(args):
    lock=locked(args)
    pack=load_private(args)
    if args.command=="anchor":
        result=fit_r4_anchor(pack["raw"],pack["fit_rows"],pack["evaluation_rows"],pack["membership"],
                             expected_union_sha256=lock["B0_R4_EXACT"]["legacy_eight_program_union_sha256"],
                             expected_legacy_source_hashes=lock["B0_R4_EXACT"]["legacy_source_sha256"],
                             legacy_root=args.legacy_root,teacher_seed=int(pack["teacher_seed"]),
                             pipeline_crossfit_seed=int(pack["pipeline_crossfit_seed"]),
                             downstream_crossfit=not args.b0_only)
        write_new(args.output,result,binary=True)
        write_new(args.receipt,{**result.receipt,"artifact_sha256":sha_file(args.output)})
    elif args.command=="risk-cell":
        result=fit_risk_cell(**pack,ridge_lambda=args.ridge_lambda,role=args.role)
        write_new(args.output,result,binary=True)
        # Re-open the actual saved object before replaying its linear head.
        with Path(args.output).open("rb") as stream:
            saved=pickle.load(stream)
        replay=replay_risk_cell(saved,pack["evaluation_features"],pack["evaluation_offset"],
                                pack["train_time"],pack["train_event"],pack["evaluation_time"],pack["evaluation_event"])
        write_new(args.receipt,{**result.receipt,"artifact_sha256":sha_file(args.output),"saved_head_replay":replay})
    else:
        candidates={float(k):v for k,v in pack["candidates"].items()}
        write_new(args.output,select_lambda(candidates))
    print(json.dumps({"status":"COMPLETE","command":args.command,"output_sha256":sha_file(args.output),
                      "lock_sha256":args.expected_lock_sha256,"confidence_intervals_computed":False}))


def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="command",required=True)
    for command in ("anchor","risk-cell","select"):
        p=sub.add_parser(command)
        for field in ("lock","expected-lock-sha256","input","expected-input-sha256","output"):
            p.add_argument("--"+field,required=True)
        if command!="select":
            p.add_argument("--receipt",required=True)
        if command=="anchor":
            p.add_argument("--legacy-root",required=True)
            p.add_argument("--b0-only",action="store_true")
        if command=="risk-cell":
            p.add_argument("--ridge-lambda",type=float,choices=(.1,1.,10.),required=True)
            p.add_argument("--role",choices=("inner","outer_final"),required=True)
    return cli


if __name__=="__main__":
    main(parser().parse_args())
