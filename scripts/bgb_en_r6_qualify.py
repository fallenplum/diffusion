"""Qualify source evidence/cache only; this entry point never trains a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.bgb_en.r6_source import (SourceContract,SourcePaths,load_source_spots,
    load_private_cache,write_private_cache,sha256,require)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-data',type=Path,required=True)
    parser.add_argument('--prepared-manifest',type=Path,required=True)
    parser.add_argument('--programs',type=Path,required=True)
    parser.add_argument('--fresh-certificate',type=Path,required=True)
    parser.add_argument('--fresh-certificate-sha256',required=True)
    parser.add_argument('--raw-dev',type=Path,required=True)
    parser.add_argument('--cache-private',type=Path)
    parser.add_argument('--cache-manifest-sha256')
    parser.add_argument('--reuse-cache',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();began=time.monotonic()
    require(not args.output.exists(),'EXISTING_QUALIFICATION_OUTPUT_PRESERVED')
    require(not args.output.resolve().is_relative_to(args.raw_dev.resolve()),'NO_REPORT_WRITES_IN_RAW_SOURCE')
    contract=SourceContract(fresh_certificate_sha256=args.fresh_certificate_sha256)
    paths=SourcePaths(args.prepared_data,args.prepared_manifest,args.programs,args.fresh_certificate,args.raw_dev)
    cache=None
    if args.reuse_cache:
        require(args.cache_private is not None,'CACHE_PATH_REQUIRED_FOR_REUSE')
        require(args.cache_manifest_sha256 is not None,'PRELOCKED_CACHE_MANIFEST_SHA_REQUIRED_FOR_REUSE')
        dataset=load_private_cache(args.cache_private,contract,expected_manifest_sha256=args.cache_manifest_sha256)
        cache=dict(status='REUSED_EXACT_SHA_PRIVATE_CACHE',manifest_sha256=sha256(args.cache_private/'manifest.private.json'))
    else:
        dataset=load_source_spots(paths,contract)
        if args.cache_private is not None:
            cache=write_private_cache(dataset,args.cache_private,repository_root=ROOT,raw_root=args.raw_dev)
    audit=dataset.public_audit()
    audit.update(scope='DATASET_ONLY_R6_FORMAL_TRAINING_NOT_STARTED',cache=cache,
        real_gene_operator_GPU_gradient_qualification='NOT_RUN_MODEL_IMPLEMENTATION_AND_GATE_REQUIRED',
        R6_formal_start='WAIT_R5_PRELOCKED_TRIGGER',
        qualification_script_sha256=sha256(Path(__file__)),source_module_sha256=sha256(ROOT/'src/bgb_en/r6_source.py'),
        elapsed_seconds=time.monotonic()-began)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(audit,stream,indent=2,allow_nan=False)
    print(json.dumps({key:audit[key] for key in ('status','primary_spots','fixed_program_gene_union',
        'measured_gene_count_min','measured_gene_count_max','program_replay_max_abs_error','new_model_fits','GPU_used','elapsed_seconds')}))


if __name__=='__main__':main()
