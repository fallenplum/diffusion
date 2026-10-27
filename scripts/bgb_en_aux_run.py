"""Matched UNI2 external-program initialization experiment, separate from core.

Reuse verified common teachers without relabeling their original source. Only
the same 1536D projection differs between the two truly retrained treatments.
"""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts import bgb_en_run as driver
from src.bgb_en.external import initialize_pathology_project

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path)
    p.add_argument('--primary-private',required=True,type=Path);p.add_argument('--private',required=True,type=Path)
    p.add_argument('--project',type=Path);p.add_argument('--arm',choices=['F5_UNI2','F11_UNI2'],required=True)
    p.add_argument('--partition',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    args=p.parse_args();args.stage='R3';args.outer_fold=1;args.config=ROOT/'configs/bgb_en/prototype_v1.json'
    torch.set_num_threads(2)
    if args.arm=='F11_UNI2' and (args.project is None or not args.project.is_file()):raise ValueError('Qualified source projection is required')
    assets,config,raw,member,core_lineage=driver.load_inputs(args)
    expected=copy.deepcopy(core_lineage)
    expected['auxiliary_source']={name:driver.sha256(ROOT/name) for name in ('scripts/bgb_en_aux_run.py','src/bgb_en/external.py')}
    expected['shared_teacher_origin_lineage']=core_lineage
    expected['UNI2_cache_receipt_sha256']=driver.sha256(Path(assets['patch_uni2'])/'READY.json')
    expected['external_projection_sha256']=driver.sha256(args.project) if args.project else None
    core_prepare=driver.prepare
    def reuse_prepared(inner_args,raw,member,lineage):
        original_args=copy.copy(inner_args);original_args.private=args.primary_private
        path,receipt=driver.prep_paths(original_args)
        if not path.is_file() or not receipt.is_file():raise ValueError('Auxiliary experiment cannot silently create teachers')
        return core_prepare(original_args,raw,member,core_lineage)
    driver.prepare=reuse_prepared
    original_model=driver.MechanismModel
    class SourceInitialized(original_model):
        def __init__(self,*values,**kwargs):
            super().__init__(*values,**kwargs)
            if self.arm=='F11_UNI2':initialize_pathology_project(self.path,args.project,driver.sha256(args.project))
    driver.MechanismModel=SourceInitialized
    driver.run_unit(args,assets,config,raw,member,expected)

if __name__=='__main__':main()
