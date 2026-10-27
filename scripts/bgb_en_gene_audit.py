"""Distinguish the historical RNA512 selection universe from Hallmark50."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.bgb_en.data import load_raw, require, sha256


def audit(raw, direct_groups, hallmark_gmt):
    old = json.loads(Path(direct_groups).read_text(encoding='utf-8-sig'))
    require(len(old) == 8 and all(isinstance(value, list) for value in old.values()), 'RNA512 requires the exact legacy eight-group registry')
    hallmark = {}
    for line in Path(hallmark_gmt).read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        require(len(fields) > 2 and fields[0].startswith('HALLMARK_'), 'invalid external Hallmark row')
        require(fields[0] not in hallmark, 'duplicate Hallmark program name')
        hallmark[fields[0]] = set(fields[2:])
    require(len(hallmark) == 50, 'expected 50 fixed Hallmark programs')
    old = {key: set(value) for key, value in old.items()}
    RNA, CNA = set(raw['rna_symbols']), set(raw['cna_symbols'])
    direct_union, hallmark_union = set().union(*old.values()), set().union(*hallmark.values())
    row = lambda key, members: {'program': key, 'external_members': len(members),
                               'canonical_RNA_overlap': len(members & RNA),
                               'canonical_CNA_overlap': len(members & CNA),
                               'RNA_CNA_overlap': len(members & RNA & CNA)}
    return {'schema': 'bgb_en_R0_gene_sources_v1', 'status': 'PASS', 'outcome_reads': 0,
            'direct_RNA512': {'source_sha256': sha256(direct_groups), 'programs': 8,
                              'union_members': len(direct_union), 'canonical_RNA_union': len(direct_union & RNA),
                              'program_coverage': [row(key, value) for key, value in old.items()],
                              'selection_rule': 'fit-local coverage>=0.90; descending fit SD inside eight-program union; legacy fallback explicit'},
            'pathway_tokens': {'source_sha256': sha256(hallmark_gmt), 'programs': 50,
                               'union_members': len(hallmark_union), 'canonical_RNA_union': len(hallmark_union & RNA),
                               'canonical_CNA_union': len(hallmark_union & CNA),
                               'program_coverage': [row(key, value) for key, value in hallmark.items()]},
            'union_RNA_intersection': len(direct_union & hallmark_union & RNA),
            'legacy_programs_exactly_equal_to_same_named_Hallmark': {
                key: value == hallmark.get(key) for key, value in old.items()},
            'gene_membership_public': False, 'gene_selection_or_statistics_fits': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-json', required=True)
    parser.add_argument('--direct-groups', required=True)
    parser.add_argument('--hallmark-gmt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = load_raw(json.loads(Path(args.assets_json).read_text(encoding='utf-8-sig')), include_endpoints=False)
    result = audit(raw, args.direct_groups, args.hallmark_gmt)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
