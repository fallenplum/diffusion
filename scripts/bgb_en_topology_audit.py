"""Measure real selected-grid connectivity before survival fitting."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.bgb_en.data import read_patch_cache, require


def topology(coords, step, region=None):
    xy = np.asarray(coords, np.int64)
    points = set(map(tuple, xy))
    require(len(points) == len(xy), 'duplicate selected geometry')
    labels = {tuple(point): int(label) for point,label in zip(xy, np.zeros(len(xy)) if region is None else region)}
    allowed = lambda source,target: target in points and labels[source] == labels[target]
    edges = sum(allowed((x,y),(x+step,y)) for x,y in points)+sum(allowed((x,y),(x,y+step)) for x,y in points)
    degree = np.array([sum(allowed((x,y),(x+dx,y+dy)) for dx,dy in ((step,0),(-step,0),(0,step),(0,-step))) for x,y in xy])
    remaining, components = set(points), []
    while remaining:
        first = next(iter(remaining)); remaining.remove(first); queue = [first]
        for x,y in queue:
            for dx,dy in ((step,0),(-step,0),(0,step),(0,-step)):
                at=(x+dx,y+dy)
                if at in remaining and allowed((x,y),at):
                    remaining.remove(at); queue.append(at)
        components.append(len(queue))
    return {'nodes':len(xy), 'undirected_edges':int(edges), 'nodes_with_neighbor':int((degree>0).sum()),
            'neighbor_node_fraction':float((degree>0).mean()), 'components':len(components),
            'singleton_components':int(np.count_nonzero(np.asarray(components)==1)),
            'largest_component':max(components), 'four_axis_scan_transitions':int(2*edges)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cache=read_patch_cache(args.cache,verify_hash=False)
    rows=[topology(cache['coords'][index,cache['mask'][index]],int(cache['coordinate_step'][index]),
          None if 'region_ids' not in cache else cache['region_ids'][index,cache['mask'][index]]) for index in range(len(cache['owner']))]
    quantiles={key:{str(q):float(np.quantile([row[key] for row in rows],q)) for q in (0,.25,.5,.75,1)}
               for key in ('neighbor_node_fraction','largest_component','components','four_axis_scan_transitions')}
    result={'schema':'bgb_en_R0_selected_topology_v1','status':'PASS_GEOMETRY_DIAGNOSTIC','slides':len(rows),
            'selector':cache['audit']['selector'],'budget_per_patient':cache['audit']['budget_per_patient'],
            'selected_nodes':sum(row['nodes'] for row in rows),'undirected_edges':sum(row['undirected_edges'] for row in rows),
            'nodes_with_neighbor':sum(row['nodes_with_neighbor'] for row in rows),
            'neighbor_node_fraction':sum(row['nodes_with_neighbor'] for row in rows)/sum(row['nodes'] for row in rows),
            'all_singleton_slides':sum(row['undirected_edges']==0 for row in rows),
            'four_axis_scan_transitions':sum(row['four_axis_scan_transitions'] for row in rows),
            'slide_quantiles':quantiles,'outcome_reads':0,'model_fits':0,
            'geometry':'exact four-neighbor coordinates at raw-native hop; no fake adjacency across gaps',
            'window_boundary_reset_respected':'region_ids' in cache}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
