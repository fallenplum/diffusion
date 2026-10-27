"""Run the installed dual-engine skill, retaining metadata rather than papers.

The skill scripts are not vendored. Pass --skill-scripts to the installed
paper_search/scripts directory. No patient data, model or endpoint is opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-scripts', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/bgb_en/LITERATURE_SEARCH.json'))
    parser.add_argument('--probe-only', action='store_true', help='Preserve discovery; append official code version probes')
    args = parser.parse_args()
    if args.probe_only:
        receipt = json.loads(args.output.read_text(encoding='utf-8'))
        references = {
            'mahmoodlab/SurvPath': ['models/model_SurvPath.py', 'models/layers/cross_attention.py'],
            'cs-w0r1d/qcagent': ['pipeline.py'],
            'synlp/SliceWorld': ['sliceworld/models/sliceworld.py'],
        }
        probes = []
        for repository, paths in references.items():
            item = {'repository': repository, 'requested_paths': paths, 'source': 'official_author_repository'}
            try:
                with urllib.request.urlopen(f'https://api.github.com/repos/{repository}/commits/main', timeout=30) as response:
                    commit = json.load(response)['sha']
                files = []
                for path in paths:
                    url = f'https://raw.githubusercontent.com/{repository}/{commit}/{path}'
                    with urllib.request.urlopen(url, timeout=30) as response:
                        source = response.read()
                    files.append({'path': path, 'url': url, 'bytes': len(source),
                                  'sha256': hashlib.sha256(source).hexdigest()})
                item.update(status='PASS_SOURCE_ACCESS_ONLY_NOT_REPRODUCED', commit=commit, files=files)
            except Exception as exc:
                item.update(status='BLOCKED_SOURCE_PROBE', error=f'{type(exc).__name__}: {exc}')
            probes.append(item)
        receipt['code_source_probes'] = probes
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(probes, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(args.skill_scripts))
    from hybrid_scholar import HybridScholar

    queries = ['SurvPath', 'QCAgent', 'SliceWorld', 'ME-Mamba', 'MCAMamba', 'TriBind-Mamba']

    def search(query):
        searcher = HybridScholar()
        result = json.loads(searcher.results_to_json(searcher.search_papers(query, limit=3)))
        for group in ('cross_validated', 'openalex_only', 'anysearch_only'):
            for paper in result[group]:
                paper.pop('abstract', None)
                paper.pop('citations', None)
        return {'query': query, 'engines_requested': ['openalex', 'anysearch'],
                'discovery_only': True, **result}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(search, queries))
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'SEARCH_DISCOVERY_REQUIRES_PRIMARY_SOURCE_VERIFICATION', 'queries': results,
               'caution': 'Zero matches and absence of cross-validation do not prove absence of a paper.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output),
                      'queries': [{q['query']: q['stats']} for q in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
