"""Lossless JSON whitespace compaction for the fixed publication size limit."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--private-backup',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();backup=args.private_backup.resolve()
    if backup.is_relative_to(root):
        raise ValueError('ORIGINAL_REPORT_BACKUP_MUST_BE_OUTSIDE_REPOSITORY')
    backup.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in sorted((root/'reports/bgb_en_r6').glob('*.json')):
        before=path.read_bytes()
        if len(before)<=2*1024**2:
            continue
        value=json.loads(before)
        after=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')+b'\n'
        if json.loads(after)!=value:
            raise ValueError('REPORT_VALUE_CHANGED')
        if len(after)>2*1024**2:
            raise ValueError('LOSSLESS_COMPACTION_STILL_OVERSIZED_NEEDS_EXPLICIT_SPLIT')
        (backup/path.name).write_bytes(before)
        path.write_bytes(after)
        if json.loads(path.read_bytes())!=value:
            raise ValueError('WRITTEN_REPORT_VALUE_CHANGED')
        rows.append(dict(file=path.relative_to(root).as_posix(),
            original_bytes=len(before),compact_bytes=len(after),
            original_sha256=sha(before),compact_sha256=sha(after),
            parsed_JSON_values_unchanged=True))
    receipt=dict(status='PASS_JSON_WHITESPACE_ONLY_ALL_VALUES_PRESERVED',files=rows,
        new_training_or_model_selection=False,publication_size_limit_changed=False)
    (backup/'SERIALIZATION_RECEIPT.private.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
