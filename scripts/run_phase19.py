import argparse
import json
import os
from pathlib import Path
import torch
from src.phase19_attention.data import P, R, cfg, source, write
from src.phase19_attention.train import train_cell
from src.phase10_trimodal.data import digest


def stem(arm, seed, fold):
    return f'{arm}_seed{seed}_fold{fold}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--stage', choices=['inner', 'outer'], required=True)
    args = parser.parse_args()
    assert args.arm in cfg()['arms'] and args.seed in cfg()['seeds'] and args.fold in cfg()['folds']
    label = stem(args.arm, args.seed, args.fold)
    target = P / args.stage / (label + '.pt')
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = source()
    assert json.loads((R/'PROTOCOL_LOCK.json').read_text())['source'] == identity
    assert json.loads((P/'PREPARATION_DONE.json').read_text())['source'] == identity
    for name in ['OPERATOR_AND_PIPELINE_TESTS', 'DATA_AUDIT', 'DATA_LEAKAGE_AUDIT', 'NATIVE_QUALIFICATION']:
        gate = json.loads((R/(name+'.json')).read_text())
        assert gate['status'] == 'PASS'
        assert gate.get('source', gate.get('training_source')) == identity
    if args.stage == 'outer':
        lock = json.loads((R/'EPOCH_SELECTION_LOCK.json').read_text())
        assert lock['source'] == identity and lock['status'] == 'ALL150_INNER_SELECTIONS_LOCKED'
    receipt = target.with_suffix('.json')
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert row['source'] == identity and row['checkpoint_sha256'] == digest(target)
        print('VERIFIED_COMPLETE_CELL', label, flush=True)
        return
    epochs, inner_hash = None, None
    if args.stage == 'outer':
        selection = json.loads((P / 'inner' / (label + '.json')).read_text())
        assert selection['source'] == identity
        inner_path = P / 'inner' / (label + '.pt')
        inner_hash = digest(inner_path)
        assert inner_hash == selection['checkpoint_sha256']
        epochs = selection['selected_epoch']
    ck = train_cell(args.arm, args.seed, args.fold, args.stage, epochs=epochs)
    assert ck['source'] == identity == source()
    ck['inner_checkpoint_sha256'] = inner_hash
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    torch.save(ck, tmp); tmp.replace(target)
    write(receipt, {'status': 'PASS', 'source': identity, 'arm': args.arm, 'seed': args.seed,
                    'fold': args.fold, 'role': args.stage, 'selected_epoch': ck['selected_epoch'],
                    'actual_epochs': ck['summary']['actual_epochs'], 'summary': ck['summary'],
                    'checkpoint_sha256': digest(target), 'inner_checkpoint_sha256': inner_hash})
    print('PHASE19_CELL_COMPLETE', label, args.stage, flush=True)


if __name__ == '__main__':
    main()
