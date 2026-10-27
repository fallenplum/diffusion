"""Read-only cast-only anchor ranking audit and teacher-local scale replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def comparable(t, e):
    left, right = [], []
    for event in np.flatnonzero(e == 1):
        later = np.flatnonzero((t > t[event]) | ((t == t[event]) & (e == 0)))
        left.extend([int(event)]*len(later)); right.extend(later.tolist())
    return np.asarray(left, int), np.asarray(right, int)


def censor_weights(train_t, train_e, t, e):
    times = np.unique(train_t)
    probability = []
    survival = 1.
    for time in times:
        cens = np.count_nonzero((train_t == time) & (train_e == 0))
        denominator = np.count_nonzero(train_t >= time)-np.count_nonzero((train_t == time) & (train_e == 1))
        if cens:
            survival *= 1-float(cens)/float(denominator)
        probability.append(survival)
    weight = np.zeros(len(t), dtype=np.float64)
    for i in np.flatnonzero((e == 1) & (t < 60)):
        before = np.searchsorted(times, t[i], side='right')-1
        value = 1. if before < 0 else probability[before]
        if value <= 0:
            raise ValueError('Independent reverse-KM support failure')
        weight[i] = 1/value**2
    return weight


def compare_cast(risk, valid, t, e, train_t, train_e):
    original = np.asarray(risk, np.float64)
    cast = torch.as_tensor(original, dtype=torch.float32).numpy().astype(np.float64)
    r, c = original[valid], cast[valid]
    left, right = comparable(t, e)
    weight = censor_weights(train_t, train_e, t, e)
    def credits(value):
        margin = value[left]-value[right]
        return np.where(np.abs(margin) <= 1e-8, .5, (margin > 0).astype(float))
    old, new = credits(r), credits(c)
    difference = new-old
    hd = len(left); ud = float(weight[left].sum())
    error = c-r
    return dict(max_abs_error=float(np.abs(error).max()), RMS_error=float(np.sqrt(np.mean(error**2))),
        all_rows_max_abs_error=float(np.abs(cast-original).max()),
        Harrell64=float(old.sum()/hd), Harrell_cast32=float(new.sum()/hd),
        delta_Harrell=float(difference.sum()/hd), Harrell_num_change=float(difference.sum()),
        Harrell_eligible_pairs=hd, Uno6064=float(np.dot(weight[left], old)/ud),
        Uno60_cast32=float(np.dot(weight[left], new)/ud),
        delta_Uno60=float(np.dot(weight[left], difference)/ud),
        Uno_num_change=float(np.dot(weight[left], difference)), Uno_eligible_weight=ud,
        changed_pair_credits=int(np.count_nonzero(difference)),
        improved_pair_credit=float(np.maximum(difference, 0).sum()),
        damaged_pair_credit=float(np.maximum(-difference, 0).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher-replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    assets = json.loads(args.assets.read_text())
    with np.load(assets['endpoints'], allow_pickle=False) as f:
        time = np.array(f['DSS_time'], dtype=float)
        event = np.array(f['DSS_event'], dtype=int)
    raw = None
    if args.teacher_replay:
        from src.bgb_en.data import load_raw, read_patch_cache
        from src.bgb_en.replay import apply_saved_preprocessing
        raw = load_raw(assets, include_endpoints=False)
        cache = read_patch_cache(assets['patch_grid'], verify_hash=False)
        raw['conch_mean'] = np.asarray(cache['full_mean'], np.float64)
    rows = []
    for partition in range(3):
        path = args.private/'prepared'/f'outer1_inner{partition}.pt'
        prepared = torch.load(path, map_location='cpu', weights_only=False)
        if prepared['lineage']['endpoints_sha256'] != sha(assets['endpoints']):
            raise ValueError('Endpoint SHA differs from prepared lineage')
        valid, train = prepared['valid'], prepared['train']
        comparisons = {name: compare_cast(prepared[key], valid, time[valid], event[valid], time[train], event[train])
                       for name, key in [('F0_CP', 'CP'), ('R1_CS', 'CS'), ('RNA_only', 'RNA'), ('F1_CP_betaRNA', 'base')]}
        teacher_checks = []
        if args.teacher_replay:
            for group, teacher in enumerate(prepared['teachers']):
                fit, held = np.asarray(teacher['train']), np.asarray(teacher['held'])
                if (np.intersect1d(fit, held).size or np.intersect1d(fit, valid).size
                    or np.intersect1d(held, valid).size or np.any(prepared['teacher_groups'][held] != group)):
                    raise ValueError('Teacher fit/held/validation isolation differs')
                features = apply_saved_preprocessing(raw, teacher['preprocessing'])
                for key in ('LOCAL', 'CS', 'R'):
                    model = teacher['models'][key]
                    prediction = features[key] @ model['coef']
                    saved = model['scale']
                    own_mean, own_sd = float(prediction[fit].mean()), float(prediction[fit].std())
                    held_mean, held_sd = float(prediction[held].mean()), float(prediction[held].std())
                    mean_error, sd_error = abs(own_mean-saved['mean']), abs(own_sd-saved['sd'])
                    if max(mean_error, sd_error) > 1e-12:
                        raise ValueError('Teacher risk scale is not derived from its own fitting rows')
                    teacher_checks.append(dict(teacher_group=group, representation=key,
                        fit_mean_replay_abs_error=mean_error, fit_SD_replay_abs_error=sd_error,
                        held_SD_minus_saved_SD=float(held_sd-saved['sd']),
                        held_mean_minus_saved_mean=float(held_mean-saved['mean']),
                        own_teacher_fit_rows=len(fit), same_teacher_held_rows=len(held),
                        scale_source='teacher_own_training_rows_not_heldout'))
        rows.append(dict(partition=partition, prepared_sha256=sha(path), validation_rows=len(valid),
                         cast_only=comparisons, teacher_scale_replay=teacher_checks))
    result = dict(status='PASS_READ_ONLY_CAST_AND_SCALE_AUDIT', optimizer_updates=0,
                  risk_cast_definition='float64 prepared offset -> torch.float32 -> numpy.float64 before evaluator',
                  tied_tolerance=1e-8, reverse_KM='training-partition-only; event removed before tied censor jump',
                  rows=rows,
                  any_cast_pair_credit_change=any(v['changed_pair_credits'] for row in rows for v in row['cast_only'].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
