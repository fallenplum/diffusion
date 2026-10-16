"""Fit-local selection, exact complete-patient Efron gradients, fresh refits."""
import copy
import json
import random
import time
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .data import Data, P, cfg, source, write
from .model import AttentionModel, NativeModel
from src.phase10_trimodal.math import fit_ridge
from src.phase17_ssd.losses import full_risk_efron
from src.phase13_rna.math import inner_score
from src.phase10_trimodal.data import digest


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def file_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def simple_models(ds):
    target = P / 'simple' / f'{ds.role}_fold{ds.fold}.pt'
    # Permutations vary by seed, whereas convex raw-feature fitting does not.
    feature_audit = json.loads(json.dumps(
        {k: v for k, v in ds.audit.items() if k != 'wrong_controls'},
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), sort_keys=True))
    with file_lock(target.with_suffix('.lock')):
        if target.exists():
            receipt = json.loads(target.with_suffix('.json').read_text())
            assert receipt['checkpoint_sha256'] == digest(target) and receipt['source'] == source()
            saved = torch.load(target, map_location='cpu', weights_only=False)
            assert saved['source'] == source()
            assert np.array_equal(saved['fit'], ds.fit) and np.array_equal(saved['valid'], ds.valid)
            assert saved['preprocess_audit'] == feature_audit
            return saved
        models = {}
        for name, key in [('J0', 'X'), ('J_CS', 'CS')]:
            x = ds.f[key]
            w, diagnostic = fit_ridge(x[ds.fit], ds.ep['time'][ds.fit], ds.ep['event'][ds.fit],
                                     cfg()['training']['main_effect_l2'])
            models[name] = {'w': w, 'risk': np.asarray(x[ds.valid] @ w, np.float32),
                            'lambda': cfg()['training']['main_effect_l2'], 'diagnostic': diagnostic}
        saved = {'source': source(), 'fold': ds.fold, 'role': ds.role,
                 'fit': ds.fit, 'valid': ds.valid, 'preprocess_audit': feature_audit, 'models': models}
        tmp = target.with_suffix('.tmp')
        torch.save(saved, tmp); tmp.replace(target)
        write(target.with_suffix('.json'), {'status': 'PASS', 'source': source(),
              'fold': ds.fold, 'role': ds.role, 'checkpoint_sha256': digest(target),
              'unique_convex_solver_calls': 2})
        return saved


def make_model(ds, arm, w):
    config = cfg()
    if arm.startswith('A'):
        model = AttentionModel(ds.f['C'].shape[1], ds.f['membership'], w,
                               ds.f['C'][ds.fit].mean(0), arm, config)
    else:
        model = NativeModel(ds.f['C'].shape[1], ds.f['membership'], w, arm, config)
        model.core.prepare_data(ds)
    return model.to(ds.visual.device)


def assigned_mode(arm):
    return {'A4': 'wrongC', 'A5': 'wrongO'}.get(arm, 'matched')


def forward(model, ds, indices, mode='matched', training=False, diagnostic=False, with_aux=False):
    if mode not in ['matched', 'wrongC', 'wrongO', 'constantC', 'off']:
        raise ValueError(mode)
    indices = ds.check_indices(indices)
    risks, diagnostics, losses = [], [], []
    chunk = cfg()['training']['microbatch_patients']
    for begin in range(0, len(indices), chunk):
        patients = np.asarray(indices[begin:begin+chunk], int)
        slides = np.flatnonzero(np.isin(ds.owner, patients))
        owners = ds.owner[slides]
        if set(owners) != set(patients):
            raise ValueError('patient without a WSI')
        f = ds.ft
        ci = ds.wrong_c[owners] if mode == 'wrongC' else owners
        oi = ds.wrong_o[owners] if mode == 'wrongO' else owners
        ct = f['C'][ci]
        if mode == 'constantC':
            ct = f['C'][ds.fit].mean(0).expand_as(ct)
        context = tuple(f[k][oi] for k in ['R', 'G', 'L', 'RM', 'CM'])
        own = tuple(f[k][owners] for k in ['R', 'G', 'L', 'RM', 'CM'])
        native_rows = {k: v[owners] for k, v in f.items()
                       if k not in ['membership', 'X'] and v.ndim >= 1 and len(v) == len(f['X'])}
        for k in ['RPCA32', 'CNAPCA32']:
            if k in f:
                native_rows[k] = f[k][oi]
        wsi_ids = torch.as_tensor(ds.global_wsi_indices[slides], device=ds.visual.device)
        args = (ds.visual[slides], ds.mask[slides], f['C'][owners], *own, f['X'][owners], ct, *context)
        native = isinstance(model, NativeModel)
        def call(vv, mm, cc, rr, gg, ll, rmm, cmm, xx, ctxc, *ctxo,
                 nrows=native_rows, wi=wsi_ids, is_native=native):
            return model(vv, mm, cc, rr, gg, ll, rmm, cmm, xx,
                         context_clinical=ctxc, context_omics=ctxo,
                         interaction_off=mode == 'off', diagnostic=diagnostic,
                         native_features=nrows, wsi_indices=wi,
                         return_extra=with_aux and is_native)
        result = checkpoint(call, *args, use_reentrant=False, preserve_rng_state=True) if training else call(*args)
        if diagnostic:
            result, dg = result
            diagnostics.append({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in dg.items()})
        elif with_aux and native:
            result, extra = result
            losses.append(extra * len(patients) / len(indices))
        # Multi-slide survival is an equal-WSI patient scalar, never concatenated latent axes.
        risks.append(torch.stack([result[torch.as_tensor(owners == pi, device=result.device)].mean()
                                  for pi in patients]))
    risk = torch.cat(risks)
    if diagnostic:
        return risk, diagnostics
    if with_aux:
        return risk, sum(losses, risk.new_zeros(()))
    return risk


def _group_gradients(model):
    groups = {'main': [], 'visual': [], 'molecular': [], 'clinical_factor': [],
              'path_factor': [], 'omics_factor': [], 'qk': [], 'readout': [], 'native': []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = ('main' if name == 'main' else 'clinical_factor' if name.startswith('attention.c.')
                 else 'path_factor' if name.startswith('attention.a.') else 'omics_factor' if name.startswith('attention.b.')
                 else 'qk' if name.startswith(('attention.q.', 'attention.k.'))
                 else 'visual' if name.startswith('visual.') else 'molecular' if name.startswith('molecular.')
                 else 'native' if name.startswith('core.') else 'readout')
        groups[group].append(float(p.grad.detach().norm()) ** 2)
    result = {key+'_gradient': float(sum(value) ** .5) for key, value in groups.items()}
    cp = getattr(getattr(model, 'attention', None), 'c', None)
    gradient = cp.weight.grad if cp is not None else None
    result['clinical_nonintercept_gradient'] = 0. if gradient is None else float(gradient[:, 1:].norm())
    result['CP_global_intercept_gradient'] = 0. if gradient is None else float(gradient[:, :1].norm())
    return result


def _discriminator_step(model, ds, optimizer, mode):
    if optimizer is None:
        return 0.
    pars = list(model.core.discriminator_parameters())
    for p in pars:
        p.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    chunk, total = cfg()['training']['microbatch_patients'], 0.
    for start in range(0, len(ds.fit), chunk):
        patients = ds.fit[start:start+chunk]
        slides = np.flatnonzero(np.isin(ds.owner, patients)); own = ds.owner[slides]
        oi = ds.wrong_o[own] if mode == 'wrongO' else own
        frows = {k: v[own] for k, v in ds.ft.items() if k != 'membership' and len(v) == len(ds.f['X'])}
        for k in ['R', 'G', 'L', 'RM', 'CM', 'RPCA32', 'CNAPCA32']:
            frows[k] = ds.ft[k][oi]
        loss = model.core.discriminator_loss(ds.visual[slides], ds.mask[slides], frows,
                                             wsi_indices=ds.global_wsi_indices[slides])
        weighted = loss * len(patients) / len(ds.fit)
        weighted.backward(); total += float(weighted.detach())
    torch.nn.utils.clip_grad_norm_(pars, cfg()['training']['gradient_clip'])
    optimizer.step()
    for p in pars:
        p.requires_grad_(False)
    return total


def train_cell(arm, seed, fold, role, epochs=None, dataset=None, progress_callback=None):
    if role == 'outer' and epochs is None:
        raise ValueError('fresh outer fit requires the inner-selected epoch')
    seed_all(seed + 100*fold)
    ds = Data(fold, role, seed) if dataset is None else dataset
    config = cfg(); tc = config['training']
    initial = simple_models(ds)['models']['J0']['w']
    model = make_model(ds, arm, initial)
    dparams = (list(model.core.discriminator_parameters())
               if isinstance(model, NativeModel) and hasattr(model.core, 'discriminator_parameters') else [])
    dids = {id(p) for p in dparams}
    other = [p for name, p in model.named_parameters() if name != 'main' and id(p) not in dids]
    optimizer = torch.optim.AdamW([{'params': [model.main], 'weight_decay': 0.},
                                   {'params': other, 'weight_decay': tc['weight_decay']}], lr=tc['learning_rate'])
    dopt = torch.optim.AdamW(dparams, lr=tc['learning_rate'], weight_decay=tc['weight_decay']) if dparams else None
    for p in dparams:
        p.requires_grad_(False)
    mode = assigned_mode(arm)
    history, best_state, best_epoch, best, ema = [], None, 0, float('inf'), None
    length = int(epochs if epochs is not None else tc['max_epochs'])
    begin = time.monotonic()
    if ds.visual.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, length+1):
        model.train()
        dl = _discriminator_step(model, ds, dopt, mode)
        if dopt is not None and progress_callback is not None:
            progress_callback('discriminator', epoch)
        optimizer.zero_grad(set_to_none=True)
        risk, auxiliary = forward(model, ds, ds.fit, mode, training=True, with_aux=True)
        cox = full_risk_efron(risk.double(), ds.ep['time'][ds.fit], ds.ep['event'][ds.fit])
        penalty = .5 * tc['main_effect_l2'] * model.main.square().sum()
        loss = cox + penalty + auxiliary
        if not torch.isfinite(loss):
            raise ValueError('nonfinite complete risk-set objective')
        loss.backward()
        groups = _group_gradients(model)
        active = sum(p.numel() for p in model.parameters() if p.grad is not None)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'] + other, tc['gradient_clip']))
        optimizer.step()
        if progress_callback is not None:
            progress_callback('generator', epoch)
        row = {'epoch': epoch, 'train_loss': float(cox.detach()), 'penalty': float(penalty.detach()),
               'auxiliary_loss': float(auxiliary.detach()), 'discriminator_loss': dl,
               'gradient_norm': gradient_norm, **groups}
        if role == 'inner':
            model.eval()
            with torch.no_grad():
                vr = forward(model, ds, ds.valid, mode)
                vl = float(full_risk_efron(vr.double(), ds.ep['time'][ds.valid], ds.ep['event'][ds.valid]))
                hc = inner_score(ds.ep['time'][ds.valid], ds.ep['event'][ds.valid], vr.cpu().numpy())
            ema = vl if ema is None else tc['validation_ema_alpha']*vl + (1-tc['validation_ema_alpha'])*ema
            row.update(validation_loss=vl, validation_loss_EMA=ema, validation_Harrell=hc)
            if epoch >= min(tc['minimum_epoch'], length) and ema < best-1e-8:
                best, best_epoch = ema, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if epoch == 1 or epoch % 20 == 0:
            print('TRAIN', arm, seed, fold, role, epoch, round(row['train_loss'], 6),
                  row.get('validation_Harrell', 'NA'), round(time.monotonic()-begin, 1), flush=True)
        if role == 'inner' and epoch-best_epoch >= tc['early_stopping_patience']:
            break
    if role == 'inner':
        if best_state is None:
            raise RuntimeError('no legal inner checkpoint selected')
        model.load_state_dict(best_state)
    else:
        best_epoch = length
    model.eval()
    with torch.no_grad():
        predictions = {m: forward(model, ds, ds.valid, m).cpu().numpy()
                       for m in ['matched', 'wrongC', 'wrongO', 'constantC', 'off']}
    mechanism = probe(model, ds, mode)
    summary = {'registered_parameters': sum(p.numel() for p in model.parameters()),
               'last_step_gradient_parameters': active, 'discriminator_parameters': sum(p.numel() for p in dparams),
               'actual_epochs': len(history), 'selected_epoch': best_epoch,
               'hit_maximum': role == 'inner' and len(history) == tc['max_epochs'],
               'seconds': time.monotonic()-begin,
               'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20 if ds.visual.is_cuda else 0.,
               'last_step_gradients': groups,
               'discriminator_optimizer_steps': len(history) if dopt is not None else 0}
    return {'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'arm': arm, 'seed': seed, 'fold': fold, 'role': role, 'source': source(),
            'fit': ds.fit, 'validation_or_test': ds.valid, 'main_initial_w': initial,
            'selected_epoch': best_epoch, 'history': history, 'preprocess_audit': ds.audit,
            'risk': predictions[mode], 'assigned_mode': mode, 'matched_risk': predictions['matched'],
            'wrongC_risk': predictions['wrongC'], 'wrongO_risk': predictions['wrongO'],
            'constantC_risk': predictions['constantC'], 'interaction_off_risk': predictions['off'],
            'summary': summary, 'mechanism': mechanism}


def probe(model, ds, mode):
    """Label-blind first32 training patients, aggregate-only diagnostic outputs."""
    ix = ds.fit[:32]
    model.eval()
    with torch.no_grad():
        primary, diag = forward(model, ds, ix, mode, diagnostic=True)
        true, true_diag = forward(model, ds, ix, 'matched', diagnostic=True)
        wrongc, dc = forward(model, ds, ix, 'wrongC', diagnostic=True)
        wrongo = forward(model, ds, ix, 'wrongO')
        main = forward(model, ds, ix, 'off')
    out = {'probe_training_patients': len(ix), 'probe_WSI_instances': int(np.isin(ds.owner, ix).sum()),
           'main_risk_SD': float(main.std(unbiased=False)),
           'interaction_risk_SD': float((primary-main).std(unbiased=False)),
           'matched_wrongC_risk_MAE': float((true-wrongc).abs().mean()),
           'matched_wrongO_risk_MAE': float((true-wrongo).abs().mean())}
    def flat(key, which):
        return torch.cat([d[key].reshape(-1) for d in which if key in d])
    if diag and 'p_to_o' in diag[0]:
        a, b = flat('p_to_o', true_diag), flat('p_to_o', dc)
        out['clinical_attention_relative_L2'] = float((a-b).norm()/a.norm().clamp_min(1e-12))
        for key in ['base_scores', 'trilinear_scores']:
            values = torch.cat([d[key][d['patch_mask'][:, :, None] & d['omics_mask'][:, None, :]] for d in diag])
            out[key+'_SD'] = float(values.std(unbiased=False))
        used = torch.cat([d['scores'][d['patch_mask'][:, :, None] & d['omics_mask'][:, None, :]] for d in diag])
        correction = torch.cat([(d['scores']-d['base_scores'])[d['patch_mask'][:, :, None] & d['omics_mask'][:, None, :]] for d in diag])
        out['actual_combined_scores_SD'] = float(used.std(unbiased=False))
        out['actual_score_correction_SD'] = float(correction.std(unbiased=False))
        out['CP_scores_used_in_attention'] = model.arm in ['A2', 'A3', 'A4', 'A5']
        for key, mk in [('p_to_o', 'patch_mask'), ('o_to_p', 'omics_mask')]:
            distributions = torch.cat([d[key][d[mk]] for d in diag])
            h = -(distributions * distributions.clamp_min(1e-12).log()).sum(-1)
            out[key+'_entropy_mean'] = float(h.mean())
            out[key+'_maximum_weight_mean'] = float(distributions.max(-1).values.mean())
    out['isolated_input_gradient_available'] = isinstance(model, AttentionModel)
    for key, name in ([('C', 'clinical'), ('R', 'RNA')] if isinstance(model, AttentionModel) else []):
        values = ds.ft[key]
        values.requires_grad_(True)
        isolated = forward(model, ds, ix[:1], mode) - forward(model, ds, ix[:1], 'off')
        grad = torch.autograd.grad(isolated.sum(), values, allow_unused=True)[0]
        out['isolated_interaction_'+name+'_gradient_L2'] = 0. if grad is None else float(grad.norm())
        values.requires_grad_(False)
    # Raw risks and maps are deliberately not returned by this public aggregate probe.
    return out
