"""Full risk-set training with activation-checkpointed patient encoding."""
import copy, time
import numpy as np, torch
from torch.utils.checkpoint import checkpoint
from src.phase10_trimodal.math import fit_ridge, log_weights, torch_cox
from src.phase10_trimodal.features import derangement, ahash
from .models import SemanticModel


def modality_indices(d, ix, arm, seed):
    ix = np.asarray(ix, np.int64)
    ri = derangement(d, ix, seed) if arm == 'B7' else ix
    si = derangement(d, ix, seed+9000) if arm == 'D0' else ix
    return ri, si


def choose_ridge(x_train, x_val, t_train, e_train, t_val, e_val, grid):
    # This helper scores only inner validation, never accepts outer-test arrays.
    rows = []; best = None
    for lam in grid:
        w, diag = fit_ridge(x_train, t_train, e_train, lam)
        r = x_val @ w
        a, b = np.where((e_val[:,None] == 1)&((t_val[:,None] < t_val[None,:])|((t_val[:,None] == t_val[None,:])&(e_val[None,:] == 0))))
        if not len(a): raise ValueError('zero inner comparable pairs')
        diff = r[a]-r[b]; score = float(np.where(np.abs(diff) <= 1e-8, .5, diff > 0).mean())
        rows.append({'lambda': lam, 'inner_Harrell': score, **diag})
        if best is None or score > best[0]+1e-12: best = (score, lam)
    return best[1], rows


def anchor_inputs(f, d, ix, arm, seed):
    _, si = modality_indices(d, ix, arm, seed)
    ck = 'C_min' if arm == 'D2' else 'C'
    return np.c_[f[ck][ix], f['S'][si]].astype(float)


def forward_partition(model, d, f, cache, ix, arm, seed, chunk, gradients=False):
    ix = np.asarray(ix, np.int64); device = next(model.parameters()).device
    ri, si = modality_indices(d, ix, arm, seed)
    ck = 'C_min' if arm == 'D2' else 'C'; outs = []
    for start in range(0, len(ix), chunk):
        end = min(start+chunk, len(ix)); js = ix[start:end]
        c = torch.tensor(f[ck][js], device=device)
        s = torch.tensor(f['S'][si[start:end]], device=device)
        r = torch.tensor(f['R'][ri[start:end]], device=device)
        p, mask = cache.batch(js, device)
        def call(c, s, r, p, mask): return model(c, s, r, p, mask, details=True)
        outputs = checkpoint(call, c, s, r, p, mask, use_reentrant=False) if gradients else call(c, s, r, p, mask)
        outs.append(outputs)
    return tuple(torch.cat([o[k] for o in outs], 0) if k != 4 else torch.stack([o[k] for o in outs]).mean() for k in range(5))


def gradients(model):
    result = {}
    for name, module in [('S32', model.concept), ('RNA', model.rna), ('Patch', model.patch_projection),
                         ('semantic_query', model.cp), ('RNA_query', model.rp), ('clinical_RNA', model.cr), ('head', model.head), ('aux', model.aux)]:
        grad = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
        result[name] = float(torch.cat(grad).norm()) if grad else 0.
    result['clinical'] = float(model.clinical_weight.grad.norm()) if model.clinical_weight.grad is not None else 0.
    return result


def train(d, f, masks, cache, fit, val, arm, seed, cfg, lam, epochs=None, checkpoint_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.use_deterministic_algorithms(True)
    if device == 'cuda': torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    ck = 'C_min' if arm == 'D2' else 'C'
    membership = masks['true'] if arm == 'C3' else (masks['random'] if arm == 'C4' else masks['ordered'])
    model = SemanticModel(arm, masks[ck], membership, cfg['hidden_dim']).to(device)
    xa = anchor_inputs(f, d, fit, arm, cfg['split_seed']+101)
    w, _ = fit_ridge(xa, d['DSS_time'][fit], d['DSS_event'][fit], lam)
    with torch.no_grad(): model.anchor.weight.copy_(torch.tensor(w, device=device, dtype=torch.float32)[None, :])
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad(): before = forward_partition(model, d, f, cache, fit, arm, cfg['split_seed']+101, cfg['patient_chunk'])[1].cpu().numpy()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    loga = log_weights(d['DSS_time'][fit], d['DSS_event'][fit], device)
    e = torch.tensor(d['DSS_event'][fit], device=device)
    if len(val):
        lv = log_weights(d['DSS_time'][val], d['DSS_event'][val], device)
        ev = torch.tensor(d['DSS_event'][val], device=device)
    n = epochs if epochs is not None else cfg['inner_epochs']; ema = None; best = None; best_state = None; logs = []; start = time.monotonic()
    for epoch in range(1, n+1):
        model.train(); opt.zero_grad(set_to_none=True)
        r, reps, aux, correction, entropy = forward_partition(model, d, f, cache, fit, arm, cfg['split_seed']+101, cfg['patient_chunk'], True)
        main = torch_cox(r, loga, e)
        al = torch_cox(aux, loga, e) if arm in ['C5', 'D3'] else main.new_zeros(())
        loss = main + cfg.get(arm+'_aux_weight', .1)*al if arm in ['C5','D3'] else main
        loss.backward(); gs = gradients(model)
        connected = sum(p.numel() for p in model.parameters() if p.grad is not None)
        nonzero = sum(int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); opt.step(); model.eval()
        vl = vh = None
        with torch.no_grad():
            trisk = forward_partition(model, d, f, cache, fit, arm, cfg['split_seed']+101, cfg['patient_chunk'])[0]
            tl = float(torch_cox(trisk, loga, e))
            if len(val):
                vr = forward_partition(model, d, f, cache, val, arm, cfg['split_seed']+202, cfg['patient_chunk'])[0]
                vl = float(torch_cox(vr, lv, ev)); ema = vl if ema is None else cfg['ema_alpha']*vl+(1-cfg['ema_alpha'])*ema
                tv, ee = d['DSS_time'][val], d['DSS_event'][val]
                a, b = np.where((ee[:,None] == 1)&((tv[:,None] < tv[None,:])|((tv[:,None] == tv[None,:])&(ee[None,:] == 0))))
                vn = vr.cpu().numpy(); delta = vn[a]-vn[b]; vh = float(np.where(abs(delta) <= 1e-8, .5, delta > 0).mean())
                if epoch >= cfg['minimum_epoch'] and (best is None or ema < best[0]):
                    best = (ema, epoch, vh); best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        logs.append({'epoch': epoch, 'train_loss_before': float(main.detach()), 'train_loss_after': tl,
                     'aux_loss': float(al.detach()), 'val_loss': vl, 'val_EMA': ema, 'val_Harrell': vh,
                     'correction_sd': float(correction.detach().std()), 'correction_max': float(correction.detach().abs().max()),
                     'entropy': float(entropy.detach()), 'gradients': gs, 'connected_parameters': connected,
                     'nonzero_gradient_elements': nonzero})
        if not np.isfinite(tl) or not all(np.isfinite(v) for v in gs.values()): raise RuntimeError('nonfinite')
    selected = best[1] if best else n
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): after = forward_partition(model, d, f, cache, fit, arm, cfg['split_seed']+101, cfg['patient_chunk'])[1].cpu().numpy()
    if checkpoint_path:
        torch.save({'state': model.state_dict(), 'initial_state': initial_state, 'arm': arm, 'seed': seed,
                    'epochs': selected, 'clinical_mask': masks[ck], 'rna_membership': membership, 'cfg': cfg, 'fit_hash': ahash(np.asarray(fit, np.int64))}, checkpoint_path)
    info = {'parameters': sum(p.numel() for p in model.parameters()), 'selected_epoch': selected, 'actual_epochs': n,
            'selected_inner_Harrell': best[2] if best else None, 'logs': logs, 'seconds': time.monotonic()-start,
            'peak_cuda_MB': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else 0,
            'device': device, 'epoch_boundary': selected == n, 'initial_state': initial_state}
    return model, info, (before, after)
