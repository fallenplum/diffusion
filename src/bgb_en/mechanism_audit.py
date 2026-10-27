"""Post-training mechanism arithmetic, without optimization or regularization."""
from __future__ import annotations
import numpy as np
import torch


def statistics(values):
    a = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    a = np.asarray(a, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), SD=float(a.std()), RMS=float(np.sqrt(np.mean(a*a))),
                max_abs=float(np.abs(a).max())) if a.size else dict(n=0, mean=None, SD=None, RMS=None, max_abs=None)


def bounded_components(output, gamma):
    """Mirror the frozen trainer; divide two-direction tanh by numeric Shapley.

    Directional Shapley terms sum exactly to the bounded interaction readout.
    They allocate a two-input mathematical readout, not causal contributions.
    """
    a, b = output['OtoP_risk'], output['PtoO_risk']
    cross = gamma*.5*torch.tanh(a+b)
    return dict(pooled_path=gamma*.25*torch.tanh(output['pooled_path_risk']),
                pooled_omics=gamma*.25*torch.tanh(output['pooled_omics_risk']),
                interaction=cross,
                OtoP_drop_one=gamma*.5*(torch.tanh(a+b)-torch.tanh(b)),
                PtoO_drop_one=gamma*.5*(torch.tanh(a+b)-torch.tanh(a)),
                OtoP_SharedReadout=gamma*.25*(torch.tanh(a)+torch.tanh(a+b)-torch.tanh(b)),
                PtoO_SharedReadout=gamma*.25*(torch.tanh(b)+torch.tanh(a+b)-torch.tanh(a)))


def independent_efron_gradient(time, event, risk):
    """Full validation loss derivative; all later probes freeze this derivative."""
    t, e = np.asarray(time, float), np.asarray(event, int)
    if not e.sum():
        raise ValueError('No validation events for the Cox chain-gradient probe')
    r = torch.as_tensor(np.asarray(risk, float), dtype=torch.float64).requires_grad_(True)
    loss = -r[torch.as_tensor(e == 1)].sum()
    for instant in np.unique(t[e == 1]):
        deaths = np.flatnonzero((t == instant) & (e == 1))
        at_risk = np.flatnonzero(t >= instant)
        log_risk = torch.logsumexp(r[torch.as_tensor(at_risk)], 0)
        log_death = torch.logsumexp(r[torch.as_tensor(deaths)], 0)
        ratio = torch.exp(log_death-log_risk)
        for tie in range(len(deaths)):
            loss = loss+log_risk+torch.log1p(-float(tie)/len(deaths)*ratio)
    loss = loss/int(e.sum())
    grad, = torch.autograd.grad(loss, r)
    if not torch.isfinite(grad).all():
        raise ValueError('Nonfinite independent Efron risk gradient')
    return dict(loss=float(loss.detach()), gradient=grad.detach().numpy(), gradient_sum=float(grad.sum()))


def _probabilities(row):
    attention = row['attention'].detach().to(dtype=torch.float64)
    null = row['null'].detach().to(dtype=torch.float64)
    total = torch.cat((attention, null[..., None]), -1)
    present = total.sum(-1) > 0
    return attention, null, total, present


def attention_summary(row, valid_key_count=None):
    attention, null, full, present = _probabilities(row)
    entropy = -(full*full.clamp_min(1e-300).log()).sum(-1)
    real_mass = attention.sum(-1)
    conditional = attention/real_mass[..., None].clamp_min(1e-300)
    relative_entropy = -(conditional*conditional.clamp_min(1e-300).log()).sum(-1)
    evidence = present & (real_mass > 0)
    ranks = []
    # Collapse each patient's WSI independently. Invalid query/key rows are
    # removed; the entropy rank uses squared singular values as its spectrum.
    for batch in range(len(conditional)):
        for slide in range(conditional.shape[1]):
            matrix = conditional[batch, slide].cpu().numpy()
            rows = matrix.sum(-1) > 0
            columns = matrix.sum(0) > 0
            matrix = matrix[rows][:, columns]
            if matrix.size:
                singular = np.linalg.svd(matrix, compute_uv=False)
                spectrum = singular**2
                spectrum = spectrum/spectrum.sum()
                ranks.append(float(np.exp(-np.sum(spectrum*np.log(np.maximum(spectrum, 1e-300))))))
    gate = row['gate'].detach().to(dtype=torch.float64)
    gate_valid = present.any(1) if gate.ndim == 3 else present
    message_norm = row['message'].detach().norm(dim=-1)
    message_valid = present.any(1) if message_norm.ndim == 2 else present
    fraction = None
    if valid_key_count is not None:
        count = torch.broadcast_to(valid_key_count.to(device=present.device, dtype=torch.float64), present.shape)
        fraction = statistics((relative_entropy.exp()/count.clamp_min(1))[evidence])
    return dict(valid_query_rows=int(present.sum()), attention_entropy_including_null=statistics(entropy[present]),
                relative_effective_keys=statistics(relative_entropy[evidence].exp()),
                relative_effective_key_fraction=fraction,
                null_mass=statistics(null[present]), nonnull_mass=statistics(real_mass[present]),
                nonzero_selected_keys=statistics((attention > 0).sum(-1)[present]),
                query_key_effective_rank=statistics(np.asarray(ranks)),
                rank_definition='exp entropy of normalized squared singular values of conditional real-key attention',
                gate=statistics(gate[gate_valid]),
                gate_fraction_below_005=float((gate[gate_valid] < .05).double().mean()) if gate_valid.any() else None,
                gate_fraction_above_095=float((gate[gate_valid] > .95).double().mean()) if gate_valid.any() else None,
                message_L2=statistics(message_norm[message_valid]))


def attention_change(left, right):
    _, _, a, present_a = _probabilities(left)
    _, _, b, present_b = _probabilities(right)
    if a.shape != b.shape:
        raise ValueError('Attention controls changed canonical support dimensions')
    both = present_a & present_b
    tv = .5*(a-b).abs().sum(-1)
    aa, bb = a[..., :-1], b[..., :-1]
    topk = min(64, aa.shape[-1])
    atop = torch.zeros_like(aa, dtype=torch.bool).scatter(-1, aa.topk(topk, dim=-1).indices, True) & (aa > 0)
    btop = torch.zeros_like(bb, dtype=torch.bool).scatter(-1, bb.topk(topk, dim=-1).indices, True) & (bb > 0)
    intersection = (atop & btop).sum(-1).double()
    union = (atop | btop).sum(-1).double()
    return dict(TV=statistics(tv[both]), both_present_query_rows=int(both.sum()),
                top64_key_Jaccard=statistics((intersection/union.clamp_min(1))[both]),
                one_sided_query_rows=int((present_a ^ present_b).sum()),
                both_absent_query_rows=int((~present_a & ~present_b).sum()))


def loss_chain_probe(contribution, full_loss_risk_gradient, model, rna_input, patch_input,
                     *, retain_graph=True):
    """Probe only loss -> actual bounded interaction -> routes and raw inputs.

    No parameter regularizer, pooled readout, anchor or synthetic norm loss is
    included. Frozen full-validation dLoss/dr multiplies the sampled patients'
    interaction contributions. This is a finite-sample chain sensitivity audit.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    targets = [p for _, p in named]+[rna_input, patch_input]
    incoming = torch.as_tensor(full_loss_risk_gradient, device=contribution.device, dtype=contribution.dtype)
    objective = (incoming*contribution).sum()
    if objective.requires_grad:
        gradients = torch.autograd.grad(objective, targets, allow_unused=True, retain_graph=retain_graph)
    else:
        gradients = [None]*len(targets)
    records = []
    for (name, parameter), gradient in zip(named, gradients[:-2]):
        records.append(dict(name=name, parameters=parameter.numel(), connected=gradient is not None,
                            gradient_L2=0. if gradient is None else float(gradient.detach().norm()),
                            gradient_max_abs=0. if gradient is None else float(gradient.detach().abs().max())))
    inputs = {}
    for name, gradient in zip(('RNA_input', 'patch_input'), gradients[-2:]):
        inputs[name] = dict(connected=gradient is not None, gradient_L2=0. if gradient is None else float(gradient.detach().norm()),
                            gradient_max_abs=0. if gradient is None else float(gradient.detach().abs().max()))
    return dict(objective=float(objective.detach()), parameter_gradients=records, input_gradients=inputs,
                regularization_included=False, pooled_main_effects_included=False,
                gradient_source='frozen full-validation Efron dLoss/dr times sampled actual interaction risk')
