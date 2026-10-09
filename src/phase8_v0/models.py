"""C3R-Surv low-capacity residual heads and full-batch training."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from .cox import concordance


ARM_SPEC = {
    "A1": {"p": True, "o": False, "i": False, "residual": False, "gate": False},
    "A2": {"p": False, "o": True, "i": False, "residual": False, "gate": False},
    "A3": {"p": True, "o": True, "i": False, "residual": False, "gate": False},
    "A4": {"p": True, "o": True, "i": True, "residual": False, "gate": False},
    "A5": {"p": True, "o": True, "i": True, "residual": True, "gate": False},
    "A6": {"p": True, "o": True, "i": True, "residual": True, "gate": True},
}


class C3RSurv(nn.Module):
    def __init__(self, c_dim, p_dim=64, o_dim=32, s_dim=32, rank=4, use_p=True, use_o=True, use_i=False, use_gate=False):
        super().__init__()
        self.use_p, self.use_o, self.use_i, self.use_gate = use_p, use_o, use_i, use_gate
        self.p_head = nn.Linear(p_dim, 1, bias=False) if use_p else None
        self.o_head = nn.Linear(o_dim, 1, bias=False) if use_o else None
        self.q = nn.Linear(s_dim, rank, bias=False) if use_i else None
        self.k = nn.Linear(o_dim, rank, bias=False) if use_i else None
        self.i_head = nn.Linear(rank, 1, bias=False) if use_i else None
        self.gate = nn.Linear(c_dim, 1) if use_gate else None
        self.alpha_p = nn.Parameter(torch.zeros(())) if use_p else None
        self.alpha_o = nn.Parameter(torch.zeros(())) if use_o else None
        self.alpha_i = nn.Parameter(torch.zeros(())) if use_i else None
        self.reset_nonzero()

    def reset_nonzero(self):
        for module in (self.p_head, self.o_head, self.q, self.k, self.i_head, self.gate):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, c, p, o, s, r_c):
        zero = r_c.new_zeros(r_c.shape)
        r_p = self.p_head(p).squeeze(-1) if self.p_head is not None else zero
        r_o = self.o_head(o).squeeze(-1) if self.o_head is not None else zero
        if self.i_head is not None:
            q = self.q(s); k = self.k(o); h = q * k; r_i = self.i_head(h).squeeze(-1)
        else:
            q = k = h = None; r_i = zero
        gate = torch.sigmoid(self.gate(c).squeeze(-1)) if self.gate is not None else torch.ones_like(r_c)
        risk = r_c + (self.alpha_p * r_p if self.alpha_p is not None else 0.) + (self.alpha_o * r_o if self.alpha_o is not None else 0.) + (self.alpha_i * gate * r_i if self.alpha_i is not None else 0.)
        z = torch.stack([r_c, r_p, r_o, r_i, gate, risk], dim=1)
        return z, risk, {"r_C": r_c, "delta_P": r_p, "delta_O": r_o, "delta_I": r_i, "gate": gate, "h_int": h}

    def zero_residual_risk(self, r_c):
        return r_c


def efron_loss(risk, time_values, event_values):
    losses = []
    for t in torch.unique(time_values[event_values == 1]):
        die = (time_values == t) & (event_values == 1)
        count = int(die.sum().item())
        riskset = time_values >= t
        max_r = risk[riskset].max()
        exp_all = torch.exp(risk[riskset] - max_r).sum()
        exp_die = torch.exp(risk[die] - max_r).sum()
        for l in range(count):
            losses.append(torch.log(exp_all - (float(l) / count) * exp_die) + max_r)
        losses.append(-risk[die].sum())
    return torch.stack(losses).sum() / event_values.sum().clamp_min(1)


def _tensor_block(blocks, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in blocks.items()}


def train_head(spec, blocks, r_c, time_values, event_values, epochs, seed, lr=1e-3, weight_decay=1e-3, lambda_delta=1.0, lambda_l2=1e-3, inner_val=None, val_blocks=None, val_r_c=None, val_time=None, val_event=None, patience=15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = C3RSurv(c_dim=blocks["C"].shape[1], use_p=spec["p"], use_o=spec["o"], use_i=spec["i"], use_gate=spec["gate"]).to(device)
    x = _tensor_block(blocks, device); rc = torch.as_tensor(r_c, dtype=torch.float32, device=device)
    xv = _tensor_block(val_blocks, device) if val_blocks is not None else None
    rcv = torch.as_tensor(val_r_c, dtype=torch.float32, device=device) if val_r_c is not None else None
    tt = torch.as_tensor(time_values, dtype=torch.float32, device=device); ee = torch.as_tensor(event_values, dtype=torch.int64, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = -np.inf; best_epoch = 1; stale = 0; losses = []; grad_trace = []; peak = 0
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        _, risk, _ = model(x["C"], x["P"], x["O"], x["S"], rc)
        loss = efron_loss(risk, tt, ee) + lambda_delta * torch.mean((risk - rc) ** 2)
        l2 = torch.zeros((), device=device)
        for name, par in model.named_parameters():
            if "alpha" not in name: l2 = l2 + (par ** 2).sum()
        loss = loss + lambda_l2 * l2
        loss.backward(); grad = max(float(par.grad.detach().abs().max()) for par in model.parameters() if par.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step()
        losses.append(float(loss.detach().cpu())); grad_trace.append(float(grad)); peak = max(peak, int(torch.cuda.max_memory_allocated() if device == "cuda" else 0))
        if val_blocks is not None:
            model.eval()
            with torch.no_grad():
                _, rv, _ = model(xv["C"], xv["P"], xv["O"], xv["S"], rcv)
            metric = concordance(val_time, val_event, rv.detach().cpu().numpy())
            if metric > best + 1e-12:
                best, best_epoch, stale = metric, epoch, 0
            else:
                stale += 1
                if stale >= patience: break
    return model, {"selected_epoch": int(best_epoch), "epochs_run": len(losses), "inner_best_harrell": float(best) if val_blocks is not None else None,
                   "train_loss_last": losses[-1], "max_grad": float(grad), "peak_gpu_bytes": peak, "device": device,
                   "loss_trace": losses, "gradient_trace": grad_trace}


def fit_for_epochs(spec, blocks, r_c, time_values, event_values, epochs, seed, **kwargs):
    return train_head(spec, blocks, r_c, time_values, event_values, epochs=epochs, seed=seed, inner_val=None, **kwargs)
