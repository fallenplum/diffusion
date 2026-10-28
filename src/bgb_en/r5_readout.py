"""R5 risk-readout changes, leaving the locked R0--R4 modules untouched.

``legacy`` preserves the original multiplication/addition order, not just the
real-number formula.  The formal rank-four replacement has no legacy skip. Its
first two GELU channels are initialized antisymmetrically from the saved
directional heads, so GELU(z)-GELU(-z)=z gives the linear A4 initialization (up to
float arithmetic).  Its proximity to tanh on a real initial batch is a separate
qualification, not an identity for all z.

All molecular/path summaries used by rank4 are *updated minus initial* states.
The sham readout instead adds two unimodal functions, with a shared up matrix;
sharing parameters does not introduce a cross-modal term in its forward map.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .interaction import masked_mean, path_mean
from .training import MechanismModel


ARM_MODES = {
    'A2': 'additive', 'A3': 'legacy', 'A4': 'linear', 'A5': 'temp4',
    'A6': 'directional', 'A7': 'rank4', 'A8': 'sham',
}
ARM_NAMES = {
    'F2_ADD': 'A2', 'F5_LEGACY': 'A3', 'F5_LINEAR': 'A4',
    'F5_TEMP4': 'A5', 'F5_DIRECTIONAL': 'A6', 'F5_RANK4': 'A7',
    'F2_CAPACITY_SHAM': 'A8',
}
MODES = tuple(ARM_MODES.values())
DIRECTIONS = ('both', 'off', 'O2P', 'P2O')
CONTROLS = (None, 'matched_query', 'wrong_query', 'mean_query', 'query_mean',
            'interaction_off', 'off', 'O2P', 'P2O', 'otop_only', 'ptoo_only',
            'attention_permutation', 'direct_off', 'whole_omics')


def _require(value: Any, message: str) -> None:
    if not bool(value):
        raise ValueError(message)


def evidence_summaries(output, patch_mask, omics_mask):
    """Return delta and initial summaries; padding contributes exactly zero."""
    pi, pu = output['path_initial'], output['path_updated']
    gi, gu = output['omics_initial'], output['omics_updated']
    _require(patch_mask.shape == pi.shape[:-1] and omics_mask.shape == gi.shape[:-1],
             'summary masks do not match interaction evidence')
    _require(patch_mask.dtype == omics_mask.dtype == torch.bool, 'summary masks must be bool')
    pd = torch.where(patch_mask[..., None], pu-pi, torch.zeros_like(pi))
    gd = torch.where(omics_mask[..., None], gu-gi, torch.zeros_like(gi))
    p0 = torch.where(patch_mask[..., None], pi, torch.zeros_like(pi))
    g0 = torch.where(omics_mask[..., None], gi, torch.zeros_like(gi))
    return dict(summary_path_update=path_mean(pd, patch_mask),
                summary_omics_update=masked_mean(gd, omics_mask, -2),
                summary_path_initial=path_mean(p0, patch_mask),
                summary_omics_initial=masked_mean(g0, omics_mask, -2))


class RiskReadout(nn.Module):
    """Common unimodal part plus a single, explicitly named cross readout.

The returned ``cross_delta`` is the sole quantity for an additional independent
risk-amplitude penalty. The existing complete-risk-set ``extra_weight=.1``
penalty on total delta is managed by the R5 trainer, not changed here.
"""
    def __init__(self, mode, gamma=.1, *, width=32, rank=4, sham_rank=75,
                 directional_lambda=1e-4, extra_init_std=.02):
        super().__init__()
        _require(mode in MODES, 'unknown R5 readout mode')
        _require(math.isfinite(gamma) and gamma > 0 and width > 0 and rank == 4,
                 'R5 requires positive scale/width and the locked rank4')
        _require(math.isfinite(directional_lambda) and directional_lambda >= 0,
                 'directional coefficient regularization must be nonnegative')
        _require(math.isfinite(extra_init_std) and extra_init_std > 0,
                 'extra-channel initialization must be nonzero')
        _require(isinstance(sham_rank, int) and not isinstance(sham_rank, bool) and sham_rank > 0,
                 'positive integer sham rank required')
        self.mode, self.gamma, self.width, self.rank = mode, float(gamma), int(width), int(rank)
        self.sham_rank = sham_rank
        self.directional_lambda, self.extra_init_std = float(directional_lambda), float(extra_init_std)
        if mode == 'directional':
            self.a_o = nn.Parameter(torch.ones(()))
            self.a_p = nn.Parameter(torch.ones(()))
        elif mode == 'rank4':
            self.down = nn.Linear(2*width, rank, bias=False)
            self.up = nn.Linear(rank, 1, bias=False)
            nn.init.normal_(self.down.weight, std=extra_init_std)
            nn.init.zeros_(self.up.weight)
        elif mode == 'sham':
            # At width32, 75 is the nearest integer to (4672-64+260)/65.
            # This matches whole-model active budgets within seven parameters,
            # not head widths or FLOPs; no validation performance chooses it.
            self.down_omics = nn.Linear(width, sham_rank, bias=False)
            self.down_path = nn.Linear(width, sham_rank, bias=False)
            self.up = nn.Linear(sham_rank, 1, bias=False)
            nn.init.normal_(self.down_omics.weight, std=extra_init_std)
            nn.init.normal_(self.down_path.weight, std=extra_init_std)
            nn.init.zeros_(self.up.weight)

    @torch.no_grad()
    def initialize_rank4_from_directional(self, otop_head, ptoo_head):
        _require(self.mode == 'rank4', 'directional initialization applies only to rank4')
        wo, wp = otop_head.weight, ptoo_head.weight
        _require(wo.shape == wp.shape == (1, self.width), 'directional head dimensions differ')
        joined = torch.cat((wo[0], wp[0])).to(self.down.weight)
        self.down.weight[0].copy_(joined)
        self.down.weight[1].copy_(-joined)
        self.up.weight.zero_()
        self.up.weight[0, 0] = 1.
        self.up.weight[0, 1] = -1.

    def coefficient_regularization(self):
        if self.mode == 'directional':
            return self.directional_lambda*((self.a_o-1).square()+(self.a_p-1).square())
        parameter = next(self.parameters(), None)
        return torch.zeros((), device=parameter.device if parameter is not None else None)

    def scalar_response(self, z):
        """One-dimensional curve only when the cross readout is a function of z."""
        if self.mode == 'legacy': return .5*self.gamma*torch.tanh(z)
        if self.mode == 'linear': return .5*self.gamma*z
        if self.mode == 'temp4': return .5*self.gamma*4*torch.tanh(z/4)
        if self.mode == 'additive': return torch.zeros_like(z)
        raise ValueError('directional/rank4/sham have no unique one-dimensional z response')

    def scalar_derivative(self, z):
        if self.mode == 'legacy': return .5*self.gamma*(1-torch.tanh(z).square())
        if self.mode == 'linear': return torch.full_like(z, .5*self.gamma)
        if self.mode == 'temp4': return .5*self.gamma*(1-torch.tanh(z/4).square())
        if self.mode == 'additive': return torch.zeros_like(z)
        raise ValueError('directional/rank4/sham require directional or summary Jacobians')

    def directional_derivatives(self, z):
        """Partial derivatives in the two original scalar risk coordinates."""
        if self.mode == 'directional':
            return (torch.ones_like(z)*(.5*self.gamma*self.a_o),
                    torch.ones_like(z)*(.5*self.gamma*self.a_p))
        if self.mode in ('legacy', 'linear', 'temp4', 'additive'):
            derivative = self.scalar_derivative(z)
            return derivative, derivative
        if self.mode == 'sham': return torch.zeros_like(z), torch.zeros_like(z)
        raise ValueError('rank4 has a summary Jacobian, not a unique scalar-z derivative')

    def summary_jacobian(self, joined):
        """Analytic d(cross risk)/d[omics delta summary;path delta summary]."""
        _require(self.mode == 'rank4' and joined.ndim == 2
                 and joined.shape[-1] == 2*self.width, 'rank4 joined summaries required')
        latent = self.down(joined)
        prime = .5*(1+torch.erf(latent/math.sqrt(2))) + latent*torch.exp(-.5*latent.square())/math.sqrt(2*math.pi)
        return .5*self.gamma*((prime*self.up.weight)@self.down.weight)

    def forward(self, output, patch_mask, omics_mask, *, interaction_off=False):
        p, o = output['pooled_path_risk'], output['pooled_omics_risk']
        zo, zp = output['OtoP_risk'], output['PtoO_risk']
        _require(p.shape == o.shape == zo.shape == zp.shape and p.ndim == 1,
                 'R5 scalar inputs must have the same [B] shape')
        z = zo+zp
        summaries = evidence_summaries(output, patch_mask, omics_mask)
        common = .25*torch.tanh(p)+.25*torch.tanh(o)
        main = self.gamma*common
        zero = torch.zeros_like(p)
        sham = zero
        derivative = None
        if self.mode in ('legacy', 'linear', 'temp4', 'additive'):
            cross = self.scalar_response(z)
            derivative = self.scalar_derivative(z)
        elif self.mode == 'directional':
            cross = .5*self.gamma*(self.a_o*zo+self.a_p*zp)
        elif self.mode == 'rank4':
            joined = torch.cat((summaries['summary_omics_update'], summaries['summary_path_update']), -1)
            cross = .5*self.gamma*self.up(F.gelu(self.down(joined))).squeeze(-1)
        else:
            # Addition after each unimodal nonlinear map is functional additivity.
            sham = .5*self.gamma*(self.up(F.gelu(self.down_omics(summaries['summary_omics_initial']))).squeeze(-1)
                 +self.up(F.gelu(self.down_path(summaries['summary_path_initial']))).squeeze(-1))
            cross = zero
        if interaction_off:
            cross = zero
            if derivative is not None:
                derivative = zero
        if self.mode == 'legacy' and not interaction_off:
            # Preserve original parenthesization for exact old-checkpoint replay.
            delta = self.gamma*(common+.5*torch.tanh(z))
        else:
            delta = main+cross+sham
        return dict(delta=delta, main_delta=main, cross_delta=cross, sham_delta=sham,
                    pool_p=p, pool_o=o, z_o=zo, z_p=zp, z=z,
                    scalar_derivative=derivative, **summaries)


class R5MechanismModel(MechanismModel):
    """F5-compatible wrapper with identical shared construction order for A2--A8.

Call ``refresh_training_query_mean`` before each QUERY_MEAN risk-set update and
before evaluation. It is no-grad and fixed across both gradient-cache passes.
No patients outside ``fit_indices`` contribute to that query or source mask.
"""
    def __init__(self, arm, prepared, membership, patch_bank, device, *, gamma=.1,
                 direction='both', intervention=None, readout_options=None):
        arm = ARM_NAMES.get(arm, arm)
        _require(arm in ARM_MODES and direction in DIRECTIONS, 'unknown R5 arm/direction')
        # Always construct F5 first. New-head RNG consumption cannot alter the
        # shared encoders/kernel/update initialization in another readout arm.
        super().__init__('R3', 'F5', prepared, membership, patch_bank, device,
                         gamma=gamma, intervention=intervention)
        self.r5_arm, self.readout_mode, self.direction = arm, ARM_MODES[arm], direction
        self.readout = RiskReadout(self.readout_mode, gamma=gamma,
                                   width=self.interaction.width, **(readout_options or {}))
        self.readout.to(device)
        self.context_mean_mask = None
        if self.readout_mode == 'rank4':
            self.readout.initialize_rank4_from_directional(self.interaction.otop_head,
                                                          self.interaction.ptoo_head)
        self._configure_trainable_routes()

    def _configure_trainable_routes(self):
        disabled = self.intervention in ('interaction_off', 'off')
        self.route_otop = not disabled and self.r5_arm not in ('A2', 'A8') and self.direction in ('both', 'O2P')
        self.route_ptoo = not disabled and self.r5_arm not in ('A2', 'A8') and self.direction in ('both', 'P2O')
        self.interaction.use_otop, self.interaction.use_ptoo = self.route_otop, self.route_ptoo
        names = []
        if not (self.route_otop or self.route_ptoo): names += ['path_match', 'omics_match']
        if not self.route_otop: names += ['path_value', 'update_omics', 'otop_head']
        if not self.route_ptoo: names += ['omics_value', 'update_path', 'ptoo_head']
        if self.readout_mode == 'rank4': names += ['otop_head', 'ptoo_head']
        for name in set(names): getattr(self.interaction, name).requires_grad_(False)
        if self.readout_mode == 'directional':
            self.readout.a_o.requires_grad_(self.route_otop)
            self.readout.a_p.requires_grad_(self.route_ptoo)
        if not (self.route_otop or self.route_ptoo) and self.readout_mode == 'rank4':
            self.readout.requires_grad_(False)

    @torch.no_grad()
    def refresh_training_query_mean(self, batch=16):
        _require(batch > 0 and len(self.fit_indices) > 0, 'nonempty training partition required')
        sums = counts = None
        for start in range(0, len(self.fit_indices), batch):
            idx = torch.as_tensor(self.fit_indices[start:start+batch], device=self.device)
            out = self.omics(**{k:v[idx] for k,v in self.inputs.items()})
            tokens, valid = out['tokens'], out['token_mask']
            current = torch.where(valid[..., None], tokens, torch.zeros_like(tokens)).sum(0, keepdim=True)
            number = valid.sum(0, keepdim=True)
            sums = current if sums is None else sums+current
            counts = number if counts is None else counts+number
        self.context_mean_mask = counts > 0
        self.context_mean = sums/counts.clamp_min(1)[..., None]
        self.context_mean = self.context_mean*self.context_mean_mask[..., None]
        return self.context_mean, self.context_mean_mask

    def training_query_mean(self, batch=16):
        return self.refresh_training_query_mean(batch)

    @contextmanager
    def _directions(self, control):
        old = self.interaction.use_otop, self.interaction.use_ptoo
        if control in ('O2P', 'otop_only'): self.interaction.use_ptoo = False
        if control in ('P2O', 'ptoo_only'): self.interaction.use_otop = False
        try: yield
        finally: self.interaction.use_otop, self.interaction.use_ptoo = old

    def forward(self, idx, control=None):
        _require(control in CONTROLS and self.intervention in CONTROLS,
                 'unknown R5 query/control intervention')
        if not torch.is_tensor(idx): idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        offset = self.offset[idx]
        path = self._path(idx)
        args = {k:v[idx] for k,v in self.inputs.items()}
        if control == 'whole_omics':
            other = torch.as_tensor(self.query_map[idx.detach().cpu().numpy()], device=self.device)
            args = {k:v[other] for k,v in self.inputs.items()}
            component = torch.as_tensor(self._rna_component, device=self.device)
            offset = offset-component[idx]+component[other]
        omics = self.omics(**args)
        context = None
        effective = control if control is not None else self.intervention
        if effective == 'wrong_query':
            other = torch.as_tensor(self.query_map[idx.detach().cpu().numpy()], device=self.device)
            source = self.omics(**{k:v[other] for k,v in self.inputs.items()})
            context = source['tokens'], source['token_mask']
        elif effective in ('mean_query', 'query_mean'):
            _require(self.context_mean is not None and self.context_mean_mask is not None,
                     'refresh the train-only mean query and its source mask first')
            context = (self.context_mean.expand(len(idx), -1, -1),
                       self.context_mean_mask.expand(len(idx), -1))
        off = effective in ('interaction_off', 'off') or not (self.route_otop or self.route_ptoo)
        with self._directions(effective):
            output = self.interaction(path['patch_states'], path['patch_mask'],
                omics['tokens'], omics['token_mask'], clinical=self.clinical[idx],
                context_omics=context, interaction_off=off,
                permute_attention=effective == 'attention_permutation', return_details=True)
        readout = self.readout(output, path['patch_mask'], omics['token_mask'], interaction_off=off)
        if control == 'direct_off':
            offset = offset-torch.as_tensor(self._rna_component, device=self.device)[idx]
        diagnostics = dict(output.get('diagnostics', {}), r5_arm=self.r5_arm,
                           readout_mode=self.readout_mode, direction=self.direction)
        return dict(risk=offset+readout['delta'], diagnostics=diagnostics,
                    interaction_details=output, **readout)

    def regularization(self):
        # Directional coefficients have an explicit prior around one, not an
        # accidental second zero-centered L2 prior inherited from all parameters.
        excluded = {'readout.a_o', 'readout.a_p'}
        values = [p.square().sum() for n,p in self.named_parameters()
                  if p.requires_grad and n not in excluded]
        penalty = 1e-4*sum(values)
        if self.readout_mode == 'directional': penalty = penalty+self.readout.coefficient_regularization()
        return penalty

    def load_legacy_backbone(self, state_dict):
        """Load strict original F5 keys; initialize new heads only afterwards."""
        expected = {k for k in self.state_dict() if not k.startswith('readout.')}
        _require(set(state_dict) == expected, 'legacy checkpoint keys differ from the F5 backbone')
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        _require(not unexpected and all(k.startswith('readout.') for k in missing),
                 'legacy backbone did not load exactly')
        if self.readout_mode == 'rank4':
            self.readout.initialize_rank4_from_directional(self.interaction.otop_head,
                                                          self.interaction.ptoo_head)
        return dict(backbone_tensors=len(expected), new_readout_tensors=len(missing))

    def parameter_budget(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),
                    requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    added_readout_registered=sum(p.numel() for p in self.readout.parameters()),
                    added_readout_requires_grad=sum(p.numel() for p in self.readout.parameters() if p.requires_grad),
                    legacy_directional_heads_diagnostic_only=self.readout_mode == 'rank4',
                    gradient_active_count_not_inferred_from_requires_grad=True)
