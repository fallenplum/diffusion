"""Outcome-independent tensor operators and the no-semantic core candidate.

Cross-modal values are used only to change queries in the re-reading arms.
The ordinary DIRECT control deliberately uses cross-modal feature messages.
No attention dropout, clinical query input, semantic bridge or auxiliary task.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

ARMS = ('C00', 'C10', 'C01', 'C11', 'DIRECT')


def masked_softmax(score, mask, dim=-1):
    """Exactly zero empty rows, without a softmax over an all-minus-inf row."""
    mask = torch.broadcast_to(mask.to(device=score.device, dtype=torch.bool), score.shape)
    present = mask.any(dim=dim, keepdim=True)
    safe = torch.where(present, score.masked_fill(~mask, -torch.inf), torch.zeros_like(score))
    probability = torch.softmax(safe, dim=dim)
    return torch.where(mask & present, probability, torch.zeros_like(probability))


def valid_mean(value, mask, dim=0):
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum(dim) / mask.sum(dim).clamp_min(1.)


class EvidenceRead(nn.Module):
    def __init__(self, input_dim, width=128, read_width=32):
        super().__init__()
        self.query = nn.Linear(width, read_width, bias=False)
        self.key = nn.Linear(input_dim, read_width, bias=False)
        self.value = nn.Linear(input_dim, width, bias=False)
        self.scale = math.sqrt(read_width)

    def prepare(self, evidence):
        return self.key(evidence), self.value(evidence)

    def forward(self, query, prepared, support, *, constant_value=False):
        key, value = prepared
        weight = masked_softmax(self.query(query) @ key.T / self.scale, support)
        if constant_value:
            # Same per-query value at every permitted evidence position. The
            # original branch anchor is untouched; only the two difference
            # reads use this control. Query changes cannot alter this mean.
            mean = support.to(value.dtype) @ value / support.sum(-1, keepdim=True).clamp_min(1)
            output = mean * support.any(-1, keepdim=True)
        else:
            output = weight @ value
        return output, weight


class GeneEvidence(nn.Module):
    def __init__(self, genes, identity_width=8, gene_width=64):
        super().__init__()
        self.identity = nn.Embedding(genes, identity_width)
        nn.init.normal_(self.identity.weight, std=.02)
        self.encode = nn.Sequential(nn.Linear(identity_width + 6, gene_width), nn.GELU(), nn.LayerNorm(gene_width))

    def forward(self, values, mask, gene_ids):
        if values.ndim != 2 or values.shape[-1] != 3 or mask.shape != values.shape:
            raise ValueError('one patient requires G by RNA/gain/loss plus measurement masks')
        if not torch.isfinite(values).all():
            raise ValueError('gene values must be fit-scaled and finitely imputed before encoding')
        if gene_ids.shape != (len(values),) or len(values) != self.identity.num_embeddings or not torch.equal(
                torch.sort(gene_ids).values, torch.arange(len(values), device=gene_ids.device)):
            raise ValueError('gene identities must be a jointly permuted complete canonical index set')
        observed = mask.bool().any(-1)
        joined = torch.cat((values * mask, mask.to(values.dtype), self.identity(gene_ids)), -1)
        evidence = self.encode(joined) * observed[:, None]
        return evidence, observed


class SetBlock(nn.Module):
    """One permutation-equivariant masked set-attention block, not gene scanning."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('set-attention dimensions must divide')
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, tokens, valid):
        k, h, d = len(tokens), self.heads, self.width // self.heads
        q, key, value = self.qkv(self.norm1(tokens)).reshape(k, 3, h, d).unbind(1)
        q, key, value = (x.transpose(0, 1) for x in (q, key, value))
        weight = masked_softmax(q @ key.transpose(-1, -2) / math.sqrt(d), valid[None, None, :])
        message = (weight @ value).transpose(0, 1).reshape(k, self.width)
        hidden = (tokens + self.output(message)) * valid[:, None]
        return (hidden + self.ff(self.norm2(hidden))) * valid[:, None]


class SharedRouting(nn.Module):
    """Same content score for both directions; axis-specific zero-value null."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.path_key = nn.Linear(width, width, bias=False)
        self.gene_key = nn.Linear(width, width, bias=False)
        self.path_value = nn.Linear(width, width, bias=False)
        self.gene_value = nn.Linear(width, width, bias=False)
        self.null_path = nn.Linear(width, heads, bias=True)
        self.null_gene = nn.Linear(width, heads, bias=True)
        nn.init.zeros_(self.null_path.weight); nn.init.zeros_(self.null_path.bias)
        nn.init.zeros_(self.null_gene.weight); nn.init.zeros_(self.null_gene.bias)

    def forward(self, path, gene, path_valid, gene_valid):
        nr, nk, h, d = len(path), len(gene), self.heads, self.width // self.heads
        pk = self.path_key(path).reshape(nr, h, d).transpose(0, 1)
        gk = self.gene_key(gene).reshape(nk, h, d).transpose(0, 1)
        score = pk @ gk.transpose(-1, -2) / math.sqrt(d)
        np_ = self.null_path(path).T.unsqueeze(-1)
        ng = self.null_gene(gene).T.unsqueeze(-1)
        # Null is always permitted. Invalid target rows are subsequently zero.
        ap = torch.softmax(torch.cat((score.masked_fill(~gene_valid[None, None, :], -torch.inf), np_), -1), -1)
        ag = torch.softmax(torch.cat((score.transpose(-1, -2).masked_fill(~path_valid[None, None, :], -torch.inf), ng), -1), -1)
        gv = self.gene_value(gene).reshape(nk, h, d).transpose(0, 1)
        pv = self.path_value(path).reshape(nr, h, d).transpose(0, 1)
        mp = (ap[..., :-1] @ gv).transpose(0, 1).reshape(nr, self.width) * path_valid[:, None]
        mg = (ag[..., :-1] @ pv).transpose(0, 1).reshape(nk, self.width) * gene_valid[:, None]
        return mp, mg, {'score': score, 'path_probability': ap, 'gene_probability': ag}


class SlotLinear(nn.Module):
    """Fixed slot identity, low-rank coefficient table, no nonlinear risk head."""
    def __init__(self, slots, width=128, rank=4, zero_output=False):
        super().__init__()
        self.slot = nn.Parameter(torch.empty(slots, rank))
        self.axis = nn.Parameter(torch.empty(rank, width))
        nn.init.normal_(self.slot, std=.02)
        if zero_output:
            nn.init.zeros_(self.axis)  # The nonzero slot factor gives first-step head gradients.
        else:
            nn.init.normal_(self.axis, std=.02)

    def forward(self, values, valid):
        coefficient = self.slot @ self.axis
        if len(coefficient) != len(values):
            raise ValueError('slot risk table does not match token identity')
        return valid_mean((values * coefficient).sum(-1), valid)


class PathLinear(nn.Module):
    """Region permutation invariance, with distinct weights for the two queries."""
    def __init__(self, width=128, zero_output=False):
        super().__init__()
        self.coefficient = nn.Parameter(torch.empty(2, width))
        if zero_output:
            nn.init.zeros_(self.coefficient)
        else:
            nn.init.normal_(self.coefficient, std=.02)

    def forward(self, values, valid):
        if len(values) % 2:
            raise ValueError('path tokens must contain two queries for each region')
        scores = (values.reshape(-1, 2, values.shape[-1]) * self.coefficient).sum(-1)
        return valid_mean(scores.flatten(), valid)


class UpdateGate(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.state_norm, self.message_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.project = nn.Linear(2 * width, width)
        nn.init.zeros_(self.project.bias)

    def forward(self, state, message):
        return torch.sigmoid(self.project(torch.cat((self.state_norm(state), self.message_norm(message)), -1)))


class BidirectionalReread(nn.Module):
    def __init__(self, genes, membership, spatial_encoder, *, arm='C11', path_dim=1536,
                 width=128, gene_width=64, rank=4, rounds=2, eta=.1):
        super().__init__()
        if arm not in ARMS or membership.shape != (108, genes) or width != 128 or rounds != 2 or eta != .1:
            raise ValueError('outside Phase28 starting tensor/operator contract')
        self.arm, self.width, self.rounds, self.eta = arm, width, rounds, eta
        self.register_buffer('membership', torch.as_tensor(membership, dtype=torch.bool))
        self.gene_evidence = GeneEvidence(genes, gene_width=gene_width)
        self.path_project = nn.Sequential(nn.LayerNorm(path_dim), nn.Linear(path_dim, width, bias=False), nn.LayerNorm(width))
        self.spatial = spatial_encoder
        self.path_read, self.gene_read = EvidenceRead(width, width), EvidenceRead(gene_width, width)
        self.path_query = nn.Parameter(torch.randn(2, width) * .02)
        self.gene_query = nn.Parameter(torch.randn(108, width) * .02)
        self.set_block = SetBlock(width)
        self.routing = SharedRouting(width)
        self.up = nn.Linear(width, width, bias=False)
        self.ug = nn.Linear(width, width, bias=False)
        self.delta_path = nn.Linear(width, width, bias=False)
        self.delta_gene = nn.Linear(width, width, bias=False)
        for module in (self.up, self.ug, self.delta_path, self.delta_gene):
            nn.init.normal_(module.weight, std=.02)
        self.gate_path, self.gate_gene = UpdateGate(width), UpdateGate(width)
        self.main_path, self.main_gene = PathLinear(width, True), SlotLinear(108, width, rank, True)
        self.rho_path, self.rho_gene = PathLinear(width), SlotLinear(108, width, rank)

    def encode(self, values, mask, gene_ids, slides):
        eg, observed = self.gene_evidence(values, mask, gene_ids)
        support = self.membership[:, gene_ids] & observed[None, :]
        gene_valid = support.any(-1)
        g0raw, gene_weight = self.gene_read(self.gene_query, self.gene_read.prepare(eg), support)
        g0 = self.set_block(g0raw, gene_valid)
        path_evidence, path_support, path_valid, path_prepared, p0, queries = [], [], [], [], [], []
        for slide in slides:
            ep = self.spatial(self.path_project(slide['features']), slide['plan'])
            regions = slide['plan']['region_indices']
            region_owner = np.full(len(ep), -1, dtype=np.int64)
            for index, members in enumerate(regions):
                if (region_owner[np.asarray(members)] != -1).any():
                    raise ValueError('overlapping path evidence regions')
                region_owner[np.asarray(members)] = index
            if (region_owner < 0).any():
                raise ValueError('region evidence does not cover selected actual patches')
            owner = torch.as_tensor(region_owner, device=ep.device)
            omega = (torch.arange(len(regions) * 2, device=ep.device) // 2)[:, None] == owner[None, :]
            q = self.path_query.repeat(len(regions), 1)
            prepared = self.path_read.prepare(ep)
            base, _ = self.path_read(q, prepared, omega)
            path_evidence.append(ep); path_support.append(omega); path_valid.append(omega.any(-1))
            path_prepared.append(prepared); p0.append(base); queries.append(q)
        if not p0:
            raise ValueError('a patient requires at least one real WSI')
        return {'gene_evidence': eg, 'gene_support': support, 'gene_valid': gene_valid,
                'gene_prepared': self.gene_read.prepare(eg), 'gene0': g0, 'gene_base_read': g0raw,
                'gene_initial_weight': gene_weight, 'path_evidence': path_evidence,
                'path_support': path_support, 'path_valid': path_valid,
                'path_prepared': path_prepared, 'path0': p0, 'path_query': queries}

    def read_encoded(self, encoded, offset, *, context=None, interaction_off=False,
                     constant_path_value=False, constant_gene_value=False, details=False):
        p0, g0 = encoded['path0'], encoded['gene0']
        ps, gs = p0, g0
        use_p, use_g = self.arm in ('C10', 'C11', 'DIRECT'), self.arm in ('C01', 'C11', 'DIRECT')
        trace, states = [], []
        for layer in range(self.rounds):
            states.append({'path': ps, 'gene': gs})
            # Matched mode computes each shared graph once for BOTH directions.
            own = [self.routing(p, gs, v, encoded['gene_valid']) for p, v in zip(ps, encoded['path_valid'])]
            path_routes, gene_routes = own, own
            mp = [row[0] for row in own]
            mg = torch.stack([row[1] for row in own]).mean(0)  # Equal WSI mean, not a slide-count sum.
            if context is not None:
                current = context[layer]
                if current.get('gene') is not None:
                    path_routes = [self.routing(p, current['gene'], v, current['gene_valid'])
                                   for p, v in zip(ps, encoded['path_valid'])]
                    mp = [row[0] for row in path_routes]
                if current.get('path') is not None:
                    if not current['path']:
                        raise ValueError('context needs real slides')
                    gene_routes = [self.routing(p, gs, v, encoded['gene_valid'])
                                   for p, v in zip(current['path'], current['path_valid'])]
                    mg = torch.stack([row[1] for row in gene_routes]).mean(0)
            if interaction_off:
                mp, mg = [torch.zeros_like(p) for p in mp], torch.zeros_like(mg)
            next_p, path_rows = [], []
            for index, (state, message) in enumerate(zip(ps, mp)):
                shifted = encoded['path_query'][index] + self.up(message)
                conditional, wc = self.path_read(shifted, encoded['path_prepared'][index], encoded['path_support'][index], constant_value=constant_path_value)
                base, wb = self.path_read(encoded['path_query'][index], encoded['path_prepared'][index], encoded['path_support'][index], constant_value=constant_path_value)
                difference = self.up(message) if self.arm == 'DIRECT' else conditional - base
                gate = self.gate_path(state, message)
                change = self.eta * gate * torch.tanh(self.delta_path(difference))
                change = change * encoded['path_valid'][index][:, None] if use_p else torch.zeros_like(change)
                next_p.append(p0[index] + change)
                if details:
                    path_rows.append({'difference': difference, 'change': change, 'gate': gate,
                                      'base_weight': wb, 'conditional_weight': wc, 'message': message})
            conditional_g, wgc = self.gene_read(self.gene_query + self.ug(mg), encoded['gene_prepared'], encoded['gene_support'], constant_value=constant_gene_value)
            base_g, wgb = self.gene_read(self.gene_query, encoded['gene_prepared'], encoded['gene_support'], constant_value=constant_gene_value)
            dg = self.ug(mg) if self.arm == 'DIRECT' else conditional_g - base_g
            gg = self.gate_gene(gs, mg)
            change_g = self.eta * gg * torch.tanh(self.delta_gene(dg))
            change_g = change_g * encoded['gene_valid'][:, None] if use_g else torch.zeros_like(change_g)
            next_g = g0 + change_g
            if details:
                trace.append({'path': path_rows, 'gene_difference': dg, 'gene_change': change_g,
                              'gene_gate': gg, 'gene_base_weight': wgb, 'gene_conditional_weight': wgc,
                              'gene_message': mg, 'own_routing_reference': [row[2] for row in own],
                              'actual_path_routing': [row[2] for row in path_routes],
                              'actual_gene_routing': [row[2] for row in gene_routes]})
            ps, gs = next_p, next_g  # Synchronous update from the same old layer states.
        path_main = torch.stack([self.main_path(p, v) for p, v in zip(p0, encoded['path_valid'])]).mean()
        gene_main = self.main_gene(g0, encoded['gene_valid'])
        path_interaction = torch.stack([self.rho_path(p, v) - self.rho_path(base, v)
                                        for p, base, v in zip(ps, p0, encoded['path_valid'])]).mean()
        gene_interaction = self.rho_gene(gs, encoded['gene_valid']) - self.rho_gene(g0, encoded['gene_valid'])
        offset = torch.as_tensor(offset, device=g0.device, dtype=torch.float64)
        learned_main = (path_main + gene_main).double()
        interaction = (path_interaction + gene_interaction).double()
        output = {'risk': offset + learned_main + interaction, 'base': offset + learned_main,
                  'offset': offset, 'learned_path_main': path_main, 'learned_gene_main': gene_main,
                  'interaction': interaction, 'path_interaction': path_interaction,
                  'gene_interaction': gene_interaction, 'path_final': ps, 'gene_final': gs,
                  'states': states, 'trace': trace}
        if not torch.isfinite(output['risk']):
            raise ValueError('nonfinite total risk')
        return output

    def forward(self, values, mask, gene_ids, slides, offset=0., **options):
        return self.read_encoded(self.encode(values, mask, gene_ids, slides), offset, **options)

    def parameter_counts(self):
        return {'allocated_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'arm': self.arm, 'dormant_direction_modules_possible': self.arm != 'C11',
                'readout': sum(p.numel() for module in (self.main_path, self.main_gene, self.rho_path, self.rho_gene) for p in module.parameters())}
