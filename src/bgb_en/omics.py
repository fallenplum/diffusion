"""Sparse, mask-aware molecular tokens for the BG-BEN inner experiments.

No data preprocessing or outcome-dependent gene selection occurs here. Inputs
must have been aligned and scaled by the current training partition. The direct
RNA512 Cox risk is deliberately managed by the experiment trainer, outside this
encoder. A learnable pathway token is not a measured cellular program.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(scores, valid, dim=-1):
    valid = torch.broadcast_to(valid.bool(), scores.shape)
    present = valid.any(dim, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~valid, -torch.inf), torch.zeros_like(scores))
    return torch.softmax(safe, dim=dim).masked_fill(~valid | ~present, 0.)


def load_gmt_membership(path, gene_symbols, *, expected_sha256=None):
    """Exact-symbol external annotation; never rank/filter using cohort labels.

    Pathways remain in the external file order, including empty measured
    support. A symbol may belong to multiple pathways. Nonunique expression
    symbols must be resolved upstream under a documented measurement contract.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('GMT source SHA256 differs from the protocol lock')
    symbols = list(map(str, gene_symbols))
    if not symbols or len(symbols) != len(set(symbols)) or any(not x or x.strip() != x for x in symbols):
        raise ValueError('gene symbols must be nonblank, unique, canonical exact symbols')
    location = {s: n for n, s in enumerate(symbols)}
    names, rows, sources, counts = [], [], [], []
    for line in raw.decode('utf-8-sig').splitlines():
        fields = line.split('\t')
        if len(fields) < 3 or not fields[0] or fields[0] in names:
            raise ValueError('GMT requires unique pathway names and at least one annotated gene')
        genes = set(fields[2:])
        if '' in genes:
            raise ValueError('GMT includes an empty gene symbol')
        row = torch.zeros(len(symbols), dtype=torch.bool)
        for gene in genes & location.keys():
            row[location[gene]] = True
        names.append(fields[0]); sources.append(fields[1]); rows.append(row); counts.append(len(genes))
    if not rows:
        raise ValueError('GMT has no pathways')
    membership = torch.stack(rows)
    return membership, {'pathways': names, 'annotation_sources': sources, 'sha256': digest,
                        'external_gene_counts': counts, 'mapped_gene_counts': membership.sum(1).tolist(),
                        'gene_order_sha256': hashlib.sha256('\n'.join(symbols).encode()).hexdigest(),
                        'symbol_mapping': 'exact_symbol_no_outcome_selection'}


def shuffled_membership(membership, *, seed):
    """Column permutation breaks gene identity while preserving all row sizes.

    It also preserves pairwise pathway intersections and each column's degree
    multiset. This is one locked null annotation, not a new permutation per fit.
    """
    member = torch.as_tensor(membership).bool().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(member.shape[1], generator=generator)
    if member.shape[1] > 1 and torch.equal(permutation, torch.arange(member.shape[1])):
        permutation = permutation.roll(1)
    return member[:, permutation], permutation


class SparseProgramPool(nn.Module):
    """Multiple independently weighted gene patterns per pathway, O(B E M).

    E is the number of annotated gene--pathway edges, M the retained patterns.
    No dense G by G attention or G by K trainable matrix is constructed. The
    first pattern starts as a uniform sum, others as signed random projections;
    all edge weights can learn from the training objective. This is a lossy
    structured projection, not an assertion of information preservation.
    """
    def __init__(self, membership, *, width=32, patterns=4):
        super().__init__()
        member = torch.as_tensor(membership).bool()
        if member.ndim != 2 or min(member.shape) < 1 or patterns < 2:
            raise ValueError('membership must be K by G and patterns must be at least two')
        self.pathways, self.genes = member.shape
        self.patterns, self.width = int(patterns), int(width)
        edge_path, edge_gene = torch.where(member)
        self.register_buffer('edge_path', edge_path)
        self.register_buffer('edge_gene', edge_gene)
        self.register_buffer('membership_count', member.sum(1))
        initial = torch.randn(len(edge_gene), patterns)
        initial[:, 0] = 1.
        self.edge_weight = nn.Parameter(initial)
        self.project = nn.Linear(patterns + 1, width)
        self.identity = nn.Parameter(torch.randn(self.pathways, width) * .02)

    def forward(self, values, measured):
        if values.ndim != 2 or values.shape != measured.shape or values.shape[1] != self.genes:
            raise ValueError('measurements and mask must have shape B by canonical G')
        mask = measured.bool()
        if not torch.isfinite(values[mask]).all():
            raise ValueError('observed molecular values must be finite')
        # Masked nonfinite values cannot enter arithmetic, including 0 * NaN.
        safe_values = values.masked_fill(~mask, 0.)
        observed = mask[:, self.edge_gene]
        count = values.new_zeros(len(values), self.pathways)
        count.index_add_(1, self.edge_path, observed.to(values.dtype))
        contribution = safe_values[:, self.edge_gene, None] * self.edge_weight[None]
        numerator = values.new_zeros(len(values), self.pathways, self.patterns)
        numerator.index_add_(1, self.edge_path, contribution)
        norm_sq = values.new_zeros(len(values), self.pathways, self.patterns)
        norm_sq.index_add_(1, self.edge_path,
                          observed[..., None].to(values.dtype) * self.edge_weight.square()[None])
        patterns = numerator / norm_sq.clamp_min(1e-12).sqrt()
        coverage = count / self.membership_count.clamp_min(1).to(values.dtype)[None]
        valid = count > 0
        tokens = self.project(torch.cat((patterns, coverage[..., None]), -1)) + self.identity[None]
        tokens = tokens * valid[..., None]
        return tokens, valid, coverage, patterns


class SetAttention(nn.Module):
    """Permutation-equivariant token attention, excluding invalid query/key rows."""
    def __init__(self, width=32, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('attention width must be divisible by heads')
        self.heads, self.width = int(heads), int(width)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, tokens, valid):
        batch, length, width = tokens.shape
        h, d = self.heads, width // self.heads
        q, k, v = self.qkv(self.norm(tokens)).reshape(batch, length, 3, h, d).unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        attention = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(d), valid[:, None, None])
        message = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        hidden = (tokens + self.output(message)) * valid[..., None]
        return (hidden + self.ff(self.ff_norm(hidden))) * valid[..., None]


class DiagonalSelectiveSSM(nn.Module):
    """Explicit real diagonal input-dependent SSM; NOT an official Mamba block.

    h_t = exp(-softplus(a) * dt_t) h_(t-1) + dt_t * B_t * u_t,
    y_t = C_t * h_t + D * u_t; dt=softplus(dt_projection(x)).
    State dimension is one per channel. Invalid tokens do not update the state.
    This branch provides a named reproducible SSM control when requested; it
    never silently substitutes for the separate official-Mamba backend.
    """
    def __init__(self, width=32):
        super().__init__()
        self.width = int(width)
        self.in_projection = nn.Linear(width, 4 * width)
        self.a = nn.Parameter(torch.zeros(width))
        self.skip = nn.Parameter(torch.ones(width))
        self.out_projection = nn.Linear(width, width)

    def forward(self, sequence, valid):
        u, dt, b, c = self.in_projection(sequence).chunk(4, -1)
        dt = F.softplus(dt - 2.)
        state = sequence.new_zeros(len(sequence), self.width)
        output = []
        for step in range(sequence.shape[1]):
            updated = torch.exp(-F.softplus(self.a) * dt[:, step]) * state
            updated = updated + dt[:, step] * torch.tanh(b[:, step]) * u[:, step]
            state = torch.where(valid[:, step, None], updated, state)
            y = torch.tanh(c[:, step]) * state + self.skip * u[:, step]
            output.append(self.out_projection(y) * valid[:, step, None])
        return torch.stack(output, 1)


class MultiOrderSequence(nn.Module):
    """Fixed multi-order averaging with inverse placement into canonical slots.

    Orders are enumeration, reverse, and two seeded permutations. These are
    order-robustness controls, not a claim that Hallmark has a biological axis.
    The official backend raises if unavailable and cannot consume padding.
    """
    def __init__(self, pathways, width=32, *, backend='ssm', seed=901, orders=4):
        super().__init__()
        if orders not in (1, 2, 4):
            raise ValueError('orders must be one, two, or four')
        base = torch.arange(pathways)
        generator = torch.Generator().manual_seed(int(seed))
        candidates = [base, base.flip(0), torch.randperm(pathways, generator=generator),
                      torch.randperm(pathways, generator=generator)]
        self.register_buffer('orders', torch.stack(candidates[:orders]))
        self.norm = nn.LayerNorm(width)
        self.backend = backend
        if backend == 'ssm':
            self.scan = DiagonalSelectiveSSM(width)
        elif backend == 'mamba':
            try:
                from mamba_ssm import Mamba
            except (ImportError, OSError) as exc:
                raise RuntimeError('BLOCKED_OFFICIAL_MAMBA_DEPENDENCY: no SSM fallback is used') from exc
            self.scan = Mamba(d_model=width, d_state=8, d_conv=4, expand=1)
        else:
            raise ValueError('backend must be ssm or mamba')

    def forward(self, tokens, valid):
        normalized = self.norm(tokens)
        messages = []
        for order in self.orders:
            x, m = normalized[:, order], valid[:, order]
            if self.backend == 'mamba':
                # Compact equal-length patient sets into one scan call. Each
                # patient retains its own relative valid-token order and its
                # own recurrent state; no padding enters convolution or SSM.
                out = torch.zeros_like(x)
                lengths = m.sum(1)
                for length in torch.unique(lengths).tolist():
                    if not length:
                        continue
                    patients = torch.where(lengths == length)[0]
                    # nonzero is row-major, so reshaping retains the current
                    # scan order separately for every patient's valid slots.
                    positions = torch.nonzero(m[patients], as_tuple=True)[1].reshape(len(patients), length)
                    observed = self.scan(x[patients[:, None], positions])
                    out[patients[:, None], positions] = observed
            else:
                out = self.scan(x, m)
            messages.append(out[:, torch.argsort(order)])
        return (tokens + torch.stack(messages).mean(0)) * valid[..., None]


class SparsePathwayEncoder(nn.Module):
    """RNA and optional separate CNA gain/loss into K pathway tokens.

    Modes: pool, set, ssm, set_ssm, cna_gate, factors. `factors` exposes two,
    four or eight supervised readout coordinates. Their biological meanings are
    not prespecified. The independent RNA512 risk is kept outside this class.
    """
    def __init__(self, membership, *, mode='pool', width=32, patterns=4,
                 factors=4, sequence_backend='ssm', order_seed=901, orders=4):
        super().__init__()
        if mode not in ('pool', 'set', 'ssm', 'set_ssm', 'cna_gate', 'factors'):
            raise ValueError('unsupported molecular mode')
        if width != 32:
            raise ValueError('BG-BEN molecular output width is locked to 32')
        if factors not in (2, 4, 8):
            raise ValueError('supervised low-rank factors must be 2, 4 or 8')
        self.mode, self.width = mode, width
        self.rna = SparseProgramPool(membership, width=width, patterns=patterns)
        if mode == 'cna_gate':
            self.gain = SparseProgramPool(membership, width=width, patterns=patterns)
            self.loss = SparseProgramPool(membership, width=width, patterns=patterns)
            self.cna_project = nn.Linear(2 * width, width)
            self.cna_gate = nn.Linear(3 * width, width)
        self.set_attention = SetAttention(width) if mode in ('set', 'set_ssm') else None
        self.sequence = MultiOrderSequence(self.rna.pathways, width, backend=sequence_backend,
                                           seed=order_seed, orders=orders) if mode in ('ssm', 'set_ssm') else None
        self.pool_query = nn.Parameter(torch.randn(width) * .02)
        # Nonlinearity is explicit: Linear(32,r) followed immediately by a
        # linear scalar head would collapse algebraically to the same linear
        # scalar model for r=2,4,8. These are jointly supervised latent risk
        # coordinates, not independently identified biological factors.
        self.factor = nn.Sequential(nn.Linear(width, factors, bias=False), nn.GELU()) if mode == 'factors' else None
        self.risk_head = nn.Linear(factors if mode == 'factors' else width, 1, bias=False)

    def forward(self, rna, rna_mask, cna_gain=None, cna_loss=None, cna_mask=None, *, gene_ids=None):
        if self.mode == 'cna_gate' and (cna_gain is None or cna_loss is None or cna_mask is None):
            raise ValueError('CNA-gated mode requires separate gain/loss and an explicit measurement mask')
        if self.mode == 'cna_gate' and (cna_gain.shape != rna.shape or cna_loss.shape != rna.shape
                                        or cna_mask.shape != rna.shape):
            raise ValueError('CNA gain/loss and measurement mask must align with canonical RNA G')
        if gene_ids is not None:
            if gene_ids.shape != (self.rna.genes,) or gene_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('gene_ids must be the jointly permuted canonical G index set')
            canonical = torch.arange(self.rna.genes, device=gene_ids.device)
            if not torch.equal(torch.sort(gene_ids).values, canonical):
                raise ValueError('gene IDs must be a complete, unique canonical index set')
            order = torch.argsort(gene_ids)
            rna, rna_mask = rna[:, order], rna_mask[:, order]
            if cna_gain is not None:
                cna_gain, cna_loss, cna_mask = cna_gain[:, order], cna_loss[:, order], cna_mask[:, order]
        tokens, valid, coverage, patterns = self.rna(rna, rna_mask)
        cna_coverage = None
        if self.mode == 'cna_gate':
            # Gain/loss were split before fold-local scaling; a centered gain
            # channel may legitimately be negative. Splitting provenance is
            # checked upstream, not guessed from a scaled tensor's sign.
            gain, gvalid, gcov, _ = self.gain(cna_gain, cna_mask)
            loss, lvalid, lcov, _ = self.loss(cna_loss, cna_mask)
            cna_valid = gvalid | lvalid
            cna_message = self.cna_project(torch.cat((gain, loss), -1)) * cna_valid[..., None]
            gate = torch.sigmoid(self.cna_gate(torch.cat((tokens, gain, loss), -1)))
            tokens = tokens + gate * cna_message
            valid = valid | cna_valid
            cna_coverage = torch.stack((gcov, lcov), -1)
        if self.set_attention is not None:
            tokens = self.set_attention(tokens, valid)
        if self.sequence is not None:
            tokens = self.sequence(tokens, valid)
        pool_weight = masked_softmax(tokens @ self.pool_query / math.sqrt(self.width), valid)
        embedding = (pool_weight[..., None] * tokens).sum(1)
        factor = self.factor(embedding) if self.factor is not None else None
        risk = self.risk_head(factor if factor is not None else embedding).squeeze(-1)
        return {'tokens': tokens, 'token_mask': valid, 'embedding': embedding, 'risk': risk,
                'coverage': coverage, 'cna_coverage': cna_coverage, 'rna_patterns': patterns,
                'factors': factor, 'pool_weights': pool_weight}

    def parameter_audit(self):
        return {'mode': self.mode, 'parameters': sum(p.numel() for p in self.parameters()),
                'annotated_edges': self.rna.edge_gene.numel(), 'patterns_per_program': self.rna.patterns,
                'output_width': self.width, 'pathways': self.rna.pathways,
                'dense_gene_attention': False,
                'sequence_backend': self.sequence.backend if self.sequence is not None else None,
                'sequence_orders': self.sequence.orders.cpu().tolist() if self.sequence is not None else None,
                'direct_rna_risk_is_external': True}
