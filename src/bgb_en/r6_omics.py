"""Sparse source-program gene evidence, with no survival or R5 dependency.

The 8-D symbol hash records gene identity, not medical language semantics. All
measurements must already have the caller's training-only scale. A gene bank
retains original per-gene values/masks; each program read or reread recomputes
K/V from those values. A pooled program tensor is not an accepted replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


PROGRAM_ORDER = ('iCMS2', 'iCMS3', 'FAP_fibroblast_associated_top25',
                 'MARCO_macrophage_associated_top25')
SOURCE_PROGRAM_SHA256 = 'f8031c03cbb9ec2409d72c799129a6de74f080860afbc81cb30fb187d2560143'


def require(value: Any, message: str):
    if not bool(value): raise ValueError(message)


def symbol_hash(symbols: Sequence[str]):
    return hashlib.sha256(json.dumps(list(symbols), separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def newline_symbol_hash(symbols: Sequence[str]):
    """Legacy canonical export's order digest; distinct from JSON-list hash."""
    return hashlib.sha256('\n'.join(symbols).encode()).hexdigest()


def checked_symbols(symbols, name):
    values = tuple(map(str, symbols))
    require(bool(values) and len(set(values)) == len(values)
            and all(value and value == value.strip() for value in values),
            name+'_REQUIRES_UNIQUE_NONBLANK_EXACT_SYMBOLS')
    return values


def frozen_symbol_identity(symbols):
    rows = []
    for symbol in symbols:
        raw = hashlib.sha256(('BG-BEN-R6/gene-id/v1/'+symbol).encode()).digest()
        row = [2*(int.from_bytes(raw[j:j+4], 'little')+.5)/2**32-1 for j in range(0, 32, 4)]
        norm = math.sqrt(sum(x*x for x in row)); rows.append([x/norm for x in row])
    return torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class SignedProgramMapping:
    program_names: tuple[str, ...]
    gene_symbols: tuple[str, ...]
    edge_program: torch.Tensor
    edge_gene: torch.Tensor
    edge_sign: torch.Tensor
    definition_sha256: str | None = None

    def __post_init__(self):
        names = checked_symbols(self.program_names, 'PROGRAM')
        genes = checked_symbols(self.gene_symbols, 'PROGRAM_GENE')
        p, g = self.edge_program.detach().cpu(), self.edge_gene.detach().cpu()
        s = self.edge_sign.detach().cpu()
        require(p.dtype == g.dtype == torch.long and p.ndim == g.ndim == s.ndim == 1
                and len(p) == len(g) == len(s) and len(p) > 0, 'BAD_SPARSE_SIGNED_EDGE_SHAPES')
        require(((p >= 0) & (p < len(names))).all() and ((g >= 0) & (g < len(genes))).all()
                and ((s == 1) | (s == -1)).all(), 'BAD_SPARSE_SIGNED_EDGE_ID_OR_SIGN')
        pairs = list(zip(p.tolist(), g.tolist()))
        require(len(set(pairs)) == len(pairs), 'DUPLICATE_OR_CONFLICTING_PROGRAM_GENE_EDGE')
        require(set(p.tolist()) == set(range(len(names))), 'EMPTY_PROGRAM_DEFINITION')
        # Canonical symbol order makes reindexing a vocabulary a true rename.
        order = sorted(range(len(p)), key=lambda j:(int(p[j]), genes[int(g[j])]))
        index = torch.tensor(order, dtype=torch.long)
        object.__setattr__(self, 'program_names', names); object.__setattr__(self, 'gene_symbols', genes)
        object.__setattr__(self, 'edge_program', p[index].clone())
        object.__setattr__(self, 'edge_gene', g[index].clone())
        object.__setattr__(self, 'edge_sign', s[index].float().clone())

    def reindex_genes(self, permutation):
        permutation = checked_permutation(permutation, len(self.gene_symbols), 'cpu')
        inverse = torch.argsort(permutation)
        return SignedProgramMapping(self.program_names, tuple(self.gene_symbols[i] for i in permutation.tolist()),
            self.edge_program, inverse[self.edge_gene], self.edge_sign, self.definition_sha256)

    def public_audit(self, canonical_gene_symbols=None):
        available = set(self.gene_symbols if canonical_gene_symbols is None
                        else checked_symbols(canonical_gene_symbols, 'CANONICAL_GENE'))
        rows = []
        for k, name in enumerate(self.program_names):
            row = dict(program=name)
            for sign, label in ((1, 'up'), (-1, 'down')):
                edge = (self.edge_program == k) & (self.edge_sign == sign)
                symbols = [self.gene_symbols[g] for g in self.edge_gene[edge].tolist()]
                row[label+'_defined'] = len(symbols)
                row[label+'_in_canonical_vocabulary'] = sum(g in available for g in symbols)
            rows.append(row)
        missing = sorted(set(self.gene_symbols)-available)
        return dict(programs=list(self.program_names), program_count=len(self.program_names),
            union_genes=len(self.gene_symbols), signed_edges=len(self.edge_program),
            union_gene_order_sha256=symbol_hash(self.gene_symbols),
            program_definition_sha256=self.definition_sha256,
            canonical_gene_order_sha256=symbol_hash(tuple(canonical_gene_symbols))
                if canonical_gene_symbols is not None else None,
            canonical_gene_order_newline_sha256=newline_symbol_hash(tuple(canonical_gene_symbols))
                if canonical_gene_symbols is not None else None,
            gene_order_hash_definition='compact_ASCII_JSON_list; legacy_newline_digest_separate',
            canonical_covered_union=len(self.gene_symbols)-len(missing),
            absent_symbol_count=len(missing), absent_symbols=missing, per_program=rows,
            sign_semantics='up_and_down_normalized_separately_then_up_minus_down_not_log_negative_sign',
            Hallmark_tokens_claimed_ST_aligned=False, outcome_labels_used=False)


def mapping_from_programs(programs, *, definition_sha256=None):
    require(bool(programs), 'NO_SIGNED_PROGRAMS')
    names = tuple(p['id'] for p in programs)
    for program in programs:
        up, down = list(program['up_genes']), list(program['down_genes'])
        require(len(set(up+down)) == len(up+down), 'DUPLICATE_OR_CONFLICTING_PROGRAM_MEMBER')
        if 'genes' in program:
            require(list(program['genes']) == up+down, 'PROGRAM_GENES_NOT_EXACT_UP_PLUS_DOWN')
    genes = checked_symbols(sorted({g for p in programs for g in p['up_genes']+p['down_genes']}), 'PROGRAM_GENE')
    lookup = {g:i for i,g in enumerate(genes)}; ep=[]; eg=[]; es=[]
    for k, program in enumerate(programs):
        for field, sign in (('up_genes', 1), ('down_genes', -1)):
            for gene in program[field]:ep.append(k);eg.append(lookup[gene]);es.append(sign)
    return SignedProgramMapping(names, genes, torch.tensor(ep, dtype=torch.long),
        torch.tensor(eg, dtype=torch.long), torch.tensor(es), definition_sha256)


def load_source_program_mapping(path, *, expected_sha256=SOURCE_PROGRAM_SHA256):
    raw = Path(path).read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    require(digest == expected_sha256, 'SOURCE_PROGRAM_DEFINITION_SHA_CHANGED')
    result = mapping_from_programs(json.loads(raw.decode('utf-8-sig'))['programs'], definition_sha256=digest)
    require(result.program_names == PROGRAM_ORDER and len(result.gene_symbols) == 760
            and len(result.edge_program) == 765, 'FIXED_FOUR_SOURCE_PROGRAMS_CHANGED')
    counts = result.public_audit()['per_program']
    require([r['up_defined'] for r in counts] == [308, 74, 25, 25]
            and [r['down_defined'] for r in counts] == [279, 54, 0, 0], 'SOURCE_SIGNED_GROUP_SIZES_CHANGED')
    return result


def checked_permutation(ids, size, device):
    value = torch.as_tensor(ids, device=device)
    require(value.dtype == torch.long and value.shape == (size,)
            and torch.equal(value.sort().values, torch.arange(size, device=device)),
            'GENE_IDENTITIES_MUST_BE_A_COMPLETE_COLUMN_BIJECTION')
    return value


@dataclass(frozen=True)
class GeneMeasurements:
    values: torch.Tensor
    measured_mask: torch.Tensor
    identity_features: torch.Tensor
    gene_symbols: tuple[str, ...]
    gene_ids: torch.Tensor
    cna_enabled: bool = False

    def validate(self):
        require(self.values.ndim == 3 and self.measured_mask.shape == self.values.shape
                and self.measured_mask.dtype == torch.bool and self.values.is_floating_point(),
                'GENE_MEASUREMENTS_REQUIRE_B_U_CHANNEL_VALUES_AND_BOOL_MASK')
        b, u, c = self.values.shape
        require(b > 0 and c == (3 if self.cna_enabled else 1) and len(self.gene_symbols) == u
                and len(set(self.gene_symbols)) == u, 'GENE_MEASUREMENTS_NOT_RAW_GENE_CHANNELS')
        require(self.identity_features.shape == (u, 8)
                and self.identity_features.device == self.values.device == self.measured_mask.device
                and self.identity_features.dtype == self.values.dtype,
                'GENE_IDENTITY_FEATURE_SHAPE_DTYPE_OR_DEVICE_CHANGED')
        checked_permutation(self.gene_ids, u, self.values.device)
        require(torch.isfinite(self.values[self.measured_mask]).all()
                and torch.isfinite(self.identity_features).all(), 'NONFINITE_MEASURED_GENE_EVIDENCE')
        return self

    def reindex(self, permutation):
        order = checked_permutation(permutation, self.values.shape[1], self.values.device)
        return GeneMeasurements(self.values[:, order], self.measured_mask[:, order], self.identity_features[order],
            tuple(self.gene_symbols[j] for j in order.cpu().tolist()), self.gene_ids[order], self.cna_enabled)


class GeneValueBank(nn.Module):
    """Gather this patient's original measurements; has zero trainable weights."""
    def __init__(self, rna_gene_symbols, union_gene_symbols, *, enable_cna=False, cna_gene_symbols=None):
        super().__init__()
        self.rna_gene_symbols = checked_symbols(rna_gene_symbols, 'RNA')
        self.gene_symbols = checked_symbols(union_gene_symbols, 'UNION')
        self.enable_cna = bool(enable_cna)
        require(self.enable_cna or cna_gene_symbols is None, 'CNA_DISABLED_BUT_VOCABULARY_SUPPLIED')
        self.cna_gene_symbols = checked_symbols(cna_gene_symbols, 'CNA') if self.enable_cna else None
        for prefix, source in (('rna', self.rna_gene_symbols), ('cna', self.cna_gene_symbols)):
            if source is None:continue
            lookup = {g:i for i,g in enumerate(source)}
            present = torch.tensor([g in lookup for g in self.gene_symbols], dtype=torch.bool)
            positions = torch.tensor([lookup.get(g, 0) for g in self.gene_symbols], dtype=torch.long)
            self.register_buffer('_'+prefix+'_positions', positions, persistent=False)
            self.register_buffer('_'+prefix+'_present', present, persistent=False)
        self.register_buffer('_identity', frozen_symbol_identity(self.gene_symbols), persistent=False)

    @staticmethod
    def _gather(values, mask, positions, present, vocabulary_size, gene_ids):
        require(values.ndim == 2 and values.shape == mask.shape and values.shape[1] == vocabulary_size
                and values.shape[0] > 0 and values.is_floating_point() and mask.dtype == torch.bool
                and values.device == mask.device, 'RAW_GENE_MATRIX_OR_MEASUREMENT_MASK_CHANGED')
        positions, present = positions.to(values.device), present.to(values.device)
        if gene_ids is not None:
            ids = checked_permutation(gene_ids, vocabulary_size, values.device)
            positions = torch.argsort(ids)[positions]
        observed = mask[:, positions]&present[None]
        gathered = values[:, positions]
        require(torch.isfinite(gathered[observed]).all(), 'NONFINITE_OBSERVED_GENE_MEASUREMENT')
        return torch.where(observed, gathered, torch.zeros_like(gathered)), observed

    def forward(self, rna, rna_mask, *, gene_ids=None, cna_gain=None, cna_loss=None,
                cna_gain_mask=None, cna_loss_mask=None, cna_gene_ids=None):
        r, mr = self._gather(rna, rna_mask, self._rna_positions, self._rna_present,
                            len(self.rna_gene_symbols), gene_ids)
        values, masks = [r], [mr]
        cna = (cna_gain, cna_loss, cna_gain_mask, cna_loss_mask, cna_gene_ids)
        if not self.enable_cna:
            require(all(x is None for x in cna), 'CNA_DISABLED_BUT_VALUES_OR_MASKS_SUPPLIED')
        else:
            require(all(x is not None for x in cna[:4]), 'CNA_GAIN_LOSS_NEED_SEPARATE_EXPLICIT_MASKS')
            for x, m in ((cna_gain, cna_gain_mask), (cna_loss, cna_loss_mask)):
                require(x.device == rna.device and x.dtype == rna.dtype and len(x) == len(rna),
                        'CNA_RNA_BATCH_DTYPE_OR_DEVICE_DIFFERS')
                v, observed = self._gather(x, m, self._cna_positions, self._cna_present,
                                          len(self.cna_gene_symbols), cna_gene_ids)
                values.append(v); masks.append(observed)
        return GeneMeasurements(torch.stack(values, -1), torch.stack(masks, -1), self._identity.to(rna),
            self.gene_symbols, torch.arange(len(self.gene_symbols), device=rna.device), self.enable_cna).validate()

    def public_audit(self):
        result = dict(union_genes=len(self.gene_symbols), union_gene_order_sha256=symbol_hash(self.gene_symbols),
            RNA_canonical_genes=len(self.rna_gene_symbols), RNA_gene_order_sha256=symbol_hash(self.rna_gene_symbols),
            RNA_gene_order_newline_sha256=newline_symbol_hash(self.rna_gene_symbols),
            RNA_union_in_vocabulary=int(self._rna_present.sum()), RNA_union_absent=int((~self._rna_present).sum()),
            trainable_parameters=0, identity='frozen_symbol_SHA256_8D_not_medical_semantics',
            input_is_original_gene_matrix_not_pooled_token=True, CNA_enabled=self.enable_cna,
            gene_order_hash_definition='compact_ASCII_JSON_list; legacy_newline_digest_separate',
            scale_fit_here=False, patients_or_outcomes_read=False)
        if self.enable_cna:
            result.update(CNA_canonical_genes=len(self.cna_gene_symbols), CNA_gene_order_sha256=symbol_hash(self.cna_gene_symbols),
                CNA_gene_order_newline_sha256=newline_symbol_hash(self.cna_gene_symbols),
                CNA_union_in_vocabulary=int(self._cna_present.sum()), CNA_union_absent=int((~self._cna_present).sum()),
                gain_loss_masks_separate=True)
        return result


class GeneEvidenceTransform(nn.Module):
    """Shared original-value/identity transform: RNA8832, RNA+CNA9088 weights."""
    def __init__(self, *, width=64, enable_cna=False):
        super().__init__(); require(width == 64, 'R6_LOCKED_GENE_WIDTH_IS_64')
        self.width, self.enable_cna = int(width), bool(enable_cna)
        self.channels = 3 if self.enable_cna else 1
        self.value_encoder = nn.Linear(2*self.channels+8, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, batch):
        require(isinstance(batch, GeneMeasurements), 'ORIGINAL_GENE_MEASUREMENTS_REQUIRED_NOT_POOLED_TOKENS')
        batch.validate()
        require(batch.cna_enabled == self.enable_cna, 'CNA_ENCODER_NOT_EXPLICITLY_ENABLED')
        require(batch.values.dtype == self.value_encoder.weight.dtype
                and batch.values.device == self.value_encoder.weight.device, 'GENE_ENCODER_DTYPE_OR_DEVICE_CHANGED')
        valid = batch.measured_mask.any(-1)
        safe = torch.where(batch.measured_mask, batch.values, torch.zeros_like(batch.values))
        identity = batch.identity_features[None].expand(len(safe), -1, -1)
        raw = torch.cat((safe, batch.measured_mask.to(safe.dtype), identity), -1)
        state = F.gelu(self.value_encoder(raw))*valid[..., None]
        return dict(keys=self.key(state)*valid[..., None], values=self.value(state)*valid[..., None],
                    gene_mask=valid, state=state)


def masked_softmax(scores, valid):
    present = valid.any(-1, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~valid, -torch.inf), torch.zeros_like(scores))
    return torch.softmax(safe, -1)*valid


class _SparseGeneRead(nn.Module):
    def __init__(self, mapping, gene_encoder, width):
        super().__init__(); require(isinstance(mapping, SignedProgramMapping), 'SIGNED_PROGRAM_MAPPING_REQUIRED')
        require(width == 64, 'R6_LOCKED_READ_WIDTH_IS_64')
        self.mapping = mapping; self.width = width
        self.gene_encoder = gene_encoder if gene_encoder is not None else GeneEvidenceTransform(width=width)
        require(self.gene_encoder.width == width, 'SHARED_GENE_ENCODER_WIDTH_DIFFERS')
        self.register_buffer('edge_program', mapping.edge_program.clone())
        self.register_buffer('edge_gene', mapping.edge_gene.clone())
        self.register_buffer('edge_sign', mapping.edge_sign.clone())

    def _read(self, query, batch, query_valid, topk):
        require(isinstance(batch, GeneMeasurements) and batch.gene_symbols == self.mapping.gene_symbols,
                'GENE_VOCABULARY_ORDER_DOES_NOT_MATCH_PROGRAM_EDGES')
        require(query.shape == (len(batch.values), len(self.mapping.program_names), self.width)
                and query_valid.shape == query.shape[:-1] and query_valid.dtype == torch.bool,
                'PROGRAM_QUERY_OR_MASK_SHAPE_CHANGED')
        require(query.device == batch.values.device == query_valid.device and torch.isfinite(query[query_valid]).all(),
                'PROGRAM_QUERY_DEVICE_OR_FINITE_GUARD')
        require(topk is None or isinstance(topk, int) and not isinstance(topk, bool) and topk > 0,
                'TOPK_MUST_BE_EXPLICIT_POSITIVE_OR_ALL_MEMBERS')
        encoded = self.gene_encoder(batch)  # Recomputed from original values on every read.
        ep, eg, sign = self.edge_program, self.edge_gene, self.edge_sign
        edge_keys, edge_values = encoded['keys'][:, eg], encoded['values'][:, eg]
        safe_query = torch.where(query_valid[..., None], query, torch.zeros_like(query))
        scores = (safe_query[:, ep]*edge_keys).sum(-1)/math.sqrt(self.width)
        measured = encoded['gene_mask'][:, eg]
        valid = measured&query_valid[:, ep]
        retained = valid.clone()
        if topk is not None:
            retained.zero_()
            for k in range(len(self.mapping.program_names)):
                columns = torch.where(ep == k)[0]; keep = min(topk, len(columns))
                chosen = scores[:, columns].masked_fill(~valid[:, columns], -torch.inf).topk(keep, -1).indices
                selected = torch.zeros_like(valid[:, columns]).scatter_(-1, chosen, True)&valid[:, columns]
                retained[:, columns] = selected
        weights = scores.new_zeros(scores.shape)
        components=[]; component_masks=[]; coverage=[]; measured_counts=[]; retained_counts=[]; defined=[]
        for k in range(len(self.mapping.program_names)):
            states=[]; masks=[]; cov=[]; counts=[]; kept=[]; definition=[]
            for s in (1, -1):
                columns = torch.where((ep == k)&(sign == s))[0]
                definition.append(bool(len(columns)))
                if not len(columns):
                    states.append(edge_values.new_zeros(len(query), self.width))
                    masks.append(torch.zeros(len(query), device=query.device, dtype=torch.bool))
                    counts.append(torch.zeros(len(query), device=query.device, dtype=torch.long)); kept.append(counts[-1])
                    cov.append(scores.new_zeros(len(query))); continue
                probability = masked_softmax(scores[:, columns], retained[:, columns])
                weights[:, columns] = probability
                states.append((probability[..., None]*edge_values[:, columns]).sum(1))
                masks.append(retained[:, columns].any(-1)); counts.append(measured[:, columns].sum(-1))
                kept.append(retained[:, columns].sum(-1)); cov.append(counts[-1].to(scores.dtype)/len(columns))
            components.append(torch.stack(states, 1)); component_masks.append(torch.stack(masks, 1))
            coverage.append(torch.stack(cov, 1)); measured_counts.append(torch.stack(counts, 1))
            retained_counts.append(torch.stack(kept, 1)); defined.append(definition)
        components = torch.stack(components, 1); component_masks = torch.stack(component_masks, 1)
        result = components[:, :, 0]-components[:, :, 1]
        return dict(tokens=result, program_mask=component_masks.any(-1), component_mask=component_masks,
            up_tokens=components[:, :, 0], down_tokens=components[:, :, 1],
            coverage=torch.stack(coverage, 1), measured_counts=torch.stack(measured_counts, 1),
            retained_counts=torch.stack(retained_counts, 1),
            component_defined=torch.tensor(defined, device=query.device, dtype=torch.bool),
            weights=weights, signed_weights=weights*sign[None], query=query,
            all_members_used=topk is None, topk_per_program_before_signed_normalization=topk,
            sparse_membership_edges=len(ep), dense_gene_gene_attention_used=False,
            original_gene_values_reencoded_this_call=True)


class SparseProgramEncoder(_SparseGeneRead):
    """Independent molecular tokens; accepts no image, tissue or outcome input."""
    def __init__(self, mapping, *, gene_encoder=None, width=64):
        super().__init__(mapping, gene_encoder, width)
        self.queries = nn.Parameter(torch.randn(len(mapping.program_names), width)*.02)

    def forward(self, batch, *, topk=None):
        require(isinstance(batch, GeneMeasurements), 'ORIGINAL_GENE_MEASUREMENTS_REQUIRED_NOT_POOLED_TOKENS')
        query = self.queries[None].expand(len(batch.values), -1, -1)
        valid = torch.ones(query.shape[:-1], device=query.device, dtype=torch.bool)
        return self._read(query, batch, valid, topk)


class OriginalGeneRereader(_SparseGeneRead):
    """Path evidence changes gene reads; own measured K/V never get replaced."""
    def __init__(self, mapping, *, gene_encoder=None, width=64):
        super().__init__(mapping, gene_encoder, width)
        self.query_projection = nn.Linear(width, width, bias=False)

    def forward(self, evidence, batch, *, evidence_mask=None, query_context=None, query_mask=None, topk=None):
        require(isinstance(batch, GeneMeasurements), 'ORIGINAL_GENE_MEASUREMENTS_REQUIRED_NOT_POOLED_TOKENS')
        expected = (len(batch.values), len(self.mapping.program_names), self.width)
        require(evidence.shape == expected and evidence.device == batch.values.device,
                'PATH_EVIDENCE_SHAPE_OR_DEVICE_CHANGED')
        if evidence_mask is None:
            evidence_mask = torch.ones(expected[:-1], device=evidence.device, dtype=torch.bool)
        require(evidence_mask.shape == expected[:-1] and evidence_mask.dtype == torch.bool
                and evidence_mask.device == evidence.device, 'PATH_EVIDENCE_MASK_CHANGED')
        require((query_context is None) == (query_mask is None), 'QUERY_ONLY_CONTEXT_REQUIRES_SOURCE_MASK')
        context, valid = evidence, evidence_mask
        if query_context is not None:
            require(query_context.shape == expected and query_context.device == evidence.device
                    and query_mask.shape == expected[:-1] and query_mask.dtype == torch.bool
                    and query_mask.device == evidence.device, 'QUERY_SOURCE_SHAPE_OR_MASK_CHANGED')
            context, valid = query_context, evidence_mask&query_mask
        safe = torch.where(valid[..., None], context, torch.zeros_like(context))
        return self._read(self.query_projection(safe), batch, valid, topk)
