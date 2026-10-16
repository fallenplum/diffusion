"""Pinned official cores with explicit Phase19 input/objective adaptations.

Only official function/class AST definitions are executed, avoiding upstream CLI
imports and unrelated optional dependencies. No upstream source is vendored.
MGCT's batch/mask port is parity-tested; CAMR's one gate-query edit is counted.
All methods emit a scalar contribution; Clinical joint main effects live outside.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MANIFEST = Path(__file__).resolve().parents[2] / 'configs/phase19_baseline_sources.json'


def _manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def _external_root(config=None):
    return Path((config or {}).get('baseline_external_root',
                os.environ.get('PHASE19_BASELINE_EXTERNAL_ROOT', _manifest()['private_external_root'])))


def _hashes(content):
    blob = b'blob ' + str(len(content)).encode() + b'\0' + content
    return {'sha256': hashlib.sha256(content).hexdigest(), 'git_blob': hashlib.sha1(blob).hexdigest()}


def verify_baseline_sources(config=None):
    """Fail closed on missing/changed content, with no source execution or GPU use."""
    manifest, root = _manifest(), _external_root(config)
    count = 0
    for name, spec in manifest['repositories'].items():
        for relative, expected in spec['files'].items():
            path = root / name / relative
            if not path.is_file():
                raise FileNotFoundError('pinned baseline source missing: ' + name + '/' + relative)
            if _hashes(path.read_bytes()) != expected:
                raise ValueError('pinned baseline source changed: ' + name + '/' + relative)
            count += 1
    return {'status': 'PASS', 'files_verified': count,
            'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'commits': {k: v['commit'] for k, v in manifest['repositories'].items()}}


def prepare_baseline_sources(config=None):
    """Copy verified existing private files or fetch only pinned official small PYs."""
    manifest, root = _manifest(), _external_root(config)
    legacy = Path(manifest['verified_legacy_root'])
    copied, fetched, present = 0, 0, 0
    for name, spec in manifest['repositories'].items():
        for relative, expected in spec['files'].items():
            target, previous = root/name/relative, legacy/name/relative
            if target.exists():
                if _hashes(target.read_bytes()) != expected:
                    raise ValueError('refusing to replace changed baseline source')
                present += 1
                continue
            if previous.is_file() and _hashes(previous.read_bytes()) == expected:
                content = previous.read_bytes(); copied += 1
            else:
                repo = spec['repository'].removeprefix('https://github.com/')
                url = 'https://raw.githubusercontent.com/' + repo + '/' + spec['commit'] + '/' + relative
                with urlopen(url, timeout=60) as response:
                    content = response.read()
                fetched += 1
            if _hashes(content) != expected:
                raise ValueError('download/copy does not match official fixed source')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.' + str(os.getpid()) + '.tmp')
            temporary.write_bytes(content); temporary.replace(target)
    return {**verify_baseline_sources(config), 'copied_existing': copied,
            'downloaded_official': fetched, 'already_present': present}


def _namespace():
    from einops import rearrange, reduce
    return {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'Parameter': nn.Parameter,
            'xavier_normal': nn.init.xavier_normal_, 'einsum': torch.einsum,
            'rearrange': rearrange, 'reduce': reduce, '__name__': 'phase19_pinned_official'}


def _definitions(path, names, environment, repair_camr=False):
    """Execute unchanged official definitions, except the one named audited repair."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [copy.deepcopy(n) for n in tree.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if set(n.name for n in nodes) != set(names):
        raise ValueError('required official definition missing')
    if repair_camr:
        edits = 0
        for node in ast.walk(ast.Module(body=nodes, type_ignores=[])):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'z3'):
                call = node.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'linear_z1' and len(call.args) == 2
                        and isinstance(call.args[0], ast.Name) and call.args[0].id == 'x_path_common'):
                    raise ValueError('CAMR third query repair no longer matches pinned AST')
                call.args[0].id = 'x_can_common'; edits += 1
        if edits != 1:
            raise ValueError('CAMR repair must edit exactly one third-query identifier')
    environment['__file__'] = str(path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), environment)
    return environment


def official_definitions(method, config=None, repaired=False):
    """Public audit hook: the actual private source bodies, not local approximations."""
    verify_baseline_sources(config)
    root, env = _external_root(config) / method, _namespace()
    if method == 'MGCT':
        _definitions(root/'models/model_utils.py', ['SNN_Block', 'Attn_Net_Gated'], env)
        _definitions(root/'models/MGCT.py', ['MGCT', 'CrossModalityFusionLayer', 'FeedForward'], env)
    elif method == 'CAMR':
        _definitions(root/'fusion.py', ['LMF', 'Discriminator'], env)
        _definitions(root/'utils.py', ['DiffLoss', 'MSE'], env)
        _definitions(root/'networks.py', ['MisaLMFGatedRec'], env, repair_camr=repaired)
    elif method == 'MMP':
        _definitions(root/'src/mil_models/components.py', ['SNN_Block', 'FeedForward',
                     'FeedForwardEnsemble', 'MMAttention', 'MMAttentionLayer'], env)
        _definitions(root/'src/mil_models/model_multimodal.py', ['init_per_path_model', 'agg_histo',
                     'construct_proto_embedding', 'coattn'], env)
        _definitions(root/'src/mil_models/PANTHER/networks.py', ['mog_eval', 'DirNIWNet'], env)
        _definitions(root/'src/mil_models/PANTHER/layers.py', ['PANTHERBase'], env)
        _definitions(root/'src/mil_models/tokenizer.py', ['PrototypeTokenizer'], env)
    else:
        raise ValueError('unknown official source family')
    return env


def _membership(value):
    result = torch.as_tensor(value, dtype=torch.bool)
    if result.ndim != 2 or result.shape != (8, 512) or not result.any(1).all():
        raise ValueError('expected eight nonempty groups over the same 512 genes')
    return result


def _groups(rows, membership):
    # The same explicit five-channel group input for both patch/token methods.
    values = torch.stack([rows[key] for key in ['R', 'G', 'L', 'RM', 'CM']], -1)
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite native group inputs')
    return [values[:, group].flatten(1) for group in membership]


class _Core(nn.Module):
    def prepare_data(self, ds):
        return {'status': 'NOT_REQUIRED'}


class MGCTCore(_Core):
    def __init__(self, membership, config):
        super().__init__()
        env = official_definitions('MGCT', config)
        self.register_buffer('membership', _membership(membership))
        self.core = env['MGCT'](stage1_num_layers=1, stage2_num_layers=2, n_classes=1,
                                dropout=config.get('training', {}).get('dropout', .1),
                                omic_sizes=[int(g.sum())*5 for g in self.membership], fusion='concat')
        self.core.wsi_net[0] = nn.Linear(1536, 256)

    @staticmethod
    def batch_layer(layer, wsi, omic, wsi_padding=None, omic_padding=None):
        a, _ = layer.wsi_omic_attn(omic, wsi, wsi, key_padding_mask=wsi_padding)
        weights, features = layer.wsi_omic_pool(a.transpose(0, 1))
        weights = weights.transpose(1, 2)
        if omic_padding is not None:
            weights = weights.masked_fill(omic_padding[:, None, :], -torch.inf)
        wa = layer.wsi_rho(layer.wsi_omic_ffn((weights.softmax(-1) @ features).squeeze(1)))
        b, _ = layer.omic_wsi_attn(wsi, omic, omic, key_padding_mask=omic_padding)
        weights, features = layer.omic_wsi_pool(b.transpose(0, 1))
        weights = weights.transpose(1, 2)
        if wsi_padding is not None:
            weights = weights.masked_fill(wsi_padding[:, None, :], -torch.inf)
        wb = layer.omic_rho(layer.omic_wsi_ffn((weights.softmax(-1) @ features).squeeze(1)))
        return layer.fusion_layer(torch.cat([wa, wb], -1))

    def forward(self, visual, mask, f_rows, wsi_indices=None, diagnostic=False):
        if not mask.any(1).all():
            raise ValueError('empty MGCT WSI')
        x = self.core.wsi_net(visual.masked_fill(~mask[..., None], 0)).transpose(0, 1)
        omic = torch.stack([net(v) for net, v in zip(self.core.omic_net, _groups(f_rows, self.membership))])
        # Preserve the fixed public graph: each layer reads the same stage input.
        # Consequently stage2[0] is disconnected from scalar output in native code.
        for layer in self.core.stage1_layers:
            stage1 = self.batch_layer(layer, x, omic, ~mask)
        for layer in self.core.stage2_layers:
            stage2 = self.batch_layer(layer, stage1[None], x, None, ~mask)
        risk = self.core.classifier(self.core.act(self.core.cls_drop(stage2))).reshape(-1)
        return {'risk': risk, 'extra_loss': risk.new_zeros(()),
                'diagnostics': {'native_stage2_disconnected_layers': 1} if diagnostic else {}}


class CAMRCore(_Core):
    def __init__(self, config):
        super().__init__()
        env = official_definitions('CAMR', config, repaired=True)
        self.core = env['MisaLMFGatedRec'](32, 1, hidden_size1=32,
                         dropout=config.get('training', {}).get('dropout', .1))
        self.discriminator = env['Discriminator'](60)
        self.difference, self.reconstruction = env['DiffLoss'](), env['MSE']()
        self.coefficients = _manifest()['CAMR']

    @staticmethod
    def inputs(rows):
        values = (rows['RPCA32'], rows['UNI32'], rows['CNAPCA32'])
        if any(v.ndim != 2 or v.shape[1] != 32 or not torch.isfinite(v).all() for v in values):
            raise ValueError('CAMR requires three train-local PCA32 inputs')
        return values

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    def generator_parameters(self):
        return self.core.parameters()

    def discriminator_loss(self, visual, mask, f_rows, wsi_indices=None):
        with torch.no_grad():
            shared = [self.core.common(x) for x in self.inputs(f_rows)]
        scores = [self.discriminator(x) for x in shared]
        return .5 * (F.binary_cross_entropy(scores[0], torch.ones_like(scores[0]))
                    + sum(F.binary_cross_entropy(s, torch.zeros_like(s)) for s in scores[1:]))

    def forward(self, visual, mask, f_rows, wsi_indices=None, diagnostic=False):
        out = self.core(*self.inputs(f_rows))
        risk, extra = out[0].reshape(-1), out[0].new_zeros(())
        if self.training:
            # D gets a separate minimization step. During G, frozen D still
            # transmits derivatives to the pathology/CNA shared encoders.
            params = list(self.discriminator.parameters())
            flags = [p.requires_grad for p in params]
            try:
                for p in params:
                    p.requires_grad_(False)
                targets = [self.discriminator(out[j]) for j in [8, 9]]
                adversarial = .5 * sum(F.binary_cross_entropy(s, torch.ones_like(s)) for s in targets)
            finally:
                for p, flag in zip(params, flags):
                    p.requires_grad_(flag)
            reconstruction = sum(self.reconstruction(out[i+1], out[i]) for i in [1, 3, 5])
            difference = sum(self.difference(out[i], out[i+3]) for i in [7, 8, 9]) / 3
            l1 = sum(p.abs().sum() for p in self.core.parameters() if p.requires_grad)
            co = self.coefficients
            extra = (co['adversary_weight'] * adversarial + co['reconstruction_weight'] * reconstruction
                     + co['difference_weight'] * difference + co['native_l1'] * l1)
        return {'risk': risk, 'extra_loss': extra, 'diagnostics': {
            'third_query_identifier_repairs': 1, 'separate_discriminator_optimizer_required': True,
            'auxiliary_objective': 'native microbatch means; repaired generator fooling'} if diagnostic else {}}


def panther_tokens(visual, mask, centroids, config=None):
    """Frozen official PANTHER EM+tokenizer; no labels or cross-patient fitting."""
    env, mc = official_definitions('MMP', config), _manifest()['MMP']
    with torch.random.fork_rng(devices=[]):
        panther = env['PANTHERBase'](visual.shape[-1], p=len(centroids), L=mc['EM_iterations'],
                 tau=mc['tau'], out='allcat', ot_eps=mc['prior_covariance_scale'],
                 load_proto=False, fix_proto=True).to(visual.device)
    with torch.no_grad():
        panther.priors.m.copy_(torch.as_tensor(centroids, device=visual.device))
        flat, _ = panther(visual.masked_fill(~mask[..., None], 0), mask=mask.to(visual.dtype))
        prob, mean, covariance = env['PrototypeTokenizer']('PANTHER', 'allcat', len(centroids))(flat)
        tokens = torch.cat([prob[..., None], mean, covariance], -1)
    if not torch.isfinite(tokens).all() or not (covariance > 0).all():
        raise ValueError('official PANTHER generated invalid distribution parameters')
    return tokens


class MMPCore(_Core):
    def __init__(self, membership, config):
        super().__init__()
        env, mc = official_definitions('MMP', config), _manifest()['MMP']
        self.config = config
        self.register_buffer('membership', _membership(membership))
        self.core = env['coattn'](omic_sizes=[int(g.sum())*5 for g in self.membership], histo_in_dim=1536,
                    dropout=config.get('training', {}).get('dropout', .1), num_classes=1,
                    path_proj_dim=mc['coattention_dimension'], num_coattn_layers=1, modality='both',
                    histo_agg='mean', histo_model='PANTHER', append_embed='none', mult=1,
                    net_indiv=False, numOfproto=mc['prototypes'])
        self.register_buffer('prototype_features', torch.empty(0), persistent=False)
        self._wsi_to_cache = {}
        self.preparation_audit = None

    def prepare_data(self, ds):
        from sklearn.cluster import KMeans
        from .data import _cached_bank
        mc = _manifest()['MMP']
        if ds.visual.shape[-1] != 1536 or ds.mask.shape != ds.visual.shape[:2]:
            raise ValueError('MMP requires matching frozen UNI2 bags and masks')
        train_slides = np.flatnonzero(np.isin(np.asarray(ds.owner), ds.fit))
        if not len(train_slides):
            raise ValueError('no fit-patient patches for MMP prototype prior')
        fit_patch_mask = ds.mask[train_slides].detach().cpu().numpy()
        eligible = np.argwhere(fit_patch_mask)
        if len(eligible) < mc['prototypes']:
            raise ValueError('too few training-only patches for sixteen prototypes')
        selected = np.random.default_rng(mc['kmeans_seed']).choice(
                    len(eligible), min(mc['initial_pool_max'], len(eligible)), replace=False)
        pairs = eligible[selected]
        sample = ds.visual[train_slides[pairs[:, 0]], pairs[:, 1]].detach().cpu().numpy().astype(np.float32)
        def ah(x):
            a = np.ascontiguousarray(x)
            return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
        provenance = {'manifest': verify_baseline_sources(self.config)['manifest_sha256'],
                      'fold': ds.fold, 'role': ds.role, 'fit_hash': ah(ds.fit),
                      'owner_hash': ah(ds.owner), 'mask_hash': ah(ds.mask.detach().cpu().numpy()),
                      'global_wsi_hash': ah(ds.global_wsi_indices), 'sample_hash': ah(sample),
                      'preprocess_audit': {k: v for k, v in ds.audit.items()
                          if k not in ['wrong_controls', 'wrong_clinical_rule', 'wrong_omics_rule']},
                      'parameters': mc}
        def json_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        # Parent public audits contain tuple-valued feature blocks. Normalize the
        # private provenance before both serialization and receipt comparisons.
        provenance = json.loads(json.dumps(provenance, sort_keys=True, default=json_value))
        signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        private = Path(self.config.get('private_root', '/root/autodl-tmp/phase19_attention_private'))
        path = private / 'baseline_features' / f'MMP_{ds.role}_fold{ds.fold}_{signature[:16]}.pt'
        def produce():
            centers = KMeans(n_clusters=mc['prototypes'], n_init=mc['kmeans_n_init'],
                        max_iter=mc['kmeans_max_iter'], random_state=mc['kmeans_seed'],
                        algorithm='lloyd').fit(sample).cluster_centers_.astype(np.float32)
            tokens = []
            for start in range(0, len(ds.visual), 8):
                tokens.append(panther_tokens(ds.visual[start:start+8], ds.mask[start:start+8],
                                            centers, self.config).cpu().numpy())
            return {'provenance': provenance, 'centroids': centers, 'tokens': np.concatenate(tokens),
                    'global_wsi_indices': np.asarray(ds.global_wsi_indices, dtype=np.int64)}
        saved, cache_sha = _cached_bank(path, provenance, produce)
        self.prototype_features = torch.as_tensor(saved['tokens'], device=ds.visual.device)
        self._wsi_to_cache = {int(v): j for j, v in enumerate(saved['global_wsi_indices'])}
        self.preparation_audit = {'status': 'PASS', 'fit_patient_count': len(ds.fit),
                    'fit_WSI_count': len(train_slides), 'fit_patch_pool_count': len(sample),
                    'prototype_count': mc['prototypes'], 'WSI_count': len(ds.visual),
                    'source_manifest_sha256': provenance['manifest'], 'cache_signature': signature,
                    'cache_sha256': cache_sha, 'labels_used': False,
                    'test_EM': 'per WSI only; frozen fit-patient centroid prior',
                    'private_cache': True}
        return self.preparation_audit

    def forward(self, visual, mask, f_rows, wsi_indices=None, diagnostic=False):
        if wsi_indices is None or not self._wsi_to_cache:
            raise RuntimeError('MMP needs prepared training-only prototypes and explicit global WSI indices')
        indices = torch.as_tensor(wsi_indices).detach().cpu().tolist()
        try:
            positions = [self._wsi_to_cache[int(i)] for i in indices]
        except KeyError as exc:
            raise ValueError('unknown WSI index in prototype cache') from exc
        if len(positions) != len(visual):
            raise ValueError('MMP WSI cache indexing mismatch')
        tokens = self.prototype_features[positions].to(visual.device)
        result = self.core.forward_no_loss(tokens, _groups(f_rows, self.membership))
        risk = result['logits'].reshape(-1)
        return {'risk': risk, 'extra_loss': risk.new_zeros(()),
                'diagnostics': {'native_prototype_count': 16, 'native_coattention_dimension': 256} if diagnostic else {}}


def build_baseline(name, clinical_dim, membership, config):
    """Cox+Clinical adaptations only: no claim of native input/endpoint reproduction."""
    if clinical_dim < 1:
        raise ValueError('Clinical main effects require legal structured input')
    if name == 'MGCT':
        return MGCTCore(membership, config)
    if name == 'CAMR_REPAIRED':
        return CAMRCore(config)
    if name == 'MMP':
        return MMPCore(membership, config)
    raise ValueError('unsupported native method adaptation: ' + str(name))
