"""Phase6B-R: true token-level TITAN prompt conditioning.

S0 static TITAN concepts; S1 static learnable token prompt; S2 RNA-conditioned
token prompt; S3 static concepts plus RNA additive; S4 matched S2 under a
wrong-patient RNA permutation.  TITAN image/text parameters are frozen.  The
continuous prompt pool is inserted after BOS and is passed through the frozen
TITAN text transformer before concept scores are computed.

This script intentionally keeps endpoint labels restricted to T/N/M/Stage and
fits RNA transforms and all trainable prompt/head parameters inside outer
training splits, using a fixed schedule with no inner selection. It depends on the official local TITAN snapshot and is
therefore meant for the authorized GPU server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from phase6b_rna_conditioned_titan_concepts import (
    ENDPOINTS,
    fit_numeric,
    load_rna_only,
    normalize_label,
    sha256,
    unit,
)

TOKEN_METHODS = ("S0_static_concepts", "S1_static_token_prompt", "S2_rna_token_prompt", "S3_static_concepts_plus_rna")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_titan(model_path: Path, device: str):
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast
    from unittest.mock import patch

    # Transformers 4.46 discovers second-level relative imports through its
    # dynamic module cache.  Populate that cache with the exact verified local
    # source files, as in the Phase6A official loader.
    module_root = Path(os.environ.get("HF_MODULES_CACHE", str(Path.home() / ".cache/huggingface/modules")))
    package = module_root / "transformers_modules" / model_path.name
    package.mkdir(parents=True, exist_ok=True)
    for parent in (module_root, module_root / "transformers_modules", package):
        (parent / "__init__.py").touch(exist_ok=True)
    for source in model_path.glob("*.py"):
        shutil.copyfile(source, package / source.name)

    original = PreTrainedTokenizerFast.from_pretrained

    def local_tokenizer(name, *args, **kwargs):
        if name == "MahmoodLab/TITAN":
            name = str(model_path)
            kwargs["local_files_only"] = True
        return original(name, *args, **kwargs)

    with patch.object(PreTrainedTokenizerFast, "from_pretrained", side_effect=local_tokenizer):
        model, loading = AutoModel.from_pretrained(str(model_path), trust_remote_code=True,
                                                   local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"nonexact TITAN load: {loading}")
    model.eval().requires_grad_(False)
    model.vision_encoder.to("cpu")
    model.text_encoder.to(device)
    return model, loading


def text_tokens(model, concepts_path: Path):
    vocab = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts = vocab["concepts"]
    templates = vocab["templates"]
    if len(concepts) != 32 or any("CLASSNAME" not in x for x in templates):
        raise ValueError("Phase6A vocabulary lock changed")
    texts = [template.replace("CLASSNAME", c["text"]) for c in concepts for template in templates]
    ids = model.text_encoder.tokenizer(texts)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids, len(templates), vocab


def official_text_prototypes(model, vocab, device: str) -> np.ndarray:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prototypes = model.zero_shot_classifier([[c["text"]] for c in vocab["concepts"]], vocab["templates"], device=device)
    return unit(prototypes.float().detach().cpu().numpy().T)


class FrozenTextProjector:
    """Run the frozen TITAN text transformer with continuous BOS-adjacent prompts."""

    def __init__(self, model, input_ids, template_count: int, prompt_len: int, device: str):
        import torch

        self.text = model.text_encoder
        self.device = device
        self.prompt_len = prompt_len
        self.template_count = template_count
        ids = input_ids.to(device) if hasattr(input_ids, "to") else torch.as_tensor(input_ids, device=device)
        if ids.ndim != 2 or ids.shape[0] != 64:
            raise ValueError(f"unexpected tokenized concept shape: {tuple(ids.shape)}")
        self.ids = ids[:, :-1].contiguous()
        self.base_embeddings = self.text.token_embedding(self.ids).detach()
        if self.base_embeddings.shape[1] <= prompt_len + 2:
            raise ValueError("prompt length leaves no concept tokens")
        if prompt_len and (self.ids[:, -prompt_len:] != self.text.pad_id).any():
            raise ValueError("prompt insertion would truncate nonpadding concept tokens")
        # The CLS padding mask must describe the INSERTED sequence. Retaining
        # the original mask would hide the last L shifted concept/EOS tokens.
        sentinel = 1 if self.text.pad_id != 1 else 2
        if prompt_len:
            prompt_ids = torch.full((len(self.ids), prompt_len), sentinel,
                                    dtype=self.ids.dtype, device=device)
            self.mask_ids = torch.cat([self.ids[:, :1], prompt_ids,
                                       self.ids[:, 1:-prompt_len]], dim=1)
        else:
            self.mask_ids = self.ids
        self.compact = True
        positions = torch.arange(self.mask_ids.shape[1], device=device)
        # Globally trailing padding has no path to valid concept tokens under
        # causal attention and is masked from CLS. Keep CLS at its ORIGINAL
        # learned position, rather than moving its positional embedding.
        self.prefix_len = int(positions[(self.mask_ids != self.text.pad_id).any(0)].max().item()) + 1

    def encode(self, prompts):
        """prompts: [batch, prompt_len, 768]; returns [batch,32,768]."""
        import torch

        batch = prompts.shape[0]
        n_text = self.base_embeddings.shape[0]
        base = self.base_embeddings.unsqueeze(0).expand(batch, -1, -1, -1)
        prompt = prompts.unsqueeze(1).expand(-1, n_text, -1, -1)
        # Keep the original BOS token at position 0, insert continuous tokens,
        # then retain the original concept sequence.  The final token is
        # already removed because official encode_text reserves it for CLS.
        tail = base[:, :, 1:-self.prompt_len, :] if self.prompt_len else base[:, :, 1:, :]
        embedded = torch.cat([base[:, :, :1, :], prompt, tail], dim=2)
        flat = embedded.reshape(batch * n_text, embedded.shape[2], embedded.shape[3])
        text = self.text
        mask_ids = self.mask_ids
        if self.compact:
            flat = flat[:, :self.prefix_len]
            mask_ids = mask_ids[:, :self.prefix_len]
            position_ids = torch.cat([torch.arange(self.prefix_len, device=self.device),
                                      torch.tensor([self.ids.shape[1]], device=self.device)])
        else:
            position_ids = torch.arange(self.ids.shape[1] + 1, device=self.device)
        cast_dtype = text.transformer.get_cast_dtype()
        # Match the official TITAN fp16 inference regime while retaining
        # gradients through the prompt tensors.  All encoder parameters remain
        # frozen; autocast only changes the temporary arithmetic dtype.
        with torch.autocast("cuda", dtype=torch.float16):
            x = flat.to(cast_dtype)
            seq_len = x.shape[1] + 1
            x = torch.cat([x, text.cls_emb.to(cast_dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)], dim=1)
            attn_mask = text.attn_mask
            cls_mask = text.build_cls_mask(mask_ids.repeat(batch, 1), cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[position_ids][:, position_ids][None] + cls_mask[:, :seq_len, :seq_len]
            x = x + text.positional_embedding[position_ids].to(cast_dtype)
            x = text.transformer(x, attn_mask=attn_mask)
            pooled = text.ln_final(x[:, -1])
            if text.text_projection is not None:
                if isinstance(text.text_projection, torch.nn.Linear):
                    pooled = text.text_projection(pooled)
                else:
                    pooled = pooled @ text.text_projection
        # Official zero_shot_classifier normalizes EACH template before
        # averaging; normalize the concept ensemble a second time afterwards.
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        pooled = pooled.reshape(batch, 32, self.template_count, -1).mean(dim=2)
        return torch.nn.functional.normalize(pooled, dim=-1)


class PromptNet:
    def __init__(self, method: str, n_classes: int, model_dim: int, prompt_len: int,
                 pool_size: int, seed: int, device: str):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.method = method
                if method in {"S1_static_token_prompt", "S2_rna_token_prompt"}:
                    self.pool = nn.Parameter(torch.randn(pool_size, prompt_len, model_dim) * 0.01)
                if method == "S1_static_token_prompt":
                    self.pool_logits = nn.Parameter(torch.zeros(pool_size))
                if method == "S2_rna_token_prompt":
                    self.adapter = nn.Linear(512, 16)
                    self.gate = nn.Linear(16, pool_size)
                if method == "S3_static_concepts_plus_rna":
                    self.adapter = nn.Linear(512, 16)
                self.head = nn.Linear(48 if method == "S3_static_concepts_plus_rna" else 32, n_classes)

            def prompt(self, rna):
                if self.method == "S1_static_token_prompt":
                    alpha = torch.softmax(self.pool_logits, dim=0).view(1, -1)
                    return torch.einsum("bm,mld->bld", alpha.expand(rna.shape[0], -1), self.pool)
                h = self.adapter(rna)
                alpha = torch.softmax(self.gate(h), dim=1)
                return torch.einsum("bm,mld->bld", alpha, self.pool)

            def forward_features(self, static, rna, projector):
                if self.method == "S0_static_concepts":
                    return static
                if self.method == "S3_static_concepts_plus_rna":
                    return torch.cat([static, self.adapter(rna)], dim=1)
                if self.method == "S1_static_token_prompt":
                    text = projector.encode(self.prompt(rna[:1])).expand(rna.shape[0], -1, -1)
                else:
                    text = projector.encode(self.prompt(rna))
                return torch.einsum("bd,bkd->bk", projector.image_batch, text)

            def forward(self, static, rna, projector):
                return self.head(self.forward_features(static, rna, projector))

        self.net = Net().to(device)
        self.device = device
        self.method = method

    def count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


def class_weights(y, n_classes, device):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    present = counts > 0
    result = np.zeros(n_classes, dtype=np.float32)
    result[present] = len(y) / (present.sum() * counts[present])
    result[present] /= result[present].mean()
    return torch.tensor(result, dtype=torch.float32, device=device)


def run_epochs(net: PromptNet, image, static, rna, y, train_idx, projector, max_epoch, checkpoints, device, batch_size=12):
    import torch
    import torch.nn.functional as F

    # The frozen image representation is attached to the projector so the
    # prompt model's forward stays compact and does not copy a second tensor.
    projector.image_batch = image
    net.net.train()
    optimizer = torch.optim.AdamW(net.net.parameters(), lr=0.005, weight_decay=1e-3)
    idx = np.asarray(train_idx, dtype=np.int64)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    weight = class_weights(y[idx], int(net.net.head.out_features), device)
    result = []
    rng = np.random.default_rng(20261007)
    for epoch in range(1, max_epoch + 1):
        losses = []
        # One deterministic permutation per epoch; labels never enter ordering.
        order = idx[rng.permutation(len(idx))]
        for start in range(0, len(order), batch_size):
            batch = torch.tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            rna_b = rna[batch]
            static_b = static[batch]
            image_b = image[batch]
            projector.image_batch = image_b
            optimizer.zero_grad(set_to_none=True)
            logits = net.net(static_b, rna_b, projector)
            loss = F.cross_entropy(logits, y_t[batch], weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.net.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        result.append({"epoch": epoch, "mean_batch_train_ce": float(np.mean(losses))})
    return result


def predict(net: PromptNet, image, static, rna, projector, batch_size, device):
    import torch

    net.net.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(image), batch_size):
            sl = slice(start, start + batch_size)
            projector.image_batch = image[sl]
            logits = net.net(static[sl], rna[sl], projector)
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def fit_outer(method, image_np, static_np, rna_raw, y, outer_train, outer_test, n_classes, seed, epoch, projector, device):
    import torch

    rna_all, audit = fit_numeric(rna_raw, outer_train)
    audit["text_runtime"] = "compact_padding_original_CLS_position_shared_static_prompt"
    image = torch.tensor(image_np, dtype=torch.float32, device=device)
    static = torch.tensor(static_np, dtype=torch.float32, device=device)
    rna = torch.tensor(rna_all, dtype=torch.float32, device=device)
    net = PromptNet(method, n_classes, 768, 4, 4, seed, device)
    audit["training_history"] = run_epochs(net, image, static, rna, y, outer_train, projector, epoch, {epoch}, device)
    matched = predict(net, image[outer_test], static[outer_test], rna[outer_test], projector, 12, device)
    shift = 1 + (seed + int(outer_test[0])) % max(1, len(outer_test) - 1)
    wrong_idx = np.roll(np.arange(len(outer_test)), shift)
    wrong = predict(net, image[outer_test], static[outer_test], rna[outer_test][wrong_idx], projector, 12, device) if method == "S2_rna_token_prompt" else None
    audit["wrong_patient_fixed_points"] = int(np.sum(wrong_idx == np.arange(len(outer_test)))) if wrong is not None else None
    count = net.count()
    del net, image, static, rna
    torch.cuda.empty_cache()
    return matched, wrong, count, audit


def metrics(y, probs, n_classes):
    pred = probs.argmax(1)
    result = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, average="macro")),
              "accuracy": float(accuracy_score(y, pred))}
    if n_classes == 2:
        result["macro_ovr_auroc"] = float(roc_auc_score(y, probs[:, 1]))
    else:
        result["macro_ovr_auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=np.arange(n_classes)))
    return result


def bootstrap_delta(y, p_a, p_b, reps, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        # Preserve rare endpoint classes and use identical patient draws for
        # the two methods. Bootstrap dispersion is not the point estimate.
        idx = np.concatenate([rng.choice(np.flatnonzero(y == label), size=np.sum(y == label), replace=True)
                              for label in np.unique(y)])
        values[i] = balanced_accuracy_score(y[idx], p_a[idx].argmax(1)) - balanced_accuracy_score(y[idx], p_b[idx].argmax(1))
    lo, hi = np.quantile(values, [0.00625, 0.99375])
    point = balanced_accuracy_score(y, p_a.argmax(1)) - balanced_accuracy_score(y, p_b.argmax(1))
    return {"delta_balanced_accuracy": float(point), "bootstrap_mean_delta": float(values.mean()),
            "ci98_75_low": float(lo), "ci98_75_high": float(hi), "bootstrap_replicates": reps,
            "bootstrap_scheme": "class_stratified_paired_patient", "bootstrap_seed": seed}


def audit_projector(model, input_ids, templates, official, projector, device):
    """Check official parity, shifted mask semantics and prompt gradients."""
    import torch

    empty = FrozenTextProjector(model, input_ids, templates, 0, device)
    with torch.no_grad():
        baseline = empty.encode(torch.empty((1, 0, 768), device=device))[0]
    parity = float(np.max(np.abs(baseline.cpu().numpy() - official)))
    if parity > 2e-4:
        raise RuntimeError(f"custom no-prompt text path diverges from official TITAN: {parity}")
    if not bool((projector.mask_ids[:, 1:5] != model.text_encoder.pad_id).all()):
        raise RuntimeError("prompt tokens not visible to CLS")
    if not torch.equal(projector.mask_ids[:, 5:], projector.ids[:, 1:-4]):
        raise RuntimeError("shifted concept padding mask differs from inserted sequence")
    torch.manual_seed(20261007)
    prompt = (torch.randn((1, 4, 768), device=device) * 0.01).requires_grad_()
    encoded = projector.encode(prompt)
    probe = torch.randn_like(encoded)
    (encoded * probe).sum().backward()
    grad_norm = float(prompt.grad.norm().detach().cpu())
    if not bool(torch.isfinite(prompt.grad).all()) or grad_norm <= 0:
        raise RuntimeError("nonfinite or zero token-prompt gradient")
    trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    if trainable or any(p.grad is not None for p in model.text_encoder.parameters()):
        raise RuntimeError("TITAN text parameters acquired gradients")
    shifted_count = int((projector.mask_ids != projector.ids).sum().cpu())
    # Numerical gate for the runtime optimization, without endpoint labels.
    # Compare full padding with compact padding, retaining the original CLS
    # position and causal visibility in both paths.
    test_prompts = (torch.randn((2, 4, 768), device=device) * 0.01).requires_grad_()
    projector.compact = False
    full = projector.encode(test_prompts)
    probe = torch.randn_like(full)
    full_grad = torch.autograd.grad((full * probe).sum(), test_prompts)[0]
    projector.compact = True
    compact = projector.encode(test_prompts)
    compact_grad = torch.autograd.grad((compact * probe).sum(), test_prompts)[0]
    compact_error = float((full - compact).abs().max().detach().cpu())
    compact_grad_relative = float((full_grad - compact_grad).norm().detach().cpu() / full_grad.norm().detach().cpu())
    if compact_error > 3e-4 or compact_grad_relative > 0.01:
        raise RuntimeError(f"compact text optimization diverged: {compact_error}, {compact_grad_relative}")
    shared_prompt = (torch.randn((1, 4, 768), device=device) * 0.01).requires_grad_()
    repeated = projector.encode(shared_prompt.expand(2, -1, -1))
    repeat_grad = torch.autograd.grad((repeated * probe).sum(), shared_prompt)[0]
    shared = projector.encode(shared_prompt).expand(2, -1, -1)
    shared_grad = torch.autograd.grad((shared * probe).sum(), shared_prompt)[0]
    shared_error = float((shared - repeated).abs().max().detach().cpu())
    shared_grad_relative = float((shared_grad - repeat_grad).norm().detach().cpu() / repeat_grad.norm().detach().cpu())
    if shared_error > 3e-4 or shared_grad_relative > 0.01:
        raise RuntimeError(f"shared static prompt optimization diverged: {shared_error}, {shared_grad_relative}")
    return {"status": "PASS", "no_prompt_official_max_abs_error": parity,
            "prompt_grad_norm": grad_norm, "text_transformer_trainable_parameters": trainable,
            "shifted_mask_different_positions": shifted_count,
            "template_aggregation": "normalize_each_template_then_mean_then_normalize",
            "cls_mask": "BOS + nonpad_prompt + shifted_original_concept_mask",
            "compact_prefix_tokens": projector.prefix_len, "original_cls_position": projector.ids.shape[1],
            "compact_full_embedding_max_abs_error": compact_error,
            "compact_full_gradient_relative_error": compact_grad_relative,
            "shared_static_embedding_max_abs_error": shared_error,
            "shared_static_gradient_relative_error": shared_grad_relative}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rna", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--clinical-targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--reuse-baselines", type=Path, help="reuse validated v1 S0/S3 OOF cells only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--audit-only", action="store_true", help="run encoder gates without reading endpoint labels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oof").mkdir(exist_ok=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("token-level TITAN prompt requires CUDA")
    torch.set_float32_matmul_precision("high")
    started = time.time()

    rep = np.load(args.cache / "patient_representations.npz", allow_pickle=False)
    ids = rep["patient_ids"].astype(str)
    image_np = unit(rep["semantic"].astype(np.float32))
    static_np = rep["concept_cosine"].astype(np.float32)
    cached_text = unit(rep["text_prototypes"].astype(np.float32))
    model, loading = load_titan(args.model, args.device)
    input_ids, template_count, vocab = text_tokens(model, args.concepts)
    official = official_text_prototypes(model, vocab, args.device)
    if float(np.max(np.abs(official - cached_text))) > 2e-4:
        raise RuntimeError("official TITAN text prototypes do not match Phase6A cache")
    projector = FrozenTextProjector(model, input_ids, template_count, 4, args.device)
    encoder_audit = audit_projector(model, input_ids, template_count, official, projector, args.device)
    write_json(args.out / "encoder_audit.json", encoder_audit)
    if args.audit_only:
        print(json.dumps(encoder_audit), flush=True)
        return 0

    folds = pd.read_csv(args.folds, dtype={"patient_id": str, "fold": int})
    folds["patient_id"] = folds.patient_id.str.upper()
    if folds.patient_id.duplicated().any() or len(folds) != len(ids) or set(folds.patient_id) != set(ids) or set(folds.fold) != set(range(1, 6)):
        raise ValueError("fixed folds do not match Phase6A patients")
    fold_map = dict(zip(folds.patient_id, folds.fold))
    fold_ids = np.array([fold_map[x] for x in ids])
    targets = pd.read_csv(args.clinical_targets, dtype=str, usecols=["patient_id", *ENDPOINTS]).set_index("patient_id").loc[ids]
    raw_rna, rna_audit = load_rna_only(args.rna, list(ids))
    config = {"version": "phase6b_token_prompt_titan_v2_mask_repair", "seeds": [20261007, 20261008, 20261009],
              "outer_folds": 5, "inner_folds": 0, "targets": list(ENDPOINTS),
              "controls": ["S0_static_concepts", "S1_static_token_prompt", "S2_rna_token_prompt", "S3_static_concepts_plus_rna", "S4_wrong_patient_rna_token_prompt"],
              "prompt": {"insertion": "after BOS and before original concept tokens", "prompt_len": 4, "pool_size": 4, "basis_shape": [4, 4, 768], "text_templates": template_count},
              "schedule": {"max_epochs": 3, "checkpoint_epochs": [3], "batch_size": 12, "optimizer": "AdamW(lr=0.005,weight_decay=0.001)", "selection": "fixed before OOF; no label-driven epoch selection"},
              "frozen": ["TITAN text transformer", "TITAN image encoder", "Phase6A image representations", "Phase6A static text concepts"],
              "survival_columns_read": [], "clinical_columns_used": ["T", "N", "M", "Stage"],
              "repair": {"cls_mask": "shifted_with_prompt_insertion", "template_aggregation": "official_per_template_normalization",
                         "reuse_baselines": str(args.reuse_baselines) if args.reuse_baselines else None,
                         "S4_predictions_persisted": True}}
    if (args.out / "config.json").exists() and args.resume:
        if json.loads((args.out / "config.json").read_text()) != config:
            raise RuntimeError("resume configuration mismatch")
    write_json(args.out / "config.json", config)
    input_audit = {"model_weight_sha256": sha256(args.model / "model.safetensors"),
                                                 "representation_cache_sha256": sha256(args.cache / "patient_representations.npz"),
                                                 "rna": rna_audit, "titan_official_text_max_abs_error": float(np.max(np.abs(official - cached_text))),
                                                 "text_transformer_trainable_parameters": 0,
                                                 "folds_sha256": sha256(args.folds), "targets_sha256": sha256(args.clinical_targets),
                                                 "concepts_sha256": sha256(args.concepts)}
    write_json(args.out / "input_audit.json", input_audit)
    if args.reuse_baselines:
        source_done = json.loads((args.reuse_baselines / "DONE.json").read_text())
        source_config = json.loads((args.reuse_baselines / "config.json").read_text())
        source_audit = json.loads((args.reuse_baselines / "input_audit.json").read_text())
        if source_done["status"] != "PASS" or source_done["config_sha256"] != sha256(args.reuse_baselines / "config.json"):
            raise RuntimeError("baseline source incomplete or configuration tampered")
        if source_config["seeds"] != config["seeds"] or source_config["schedule"] != config["schedule"]:
            raise RuntimeError("baseline source seed/schedule mismatch")
        for key in ("model_weight_sha256", "representation_cache_sha256"):
            if source_audit[key] != input_audit[key]:
                raise RuntimeError(f"baseline source mismatch: {key}")
        if source_audit["rna"]["input_sha256"] != rna_audit["input_sha256"]:
            raise RuntimeError("baseline source RNA mismatch")
    (args.out / "audits").mkdir(exist_ok=True)

    metric_rows, pair_rows, selection_rows = [], [], []
    counts = {}
    reuse_rows = []
    training_rows = []
    for seed in config["seeds"]:
        for endpoint in ENDPOINTS:
            labels = [normalize_label(v, endpoint) for v in targets[endpoint]]
            valid = np.array([v is not None for v in labels])
            classes = sorted({v for v in labels if v is not None})
            mapping = {v: i for i, v in enumerate(classes)}
            y = np.array([mapping[v] if v is not None else -1 for v in labels])
            outer_predictions = {m: np.full((len(ids), len(classes)), np.nan, dtype=np.float32) for m in TOKEN_METHODS + ("S4_wrong_patient_rna_token_prompt",)}
            y_oof = np.full(len(ids), -1, dtype=int)
            for fold in range(1, 6):
                outer_test = np.flatnonzero((fold_ids == fold) & valid)
                outer_train = np.flatnonzero((fold_ids != fold) & valid)
                for method in TOKEN_METHODS:
                    epoch, inner_score, details = 3, float("nan"), {"selection": "fixed_schedule_no_inner_selection"}
                    selection_rows.append({"seed": seed, "endpoint": endpoint, "outer_fold": fold, "method": method, "selected_epoch": epoch, "inner_balanced_accuracy": inner_score})
                    cell_name = f"{endpoint}__{method}__seed{seed}__fold{fold}.npz"
                    cell_path = args.out / "oof" / cell_name
                    wrong_path = args.out / "oof" / cell_name.replace(method, "S4_wrong_patient_rna_token_prompt")
                    source = None
                    if args.resume and cell_path.exists() and (method != "S2_rna_token_prompt" or wrong_path.exists()):
                        source = cell_path
                    elif args.reuse_baselines and method in {"S0_static_concepts", "S3_static_concepts_plus_rna"}:
                        source = args.reuse_baselines / "oof" / cell_name
                    count = 33 * len(classes) if method == "S0_static_concepts" else 49 * len(classes) + 8208 if method == "S3_static_concepts_plus_rna" else 12292 + 33 * len(classes) if method == "S1_static_token_prompt" else 12288 + 8208 + 68 + 33 * len(classes)
                    wrong = None
                    if source:
                        with np.load(source, allow_pickle=False) as saved:
                            if not np.array_equal(saved["patient_ids"].astype(str), ids[outer_test]) or not np.array_equal(saved["y_true"], y[outer_test]) or not np.array_equal(saved["classes"], classes) or int(saved["selected_epoch"]) != epoch:
                                raise RuntimeError(f"baseline/resume cell provenance mismatch: {cell_name}")
                            matched = saved["probabilities"].copy()
                        if source != cell_path:
                            shutil.copyfile(source, cell_path)
                        if args.reuse_baselines and method in {"S0_static_concepts", "S3_static_concepts_plus_rna"}:
                            baseline = args.reuse_baselines / "oof" / cell_name
                            if sha256(baseline) != sha256(cell_path):
                                raise RuntimeError(f"resumed baseline differs from original valid cell: {cell_name}")
                            reuse_rows.append({"endpoint": endpoint, "seed": seed, "fold": fold, "method": method, "source_sha256": sha256(baseline)})
                        if method == "S2_rna_token_prompt":
                            with np.load(wrong_path, allow_pickle=False) as saved:
                                wrong = saved["probabilities"].copy()
                    else:
                        matched, wrong, count, fit_audit = fit_outer(method, image_np, static_np, raw_rna, y, outer_train, outer_test, len(classes), seed + fold * 10000, epoch, projector, args.device)
                        write_json(args.out / "audits" / cell_name.replace(".npz", ".json"), fit_audit)
                    audit_path = args.out / "audits" / cell_name.replace(".npz", ".json")
                    if audit_path.exists():
                        fit_audit = json.loads(audit_path.read_text())
                        for row in fit_audit["training_history"]:
                            training_rows.append({"seed": seed, "endpoint": endpoint, "fold": fold, "method": method, **row})
                    if not np.isfinite(matched).all() or not np.allclose(matched.sum(axis=1), 1.0, atol=1e-5):
                        raise RuntimeError(f"invalid OOF probabilities: {cell_name}")
                    outer_predictions[method][outer_test] = matched
                    if method == "S2_rna_token_prompt":
                        outer_predictions["S4_wrong_patient_rna_token_prompt"][outer_test] = wrong
                    counts[f"{endpoint}/{method}"] = count
                    np.savez_compressed(cell_path, patient_ids=ids[outer_test], y_true=y[outer_test], probabilities=matched, classes=np.asarray(classes, dtype=str), selected_epoch=np.asarray(epoch))
                    if wrong is not None:
                        np.savez_compressed(wrong_path, patient_ids=ids[outer_test], y_true=y[outer_test], probabilities=wrong, classes=np.asarray(classes, dtype=str), selected_epoch=np.asarray(epoch))
                    print(f"CELL_DONE endpoint={endpoint} seed={seed} fold={fold} method={method} source={'reuse' if source else 'fit'}", flush=True)
                y_oof[outer_test] = y[outer_test]
            keep = y_oof >= 0
            for method, probs in outer_predictions.items():
                if not np.isfinite(probs[keep]).all():
                    raise RuntimeError(f"missing OOF {endpoint} {method} seed {seed}")
                metric_rows.append({"seed": seed, "endpoint": endpoint, "method": method, "n": int(keep.sum()), **metrics(y_oof[keep], probs[keep], len(classes))})
            comparisons = [("S2-S1", "S2_rna_token_prompt", "S1_static_token_prompt"), ("S2-S3", "S2_rna_token_prompt", "S3_static_concepts_plus_rna"), ("S2-S4", "S2_rna_token_prompt", "S4_wrong_patient_rna_token_prompt")]
            for label, a, b in comparisons:
                stable = int.from_bytes(hashlib.sha256(f"{endpoint}|{label}".encode()).digest()[:4], "little")
                pair_rows.append({"seed": seed, "endpoint": endpoint, "comparison": label, **bootstrap_delta(y_oof[keep], outer_predictions[a][keep], outer_predictions[b][keep], args.bootstrap_replicates, seed + stable % 100000)})
            pd.DataFrame(metric_rows).to_csv(args.out / "metrics.csv", index=False)
            pd.DataFrame(pair_rows).to_csv(args.out / "paired_comparisons_by_seed.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.out / "paired_comparisons_by_seed.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.out / "selection_summary.csv", index=False)
    pd.DataFrame(reuse_rows).to_csv(args.out / "reused_baselines.csv", index=False)
    pd.DataFrame(training_rows).to_csv(args.out / "training_history.csv", index=False)
    write_json(args.out / "parameter_counts.json", counts)
    validation = {"status": "PASS", "patients": len(ids), "seeds": 3, "outer_folds": 5, "inner_folds": 0,
                  "metrics_rows": len(metric_rows), "pair_rows": len(pair_rows), "oof_files": len(list((args.out / "oof").glob("*.npz"))),
                  "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": time.time() - started, "survival_columns_read": [],
                  "text_transformer_trainable_parameters": 0, "patient_level_outputs_published": False,
                  "reused_baseline_cells": len(reuse_rows), "encoder_audit": encoder_audit}
    if validation["oof_files"] != 300 or len(metric_rows) != 60 or len(pair_rows) != 36:
        raise RuntimeError(f"incomplete outputs: {validation}")
    write_json(args.out / "validation.json", validation)
    write_json(args.out / "DONE.json", {"status": "PASS", "validation": validation, "config_sha256": sha256(args.out / "config.json")})
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
