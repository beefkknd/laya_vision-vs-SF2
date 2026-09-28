"""LoRA for text laya's ModernBERT encoder, in plain MLX. Copied from laya_othello (az/lora.py), which trained the
same encoder this way: every base weight frozen, a low-rank update beside each target linear layer, B starting at
zero so the adapted model starts as exactly pretrained laya.

    y = W x + (x A B) * (alpha / r)          A: (in, r)   B: (r, out)
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

#: ModernBERT's linear layers: attention Wqkv/Wo and MLP Wi/Wo.
TARGETS = ("Wqkv", "Wo", "Wi")
#: Attention projections only - the smaller, "surgical" LoRA scope. Qualified
#: by the parent module: ModernBERT has BOTH attn.Wo and mlp.Wo.
ATTENTION = ("attn.Wqkv", "attn.Wo")


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        out_dim, in_dim = base.weight.shape
        self.base = base
        self.base.freeze()
        bound = 1.0 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dim, r))
        self.lora_b = mx.zeros((r, out_dim))          # zero: starts as the base
        self.scale = alpha / r                        # a float, not a parameter

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        # Back to the BASE dtype: with a bf16 base, an fp32 update would promote
        # every activation downstream to fp32 and erase bf16's speed.
        return y + ((x @ self.lora_a @ self.lora_b) * self.scale).astype(y.dtype)


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0,
               targets=TARGETS, top_k: "int | None" = None) -> int:
    """Freeze ALL of `model`, then wrap every nn.Linear attribute named in
    `targets` with a LoRALinear (in place). Returns how many were wrapped.

    `top_k`: adapt only the LAST k entries of `model.layers`. Nothing below the
    lowest adapter needs a gradient, so the backward pass stops there - the
    main speed lever once training is compute-bound (measured ~20 TFLOP/s at
    bf16 batch 128 with all 28 layers adapted)."""
    model.freeze()
    if top_k is None:
        scopes = [model]
    else:
        layers = getattr(model, "layers", None)
        if not layers or not 1 <= top_k <= len(layers):
            raise ValueError(f"top_k={top_k} needs 1..{len(layers or [])} `layers`")
        scopes = layers[-top_k:]
    split = [t.rpartition(".") for t in targets]          # "attn.Wo" -> ("attn", ".", "Wo")
    sites = [(mod, attr) for scope in scopes for name, mod in scope.named_modules()
             for parent, _, attr in split
             if isinstance(getattr(mod, attr, None), nn.Linear)
             and (not parent or name.rsplit(".", 1)[-1] == parent)]
    for mod, attr in sites:
        setattr(mod, attr, LoRALinear(getattr(mod, attr), r, alpha))
    if not sites:
        raise ValueError(f"no nn.Linear named {targets} found - nothing to adapt")
    return len(sites)


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRALinear's update into its base weight and unwrap it, in
    place: W <- W + (A B scale)^T (nn.Linear computes x W^T). The merged model
    computes what the adapted one did. Returns how many were merged."""
    sites = [(mod, key) for _, mod in model.named_modules()
             for key, child in list(mod.items()) if isinstance(child, LoRALinear)]
    for mod, key in sites:
        adapted = mod[key]
        base = adapted.base
        delta = (adapted.lora_a @ adapted.lora_b * adapted.scale).T
        base.weight = base.weight + delta.astype(base.weight.dtype)
        setattr(mod, key, base)
    return len(sites)


def trainable_count(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def save_adapter(model: nn.Module, path, r: int, alpha: float, meta: dict,
                 top_k: "int | None" = None, targets=TARGETS) -> None:
    """Write the adapter (+ head) and a JSON sidecar with the LoRA shape - the
    only thing a later load needs besides the base. ~15 MB, not 1.58 GB."""
    import json
    from pathlib import Path
    path = Path(path)
    mx.save_safetensors(str(path), adapter_state(model))
    sidecar = {"r": r, "alpha": alpha, "targets": list(targets), "top_k": top_k, **meta}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def load_chain(model: nn.Module, base_path, adapter_paths=()) -> nn.Module:
    """Base weights, then each adapter in order: wrap, load, merge. The result
    has no LoRALinear left - ready to be adapted again for the next round.
    `model.encoder` is what gets adapted; each adapter also carries the head weights it trained."""
    import json
    from pathlib import Path
    if base_path is not None:                  # None: the original (vanilla) weights
        model.load_weights(str(base_path))
    for ad in map(Path, adapter_paths):
        meta = json.loads(ad.with_suffix(".json").read_text())
        apply_lora(model.encoder, r=meta["r"], alpha=meta["alpha"],
                   targets=tuple(meta["targets"]), top_k=meta.get("top_k"))
        model.load_weights(str(ad), strict=False)
        merge_lora(model.encoder)
    return model


def adapter_state(model: nn.Module) -> dict:
    """What a LoRA run must save: the trainable parameters plus any frozen
    head statistics (StdHead mu/sd) - NOT the base encoder weights."""
    keep = dict(tree_flatten(model.trainable_parameters()))
    for k, v in tree_flatten(model.parameters()):
        if k.startswith("head."):
            keep[k] = v
    return keep
