"""Minimal LoRA for laya-vision's language tower, merged back into plain weights before saving.

laya-vision's trainer only knows freezing stages (head / last_n / full). LoRA here = freeze the backbone, train the
decision head fully, and add rank-r adapters to the attention and MLP projections of the SmolVLM text layers.
The saved checkpoint is an ordinary full laya-vision checkpoint, so ``laya.load_vlm`` needs nothing extra.
"""
import copy
import math
from typing import Iterable, List

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype,
                                               device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                               device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += (self.lora_B @ self.lora_A) * self.scale
        return lin


def text_layers(encoder) -> nn.ModuleList:
    try:
        from laya.vlm import _text_tower

        return _text_tower(encoder)[0]
    except ImportError:
        return encoder.text_model.layers


def inject(encoder, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05,
           targets: Iterable[str] = TARGETS) -> int:
    """Wrap every target Linear in the text layers. Returns the number of adapters added."""
    n = 0
    for layer in text_layers(encoder):
        for parent in layer.modules():
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                    n += 1
    return n


def lora_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def merge(model: nn.Module) -> nn.Module:
    """A deep copy with every LoRALinear replaced by its merged Linear (the original keeps training)."""
    out = copy.deepcopy(model)
    for parent in list(out.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return out
