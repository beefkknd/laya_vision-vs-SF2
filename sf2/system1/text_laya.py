"""Text laya (ModernBERT-large, laya-mlx on Apple MLX) as the advice follower: load it, optionally with a LoRA
checkpoint from scripts/train_text_laya.py, turn advice-data rows into model inputs, and score it.

Runs in the laya-mlx venv (~/work/laya_mlx/.venv), not this project's torch venv.
"""
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TEXT_LAYA_BASE

BASE = TEXT_LAYA_BASE
TRAINABLE = ("head", "scorer", "type_emb")   # the decision head trains with the LoRA; act_head stays as it is


def load(adapter_dir: Optional[str] = None, dtype: str = "float32"):
    """The laya-mlx Agent; with ``adapter_dir``, its LoRA folded into the encoder and its head weights loaded."""
    import laya_mlx

    from . import mlx_lora

    agent = laya_mlx.load(BASE, dtype=dtype)
    if adapter_dir:
        meta = json.load(open(os.path.join(adapter_dir, "adapter.json")))
        mlx_lora.apply_lora(agent.model.encoder, r=meta["r"], alpha=meta["alpha"], targets=tuple(meta["targets"]),
                            top_k=meta.get("top_k"))
        agent.model.load_weights(os.path.join(adapter_dir, "adapter.safetensors"), strict=False)
        mlx_lora.merge_lora(agent.model.encoder)
    return agent


def encode(agent, row: Dict) -> Tuple[Dict, List[str], np.ndarray]:
    """One advice row -> (laya item, option names in label order, target distribution over the options)."""
    items, internal = agent.prepare(row["text"], {"q": row["question"]})
    names = list(internal[0]["crit"])
    if len(items[0]["markers"]) != len(names):
        raise ValueError("%s: %d markers for %d options" % (row.get("id"), len(items[0]["markers"]), len(names)))
    target = np.array([1.0 if n in row["answers"] else 0.0 for n in names], dtype=np.float32)
    if not target.sum():
        raise ValueError("%s: no answer among the options" % row.get("id"))
    return items[0], names, target / target.sum()


def read(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path)]


def predict_rows(agent, rows: Sequence[Dict]) -> List[Dict[str, float]]:
    return [agent.predict(r["text"], {"q": r["question"]})["answers"]["q"]["probabilities"] for r in rows]


def score(rows: Sequence[Dict], probs: Sequence[Dict[str, float]]) -> Dict:
    """Accuracy (the top option is one of the answers) overall and by round, case, wording and character."""
    hit = [max(p, key=p.get) in r["answers"] for r, p in zip(rows, probs)]
    out = {"n": len(rows), "accuracy": round(float(np.mean(hit)), 4)}
    for key in ("round", "case", "words", "me"):
        groups = collections.defaultdict(list)
        for r, h in zip(rows, hit):
            groups[r.get(key, "-")].append(h)
        out[key] = {k: round(float(np.mean(v)), 3) for k, v in sorted(groups.items())}
    return out
