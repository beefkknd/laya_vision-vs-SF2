"""laya-vision as the student policy: two frames + the text note -> probabilities over the 12 options.

The state built here must match what ``laya.vlm_train.jsonl_example`` builds from a dataset record:
``{"images": [prev, cur], "context": state_text}``. Both sides load images as RGB PIL images (PNG is lossless),
so train and play see the same pixels.
"""
from typing import Dict, Tuple

import numpy as np

from . import actions as A


def make_state(prev: np.ndarray, cur: np.ndarray, text: str) -> Dict:
    from PIL import Image

    return {"images": [Image.fromarray(prev).convert("RGB"), Image.fromarray(cur).convert("RGB")], "context": text}


class LayaPolicy:
    def __init__(self, model: str, device: str = None, sample: bool = False, seed: int = 0):
        import laya

        self.agent = laya.load_vlm(model, device=device)
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.q = {"action": A.question()}

    def act(self, prev: np.ndarray, cur: np.ndarray, text: str) -> Tuple[str, Dict[str, float]]:
        ans = self.agent.predict(make_state(prev, cur, text), self.q)["answers"]["action"]
        probs = ans["probabilities"]
        if self.sample:
            p = np.array([probs[a] for a in A.ACTIONS], dtype=np.float64)
            return A.ACTIONS[int(self.rng.choice(len(p), p=p / p.sum()))], probs
        return ans["choice"], probs
