"""The log of fighter sprites the reader could not match confidently (owner decision 2026-10-02, the last section of
docs/laya_text_only_plan.md): the reader answers with a default ("stand") and the sprite is saved here for a later
catalog top-up. Images only (the crop of the frame), never RAM.

<dir>/crops/<sha1 of the crop>.png   each distinct crop once (the best match's box, padded; the whole frame when the
                                     fighter was not located at all)
<dir>/unknown.jsonl                  one line per unknown fighter-frame: frame index, side, character, best sprite,
                                     best score, box, crop file
"""
import hashlib
import json
import os
from typing import Optional

import numpy as np
from PIL import Image

from .assets import H, READER_DIR, W
from .facts import ScreenFacts

UNKNOWN_DIR = os.path.join(os.path.dirname(READER_DIR), "screen_reader_unknown")
MARGIN = 8


def crop(frame: np.ndarray, box) -> np.ndarray:
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    return frame[max(0, y0 - MARGIN):min(H, y1 + MARGIN), max(0, x0 - MARGIN):min(W, x1 + MARGIN)]


class UnknownLog:
    """Appends the unknown fighters of each frame fed; ``count`` lines written so far."""

    def __init__(self, path: str = UNKNOWN_DIR):
        self.path = path
        os.makedirs(os.path.join(path, "crops"), exist_ok=True)
        self.count = 0

    def add(self, frame: np.ndarray, facts: ScreenFacts, k: Optional[int] = None) -> int:
        n = 0
        for f in (facts.left, facts.right):
            if not f.unknown:
                continue
            img = np.ascontiguousarray(crop(frame, f.box))
            name = hashlib.sha1(repr(img.shape).encode() + img.tobytes()).hexdigest()[:16] + ".png"
            dst = os.path.join(self.path, "crops", name)
            if not os.path.exists(dst):
                Image.fromarray(img).save(dst)
            with open(os.path.join(self.path, "unknown.jsonl"), "a") as out:
                out.write(json.dumps(dict(k=k, side=f.side, character=f.character, found=f.found,
                                          best_sprite=f.sprite, best_score=f.confidence,
                                          box=list(f.box) if f.box else None, crop=name)) + "\n")
            n += 1
        self.count += n
        return n
