"""laya-vision's input: two frames + the text note.

The state built here must match what ``laya.vlm_train.jsonl_example`` builds from a dataset record:
``{"images": [prev, cur], "context": state_text}``. Both sides load images as RGB PIL images (PNG is lossless),
so train and play see the same pixels.
"""
from typing import Dict

import numpy as np

from .data.frames import model_frame


def make_state(prev: np.ndarray, cur: np.ndarray, text: str) -> Dict:
    from PIL import Image

    return {"images": [Image.fromarray(model_frame(prev)).convert("RGB"),
                       Image.fromarray(model_frame(cur)).convert("RGB")], "context": text}
