"""Records on disk, in the JSONL layout laya-vision's ``laya.vlm_train.load_jsonl_examples`` reads
(<root>/<name>/train.jsonl, val.jsonl, images/*.png; built by scripts/vs_dataset.py)."""
import json
from typing import Dict, Iterable, List

import numpy as np


def save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, optimize=False, compress_level=1)


def read(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, recs: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n
