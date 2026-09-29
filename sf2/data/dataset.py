"""Records on disk, in the JSONL layout laya-vision's ``laya.vlm_train.load_jsonl_examples`` reads
(<root>/<name>/train.jsonl, val.jsonl, images/*.png; built by scripts/vs_dataset.py)."""
import json
import os
from typing import Dict, Iterable, List

import numpy as np

from ..vocab import FIGHTERS


def save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, optimize=False, compress_level=1)


def read(path: str, missing_ok: bool = False) -> List[Dict]:
    """The records of a JSONL file (blank lines skipped); [] for a missing file when ``missing_ok``."""
    if missing_ok and not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, recs: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def dataset_chars(root: str) -> List[str]:
    """The characters with a folder in ``root`` (test_data/): other folders (_shards, advice) are not characters."""
    return sorted(d for d in os.listdir(root) if d in FIGHTERS and os.path.isdir(os.path.join(root, d)))
