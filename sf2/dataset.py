"""Records on disk, in the exact JSONL layout laya-vision's ``laya.vlm_train.load_jsonl_examples`` reads.

    <root>/<name>/train.jsonl, val.jsonl, images/*.png

One record = one decision frame:
    {"id", "images": [prev.png, cur.png], "state_text", "question", "label", "target",
     "episode", "step", "source", "meta": {...}}

``label`` / ``target`` are the gold (teacher). ``meta`` holds everything the relabel and gate steps need (who
acted, what the student chose, life totals, frame counter, round); laya ignores it.
Rollouts use the same layout, so a rollout directory is already a dataset whose gold is the teacher's.
"""
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import actions as A


def save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, optimize=False, compress_level=1)


class Writer:
    """Append decision frames to <root>/<name>/{split}.jsonl; val = every ``val_every``-th episode.

    ``skip_uncontrollable``: training data leaves out decisions where the stick did nothing (meta controllable
    False: hit or knocked down); rollouts keep them for the gate."""

    def __init__(self, root: str, name: str, source: str, val_every: int = 10, skip_uncontrollable: bool = False):
        self.dir = os.path.join(root, name)
        if os.path.exists(os.path.join(self.dir, "train.jsonl")):
            raise FileExistsError("%s already holds a dataset; pick another name or delete it" % self.dir)
        os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
        self.source, self.val_every, self.skip_uncontrollable = source, val_every, skip_uncontrollable
        self.files = {s: open(os.path.join(self.dir, s + ".jsonl"), "w") for s in ("train", "val")}
        self.n = {"train": 0, "val": 0}

    def split(self, episode: int) -> str:
        return "val" if self.val_every and episode % self.val_every == self.val_every - 1 else "train"

    def record(self, episode: int, step: int, prev: np.ndarray, cur: np.ndarray, text: str,
               target: Dict[str, float], meta: Optional[Dict] = None) -> Dict:
        """Save the images and build the record; ``add`` it once its meta is final."""
        rid = "%s-e%05d-s%06d" % (self.source, episode, step)
        if self._skip(meta or {}):
            return {"id": rid, "episode": episode, "step": step, "meta": meta or {}}
        cur_p = "images/%s.png" % rid
        save_png(cur, os.path.join(self.dir, cur_p))
        if np.array_equal(prev, cur):
            prev_p = cur_p
        else:
            prev_p = "images/%s_prev.png" % rid
            save_png(prev, os.path.join(self.dir, prev_p))
        tvec = [float(target.get(a, 0.0)) for a in A.ACTIONS]
        rec = {"id": rid, "images": [prev_p, cur_p], "state_text": text, "question": A.question(),
               "label": int(np.argmax(tvec)), "target": tvec, "episode": episode, "step": step,
               "source": self.source, "meta": meta or {}}
        return rec

    def _skip(self, meta: Dict) -> bool:
        return self.skip_uncontrollable and meta.get("controllable") is False

    def add(self, rec: Dict) -> None:
        if self._skip(rec["meta"]):
            return
        s = self.split(rec["episode"])
        self.files[s].write(json.dumps(rec) + "\n")
        self.files[s].flush()
        self.n[s] += 1

    def close(self):
        for f in self.files.values():
            f.close()
        with open(os.path.join(self.dir, "_READY"), "w") as f:
            json.dump(self.n, f)


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


def one_hot(action: str) -> List[float]:
    return [1.0 if a == action else 0.0 for a in A.ACTIONS]
