"""What scripts/train.py trains and validates on: load the --data dirs, split validation by screen position, and
check coverage before any model is loaded.

The coverage gate exists because of a real bug: a validation split sampled across all characters left Ken and
Dhalsim with no validation rows at all, and nothing failed. ``coverage_problems`` makes every such gap a hard stop:
each character must be in train and in validation in equal shares, and every (action, range, posture, side) a
character has must have enough training rows.
"""
import collections
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

MIN_COMBO_ROWS = 5       # training rows per (character, action, range, posture, side)
MIN_SHARE_RATIO = 0.8    # smallest / largest character, in train rows and in validation rows


def position(ex: Dict) -> Tuple:
    """The screen position a row was asked about: its dataset and current frame, a mirrored frame counted as its
    source. Every action asked at one position, and its mirrored twin, share this key."""
    now = os.path.basename(ex["state"]["images"][-1]).replace("mirror_", "")
    return ex.get("dataset"), now


def split_by_position(rows: List[Dict], rng: random.Random, share: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
    """Hold out ``share`` of each dataset's positions (all their rows) for early stopping, so validation never shows
    a frame the model trains on and every dataset (character) is in it equally. Splitting rows at random would put a
    position's other actions and its mirror in training."""
    held = set()
    for d in sorted({ex.get("dataset") for ex in rows}):
        keys = sorted({position(ex) for ex in rows if ex.get("dataset") == d})
        held |= set(rng.sample(keys, max(1, round(share * len(keys)))))
    return [ex for ex in rows if position(ex) not in held], [ex for ex in rows if position(ex) in held]


def load_data(dirs: Sequence[str], seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Every dir's train.jsonl (and val.jsonl if it has one) as laya-vision examples; without any val.jsonl, a
    position split of the training rows."""
    import laya.vlm_train as vt

    train, val = [], []
    for d in dirs:
        root, name = os.path.split(os.path.normpath(d))
        train += vt.load_jsonl_examples(root, name, "train")
        if os.path.exists(os.path.join(d, "val.jsonl")):
            val += vt.load_jsonl_examples(root, name, "val")
    if not val:
        train, val = split_by_position(train, random.Random(seed))
    return train, val


def _records(dirs: Sequence[str]) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        for f in ("train", "val"):
            path = os.path.join(d, f + ".jsonl")
            if os.path.exists(path):
                for line in open(path):
                    r = json.loads(line)
                    out[(name, r.get("id"))] = r
    return out


def coverage_problems(train: List[Dict], val: List[Dict], dirs: Sequence[str]) -> List[str]:
    """Everything wrong with the representation of each dataset in what is about to be trained on; empty = OK."""
    from .vs_sweep import RANGES, SPECIALS, STAGE1_POSTURES, actions

    names = [os.path.basename(os.path.normpath(d)) for d in dirs]
    problems: List[str] = []
    n_train = collections.Counter(ex.get("dataset") for ex in train)
    n_val = collections.Counter(ex.get("dataset") for ex in val)
    for split, n in (("train", n_train), ("val", n_val)):
        for name in names:
            if n[name] == 0:
                problems.append("%s has no %s rows" % (name, split))
        present = [n[x] for x in names if n[x]]
        if present and min(present) / max(present) < MIN_SHARE_RATIO:
            problems.append("%s rows unequal across datasets: %s" % (split, {x: n[x] for x in names}))
    shared = {position(ex) for ex in train} & {position(ex) for ex in val}
    if shared:
        problems.append("%d validation positions also in train, e.g. %s" % (len(shared), sorted(shared)[:2]))
    recs = _records(dirs)
    combos = collections.Counter()
    labels = collections.defaultdict(set)
    for ex in train:
        r = recs.get((ex.get("dataset"), ex.get("id")))
        if r and "action" in r:
            combos[(ex["dataset"], r["action"], r["range"], r["posture"], r["side"])] += 1
            labels[ex["dataset"]].add(r["outcome"])
    for name in names:
        if name not in SPECIALS:
            continue                      # not a stage-1 character dir: only the share checks above apply
        for a in actions(name):
            for rng in RANGES:
                for p in STAGE1_POSTURES:
                    for side in ("left", "right"):
                        got = combos[(name, a, rng, p, side)]
                        if got < MIN_COMBO_ROWS:
                            problems.append("%s %s@%s/%s/%s: %d training rows < %d" % (
                                name, a, rng, p, side, got, MIN_COMBO_ROWS))
        if len(labels[name]) < 3:
            problems.append("%s training outcomes only %s" % (name, sorted(labels[name])))
    return problems


def coverage_table(train: List[Dict], val: List[Dict], dirs: Sequence[str]) -> str:
    names = [os.path.basename(os.path.normpath(d)) for d in dirs]
    nt = collections.Counter(ex.get("dataset") for ex in train)
    nv = collections.Counter(ex.get("dataset") for ex in val)
    pv = collections.Counter(d for d, _ in {position(ex) for ex in val})
    return "\n".join("  %-8s train %5d  val %4d rows (%d positions)" % (n, nt[n], nv[n], pv[n]) for n in names)
