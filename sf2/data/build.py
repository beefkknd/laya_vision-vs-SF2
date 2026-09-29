"""Building laya-vision's stage-1 dataset from the shards (scripts/vs_dataset.py build): merge, today's notes,
mirror, the live-play cap, and per-character train / test files in test_data/<char>/. It exits 1 if any (action,
range) combination is short of examples, or if an attack never came out (a harness bug, not a data point)."""
import collections
import json
import os
from typing import Dict, List

import numpy as np

from ..config import TEST_DATA

from .dataset import read, save_png, write_jsonl
from .frames import HUD_ROWS, mirror_frame, model_frame
from . import vs_defense as D
from ..vocab import RANGES, range_of
from .vs_sweep import (OUTCOMES, STAGE1_POSTURES, actions,
                               mirror_record, outcome_question, static_actions, current_note)


ROOT = TEST_DATA
MIN_TRAIN, MIN_TEST = 14, 6   # per (action, range): 7 gaps x 2 postures train, 3 x 2 test
MIN_DEF_TRAIN, MIN_DEF_TEST = 18, 8   # blocks, per (answer, probe): 7 gaps x 3 ranges train, 3 x 3 test
MIN_LIVE = 20                  # live play: training rows per move


def _load(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def _write_frames(base: str, recs: List[Dict], mirrored: List[Dict]) -> List[str]:
    """frames/<x>.png = model_frame(images/<x>.png); frames/mirror_<x>.png = its whole-frame flip. Checked: every
    referenced frame exists, its HUD rows are black, and each mirror is exactly the flip of its source."""
    os.makedirs(os.path.join(base, "frames"), exist_ok=True)
    problems = []
    for r in recs:
        for p in r["images"]:
            raw = _load(os.path.join(base, p.replace("frames/", "images/")))
            save_png(model_frame(raw), os.path.join(base, p))
    for m in mirrored:
        for p in m["images"]:
            src = p.replace("frames/mirror_", "frames/")
            save_png(mirror_frame(_load(os.path.join(base, src.replace("frames/", "images/")))), os.path.join(base, p))
    checked = set()
    for r in recs + mirrored:
        for p in r["images"]:
            if p in checked:
                continue
            checked.add(p)
            img = _load(os.path.join(base, p))
            if img[:HUD_ROWS].any():
                problems.append("%s: HUD not blank" % p)
            if "mirror_" in p and not np.array_equal(img, _load(os.path.join(base, p.replace("mirror_", "")))[:, ::-1]):
                problems.append("%s: not the exact flip of its source" % p)
    return problems


def _laya(recs: List[Dict]) -> List[Dict]:
    return [dict(r, question=outcome_question(r["action"]), label=OUTCOMES.index(r["outcome"])) for r in recs]


def _file_side(r: Dict) -> str:
    """The side a row was COLLECTED on (which file it belongs in); a cross-up block row's own side is where the
    character ended up at the decision frame."""
    return r.get("collected_side", r["side"])


def build(args) -> int:
    shards = [os.path.join(ROOT, "_shards", f) for f in sorted(os.listdir(os.path.join(ROOT, "_shards")))
              if f.endswith(".jsonl")]
    by_char: Dict[str, List[Dict]] = collections.defaultdict(list)
    for s in shards:
        for r in read(s):
            if r["kind"] == "live" and r["dx"] == 0:
                continue        # the fighters at the same x (crossing): no left or right, a mirror would contradict it
            if r["posture"] in STAGE1_POSTURES or r["kind"] == "live":
                r["state_text"] = current_note(r)    # today's note: no opponent name, general bars, no constants
                if r["kind"] == "defense":     # what laya SEES at the decision frame (a jump-in has flown in, and
                    r["range"] = range_of(r["gap"])    # may have crossed over: then the character is on the other side)
                    r["collected_side"] = r.get("collected_side", r["side"])
                    now = "left" if r["dx"] > 0 else "right"
                    if now != r["side"]:
                        r["state_text"] = r["state_text"].replace("side=" + r["side"], "side=" + now)
                        r["side"], r["facing"] = now, "right" if now == "left" else "left"
                by_char[r["char"]].append(r)
    problems = []
    # live play: the same number of training rows for every character, and never more than its other data (the
    # still-opponent + block rows), so live play cannot dominate laya (System 1 stays general); earliest games first
    live_n = {c: sum(r["kind"] == "live" and r["split"] == "train" for r in rs) for c, rs in by_char.items()}
    other_n = {c: sum(r["kind"] != "live" and r["split"] == "train" and _file_side(r) == "left" for r in rs)
               for c, rs in by_char.items()}
    cap = min([n for n in live_n.values() if n] + list(other_n.values())) if any(live_n.values()) else 0
    for c, rs in by_char.items():
        live_train = sorted((r for r in rs if r["kind"] == "live" and r["split"] == "train"),
                            key=lambda r: (r["game"], r["id"]))
        drop = {id(r) for r in live_train[cap:]}
        by_char[c] = [r for r in rs if id(r) not in drop]
    if cap:
        print("live training rows per character: %d (capped; before %s)" % (cap, live_n))
    for char, recs in sorted(by_char.items()):
        base = os.path.join(ROOT, char)
        # model frames: raw captures (images/) with the HUD blanked, in frames/; mirrored ones flipped whole
        recs = [dict(r, images=[p.replace("images/", "frames/") for p in r["images"]]) for r in recs]
        # training: the left-side rows (their mirror is the right side) and every live row (live play has both sides)
        train = [r for r in recs if r["split"] == "train" and (_file_side(r) == "left" or r["kind"] == "live")]
        mirrored = [dict(m, images=[p.replace("frames/", "frames/mirror_") for p in m["images"]])
                    for m in map(mirror_record, train)]
        problems += _write_frames(base, recs, mirrored)
        tests = {s: [r for r in recs if r["split"] == "test" and _file_side(r) == s] for s in ("left", "right")}
        # laya-vision's loader (laya.vlm_train.jsonl_example) needs a question dict and an int label
        train, mirrored = _laya(train), _laya(mirrored)
        tests = {s: _laya(rs) for s, rs in tests.items()}
        write_jsonl(os.path.join(base, "train.jsonl"), train + mirrored)   # what scripts/train.py reads
        write_jsonl(os.path.join(base, "train_real.jsonl"), train)
        write_jsonl(os.path.join(base, "train_mirrored.jsonl"), mirrored)
        for s, rs in tests.items():
            write_jsonl(os.path.join(base, "test_real_%s.jsonl" % s), rs)
        # gates: every (action, range) combination filled; every attack came out
        count = collections.Counter((r["action"], r["range"], r["split"], r["side"]) for r in recs
                                    if r["kind"] != "defense")
        for action in static_actions(char):
            for rng in RANGES:
                need_sides = (("train", "left", MIN_TRAIN), ("test", "left", MIN_TEST)) + (
                    (("test", "right", MIN_TEST),) if args.right_test else ())
                for split, side, need in need_sides:
                    if count[(action, rng, split, side)] < need:
                        problems.append("%s %s %s %s/%s: %d < %d" % (char, action, rng, split, side,
                                                                       count[(action, rng, split, side)], need))
        problems += ["%s %s never came out (%s)" % (char, r["action"], r["id"]) for r in recs if not r["executed"]]
        live = [r for r in recs if r["kind"] == "live"]
        if live:     # live play: every move tried in training, and a held-out share
            lc = collections.Counter(r["action"] for r in live if r["split"] == "train")
            problems += ["%s live: %s has %d training rows < %d" % (char, a, lc[a], MIN_LIVE) for a in actions(char)
                         if lc[a] < MIN_LIVE]
            if not [r for r in live if r["split"] == "test"]:
                problems.append("%s live: no held-out games" % char)
        # blocks: every answer to every probe, on the training side and on both test sides
        dcount = collections.Counter((r["action"], r["probe"], r["split"], _file_side(r)) for r in recs
                                     if r["kind"] == "defense")
        for answer in D.ANSWERS:
            for probe in ("s.hk", "c.mk", "sweep", "jump_in"):
                for split, side, need in (("train", "left", MIN_DEF_TRAIN), ("test", "left", MIN_DEF_TEST)) + (
                        (("test", "right", MIN_DEF_TEST),) if args.right_test else ()):
                    if dcount[(answer, probe, split, side)] < need:
                        problems.append("%s %s vs %s %s/%s: %d < %d" % (char, answer, probe, split, side,
                                                                      dcount[(answer, probe, split, side)], need))
        table = collections.defaultdict(collections.Counter)
        for r in train:
            table[(r["action"], r["range"])][r["outcome"]] += 1
        stats = {"train": len(train), "train_mirrored": len(mirrored),
                 "test_real_left": len(tests["left"]), "test_real_right": len(tests["right"]),
                 "outcomes_train": {"%s@%s" % k: dict(v) for k, v in sorted(table.items())}}
        with open(os.path.join(base, "stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print("%-7s train %d (+%d mirrored)  test real left %d / right %d" % (
            char, len(train), len(mirrored), len(tests["left"]), len(tests["right"])))
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0
