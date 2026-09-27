"""Audit test_data/<char>/ (scripts/vs_dataset.py output) before it is used for training: mechanical checks only,
every one a yes/no per record or per frame. Exit 1 if any check has a violation.

    python scripts/audit_dataset.py                 # every character in test_data/
    python scripts/audit_dataset.py --chars ryu,ken

Writes out/audit/audit.json: per character, per check, the number of violations and up to 5 examples.
"""
import argparse
import collections
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

import _path  # noqa: F401
from sf2.config import PAD
from sf2.frames import HUD_ROWS
from sf2.vs import physical
from sf2.vs_sweep import MOVEMENT, OUTCOMES, POSTURES, RANGES, TEST_INDEX, actions, outcome_question, range_of
from laya.vlm_train import jsonl_example

ROOT = "test_data"
FILES = ("train_real", "train_mirrored", "test_real_left", "test_real_right")
FIELDS = {"id": str, "char": str, "opp": str, "side": str, "facing": str, "range": str, "gap": int, "gap_index": int,
          "dx": int, "posture": str, "action": str, "kind": str, "buttons": list, "images": list, "state_text": str,
          "question": dict, "label": int, "split": str, "mirrored": bool, "source": str, "outcome": str, "thrown": bool,
          "executed": bool, "damage": int, "busy_frames": int, "travel": int}
PER_COMBO = {"train_real": 18, "test_real_left": 6, "test_real_right": 6}
_SWAP = {"left": "right", "right": "left"}


def set_root(root: str) -> None:
    global ROOT
    ROOT = root


class Audit:
    def __init__(self):
        self.bad: Dict[str, List[str]] = collections.defaultdict(list)
        self.ran: collections.Counter = collections.Counter()

    def check(self, name: str, ok: bool, where: str) -> None:
        self.ran[name] += 1
        if not ok:
            self.bad[name].append(where)


def load_img(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def expected_buttons(char: str, action: str, side: str) -> List[List[str]]:
    frames = [tok for tok, n in actions(char)[action] for _ in range(n)]
    return [physical(t, side == "left", PAD) for t in frames]


def note_fields(text: str) -> Dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", text))


def audit_record(a: Audit, char: str, f: str, r: Dict) -> None:
    rid = "%s:%s" % (f, r.get("id"))
    a.check("schema", all(isinstance(r.get(k), t) and not (t is int and isinstance(r.get(k), bool))
                          for k, t in FIELDS.items()), rid)
    if a.bad["schema"] and a.bad["schema"][-1] == rid:
        return
    want_side = "right" if f in ("train_mirrored", "test_real_right") else "left"
    a.check("side_matches_file", r["side"] == want_side and r["facing"] == _SWAP[want_side], rid)
    a.check("split_matches_file", r["split"] == ("train" if f.startswith("train") else "test"), rid)
    a.check("mirrored_flag", r["mirrored"] == (f == "train_mirrored"), rid)
    a.check("char", r["char"] == char and r["opp"] != char, rid)
    a.check("action_known", r["action"] in actions(char), rid)
    a.check("posture_known", r["posture"] in POSTURES, rid)
    a.check("gap_index_split", (r["gap_index"] in TEST_INDEX) == (r["split"] == "test"), rid)
    a.check("range_of_gap", r["range"] in RANGES and range_of(r["gap"]) == r["range"], rid)
    a.check("dx_is_gap_and_side", abs(r["dx"]) == r["gap"] and (r["dx"] > 0) == (r["side"] == "left"), rid)
    n = note_fields(r["state_text"])
    a.check("note_matches_fields", n.get("me") == char and n.get("opp") == r["opp"] and n.get("dist") == r["range"]
            and n.get("side") == r["side"] and n.get("dx") == "%+d" % r["dx"]
            and n.get("opp_crouch") == str(int(r["posture"] != "stand")), rid)
    a.check("question_is_the_play_question", r["question"] == outcome_question(r["action"]), rid)
    a.check("label_is_outcome", r["label"] == OUTCOMES.index(r["outcome"]), rid)
    ex = jsonl_example(r, os.path.join(ROOT, char), char)     # laya-vision's own loader
    a.check("laya_loads", ex is not None and ex["label"] == r["label"] and len(ex["target"]) == len(OUTCOMES)
            and ex["state"]["context"] == r["state_text"] and len(ex["state"]["images"]) == 2, rid)
    a.check("buttons_match_action", r["buttons"] == expected_buttons(char, r["action"], r["side"]), rid)
    a.check("kind", r["kind"] == ("movement" if r["action"] in MOVEMENT else "attack"), rid)
    a.check("movement_iff_none", (r["outcome"] == "none") == (r["action"] in MOVEMENT), rid)
    a.check("outcome_known", r["outcome"] in ("hit", "whiff", "blocked", "none"), rid)
    a.check("executed", r["executed"], rid)
    a.check("hit_has_damage", r["outcome"] != "hit" or r["damage"] > 0 or r["thrown"], rid)
    a.check("whiff_no_damage", r["outcome"] not in ("whiff", "none") or r["damage"] == 0, rid)
    a.check("thrown_is_hit", not r["thrown"] or r["outcome"] == "hit", rid)
    a.check("boot_state_exists", os.path.exists(os.path.join(ROOT, r.get("boot") or "-")), rid)
    a.check("images_are_frames", len(r["images"]) == 2 and all(p.startswith("frames/") for p in r["images"]), rid)


def audit_char(a: Audit, char: str) -> Dict[str, int]:
    base = os.path.join(ROOT, char)
    recs = {}
    for f in FILES:
        path = os.path.join(base, f + ".jsonl")
        a.check("file_exists", os.path.exists(path), path)
        recs[f] = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    for f, rs in recs.items():
        ids = [r.get("id") for r in rs]
        a.check("unique_ids", len(ids) == len(set(ids)), "%s/%s" % (char, f))
        for r in rs:
            audit_record(a, char, f, r)
    # coverage: every (action, range, posture) in each file; counts per (action, range)
    for f, need in PER_COMBO.items():
        c = collections.Counter((r["action"], r["range"]) for r in recs[f])
        cp = collections.Counter((r["action"], r["range"], r["posture"]) for r in recs[f])
        for act in actions(char):
            for rng in RANGES:
                a.check("combo_count", c[(act, rng)] >= need, "%s/%s %s@%s: %d" % (char, f, act, rng, c[(act, rng)]))
                for p in POSTURES:
                    a.check("posture_coverage", cp[(act, rng, p)] > 0, "%s/%s %s@%s/%s" % (char, f, act, rng, p))
    # no leakage: test frames never appear in train, test gaps never in train
    train_imgs = {p for f in ("train_real", "train_mirrored") for r in recs[f] for p in r["images"]}
    for f in ("test_real_left", "test_real_right"):
        for r in recs[f]:
            a.check("no_image_leak", not set(r["images"]) & train_imgs, "%s:%s" % (f, r["id"]))
    # the mirrored set is exactly the train set, flipped
    src = {r["id"]: r for r in recs["train_real"]}
    a.check("mirror_count", len(recs["train_mirrored"]) == len(recs["train_real"]), char)
    # train.jsonl (what scripts/train.py reads) is exactly real + mirrored
    tpath = os.path.join(base, "train.jsonl")
    a.check("train_is_real_plus_mirrored", os.path.exists(tpath) and [json.loads(x) for x in open(tpath)]
            == recs["train_real"] + recs["train_mirrored"], char)
    same = ("action", "range", "gap", "gap_index", "posture", "kind", "outcome", "damage", "thrown", "busy_frames",
            "travel", "split", "opp", "question", "label")
    for m in recs["train_mirrored"]:
        s = src.get(m["id"][:-2]) if m["id"].endswith("-m") else None
        a.check("mirror_has_source", s is not None, m["id"])
        if s is None:
            continue
        a.check("mirror_same_labels", all(m[k] == s[k] for k in same), m["id"])
        a.check("mirror_side_fields", m["dx"] == -s["dx"] and m["side"] == _SWAP[s["side"]]
                and m["buttons"] == [[_SWAP.get(b, b) for b in fr] for fr in s["buttons"]], m["id"])
    # frames: exist, 256x224 RGB, HUD black; mirrors are exact flips; prev differs from now
    cache: Dict[str, np.ndarray] = {}
    for f, rs in recs.items():
        for r in rs:
            for p in r["images"]:
                if p in cache:
                    continue
                full = os.path.join(base, p)
                if not os.path.exists(full):
                    a.check("frame_exists", False, full)
                    continue
                a.check("frame_exists", True, full)
                try:
                    img = load_img(full)
                except Exception as e:  # truncated / corrupt file (e.g. a build killed mid-write)
                    a.check("frame_readable", False, "%s: %s" % (full, type(e).__name__))
                    continue
                a.check("frame_readable", True, full)
                cache[p] = img
                a.check("frame_shape", img.shape == (224, 256, 3), full)
                a.check("hud_black", not img[:HUD_ROWS].any(), full)
                a.check("frame_not_blank", img[HUD_ROWS:].std() > 5, full)
    for m in recs["train_mirrored"]:
        s = src.get(m["id"][:-2])
        if s is None:
            continue
        for mp, sp in zip(m["images"], s["images"]):
            if mp in cache and sp in cache:
                a.check("mirror_exact_flip", np.array_equal(cache[mp], cache[sp][:, ::-1]), mp)
    return {f: len(rs) for f, rs in recs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", help="comma-separated (default: every dir in test_data/ but _shards)")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="out/audit/audit.json")
    args = ap.parse_args()
    set_root(args.root)
    chars = args.chars.split(",") if args.chars else sorted(d for d in os.listdir(ROOT) if not d.startswith("_"))
    report = {}
    failed = False
    for char in chars:
        a = Audit()
        counts = audit_char(a, char)
        viol = {k: len(v) for k, v in a.bad.items() if v}
        failed |= bool(viol)
        report[char] = {"counts": counts, "checks_run": dict(a.ran), "violations": viol,
                        "examples": {k: v[:5] for k, v in a.bad.items() if v}}
        print("%-8s %s  checks %d  %s" % (char, counts, sum(a.ran.values()),
                                          "OK" if not viol else "VIOLATIONS %s" % viol))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
