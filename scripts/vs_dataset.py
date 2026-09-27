"""Stage-1 dataset (still opponent) per character, into test_data/<char>/. See sf2/vs_sweep.py for the design.

    # 1. collect shards (each its own headless Mesen; run them in parallel on different ports)
    python scripts/vs_dataset.py collect --p1 ryu --p2 chunli --who 1 --range close --port 48001   # Ryu, left
    python scripts/vs_dataset.py collect --p1 ryu --p2 chunli --who 2 --port 48004                  # Chun-Li, right
    # 2. merge, mirror, write per character
    python scripts/vs_dataset.py build
    # or both steps, all shards in parallel (the usual way):
    python scripts/vs_dataset.py run --chars ryu,chunli              # needs $SF2_ROM
    python scripts/vs_dataset.py run --chars ryu,chunli --no-right-test  # skip the real right-side check set
    python scripts/vs_dataset.py run --pairs ken:guile,honda:blanka,zangief:dhalsim   # 3 pairs, 36 jobs at once

A fighter on the left (``--who 1``) gives train + test; on the right (``--who 2``) test only (real frames for the
mirroring check). ``build`` exits 1 if any (action, range) combination is short of examples, or if an attack
never came out (a harness bug, not a data point).
"""
import argparse
import collections
import json
import os
import sys
from typing import Dict, List

import numpy as np

import _path  # noqa: F401
from sf2.config import PAD
from sf2.dataset import read, save_png, write_jsonl
from sf2.headless import launch_argv
from sf2.mesen import MesenBridge
from sf2.vs import boot_vs, gap_state, record, view
from sf2.vs_moves import CONDS
from sf2.frames import HUD_ROWS, mirror_frame, model_frame
from sf2.vs_sweep import (GAPS, LEAD, MOVEMENT, OUTCOMES, POSTURES, PREV_GAP, RANGES, STAGE1_POSTURES, actions, mirror_record, note,
                          outcome, outcome_question, range_of, split_of)

ROOT = "test_data"
MIN_TRAIN, MIN_TEST = 14, 6   # per (action, range): 7 gaps x 2 postures train, 3 x 2 test


def collect(args) -> int:
    me, opp = (args.p1, args.p2) if args.who == 1 else (args.p2, args.p1)
    side = "left" if args.who == 1 else "right"
    splits = {"train", "test"} if args.who == 1 else {"test"}
    ranges = [args.range] if args.range else list(RANGES)
    img_dir = os.path.join(ROOT, me, "images")
    shard_dir = os.path.join(ROOT, "_shards")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)
    shard = os.path.join(shard_dir, "%s_%s_%s.jsonl" % (me, side, "_".join(ranges)))
    acts = actions(me)

    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom, args.mesen))
    recs: List[Dict] = []
    skipped = []
    try:
        b.set_capture("raw")
        start = boot_vs(b, args.p1, args.p2)
        # power-on is not bit-exact (the reset's sub-frame timing leaves the RNG and animation counters differing:
        # damage and some frames vary between boots), so every record names the boot it came from, for replay
        boot = os.path.splitext(os.path.basename(shard))[0] + ".start.state"
        with open(os.path.join(shard_dir, boot), "wb") as f:
            f.write(start)
        for rng in ranges:
            for gi, target in enumerate(GAPS[rng]):
                split = split_of(gi)
                if split not in splits:
                    continue
                state, got = gap_state(b, start, target)
                if range_of(got) != rng:
                    skipped.append((rng, target, got))
                    continue
                for posture, hold in POSTURES.items():
                    key = "%s_%s_%d_%s" % (side, rng, gi, posture)
                    paths = ["images/%s_prev.png" % key, "images/%s_now.png" % key]
                    for n, (action, steps) in enumerate(acts.items()):
                        b.load_state(state)
                        take = record(b, args.who, (((), LEAD),) + steps, (), 240, CONDS, PAD,
                                      shots={LEAD - PREV_GAP, LEAD} if n == 0 else set(), d_hold=hold)
                        if n == 0:
                            for p, k in zip(paths, (LEAD - PREV_GAP, LEAD)):
                                save_png(take.images[k], os.path.join(ROOT, me, p))
                        rows = [view(r, args.who) for r in take.rows]
                        now = rows[LEAD]
                        pin = take.p1 if args.who == 1 else take.p2
                        nframes = sum(s[1] for s in steps)
                        rec = {"id": "%s-%s-%s" % (me, key, action), "char": me, "opp": opp, "side": side,
                               "facing": "right" if side == "left" else "left", "range": rng, "gap": abs(now["d_x"] -
                               now["a_x"]), "gap_index": gi, "dx": now["d_x"] - now["a_x"], "posture": posture,
                               "action": action, "kind": "movement" if action in MOVEMENT else "attack",
                               "buttons": pin[LEAD + 1: LEAD + 1 + nframes], "images": paths,
                               "state_text": note(me, opp, now, side),
                               "split": split, "mirrored": False, "source": "real", "boot": "_shards/" + boot}
                        rec.update(outcome(rows[LEAD:], action))
                        recs.append(rec)
                print("%s %s %-5s gap %3d (target %3d) %-5s: %d examples" % (me, side, rng, got, target, split,
                                                                               len(recs)), flush=True)
    finally:
        b.close()
    write_jsonl(shard, recs)
    print("wrote %d examples to %s; gaps skipped (out of range): %s" % (len(recs), shard, skipped or "none"))
    return 0


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


def build(args) -> int:
    shards = [os.path.join(ROOT, "_shards", f) for f in sorted(os.listdir(os.path.join(ROOT, "_shards")))
              if f.endswith(".jsonl")]
    by_char: Dict[str, List[Dict]] = collections.defaultdict(list)
    for s in shards:
        for r in read(s):
            if r["posture"] in STAGE1_POSTURES:
                by_char[r["char"]].append(r)
    problems = []
    for char, recs in sorted(by_char.items()):
        base = os.path.join(ROOT, char)
        # model frames: raw captures (images/) with the HUD blanked, in frames/; mirrored ones flipped whole
        recs = [dict(r, images=[p.replace("images/", "frames/") for p in r["images"]]) for r in recs]
        train = [r for r in recs if r["split"] == "train" and r["side"] == "left"]
        mirrored = [dict(m, images=[p.replace("frames/", "frames/mirror_") for p in m["images"]])
                    for m in map(mirror_record, train)]
        problems += _write_frames(base, recs, mirrored)
        tests = {s: [r for r in recs if r["split"] == "test" and r["side"] == s] for s in ("left", "right")}
        # laya-vision's loader (laya.vlm_train.jsonl_example) needs a question dict and an int label
        train, mirrored = _laya(train), _laya(mirrored)
        tests = {s: _laya(rs) for s, rs in tests.items()}
        write_jsonl(os.path.join(base, "train.jsonl"), train + mirrored)   # what scripts/train.py reads
        write_jsonl(os.path.join(base, "train_real.jsonl"), train)
        write_jsonl(os.path.join(base, "train_mirrored.jsonl"), mirrored)
        for s, rs in tests.items():
            write_jsonl(os.path.join(base, "test_real_%s.jsonl" % s), rs)
        # gates: every (action, range) combination filled; every attack came out
        count = collections.Counter((r["action"], r["range"], r["split"], r["side"]) for r in recs)
        for action in actions(char):
            for rng in RANGES:
                need_sides = (("train", "left", MIN_TRAIN), ("test", "left", MIN_TEST)) + (
                    (("test", "right", MIN_TEST),) if args.right_test else ())
                for split, side, need in need_sides:
                    if count[(action, rng, split, side)] < need:
                        problems.append("%s %s %s %s/%s: %d < %d" % (char, action, rng, split, side,
                                                                       count[(action, rng, split, side)], need))
        problems += ["%s %s never came out (%s)" % (char, r["action"], r["id"]) for r in recs if not r["executed"]]
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


def run(args) -> int:
    """Everything, in parallel: per character, one collect per range on the left (train + test) and, with
    --right-test, one per range on the right (real test frames, the mirroring check); then build. Each character's
    still opponent is --dummy, or else the next character in --chars (the last one's is the first)."""
    import subprocess

    if args.pairs:   # "a:b,c:d": a's dummy is b and b's is a
        who_vs = [(x, y) for pair in args.pairs.split(",") for a, b in [pair.split(":")] for x, y in ((a, b), (b, a))]
    else:
        chars = args.chars.split(",")
        who_vs = [(me, args.dummy or chars[(i + 1) % len(chars)]) for i, me in enumerate(chars)]
    jobs, port = [], args.base_port
    os.makedirs(os.path.join("logs", "dataset"), exist_ok=True)
    for me, dummy in who_vs:
        if dummy == me:
            raise SystemExit("%s needs a different dummy (--dummy, or two or more --chars)" % me)
        for who in (1, 2) if args.right_test else (1,):
            p1, p2 = (me, dummy) if who == 1 else (dummy, me)
            for rng in RANGES:
                log = os.path.join("logs", "dataset", "%s_%s_%s.log" % (me, "left" if who == 1 else "right", rng))
                cmd = [sys.executable, os.path.abspath(__file__), "collect", "--p1", p1, "--p2", p2, "--who",
                       str(who), "--range", rng, "--port", str(port)] + (["--rom", args.rom] if args.rom else [])
                jobs.append((log, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
                port += 1
    print("%d collect jobs running (logs in logs/dataset/)" % len(jobs), flush=True)
    failed = [log for log, proc in jobs if proc.wait() != 0]
    for log in failed:
        print("FAILED collect, see", log)
    if failed:
        return 1
    return build(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--p1", required=True)
    c.add_argument("--p2", required=True)
    c.add_argument("--who", type=int, choices=(1, 2), required=True)
    c.add_argument("--range", choices=RANGES)
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    c.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    bd = sub.add_parser("build")
    r = sub.add_parser("run", help="collect every shard in parallel, then build")
    r.add_argument("--chars", default="ryu,chunli", help="comma-separated; each gets its own test_data/<char>/")
    r.add_argument("--pairs", help="a:b,c:d - each pair are each other's still opponent (overrides --chars)")
    r.add_argument("--dummy", help="the still opponent for every character (default: the next in --chars)")
    r.add_argument("--base-port", type=int, default=48001)
    r.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    for p in (bd, r):
        p.add_argument("--no-right-test", dest="right_test", action="store_false",
                       help="skip the real right-side test set (the one-time mirroring check)")
    args = ap.parse_args()
    return {"collect": collect, "build": build, "run": run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
