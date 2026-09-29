"""Replay check: re-record a random sample of test_data/ examples from a fresh power-on and compare with what is
stored. Proves the data is what the emulator really does (labels, measurements) and that every record points at
the right frames (raw captures compared byte for byte).

    python scripts/verify_replay.py                    # 40 records per character and side, all characters
    python scripts/verify_replay.py --chars ryu --n 100

One headless Mesen per (character, side), all in parallel. Exit 1 on any mismatch.
"""
import argparse
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np

import _path  # noqa: F401
from sf2.eval.runner import exit_on_sigterm, fan_out
from sf2.config import PAD
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import boot_vs, gap_state, record, view
from sf2.data.vs_moves import CONDS
from sf2.data.vs_sweep import GAPS, LEAD, POSTURES, PREV_GAP, actions, outcome

ROOT = "test_data"
COMPARE = ("gap", "outcome", "damage", "busy_frames", "travel", "thrown", "executed")


def load_img(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def sample(char: str, side: str, n: int, seed: int) -> List[Dict]:
    files = ["train_real", "test_real_left"] if side == "left" else ["test_real_right"]
    recs = [r for f in files for r in map(json.loads, open(os.path.join(ROOT, char, f + ".jsonl")))
            if r.get("kind") != "defense"]      # still-opponent rows; block rows are checked by their own ROM probe
    return random.Random("%s-%s-%d" % (char, side, seed)).sample(recs, min(n, len(recs)))


def replay(char: str, side: str, n: int, seed: int, port: int, rom: str) -> int:
    recs = sample(char, side, n, seed)
    who = 1 if side == "left" else 2
    opp = recs[0]["opp"]
    p1, p2 = (char, opp) if who == 1 else (opp, char)
    b = MesenBridge(port, launch=launch_argv(port, rom))
    bad = []
    try:
        b.set_capture("raw")
        fresh_boot = boot_vs(b, p1, p2)       # also proves the pair still boots
        states = {}
        for r in recs:
            # the record's own boot (power-on is not bit-exact); a record without one replays from a fresh boot
            start = open(os.path.join(ROOT, r["boot"]), "rb").read() if r.get("boot") else fresh_boot
            key = (r.get("boot"), r["range"], r["gap_index"])
            if key not in states:
                states[key] = gap_state(b, start, GAPS[r["range"]][r["gap_index"]])[0]
            b.load_state(states[key])
            take = record(b, who, (((), LEAD),) + actions(char)[r["action"]], (), 240, CONDS, PAD,
                          shots={LEAD - PREV_GAP, LEAD}, d_hold=POSTURES[r["posture"]])
            rows = [view(x, who) for x in take.rows]
            fresh = dict(outcome(rows[LEAD:], r["action"]), gap=abs(rows[LEAD]["d_x"] - rows[LEAD]["a_x"]))
            diff = {k: (r[k], fresh[k]) for k in COMPARE if r[k] != fresh[k]}
            for p, k in zip(r["images"], (LEAD - PREV_GAP, LEAD)):
                raw = os.path.join(ROOT, char, p.replace("frames/", "images/"))
                if not np.array_equal(load_img(raw), take.images[k]):
                    diff[p] = "raw frame differs"
            if diff:
                bad.append({"id": r["id"], "diff": diff})
    finally:
        b.close()
    print("%-8s %-5s %d replayed, %d mismatched%s" % (char, side, len(recs), len(bad),
                                                       "" if not bad else ": %s" % json.dumps(bad[:3])), flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-port", type=int, default=48401)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--one", nargs=3, metavar=("CHAR", "SIDE", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        c, s, p = args.one
        return replay(c, s, args.n, args.seed, int(p), args.rom)
    chars = args.chars.split(",") if args.chars else sorted(d for d in os.listdir(ROOT) if not d.startswith("_"))
    groups = [(c, s) for c in chars for s in ("left", "right")]
    cmds = [((c, s), [sys.executable, os.path.abspath(__file__), "--one", c, s, str(args.base_port + i), "--n",
                      str(args.n), "--seed", str(args.seed)] + (["--rom", args.rom] if args.rom else []))
            for i, (c, s) in enumerate(groups)]
    failed = fan_out(cmds, os.path.join("logs", "replay"))
    print("replay: %d of %d (character, side) groups match (logs/replay/)" % (len(groups) - len(failed), len(groups)))
    for c, s in failed:
        print("MISMATCH or error: %s %s, see logs/replay/%s_%s.log" % (c, s, c, s))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
