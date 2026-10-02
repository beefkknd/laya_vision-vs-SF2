"""Savestates for Plan B of the movement pairs (docs/prereg_movement_pairs.md, "Plan B"): 2-player VS BATTLE, BOTH
controllers ours, A = player 1 (left), B = player 2 (right), for all 56 ordered pairs. sf2.emu.vs.boot_vs: title,
V.S. BATTLE, both cursors walked (cursor_plans), both jab, handicap START, the first frame both answer the stick.

    python scripts/make_vs_pair_states.py [--p1 ryu,ken] [--p2 ...] [--port N] [--frames out/vs_pair_states]

Writes states/vs_<A>_vs_<B>.state (never overwrites one that exists) and checks every one, new or old (``checks``):
  chars     RAM p1_char (0x0CD1) / p2_char (0x0ED1) == (A, B) on load and on every one of the idle frames
  no_cpu    IDLE frames with both pads empty: neither x moves, both state bytes stay 0 (nobody drives either side)
  p1_only   P1 holds forward (toward P2) STEP frames, P2 empty: P1's x moves toward, P2's x does not move
  p2_only   the same with the roles swapped
<frames>/<A>_vs_<B>.png is the frame 30 idle frames in (for a look). One JSON line per pair on stdout ("RESULT ");
exit 1 if any pair fails.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from PIL import Image

from sf2.config import PORTS
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import VARS, boot_vs, rows_of
from sf2.vocab import IDS

CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
IDLE, STEP, MIN_STEP_PX = 300, 20, 8


def vs_state(a: str, b: str) -> str:
    return os.path.join("states", "vs_%s_vs_%s.state" % (a, b))


def ordered_pairs(p1s=CHARS, p2s=CHARS):
    return [(a, b) for a in p1s for b in p2s if a != b]


def drive(bridge, state: bytes, p1: List[str], p2: List[str], n: int) -> List[Dict[str, int]]:
    bridge.load_state(state)
    return rows_of(bridge.run([p1] * n, p2=[p2] * n))


def checks(bridge, state: bytes, a: str, b: str) -> Dict[str, str]:
    """name -> "" (ok) or what failed; pure reads of the state, nothing written."""
    out = {}
    first = rows_of(bridge.load_state(state))[-1]
    idle = drive(bridge, state, [], [], IDLE)
    rows = [first] + idle
    bad = [k for k, r in enumerate(rows) if (r["p1_char"], r["p2_char"]) != (IDS[a], IDS[b])]
    out["chars"] = "" if not bad else "row %d holds %s" % (bad[0], (rows[bad[0]]["p1_char"], rows[bad[0]]["p2_char"]))
    moved = {p: {r["p%d_x" % p] for r in rows} for p in (1, 2)}
    states = {p: {r["p%d_state" % p] for r in rows} for p in (1, 2)}
    out["no_cpu"] = "" if all(len(moved[p]) == 1 and states[p] == {0} for p in (1, 2)) else \
        "idle: x %s, states %s" % (moved, states)
    for me in (1, 2):
        him = 3 - me
        right = first["p%d_x" % me] < first["p%d_x" % him]
        pad = ["right" if right else "left"]
        r = drive(bridge, state, pad if me == 1 else [], pad if me == 2 else [], STEP)
        d_me = (r[-1]["p%d_x" % me] - first["p%d_x" % me]) * (1 if right else -1)
        d_him = r[-1]["p%d_x" % him] - first["p%d_x" % him]
        ok = d_me >= MIN_STEP_PX and d_him == 0
        out["p%d_only" % me] = "" if ok else "P%d toward %+d px, P%d moved %+d px" % (me, d_me, him, d_him)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default=",".join(CHARS))
    ap.add_argument("--p2", default=",".join(CHARS))
    ap.add_argument("--frames", default=os.path.join("out", "vs_pair_states"))
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--port", type=int, default=PORTS["pairs"][0])
    args = ap.parse_args(argv)
    p1s, p2s = args.p1.split(","), args.p2.split(",")
    bad = sorted(set(p1s + p2s) - set(CHARS))
    if bad:
        raise SystemExit("unknown characters %s" % bad)
    os.makedirs(args.frames, exist_ok=True)
    br = MesenBridge(args.port, launch=launch_argv(args.port, args.rom))
    failed = []
    try:
        br.set_capture("raw")
        for a, b in ordered_pairs(p1s, p2s):
            path, how = vs_state(a, b), "exists"
            try:
                if not os.path.exists(path):
                    state = boot_vs(br, a, b)
                    with open(path + ".tmp", "wb") as f:
                        f.write(state)
                    os.replace(path + ".tmp", path)
                    how = "booted"
                with open(path, "rb") as f:
                    state = f.read()
                br.set_vars(VARS)
                res = checks(br, state, a, b)
                br.load_state(state)
                obs = br.run([[]] * 30, caps=[30])
                Image.fromarray(obs.images[30]).save(os.path.join(args.frames, "%s_vs_%s.png" % (a, b)))
            except (ValueError, RuntimeError) as e:
                res = {"error": str(e)}
            ok = not any(res.values())
            print("RESULT " + json.dumps({"pair": [a, b], "state": path, "how": how, "ok": ok, "checks": res}),
                  flush=True)
            if not ok:
                failed.append((a, b))
    finally:
        br.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
