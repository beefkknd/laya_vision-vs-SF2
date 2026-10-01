"""Savestates for the movement pairs (docs/prereg_movement_pairs.md, "Fallback"): arcade mode, player 1 = A (the
directed player), the CPU's opponent = B (player 2), for all 56 ordered pairs of the 8 characters. Route 1 (as
scripts/make_cpu_chunli_states.py): sf2.emu.boot.boot searches the idle count before the jab until the CPU picks B
first. Route 2, when B is never A's first opponent (Guile is nobody's, 2026-10-01; the "via" route of
make_cpu_chunli_states needs B as some character's first opponent): pick A, and DECIDED frames after the jab, when the
CPU's pick is in player 2's character byte (0x0ED1), write B there; the VS screen and the fight then load B (the stage
stays the original opponent's). Checked as every state is: RAM p1_char / p2_char and a frame to look at.

    python scripts/make_pair_states.py [--p1 ryu,ken] [--p2 ...] [--port N] [--frames out/pair_states]

Writes states/p1_<A>_vs_<B>.state (never overwrites one that exists) and checks every one, new or old: the RAM's
p1_char / p2_char after loading it and on 30 idle frames must be (A, B); <frames>/<A>_vs_<B>.png is the frame shown
30 idle frames in (for a look). Exit 1 if any pair fails.
"""
import argparse
import os
import sys

import _path  # noqa: F401
from PIL import Image

from sf2.config import PORTS
from sf2.data.action_collect_io import char_problems
from sf2.emu.boot import BOOT, DECIDED, P2_CHAR, _to_fight, boot, cursor_steps, frames_of
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, VARS
from sf2.eval.runner import savestate
from sf2.vocab import IDS

CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")


def ordered_pairs(p1s=CHARS, p2s=CHARS):
    return [(a, b) for a in p1s for b in p2s if a != b]


def set_opponent(b, me: str, opp: str) -> bytes:
    """Route 2: pick ``me`` (any first opponent), write ``opp`` into player 2's character byte once the CPU has
    picked, and go on to the first controllable frame of round 1."""
    b.reset()
    b.run(frames_of(BOOT[:5]))
    b.run(frames_of(cursor_steps(me) + [((), 10), (("y",), 2), ((), DECIDED)]))
    b.poke({P2_CHAR: IDS[opp]})
    return _to_fight(b, None, VARS)


def make(b, me: str, opp: str, path: str) -> str:
    """Boot ``me`` vs CPU ``opp`` and save the state at ``path``; how it got there."""
    try:
        state, how = boot(b, None, me, opp, ram_map=VARS), "first opponent"
    except ValueError:
        state, how = set_opponent(b, me, opp), "player 2 byte set"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(state)
    os.replace(tmp, path)
    return how


def check(b, me: str, opp: str, path: str, frames: str) -> list:
    b.set_vars(VARS)
    with open(path, "rb") as f:
        rows = [dict(zip(NAMES, x)) for x in b.load_state(f.read()).rams]
    obs = b.run([[]] * 30, caps=[30])
    rows += [dict(zip(NAMES, x)) for x in obs.rams]
    os.makedirs(frames, exist_ok=True)
    Image.fromarray(obs.images[30]).save(os.path.join(frames, "%s_vs_%s.png" % (me, opp)))
    return char_problems(rows, {1: IDS[me], 2: IDS[opp]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default=",".join(CHARS))
    ap.add_argument("--p2", default=",".join(CHARS))
    ap.add_argument("--frames", default=os.path.join("out", "pair_states"))
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--port", type=int, default=PORTS["pairs"][0])
    args = ap.parse_args(argv)
    p1s, p2s = args.p1.split(","), args.p2.split(",")
    bad = sorted(set(p1s + p2s) - set(CHARS))
    if bad:
        raise SystemExit("unknown characters %s" % bad)
    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom))
    failed = []
    try:
        b.set_capture("raw")
        for me, opp in ordered_pairs(p1s, p2s):
            path = savestate(me, opp)
            how = "exists"
            try:
                if not os.path.exists(path):
                    how = make(b, me, opp, path)
                probs = check(b, me, opp, path, args.frames)
            except (ValueError, RuntimeError) as e:
                probs = [str(e)]
            print("%-8s vs %-8s %s: %s" % (me, opp, path, "; ".join(probs) if probs else "OK (%s)" % how),
                  flush=True)
            if probs:
                failed.append((me, opp))
    finally:
        b.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
