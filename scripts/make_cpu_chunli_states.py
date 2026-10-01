"""Savestates for the CPU Chun-Li games (docs/prereg_movement_data.md, "Addition: CPU Chun-Li games"): arcade mode,
player 1 = each of the 7 other characters, the CPU's first opponent = Chun-Li (player 2). sf2.emu.boot.boot searches
the idle count before the jab until the CPU picks Chun-Li (pick_opponent); for a character Chun-Li is never the first
opponent of, ``--via`` boots another character against Chun-Li, loses, and continues as the wanted one (the continue
keeps the opponent).

    python scripts/make_cpu_chunli_states.py [--chars blanka,dhalsim,...] [--frames out/cpu_chunli_states]

Writes states/p1_<char>_vs_chunli.state (never overwrites one that exists) and checks it: the RAM's p1_char /
p2_char after loading it must be (<char>, Chun-Li), and <frames>/<char>.png is the frame shown 30 idle frames in (for a
look). Exit 1 if any character fails.
"""
import argparse
import os
import sys

import _path  # noqa: F401
from PIL import Image

from sf2.config import PORTS
from sf2.data.action_collect_io import char_problems
from sf2.emu.boot import boot
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, VARS
from sf2.eval.runner import savestate
from sf2.vocab import IDS

CPU = "chunli"
P1_CHARS = ("blanka", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
VIA = ("ryu", "ken", "guile", "blanka", "dhalsim", "honda", "zangief")


def make(b, me: str, path: str) -> str:
    """Boot ``me`` vs CPU Chun-Li and save the state at ``path``; how it got there."""
    try:
        state, how = boot(b, None, me, CPU, ram_map=VARS), "first opponent"
    except ValueError:
        for via in VIA:
            if via == me:
                continue
            try:
                state, how = boot(b, None, me, CPU, ram_map=VARS, via=via), "via %s" % via
                break
            except (ValueError, RuntimeError):
                continue
        else:
            raise RuntimeError("no route to %s vs CPU Chun-Li" % me)
    with open(path, "wb") as f:
        f.write(state)
    return how


def check(b, me: str, path: str, frames: str) -> list:
    b.set_vars(VARS)
    with open(path, "rb") as f:
        rows = [dict(zip(NAMES, x)) for x in b.load_state(f.read()).rams]
    obs = b.run([[]] * 30, caps=[30])
    rows += [dict(zip(NAMES, x)) for x in obs.rams]
    os.makedirs(frames, exist_ok=True)
    Image.fromarray(obs.images[30]).save(os.path.join(frames, "%s.png" % me))
    return char_problems(rows, {1: IDS[me], 2: IDS[CPU]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", default=",".join(P1_CHARS))
    ap.add_argument("--frames", default=os.path.join("out", "cpu_chunli_states"))
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--port", type=int, default=PORTS["movement"][0])
    args = ap.parse_args(argv)
    chars = args.chars.split(",")
    bad = sorted(set(chars) - set(P1_CHARS))
    if bad:
        raise SystemExit("unknown player-1 characters %s" % bad)
    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom))
    failed = []
    try:
        b.set_capture("raw")
        for me in chars:
            path = savestate(me, CPU)
            how = "exists"
            if not os.path.exists(path):
                how = make(b, me, path)
            probs = check(b, me, path, args.frames)
            print("%-8s %s: %s" % (me, path, "; ".join(probs) if probs else "OK (%s)" % how), flush=True)
            if probs:
                failed.append(me)
    finally:
        b.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
