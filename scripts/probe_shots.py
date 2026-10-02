"""Probe (round 3, docs/prereg_movement_finetunes.md): is a projectile DRAWN on every frame its shot slot is on? Plays
2P versus from the pair's savestate, player 1 presses only its projectile word (then idles), player 2 idles; saves
EVERY captured frame (hud_frame, 256 px) from a few frames before each spawn to a few after the slot clears, with the
RAM rows. Scratch output only; no dataset is written.

    python scripts/probe_shots.py --out <scratch dir> --pairs ryu:ken,guile:dhalsim [--flights 3]
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.data import movement_collect as C
from sf2.data import pairs_collect as PC
from sf2.data.action_codes import EXTRA_NAMES, EXTRA_VARS
from sf2.data.movement_collect_io import ImageSaver
from sf2.emu.ram import Var
from sf2.emu.vs import NAMES, VARS

STRUCT = [Var("s1_%02x" % i, 0x1000 + i, 1, False) for i in range(0x50)] + [Var("s2_%02x" % i, 0x1050 + i, 1, False) for i in range(0x50)] + [
    Var("g_%04x" % a, a, 1, False) for a in (0x0040, 0x0042, 0x0044, 0x0046)]
from sf2.eval.runner import open_fight
from make_vs_pair_states import vs_state

WORD = {"ryu": "hadoken", "ken": "hadoken", "guile": "sonic_boom", "dhalsim": "yoga_fire"}
PAD_BEFORE, PAD_AFTER = 4, 4


def probe(a, b, port, out, flights, rom, slot=1):
    rows, imgs = [], {}
    save = ImageSaver(os.path.join(out, "%s_vs_%s" % (a, b)), 0)
    with open_fight(a, b, port, rom, state=vs_state(a, b)) as (br, state):
        br.set_vars(VARS + EXTRA_VARS + STRUCT)
        proxy = C.CapturingBridge(br, NAMES + EXTRA_NAMES + [v.name for v in STRUCT])

        def sink(r, im):
            k = len(rows)
            rows.append(r)
            if im is not None:
                imgs[k] = im
                for j in [j for j in imgs if j < k - 200]:
                    del imgs[j]
        proxy.sink = sink
        proxy.load_state(state)
        proxy.run([[]] * 10)
        spans = []
        while len(spans) < flights and len(rows) < 6000:
            r = rows[-1]
            me = a if slot == 1 else b
            if PC.can_act(r, slot) and not r["shot%d" % slot]:
                fr = PC.press_frames(me, WORD[me], r, slot)
                proxy.run(fr if slot == 1 else [[]] * len(fr), p2=None if slot == 1 else fr)
            proxy.run([[]] * 4)
            last = spans[-1][1] if spans else -1
            on = [k for k in range(last + 1, len(rows)) if rows[k]["shot%d" % slot]]
            # a finished flight: on, then off for PAD_AFTER rows
            if on and not any(rows[k]["shot%d" % slot] for k in range(len(rows) - PAD_AFTER, len(rows))):
                s = on[0]
                e = s
                while rows[e + 1]["shot%d" % slot]:
                    e += 1
                if True:
                    spans.append((s, e))
                    for k in range(s - PAD_BEFORE, e + PAD_AFTER + 2):
                        if k in imgs:
                            save(k, imgs[k])
    with open(os.path.join(out, "%s_vs_%s.json" % (a, b)), "w") as f:
        json.dump({"spans": spans, "rows": {k: rows[k] for s, e in spans for k in range(s - PAD_BEFORE - 1,
                                                                                         min(e + PAD_AFTER + 2, len(rows)))}}, f)
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="ryu:ken,guile:dhalsim")
    ap.add_argument("--flights", type=int, default=3)
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for i, item in enumerate(args.pairs.split(",")):
        a, b = item.split(":")
        print(a, b, probe(a, b, PORTS["pairs"][0] + 60 + i, args.out, args.flights, args.rom, args.slot), flush=True)


if __name__ == "__main__":
    sys.exit(main())
