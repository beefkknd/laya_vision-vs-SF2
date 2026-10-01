"""Press every move of a character from a still VS BATTLE stance and record both fighters' whole structs
(0x0C00-0x0FFF) every frame: the ground truth for the action-ID probe (sf2.data.action_probe).

    python scripts/probe_action_moves.py --out <dir> --p1 chunli --p2 ken [--gaps 30,150]

For each gap (world px, sf2.emu.vs.gap_state) and each side (player 1 presses, then player 2 presses; the other stands
still), each action of sf2.data.vs_sweep.actions(char) plus the six neutral-jump normals is pressed from the same
savestate, then both idle until settled (or TAIL frames). Writes <out>/<p1>_vs_<p2>.npz: ``struct`` uint8
[rows, 1024], ``fields`` int32 [rows, len(NAMES)], ``take`` int32 [rows] (index into ``takes``), ``takes`` (JSON
list of {side, char, action, gap}), and images/<p1>_vs_<p2>/t<take>_k<NNN>.png every 4th frame (frame k's image is
captured after frame k, i.e. it shows row k + 1 of the take, which is where the RAM row lands after the frame).
"""
import argparse
import json
import os
import sys

import numpy as np

import _path  # noqa: F401
from sf2.config import PAD, PORTS
from sf2.data.vs_sweep import actions
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, VARS, boot_vs, gap_state, physical

BASE, SPAN = 0x0C00, 0x400
TAIL = 90            # idle frames after the inputs, at most
SETTLE = 12          # ... or this many in a row with the presser back in a neutral stance (state 0 / 2)
JUMP_NORMALS = {"j.%s" % b: ((("U",), 2), ((), 10), ((b,), 2), ((), 2)) for b in ("lp", "mp", "hp", "lk", "mk", "hk")}


def _steps(char):
    return dict(actions(char), **JUMP_NORMALS)


def _struct_vars():
    from sf2.emu.ram import Var
    return [Var("b%03x" % i, BASE + i, 1, False) for i in range(SPAN)]


def press(bridge, side: int, steps, every: int):
    """Rows of one take: inputs for ``side``, the other idle; then idle until settled. (rows, {k: image})."""
    obs = bridge.run([])
    rows, images = [obs.rams[-1]], {}
    r0 = dict(zip(NAMES, rows[0]))
    right = r0["p%d_x" % side] < r0["p%d_x" % (3 - side)]
    frames = [physical(t, right, PAD) for toks, n in steps for t in [toks] * n]
    calm, tail = 0, 0
    st = NAMES.index("p%d_state" % side)
    k = 0
    while k < len(frames) or (tail < TAIL and calm < SETTLE):
        pad = frames[k] if k < len(frames) else []
        mine, other = ([pad], [[]]) if side == 1 else ([[]], [pad])
        o = bridge.run(mine, caps=[1] if k % every == 0 else [], p2=other)
        rows.append(o.rams[-1])
        if 1 in o.images:
            images[k] = o.images[1]
        if k >= len(frames):
            tail += 1
            calm = calm + 1 if rows[-1][st] in (0, 2) else 0
        k += 1
    return rows, images


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p1", default="chunli")
    ap.add_argument("--p2", default="ken")
    ap.add_argument("--gaps", default="30,150")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--port", type=int, default=PORTS["movement"][0] + 12)
    args = ap.parse_args()
    tag = "%s_vs_%s" % (args.p1, args.p2)
    img_dir = os.path.join(args.out, "images", tag)
    os.makedirs(img_dir, exist_ok=True)
    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom))
    all_rows, take_of, takes = [], [], []
    try:
        b.set_capture("raw")
        start = boot_vs(b, args.p1, args.p2)
        b.set_vars(VARS + _struct_vars())
        for gap in map(int, args.gaps.split(",")):
            at, reached = gap_state(b, start, gap)
            for side, char in ((1, args.p1), (2, args.p2)):
                for name, steps in _steps(char).items():
                    b.load_state(at)
                    rows, images = press(b, side, steps, args.every)
                    t = len(takes)
                    takes.append({"side": side, "char": char, "action": name, "gap": reached})
                    for k, im in images.items():
                        Image.fromarray(im).save(os.path.join(img_dir, "t%03d_k%03d.png" % (t, k)), compress_level=1)
                    all_rows += rows
                    take_of += [t] * len(rows)
            print("gap %d (reached %d): %d takes" % (gap, reached, len(takes)), flush=True)
    finally:
        b.close()
    arr = np.asarray(all_rows, np.int64)
    nf = len(NAMES)
    np.savez_compressed(os.path.join(args.out, tag + ".npz"), struct=arr[:, nf:].astype(np.uint8),
                        fields=arr[:, :nf].astype(np.int32), names=np.asarray(NAMES),
                        take=np.asarray(take_of, np.int32), takes=json.dumps(takes))
    print("%s: %d takes, %d rows" % (tag, len(takes), len(all_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
