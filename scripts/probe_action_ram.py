"""Record both fighters' whole structs (0x0C00-0x0FFF) every frame of a few games, for the action-ID probe
(docs/prereg_movement_data.md, last section: "<actor> act<NN> stg<1-3>"). Probe only: no pairs, no dataset.

    python scripts/probe_action_ram.py --out <dir> --me chunli --opps ken,ryu --games 2 [--every 1]

Writes <out>/<me>_vs_<opp>_g<game>.npz: ``struct`` uint8 [rows, 1024] (row k = WRAM 0x0C00 + i before frame k of the
round's stream, same stream as the movement collector) and ``fields`` int32 [rows, len(NAMES)] (sf2.emu.vs.NAMES);
and images/<me>_vs_<opp>_g<game>/k<NNNNN>.png every ``--every`` captures (capture k shows row k - 1, perception.LAG).
Analysis: sf2.data.action_probe.
"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.data.movement_collect import CapturingBridge
from sf2.emu.ram import Var
from sf2.emu.vs import NAMES, VARS
from sf2.eval.runner import exit_on_sigterm, open_fight

BASE, SPAN = 0x0C00, 0x400
STRUCT = [Var("b%03x" % i, BASE + i, 1, False) for i in range(SPAN)]


def play_games(args, opp: str, port: int) -> None:
    from PIL import Image

    from sf2.system1.system1 import System1, play_round

    s1 = System1(None if args.model in ("random", "none") else args.model, args.me, 0.5, args.device, args.seed,
                 explore=args.explore)
    pool = ThreadPoolExecutor(8)
    with open_fight(args.me, opp, port, args.rom) as (b, state):
        b.set_vars(VARS + STRUCT)   # play_round zips NAMES with each row: the struct bytes ride along at the end
        proxy = CapturingBridge(b, None)
        for game in range(args.games):
            tag = "%s_vs_%s_g%d" % (args.me, opp, game)
            img_dir = os.path.join(args.out, "images", tag)
            os.makedirs(img_dir, exist_ok=True)
            rows = []

            def sink(r, im, rows=rows, img_dir=img_dir):
                k = len(rows)
                rows.append(r)
                if im is not None and k % args.every == 0:
                    pool.submit(lambda im=im, k=k: Image.fromarray(im).save(
                        os.path.join(img_dir, "k%05d.png" % k), compress_level=1))

            proxy.sink = sink
            s1.rng = random.Random("s1:%d:%s:%d" % (args.seed, opp, game))
            rnd = play_round(proxy, s1, opp, state, random.Random("start:%d:%s:%d" % (args.seed, opp, game)),
                             None, game)
            proxy.sink = None
            arr = np.asarray(rows, np.int64)
            nf = len(NAMES)
            np.savez_compressed(os.path.join(args.out, tag + ".npz"), struct=arr[:, nf:].astype(np.uint8),
                                fields=arr[:, :nf].astype(np.int32), names=np.asarray(NAMES))
            print("%s: %d rows, %s" % (tag, len(rows), rnd.result), flush=True)
    pool.shutdown(wait=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--opps", default="ken")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--model", default=os.path.join("runs", "all8", "best"))
    ap.add_argument("--explore", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--port", type=int, default=PORTS["movement"][0])
    args = ap.parse_args()
    exit_on_sigterm()
    os.makedirs(args.out, exist_ok=True)
    for i, opp in enumerate(args.opps.split(",")):
        play_games(args, opp, args.port + i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
