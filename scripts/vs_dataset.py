"""Stage-1 dataset (still opponent) per character, into test_data/<char>/. See sf2/data/vs_sweep.py for the design.

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
import os
import sys


import _path  # noqa: F401
from sf2.config import PORTS
from sf2.data.build import build
from sf2.data.collect import collect, collect_defense, import_live
from sf2.eval.runner import fan_out
from sf2.vocab import RANGES

def run(args) -> int:
    """Everything, in parallel: per character, one collect per range on the left (train + test) and, with
    --right-test, one per range on the right (real test frames, the mirroring check); then build. Each character's
    still opponent is --dummy, or else the next character in --chars (the last one's is the first)."""
    if args.pairs:   # "a:b,c:d": a's dummy is b and b's is a
        who_vs = [(x, y) for pair in args.pairs.split(",") for a, b in [pair.split(":")] for x, y in ((a, b), (b, a))]
    else:
        chars = args.chars.split(",")
        who_vs = [(me, args.dummy or chars[(i + 1) % len(chars)]) for i, me in enumerate(chars)]
    cmds, port = [], args.base_port
    for me, dummy in who_vs:
        if dummy == me:
            raise SystemExit("%s needs a different dummy (--dummy, or two or more --chars)" % me)
        for who in (1, 2) if args.right_test else (1,):
            p1, p2 = (me, dummy) if who == 1 else (dummy, me)
            for rng in RANGES:
                for kind in args.kinds.split(","):
                    key = (me, "left" if who == 1 else "right", rng, kind)
                    cmd = [sys.executable, os.path.abspath(__file__), "collect" if kind == "static" else
                           "collect-defense", "--p1", p1, "--p2", p2, "--who", str(who), "--range", rng,
                           "--port", str(port)] + (["--rom", args.rom] if args.rom else [])
                    cmds.append((key, cmd))
                    port += 1
    print("%d collect jobs running (logs in logs/dataset/)" % len(cmds), flush=True)
    failed = fan_out(cmds, os.path.join("logs", "dataset"))
    for key in failed:
        print("FAILED collect, see logs/dataset/%s.log" % "_".join(key))
    if failed:
        return 1
    return build(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("collect", "still-opponent data"), ("collect-defense", "block data (sf2/data/vs_defense.py)")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--p1", required=True)
        c.add_argument("--p2", required=True)
        c.add_argument("--who", type=int, choices=(1, 2), required=True)
        c.add_argument("--range", choices=RANGES)
        c.add_argument("--port", type=int, required=True)
        c.add_argument("--rom", default=os.environ.get("SF2_ROM"))
        c.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    il = sub.add_parser("import-live", help="game logs of live play -> shards (scripts/play_system1.py output)")
    il.add_argument("--log", required=True, help="e.g. rollouts/live_dhalsim")
    bd = sub.add_parser("build")
    r = sub.add_parser("run", help="collect every shard in parallel, then build")
    r.add_argument("--kinds", default="static,defense", help="static (still opponent), defense (blocks), or both")
    r.add_argument("--chars", default="ryu,chunli", help="comma-separated; each gets its own test_data/<char>/")
    r.add_argument("--pairs", help="a:b,c:d - each pair are each other's still opponent (overrides --chars)")
    r.add_argument("--dummy", help="the still opponent for every character (default: the next in --chars)")
    r.add_argument("--base-port", type=int, default=PORTS["dataset"][0])
    r.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    for p in (bd, r):
        p.add_argument("--no-right-test", dest="right_test", action="store_false",
                       help="skip the real right-side test set (the one-time mirroring check)")
    args = ap.parse_args()
    return {"collect": collect, "collect-defense": collect_defense, "import-live": import_live, "build": build,
            "run": run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
