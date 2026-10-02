"""The sprite catalog (docs/laya_text_only_plan.md step 1): one 2P-versus game per ordered pair, both sides driven by
their own move lists (as scripts/collect_pairs.py --mode vs), every frame's sprites cut from the OAM snapshot
(sf2.sprites), labelled from RAM, deduplicated by pixels. One headless Mesen per pair, many at once.

    .venv/bin/python scripts/collect_sprites.py                                  # all 56 pairs, then build + report
    .venv/bin/python scripts/collect_sprites.py --pairs ryu:ken --out /tmp/sc     # a smoke
    .venv/bin/python scripts/collect_sprites.py --build-only                     # re-join labels, re-report

Writes <out>/_pairs/<A>_vs_<B>/ (sf2.sprites.collect), then <out>/<char>/<key>.png, <out>/_projectiles/,
<out>/catalog.json, <out>/_sheets/<char>.png and <out>/report.json. A pair with done.json is not played again.
"""
import argparse
import json
import os
import sys
import time

import _path  # noqa: F401
from sf2.config import REPO
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out
from sf2.sprites import build as B
from sf2.sprites import report as R
from sf2.sprites.collect import collect_pair
from collect_pairs import CHARS, parse_pairs
from make_vs_pair_states import vs_state

OUT = os.path.join(REPO, "out", "sprite_catalog")
BASE_PORT = 52101            # 52101 .. 52156: outside every range in sf2.config.PORTS
JOB_GB = 1.5


def summarize(out: str, seconds: float) -> dict:
    cat = R.load(out)
    sprites = cat["sprites"]
    rep = dict(seconds_wall=round(seconds, 1), per_char=R.per_char(sprites),
               shared=_jsonable(R.shared(sprites, "labels")), shared_lag0=_jsonable(R.shared(sprites, "lag0")),
               rerender_bad=R.rerender_check(out, sprites, 50),
               sheets=[p for c in CHARS + (R.PROJ,) if (p := R.contact_sheet(out, sprites, c))],
               stats=cat["stats"])
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def _jsonable(s: dict) -> dict:
    return {"chars": {c: dict(v) for c, v in s["chars"].items()},
            "pairs": {"%s|%s" % k: dict(v) for k, v in sorted(s["pairs"].items(), key=lambda kv: -kv[1]["frames"])}}


def print_report(rep: dict) -> None:
    print("wall %.0f s" % rep["seconds_wall"])
    print("%-12s %8s %8s %10s %9s %8s %8s" % ("char", "sprites", "idx_keys", "frames", "shared", "shr_frm%", "minor%"))
    for c, d in sorted(rep["per_char"].items()):
        s = rep["shared"]["chars"].get(c, {})
        fr = max(s.get("frames", 0), 1)
        print("%-12s %8d %8d %10d %9d %8.1f %8.1f" % (c, d["sprites"], d["index_keys"], d["frames"], s.get("shared", 0),
                                                       100 * s.get("shared_frames", 0) / fr,
                                                       100 * s.get("minority_frames", 0) / fr))
    print("action pairs (sprites, frames, minority frames):")
    for k, v in rep["shared"]["pairs"].items():
        print("  %-24s %6d %8d %8d" % (k, v["sprites"], v["frames"], v["minor"]))
    print("re-render check bad:", rep["rerender_bad"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--base-port", type=int, default=BASE_PORT)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("A", "B", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        a, b, port = args.one
        s = collect_pair(args.out, a, b, int(port), os.path.join(REPO, vs_state(a, b)), args.seed, rom=args.rom)
        print("%s vs %s: %s, %d frames, %.0f s" % (a, b, s["result"], s["frames"], s["seconds"]), flush=True)
        return 0
    t0 = time.time()
    if not args.build_only:
        pairs = parse_pairs(args.pairs)
        todo = [(i, a, b) for i, (a, b) in enumerate(pairs)
                if not os.path.exists(os.path.join(args.out, "_pairs", "%s_vs_%s" % (a, b), "done.json"))]
        cmds = [(("%s_vs_%s" % (a, b),), [sys.executable, os.path.abspath(__file__), "--one", a, b,
                                          str(args.base_port + i), "--out", args.out, "--seed", str(args.seed)]
                 + (["--rom", args.rom] if args.rom else [])) for i, a, b in todo]
        print("%d pairs to play, up to %d at once" % (len(cmds), args.workers), flush=True)
        failed = fan_out(cmds, os.path.join(REPO, "logs", "sprites"), job_gb=JOB_GB,
                         budget=Budget(total_gb=JOB_GB * args.workers))
        print("played in %.0f s; failed: %s" % (time.time() - t0, failed), flush=True)
        if failed:
            return 1
    B.write(args.out, B.build(args.out))
    print_report(summarize(args.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
