"""Make an EARLY (thin, weak) value table from a trained one -- for the watch_chunli.sh demo.

Scales every cell's evidence count down so only the strongest handful of moves just cross MIN_TRIES(20)
(trusted) and the rest stay under-confident -> the quorum explores more and plays weak (~30% win).
Each move's MEAN (sum/n) is preserved; only the confidence (n) drops.

    python scripts/make_early_table.py runs/tables/chunli.json runs/tables/chunli_early.json [--maxn 30]

Measured (chunli vs ryu, frozen quorum): expert table ~67% round-win / every match; this early table
~33% round-win / wins ~1 match in 6.
"""
import argparse
import json


def thin(cells, maxn):
    mx = max((st[0] for c in cells.values() for st in c.values()), default=1)
    f = maxn / mx
    out = {}
    for wk, moves in cells.items():
        nm = {}
        for act, (n, s, ss) in moves.items():
            n2 = max(1, round(n * f))
            sc = n2 / n if n else 0.0
            nm[act] = [n2, round(s * sc, 1), round(ss * sc, 1)]
        out[wk] = nm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--maxn", type=int, default=30, help="target max evidence count (MIN_TRIES is 20)")
    a = ap.parse_args()
    with open(a.src) as f:
        cells = json.load(f)["cells"]
    out = {"cells": thin(cells, a.maxn)}
    with open(a.dst, "w") as f:
        json.dump(out, f)
    ns = [st[0] for c in out["cells"].values() for st in c.values()]
    print("wrote %s  (%d cells, n %d-%d, %d >= MIN_TRIES(20))"
          % (a.dst, len(out["cells"]), min(ns), max(ns), sum(v >= 20 for v in ns)))


if __name__ == "__main__":
    main()
