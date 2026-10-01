"""How full the movement-pairs buckets are after a collection round (sf2.data.pairs_data.fill_report): per bucket
(character, controller, movement, direction, facing) min(collected, cap) / cap, summed per character, per controller
(directed player 1 vs CPU) and overall, the empty buckets, and the moves report (scripts/pairs_moves_report.py).

    python scripts/pairs_fill_report.py --root rollouts/pairs [--cap 60] [--json out.json]
"""
import argparse
import json
import sys

import _path  # noqa: F401
from sf2.data import pairs_data as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cap", type=int, default=D.CAP)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    pairs, dropped, names = D.collection_pairs(args.root)
    chars = sorted({c for n in names for c in n.split("_vs_")})
    rep = dict(D.fill_report(pairs, chars, args.cap), pair_dirs=len(names), pairs=len(pairs), dropped=dropped)
    print("%d pair dirs, %d pairs; %d buckets x cap %d; overall %.1f%% (directed %.1f%%, cpu %.1f%%)" % (
        len(names), len(pairs), rep["buckets"], args.cap, rep["overall_pct"], rep["per_controller"]["directed"],
        rep["per_controller"]["cpu"]))
    print("%-8s %6s %9s %6s  empty buckets" % ("char", "all%", "directed%", "cpu%"))
    for c, v in rep["per_char"].items():
        print("%-8s %6.1f %9.1f %6.1f  %d: %s" % (c, v["all"], v["directed"], v["cpu"], len(rep["zero"][c]),
                                                  "; ".join(rep["zero"][c])))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
