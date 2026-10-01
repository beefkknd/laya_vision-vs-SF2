"""Build the movement dataset v2 (sf2.data.movement_data2; docs/prereg_movement_data.md) from the collection
(scripts/collect_movement.py):

    python scripts/build_movement_data2.py [--root rollouts/mv2] [--out test_data_mv2] [--opps ryu,ken] [--workers 7]

Writes <out>/<opp>/{train,val,test_real}.jsonl, frames -> <root>/<opp>/images (symlink), stats.json, and
<out>/build.json + summary.json. Prints the pairs per answer and split for each opponent. Exit 1 on any problem.
Then run the dataset gate: scripts/gate_movement_data.py --data <out>.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import movement as M
from sf2.data import movement_data2 as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("rollouts", "mv2"))
    ap.add_argument("--out", default=D.OUT)
    ap.add_argument("--opps", help="only these opponents (comma-separated)")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)
    res = D.build(args.root, args.out, args.opps.split(",") if args.opps else None, args.workers)
    for opp, s in sorted(res["counts"].items()):
        print("%-8s games %s, rows %s, dropped (uncommitted) %d, long %d, cut by the round end %d" % (
            opp, s["games"], s["rows"], s["dropped_uncommitted"], s["long"], s["cut_end"]))
        for split, c in s["counts"].items():
            print("   %-6s %s" % (split, " ".join("%s=%d" % (a, c[a]) for a in M.ANSWERS)))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"meta": res["meta"], "counts": res["counts"], "problems": res["problems"][:200]}, f, indent=1)
    for p in res["problems"][:40]:
        print("PROBLEM:", p)
    if res["problems"]:
        print("%d problems" % len(res["problems"]))
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
