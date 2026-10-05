#!/usr/bin/env python
"""Study a value table and suggest which bees fill its gaps -- the trunk-study -> bee-suggestion process, callable.

    python scripts/table_study.py <table.json>              # a table file (e.g. a run's merged_table.json)
    python scripts/table_study.py --char chunli             # a character from the per-character store (runs/tables/)
    python scripts/table_study.py <table.json> --json       # machine-readable analysis + suggested bee config

Prints: headline counts, the thick trunk (what the table confidently knows), the blind contexts and promising-thin
cells (what it lacks), fireball-slice coverage, and a suggested bee set to fill the gaps. Read-only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf2.system1 import table_store as TS          # noqa: E402
from sf2.quorum.table_study import study, suggest_bees, CONFIDENT_N    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study a value table, suggest gap-filling bees.")
    ap.add_argument("table", nargs="?", help="path to a table JSON ({cells,shadow,depth})")
    ap.add_argument("--char", help="study this character from the per-character store instead of a file path")
    ap.add_argument("--root", default=TS.ROOT, help="store root for --char (default runs/tables)")
    ap.add_argument("--confident-n", type=int, default=CONFIDENT_N, help="n a move needs to count as confidently known")
    ap.add_argument("--fb-floor", type=float, default=0.15, help="suggest the fireball bee below this fb=1 sample share")
    ap.add_argument("--json", action="store_true", help="print the analysis + suggested bee config as JSON")
    args = ap.parse_args(argv)

    if args.char:
        table = TS.load(args.char, root=args.root)
    elif args.table:
        with open(args.table) as f:
            table = json.load(f)
    else:
        ap.error("give a table path or --char")
    cells = table.get("cells", {})
    a = study(cells, confident_n=args.confident_n)
    s = suggest_bees(a, fb_share_floor=args.fb_floor)

    if args.json:
        print(json.dumps({"analysis": {k: a[k] for k in ("contexts", "cells", "confident", "thin", "fb")},
                          "blind": a["blind"], "suggest": s}, indent=1))
        return 0

    fb = a["fb"]
    print("TABLE: %d contexts, %d cells; %d confident (n>=%d), %d thin" % (
        a["contexts"], a["cells"], a["confident"], args.confident_n, a["thin"]))
    print("FIREBALL slice: fb=1 %d samples across %d cells vs fb=0 %d/%d  (fb=1 share %.1f%%)" % (
        fb["fb1_n"], fb["fb1_cells"], fb["fb0_n"], fb["fb0_cells"], 100 * fb["fb1_share"]))
    print("\nTHICK TRUNK (confident winners, top 10):")
    for when, mv, m, n in a["trunk"][:10]:
        print("  %-22s %-20s mean %+6.2f  n=%d" % (when, mv, m, n))
    print("\nBLIND contexts (no confident move): " + (", ".join(w for w, _ in a["blind"]) or "none"))
    print("\nSUGGESTED BEES: frontier=%s  fireball=%s  flavors=%s" % (s["frontier"], s["fireball"], s["flavors"]))
    for note in s["notes"]:
        print("  - " + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
