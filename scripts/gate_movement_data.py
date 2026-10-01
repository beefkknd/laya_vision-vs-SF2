"""The movement dataset's gates (sf2.data.movement_gate; docs/prereg_movement_data.md gates 1-3, the 10 GB disk
budget) and the contact sheets for the owner. The exit code decides: 0 only when every gate passes.

    python scripts/gate_movement_data.py --data test_data_mv2 [--sheets test_data_mv2/contact]

Writes <data>/gate.json (the full report) and <sheets>/contact_<opp>.png.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import movement as M
from sf2.data import movement_gate as G


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="test_data_mv2")
    ap.add_argument("--answers", default=",".join(M.ANSWERS))
    ap.add_argument("--min-train", type=int, default=G.MIN_TRAIN)
    ap.add_argument("--min-test", type=int, default=G.MIN_TEST)
    ap.add_argument("--min-share", type=float, default=G.MIN_SHARE)
    ap.add_argument("--min-disc", type=int, default=G.MIN_DISC)
    ap.add_argument("--max-gb", type=float, default=G.MAX_GB)
    ap.add_argument("--sample", type=int, default=G.SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheets", help="contact sheet dir (default <data>/contact); 'none' skips them")
    ap.add_argument("--per-answer", type=int, default=3)
    args = ap.parse_args(argv)
    answers = args.answers.split(",")
    rep = G.run_gates(args.data, answers, args.min_train, args.min_test, args.min_share, args.min_disc,
                      max_gb=args.max_gb, sample=args.sample, seed=args.seed)
    with open(os.path.join(args.data, "gate.json"), "w") as f:
        json.dump(rep, f, indent=1)
    for name in G.GATES:
        g = rep["gates"][name]
        detail = {k: v for k, v in g.items() if k != "pass"}
        if name == "counts":
            detail = {"shortfalls": len(g["shortfalls"]), "first": g["shortfalls"][:12]}
        print("%s %s %s" % ("PASS" if g["pass"] else "FAIL", name, json.dumps(detail)[:1500]))
    if args.sheets != "none":
        for p in G.contact_sheets(args.data, args.sheets or os.path.join(args.data, "contact"), args.per_answer,
                                  answers, args.seed):
            print("contact sheet:", p)
    print("dataset gate: %s (%d rows, %s)" % ("PASS" if rep["pass"] else "FAIL", rep["rows"], ",".join(rep["opps"])))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
