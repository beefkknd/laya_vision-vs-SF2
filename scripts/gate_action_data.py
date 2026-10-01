"""The gates on the three movement datasets (sf2.data.action_gate): labels re-derived independently from the stored
RAM (100%), RAM-to-image alignment (lag 1), disk, train.py's data checks; and, for the act set, the coverage report
(every (actor, code) in train and test: shortfalls listed, not a fail) and one contact sheet per actor. The exit code
decides: 0 only when every gate of every dataset given passes.

    python scripts/gate_action_data.py --data test_data_act --data test_data_where --data test_data_dist
    python scripts/gate_action_data.py --data test_data_act_cpu --compare-act test_data_act    # CPU Chun-Li set

The CPU Chun-Li set (build.json "layout": "cpu_chunli") also has the provenance gate; with --compare-act its coverage
lists her codes new against that act set's Chun-Li options and the watched jump-attack IDs 23-27, 32.

Writes <data>/gate.json and, for the act set, <data>/contact/contact_<actor>.png.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import action_codes as A
from sf2.data import action_gate as G
from sf2.data import cpu_chunli as K
from sf2.data.action_data import THRESHOLDS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--thresholds", default=THRESHOLDS)
    ap.add_argument("--max-gb", type=float, default=G.MAX_GB)
    ap.add_argument("--sample", type=int, default=G.SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=G.VAL_LIMIT)
    ap.add_argument("--no-train-check", action="store_true", help="skip train.py's data checks (tests)")
    ap.add_argument("--sheets", default="contact", help="contact sheet subdir of the act set; 'none' skips")
    ap.add_argument("--chunli-ids", default=os.path.join("lessons", "chunli_action_ids.json"))
    ap.add_argument("--compare-act", help="an act set whose Chun-Li options the CPU set's codes are compared with")
    args = ap.parse_args(argv)
    ok = True
    for data in args.data:
        rep = G.run_gates(data, args.thresholds, args.max_gb, args.sample, args.seed, args.val_limit,
                          not args.no_train_check)
        if args.compare_act and "coverage" in rep:
            ref = json.load(open(os.path.join(args.compare_act, "build.json")))
            old = [int(c[3:]) for c in ref["options"].get(K.CPU, [])]
            rep["compare"] = dict(K.compare_codes(rep["coverage"]["table"], old), against=args.compare_act)
        with open(os.path.join(data, "gate.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print("== %s (%s)" % (data, rep["dataset"]))
        for name, g in rep["gates"].items():
            detail = {k: v for k, v in g.items() if k not in ("pass", "rows_problems")}
            print("%s %s %s" % ("PASS" if g["pass"] else "FAIL", name, json.dumps(detail)[:800]))
        if "coverage" in rep:
            c = rep["coverage"]
            print("REPORT coverage: %d (actor, code), %d shortfalls" % (c["codes"], len(c["shortfalls"])))
            for s in c["shortfalls"]:
                print("   short:", s)
            if "compare" in rep:
                print("REPORT vs %s: new %s; watched %s" % (args.compare_act, rep["compare"]["new_vs_mv3"], {
                    k: v and (v["train"], v["val"], v["test"]) for k, v in rep["compare"]["watched"].items()}))
            if args.sheets != "none":
                for p in G.contact_sheets(data, os.path.join(data, args.sheets), args.chunli_ids, A.RESERVED):
                    print("contact sheet:", p)
        print("%s: %s" % (data, "PASS" if rep["pass"] else "FAIL"))
        ok = ok and rep["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
