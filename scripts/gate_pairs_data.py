"""The movement-pairs gates (sf2.data.pairs_gate; docs/prereg_movement_pairs.md) and one contact sheet per character.
The exit code decides: 0 only when every gate passes.

    python scripts/gate_pairs_data.py --data test_data_pairs [--sheets test_data_pairs/contact]

Writes <data>/gate.json (the full report, with the counts table) and <sheets>/contact_<char>.png.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="test_data_pairs")
    ap.add_argument("--max-gb", type=float, default=G.MAX_GB)
    ap.add_argument("--sample", type=int, default=G.SAMPLE)
    ap.add_argument("--min-disc", type=int, default=G.MIN_DISC)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheets", help="contact sheet dir (default <data>/contact); 'none' skips them")
    args = ap.parse_args(argv)
    rep = G.run_gates(args.data, L.poke_bands(), args.max_gb, args.sample, args.min_disc, args.seed)
    with open(os.path.join(args.data, "gate.json"), "w") as f:
        json.dump(rep, f, indent=1)
    for name in G.GATES:
        g = rep["gates"][name]
        print("%s %s %s" % ("PASS" if g["pass"] else "FAIL", name,
                            json.dumps({k: v for k, v in g.items() if k != "pass"})[:1200]))
    print("counts: %d cells filled, %d short of the cap; per question: %s" % (
        len(rep["counts"]["cells"]), len(rep["counts"]["short"]), json.dumps(rep["counts"]["questions"])[:1500]))
    if args.sheets != "none":
        for p in G.contact_sheets(args.data, args.sheets or os.path.join(args.data, "contact"), seed=args.seed):
            print("contact sheet:", p)
    print("pairs gate: %s" % ("PASS" if rep["pass"] else "FAIL"))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
