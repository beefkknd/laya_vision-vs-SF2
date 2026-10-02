"""The fireball dataset's gates (sf2.data.mv3_fireball_gate; docs/prereg_movement_finetunes.md, round 3) and the
contact sheets (one per thrower: side x flight stage bands, and one of none rows). The exit code decides: 0 only when
every gate passes.

    python scripts/gate_mv3_fireball.py --data test_data_mv3_fireball [--sheets <dir>]

Writes <data>/gate.json and <sheets>/fireball_<thrower|none>.png (default <data>_contact: outside the dataset, whose every dir is an answer).
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import mv3_fireball_gate as G


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="test_data_mv3_fireball")
    ap.add_argument("--sheets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rep = G.run_gates(args.data, seed=args.seed)
    with open(os.path.join(args.data, "gate.json"), "w") as f:
        json.dump(rep, f, indent=1)
    for name in G.GATES:
        g = rep["gates"][name]
        print("%s %s %s" % ("PASS" if g["pass"] else "FAIL", name,
                            json.dumps({k: v for k, v in g.items() if k != "pass"})[:1000]))
    print("ownership:", json.dumps(rep["ownership"]))
    print("counts:", json.dumps(rep["counts"]))
    for p in G.contact_sheets(args.data, args.sheets or os.path.normpath(args.data) + "_contact", seed=args.seed):
        print("contact sheet:", p)
    print("fireball gate: %s" % ("PASS" if rep["pass"] else "FAIL"))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
