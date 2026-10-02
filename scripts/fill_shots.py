"""How full the fireball cells are (sf2.data.mv3_fireball.fill_report): per (split, thrower, side, flight stage) the
eligible projectile samples of committed games vs the caps (40 train / 20 test). One line per call (a round).

    python scripts/fill_shots.py --root rollouts/pairs2p [--json]
"""
import argparse
import json
import sys

import _path  # noqa: F401
from sf2.data import mv3_fireball as F


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rollouts/pairs2p")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = F.fill_report(args.root)
    if args.json:
        print(json.dumps(rep, indent=1))
    print("filled %.1f%%, %d of %d cells not full: %s; dropped %s" % (
        rep["filled_pct"], len(rep["not_full"]), len(rep["cells"]), json.dumps(rep["not_full"]), json.dumps(rep["dropped"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
