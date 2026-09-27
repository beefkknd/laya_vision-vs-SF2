"""Compare two rollouts played on the same opening schedule, opening by opening (sf2/openings.py).

    python scripts/paired.py rollouts/<control> rollouts/<arm> [--effect 15]

Prints each arm's mean net damage per round over the shared openings, the paired difference with its 95%
interval, and a verdict against the effect size: better, worse, no effect, or inconclusive.
"""
import argparse
import json
import os

import _path  # noqa: F401
from sf2 import dataset as D
from sf2.openings import paired


def verdict(r, effect):
    lo, hi = r["ci95"]
    if lo > 0:
        return "better" if lo >= effect or r["mean_diff"] >= effect else "better, below the effect size"
    if hi < 0:
        return "worse"
    if hi < effect:
        return "no effect as large as %g" % effect
    return "inconclusive: the interval includes 0 and %g" % effect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control")
    ap.add_argument("arm")
    ap.add_argument("--effect", type=float, default=15.0, help="the practical effect size, net damage per round")
    args = ap.parse_args()
    r = paired(D.read(os.path.join(args.control, "rounds.jsonl")), D.read(os.path.join(args.arm, "rounds.jsonl")))
    r["verdict"] = verdict(r, args.effect)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
