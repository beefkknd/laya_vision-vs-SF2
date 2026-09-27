"""The balance gate (docs/TWO_SYSTEM_PLAN.md, B4): refuse a training mix where one character dominates.

    python scripts/balance.py data/<a> data/<b> ... [--tol 0.10] [--opponents]

Counts the training rows per player-1 character (and per opponent) from each row's note, and exits 1 if any
character is more than --tol away from an equal share. Owner rule: the all-character checkpoint must not be
Chun-Li focused (or focused on any one character). --opponents applies the same rule to opponents.
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence

import _path  # noqa: F401
from sf2.contract import parse_note


def count(dirs: Sequence[str]) -> Dict[str, Counter]:
    me, opp = Counter(), Counter()
    for d in dirs:
        with open(os.path.join(d, "train.jsonl")) as f:
            for line in f:
                s = parse_note(json.loads(line)["state_text"])
                me[s["me"]] += 1
                opp[s["opp"]] += 1
    return {"me": me, "opp": opp}


def _off(counts: Counter, tol: float, what: str) -> List[str]:
    total, n = sum(counts.values()), len(counts)
    if n < 2:
        return []
    fair = total / n
    return ["%s %s has %d rows, %.0f%% off the equal share of %.0f" % (what, k, v, 100 * (v - fair) / fair, fair)
            for k, v in sorted(counts.items()) if abs(v - fair) > tol * fair]


def check(c: Dict[str, Counter], tol: float = 0.10, opponents: bool = False) -> List[str]:
    return _off(c["me"], tol, "character") + (_off(c["opp"], tol, "opponent") if opponents else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--opponents", action="store_true")
    args = ap.parse_args()
    c = count(args.data)
    print(json.dumps({k: dict(v) for k, v in c.items()}))
    problems = check(c, args.tol, args.opponents)
    for p in problems:
        print("UNBALANCED:", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
