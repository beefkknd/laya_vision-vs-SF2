"""Compare two runs of the lesson loop file by file (e.g. an original and its repeat from a lock): per arm and log,
identical or not, and where they first part (line, round/game, which fields).

    python scripts/compare_runs.py rollouts/qwen_lessons/20260929-155448_ken rollouts/locked/lesson_loop_v1/<stamp>_ken

Exit 0 only when every compared file is identical.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import _path  # noqa: F401
from sf2.data.dataset import read

ARMS = ("none", "loop")
FILES = ("rounds.jsonl", "actions.jsonl", "ledger.jsonl")
WHERE = ("game", "round", "frame")


def _rows(path: str) -> List[Dict]:
    return read(path, missing_ok=True) if os.path.exists(path) else []


def compare(a: str, b: str, arms: Sequence[str] = ARMS, files: Sequence[str] = FILES) -> List[Dict]:
    out = []
    for arm in arms:
        for name in files:
            pa, pb = os.path.join(a, arm, name), os.path.join(b, arm, name)
            if not (os.path.exists(pa) or os.path.exists(pb)):
                continue                                  # e.g. no ledger in the none arm
            ra, rb = _rows(pa), _rows(pb)
            first = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), None)
            if first is None and len(ra) != len(rb):
                first = min(len(ra), len(rb))
            row = {"arm": arm, "file": name, "lines": (len(ra), len(rb)), "same": first is None, "first_diff": first}
            if first is not None and first < min(len(ra), len(rb)):
                x, y = ra[first], rb[first]
                row["where"] = {k: x[k] for k in WHERE if k in x}
                row["keys"] = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    rows = compare(args.a, args.b)
    for r in rows:
        print("%-5s %-14s %5d vs %5d lines  %s" % (r["arm"], r["file"], r["lines"][0], r["lines"][1],
                                                   "IDENTICAL" if r["same"] else "first differs at line %d %s %s" % (
                                                       r["first_diff"], json.dumps(r.get("where", {})),
                                                       r.get("keys", ""))))
    return 0 if rows and all(r["same"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
