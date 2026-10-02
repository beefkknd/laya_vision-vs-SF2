"""HARD GATE: no lookup table and no RAM in real (play) code. See sf2/hard_gate.py for the definitions.

    python scripts/hard_gate.py                 # the repo; exit 0 = clean, 1 = violations (each printed), 2 = usage
    python scripts/hard_gate.py --repo DIR --entry scripts/play.py
"""
import argparse
import os
import sys

import _path  # noqa: F401
from sf2.config import REPO
from sf2.hard_gate import ENTRY_POINTS, Policy, check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO, help="repo root to check (default: this repo)")
    ap.add_argument("--entry", action="append", default=None,
                    help="entry point, repo-relative (repeatable; default: sf2.hard_gate.ENTRY_POINTS)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.repo):
        print("hard gate: no such repo directory %s" % args.repo, file=sys.stderr)
        return 2
    policy = Policy(entry_points=tuple(args.entry or ENTRY_POINTS))
    found = check(os.path.abspath(args.repo), policy)
    for v in found:
        print(v)
    kinds = sorted({v.kind for v in found})
    print("hard gate: %s (%d violation%s%s)" % ("FAIL" if found else "clean", len(found), "" if len(found) == 1 else "s",
                                                ": " + ", ".join(kinds) if kinds else ""))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
