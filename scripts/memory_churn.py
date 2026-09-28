"""How much System 2 changed each memory, version by version: the playbook and every short memory of a character,
from the kept copies in logs/system2/memory/ (sf2.memory_churn). A stable playbook is the goal; RADICAL rewrites
(most lessons replaced, or a "use X" turned into "avoid X") are listed.

    python scripts/memory_churn.py                  # chunli, from today
    python scripts/memory_churn.py --since 20260928-062237 --show
Log: logs/system2/churn_<me>.log
"""
import argparse
import collections
import glob
import json
import os
import statistics

import _path  # noqa: F401
from sf2.advice import FORWARD
from sf2.memory_churn import diff
from sf2.vs_sweep import actions

KEEP = "logs/system2/memory"


def versions(me: str, since: str):
    """{memory name: [(time, memory)]} in time order; the name is 'playbook' or the opponent."""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(KEEP, "*_%s*.json" % me))):
        stamp, name = os.path.basename(f).split("_", 1)
        if stamp < since:
            continue
        name = name[:-5]
        key = "playbook" if name == me else name.split("_vs_", 1)[1]
        out[key].append((stamp, json.load(open(f))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--since", default="20260928-000000", help="version time stamp to start from")
    ap.add_argument("--show", action="store_true", help="print every rewrite's changes")
    args = ap.parse_args()
    moves = list(actions(args.char)) + [FORWARD]
    lines = []
    for name, vs in sorted(versions(args.char, args.since).items(), key=lambda kv: kv[0] != "playbook"):
        changes = [(t, diff(a, b, moves)) for (_, a), (t, b) in zip(vs, vs[1:])]
        if not changes:
            continue
        churn = [c.churn for _, c in changes]
        radical = [(t, c) for t, c in changes if c.radical]
        flips = sum(len(c.flipped) for _, c in changes)
        lines.append("%-9s %3d rewrites  mean churn %3.0f%%  unchanged %3d  radical %3d  flips %3d  size %d -> %d" % (
            name, len(changes), 100 * statistics.mean(churn), sum(c.churn == 0 for _, c in changes), len(radical),
            flips, changes[0][1].size[0], changes[-1][1].size[1]))
        for t, c in changes if args.show else radical[-5:]:
            lines.append("    %s %s" % (t, c.line()))
            for old, new in c.flipped:
                lines.append("        flip: %r -> %r" % (old, new))
            if args.show:
                lines += ["        + %s" % x for x in c.added] + ["        - %s" % x for x in c.dropped]
    os.makedirs("logs/system2", exist_ok=True)
    with open("logs/system2/churn_%s.log" % args.char, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
