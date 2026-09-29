"""How much System 2 changed each memory, version by version (scripts/report.py churn): the playbook and every short
memory of a character, from the kept copies in logs/system2/memory/ (sf2.memory_churn). A stable playbook is the
goal; RADICAL rewrites (most lessons replaced, or a "use X" turned into "avoid X") are listed.
"""
import collections
import glob
import json
import os
import statistics
from typing import List

from ..advice import FORWARD
from ..memory_churn import diff
from ..data.vs_sweep import actions

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
        out[key].append((stamp, _load(f)))
    return out


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def report(me: str, since: str, show: bool) -> List[str]:
    moves = list(actions(me)) + [FORWARD]
    lines = []
    for name, vs in sorted(versions(me, since).items(), key=lambda kv: kv[0] != "playbook"):
        changes = [(t, diff(a, b, moves)) for (_, a), (t, b) in zip(vs, vs[1:])]
        if not changes:
            continue
        churn = [c.churn for _, c in changes]
        radical = [(t, c) for t, c in changes if c.radical]
        flips = sum(len(c.flipped) for _, c in changes)
        lines.append("%-9s %3d rewrites  mean churn %3.0f%%  unchanged %3d  radical %3d  flips %3d  size %d -> %d" % (
            name, len(changes), 100 * statistics.mean(churn), sum(c.churn == 0 for _, c in changes), len(radical),
            flips, changes[0][1].size[0], changes[-1][1].size[1]))
        for t, c in changes if show else radical[-5:]:
            lines.append("    %s %s" % (t, c.line()))
            for old, new in c.flipped:
                lines.append("        flip: %r -> %r" % (old, new))
            if show:
                lines += ["        + %s" % x for x in c.added] + ["        - %s" % x for x in c.dropped]
    return lines
