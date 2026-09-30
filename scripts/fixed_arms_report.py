"""Fixed-advice arms vs no advice (scripts/ab_memory.py --fixed), pooled over the seeds of a batch, the seed (run) as
the unit (sf2.eval.stats.run_level). The batch is every run a directory of ab_memory console logs saved.

    python scripts/fixed_arms_report.py logs/ab/ryu_expert --opp ryu [--vs expert,qwenset]
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.eval.stats import paired, run_level


def roots(log_dir: str, base: str = "") -> List[str]:
    """The run each seed's log saved (its last "saved" line), under ``base`` (the saved paths are relative to the
    checkout the batch ran in)."""
    out = []
    for f in sorted(glob.glob(os.path.join(base, log_dir, "*.log"))):
        saved = [x.split()[1] for x in open(f) if x.startswith("saved ")]
        if not saved:
            raise SystemExit("%s: no run saved" % f)
        out.append(os.path.join(base, saved[-1]))
    return out


def arms(root: str, opp: str) -> List[str]:
    pre = opp + "_"
    return sorted(n[len(pre):] for n in os.listdir(root) if n.startswith(pre) and n != pre + "none"
                  and os.path.isdir(os.path.join(root, n)))


def report(log_dir: str, opp: str, vs: List[str], base: str = "") -> Dict:
    rs = roots(log_dir, base)
    names = arms(rs[0], opp)
    diff: Dict[str, List[List[int]]] = {a: [] for a in names}
    taken: Dict[str, List[List[int]]] = {a: [] for a in names}
    wins = {a: 0 for a in names + ["none"]}
    ab: List[List[int]] = []
    for r in rs:
        none = read(os.path.join(r, "%s_none" % opp, "rounds.jsonl"))
        wins["none"] += sum(x["result"] == "win" for x in none)
        got = {}
        for a in names:
            got[a] = read(os.path.join(r, "%s_%s" % (opp, a), "rounds.jsonl"))
            diff[a].append(paired(got[a], none))
            taken[a].append([n["taken"] - x["taken"] for x, n in zip(got[a], none)])
            wins[a] += sum(x["result"] == "win" for x in got[a])
        if vs:
            ab.append(paired(got[vs[0]], got[vs[1]]))
    out = {"seeds": len(rs), "rounds_per_arm": sum(len(d) for d in diff[names[0]]), "wins": wins,
           "vs_none": {a: run_level(diff[a]) for a in names}, "taken_less": {a: run_level(taken[a]) for a in names}}
    if vs:
        out["%s_minus_%s" % tuple(vs)] = run_level(ab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--opp", required=True)
    ap.add_argument("--vs", default="", help="two arms to compare directly, e.g. expert,qwenset")
    args = ap.parse_args()
    r = report(args.log_dir, args.opp, [x for x in args.vs.split(",") if x])
    print("%d seeds, %d rounds per arm; rounds won: none %d" % (r["seeds"], r["rounds_per_arm"], r["wins"]["none"]))
    for a, v in sorted(r["vs_none"].items(), key=lambda kv: -kv[1]["mean"]):
        t = r["taken_less"][a]
        print("  %-10s %+6.1f [%+6.1f, %+6.1f] %-9s | taken less %+5.1f | won %d" % (
            a, v["mean"], v["ci95"][0], v["ci95"][1], v["verdict"], t["mean"], r["wins"][a]))
    for k, v in r.items():
        if k.endswith(tuple("_minus_" + x for x in r["vs_none"])):
            print("  %s: %+.1f [%+.1f, %+.1f] %s" % (k, v["mean"], v["ci95"][0], v["ci95"][1], v["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
