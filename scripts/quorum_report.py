"""Read quorum runs back: the numbers each rollout gate in docs/quorum_integration.md asks for.

    python scripts/quorum_report.py rollouts/loop_screen/<run> [<run> ...]   # one or more run dirs (trace.jsonl)
    python scripts/quorum_report.py <run> --json                             # machine-readable

Reports, from the per-decision trace (each decision's quorum record + its delayed-hit-corrected dealt / taken):
  * agreement vs outcome: mean net hp per decision by vote-share band (Phase 0 gate: a split vote should predict
    a worse exchange; if it does not, the quorum has nothing to work with);
  * how often the quorum's move differs from text-laya's, and the net when they agree vs disagree;
  * who decided (source counts) and the escalation rate (decisions with no quorum);
  * per voter: how often its proposal was the move played, and the mean net when it was.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def load(dirs: List[str]) -> List[Dict]:
    out = []
    for d in dirs:
        with open(os.path.join(d, "trace.jsonl")) as f:
            for line in f:
                e = json.loads(line)
                if e.get("event") == "decision" and e.get("quorum"):
                    out.append(e)
    return out


def _mean(xs: List[float]):
    return round(sum(xs) / len(xs), 3) if xs else None


def report(decs: List[Dict]) -> Dict:
    net = [(d.get("dealt", 0) or 0) - (d.get("taken", 0) or 0) for d in decs]
    bands = []
    for lo, hi in BANDS:
        xs = [n for d, n in zip(decs, net) if lo <= d["quorum"]["share"] < hi]
        bands.append({"share": "%.2f-%.2f" % (lo, min(hi, 1.0)), "n": len(xs), "mean_net": _mean(xs)})
    agree = [n for d, n in zip(decs, net) if d["quorum"]["top"] == d["quorum"]["laya_move"]]
    differ = [n for d, n in zip(decs, net) if d["quorum"]["top"] != d["quorum"]["laya_move"]]
    sources: Dict[str, int] = {}
    for d in decs:
        sources[d.get("source") or "?"] = sources.get(d.get("source") or "?", 0) + 1
    voters: Dict[str, Dict] = {}
    for d, n in zip(decs, net):
        for v, act, _, _ in d["quorum"]["proposals"]:
            s = voters.setdefault(v, {"proposed": 0, "played": 0, "net": []})
            s["proposed"] += 1
            if act == d.get("action"):
                s["played"] += 1
                s["net"].append(n)
    return {
        "decisions": len(decs),
        "mean_net": _mean(net),
        "agreement_vs_outcome": bands,
        "quorum_differs_from_laya": round(len(differ) / len(decs), 3) if decs else None,
        "mean_net_when_agree": _mean(agree), "mean_net_when_differ": _mean(differ),
        "sources": sources,
        "escalation_rate": round(sum(d["quorum"]["quorum_move"] is None for d in decs) / len(decs), 3) if decs else None,
        "voters": {v: {"proposed": s["proposed"], "played": s["played"], "mean_net_when_played": _mean(s["net"])}
                   for v, s in sorted(voters.items())},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = report(load(args.runs))
    if args.json:
        print(json.dumps(r, indent=1))
        return 0
    print("decisions %d, mean net %s, escalation (no quorum) %s" % (r["decisions"], r["mean_net"], r["escalation_rate"]))
    print("agreement vs outcome (Phase 0 gate: lower share should mean lower net):")
    for b in r["agreement_vs_outcome"]:
        print("  share %-9s n=%-5d mean net %s" % (b["share"], b["n"], b["mean_net"]))
    print("quorum differs from laya on %s of decisions; net when agree %s, when differ %s"
          % (r["quorum_differs_from_laya"], r["mean_net_when_agree"], r["mean_net_when_differ"]))
    print("sources:", r["sources"])
    for v, s in r["voters"].items():
        print("  %-7s proposed %-5d played %-5d mean net when played %s" % (v, s["proposed"], s["played"],
                                                                         s["mean_net_when_played"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
