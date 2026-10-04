#!/usr/bin/env python3
"""Stage 3b: A/B the self-learning value TABLE vs the RULES system (live head-to-head), phased one opponent first.

Both arms LEARN as they play (the table credits outcomes; the rules arm churns the short memory via Qwen), so the
same seed does NOT give identical fights -- round-by-round pairing is unreliable. Instead this reports the LEARNING
CURVES (cumulative win-rate per round) and a LAST-N tail comparison: each arm's win-rate and mean hp/round over the
last N rounds, plus the table-minus-rules difference with a two-sample bootstrap 95% interval and a verdict. Reuses
sf2.eval.stats (verdict, slope).

Run the two arms first (headless, one opponent), e.g.:
    python scripts/play_career.py --opps ryu --policy table --playbook ab_tbl   --block 4 --cap 8
    python scripts/play_career.py --opps ryu --policy rules --playbook ab_rules --block 4 --cap 8
then compare:
    python scripts/ab_table.py --table playbooks/ab_tbl --rules playbooks/ab_rules --last 40
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.eval.stats import slope, verdict  # noqa: E402


def arm_rounds(session_dir):
    """Every round of a session in order, as {"hp", "result"} from the round events of its round_* subdirs."""
    out = []
    for d in sorted(glob.glob(os.path.join(session_dir, "round_*"))):
        trace = os.path.join(d, "trace.jsonl")
        if not os.path.isdir(d) or not os.path.exists(trace):
            continue
        for line in open(trace):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "round":
                out.append({"hp": e.get("hp", 0), "result": e.get("result")})
    return out


def cum_winrate(rounds):
    """Cumulative win fraction after each round (the learning curve)."""
    out, wins = [], 0
    for i, r in enumerate(rounds, 1):
        wins += 1 if r["result"] == "win" else 0
        out.append(wins / i)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _boot_diff(a, b, reps=2000, seed=20260104):
    """95% interval of mean(a) - mean(b), resampling each set's rounds independently (two-sample bootstrap)."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    rng, boot = random.Random(seed), []
    for _ in range(reps):
        boot.append(_mean([rng.choice(a) for _ in a]) - _mean([rng.choice(b) for _ in b]))
    boot.sort()
    return boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]


def _arm_stat(rounds, last):
    tail = rounds[-last:] if last else rounds
    hp = [r["hp"] for r in tail]
    return {"rounds": len(rounds), "tail": len(tail),
            "winrate": sum(r["result"] == "win" for r in tail) / max(1, len(tail)),
            "hp_mean": _mean(hp), "hp_slope": slope([r["hp"] for r in rounds])}


def compare(table_rounds, rules_rounds, last=None):
    """The A/B report dict: each arm's tail stats + the table-minus-rules hp difference with a bootstrap CI/verdict."""
    t, r = _arm_stat(table_rounds, last), _arm_stat(rules_rounds, last)
    thp = [x["hp"] for x in (table_rounds[-last:] if last else table_rounds)]
    rhp = [x["hp"] for x in (rules_rounds[-last:] if last else rules_rounds)]
    lo, hi = _boot_diff(thp, rhp)
    return {"table": t, "rules": r,
            "hp_diff": {"mean": t["hp_mean"] - r["hp_mean"], "ci95": [lo, hi], "verdict": verdict(lo, hi)},
            "winrate_diff": t["winrate"] - r["winrate"],
            "curves": {"table": cum_winrate(table_rounds), "rules": cum_winrate(rules_rounds)}}


def report(cmp):
    for arm in ("table", "rules"):
        s = cmp[arm]
        print("%-6s  %4d rounds | last %d: win-rate %3.0f%%  hp/round %+6.1f  (slope %+.2f/round over all)"
              % (arm, s["rounds"], s["tail"], 100 * s["winrate"], s["hp_mean"], s["hp_slope"]))
    d = cmp["hp_diff"]
    print("TABLE - RULES (last tail): hp %+6.1f  95%% CI [%+.1f, %+.1f]  %s | win-rate %+.0f pp"
          % (d["mean"], d["ci95"][0], d["ci95"][1], d["verdict"], 100 * cmp["winrate_diff"]))
    print("(verdict on hp/round: HELPS = table better, HURTS = rules better, NOT SHOWN = inconclusive)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="the table arm's session dir (play_career --policy table)")
    ap.add_argument("--rules", required=True, help="the rules arm's session dir (play_career --policy rules)")
    ap.add_argument("--last", type=int, default=40, help="compare over the last N rounds (the converged tail)")
    args = ap.parse_args()
    tr, rr = arm_rounds(args.table), arm_rounds(args.rules)
    if not tr or not rr:
        print("no rounds found (table=%d, rules=%d) -- run both arms first" % (len(tr), len(rr)), file=sys.stderr)
        return 2
    report(compare(tr, rr, args.last))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
