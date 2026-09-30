"""Round 5's pre-registered analysis (docs/prereg_round5.md). The runs are the ones logs/qwen_lessons/round5/*.log saved;
their old twins are the character_fgc runs (no track) of rounds 3-4 with the same opponent and seed.

    python scripts/round5_report.py [--out FILE.json]
"""
import argparse
import collections
import glob
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.eval.stats import paired, pooled, run_level

LOGS = os.path.join("logs", "qwen_lessons", "round5")
OLD_ROOT = os.path.join("rollouts", "locked", "lesson_loop_v1")


def new_runs() -> List[str]:
    out = []
    for f in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        saved = [x.split()[1] for x in open(f) if x.startswith("saved ")]
        if not saved:
            raise SystemExit("%s: no run saved" % f)
        out.append(saved[-1])
    return out


def meta(d: str) -> Dict:
    with open(os.path.join(d, "verdict.json")) as f:
        v = json.load(f)
    return dict(v, opp=os.path.basename(d).split("_")[1], dir=d)


def old_twin(opp: str, seed: int, new: List[str]) -> str:
    cands = [d for d in sorted(glob.glob(os.path.join(OLD_ROOT, "*_%s_character_fgc" % opp)))
             if d not in new and meta(d)["seed"] == seed]
    if len(cands) != 1:
        raise SystemExit("%s %d: %d old twins %s" % (opp, seed, len(cands), cands))
    return cands[0]


def by_run(pairs: Dict[str, List[List[float]]]) -> Dict:
    per = {o: run_level(rs) for o, rs in sorted(pairs.items())}
    flat = {o: [x for r in rs for x in r] for o, rs in pairs.items()}
    return {"per_opponent": per, "pooled": pooled(flat)}


def halves(loop: List[Dict], none: List[Dict]) -> float:
    d = paired(loop, none)
    early = [x for r, x in zip(loop, d) if r["game"] < 5]
    late = [x for r, x in zip(loop, d) if r["game"] >= 5]
    return sum(late) / len(late) - sum(early) / len(early)


def system1(d: str) -> Dict:
    out = {}
    for arm in ("loop", "none"):
        acts = read(os.path.join(d, arm, "actions.jsonl"))
        out[arm] = {"decisions": len(acts), "follows_rule": sum(bool(a.get("follows_rule")) for a in acts),
                    "nothing_left": sum(a.get("rule") == "nothing_left" for a in acts),
                    "blocks": sum(a["action"].startswith("block") for a in acts)}
    return out


def report() -> Dict:
    new = new_runs()
    prim, s1, total, learn = (collections.defaultdict(list) for _ in range(4))
    qwen, sys1 = collections.defaultdict(list), collections.defaultdict(lambda: collections.Counter())
    for d in new:
        m = meta(d)
        old = old_twin(m["opp"], m["seed"], new)
        loop, none = read(os.path.join(d, "loop", "rounds.jsonl")), read(os.path.join(d, "none", "rounds.jsonl"))
        oloop, onone = (read(os.path.join(old, a, "rounds.jsonl")) for a in ("loop", "none"))
        prim[m["opp"]].append(paired(loop, none))
        s1[m["opp"]].append(paired(none, onone))
        total[m["opp"]].append(paired(loop, oloop))
        learn[m["opp"]].append([halves(loop, none)])
        for k in ("qwen_hold_rate", "random_hold_rate", "registered_at_proposal_share", "refused_duplicate_share"):
            if m.get(k) is not None:
                qwen[(m["opp"], k)].append(m[k])
        for arm, c in system1(d).items():
            sys1[(m["opp"], arm)].update(c)
        sys1[(m["opp"], "old none")].update(system1(old)["none"])
        qwen[(m["opp"], "violations")].append(m["violations"])
        qwen[(m["opp"], "stopped")].append(sum(1 for x in read(os.path.join(d, "loop", "ledger.jsonl"))
                                               for r in x["registry"] if r["state"] == "rejected"
                                               and str(r.get("why", "")).startswith("stopped")))
    return {"runs": len(new),
            "primary_loop_vs_none": by_run(prim),
            "system1_new_none_vs_old_none": by_run(s1),
            "total_new_loop_vs_old_loop": by_run(total),
            "learning_late_minus_early": {o: run_level(rs) for o, rs in sorted(learn.items())},
            "qwen": {"%s %s" % k: sum(v) / len(v) if "stopped" not in k[1] and "violations" not in k[1] else sum(v)
                     for k, v in sorted(qwen.items())},
            "system1": {"%s %s" % k: dict(v) for k, v in sorted(sys1.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    args = ap.parse_args()
    r = report()
    text = json.dumps(r, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
