"""The 2x2 of docs/prereg_2x2.md: System 1 ranking {runs/all8 P(hit), lookup table} x advice {none, Qwen's loop + the
book}. Each lesson-loop run (scripts/qwen_lessons.py) plays its loop arm and its no-advice arm on one seed; a
runs/all8 run and a table run (--oracle, its run name ends "+table") of the same (opponent, seed) and label make one
unit with four cells, paired round by round (the same seed: the same start delays):

    A0  runs/all8, no advice      A1  runs/all8, loop + book
    T0  table, no advice          T1  table, loop + book

Effects per round: T0-A0 (the table without advice), T1-T0 (Qwen + book on the table), A1-A0 (Qwen + book on
runs/all8), table = mean over advice of (T - A), advice = mean over rankings of (1 - 0), interaction = (T1-T0) - (A1-A0).
Per opponent the run (seed) is the unit (sf2.eval.stats.run_level); pooled the opponent (sf2.eval.stats.pooled).
Also per cell: hp, damage taken and rounds won per round, throws per close-range decision; per ranking: Qwen's
registered lessons at the end and hold rate (the verdicts).

    python scripts/factorial_report.py [--label character_fgc+book] [--json OUT] [ROOT ...]
                                       default roots: rollouts/qwen_lessons
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import _path  # noqa: F401
from compare_prompts import _runs
from sf2.data.dataset import read
from sf2.eval.stats import hp, pooled, run_level

ROOTS = [os.path.join("rollouts", "qwen_lessons")]
LABEL = "character_fgc+book"
TABLE = "+table"
CELLS = {"A0": ("all8", "none"), "A1": ("all8", "loop"), "T0": ("table", "none"), "T1": ("table", "loop")}
EFFECTS = ("table", "advice", "interaction", "T0-A0", "T1-T0", "A1-A0")
Pairs = Dict[Tuple[str, int], Dict[str, str]]


def pairs(roots: Sequence[str], label: str = LABEL) -> Tuple[Pairs, List[Tuple[str, int]]]:
    """({(opp, seed): {"all8": dir, "table": dir}} for (opponent, seed)s run under both rankings with ``label`` (the
    first run of each, oldest first), [(opp, seed)s run under one ranking only])."""
    seen: Pairs = {}
    for d, opp, seed, lab in _runs(roots):
        ranking = "table" if lab == label + TABLE else "all8" if lab == label else None
        if ranking:
            seen.setdefault((opp, seed), {}).setdefault(ranking, d)
    return ({k: v for k, v in sorted(seen.items()) if len(v) == 2},
            [k for k, v in sorted(seen.items()) if len(v) != 2])


def _rounds(d: str, arm: str) -> List[Dict]:
    return read(os.path.join(d, arm, "rounds.jsonl"))


def cells(pair: Dict[str, str]) -> Dict[str, List[Dict]]:
    """The four cells' rounds; unequal round counts are an error (a crashed or cut arm), never truncated."""
    out = {c: _rounds(pair[rk], arm) for c, (rk, arm) in CELLS.items()}
    n = {c: len(rs) for c, rs in out.items()}
    if len(set(n.values())) != 1:
        raise ValueError("the four cells played different numbers of rounds: %s" % n)
    return out


def effects(c: Dict[str, List[Dict]]) -> Dict[str, List[float]]:
    """Per round: the simple effects, both main effects and the interaction (in hp: dealt - taken)."""
    a0, a1, t0, t1 = ([hp(r) for r in c[k]] for k in ("A0", "A1", "T0", "T1"))
    rows = list(zip(a0, a1, t0, t1))
    return {"T0-A0": [t0_ - a0_ for a0_, _, t0_, _ in rows],
            "T1-T0": [t1_ - t0_ for _, _, t0_, t1_ in rows],
            "A1-A0": [a1_ - a0_ for a0_, a1_, _, _ in rows],
            "table": [((t1_ + t0_) - (a1_ + a0_)) / 2 for a0_, a1_, t0_, t1_ in rows],
            "advice": [((t1_ - t0_) + (a1_ - a0_)) / 2 for a0_, a1_, t0_, t1_ in rows],
            "interaction": [(t1_ - t0_) - (a1_ - a0_) for a0_, a1_, t0_, t1_ in rows]}


def _ratio(k: int, n: int):
    return k / n if n else None


def cell_stats(prs: Pairs) -> Dict[str, Dict]:
    """Per cell over every paired run: hp, damage taken and rounds won per round; throws per close-range decision."""
    out = {}
    for c, (rk, arm) in CELLS.items():
        rs = [r for p in prs.values() for r in _rounds(p[rk], arm)]
        acts = [a for p in prs.values() for a in read(os.path.join(p[rk], arm, "actions.jsonl"), missing_ok=True)]
        close = [a for a in acts if a.get("range") == "close"]
        out[c] = {"rounds": len(rs), "hp": _ratio(sum(map(hp, rs)), len(rs)),
                  "taken": _ratio(sum(r["taken"] for r in rs), len(rs)),
                  "won": sum(r["result"] == "win" for r in rs), "close": len(close),
                  "throws_per_close": _ratio(sum(a["action"] == "throw" for a in close), len(close))}
    return out


def qwen_stats(prs: Pairs) -> Dict[str, Dict]:
    """Per ranking, over its loop arms: Qwen's registered lessons at the end and hold rate (mean per run)."""
    out = {}
    for rk in ("all8", "table"):
        vs = [json.load(open(os.path.join(p[rk], "verdict.json"))) for p in prs.values()]
        holds = [v["qwen_hold_rate"] for v in vs if v.get("qwen_hold_rate") is not None]
        out[rk] = {"runs": len(vs), "registered_mean": _ratio(sum(len(v.get("registered_at_end", [])) for v in vs),
                                                              len(vs)),
                   "hold_rate_mean": _ratio(sum(holds), len(holds))}
    return out


def report(roots: Sequence[str], label: str = LABEL) -> Dict:
    prs, unpaired = pairs(roots, label)
    problems: List[str] = []
    good: Pairs = {}
    by: Dict[str, Dict[str, List[List[float]]]] = {e: {} for e in EFFECTS}
    for (opp, seed), p in prs.items():
        try:
            e = effects(cells(p))
        except (ValueError, OSError) as err:
            problems = problems + ["%s seed %d: %s" % (opp, seed, err)]
            continue
        good[(opp, seed)] = p
        for k in EFFECTS:
            by[k].setdefault(opp, []).append(e[k])
    return {"label": label, "pairs": [[o, s, p["all8"], p["table"]] for (o, s), p in good.items()],
            "problems": problems, "unpaired": [list(k) for k in unpaired],
            "effects": {k: {"per_opp": {o: run_level(runs) for o, runs in sorted(v.items())},
                            "pooled": pooled({o: [x for r in runs for x in r] for o, runs in v.items()})}
                        for k, v in by.items()},
            "cells": cell_stats(good), "qwen": qwen_stats(good)}


def _fmt(r: Dict) -> str:
    if "ci95" not in r:
        return r["verdict"]
    return "%+.1f [%+.1f, %+.1f] %s" % (r["mean"], r["ci95"][0], r["ci95"][1], r["verdict"])


def show(rep: Dict) -> None:
    for p in rep["problems"]:
        print("REFUSED:", p)
    print("%s: %d paired (opponent, seed)s: %s" % (rep["label"], len(rep["pairs"]),
                                                   ", ".join("%s %d" % (o, s) for o, s, _, _ in rep["pairs"])))
    if rep["unpaired"]:
        print("run under one ranking only:", ", ".join("%s %d" % tuple(k) for k in rep["unpaired"]))
    for k, v in rep["effects"].items():
        print("\n%s (hp per round; run as unit per opponent, opponent pooled):" % k)
        for opp, r in v["per_opp"].items():
            print("   %-8s %s  runs=%d rounds=%d" % (opp, _fmt(r), r["runs"], r["rounds"]))
        print("   pooled   %s" % _fmt(v["pooled"]))
    print("\ncells:")
    for c, s in rep["cells"].items():
        print("   %s  rounds %d  hp %s  taken %s  won %d  throws/close %s (%d close)" % (
            c, s["rounds"], _num(s["hp"]), _num(s["taken"]), s["won"], _num(s["throws_per_close"], 2), s["close"]))
    print("\nQwen per ranking:", json.dumps(rep["qwen"]))


def _num(x, digits: int = 1) -> str:
    return "-" if x is None else "%.*f" % (digits, x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=LABEL, help="the runs/all8 runs' label (compare_prompts.label); the table "
                                                   "runs are the same label + '+table'")
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("roots", nargs="*")
    args = ap.parse_args()
    rep = report(args.roots or ROOTS, args.label)
    show(rep)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
