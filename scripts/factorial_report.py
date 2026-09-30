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

Which run (blind review 2026-09-30): per (opponent, seed, ranking) the NEWEST run whose arms both played at least
--min-rounds rounds (MIN_ROUNDS: the registered 10 games x 3 rounds, qwen_lessons' defaults as the book round ran them)
and the same number; the others are listed in "passed_over" with why (a crashed run never shadows its rerun, a smoke
run with the same seed is never paired). A pair whose settings differ where they must be equal (PAIR_FIELDS, from the
verdict and the arms' run.json, games and rounds per game read from the rounds when not recorded; a field recorded
by one run only is not compared) is refused, and every table run must use one table (oracle_sha256), every runs/all8
run one model: any of these is a problem (exit 1).

    python scripts/factorial_report.py [--label character_fgc+book] [--min-rounds 30] [--json OUT] [ROOT ...]
                                       default roots: rollouts/qwen_lessons
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import _path  # noqa: F401
from compare_prompts import label as run_label
from compare_prompts import opponent
from sf2.data.dataset import read
from sf2.eval.stats import hp, pooled, run_level

ROOTS = [os.path.join("rollouts", "qwen_lessons")]
LABEL = "character_fgc+book"
TABLE = "+table"
CELLS = {"A0": ("all8", "none"), "A1": ("all8", "loop"), "T0": ("table", "none"), "T1": ("table", "loop")}
EFFECTS = ("table", "advice", "interaction", "T0-A0", "T1-T0", "A1-A0")
MIN_ROUNDS = 30        # rounds per arm registered for the 2x2: 10 games x 3 rounds (qwen_lessons' defaults)
PAIR_FIELDS = ("seed", "prompt", "book_sha256", "forward_lessons", "lock", "commit", "advisor", "games", "rounds")
RUN_FIELDS = ("advisor", "commit", "games", "rounds", "model")
Pairs = Dict[Tuple[str, int], Dict[str, str]]


def _json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _count(d: str, arm: str) -> int:
    try:
        with open(os.path.join(d, arm, "rounds.jsonl")) as f:
            return sum(1 for x in f if x.strip())
    except OSError:
        return 0


def _candidates(roots: Sequence[str]) -> List[Tuple[str, str, int, str]]:
    """(dir, opponent, seed, label) of every run with a verdict, oldest first (by the name's time stamp)."""
    out = []
    for root in roots:
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, "verdict.json")):
                v = _json(os.path.join(d, "verdict.json"))
                out.append((name.split("_")[0], name, d, opponent(name), v["seed"], run_label(v)))
    return [x[2:] for x in sorted(out)]


def _unfit(d: str, min_rounds: int) -> str:
    """Why a run cannot be used ("" when it can): an arm short of ``min_rounds`` or the arms' counts differ."""
    n = {arm: _count(d, arm) for arm in ("loop", "none")}
    if min(n.values()) < min_rounds:
        return "fewer than %d rounds in an arm (loop %d, none %d)" % (min_rounds, n["loop"], n["none"])
    if n["loop"] != n["none"]:
        return "the arms played different numbers of rounds (%d vs %d)" % (n["loop"], n["none"])
    return ""


def pick(roots: Sequence[str], label: str = LABEL,
         min_rounds: int = MIN_ROUNDS) -> Tuple[Dict[Tuple[str, int, str], str], List[List[str]]]:
    """{(opp, seed, ranking): the newest usable run}, [[dir, why passed over], ...] for the runs of ``label``."""
    chosen: Dict[Tuple[str, int, str], str] = {}
    passed: List[List[str]] = []
    for d, opp, seed, lab in reversed(_candidates(roots)):          # newest first
        ranking = "table" if lab == label + TABLE else "all8" if lab == label else None
        if not ranking:
            continue
        key = (opp, seed, ranking)
        why = _unfit(d, min_rounds)
        if why:
            passed.append([d, why])
        elif key in chosen:
            passed.append([d, "a newer usable run of this opponent, seed and ranking: %s" % chosen[key]])
        else:
            chosen[key] = d
    return chosen, sorted(passed)


def pairs(roots: Sequence[str], label: str = LABEL,
          min_rounds: int = MIN_ROUNDS) -> Tuple[Pairs, List[Tuple[str, int]]]:
    """({(opp, seed): {"all8": dir, "table": dir}} for (opponent, seed)s with a usable run under both rankings
    (``pick``), [(opp, seed)s with one ranking only])."""
    seen: Pairs = {}
    for (opp, seed, ranking), d in pick(roots, label, min_rounds)[0].items():
        seen.setdefault((opp, seed), {})[ranking] = d
    return ({k: v for k, v in sorted(seen.items()) if len(v) == 2},
            [k for k, v in sorted(seen.items()) if len(v) != 2])


def settings(d: str) -> Dict:
    """A run's settings: its verdict's, its arms' run.json (RUN_FIELDS), and games / rounds per game read from the
    loop arm's rounds when not recorded."""
    v = _json(os.path.join(d, "verdict.json"))
    out = {k: v[k] for k in PAIR_FIELDS + ("oracle_sha256",) if k in v}
    for arm in ("loop", "none"):
        path = os.path.join(d, arm, "run.json")
        run = _json(path) if os.path.exists(path) else {}
        out = dict({k: run[k] for k in RUN_FIELDS if k in run}, **out)
    if "games" not in out or "rounds" not in out:
        games = len({r.get("game") for r in _rounds(d, "loop")})
        out = dict({"games": games, "rounds": _count(d, "loop") // max(games, 1)}, **out)
    return out


def mismatch(a: Dict, b: Dict) -> List[str]:
    """PAIR_FIELDS both runs record and that differ, as "field (a vs b)"."""
    return ["%s (%r vs %r)" % (k, a[k], b[k]) for k in PAIR_FIELDS if k in a and k in b and a[k] != b[k]]


def one_of_each(prs: Pairs) -> List[str]:
    """Every table run on one table, every runs/all8 run on one model (among those that record it)."""
    out = []
    for rk, field in (("table", "oracle_sha256"), ("all8", "model")):
        got = sorted({str(settings(p[rk])[field]) for p in prs.values() if field in settings(p[rk])})
        if len(got) > 1:
            out.append("the %s runs differ in %s: %s" % (rk, field, ", ".join(got)))
    return out


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


def report(roots: Sequence[str], label: str = LABEL, min_rounds: int = MIN_ROUNDS) -> Dict:
    prs, unpaired = pairs(roots, label, min_rounds)
    problems: List[str] = one_of_each(prs)
    good: Pairs = {}
    by: Dict[str, Dict[str, List[List[float]]]] = {e: {} for e in EFFECTS}
    for (opp, seed), p in prs.items():
        differ = mismatch(settings(p["all8"]), settings(p["table"]))
        if differ:
            problems = problems + ["%s seed %d: the runs differ in %s" % (opp, seed, ", ".join(differ))]
            continue
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
            "passed_over": pick(roots, label, min_rounds)[1],
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
    for d, why in rep["passed_over"]:
        print("passed over: %s (%s)" % (d, why))
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=LABEL, help="the runs/all8 runs' label (compare_prompts.label); the table "
                                                   "runs are the same label + '+table'")
    ap.add_argument("--min-rounds", type=int, default=MIN_ROUNDS,
                    help="rounds each arm must have played (default: the registered 10 games x 3 rounds)")
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("roots", nargs="*")
    args = ap.parse_args(argv)
    rep = report(args.roots or ROOTS, args.label, args.min_rounds)
    show(rep)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
