"""Character prompt vs the two-view prompt: lesson-loop runs paired by (opponent, seed), each prompt's loop arm against
the same no-advice arm (byte-identical across the two runs, or the pair is refused). Per opponent with the RUN (seed)
as the unit (every round of a run shares its lessons: sf2.eval.stats.run_level, a verdict needs 3+ runs), pooled with
the opponent as the unit (sf2.eval.stats.pooled).

    python scripts/compare_prompts.py [--a views] [--b character] [ROOT ...]
    python scripts/compare_prompts.py --each [ROOT ...]     every prompt vs its own no-advice arm, all finished runs
                                        default roots: rollouts/qwen_lessons rollouts/locked/lesson_loop_v1
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.eval.stats import paired, pooled, run_level

ROOTS = [os.path.join("rollouts", "qwen_lessons"), os.path.join("rollouts", "locked", "lesson_loop_v1")]
PROMPTS = ("views", "character")


def opponent(name: str) -> str:
    return name.split("_")[1]


def _runs(roots: Sequence[str]) -> List[Tuple[str, str, int, str]]:
    """(dir, opponent, seed, prompt) for every finished run, oldest first."""
    out = []
    for root in roots:
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, name)
            path = os.path.join(d, "verdict.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                v = json.load(f)
            out.append((name.split("_")[0], d, opponent(name), v["seed"],
                        v.get("prompt", "views") + ("+track" if v.get("track") else "")))
    return [x[1:] for x in sorted(out)]


LOG_ONLY = ("opp_move", "opp_shot")      # added to the log 2026-09-29 (sf2/system1/opp_moves.py); play is unchanged


def _none(d: str):
    """The no-advice arm's decisions without the logging-only fields (compared as data, not bytes)."""
    with open(os.path.join(d, "none", "actions.jsonl"), "rb") as f:
        raw = f.read()
    try:
        return [{k: v for k, v in json.loads(x).items() if k not in LOG_ONLY} for x in raw.splitlines() if x.strip()]
    except ValueError:
        return raw


def pairs(roots: Sequence[str], prompts: Sequence[str] = PROMPTS) -> Tuple[Dict[Tuple[str, int], Dict[str, str]],
                                                                       List[str]]:
    """{(opp, seed): {a: dir, b: dir}} for seeds run with both prompts (the first run of each)."""
    seen: Dict[Tuple[str, int], Dict[str, str]] = {}
    for d, opp, seed, prompt in _runs(roots):
        seen.setdefault((opp, seed), {}).setdefault(prompt, d)
    out, problems = {}, []
    for k, by in sorted(seen.items()):
        a, b = prompts
        if a not in by or b not in by:
            continue
        if _none(by[a]) != _none(by[b]):
            problems.append("%s seed %d: the no-advice arms differ (%s, %s)" % (k + (by[a], by[b])))
            continue
        out[k] = {a: by[a], b: by[b]}
    return out, problems


def _rounds(d: str, arm: str) -> List[Dict]:
    return read(os.path.join(d, arm, "rounds.jsonl"))


def run_diffs(prs: Dict[Tuple[str, int], Dict[str, str]],
              prompts: Sequence[str] = PROMPTS) -> Dict[str, Dict[str, List[List[float]]]]:
    """Per opponent, one list per run (seed), paired per round: each prompt vs no advice (hp and damage taken less),
    and b - a."""
    a, b = prompts
    out: Dict[str, Dict[str, List[List[float]]]] = {k: {} for k in (a, b, "b_minus_a", a + "_taken_less",
                                                                     b + "_taken_less")}
    for (opp, _), by in prs.items():
        none = _rounds(by[a], "none")
        loop = {p: _rounds(by[p], "loop") for p in prompts}
        for p in prompts:
            out[p].setdefault(opp, []).append(paired(loop[p], none))
            out[p + "_taken_less"].setdefault(opp, []).append([n["taken"] - x["taken"] for x, n in zip(loop[p], none)])
        out["b_minus_a"].setdefault(opp, []).append(paired(loop[b], loop[a]))
    return out


def flat(by_opp: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    return {o: [x for r in runs for x in r] for o, runs in by_opp.items()}


def diffs(prs: Dict[Tuple[str, int], Dict[str, str]], prompts: Sequence[str] = PROMPTS) -> Dict[str, Dict[str, List[float]]]:
    """``run_diffs`` with each opponent's runs joined (per round)."""
    return {k: flat(v) for k, v in run_diffs(prs, prompts).items()}


def each_prompt(roots: Sequence[str]) -> Tuple[Dict[str, Dict[str, List[List[float]]]], List[str]]:
    """{prompt: {opp: [per-run paired differences, loop - none]}} over every finished run (a repeated seed is another
    run: Qwen does not repeat itself); a run whose arms played different numbers of rounds is refused."""
    out: Dict[str, Dict[str, List[List[float]]]] = {}
    problems = []
    for d, opp, _, prompt in _runs(roots):
        try:
            got = paired(_rounds(d, "loop"), _rounds(d, "none"))
        except (ValueError, OSError) as e:
            problems.append("%s: %s" % (d, e))
            continue
        out.setdefault(prompt, {}).setdefault(opp, []).append(got)
    return out, problems


def report(by_opp: Dict[str, List[List[float]]]) -> Dict:
    """Per opponent the run as the unit; pooled the opponent as the unit."""
    return {"per_opp": {o: run_level(runs) for o, runs in sorted(by_opp.items())}, "pooled": pooled(flat(by_opp))}


def show(title: str, by_opp: Dict[str, List[List[float]]]) -> None:
    rep = report(by_opp)
    print("\n%s (run as unit per opponent):" % title)
    for opp, r in rep["per_opp"].items():
        if r["runs"]:
            print("   %-6s %+.1f [%+.1f, %+.1f]  runs=%d rounds=%d  %s" % (opp, r["mean"], r["ci95"][0], r["ci95"][1],
                                                                       r["runs"], r["rounds"], r["verdict"]))
    print("   pooled", json.dumps(rep["pooled"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="views")
    ap.add_argument("--b", default="character")
    ap.add_argument("--each", action="store_true", help="every prompt vs its own no-advice arm, all finished runs")
    ap.add_argument("roots", nargs="*")
    args = ap.parse_args()
    if args.each:
        each, problems = each_prompt(args.roots or ROOTS)
        for p in problems:
            print("REFUSED:", p)
        for prompt, by_opp in sorted(each.items()):
            show(prompt + " vs none", by_opp)
        return 1 if problems else 0
    prompts = (args.a, args.b)
    prs, problems = pairs(args.roots or ROOTS, prompts)
    for p in problems:
        print("REFUSED:", p)
    print("%d paired seeds: %s" % (len(prs), ", ".join("%s %d" % k for k in prs)))
    for (opp, seed), by in prs.items():
        v = {p: json.load(open(os.path.join(by[p], "verdict.json"))) for p in prompts}
        print("  %-6s %6d  hold %s %.2f %s %.2f (chance %.2f)  violations %d/%d" % (
            opp, seed, args.a, v[args.a]["qwen_hold_rate"] or 0, args.b, v[args.b]["qwen_hold_rate"] or 0,
            v[args.b]["random_hold_rate"] or 0, v[args.a]["violations"], v[args.b]["violations"]))
    for k, by_opp in run_diffs(prs, prompts).items():
        show(k, by_opp)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
