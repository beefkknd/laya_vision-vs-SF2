"""Character prompt vs the two-view prompt: lesson-loop runs paired by (opponent, seed), each prompt's loop arm against
the same no-advice arm (byte-identical across the two runs, or the pair is refused). Per opponent and pooled with the
opponent as the unit (sf2.eval.stats.pooled).

    python scripts/compare_prompts.py [--a views] [--b character] [ROOT ...]
                                        default roots: rollouts/qwen_lessons rollouts/locked/lesson_loop_v1
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.eval.stats import ci, paired, pooled

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
            out.append((name.split("_")[0], d, opponent(name), v["seed"], v.get("prompt", "views")))
    return [x[1:] for x in sorted(out)]


def _none(d: str) -> bytes:
    with open(os.path.join(d, "none", "actions.jsonl"), "rb") as f:
        return f.read()


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


def diffs(prs: Dict[Tuple[str, int], Dict[str, str]], prompts: Sequence[str] = PROMPTS) -> Dict[str, Dict[str, List[float]]]:
    """Per opponent, paired per round: each prompt vs no advice (hp and damage taken less), and b - a."""
    a, b = prompts
    out: Dict[str, Dict[str, List[float]]] = {k: {} for k in (a, b, "b_minus_a", a + "_taken_less", b + "_taken_less")}
    for (opp, _), by in prs.items():
        none = _rounds(by[a], "none")
        loop = {p: _rounds(by[p], "loop") for p in prompts}
        for p in prompts:
            out[p].setdefault(opp, []).extend(paired(loop[p], none))
            out[p + "_taken_less"].setdefault(opp, []).extend(n["taken"] - x["taken"] for x, n in zip(loop[p], none))
        out["b_minus_a"].setdefault(opp, []).extend(paired(loop[b], loop[a]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="views")
    ap.add_argument("--b", default="character")
    ap.add_argument("roots", nargs="*")
    args = ap.parse_args()
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
    for k, by_opp in diffs(prs, prompts).items():
        print("\n%s (per round):" % k)
        for opp, d in sorted(by_opp.items()):
            print("   %-6s %+.1f [%+.1f, %+.1f]  n=%d" % ((opp,) + ci(d) + (len(d),)))
        print("   pooled", json.dumps(pooled(by_opp)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
