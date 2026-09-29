"""Character prompt vs the two-view prompt: lesson-loop runs paired by (opponent, seed), each prompt's loop arm against
the same no-advice arm (byte-identical across the two runs, or the pair is refused). Per opponent and pooled with the
opponent as the unit (sf2.eval.stats.pooled).

    python scripts/compare_prompts.py [ROOT ...]        default: rollouts/qwen_lessons rollouts/locked/lesson_loop_v1
"""
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


def pairs(roots: Sequence[str]) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], List[str]]:
    """{(opp, seed): {"views": dir, "character": dir}} for seeds run with both prompts (the first run of each)."""
    seen: Dict[Tuple[str, int], Dict[str, str]] = {}
    for d, opp, seed, prompt in _runs(roots):
        seen.setdefault((opp, seed), {}).setdefault(prompt, d)
    out, problems = {}, []
    for k, by in sorted(seen.items()):
        if set(PROMPTS) - set(by):
            continue
        if _none(by["views"]) != _none(by["character"]):
            problems.append("%s seed %d: the no-advice arms differ (%s, %s)" % (k + (by["views"], by["character"])))
            continue
        out[k] = by
    return out, problems


def _rounds(d: str, arm: str) -> List[Dict]:
    return read(os.path.join(d, arm, "rounds.jsonl"))


def diffs(prs: Dict[Tuple[str, int], Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    """Per opponent, paired per round: each prompt vs no advice (hp and damage taken less), and character - views."""
    out: Dict[str, Dict[str, List[float]]] = {k: {} for k in ("views", "character", "char_minus_views",
                                                             "views_taken_less", "character_taken_less")}
    for (opp, _), by in prs.items():
        none = _rounds(by["views"], "none")
        loop = {p: _rounds(by[p], "loop") for p in PROMPTS}
        for p in PROMPTS:
            out[p].setdefault(opp, []).extend(paired(loop[p], none))
            out[p + "_taken_less"].setdefault(opp, []).extend(b["taken"] - a["taken"] for a, b in zip(loop[p], none))
        out["char_minus_views"].setdefault(opp, []).extend(paired(loop["character"], loop["views"]))
    return out


def main() -> int:
    prs, problems = pairs(sys.argv[1:] or ROOTS)
    for p in problems:
        print("REFUSED:", p)
    print("%d paired seeds: %s" % (len(prs), ", ".join("%s %d" % k for k in prs)))
    for (opp, seed), by in prs.items():
        v = {p: json.load(open(os.path.join(by[p], "verdict.json"))) for p in PROMPTS}
        print("  %-6s %6d  hold views %.2f character %.2f (chance %.2f)  violations %d/%d" % (
            opp, seed, v["views"]["qwen_hold_rate"] or 0, v["character"]["qwen_hold_rate"] or 0,
            v["character"]["random_hold_rate"] or 0, v["views"]["violations"], v["character"]["violations"]))
    for k, by_opp in diffs(prs).items():
        print("\n%s (per round):" % k)
        for opp, d in sorted(by_opp.items()):
            print("   %-6s %+.1f [%+.1f, %+.1f]  n=%d" % ((opp,) + ci(d) + (len(d),)))
        print("   pooled", json.dumps(pooled(by_opp)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
