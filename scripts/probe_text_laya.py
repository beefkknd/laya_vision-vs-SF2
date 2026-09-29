"""Does TEXT laya (ModernBERT-large, via laya-mlx) follow advice words? Situations from Chun-Li's real game log, told in
words; the same question with and without a memory line; does the advised option's probability move the right way?

Runs in the laya-mlx venv (inference only):
    ~/work/laya_mlx/.venv/bin/python scripts/probe_text_laya.py
    ~/work/laya_mlx/.venv/bin/python scripts/probe_text_laya.py --model runs/text_laya/advice_v1

Two tests:
  letters  options a / b / c, memory "more a", "avoid a", ...  (the probe laya-vision failed)
  moves    Chun-Li's moves, memory = real short-memory style lessons ("use more lp up close", "avoid sweep",
           "when he jumps, use spinning_bird_kick"); target = the move the lesson names, expected direction up/down
Results: printed table + logs/probe/text_laya_<model>.json
"""
import argparse
import json
import os
import statistics
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2 import text_laya  # noqa: E402
from sf2.advice import INSTRUCTIONS, opp_doing, situation_text  # noqa: E402
from sf2.vocab import bar  # noqa: E402

SESSION = "rollouts/learn/chunli/20260928-023610"

LETTERS = {"a": "a", "b": "b", "c": "c"}
LETTER_MEMORY = {"none": "Advice: none.", "more a": "Advice: more a.", "more b": "Advice: more b.",
                 "avoid a": "Advice: avoid a.", "always a": "Advice: always choose a, never b or c."}

MOVES = {"lp": "may work", "sweep": "may work", "spinning_bird_kick": "may work", "block_high": "may work",
         "forward": "walk in"}      # all rated alike, so only the advice can move the choice
# (name, memory line, target move, expected direction)
MOVE_MEMORY = [
    ("none", "Advice: none.", None, 0),
    ("use more lp", "Advice: use more lp: it lands.", "lp", +1),
    ("avoid sweep", "Advice: avoid sweep: it whiffs and gets punished.", "sweep", -1),
    ("sbk when jumps", "Advice: when he jumps, use spinning_bird_kick.", "spinning_bird_kick", +1),
    ("never sbk", "Advice: never use spinning_bird_kick: he punishes it.", "spinning_bird_kick", -1),
    ("block more", "Advice: use more block_high: he hits you when you walk in.", "block_high", +1),
    ("stop walking", "Advice: stop walking forward into him.", "forward", -1),
]


def situations(n: int) -> List[str]:
    """Real moments from Chun-Li's first 10 games, in the words sf2.advice gives text laya at play time."""
    rows = [json.loads(x) for x in open(os.path.join(SESSION, "actions.jsonl"))]
    rows = [r for r in rows if r["game"] < 10]
    pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, n)]
    return [situation_text(r["range"], opp_doing(r), bar(r["my_life"]), bar(r["opp_life"])) for r in pick]


def ask(agent, text: str, criteria: Dict[str, str]) -> Dict[str, float]:
    q = {"q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}
    return agent.predict(text, q)["answers"]["q"]["probabilities"]


def run(model: str, sits: List[str]) -> Dict:
    agent = text_laya.load(None if model == text_laya.BASE else model)
    rows = []
    for s in sits:
        for name, mem in LETTER_MEMORY.items():
            rows.append({"test": "letters", "memory": name, "situation": s,
                         "p": ask(agent, (s + " " + mem).strip(), LETTERS)})
        for name, mem, _, _ in MOVE_MEMORY:
            rows.append({"test": "moves", "memory": name, "situation": s,
                         "p": ask(agent, (s + " " + mem).strip(), MOVES)})
    return {"model": model, "rows": rows}


def table(res: Dict) -> None:
    rows = res["rows"]
    n = len({r["situation"] for r in rows})
    print("\n=== %s  (%d situations) ===" % (res["model"], n))
    print("letters: memory -> mean P(a) P(b) P(c) | picks a/b/c")
    for m in LETTER_MEMORY:
        rs = [r for r in rows if r["test"] == "letters" and r["memory"] == m]
        mp = [statistics.mean(r["p"][k] for r in rs) for k in "abc"]
        picks = [sum(max(r["p"], key=r["p"].get) == k for r in rs) for k in "abc"]
        print("  %-9s %.3f %.3f %.3f | %2d %2d %2d" % (m, *mp, *picks))
    base = {r["situation"]: r["p"] for r in rows if r["test"] == "moves" and r["memory"] == "none"}
    print("moves: memory -> target  P(target) none -> with  | moved the right way in k/n situations | picks")
    for name, _, target, sign in MOVE_MEMORY:
        rs = [r for r in rows if r["test"] == "moves" and r["memory"] == name]
        picks = dict(sorted(((k, sum(max(r["p"], key=r["p"].get) == k for r in rs)) for k in MOVES),
                            key=lambda kv: -kv[1]))
        if target is None:
            print("  %-15s %-19s %s" % (name, "", {k: v for k, v in picks.items() if v}))
            continue
        before = statistics.mean(base[r["situation"]][target] for r in rs)
        after = statistics.mean(r["p"][target] for r in rs)
        right = sum((r["p"][target] - base[r["situation"]][target]) * sign > 0 for r in rs)
        print("  %-15s %-19s %.3f -> %.3f  %s  %2d/%d  %s" % (name, target, before, after, "up" if sign > 0 else "down",
                                                           right, len(rs), {k: v for k, v in picks.items() if v}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=text_laya.BASE, help="the base, or a runs/text_laya/<name> checkpoint")
    ap.add_argument("--situations", type=int, default=30)
    args = ap.parse_args()
    res = run(args.model, situations(args.situations))
    os.makedirs("logs/probe", exist_ok=True)
    with open("logs/probe/text_laya_%s.json" % args.model.replace("/", "_"), "w") as f:
        json.dump(res, f)
    table(res)


if __name__ == "__main__":
    main()
