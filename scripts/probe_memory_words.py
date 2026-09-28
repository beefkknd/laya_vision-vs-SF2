"""Does laya read advice words? Ask a model a plain a/b/c choice ("which do you do now? one answer") on real game
frames, with and without a memory line ("more a", "avoid a", ...), and see if the answer moves toward the advice.
Also asks the play-time outcome question with and without "use more lp", the way System 1 feeds memory today.

    python scripts/probe_memory_words.py                          # vanilla laya-vision and runs/all8/best
    python scripts/probe_memory_words.py --models runs/all8/best --positions 20

Results: printed table + logs/probe/memory_words_<model>.json (every position's probabilities).
"""
import argparse
import glob
import json
import os
import statistics
from typing import Dict, List

import numpy as np

import _path  # noqa: F401
from sf2.config import BASE_MODEL
from sf2.policy import make_state
from sf2.vs_sweep import outcome_question

SESSION = "rollouts/learn/chunli/20260928-023610"
NOTE = "me=chunli dist=close side=left dx=+40 my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0"

CHOICES = {
    "letters": {"a": "a", "b": "b", "c": "c"},
    "moves": {"a": "lp: a quick jab punch", "b": "hk: a roundhouse kick", "c": "sweep: a low sliding kick"},
}
ASK = {"type": "choice", "instructions": "Which do you do now? Reply with one answer."}
MEMORY = {"none": "", "more a": "memory: more a", "more b": "memory: more b", "more c": "memory: more c",
          "avoid a": "memory: avoid a", "always a": "memory: always choose a, never b or c"}
OUTCOME_MEMORY = {"none": "", "use more lp": "memory vs dhalsim: use more lp up close: it lands",
                  "never lp": "memory vs dhalsim: never use lp: it always misses"}


def positions(n: int) -> List[Dict]:
    nows = sorted(glob.glob(os.path.join(SESSION, "images", "g0[0-9]_*_now.png")))
    if len(nows) < n:
        raise SystemExit("only %d frames under %s" % (len(nows), SESSION))
    pick = [nows[int(i)] for i in np.linspace(0, len(nows) - 1, n)]
    return [{"now": p, "prev": p.replace("_now.png", "_prev.png")} for p in pick]


def load(path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def text(memory: str, with_note: bool) -> str:
    return "\n".join(x for x in ((NOTE if with_note else ""), memory) if x)


def run(model: str, pos: List[Dict]) -> Dict:
    import laya

    agent = laya.load_vlm(model)
    qs = {"q_" + k: dict(ASK, criteria=v) for k, v in CHOICES.items()}
    qs["lp_outcome"] = outcome_question("lp")
    out: Dict = {"model": model, "rows": []}
    for p in pos:
        prev, now = load(p["prev"]), load(p["now"])
        for with_note in (False, True):
            for name, mem in {**MEMORY, **OUTCOME_MEMORY}.items():
                ans = agent.predict(make_state(prev, now, text(mem, with_note)), qs)["answers"]
                out["rows"].append({"frame": os.path.basename(p["now"]), "note": with_note, "memory": name,
                                    **{k: ans[k]["probabilities"] for k in qs},
                                    **{k + "_choice": ans[k]["choice"] for k in qs}})
    return out


def table(res: Dict) -> None:
    rows = res["rows"]
    print("\n=== %s  (%d positions) ===" % (res["model"], len({r["frame"] for r in rows})))
    for with_note in (False, True):
        print("-- prompt %s the game note --" % ("WITH" if with_note else "without"))
        for q in CHOICES:
            print("  choice a/b/c (%s):  memory -> mean P(a) P(b) P(c) | picks a/b/c" % q)
            for m in MEMORY:
                rs = [r for r in rows if r["note"] == with_note and r["memory"] == m]
                mp = [statistics.mean(r["q_" + q][k] for r in rs) for k in "abc"]
                picks = [sum(r["q_%s_choice" % q] == k for r in rs) for k in "abc"]
                print("    %-9s %.3f %.3f %.3f | %2d %2d %2d" % (m, *mp, *picks))
        print("  outcome 'If you do lp now, what happens?': memory -> mean P(hit), picks hit")
        for m in OUTCOME_MEMORY:
            rs = [r for r in rows if r["note"] == with_note and r["memory"] == m]
            print("    %-12s %.3f  %2d/%d" % (m, statistics.mean(r["lp_outcome"]["hit"] for r in rs),
                                            sum(r["lp_outcome_choice"] == "hit" for r in rs), len(rs)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[BASE_MODEL, "runs/all8/best"])
    ap.add_argument("--positions", type=int, default=30)
    args = ap.parse_args()
    pos = positions(args.positions)
    os.makedirs("logs/probe", exist_ok=True)
    for m in args.models:
        res = run(m, pos)
        with open("logs/probe/memory_words_%s.json" % m.replace("/", "_"), "w") as f:
            json.dump(res, f)
        table(res)


if __name__ == "__main__":
    main()
