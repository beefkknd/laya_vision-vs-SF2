"""Does System 1 (laya-vision + text laya) follow the lessons Qwen gives it? Read-only numbers from finished
lesson-loop runs (scripts/qwen_lessons.py), to budget a fine-tune of text laya (and maybe laya-vision).

    python scripts/laya_evidence.py [--root DIR ...] [--out evidence.json]

For every run with a verdict.json, for every lesson line in play in the loop arm (rounds.jsonl ``lines``), over the
loop-arm decisions of those rounds where the lesson APPLIES (sf2.system1.advisor.applicable, the same rule text laya
is shown lessons by: range and what he is doing):
    n, followed, compliance   soft/hard: share that picked the named move; neg: share that did NOT pick it
    base_n, base_followed, base   the same share in the paired no-advice arm, same rounds, same situation
    effect                    compliance - base
    top3                      share where laya-vision had the named move in its top 3; compliance_in_top3 /
                              compliance_out_top3 split compliance by it ("text laya ignored it" vs "rated out")
    shortlist                 share where the named move was on text laya's shortlist (rows that log one)
Pooled (decision-weighted) by polarity, by the named move's kind (attack / block / forward) and by prompt
(loop/run.json ``prompt``; missing = views). Unfinished runs (no verdict.json) are listed as skipped; a run that
cannot be read is listed in ``failures``, never dropped silently.

From loop/ledger.jsonl: Qwen claims mentioning an opponent move the lesson grammar cannot name (fireball, dragon
punch, ...), per word per opponent, with example ``why`` texts; and the number of ``problems`` entries.
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.data.vs_defense import BLOCKS
from sf2.system1.advice import FORWARD, opp_doing
from sf2.system1.advice import read as read_lesson
from sf2.system1.advisor import applicable
from sf2.system1.system1 import choices

ME = "chunli"
MOVES = [m for m in choices(ME) if m != FORWARD]          # as System1 passes them (attacks + blocks)
ROOTS = (os.path.join("rollouts", "locked", "lesson_loop_v1"), os.path.join("rollouts", "qwen_lessons"))
ARMS = ("loop", "none")
OPP_WORDS = ("fireball", "hadoken", "shoryuken", "uppercut", "dragon punch", "hurricane", "tatsu", "headbutt",
             "head butt", "torpedo", "hundred hand", "slap", "sumo", "spinning")
# "spinning" alone is his (hurricane kick); "spinning bird kick" / spinning_bird_kick is her own move
OPP_RE = {w: re.compile(r"\b%s\b" % (r"spinning(?![\s_-]*bird)" if w == "spinning" else re.escape(w)), re.I)
          for w in OPP_WORDS}
MAX_EXAMPLES = 5


def kind_of(move: str) -> str:
    return "forward" if move == FORWARD else "block" if move in BLOCKS else "attack"


def situation(row: Dict) -> tuple:
    return row["range"], opp_doing(row)


def applies(line: str, row: Dict) -> bool:
    return bool(applicable([line], MOVES, situation(row)))


def follows(polarity: str, move: str, row: Dict) -> bool:
    picked = row["action"] == move
    return not picked if polarity == "neg" else picked


def share(k: int, n: int) -> Optional[float]:
    return k / n if n else None


def _read_arm(run: str, arm: str) -> Dict:
    d = os.path.join(run, arm)
    acts = read(os.path.join(d, "actions.jsonl"))
    rounds = read(os.path.join(d, "rounds.jsonl"))
    bad = [a for a in acts if not {"action", "range", "round", "top3"} <= a.keys()]
    if bad:
        raise ValueError("%s: %d action rows lack action/range/round/top3" % (arm, len(bad)))
    info = {}
    if os.path.exists(os.path.join(d, "run.json")):
        with open(os.path.join(d, "run.json")) as f:
            info = json.load(f)
    return {"actions": acts, "rounds": rounds, "run": info}


def lines_by_round(rounds: List[Dict]) -> Dict[int, List[str]]:
    missing = [r for r in rounds if "lines" not in r or "round" not in r]
    if missing:
        raise ValueError("loop: %d rounds.jsonl rows lack round/lines" % len(missing))
    return {r["round"]: r["lines"] for r in rounds}


def lesson_evidence(line: str, loop: List[Dict], none: List[Dict]) -> Dict:
    les = read_lesson(line, MOVES + [FORWARD])
    mine = [a for a in loop if applies(line, a)]
    base = [a for a in none if applies(line, a)]
    followed = sum(follows(les.polarity, les.move, a) for a in mine)
    base_followed = sum(follows(les.polarity, les.move, a) for a in base)
    rated_in = [a for a in mine if les.move in [m for m, _ in a["top3"]]]
    rated_out = [a for a in mine if les.move not in [m for m, _ in a["top3"]]]
    top3 = len(rated_in)
    with_sl = [a for a in mine if a.get("shortlist")]
    comp, b = share(followed, len(mine)), share(base_followed, len(base))
    return {"line": line, "move": les.move, "polarity": les.polarity, "kind": kind_of(les.move),
            "where": les.where, "when": les.when, "n": len(mine), "followed": followed, "compliance": comp,
            "base_n": len(base), "base_followed": base_followed, "base": b,
            "effect": comp - b if comp is not None and b is not None else None,
            "top3_n": top3, "top3": share(top3, len(mine)),
            "n_in_top3": len(rated_in), "followed_in_top3": sum(follows(les.polarity, les.move, a) for a in rated_in),
            "n_out_top3": len(rated_out),
            "followed_out_top3": sum(follows(les.polarity, les.move, a) for a in rated_out),
            "shortlist_n": sum(les.move in a["shortlist"] for a in with_sl), "shortlist_rows": len(with_sl),
            "shortlist": share(sum(les.move in a["shortlist"] for a in with_sl), len(with_sl))}


def situation_mismatch(acts: List[Dict]) -> int:
    """Decisions whose logged advice_text disagrees with the situation read from the row (a check on this script's
    reading of range / what he is doing against what text laya was really told)."""
    words = {"close": "up close", "mid": "at mid range", "far": "far away"}
    return sum(1 for a in acts if a.get("advice_text")
               and not a["advice_text"].startswith("He is %s and %s." % (words.get(a["range"]), opp_doing(a))))


def run_evidence(run: str) -> Dict:
    arms = {arm: _read_arm(run, arm) for arm in ARMS}
    in_play = lines_by_round(arms["loop"]["rounds"])
    prompt = arms["loop"]["run"].get("prompt") or "views"
    opp = arms["loop"]["run"].get("opp") or os.path.basename(run).split("_")[1]
    lessons, no_move = [], []
    for line in sorted({t for ls in in_play.values() for t in ls}):
        rounds = {r for r, ls in in_play.items() if line in ls}
        if read_lesson(line, MOVES + [FORWARD]).move is None:
            no_move.append(line)
            continue
        loop = [a for a in arms["loop"]["actions"] if a["round"] in rounds]
        none = [a for a in arms["none"]["actions"] if a["round"] in rounds]
        lessons.append(dict(lesson_evidence(line, loop, none), rounds=len(rounds)))
    return {"run": run, "opp": opp, "prompt": prompt, "decisions": {a: len(arms[a]["actions"]) for a in ARMS},
            "situation_mismatch": situation_mismatch(arms["loop"]["actions"]),
            "lessons": lessons, "no_move_lines": no_move}


def pool(lessons: Sequence[Dict]) -> Dict:
    n, k = sum(x["n"] for x in lessons), sum(x["followed"] for x in lessons)
    bn, bk = sum(x["base_n"] for x in lessons), sum(x["base_followed"] for x in lessons)
    t3 = sum(x["top3_n"] for x in lessons)
    sn, sk = sum(x["shortlist_rows"] for x in lessons), sum(x["shortlist_n"] for x in lessons)
    comp, base = share(k, n), share(bk, bn)
    return {"lessons": len(lessons), "applied_lessons": sum(x["n"] > 0 for x in lessons), "n": n, "followed": k,
            "compliance": comp, "base_n": bn, "base_followed": bk, "base": base,
            "effect": comp - base if comp is not None and base is not None else None,
            "top3": share(t3, n), "shortlist": share(sk, sn),
            "compliance_in_top3": share(sum(x["followed_in_top3"] for x in lessons), sum(x["n_in_top3"] for x in lessons)),
            "compliance_out_top3": share(sum(x["followed_out_top3"] for x in lessons),
                                         sum(x["n_out_top3"] for x in lessons))}


def group(runs: List[Dict], key) -> Dict[str, Dict]:
    out: Dict[str, List[Dict]] = {}
    for r in runs:
        for x in r["lessons"]:
            out.setdefault(key(r, x), []).append(x)
    return {k: pool(v) for k, v in sorted(out.items())}


def mentions(ledgers: Dict[str, List[Dict]]) -> Dict:
    """``ledgers``: opponent -> ledger rows (every finished run's). A claim counts once per word it mentions."""
    counts: Dict[str, Dict[str, int]] = {}
    examples: List[str] = []
    claims = hits = 0
    for opp, rows in sorted(ledgers.items()):
        for row in rows:
            for c in row.get("claims") or []:
                claims += 1
                text = " ".join(str(v) for v in (c.values() if isinstance(c, dict) else [c]))
                found = [w for w, pat in OPP_RE.items() if pat.search(text)]
                if not found:
                    continue
                hits += 1
                for w in found:
                    counts.setdefault(opp, {})[w] = counts.get(opp, {}).get(w, 0) + 1
                why = c.get("why") if isinstance(c, dict) else None
                if len(examples) < MAX_EXAMPLES:
                    examples.append(why or text)
    return {"claims": claims, "claims_mentioning": hits, "counts": counts, "examples": examples}


def problems(ledgers: Dict[str, List[Dict]]) -> Dict:
    by = {opp: sum(len(r.get("problems") or []) for r in rows) for opp, rows in sorted(ledgers.items())}
    return {"total": sum(by.values()), "by_opp": {o: n for o, n in by.items() if n}}


def run_dirs(roots: Sequence[str]) -> List[str]:
    return sorted(os.path.join(root, d) for root in roots if os.path.isdir(root) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d, "loop")) or os.path.isdir(os.path.join(root, d, "none")))


def collect(roots: Sequence[str] = ROOTS) -> Dict:
    runs, failures, skipped = [], [], []
    ledgers: Dict[str, List[Dict]] = {}
    for d in run_dirs(roots):
        if not os.path.exists(os.path.join(d, "verdict.json")):
            skipped.append(d)
            continue
        try:
            r = run_evidence(d)
            led = read(os.path.join(d, "loop", "ledger.jsonl"))
        except Exception as e:           # reported, never silently skipped
            failures.append({"run": d, "error": "%s: %s" % (type(e).__name__, e)})
            continue
        runs.append(r)
        ledgers.setdefault(r["opp"], []).extend(led)
    lessons = [x for r in runs for x in r["lessons"]]
    return {"runs_read": len(runs), "all": pool(lessons),
            "by_polarity": group(runs, lambda r, x: x["polarity"]),
            "by_kind": group(runs, lambda r, x: x["kind"]),
            "by_prompt": group(runs, lambda r, x: r["prompt"]),
            "by_prompt_polarity": group(runs, lambda r, x: "%s/%s" % (r["prompt"], x["polarity"])),
            "by_opp": group(runs, lambda r, x: r["opp"]),
            "situation_mismatch": sum(r["situation_mismatch"] for r in runs),
            "opponent_moves": mentions(ledgers), "ledger_problems": problems(ledgers),
            "failures": failures, "skipped_unfinished": skipped, "runs": runs}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="a folder of runs (repeatable; default: %s)" % ", ".join(ROOTS))
    ap.add_argument("--out", help="write the JSON here (default: print it)")
    args = ap.parse_args(argv)
    doc = collect(args.root or ROOTS)
    text = json.dumps(doc, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
