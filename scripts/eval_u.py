"""Offline gates of the U arm's eye (docs/prereg_u_perception.md "Offline gates") on test_data_u (built by
scripts/build_u_data.py): one predict per decision with every question, exactly as System 1's eye asks them
(sf2.system1.eye.Eye.ask, frames v3, note "me=<char>").

    python scripts/eval_u.py --model runs/u/best [--data test_data_u] [--floors FLOORS.json] [--decisions N]

Per file (test_real, test_heldout_guile, and val for reference), per character and pooled:
  accuracy  per question (1-7; question 8's word too, reported); masked ("unknown") labels have no row, so they are
            never counted
  q2        the confusion of question 2 (true answer -> eye's answer)
  gate 2    the table's best move (lessons/value_oracle_v1.json, by the decision's RAM cell, exactly as the T arm
            plays it: perception.table_best) in the eye's top 3 by rank score; over the decisions where that best is
            not walking in; the forward-best decisions are reported apart (with how often the eye walks in there)
  q8        softness: on (decision, move)s whose target is spread (no answer >= 0.9), the share of confident answers
            (the eye's top answer >= 0.9); on confident targets, the eye's word accuracy
Writes <model>/eval_u.json. Exit 1 if a pre-registered gate fails: gate 2 >= GATE2 (0.9) on test_real and on
test_heldout_guile; gate 1 (per character and question) only with --floors {"floors": {question: floor}} - the
floors are fixed from the label-noise check before training; without the file gate 1 is report-only.
"""
import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import _path  # noqa: F401
from sf2.data import value_oracle
from sf2.data import u_data as U
from sf2.data.perception import FORWARD, Q8_ANSWERS, QUESTIONS, best_in_top3, table_best
from sf2.system1.eye import pick, q8_words, rank_scores

FILES = ("test_real", "test_heldout_guile", "val")
GATED = ("test_real", "test_heldout_guile")
GATE2 = 0.9
CONFIDENT = U.CONFIDENT
TABLE = os.path.join("lessons", "value_oracle_v1.json")

Answers = Dict[str, Dict[str, float]]


def by_decision(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = collections.OrderedDict()
    for r in rows:
        out.setdefault(r["decision"], []).append(r)
    return out


def _acc(hits: List[bool]) -> Dict:
    return {"n": len(hits), "acc": sum(hits) / len(hits) if hits else None}


def metrics(decisions: Sequence[Tuple[List[Dict], Answers]], table: Dict) -> Dict:
    """Every metric over (a decision's rows, the eye's answers to all its questions)."""
    hits: Dict[str, Dict[str, List[bool]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    conf: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    g2: Dict[str, List[bool]] = collections.defaultdict(list)
    fwd = {"n": 0, "eye_walks_in": 0}
    spread, confident = [], []
    for rows, ans in decisions:
        me = rows[0]["char"]
        for r in rows:
            if r["task"] == "q8":
                got = U.argmax(ans[r["key"]], Q8_ANSWERS)
                hits[me]["q8_word"].append(got == r["answer"])
                hits["all"]["q8_word"].append(got == r["answer"])
                (confident if r["confident"] else spread).append(
                    (max(ans[r["key"]].values()) >= CONFIDENT, got == r["answer"]))
                continue
            got = U.argmax(ans[r["key"]], QUESTIONS[r["key"]])
            hits[me][r["key"]].append(got == r["answer"])
            hits["all"][r["key"]].append(got == r["answer"])
            if r["key"] == "phase":
                conf[r["answer"]][got] += 1
        moves = U.q8_moves(me)
        rank = rank_scores(ans, moves)
        cell = tuple(rows[0]["cell"])
        top = best_in_top3(table, cell, rank)
        if top is None:
            fwd["n"] += 1
            fwd["eye_walks_in"] += pick(rank, q8_words(ans, moves))[0] == FORWARD
        else:
            g2[me].append(top)
            g2["all"].append(top)
    return {"decisions": len(decisions),
            "accuracy": {who: {q: _acc(v) for q, v in sorted(qs.items())} for who, qs in sorted(hits.items())},
            "q2_confusion": {t: dict(sorted(c.items())) for t, c in sorted(conf.items())},
            "gate2": {"all": _share(g2["all"]), "by_char": {c: _share(v) for c, v in sorted(g2.items()) if c != "all"},
                      "forward_best": fwd},
            "q8": {"spread": {"n": len(spread), "confident_answers": _mean([c for c, _ in spread])},
                   "confident": {"n": len(confident), "confident_answers": _mean([c for c, _ in confident]),
                                 "word_acc": _mean([h for _, h in confident])},
                   "word_acc": _acc([h for _, h in spread + confident])}}


def _mean(xs: List[bool]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _share(xs: List[bool]) -> Dict:
    return {"n": len(xs), "in_top3": sum(xs), "share": sum(xs) / len(xs) if xs else None}


def gate_failures(splits: Dict[str, Dict], floors: Optional[Dict[str, float]]) -> List[str]:
    """The pre-registered gates that fail (empty: pass). Gate 2 on GATED; gate 1 only with ``floors``."""
    out = []
    for name in GATED:
        m = splits.get(name)
        g = (m or {}).get("gate2", {}).get("all", {})
        if not g.get("n"):
            out.append("gate 2 %s: no decisions to judge" % name)
        elif g["share"] < GATE2:
            out.append("gate 2 %s: %.3f < %s (%d of %d)" % (name, g["share"], GATE2, g["in_top3"], g["n"]))
        for who, qs in sorted(((m or {}).get("accuracy") or {}).items()):
            if who == "all" or not floors:
                continue
            for q, floor in sorted(floors.items()):
                a = qs.get(q)
                if a and a["acc"] is not None and a["acc"] < floor:
                    out.append("gate 1 %s %s %s: %.3f < %s (n %d)" % (name, who, q, a["acc"], floor, a["n"]))
    return out


def read_floors(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    with open(path) as f:
        floors = json.load(f)["floors"]
    bad = sorted(set(floors) - set(U.PERCEPTION))
    if bad:
        raise SystemExit("%s: floors for unknown questions %s" % (path, bad))
    return {k: float(v) for k, v in floors.items()}


# ---- the model --------------------------------------------------------------------------------------------------

def ask_all(eyes: Dict, base: str, decisions: Dict[str, List[Dict]]) -> Tuple[List[Tuple[List[Dict], Answers]], float]:
    """Every decision through its character's eye (frames from ``base``); (pairs, seconds per decision)."""
    from sf2.data.build import _load

    out, t0 = [], time.time()
    for rows in decisions.values():
        prev, cur = (_load(os.path.join(base, p)) for p in rows[0]["images"])
        out.append((rows, eyes[rows[0]["char"]].ask(prev, cur)))
    return out, (time.time() - t0) / max(len(out), 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=U.OUT)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--floors", help="gate 1 floors: {\"floors\": {question: floor}} (absent: gate 1 report-only)")
    ap.add_argument("--decisions", type=int, default=0,
                    help="per character and file, only the first N decisions by u_data.val_rank (0: all)")
    ap.add_argument("--chars", help="comma-separated (default: every character dir in --data)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    floors = read_floors(args.floors)
    table = value_oracle.load(args.table)
    import laya

    from sf2.system1.eye import Eye

    agent = laya.load_vlm(args.model, device=args.device)
    chars = args.chars.split(",") if args.chars else sorted(
        c for c in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, c)))
    eyes = {c: Eye(agent, c) for c in chars}
    splits, timing = {}, {}
    for name in FILES:
        pairs: List[Tuple[List[Dict], Answers]] = []
        for c in chars:
            path = os.path.join(args.data, c, name + ".jsonl")
            if not os.path.exists(path):
                continue
            decs = by_decision([json.loads(x) for x in open(path)])
            if args.decisions:
                keep = sorted(decs, key=U.val_rank)[:args.decisions]
                decs = collections.OrderedDict((k, decs[k]) for k in keep)
            got, per = ask_all(eyes, os.path.join(args.data, c), decs)
            pairs += got
            timing["%s/%s" % (name, c)] = round(per, 4)
        if pairs:
            splits[name] = metrics(pairs, table)
    failures = gate_failures(splits, floors)
    out = {"model": args.model, "data": args.data, "table": args.table, "floors": floors, "gate2_min": GATE2,
           "splits": splits, "seconds_per_decision": timing, "failures": failures,
           "gate1": "checked" if floors else "report-only (no floors file)"}
    with open(os.path.join(args.model, "eval_u.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name, m in splits.items():
        g = m["gate2"]["all"]
        print("%-20s %6d decisions  gate2 %s (%d of %d)  forward-best %s  q8 spread confident %s" % (
            name, m["decisions"], "-" if g["share"] is None else "%.3f" % g["share"], g["in_top3"], g["n"],
            m["gate2"]["forward_best"], m["q8"]["spread"]["confident_answers"]))
        for q, a in m["accuracy"].get("all", {}).items():
            print("   %-14s %s" % (q, a))
    for x in failures:
        print("GATE FAILED:", x)
    print("wrote", os.path.join(args.model, "eval_u.json"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
