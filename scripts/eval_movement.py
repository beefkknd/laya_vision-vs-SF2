"""The movement test (docs/prereg_movement.md "Judged against RAM") on test_data_mv (scripts/build_movement_data.py):
one predict per decision of "What is he doing right now?" (frames v3, note "me=<char>"), scored against the RAM label.

    python scripts/eval_movement.py --model CKPT [--data test_data_mv] [--decisions N] [--before runs/u_eye/best]
        [--out CKPT/eval_movement.json] [--device mps]

Per file (test_real, test_heldout_guile): per-answer recall / precision, the confusion, overall and balanced
accuracy; the majority baseline (the training split's most common answer, always; balanced accuracy 1/8); per
character and per opponent; and the "before": --before (round 1's eye) asked its own question 2 "What is he doing?"
on the same decisions, mapped onto the eight answers (sf2.data.movement_eval.BEFORE_MAP; --before none skips it).
Prints the prereg's "trainable" verdict (sf2.data.movement_eval.verdict). --decisions N: per character and file, the
first N decisions by u_data.val_rank (indicative only). Writes --out (default CKPT/eval_movement.json).
"""
import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, List

import _path  # noqa: F401

from sf2.data import movement as M
from sf2.data import movement_eval as E
from sf2.data import u_data as U

FILES = ("test_real", "test_heldout_guile")
BEFORE = os.path.join("runs", "u_eye", "best")
PHASE = "phase"


def load(path: str, device):
    import laya
    return laya.load_vlm(path, device=device)


def read_rows(data: str, chars: List[str], name: str, limit: int) -> List[Dict]:
    out = []
    for c in chars:
        path = os.path.join(data, c, name + ".jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(x) for x in open(path)]
        if limit:
            rows = sorted(rows, key=lambda r: U.val_rank(r["decision"]))[:limit]
        out += rows
    return out


def train_counts(data: str, chars: List[str]) -> Dict[str, int]:
    """Natural (copy 0) training answers, pooled over the characters."""
    tot: Dict[str, int] = collections.Counter()
    for c in chars:
        for line in open(os.path.join(data, c, "train.jsonl")):
            r = json.loads(line)
            if r["copy"] == 0:
                tot[r["answer"]] += 1
    return dict(tot)


def state(base: str, row: Dict) -> Dict:
    from PIL import Image

    from sf2.data.build import _load
    return {"images": [Image.fromarray(U.hud_frame(_load(os.path.join(base, p)))).convert("RGB")
                       for p in row["images"]], "context": U.eye_note(row["char"])}


def ask(agent, data: str, rows: List[Dict], key: str, question: Dict) -> List[Dict[str, float]]:
    out = []
    for r in rows:
        got = agent.predict(state(os.path.join(data, r["char"]), r), {key: question})["answers"][key]
        out.append({a: float(p) for a, p in got["probabilities"].items()})
    return out


def evaluate(rows: List[Dict], preds: List[str], before_words, maj: str) -> Dict:
    truths = [r["answer"] for r in rows]
    return {"n": len(rows), "model": E.metrics(truths, preds),
            "balanced_lower_bound": E.balanced_lower_bound(rows, preds),
            "majority": dict(E.metrics(truths, [maj] * len(rows)), answer=maj),
            "before": E.before_metrics(truths, before_words) if before_words is not None else None,
            "by_char": E.breakdown(rows, preds, "char"), "by_opp": E.breakdown(rows, preds, "opp")}


def show(name: str, res: Dict) -> None:
    m, mj, b = res["model"], res["majority"], res["before"]
    print("%-20s %5d decisions  acc %.3f  balanced %.3f (95%% lb %.3f)  | majority (%s) acc %.3f balanced %.3f%s" % (
        name, res["n"], m["accuracy"] or 0, m["balanced_accuracy"] or 0, res["balanced_lower_bound"] or 0,
        mj["answer"], mj["accuracy"] or 0, mj["balanced_accuracy"] or 0,
        "" if b is None else "  | before acc %.3f balanced %.3f" % (b["accuracy"] or 0, b["balanced_accuracy"] or 0)))
    for a in M.ANSWERS:
        f = lambda v: "   -  " if v is None else "%.3f" % v
        print("   %-18s n %5d  recall %s  precision %s%s" % (
            a, m["support"][a], f(m["recall"][a]), f(m["precision"][a]),
            "" if b is None else "  before recall %s" % f(b["recall"][a])))
    for key in ("by_char", "by_opp"):
        print("   %s: %s" % (key, ", ".join("%s %.2f/%.2f (n %d)" % (g, v["accuracy"] or 0, v["balanced_accuracy"] or 0,
                                                                       v["n"]) for g, v in res[key].items())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="test_data_mv")
    ap.add_argument("--decisions", type=int, default=0, help="per character and file, the first N by val_rank")
    ap.add_argument("--before", default=BEFORE, help="round 1's eye, asked its question 2; 'none' to skip")
    ap.add_argument("--chars", help="comma-separated (default: every character dir in --data)")
    ap.add_argument("--out", default=None, help="default: <model>/eval_movement.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    chars = args.chars.split(",") if args.chars else sorted(
        c for c in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, c)))
    maj = E.majority(train_counts(args.data, chars))
    agent = load(args.model, args.device)
    before = None if args.before == "none" else load(args.before, args.device)
    files, timing = {}, {}
    for name in FILES:
        rows = read_rows(args.data, chars, name, args.decisions)
        if not rows:
            continue
        t0 = time.time()
        preds = [E.predicted(p) for p in ask(agent, args.data, rows, M.KEY, M.movement_question())]
        words = None
        if before is not None:
            q = U.perception_question(PHASE)
            words = [E.predicted(p, tuple(q["criteria"])) for p in ask(before, args.data, rows, PHASE, q)]
        timing[name] = round((time.time() - t0) / len(rows), 4)
        files[name] = evaluate(rows, preds, words, maj)
    verdict = E.verdict(files)
    out = {"model": args.model, "before": None if before is None else args.before, "data": args.data,
           "decisions_limit": args.decisions or None, "majority_answer": maj, "files": files, "verdict": verdict,
           "seconds_per_decision": timing}
    path = args.out or os.path.join(args.model, "eval_movement.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for name, res in files.items():
        show(name, res)
    for name, v in verdict["files"].items():
        print("%-20s trainable %s  %s" % (name, v["trainable"], v["checks"]))
    print("TRAINABLE (prereg): %s" % ("YES" if verdict["trainable"] else "NO"))
    if args.decisions:
        print("SUBSET: %d decisions per character and file; indicative only" % args.decisions)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
