"""Score a laya-vision checkpoint on the stage-1 held-out sets (real frames, both sides): per character, side,
outcome class and (action, range) combination. The pass bar: >= --bar correct on every one of a character's
20 actions x 3 ranges x 2 sides = 120 combinations. Exit 1 if any combination is under it.

    python scripts/eval_outcome.py --model runs/all8/best
    python scripts/eval_outcome.py --model runs/all8/best --chars ryu,ken --bar 0.95

Writes <model>/eval_outcome.json (every number printed, plus the confusion matrix and the failing combinations).
"""
import argparse
import collections
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.vs_sweep import OUTCOMES

ROOT = "test_data"
FILES = ("test_real_left", "test_real_right")


def score(agent, vt, char: str, split: str) -> List[Dict]:
    recs = [json.loads(line) for line in open(os.path.join(ROOT, char, split + ".jsonl"))]
    exs = vt.load_jsonl_examples(ROOT, char, split)
    if len(exs) != len(recs):
        raise RuntimeError("%s/%s: %d records but %d loaded" % (char, split, len(recs), len(exs)))
    out = vt.collect_logits(agent.model, agent.processor, exs, batch_size=32)
    return [dict(r, pred=OUTCOMES[int(o["logits"].argmax())]) for r, o in zip(recs, out)]


def acc(rows: List[Dict]):
    """Share correct; None for no rows (an absent class is not 0% recall)."""
    return sum(r["pred"] == r["outcome"] for r in rows) / len(rows) if rows else None


def fmt(x) -> str:
    return "  n/a  " if x is None else "%-7.3f" % x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir, e.g. runs/all8/best")
    ap.add_argument("--chars", help="comma-separated (default: every character in test_data/)")
    ap.add_argument("--bar", type=float, default=0.95)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    chars = args.chars.split(",") if args.chars else sorted(d for d in os.listdir(ROOT) if not d.startswith("_"))

    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(args.model, device=args.device)
    report, failing = {}, []
    print("%-8s %6s %6s %6s | %s | combos >= %.0f%%" % ("char", "left", "right", "attack", "  ".join(
        "%-7s" % o for o in OUTCOMES), 100 * args.bar))
    for char in chars:
        rows = [dict(r, file=f) for f in FILES for r in score(agent, vt, char, f)]
        combos = collections.defaultdict(list)
        for r in rows:
            combos[(r["action"], r["range"], r["side"])].append(r)
        bad = sorted(("%s@%s/%s" % k, round(acc(v), 3)) for k, v in combos.items() if acc(v) < args.bar)
        failing += [(char,) + b for b in bad]
        confusion = collections.Counter((r["outcome"], r["pred"]) for r in rows)
        per_class = {o: acc([r for r in rows if r["outcome"] == o]) for o in OUTCOMES}
        report[char] = {
            "n": len(rows), "left": acc([r for r in rows if r["side"] == "left"]),
            "right": acc([r for r in rows if r["side"] == "right"]),
            "attack_only": acc([r for r in rows if r["kind"] == "attack"]),
            "recall_by_outcome": per_class, "combos": len(combos), "combos_passing": len(combos) - len(bad),
            "failing_combos": bad, "confusion": {"%s->%s" % k: v for k, v in sorted(confusion.items())},
            "rows": [{k: r[k] for k in ("id", "side", "action", "range", "posture", "gap", "gap_index", "outcome",
                                         "pred")} for r in rows],
        }
        x = report[char]
        print("%-8s %6.3f %6.3f %6.3f | %s | %d/%d" % (char, x["left"], x["right"], x["attack_only"], "  ".join(
            fmt(per_class[o]) for o in OUTCOMES), x["combos_passing"], x["combos"]), flush=True)
    path = os.path.join(args.model, "eval_outcome.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "bar": args.bar, "chars": report}, f, indent=1)
    print("%d failing combinations of %d -> %s" % (len(failing), sum(r["combos"] for r in report.values()), path))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
