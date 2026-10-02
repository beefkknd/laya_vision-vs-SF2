"""Judge one movement fine-tune on its held-out test matches (docs/prereg_movement_finetunes.md; sf2.data.mv_eval):
every --data dir's test.jsonl (never trained or validated on), scored against the RAM labels.

    python scripts/eval_mv.py --model runs/mv_air/best --data test_data_mv_air [--out runs/mv_air/eval.json]

The model scores each row exactly as validation did in scripts/train.py (laya.vlm_train.collect_logits, the options
in label order); the answer is the arg-max. Reports balanced accuracy, per-answer recall, the confusion, per character
and per facing; vs always the most common TRAINING answer and vs chance; "learned" = the 2.5% lower bound of balanced
accuracy (1,000 resamples by whole match) above both. Writes --out (default <model>/../eval.json), with every test
row's id, truth and answer.

Round 3: the answers come from the dataset's build.json "answers" when it has them (act, fireball); --collapse act
scores a 10-movement model (mv2_move) on act's three answers (sf2.data.mv3_act mapping, truth and answer both).
"""
import argparse
import json
import os
import sys
import time

import _path  # noqa: F401
from sf2.data import mv_eval as E
from sf2.data import pairs_train as T


def load_rows(data: str, split: str):
    """(laya examples, the raw records in the same order) over every answer / dataset dir of ``data``."""
    import laya.vlm_train as vt

    exs, recs = [], []
    for d in sorted(T.read_dataset(data)):
        recs_d = [json.loads(x) for x in open(os.path.join(data, d, split + ".jsonl")) if x.strip()]
        ex_d = vt.load_jsonl_examples(data, d, split)
        if len(ex_d) != len(recs_d) or any(e["id"] != r["id"] for e, r in zip(ex_d, recs_d)):
            raise SystemExit("%s/%s/%s: laya dropped or reordered rows" % (data, d, split))
        exs += ex_d
        recs += recs_d
    return exs, recs


def show(name: str, res) -> str:
    m, mj = res["model"], res["majority"]
    lines = ["%s: n %d (%d matches)  balanced %.3f (2.5%% lb %.3f)  acc %.3f | majority '%s' balanced %.3f acc %.3f | "
             "chance %.3f | LEARNED %s" % (name, res["n"], res["matches"], m["balanced_accuracy"],
                                           res["balanced_lower_bound"], m["accuracy"], mj["answer"],
                                           mj["balanced_accuracy"], mj["accuracy"], res["chance"]["balanced_accuracy"],
                                           "YES" if res["learned"] else "NO")]
    lines.append("  recall: " + ", ".join("%s %.2f" % (a, v) for a, v in m["recall"].items() if v is not None))
    lines.append("  by char: " + ", ".join("%s %.2f" % (c, v["balanced_accuracy"]) for c, v in res["by_char"].items()))
    lines.append("  by facing: " + ", ".join("%s %.2f" % (c, v["balanced_accuracy"])
                                             for c, v in res["by_facing"].items()))
    if "by_side" in res:
        lines.append("  by side: " + ", ".join("%s %.2f" % (c, v["balanced_accuracy"])
                                               for c, v in res["by_side"].items()))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--collapse", choices=("act",), help="score the answers mapped to round 3's act answers")
    args = ap.parse_args(argv)
    meta = json.load(open(os.path.join(args.data, "build.json")))
    q = meta["question"]
    answers = list(meta.get("answers") or T.CRITERIA[q])
    exs, recs = load_rows(args.data, "test")
    train_answers = [r["answer"] for r in T.all_rows(args.data, "train")]
    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(args.model, device=args.device)
    t0 = time.time()
    out = vt.collect_logits(agent.model, agent.processor, exs, batch_size=args.batch_size)
    preds = [answers[int(o["logits"].argmax())] for o in out]
    if any(o["label"] != answers.index(r["answer"]) for o, r in zip(out, recs)):
        raise SystemExit("label order mismatch between laya and the records")
    raw = [{"id": r["id"], "truth": r["answer"], "answer": p} for r, p in zip(recs, preds)]
    if args.collapse:
        from sf2.data import mv3_act as A
        mapping = {m: A.act_of(m) for m in answers}
        recs, preds = E.collapsed(recs, preds, mapping)
        train_answers = [mapping[a] for a in train_answers]
        answers = list(A.ACT_ANSWERS)
    res = E.evaluate(recs, preds, answers, train_answers)
    res.update({"checkpoint": args.model, "data": args.data, "question": q, "seconds": round(time.time() - t0, 1),
                "collapse": args.collapse, "rows": raw})
    path = args.out or os.path.join(os.path.dirname(os.path.normpath(args.model)), "eval.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(show(os.path.basename(os.path.normpath(args.data)), res))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
