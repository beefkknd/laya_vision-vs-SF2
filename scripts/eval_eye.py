"""Judge an eye fine-tune on one question's held-out test matches (docs/eye_questions_v1.md, "Training
pre-registration"; sf2.data.eye_eval): every answer dir's test.jsonl (never trained or validated on), against RAM.

    python scripts/eval_eye.py --model runs/eye_q3/best --data test_data_eye_q3_act --out runs/eye_q3/eval_q3.json

The model scores each row as validation did in scripts/train.py (laya.vlm_train.collect_logits, the options in label
order); the answer is the arg-max. Writes --out with the summary and every test row's id, truth and answer.

    python scripts/eval_eye.py --report runs     # the results table and the eye_all keep rule from runs/eye_*/eval_*.json
"""
import argparse
import json
import os
import sys
import time

import _path  # noqa: F401
from sf2.data import eye_eval as V

QUESTIONS = {"q1": "test_data_eye_q1_fireball", "q3": "test_data_eye_q3_act", "q4": "test_data_eye_q4_air",
             "q5": "test_data_eye_q5_dist"}


def load_rows(data: str, dirs, split: str):
    import laya.vlm_train as vt

    exs, recs = [], []
    for d in dirs:
        recs_d = [json.loads(x) for x in open(os.path.join(data, d, split + ".jsonl")) if x.strip()]
        ex_d = vt.load_jsonl_examples(data, d, split)
        if len(ex_d) != len(recs_d) or any(e["id"] != r["id"] for e, r in zip(ex_d, recs_d)):
            raise SystemExit("%s/%s/%s: laya dropped or reordered rows" % (data, d, split))
        exs += ex_d
        recs += recs_d
    return exs, recs


def score(args) -> int:
    meta = json.load(open(os.path.join(args.data, "build.json")))
    q, answers = meta["question"], list(meta["answers"])
    exs, recs = load_rows(args.data, meta["dirs"], "test")
    train_answers = [json.loads(x)["answer"] for d in meta["dirs"]
                     for x in open(os.path.join(args.data, d, "train.jsonl")) if x.strip()]
    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(args.model, device=args.device)
    t0 = time.time()
    out = vt.collect_logits(agent.model, agent.processor, exs, batch_size=args.batch_size)
    if any(o["label"] != answers.index(r["answer"]) for o, r in zip(out, recs)):
        raise SystemExit("label order mismatch between laya and the records")
    preds = [answers[int(o["logits"].argmax())] for o in out]
    res = V.summary(recs, preds, answers, train_answers, q, meta.get("candidates"))
    res.update({"checkpoint": args.model, "data": args.data, "seconds": round(time.time() - t0, 1),
                "rows": [{"id": r["id"], "truth": r["answer"], "answer": p} for r, p in zip(recs, preds)]})
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    m = res["model"]
    print("%s %s: n %d (%d matches) balanced %.3f (lb %.3f) majority %.3f chance %.3f learned %s weighted %s | %s"
          % (args.model, q, res["n"], res["matches"], m["balanced_accuracy"], res["balanced_lower_bound"],
             res["majority"]["balanced_accuracy"], res["chance"], res["learned"], res["weighted_accuracy"],
             ", ".join("%s %.3f" % (a, v) for a, v in m["recall"].items())))
    print("wrote", args.out)
    return 0


def report(runs: str) -> int:
    def load(run, q):
        p = os.path.join(runs, run, "eval_%s.json" % q)
        return json.load(open(p)) if os.path.exists(p) else None

    print("| run | question | n (matches) | balanced | 2.5% lb | majority | chance | learned | weighted to real play | recall |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    sep, comb = {}, {}
    for q in QUESTIONS:
        for run in ("eye_" + q, "eye_all"):
            r = load(run, q)
            if not r:
                continue
            b = r["model"]["balanced_accuracy"]
            (comb if run == "eye_all" else sep)[q] = b
            w = r["weighted_accuracy"]
            print("| %s | %s | %d (%d) | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |" % (
                run, q, r["n"], r["matches"], b, r["balanced_lower_bound"], r["majority"]["balanced_accuracy"],
                r["chance"], "yes" if r["learned"] else "NO", "-" if w is None else "%.3f" % w,
                ", ".join("%s %.3f" % (a, v) for a, v in r["model"]["recall"].items())))
    for q in QUESTIONS:
        r = load("eye_" + q, q)
        if r and "by_side" in r:
            print("%s by side: %s | by char: %s" % (q, ", ".join("%s %.3f" % (k, v["balanced_accuracy"]) for k, v in r["by_side"].items()),
                                                  ", ".join("%s %.3f" % (k, v["balanced_accuracy"]) for k, v in r["by_char"].items())))
        if r and q == "q1":
            print("q1 'no' accuracy by tag: %s | 'yes' accuracy by frames: %s" % (
                ", ".join("%s %.3f (%d)" % (k, v["accuracy"], v["n"]) for k, v in r["by_no_tag"].items()),
                ", ".join("%s %.3f (%d)" % (k, v["accuracy"], v["n"]) for k, v in r["by_yes_frames"].items())))
    if sep and comb and set(sep) == set(comb):
        k = V.keep_combined(sep, comb)
        print("keep rule (margin %.2f): %s" % (k["margin"], "KEEP eye_all" if k["keep"] else
                                                "DROP eye_all - separate adapters for %s" % k["separate_adapters_for"]))
        for q, v in k["per_question"].items():
            print("  %s separate %.3f combined %.3f diff %+.3f %s" % (q, v["separate"], v["combined"], v["diff"],
                                                                     "within" if v["within"] else "LOSES MORE"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--report", metavar="RUNS_DIR")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    if args.report:
        return report(args.report)
    if not (args.model and args.data and args.out):
        ap.error("--model, --data and --out are required (or --report)")
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
