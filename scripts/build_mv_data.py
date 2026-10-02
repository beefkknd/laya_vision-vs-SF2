"""Build the four fine-tune datasets of docs/prereg_movement_finetunes.md (sf2.data.pairs_train) from one GATED
movement-pairs build (scripts/build_pairs_data.py, then scripts/gate_pairs_data.py must pass on it).

    python scripts/build_mv_data.py --src test_data_pairs2p_down [--datasets move,face,air,dist] [--ask side]

Writes test_data_mv_<dataset>/<dir>/{train,val,test}.jsonl (+ frames link) and test_data_mv_<dataset>/build.json.
--ask side (round 2): the question names the fighter by screen side; out test_data_mv2_<dataset>, equal-x rows
dropped and counted. Refuses an existing out dir. Checks every question's head length against laya's 256-token head (--tokenizer).
Exit 1 on any problem.
"""
import argparse
import os
import sys

import _path  # noqa: F401
from sf2.data import action_data as AD
from sf2.data import pairs_data as D
from sf2.data import pairs_train as T


def head_problems(tokenizer, ask="name"):
    if ask == "side":
        heads = {"%s %s" % (q, s): AD.head_tokens(T.question_side(q, s), tokenizer)
                 for q in D.QUESTION_ANSWERS for s in T.SIDES}
    else:
        heads = {"%s %s" % (q, c): AD.head_tokens(T.question(q, c), tokenizer)
                 for q in D.QUESTION_ANSWERS for c in T.NAMES}
    return heads, ["%s: %d head tokens > %d" % (k, n, AD.HEAD_MAX) for k, n in heads.items() if n > AD.HEAD_MAX]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a gated pairs build (build.json + gate.json with pass)")
    ap.add_argument("--datasets", default=",".join(T.DATASETS))
    ap.add_argument("--ask", choices=T.ASKS, default="name", help="name the fighter by character or by screen side")
    ap.add_argument("--out-pattern", default=None, help="out dir per dataset (default by --ask: %s)" % T.OUT_BY_ASK)
    ap.add_argument("--tokenizer", default=AD.TOKENIZER, help="laya-vision processor dir ('none' skips the check)")
    args = ap.parse_args(argv)
    import json
    gate = os.path.join(args.src, "gate.json")
    if not os.path.exists(gate) or not json.load(open(gate)).get("pass"):
        raise SystemExit("%s: no passing gate.json (run scripts/gate_pairs_data.py first)" % args.src)
    if args.tokenizer != "none":
        from transformers import AutoTokenizer
        heads, bad = head_problems(AutoTokenizer.from_pretrained(args.tokenizer), args.ask)
        print("head tokens: max %d (%s)" % (max(heads.values()), max(heads, key=heads.get)))
        if bad:
            raise SystemExit("questions too long for laya's head: %s" % bad)
    for ds in T.datasets(args.datasets.split(",")):
        out = (args.out_pattern or T.OUT_BY_ASK[args.ask]) % ds
        meta = T.build_one(args.src, ds, out, args.ask)
        print("%s -> %s: %s; matches %s" % (ds, out, T.summary(meta), meta["matches"]))
        if args.ask == "side":
            print("   dropped (equal x at t): %s; sides %s" % (
                json.dumps({k: v for k, v in meta["dropped_equal_x"].items() if k != "ids"}),
                json.dumps(meta["sides"])))
        for d, fs in meta["counts"].items():
            print("   %-6s %s" % (d, json.dumps(fs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
