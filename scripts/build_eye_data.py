"""Build the eye's aligned datasets of docs/eye_questions_v1.md ("Datasets v1"; sf2.data.eye_pool / eye_data) from
ONE frame pool: every image pair the 2P collection has on disk (committed games), one whole-match split table.

    python scripts/build_eye_data.py [--root rollouts/pairs2p] [--questions q1,q3,q4,q5] [--max-game N]
    python scripts/build_eye_data.py --questions q3v2,q4v2 --max-game 31      # questions v2 (relabel), games 0-31

Writes test_data_eye_<q>_<name>/<answer>/{train,val,test}.jsonl (+ frames links) and build.json per question.
Refuses an existing out dir. Checks every question's head length against laya's head (--tokenizer). Run
scripts/gate_eye_data.py next (labels, episode, splits, drawn, lag 1, disk, shortcut). Exit 1 on any problem.
"""
import argparse
import json
import sys

import _path  # noqa: F401
from sf2.data import action_data as AD
from sf2.data import eye_data as E
from sf2.data import eye_pool as P
from sf2.data import pairs_labels as L

PREFER = {"q1": ["test_data_mv3_fireball"], "q3": ["test_data_mv3_act"], "q3v2": ["test_data_mv3_act"],
          "q3v2b": ["test_data_mv3_act"]}


def heads(tokenizer_dir: str) -> None:
    if tokenizer_dir == "none":
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    qs = {"%s %s" % (q, s): E.question(q, s) for q in E.Q for s in (("left", "right") if E.Q[q]["side"] else (None,))}
    n = {k: AD.head_tokens(v, tok) for k, v in qs.items()}
    print("head tokens: max %d (%s)" % (max(n.values()), max(n, key=n.get)))
    bad = ["%s: %d > %d" % (k, v, AD.HEAD_MAX) for k, v in n.items() if v > AD.HEAD_MAX]
    if bad:
        raise SystemExit("questions too long for laya's head: %s" % bad)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rollouts/pairs2p")
    ap.add_argument("--questions", default=",".join(E.V1), help="of %s" % ",".join(E.Q))
    ap.add_argument("--max-game", type=int, default=None,
                    help="pool games 0 .. N only (a stable pool while a collector appends later games)")
    ap.add_argument("--out-pattern", default=None, help="out dir per question, %%s = q (default %s)" % E.OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tokenizer", default=AD.TOKENIZER, help="laya-vision processor dir ('none' skips the check)")
    args = ap.parse_args(argv)
    qs = args.questions.split(",")
    unknown = [q for q in qs if q not in E.Q]
    if unknown:
        raise SystemExit("unknown questions %s (of %s)" % (unknown, sorted(E.Q)))
    heads(args.tokenizer)
    pool = P.pool(args.root, L.poke_bands()["all"], args.workers, args.max_game)
    print("pool: %d image pairs (games %s)" % (len(pool), "all" if args.max_game is None else "0-%d" % args.max_game))
    bad = 0
    for q in qs:
        out = args.out_pattern % q if args.out_pattern else E.out_of(q)
        meta = E.build(pool, q, out, args.root, args.seed, E.prefer_keys(q, PREFER.get(q, [])))
        print("%s -> %s: %s; matches %s; strata %d" % (q, out, E.summary(meta), meta["matches"], meta["strata"]))
        print("   dropped %s; prefer %s" % (json.dumps(meta["dropped"]), json.dumps(meta["prefer"])))
        print("   extras %s" % json.dumps(meta["extras"])[:1500])
        bad += len(meta["problems"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
