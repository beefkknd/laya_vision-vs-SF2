"""Build the round-3 fine-tune datasets of docs/prereg_movement_finetunes.md ("Round 3").

    python scripts/build_mv3_data.py act --src test_data_pairs2p_down [--out test_data_mv3_act]
    python scripts/build_mv3_data.py fireball --src test_data_pairs2p_down --shots rollouts/pairs2p \
        [--out test_data_mv3_fireball]

act (sf2.data.mv3_act): round 2's movement rows (same frames, sides, splits), answer moving / attack / special, one dir
per answer. fireball (sf2.data.mv3_fireball): the projectile-trigger samples of the new rounds (none / left / right by
the thrower's side), "none" rows from the existing pairs build, one dir per answer. Both refuse an existing --out and a
source build without a passing gate.json; both check the question's head length against laya's head (--tokenizer).
Exit 1 on any problem.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import action_data as AD
from sf2.data import mv3_act as A


def _gated(src: str) -> None:
    gate = os.path.join(src, "gate.json")
    if not os.path.exists(gate) or not json.load(open(gate)).get("pass"):
        raise SystemExit("%s: no passing gate.json (run scripts/gate_pairs_data.py first)" % src)


def _heads(questions, tokenizer_dir: str) -> None:
    if tokenizer_dir == "none":
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    heads = {k: AD.head_tokens(q, tok) for k, q in questions.items()}
    print("head tokens: max %d (%s)" % (max(heads.values()), max(heads, key=heads.get)))
    bad = ["%s: %d > %d" % (k, n, AD.HEAD_MAX) for k, n in heads.items() if n > AD.HEAD_MAX]
    if bad:
        raise SystemExit("questions too long for laya's head: %s" % bad)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("act", "fireball"))
    ap.add_argument("--src", required=True, help="the gated pairs build (round 2's source)")
    ap.add_argument("--shots", help="fireball: the collection root with <pair>/shots.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=AD.TOKENIZER, help="laya-vision processor dir ('none' skips the check)")
    args = ap.parse_args(argv)
    _gated(args.src)
    if args.dataset == "act":
        _heads({s: A.question_act(s) for s in ("left", "right")}, args.tokenizer)
        meta = A.build_act(args.src, args.out or A.OUT)
    else:
        from sf2.data import mv3_fireball as F
        if not args.shots:
            raise SystemExit("fireball needs --shots <collection root>")
        _heads({"fireball": F.question_fireball()}, args.tokenizer)
        meta = F.build_fireball(args.src, args.shots, args.out or F.OUT, seed=args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("problems", "dropped_equal_x", "ids")}, indent=1)[:4000])
    print("dropped (equal x at t): %s" % meta.get("dropped_equal_x", {}).get("total"))
    print("problems: %d" % len(meta["problems"]))
    return 1 if meta["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
