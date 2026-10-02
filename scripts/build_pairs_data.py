"""Build the movement-pairs datasets (sf2.data.pairs_data; docs/prereg_movement_pairs.md) from a collection.

    python scripts/build_pairs_data.py --root rollouts/pairs --out test_data_pairs

Writes <out>/<question>/{train,test}.jsonl for movement, facing, air, distance, <out>/frames/<pair> symlinks and
<out>/build.json (counts per cell, shortfalls). Refuses an existing --out with a build.json (never overwrites).
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import pairs_data as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("rollouts", "pairs"))
    ap.add_argument("--out", default="test_data_pairs")
    ap.add_argument("--cap-train", type=int, default=D.CAPS["train"])
    ap.add_argument("--cap-test", type=int, default=D.CAPS["test"])
    ap.add_argument("--cap-movement", action="append", default=[], metavar="MOVEMENT:TRAIN:TEST",
                    help="a grid movement's own cap (repeatable; round 4: attack:100:30, special:100:30)")
    ap.add_argument("--keep-from", help="an earlier build: every one of its movement rows is selected again")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controller", action="append", choices=D.CONTROLLERS,
                    help="keep only these players' rows (repeatable; default both)")
    args = ap.parse_args(argv)
    if os.path.exists(os.path.join(args.out, "build.json")):
        raise SystemExit("%s already holds a build: pick a new --out" % args.out)
    meta = D.build(args.root, args.out, {"train": args.cap_train, "test": args.cap_test}, args.seed,
                   tuple(args.controller) if args.controller else D.CONTROLLERS,
                   movement_caps=D.parse_movement_caps(args.cap_movement), keep_from=args.keep_from)
    print("collected %d pairs, selected %d (kept %d); %d cells short of the cap" % (
        meta["collected"], meta["selected"], meta["kept"], len(meta["short"])))
    print(json.dumps(meta["questions"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
