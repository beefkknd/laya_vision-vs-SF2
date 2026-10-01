"""Build the movement dataset (sf2.data.movement_data; docs/prereg_movement.md) from test_data_u and the U collection:

    python scripts/build_movement_data.py [--root rollouts/u_perception] [--u-data test_data_u] [--out test_data_mv]
        [--val-decisions 200] [--steps 10000] [--batch 8] [--workers N]

Writes <out>/<char>/{train,val,val_rest,test_real[,test_heldout_guile]}.jsonl, frames -> test_data_u/<char>/frames
(symlink), stats.json, and <out>/build.json + summary.json. Prints per character the decisions, the masked
("unknown") ones, the class counts per split (natural) and of the balanced train, and the --epochs that make
scripts/train.py run exactly --steps at --batch. Exit 1 on any problem.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import movement as M
from sf2.data import movement_data as D
from sf2.data import u_data as U


def summary(res, steps: int, batch: int) -> dict:
    train = sum(c["train_rows"] for c in res["counts"].values())
    tot = {}
    for c in res["counts"].values():
        for name, counts in c["natural"].items():
            t = tot.setdefault(name, {})
            for a, n in counts.items():
                t[a] = t.get(a, 0) + n
    return {"train_rows": train, "val_rows": sum(c["rows"].get("val", 0) for c in res["counts"].values()),
            "natural": tot, "steps": steps, "batch": batch,
            "epochs_for_steps": D.epochs_for_steps(steps, train, batch) if train else None}


def show(res, s) -> None:
    for char, c in sorted(res["counts"].items()):
        print("%-8s %d decisions, %d unknown (masked), %d collection entries not in test_data_u; rows %s" % (
            char, c["decisions"], c["unknown"], c["collection_not_in_u"], c["rows"]))
        for name, counts in sorted(c["natural"].items()):
            print("   %-20s %s" % (name, " ".join("%s=%d" % (a, counts.get(a, 0)) for a in M.ANSWERS)))
        print("   %-20s %s" % ("train (balanced)", " ".join("%s=%d" % (a, c["balanced_train"].get(a, 0))
                                                          for a in M.ANSWERS)))
    print(json.dumps(s, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=U.U_ROOT)
    ap.add_argument("--u-data", default=U.OUT)
    ap.add_argument("--out", default=D.OUT)
    ap.add_argument("--val-decisions", type=int, default=D.VAL_DECISIONS,
                    help="per character, the val games' decisions kept in val.jsonl (by u_data.val_rank)")
    ap.add_argument("--chars", help="only these characters (comma-separated)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    res = D.build(args.root, args.u_data, args.out, args.val_decisions,
                  args.chars.split(",") if args.chars else None, args.workers)
    s = summary(res, args.steps, args.batch) if res["counts"] else {}
    show(res, s)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(s, meta=res["meta"], counts=res["counts"], problems=res["problems"][:200]), f, indent=1)
    for p in res["problems"][:40]:
        print("PROBLEM:", p)
    if res["problems"]:
        print("%d problems" % len(res["problems"]))
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
