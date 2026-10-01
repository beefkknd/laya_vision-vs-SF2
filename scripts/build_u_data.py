"""Build the U arm's dataset (sf2.data.u_data; docs/prereg_u_perception.md) from the U collection:

    python scripts/build_u_data.py [--root rollouts/u_perception] [--out test_data_u]
        [--thresholds lessons/perception_thresholds_v1.json] [--q8-targets lessons/perception_q8_targets_v1.json]
        [--workers N] [--overwrite] [--chars chunli,ryu] [--steps-per-min 135] [--hours 4 4.5 5]

Writes <out>/<char>/{train,val,test_real[,test_heldout_guile]}.jsonl, frames/ (HUD visible, frames v3), stats.json,
and <out>/build.json + summary.json. Prints per character the decisions per split, the rows per question and their
label distributions, the masked ("unknown") labels, and the training-step numbers: the steps scripts/train.py would
run for --epochs 2 at batch 8, and fixed step budgets for --hours at --steps-per-min (evaluations included). Exit 1
on any problem (a decision that does not join, a missing image, a row gate).
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import u_data as U


def summary(res, steps_per_min: float, hours) -> dict:
    train = sum(c["files"].get("train", {}).get("rows", 0) for c in res["counts"].values())
    val = sum(c["files"].get("val", {}).get("rows", 0) for c in res["counts"].values())
    decisions = sum(c["decisions"] for c in res["counts"].values())
    rows = sum(f["rows"] for c in res["counts"].values() for f in c["files"].values())
    return {"decisions": decisions, "rows": rows, "rows_per_decision": rows / decisions if decisions else None,
            "train_rows": train, "val_rows": val,
            "steps_2_epochs_batch_8": U.training_steps(train, 2.0, 8),
            "steps_per_min": steps_per_min,
            "hours_for_2_epochs": U.training_steps(train, 2.0, 8) / steps_per_min / 60 if train else None,
            "budgets": {str(h): {"steps": U.step_budget(h, steps_per_min),
                                 "epochs": U.step_budget(h, steps_per_min) * 8 / train if train else None}
                        for h in hours}}


def show(res, s) -> None:
    for char, c in sorted(res["counts"].items()):
        print("%-8s %d decisions %s, %.1f rows/decision, partial lines %d" % (
            char, c["decisions"], c["decisions_by_split"], c["rows_per_decision"] or 0, c["partial_lines"]))
        for name, f in sorted(c["files"].items()):
            print("   %-20s %7d rows" % (name, f["rows"]))
        for q, v in sorted(c["files"].get("train", {}).get("questions", {}).items()):
            print("      train %-12s %6d  %s" % (q, v["rows"], v["labels"]))
        print("   unknown (masked): %s" % c["unknown"])
        print("   eye -> text laya words vs RAM's: %s" % c["mapping"])
    print(json.dumps(s, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=U.U_ROOT)
    ap.add_argument("--out", default=U.OUT)
    ap.add_argument("--thresholds", default=os.path.join("lessons", "perception_thresholds_v1.json"))
    ap.add_argument("--q8-targets", default=os.path.join("lessons", "perception_q8_targets_v1.json"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4),
                    help="processes writing frames (default: all cores but 4)")
    ap.add_argument("--chars", help="only these characters (comma-separated)")
    ap.add_argument("--val-decisions", type=int, default=None,
                    help="per character, keep this many decisions of the val games in val.jsonl (whole decisions, "
                         "chosen by hash; the rest -> val_rest.jsonl); default: every one (the prereg's split)")
    ap.add_argument("--overwrite", action="store_true", help="remove a previous build of --out first")
    ap.add_argument("--steps-per-min", type=float, default=135.0,
                    help="training speed incl. evaluations, for the step numbers (logs/train_lv_value2.log)")
    ap.add_argument("--hours", type=float, nargs="+", default=[4.0, 4.5, 5.0])
    args = ap.parse_args(argv)
    if args.overwrite:
        U.overwrite(args.out)
    res = U.build(args.root, args.out, args.thresholds, args.q8_targets, args.workers,
                  args.chars.split(",") if args.chars else None, args.val_decisions)
    s = summary(res, args.steps_per_min, args.hours) if res["counts"] else {}
    show(res, s)
    if res["counts"]:
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(dict(s, meta=res["meta"], problems=res["problems"][:200]), f, indent=1)
    for p in res["problems"][:40]:
        print("PROBLEM:", p)
    if res["problems"]:
        print("%d problems" % len(res["problems"]))
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
