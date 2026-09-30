"""Build the value fine-tune's dataset test_data_v2/ (sf2/data/value_data.py; docs/prereg_lv_value.md): the all8 rows
in note v2, a value row next to every live row, and the new live rows from rollouts/lv_value (Chun-Li vs Guile held
out in chunli/test_heldout_guile.jsonl). Prints PROBLEM lines and exits 1 on any gate failure.

    python scripts/build_value_data.py                      # -> test_data_v2/
    python scripts/build_value_data.py --out /tmp/v2 --overwrite
"""
import argparse
import sys

import _path  # noqa: F401
from sf2.data import value_data as V


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-data", default="test_data", help="the all8 dataset (read only)")
    ap.add_argument("--lv-root", default=V.LV_ROOT, help="the value collection logs (read only)")
    ap.add_argument("--repo", default=".", help="where the old live rows' sources (live:rollouts/...) are relative to")
    ap.add_argument("--out", default=V.OUT)
    ap.add_argument("--overwrite", action="store_true", help="remove a previous build of --out first")
    ap.add_argument("--min-value-train", type=int, default=V.MIN_VALUE_TRAIN,
                    help="new real value training rows needed per move of System 1's choices")
    ap.add_argument("--no-cap", action="store_true",
                    help="no live cap: every character keeps all its training games (no test_extra); off by default")
    ap.add_argument("--forward-value-cap", choices=["median_attack"], default=None,
                    help="cut each character's new forward value training rows to its median attack's count")
    args = ap.parse_args(argv)
    if args.overwrite:
        V.overwrite(args.out)
    res = V.build(args.test_data, args.lv_root, args.out, args.repo, args.min_value_train, args.no_cap,
                  args.forward_value_cap)
    if args.no_cap:
        print("live cap: off (--no-cap): every training game kept")
    else:
        print("live cap: %s new training decisions per character" % res["cap"])
    for char, c in sorted(res["counts"].items()):
        if c["logs"]:
            print("%-8s new training decisions kept %d, past the cap -> test_extra %d" % (
                char, c["new_train_decisions"], c["extra_decisions"]))
        if "forward_value_cap" in c:
            f = c["forward_value_cap"]
            print("%-8s forward value rows %d -> %d (median attack %d); labels before %s after %s" % (
                char, f["before"], f["after"], f["median_attack"], f["labels_before"], f["labels_after"]))
    for char, c in sorted(res["counts"].items()):
        f = c["files"]
        print("%-8s %s | new logs %s, dx0 dropped %d, partial lines %d" % (
            char, "  ".join("%s %d (value %d, new %d)" % (k, v["rows"], v["value"], v["new"]) for k, v in f.items()),
            ",".join(c["logs"]) or "-", c["dropped_dx0"], c["partial_lines"]))
    for p in res["problems"]:
        print("PROBLEM", p)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
