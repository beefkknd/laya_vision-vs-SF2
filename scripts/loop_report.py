"""Per-GAME diagnostic report for a loop run (README G7), OFFLINE and OUTSIDE play.

    python scripts/loop_report.py rollouts/loop_screen/<stamp>_<opp>/
    python scripts/loop_report.py <run_dir> --table lessons/value_oracle_v1.json --out /tmp/loop_report.json

Reads an existing loop-run directory (run.json / trace.jsonl / verdict.json / score.json / per-round
replay.json); runs nothing. Prints a readable summary and writes loop_report.json (in the run dir by default).
The G5 verdicts need the offline table scorer (sf2.eval.rule_score + lessons/value_oracle_v1.json); when that is
not on the branch, G5 reports "unavailable" rather than guessing. Exit 0 always: it reports, it does not gate.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.eval.loop_report import build_report, make_g5_scorer, render_text


def main() -> int:
    ap = argparse.ArgumentParser(description="per-game diagnostic report for a loop run (G7)")
    ap.add_argument("run_dir", help="a loop-run directory, e.g. rollouts/loop_screen/<stamp>_<opp>/")
    ap.add_argument("--table", default=None, help="value lookup table for G5 (default lessons/value_oracle_v1.json)")
    ap.add_argument("--out", default=None, help="where to write loop_report.json (default <run_dir>/loop_report.json)")
    ap.add_argument("--no-g5", action="store_true", help="skip the G5 table scorer entirely")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit("not a run directory: %s" % args.run_dir)

    scorer = None if args.no_g5 else make_g5_scorer(args.table)
    if scorer is None and not args.no_g5:
        sys.stderr.write("note: G5 table scorer unavailable (sf2.eval.rule_score / value table not on this "
                         "branch); G5 verdicts will read 'unavailable'\n")

    report = build_report(args.run_dir, scorer=scorer)
    print(render_text(report))

    out = args.out or os.path.join(args.run_dir, "loop_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    sys.stderr.write("\nwrote %s\n" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
