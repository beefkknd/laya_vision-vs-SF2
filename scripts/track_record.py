"""Build the lesson track record from finished lesson-loop runs (sf2.system2.track_record) into a frozen file that
qwen_lessons.py --track reads. The same runs give the same file, byte for byte.

    python scripts/track_record.py [--out lessons/track_record.json] [ROOT ...]
                                   default roots: rollouts/qwen_lessons rollouts/locked/lesson_loop_v1
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.system2 import track_record as T

ROOTS = [os.path.join("rollouts", "qwen_lessons"), os.path.join("rollouts", "locked", "lesson_loop_v1")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("lessons", "track_record.json"))
    ap.add_argument("--ranking", choices=T.RANKINGS, default="all8",
                    help="System 1's ranking whose runs count (table: runs made with qwen_lessons.py --oracle)")
    ap.add_argument("roots", nargs="*")
    args = ap.parse_args()
    t = T.build(args.roots or ROOTS, args.ranking)
    with open(args.out + ".tmp", "w") as f:
        json.dump(t, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print("%d runs (%d skipped: %s; %d of another ranking left out) -> %s" % (
        len(t["sources"]), len(t["skipped"]), t["skipped"], len(t.get("other_ranking", [])), args.out))
    for opp, track in t["opponents"].items():
        clear = T.prompt_lines({k: v for k, v in track.items() if v["verdict"] in ("hurts", "helps")})
        print("%s: %d lines, %d clear" % (opp, len(track), len(clear)))
        for x in clear:
            print("  " + x)
    return 1 if t["skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
