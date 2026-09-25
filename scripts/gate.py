"""Day 3/6/7: compare runs on the same savestate. Round wins and damage decide, not val loss.

    python scripts/gate.py rollouts/teacher rollouts/r0 rollouts/r1

Reads each rollout's gate.json (written by play_student.py / play_teacher.py).
"""
import json
import os
import sys


def main():
    cols = ["rounds", "round_win_rate", "dmg_dealt_per_round", "dmg_taken_per_round", "hadouken_whiff_rate",
            "shoryuken_whiff_rate", "teacher_agreement"]
    print("%-22s" % "run" + "".join("%22s" % c for c in cols))
    for d in sys.argv[1:]:
        with open(os.path.join(d, "gate.json")) as f:
            g = json.load(f)
        cells = []
        for c in cols:
            v = g.get(c)
            cells.append("%22s" % ("-" if v is None else ("%.3f" % v if isinstance(v, float) else v)))
        print("%-22s" % os.path.basename(os.path.normpath(d)) + "".join(cells))
        top = list(g.get("action_mix", {}).items())[:5]
        print(" " * 22 + "mix: " + ", ".join("%s %.2f" % kv for kv in top))


if __name__ == "__main__":
    main()
