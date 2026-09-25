"""Day 3/6/7: compare runs on the same savestate. Round wins and damage decide, not val loss.

    python scripts/gate.py rollouts/teacher rollouts/r0 rollouts/r1

Reads each rollout's gate.json (written by play_student.py / play_teacher.py).
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cols = ["rounds", "damage_score", "net_damage_per_round", "round_win_rate", "dmg_dealt_per_round",
            "dmg_taken_per_round", "hadouken_whiff_rate", "shoryuken_whiff_rate", "teacher_agreement"]
    print("%-22s" % "run" + "".join("%22s" % c for c in cols))
    for d in sys.argv[1:]:
        with open(os.path.join(d, "gate.json")) as f:
            g = json.load(f)
        # Allow historical rollouts to participate in the new scorecard.
        if "damage_score" not in g:
            g["damage_score"] = 100.0 * g.get("dmg_dealt_per_round", 0.0) / 176.0
        if "net_damage_per_round" not in g:
            g["net_damage_per_round"] = g.get("dmg_dealt_per_round", 0.0) - g.get("dmg_taken_per_round", 0.0)
        cells = []
        for c in cols:
            v = g.get(c)
            cells.append("%22s" % ("-" if v is None else ("%.3f" % v if isinstance(v, float) else v)))
        print("%-22s" % os.path.basename(os.path.normpath(d)) + "".join(cells))
        top = list(g.get("action_mix", {}).items())[:5]
        print(" " * 22 + "mix: " + ", ".join("%s %.2f" % kv for kv in top))


if __name__ == "__main__":
    main()
