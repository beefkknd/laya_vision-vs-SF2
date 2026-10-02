"""Seeded faults for the eye datasets v1 (docs/eye_questions_v1.md, "Datasets v1"): each mutates one line of the frame
pool, the aligned builder, the shortcut check or the gates, runs the eye tests, and restores the file. Every fault must
turn the tests red; exit 1 if one stays green.

    python tests/faults/eye_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_eye_pool.py", "tests/test_eye_data.py"]
PL, DA, SC, GT = "sf2/data/eye_pool.py", "sf2/data/eye_data.py", "sf2/data/eye_shortcut.py", "sf2/data/eye_gate.py"
FAULTS = [
    # the pool
    ("pool: the blink bit ignored", PL, 'return not row["shot%d_hide" % s] & 1', "return True"),
    ("pool: yoga flame is a fireball", PL, 'return [name for name, u in (("n-4", t - GAP), ("n", t))\n            if any(kinds[s][u] == "fireball"',
     'return [name for name, u in (("n-4", t - GAP), ("n", t))\n            if any(kinds[s][u] in ("fireball", "yoga_flame")'),
    ("pool: only frame n looked at (the v1 bug)", PL, 'for name, u in (("n-4", t - GAP), ("n", t))',
     'for name, u in (("n", t),)'),
    ("pool: an unknown projectile is a no", PL, '    if any(k == "other" for k, _ in active):\n        return None, "unknown_active"',
     '    if False:\n        return None, "unknown_active"'),
    ("pool: off the screen counts as yes", PL, "return pad <= u <= SCREEN_W - pad", "return True"),
    ("pool: a fireball hidden in both frames is a no", PL, '    if fire:\n        return None, "off_screen" if any(fire) else "hidden_both"',
     '    if False:\n        return None, "off_screen" if any(fire) else "hidden_both"'),
    ("pool: camera not clamped to the stage", PL, "return min(max(mid - SCREEN_W / 2, CAMERA_MIN), CAMERA_MAX)",
     "return mid - SCREEN_W / 2"),
    ("pool: the flight's word read at its last row", PL, "w = words[s][a]", "w = words[s][u - 1]"),
    ("pool: before-spawn window one row", PL, "range(t + 1, min(len(rows), t + near + 1))", "range(t + 1, t + 2)"),
    ("pool: after-impact window one row", PL, "range(max(0, t - near), t - GAP)", "range(t - GAP - 1, t - GAP)"),
    ("pool: yoga flame never tagged", PL, 'if any(k[u] == "yoga_flame" and drawn(rows[u], s) for u in (t - GAP, t)):',
     "if False:"),
    ("pool: pose without the attack state", PL, "if st in (L.ATTACK, L.SPECIAL) and S.is_projectile(char, word):",
     "if S.is_projectile(char, word):"),
    ("pool: image pairs without the lag", PL, "return {g: sorted(k - LAG for k in v if k - GAP in v)",
     "return {g: sorted(k for k in v if k - GAP in v)"),
    ("pool: the band is strict", PL, 'return "close" if abs(row["p1_x"] - row["p2_x"]) <= band else "far"',
     'return "close" if abs(row["p1_x"] - row["p2_x"]) < band else "far"'),
    ("pool: take-off read at t", PL, '"air_prev": L.air(rows, t - GAP, s)', '"air_prev": L.air(rows, t, s)'),
    ("pool: special counted as attack", PL, 'return mv10 if mv10 in ("attack", "special") else "moving"',
     'return "attack" if mv10 in ("attack", "special") else "moving"'),
    # the builder
    ("data: q3 outside an episode kept", DA, 'if q == "q3" and (me["act"] is None or not me["in_episode"]):',
     'if q == "q3" and me["act"] is None:'),
    ("data: the larger answer count per stratum", DA, "n = min(len(g[a]) for a in answers)",
     "n = max(len(g[a]) for a in answers)"),
    ("data: the cap ignored", DA, "            n = min(n, cap)", "            n = n"),
    ("data: tiers ignored", DA, 'out += sorted(mine, key=lambda c: c["tier"])[:n]', "out += mine[:n]"),
    ("data: q1 not matched on the poses", DA,
     'stratum = (split, f["pair_name"], game_range(f["game"])) + tuple(fine)',
     'stratum = (split, f["pair_name"], game_range(f["game"]))'),
    ("data: q1 not matched on the game range", DA,
     'stratum = (split, f["pair_name"], game_range(f["game"])) + tuple(fine)',
     'stratum = (split, f["pair_name"]) + tuple(fine)'),
    ("data: split not by match", DA, '"question_key": Q[q]["name"], "split": T.split3(f["pair_name"], f["game"]),',
     '"question_key": Q[q]["name"], "split": "train",'),
    ("data: label not the answer's index", DA, 'label=list(spec["answers"]).index(c["answer"])', "label=0"),
    ("data: the question always asks the left", DA, 'text = spec["text"] % side if spec["side"] else spec["text"]',
     'text = spec["text"] % "left" if spec["side"] else spec["text"]'),
    ("data: unequal strata not reported", DA, 'if len({c[a] for a in spec["answers"]}) != 1]', "if False]"),
    ("data: an existing out overwritten", DA,
     '    if os.path.exists(out):\n        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)\n    cap',
     "    cap"),
    ("data: q5 equal x kept", DA, 'if a["side"] is None or f["dist"] is None:', 'if f["dist"] is None:'),
    # the shortcut check
    ("shortcut: always passes", SC, '"pass": scores[best] - chance <= margin', '"pass": True'),
    ("shortcut: plain accuracy", SC, "rec.append(sum(pred[i] == a for i in idx) / len(idx))",
     "rec.append(sum(pred[i] == a for i in idx) / len(truth))"),
    ("shortcut: lookup ignores the feature", SC, "votes[_key(feat, names)][a] += 1", "votes[()][a] += 1"),
    ("shortcut: logreg never learns", SC, "w -= LR * (x.T @ (p - y) / len(train) + L2 * w)", "w -= 0 * w"),
    # the gates
    ("gate: the blink bit ignored", GT, 'seen.append("shown" if ram[u]["shot%d_hide" % s] % 2 == 0 and inside else "unseen")',
     'seen.append("shown" if inside else "unseen")'),
    ("gate: only frame n looked at", GT, "    for u in (t - 4, t):\n        for s in (1, 2):", "    for u in (t,):\n        for s in (1, 2):"),
    ("gate: off the screen ignored", GT, "inside = 8 <= ram[u][\"shot%d_x\" % s] - left <= 248", "inside = True"),
    ("gate: a fireball never shown is a no", GT, '    return None if seen else "no"', '    return "no"'),
    ("gate: the side never checked", GT, 'and ("side" not in want or want["side"] == r.get("side")))', ")"),
    ("gate: q4 always ground", GT, 'return {"answer": "ground" if ram[t]["p%d_y" % s] == 192 else "air", "side": side}',
     'return {"answer": "ground", "side": side}'),
    ("gate: the episode's start never checked", GT, 'if "unknown" in seq or len(set(seq)) != 1:',
     'if "unknown" in seq:'),
    ("gate: split rule never checked", GT, 'if r["_file"] != split_of(*m) or r["split"] != r["_file"]:', "if False:"),
    ("gate: drawn checks either image, not the frames RAM names", GT, "all(blue(r, FRAME[f]) for f in r[\"fire_frames\"])", "any(blue(r, i) for i in (0, 1))"),
]


def run(fault) -> bool:
    name, path, old, new = fault
    full = os.path.join(ROOT, path)
    with open(full) as f:
        src = f.read()
    if src.count(old) != 1:
        raise SystemExit("fault %r: the line to mutate is not found once in %s" % (name, path))
    try:
        with open(full, "w") as f:
            f.write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS, cwd=ROOT,
                           capture_output=True, text=True)
    finally:
        with open(full, "w") as f:
            f.write(src)
    return r.returncode != 0


def main() -> int:
    missed = 0
    for fault in FAULTS:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]), flush=True)
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
