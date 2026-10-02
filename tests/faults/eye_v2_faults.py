"""Seeded faults for the eye's questions v2 (docs/eye_questions_v1.md, "Questions v2 - relabel" / "Datasets v2
(relabel)"): each mutates one line of the v2 labels, the pool, the builder or the gates, runs the v2 tests, and restores
the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/eye_v2_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_eye_v2.py"]
V2, PL, DA, GT = "sf2/data/eye_v2.py", "sf2/data/eye_pool.py", "sf2/data/eye_data.py", "sf2/data/eye_gate.py"
FAULTS = [
    # the labels
    ("labels: special folded into attack", V2, '"special": "special attack"', '"special": "attack"'),
    ("labels: crouch is its own answer", V2, '"crouch": "stand"', '"crouch": "crouch"'),
    ("labels: down is not hit", V2, '"hit": "hit", "down": "hit"', '"hit": "hit", "down": "stand"'),
    ("labels: the pressed word ignored", V2, "mv = L.movement_pressed(rows, t, p, pressed)[0]",
     "mv = L.movement(rows, t, p)"),
    ("labels: episode checks frame n only", V2, "for u in range(t - gap, t))", "for u in range(t, t))"),
    ("labels: crouching attack class ignored", V2, 'if st == L.ATTACK and row["p%d_mclass" % p] == CROUCH_ATTACK_CLASS:',
     "if st == L.ATTACK:"),
    ("labels: crouching guard sub 6 only", V2, "CROUCH_GUARD_SUBS = (0x04, 0x06)", "CROUCH_GUARD_SUBS = (0x06,)"),
    ("labels: crouching block stun ignored", V2,
     '(st == L.HIT and row["p%d_react" % p] == CROUCH_BLOCK_REACT)', "False"),
    ("labels: standing block stun is low", V2, "CROUCH_BLOCK_REACT = 0x08", "CROUCH_BLOCK_REACT = 0x06"),
    ("labels: high from the jump state, not y", V2, 'if r["p%d_y" % p] != GROUND_Y:', 'if r["p%d_state" % p] == L.JUMP:'),
    ("labels: dir keeps the space", V2, 'return answer.replace(" ", "_")', "return answer"),
    # the pool
    ("pool: games past max_game kept", PL, "if game not in logs or (max_game is not None and game > max_game):",
     "if game not in logs:"),
    ("pool: act2 without the pressed class", PL, '"act2": V2.act2(rows, t, s, classes[t])', '"act2": V2.act2(rows, t, s)'),
    # the builder
    ("data: q3v2 outside an episode kept", DA,
     'if base_of(q) == "q3v2" and (me["act2"] is None or not me["act2_in_episode"]):',
     'if base_of(q) == "q3v2" and me["act2"] is None:'),
    ("data: q4v2 asks the air", DA, '"q3v2": "act2", "q4v2": "pos", "q3v2b"', '"q3v2": "act2", "q4v2": "air", "q3v2b"'),
    ("data: q4v2b keeps the jump state on the ground", DA, 'if Q[q].get("drop_jump_ground") and me["jump_ground"]:',
     "if False:"),
    ("data: q3v2b still matches on the game", DA, 'game = (f["game"],) if Q[q].get("game_in_stratum", True) else ()',
     'game = (f["game"],)'),
    ("data: q3v2b capped like q3v2", DA, 'base="q3v2", cap=None,', 'base="q3v2", cap=2,'),
    ("pool: jump_ground ignores y", PL, 'rows[t]["p%d_state" % s] == L.JUMP and L.air(rows, t, s) == "ground"',
     'rows[t]["p%d_state" % s] == L.JUMP'),
    ("gate: q4v2b accepts the jump state on the ground", GT,
     'if q == "q4v2b" and ram[t]["p%d_state" % s] == 4 and ram[t]["p%d_y" % s] == 192:', "if False:"),
    ("data: answer dirs keep the space", DA, "base = os.path.join(out, dir_of(a))", "base = os.path.join(out, a)"),
    # the gates
    ("gate: position ignores the crouching attack", GT, "    if st == 10 and mcl == 2:\n", "    if False:\n"),
    ("gate: position ignores the crouching guard", GT, "if (st == 8 and sub in (4, 6)) or (st == 14 and react == 8):",
     "if st == 14 and react == 8:"),
    ("gate: down is not hit", GT, '"hit": "hit", "down": "hit"}', '"hit": "hit", "down": "stand"}'),
    ("gate: crouch is not stand", GT, '"stand": "stand", "crouch": "stand"', '"stand": "stand", "crouch": "low"'),
    ("gate: the dir never checked", GT, 'and r["_dir"] == r["answer"].replace(" ", "_")', ""),
    ("gate: v2 episode as the v1 grid", GT, '        if q in ("q3v2", "q3v2b"):\n            seq', '        if False:\n            seq'),
    ("gate: walk needs a direction (the v1 grid cell)", GT, 'return ACT2_FROM.get(lab["movement"], "unknown")',
     'return ACT2_FROM.get(PG._grid(lab), "unknown")'),
    ("gate: second fact always agrees", GT, "        res[why][\"agree\"] += bool(ok)", "        res[why][\"agree\"] += 1"),
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
