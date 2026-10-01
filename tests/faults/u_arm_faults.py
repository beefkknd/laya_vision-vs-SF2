"""Seeded faults for the U arm (2026-10-01: the dataset, the training hooks, the eye, the runners, the offline eval).
Each mutates the code, runs the tests that must catch it, and restores the file. Every fault must turn the tests red;
exit 1 if one stays green.

    python tests/faults/u_arm_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_u_data.py", "tests/test_u_train.py", "tests/test_eye.py", "tests/test_u_runs.py",
         "tests/test_eval_u.py", "tests/test_checkpoint_tags.py", "tests/test_factorial_report.py",
         "tests/test_table_runs.py"]
UD, TD, EYE, S1 = "sf2/data/u_data.py", "sf2/data/train_data.py", "sf2/system1/eye.py", "sf2/system1/system1.py"
QL, AB, FR, EV = "scripts/qwen_lessons.py", "scripts/ab_memory.py", "scripts/factorial_report.py", "scripts/eval_u.py"
FAULTS = [
    # 1. the dataset
    ("data: the HUD is blanked", UD, "    out[:h] = img\n    return out", "    out[:h] = img\n    out[:62] = 0\n    return out"),
    ("data: the note carries RAM", UD, 'return "me=%s" % me', 'return "me=%s dist=mid" % me'),
    ("data: unknown labels are kept", UD, "if lab[key] == UNKNOWN:", "if lab[key] == 'never':"),
    ("data: question 8 asked for walking in", UD, "return [m for m in choices(me) if m != FORWARD]",
     "return list(choices(me))"),
    ("data: question 8 hard targets", UD, "target=[float(p[a]) for a in Q8_ANSWERS]",
     "target=[float(a == word) for a in Q8_ANSWERS]"),
    ("data: thresholds argument ignored", UD, "lab = labels(rows, n, th)",
     'lab = labels(rows, n, dict(th, throw_max={"all": 40}))'),
    ("data: validation is game % 10 == 1", UD, "VAL_INDEX = (0,)", "VAL_INDEX = (1,)"),
    ("data: Chun-Li vs Guile not held out", UD, 'HELDOUT = ("chunli", "guile")', 'HELDOUT = ("chunli", "nobody")'),
    ("data: the RAM join is not checked", UD, "        bad = join_problem(e, hits[0])", "        bad = None"),
    ("data: the val cap splits a decision", UD, 'return ([r for r in rows if r["decision"] in keep]',
     'return ([r for r in rows if r["id"] in keep or r["decision"] in keep and r["task"] == "q8"]'),
    # 2. training hooks
    ("train: v3 not tagged perception", TD, '        return {"note_version": 3, "value_questions": False, "perception": True}',
     '        return {"note_version": 3, "value_questions": False}'),
    ("train: stage-1 coverage on perception dirs", TD, "if name not in SPECIALS or name in perception:",
     "if name not in SPECIALS:"),
    ("train: v3 read as v1", TD, "    if V3_NOTE.match(context):\n        return 3", "    if False:\n        return 3"),
    # 3. the eye and System 1
    ("eye: a RAM field reaches the decision", S1, "        return s1.decide(prev, cur, s1.eye.note), s1.eye.note",
     '        return s1.decide(prev, cur, s1.eye.note + ("" if r["p1_x"] else "")), s1.eye.note'),
    ("eye: rank weight of may work", EYE, "MAY_WEIGHT = 0.5", "MAY_WEIGHT = 1.0"),
    ("eye: walks in only when every word is worse than fails", EYE, "RATING[w] > RATING[FAILS]",
     "RATING[w] >= RATING[FAILS]"),
    ("eye: shortlist words not from question 8", EYE,
     "out: Dict[str, Optional[str]] = {m: words[m] for m in", "out: Dict[str, Optional[str]] = {m: WORKS for m in"),
    ("eye: avoided moves stay in the top 3", EYE, "best = sorted((m for m in rank if m not in out_ruled)",
     "best = sorted((m for m in rank)"),
    ("eye: poke mapped to mid", EYE, '"poke": "close"', '"poke": "mid"'),
    ("eye: blocking mapped to stunned", EYE, 'if phase == "being hit" or him in NOT_FREE:',
     'if phase in ("being hit", "blocking") or him in NOT_FREE:'),
    ("eye: air not first", EYE, '    if air != "grounded":\n        return "jumping"\n',
     '    if air == "never":\n        return "jumping"\n'),
    ("eye: untagged checkpoint accepted", EYE,
     'if not cfg.get("perception") or cfg.get("note_version") != NOTE_VERSION:', "if False:"),
    ("eye: answers not logged", S1, 'EYE_KEYS = ("eye", "rank", "eye_situation", "eye_ms")', 'EYE_KEYS = ("rank",)'),
    ("eye: the default entry gains eye keys", S1, "    for k in EYE_KEYS:                            # the eye: every "
     "answer, the rank scores, the situation it told text laya\n        if k in d:",
     "    for k in EYE_KEYS:                            # the eye: every answer, the rank scores, the situation it "
     "told text laya\n        if True:"),
    ("eye: frames not hud_frame'd", EYE, "Image.fromarray(hud_frame(cur))", "Image.fromarray(cur)"),
    # runners and report
    ("runs: no +eye in the run name", QL, '+ ("+eye" if getattr(args, "eye", None) else ""))', ")"),
    ("runs: --eye not passed to the arms", QL, '+ (["--eye", args.eye] if getattr(args, "eye", None) else [])',
     "+ []"),
    ("runs: the perception tag is not checked", QL, "        check_eye(args, s1)\n", "\n"),
    ("runs: ab_memory folder lacks +eye", AB, '"+eye" if eye else ""', '""'),
    ("report: eye runs counted as runs/all8", FR, 'run_label(v) + (EYE if v.get("eye") else "")))',
     'run_label(v) + ""))'),
    ("report: U0-T0 against A0", FR, '"U0-T0": ("U0", "T0")', '"U0-T0": ("U0", "A0")'),
    ("report: two checkpoints accepted", FR, "    if len(shas) > 1:", "    if len(shas) > 2:"),
    ("report: the commit compared for U", FR, 'U_PAIR_FIELDS = tuple(k for k in PAIR_FIELDS if k != "commit")',
     "U_PAIR_FIELDS = PAIR_FIELDS"),
    # 4. offline eval
    ("eval: gate 2 at 0.8", EV, "GATE2 = 0.9", "GATE2 = 0.8"),
    ("eval: forward-best decisions counted as hits", EV, "        top = best_in_top3(table, cell, rank)",
     "        top = best_in_top3(table, cell, rank) if table_best(table, cell) != FORWARD else True"),
    ("eval: floors ignored", EV, 'if who == "all" or not floors:', "if True:"),
    ("eval: val gated", EV, 'GATED = ("test_real", "test_heldout_guile")',
     'GATED = ("test_real", "test_heldout_guile", "val")'),
    ("eval: no decisions passes gate 2", EV, '        if not g.get("n"):\n            out.append',
     '        if not g.get("n"):\n            pass\n        if False:\n            out.append'),
]


def run(fault) -> bool:
    name, path, old, new = fault
    full = os.path.join(ROOT, path)
    with open(full) as f:
        src = f.read()
    if src.count(old) != 1:
        raise SystemExit("fault %r: the code to mutate is found %d times in %s" % (name, src.count(old), path))
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
