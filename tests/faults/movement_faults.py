"""Seeded faults for the movement test (docs/prereg_movement.md; 2026-10-01): each mutates one line of the label, the
builder or the eval, runs the movement tests, and restores the file. Every fault must turn the tests red; exit 1 if
one stays green.

    python tests/faults/movement_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_movement.py", "tests/test_movement_data.py", "tests/test_movement_eval.py"]
LABEL, DATA, EVAL, SCRIPT = ("sf2/data/movement.py", "sf2/data/movement_data.py", "sf2/data/movement_eval.py",
                             "scripts/eval_movement.py")
FAULTS = [
    # 1 the label
    ("label: a block react counts as being hit", LABEL,
     'if st == THROWN or (st == HIT and r["p2_react"] not in BLOCK_REACTS):', "if st in (THROWN, HIT):"),
    ("label: walking needs more than 2 px", LABEL, "if abs(dx) < walk_px:", "if abs(dx) <= walk_px:"),
    ("label: no lag (the decision row, not the displayed one)", LABEL,
     "return movement(rows, n - lag, walk_px)", "return movement(rows, n, walk_px)"),
    ("label: y above ground ignored", LABEL, 'if st == JUMP or r["p2_y"] != GROUND_Y:', "if st == JUMP:"),
    ("label: toward and away swapped", LABEL, 'return "walking toward me" if dx * side > 0 else "walking away"',
     'return "walking toward me" if dx * side < 0 else "walking away"'),
    ("label: impossible x not guarded", LABEL,
     'if not (_x_ok(p, "p2_x") and _x_ok(r, "p1_x", "p2_x")):', "if p is None:"),
    # 2 the builder
    ("data: split recomputed instead of read from test_data_u", DATA,
     'split = SPLIT_OF_FILE[dec["u_file"]]', 'split = U.split_of(char, dec["opp"], dec["game"])'),
    ("data: train left unbalanced", DATA, 'files["train"], p = balance(nat_train, char)',
     'files["train"], p = nat_train, []'),
    ("data: val cap not by u_data.val_rank", DATA,
     'val_pool = sorted(natural.pop("val", []), key=lambda r: U.val_rank(r["decision"]))',
     'val_pool = sorted(natural.pop("val", []), key=lambda r: r["decision"])'),
    ("data: frames copied, not symlinked", DATA, "    os.symlink(os.path.abspath(os.path.join(u_dir, \"frames\")),",
     "    __import__(\"shutil\").copytree(os.path.abspath(os.path.join(u_dir, \"frames\")),"),
    ("data: rows not tagged perception (train.py's stage-1 coverage would apply)", DATA,
     '"perception": True, "note_version": NOTE_VERSION', '"perception": False, "note_version": NOTE_VERSION'),
    ("data: unknown labels not masked", DATA, "if answer == UNKNOWN:", "if False:"),
    # 3 the eval
    ("eval: balanced accuracy counts absent answers as 0", EVAL,
     "return sum(vals) / len(vals) if vals else None", "return sum(vals) / len(ANSWERS) if vals else None"),
    ("eval: before's neutral also scores crouching and jumping", EVAL,
     'BEFORE_MAP: Dict[str, tuple] = {"neutral": STANDING_OR_WALKING,',
     'BEFORE_MAP: Dict[str, tuple] = {"neutral": STANDING_OR_WALKING + ("crouching", "jumping"),'),
    ("eval: before's recovering not mapped to attacking", EVAL,
     '"recovering after a miss": ("attacking",)', '"recovering after a miss": ()'),
    ("eval: verdict ignores attacking recall", EVAL,
     '"attacking_recall_ge_0.5": att is not None and att >= ATTACK_RECALL_MIN', '"attacking_recall_ge_0.5": True'),
    ("eval: verdict ignores the majority baseline", EVAL,
     "floor = max(CHANCE, maj if maj is not None else CHANCE)", "floor = CHANCE"),
    ("eval: before asked the new question, not round 1's", SCRIPT,
     "q = U.perception_question(PHASE)", "q = M.movement_question()"),
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
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]))
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
