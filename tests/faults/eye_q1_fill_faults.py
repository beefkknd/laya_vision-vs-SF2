"""Seeded faults for the q1 fireball collection (docs/eye_questions_v1.md, "Next: more fireball data"): the fill line
(sf2.data.eye_q1_fill, scripts/fill_eye_q1.py), the collector's --shot-per-game (scripts/collect_pairs.py) and the
layout rule eye_gate broke (paths from sf2.config.REPO). Each mutates one line, runs the tests, restores the file;
every fault must turn the tests red; exit 1 if one stays green. Do not run while a collection reads these files.

    python tests/faults/eye_q1_fill_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_eye_q1_fill.py", "tests/test_layout.py"]
FL, CLI, CP, GT = "sf2/data/eye_q1_fill.py", "scripts/fill_eye_q1.py", "scripts/collect_pairs.py", "sf2/data/eye_gate.py"
FAULTS = [
    ("fill: counts without the matching", FL, 'rows = E.select(cands, ANSWERS, E.Q["q1"]["cap"], seed)',
     'rows = cands'),
    ("fill: short strata over every split", FL, '        if c["split"] == "train":\n            per[c["stratum"]]',
     '        if True:\n            per[c["stratum"]]'),
    ("fill: yes lost counts the strata", FL, '"yes_lost": sum(v["yes"] - v["no"] for v in short.values())',
     '"yes_lost": len(short)'),
    ("fill: short strata include ties", FL, 'short = {k: v for k, v in per.items() if v["yes"] > v["no"]}',
     'short = {k: v for k, v in per.items() if v["yes"] >= v["no"]}'),
    ("fill: shot cells from every game", FL, 'if g["game"] >= from_game}', "}"),
    ("fill: shot cells for non-throwers", FL, "                if chars[slot] in S.THROWERS:",
     "                if True:"),
    ("fill: a cell at the cap counts as not full", FL, "not_full += taken[(g, slot, stage)] < cap",
     "not_full += taken[(g, slot, stage)] <= cap"),
    ("fill: met needs one answer only", FL, 'return all(rep["counts"][a]["train"] >= target for a in ANSWERS)',
     'return any(rep["counts"][a]["train"] >= target for a in ANSWERS)'),
    ("fill: the line shows yes twice", FL, 'c["yes"]["train"], c["no"]["train"], c["yes"]["val"]',
     'c["yes"]["train"], c["yes"]["train"], c["yes"]["val"]'),
    ("cli: exit 0 when not met", CLI, "return 0 if F.met(rep, args.target) else 2", "return 0"),
    ("cli: the report not logged", CLI, '        f.write(json.dumps(rep) + "\\n")', "        pass"),
    ("collect: the shot cap not passed", CP, "return functools.partial(S.ShotSampler, shot_per_game=shot_per_game)",
     "return S.ShotSampler"),
    ("collect: the shot cap not forwarded", CP, '(["--shots", "--shot-per-game", str(args.shot_per_game)] if args.shots',
     '(["--shots"] if args.shots'),
    ("collect: run.json keeps the default cap", CP, '"shot_per_game": args.shot_per_game,',
     '"shot_per_game": S.SHOT_PER_GAME,'),
    ("collect: the cap flag defaults to 1", CP, "ap.add_argument(\"--shot-per-game\", type=int, default=S.SHOT_PER_GAME,",
     "ap.add_argument(\"--shot-per-game\", type=int, default=1,"),
    ("layout: eye_gate finds the repo from its own file", GT,
     'THRESHOLDS = os.path.join(REPO, "lessons", "perception_thresholds_v2.json")',
     'THRESHOLDS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '
     '"lessons", "perception_thresholds_v2.json")'),
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
