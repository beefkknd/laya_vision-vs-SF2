"""Seeded faults for the movement fine-tune scoring (sf2.data.mv_eval): each mutates one line, runs
tests/test_mv_eval.py, restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/mv_eval_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = "sf2/data/mv_eval.py"
FAULTS = [
    ("balanced accuracy is plain accuracy", '"balanced_accuracy": sum(present) / len(present) if present else None',
     '"balanced_accuracy": _ratio(sum(conf[a][a] for a in answers), len(truths))'),
    ("majority tie to the later answer", "return max(answers, key=lambda a: (n[a], -list(answers).index(a)))",
     "return max(answers, key=lambda a: (n[a], list(answers).index(a)))"),
    ("learned ignores the floor", '"learned": lb is not None and lb > floor', '"learned": lb is not None and lb >= 0.0'),
    ("confusion transposed", "conf[t][p] += 1", "conf[p][t] += 1"),
    ("bootstrap by row, not by match", 'games[(r["pair_name"], r["game"])].append(i)',
     'games[(r["pair_name"], r["game"], i)].append(i)'),
    ("facing breakdown by character", 'breakdown(rows, preds, "facing", answers)', 'breakdown(rows, preds, "char", answers)'),
    ("floor without the majority baseline", 'floor = max(chance, base_maj["balanced_accuracy"] or 0.0)',
     "floor = chance / 2"),
    ("side breakdown by facing", 'breakdown(rows, preds, "side", answers)', 'breakdown(rows, preds, "facing", answers)'),
    ("side breakdown on round-1 rows too", 'if rows and all("side" in r for r in rows):', "if rows:"),
]


def run(old, new) -> bool:
    path = os.path.join(ROOT, EV)
    src = open(path).read()
    if src.count(old) != 1:
        raise SystemExit("line to mutate not found once: %r" % old)
    try:
        open(path, "w").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "tests/test_mv_eval.py"], cwd=ROOT, capture_output=True)
    finally:
        open(path, "w").write(src)
    return r.returncode != 0


def main() -> int:
    missed = 0
    for name, old, new in FAULTS:
        red = run(old, new)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", name), flush=True)
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
