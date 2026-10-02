"""Seeded faults for scripts/train.py --balance question (docs/eye_questions_v1.md, eye_all): each mutates one line
of the question weights, their check or train.py's hand-off to laya, runs the tests, and restores the file. Every
fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/question_share_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_train_question_share.py", "tests/test_train_v3.py", "tests/test_train_min_sampled.py"]
TD, TR = "sf2/data/train_data.py", "scripts/train.py"
FAULTS = [
    ("weights: every dir equal (q3 gets 1/3)", TD, "return {n: 1.0 / (len(by_q) * len(ns)) for ns in by_q.values() for n in ns}",
     "return {n: 1.0 for ns in by_q.values() for n in ns}"),
    ("weights: question = the dir itself", TD, "by_q.setdefault(os.path.basename(os.path.dirname(p)), []).append(os.path.basename(p))",
     "by_q.setdefault(os.path.basename(p), []).append(os.path.basename(p))"),
    ("weights: name collision not refused", TD, "    if len(set(names)) != len(names):\n        raise ValueError(\"answer dir",
     "    if False:\n        raise ValueError(\"answer dir"),
    ("check: question totals never compared", TD, "if max(totals.values()) - min(totals.values()) > SHARE_TOLERANCE:", "if False:"),
    ("check: answers within a question never compared", TD, "for q, v in by_q.items() if max(v) - min(v) > SHARE_TOLERANCE]",
     "for q, v in by_q.items() if False]"),
    ("check: reads the default mix, not the question mix", TD,
     "shares = sampling_shares(train, question_weights(dirs) if by_question else MIX_WEIGHTS)",
     "shares = sampling_shares(train, MIX_WEIGHTS)"),
    ("check: row minimums dropped", TD, 'problems += ["%s has %d %s rows < %d" % (x, n[x], split, need) for x in names if n[x] < need]',
     'problems += []'),
    ("train: laya gets the default mix", TR, 'return TD.question_weights(args.data) if args.balance == "question" else TD.MIX_WEIGHTS',
     "return TD.MIX_WEIGHTS"),
    ("train: question mode checked as plain sampling", TR,
     "train, val, args.data, args.min_sampled_train, args.min_sampled_val, by_question=True)",
     "train, val, args.data, args.min_sampled_train, args.min_sampled_val)"),
    ("train: default changed to question", TR, 'choices=["rows", "sampling", "question"], default="rows"',
     'choices=["rows", "sampling", "question"], default="question"'),
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
