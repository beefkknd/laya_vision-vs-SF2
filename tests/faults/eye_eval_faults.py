"""Seeded faults for sf2.data.eye_eval (the eye fine-tunes' scoring and the eye_all keep rule): each mutates one line,
runs tests/test_eye_eval.py, and restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/eye_eval_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_eye_eval.py"]
EV = "sf2/data/eye_eval.py"
FAULTS = [
    ("keep: margin ignored", EV, "within\": combined[q] >= separate[q] - margin - 1e-12}", "within\": combined[q] >= separate[q] - 1e-12}"),
    ("keep: any question within keeps", EV, '"keep": all(v["within"] for v in per.values())', '"keep": any(v["within"] for v in per.values())'),
    ("keep: margin doubled", EV, "MARGIN = 0.02", "MARGIN = 0.04"),
    ("weights: q1 fireball share wrong", EV, '"q1": {"yes": 0.035, "no": 0.965}', '"q1": {"yes": 0.5, "no": 0.5}'),
    ("weights: q3 attacking split equally", EV, 'a = candidates["attack"] / (candidates["attack"] + candidates["special"])', "a = 0.5"),
    ("weights: plain accuracy instead of weighted", EV, "return sum(w * recall[a] for a, w in weights.items())",
     "return sum(recall[a] for a in weights) / len(weights)"),
    ("learned: lower bound vs chance only", EV, "floor = max(chance, base_maj[\"balanced_accuracy\"] or 0.0)", "floor = 0.0"),
    ("q1: hard tags not broken down", EV, 'lambda r: (r.get("hard") or ["plain"]) if r["answer"] == "no" else [])',
     'lambda r: ["plain"] if r["answer"] == "no" else [])'),
    ("q1: n-4 only read as both", EV, 'else "n-4 only" if f == {"n-4"}', 'else "both" if f == {"n-4"}'),
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
