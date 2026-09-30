"""Seeded faults for the verified book and the forward refusal (2026-09-30): each mutates one line of the code, runs
the tests that must catch it, and restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/book_forward_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_verified_lessons.py", "tests/test_forward_prompts.py", "tests/test_book.py",
         "tests/test_loop_book.py"]
FAULTS = [
    ("book: review retires a verified lesson like a registered one", "sf2/system2/lessons.py",
     'elif r["state"] == "registered" and ev["cls"] != right:',
     'elif r["state"] in ("registered", "verified") and ev["cls"] != right:'),
    ("book: verified lines not put in play first", "sf2/system2/lessons.py",
     "return (verified + tests + [r[\"line\"] for r in lessons])[:MAX_LINES]",
     "return (tests + [r[\"line\"] for r in lessons] + verified)[:MAX_LINES]"),
    ("book: a claim contradicting a verified lesson is accepted", "sf2/system2/lessons.py",
     'LIVE = ("verified", "testing", "registered")', 'LIVE = ("testing", "registered")'),
    ("book: the builder admits a line whose interval touches 0", "scripts/book.py",
     'elif v["verdict"] == "HELPS" and v["ci95"][0] > 0:', 'elif v["mean"] > 0:'),
    ("book: the loop ignores --book", "scripts/qwen_lessons.py",
     "    reg: L.Registry = start_registry(args, arm)", "    reg: L.Registry = []"),
    ("forward: the verifier accepts claims naming forward", "sf2/system2/lessons.py",
     "UNFOLLOWABLE = (FORWARD,)", "UNFOLLOWABLE = ()"),
    ("forward: the prompts still offer forward for lessons", "sf2/system2/lesson_prompt.py",
     "return list(moves) if forward_lessons else [m for m in moves if m not in L.UNFOLLOWABLE]",
     "return list(moves)"),
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
