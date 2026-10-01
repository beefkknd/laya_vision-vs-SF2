"""Seeded faults for the movement dataset v2 (docs/prereg_movement_data.md, gate 4): each mutates one line of the
collector, its files, the builder or the dataset gate, runs their tests, and restores the file. Every fault must
turn the tests red; exit 1 if one stays green.

    python tests/faults/movement_data2_faults.py
"""
import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMEOUT = 180            # a fault that makes a loop endless is caught (red), not waited for
TESTS = ["tests/test_movement_collect.py", "tests/test_movement_collect_io.py", "tests/test_movement_data2.py",
         "tests/test_movement_gate.py"]
COL, IO, DATA, GATE = ("sf2/data/movement_collect.py", "sf2/data/movement_collect_io.py",
                       "sf2/data/movement_data2.py", "sf2/data/movement_gate.py")
FAULTS = [
    # 1 the collector's core
    ("collector: the now image is the decision row's, not lag 1", COL,
     "k_prev, k_now = u - 4 + LAG, u + LAG", "k_prev, k_now = u - 4 + LAG, u"),
    ("collector: the prev image is 5 frames back", COL,
     "k_prev, k_now = u - 4 + LAG, u + LAG", "k_prev, k_now = u - 5 + LAG, u + LAG"),
    ("collector: the quota is not enforced", COL, "if self.left[ans] <= 0:\n                break",
     "if False:\n                break"),
    ("collector: rows before t = 4 start episodes", COL,
     "ans = M.movement(self.rows, k) if k >= FIRST_T else UNKNOWN", "ans = M.movement(self.rows, k)"),
    ("collector: one pair for every episode", COL, "if L < SHORT:", "if L < 10 ** 9:"),
    ("collector: thirds all taken from the start", COL,
     "cand = [u for u in range(a, b + 1) if 3 * (u - base) // m == i]",
     "cand = [u for u in range(a, b + 1) if 3 * (u - base) // m == 0]"),
    ("collector: long episodes not flagged", COL, '"long": s < lo,', '"long": False,'),
    ("collector: the ring's oldest capture not respected", COL,
     "lo = max(FIRST_T, oldest + 4 - LAG)", "lo = FIRST_T"),
    ("collector: the answer read one row early", COL,
     "ans = M.movement(self.rows, k) if k >= FIRST_T else UNKNOWN",
     "ans = M.movement(self.rows, k - 1) if k >= FIRST_T else UNKNOWN"),
    ("collector: the last episode not closed at the round end", COL,
     "self._close(len(self.rows) - 1, cut_end=True)", "pass"),
    ("collector: the round end episode not flagged", COL,
     "self._close(len(self.rows) - 1, cut_end=True)", "self._close(len(self.rows) - 1, cut_end=False)"),
    ("collector: a later RUN's row 0 fed again", COL, "first = 0 if self.fresh else 1", "first = 0"),
    ("collector: split by game wrong (val as train)", COL, 'return "val" if game % 10 in VAL_INDEX else "train"',
     'return "train"'),
    # 2 the collector's files and loop
    ("io: resume counts pairs of uncommitted games", IO,
     'if p.get("game") in done and (p["answer"], p["split"]) in counts:', 'if (p["answer"], p["split"]) in counts:'),
    ("io: the game cap ignored", IO, "if prog.played >= cap:", "if False:"),
    ("io: games of a full split not skipped", IO,
     "while not _open(prog, targets, split_of_game(game), answers):", "while False:"),
    ("io: the memory check ignored", IO, "if rss > mem_cap_gb:", "if False:"),
    ("io: images saved raw (no HUD frame padding)", IO, "Image.fromarray(hud_frame(im))", "Image.fromarray(im)"),
    ("io: a crashed game's number reused", IO,
     '| _games_on_disk(base)', ''),
    # 3 the builder
    ("data: uncommitted pairs kept", DATA, "pairs, dropped = IO.committed_pairs(src)",
     'pairs, dropped = IO.read_jsonl(os.path.join(src, "pairs.jsonl")), 0'),
    ("data: rows not tagged perception", DATA, '"perception": True,', '"perception": False,'),
    ("data: the split not checked against the game", DATA,
     'if p["split"] != C.split_of_game(p["game"]):', "if False:"),
    ("data: the quota not checked", DATA, "for s, c in counts.items() for a, n in c.items() if n > quota[s]]",
     "for s, c in counts.items() for a, n in c.items() if False]"),
    ("data: missing images not checked", DATA,
     "if not os.path.exists(os.path.join(img_dir, n))]", "if False]"),
    # 4 the gate
    ("gate: label re-derived at the capture row (lag 0)", GATE, "got = independent_answer(ram, t)",
     "got = independent_answer(ram, t + 1)"),
    ("gate: label mismatches not counted", GATE, "if not ok:\n                    bad += 1", "if False:\n                    bad += 1"),
    ("gate: stage share threshold ignored", GATE, "if c[s] / n < min_share}", "if False}"),
    ("gate: counts read val instead of test", GATE, 'for split, need in (("train", min_train), ("test", min_test)):',
     'for split, need in (("train", min_train), ("val", min_test)):'),
    ("gate: disk budget ignored", GATE, '"pass": gb < max_gb,', '"pass": True,'),
    ("gate: alignment accepts any best lag", GATE,
     "strict = all(ad[str(LAG)] > ad[str(L)] for L in LAGS if L != LAG)", "strict = True"),
    ("gate: alignment needs no evidence", GATE, "n_disc < min_disc or n_disc == 0", "False"),
    ("gate: alignment ignores the random-sample agreement", GATE, "return ag[str(LAG)] >= min_agree and",
     "return True and"),
    ("gate: independent label ignores block reacts", GATE,
     'return "blocking" if react in (0x06, 0x08) else "being hit"', 'return "being hit"'),
    ("gate: missing RAM passes", GATE, "and not missing,", ","),
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
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS,
                               cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
            red = r.returncode != 0
        except subprocess.TimeoutExpired:
            red = True
    finally:
        with open(full, "w") as f:
            f.write(src)
    return red


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # a kill still restores the mutated file
    missed = 0
    for fault in FAULTS:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]), flush=True)
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
