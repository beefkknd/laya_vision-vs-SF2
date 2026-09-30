"""Seeded faults for the table x advice 2x2 (docs/prereg_2x2.md): each mutates one line of the code, runs the tests
that must catch it, and restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/table_2x2_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_table_advice.py", "tests/test_table_runs.py", "tests/test_factorial_report.py",
         "tests/test_advice.py", "tests/test_system1_value.py", "tests/test_value_oracle.py", "tests/test_loop_book.py",
         "tests/test_track_record.py", "tests/test_compare_prompts.py",
         "tests/test_table_guards.py", "tests/test_boundary.py", "tests/test_laya_evidence.py"]
FAULTS = [
    ("1 rating: the works margin off", "sf2/system1/advice.py",
     "NET_WORKS_MARGIN = 3.0", "NET_WORKS_MARGIN = 2.0"),
    ("1 rating: a move worth exactly walking in 'may work'", "sf2/system1/advice.py",
     "MAY if score > walk else FAILS", "MAY if score >= walk else FAILS"),
    ("1 rating: the absolute scale (forward's value ignored)", "sf2/system1/system1.py",
     "walk=values.get(FORWARD, 0.0))", "walk=0.0)"),
    ("1 rating: blocks on the P(blocked) scale under the table", "sf2/system1/advice.py",
     "    if scale == \"net\":\n        return WORKS",
     "    if scale == \"net\" and move not in BLOCK_MOVES:\n        return WORKS"),
    ("1 rating: the P(hit) path rated on the net scale", "sf2/system1/advisor.py",
     "    out = {m: rating(scores[m], m, scale, walk) for m in best + sorted(set(named))}",
     "    out = {m: rating(scores[m], m, \"net\", walk) for m in best + sorted(set(named))}"),
    ("2 table mode: the advisor path rates nets on the P(hit) scale", "sf2/system1/system1.py",
     "self.lessons(), self.attacks + self.blocks, scale=\"net\",",
     "self.lessons(), self.attacks + self.blocks, scale=\"p_hit\","),
    ("2 table mode: the advisor is ignored (returns argmax as before)", "sf2/system1/system1.py",
     "        if self.advisor is None:\n            return {\"action\": best, **out}",
     "        if True:\n            return {\"action\": best, **out}"),
    ("2 table mode: the no-advice arm still gets the lessons", "sf2/system1/system1.py",
     "        if not self.advice_on:\n            return []",
     "        if False:\n            return []"),
    ("3 track record: table runs mixed into runs/all8's record", "sf2/system2/track_record.py",
     "        if rk != ranking:", "        if False:"),
    ("4 qwen_lessons: the table arm still loads laya-vision", "scripts/qwen_lessons.py",
     "s1 = System1(None if table is not None else args.model, ME,", "s1 = System1(args.model, ME,"),
    ("4 qwen_lessons: --oracle not passed to the arms", "scripts/qwen_lessons.py",
     "              + ([\"--oracle\", args.oracle] if getattr(args, \"oracle\", None) else [])\n", ""),
    ("4 qwen_lessons: shared text laya reserves the full model budget", "scripts/qwen_lessons.py",
     "return RUN_JOB_SHARED_GB if getattr(args, \"shared_text_laya\", False) else MODEL_JOB_GB",
     "return MODEL_JOB_GB"),
    ("4 qwen_lessons: no +table in the run name", "scripts/qwen_lessons.py",
     "+ (\"+table\" if getattr(args, \"oracle\", None) else \"\")", "+ \"\""),
    ("4 ab_memory: run.json names no table", "scripts/ab_memory.py",
     "        return {\"model\": None, \"oracle\": path, \"oracle_sha256\": hashlib.sha256(f.read()).hexdigest()}",
     "        return {\"model\": None}"),
    ("4 compare_prompts: table runs labelled like runs/all8's", "scripts/compare_prompts.py",
     "            + (\"+table\" if v.get(\"oracle\") else \"\"))", "            + \"\")"),
    ("guard: a +table name not seen as a table run", "sf2/eval/logs.py",
     "        if os.path.basename(x).endswith(TABLE_SUFFIX):", "        if False:"),
    ("guard: a run.json naming an oracle not seen", "sf2/eval/logs.py",
     "            if oracle:\n                return", "            if False:\n                return"),
    ("guard: table decisions not counted", "sf2/eval/logs.py",
     'return sum("values" in r and not r.get("top3") for r in rows)', "return 0"),
    ("guard: boundary traces table rows", "sf2/eval/boundary.py", "    if table:", "    if False:"),
    ("guard: boundary CLI takes a table batch", "scripts/boundary.py", "        if why:", "        if False:"),
    ("guard: boundary CLI reads table ledgers", "scripts/boundary.py",
     "return [read(p) for p in found if not table_run(os.path.dirname(os.path.dirname(p)))]",
     "return [read(p) for p in found]"),
    ("guard: laya_evidence reads table runs", "scripts/laya_evidence.py", "        if why:", "        if False:"),
    ("guard: gaps reports table decisions", "sf2/eval/gaps.py", "    if table:", "    if False:"),
    ("guard: report gaps on a table session", "scripts/report.py", "        if why:", "        if False:"),
    ("5 report: interaction sign flipped", "scripts/factorial_report.py",
     "\"interaction\": [(t1_ - t0_) - (a1_ - a0_) for", "\"interaction\": [(a1_ - a0_) - (t1_ - t0_) for"),
    ("5 report: cells taken from the wrong arm", "scripts/factorial_report.py",
     "CELLS = {\"A0\": (\"all8\", \"none\"), \"A1\": (\"all8\", \"loop\"),",
     "CELLS = {\"A0\": (\"all8\", \"loop\"), \"A1\": (\"all8\", \"none\"),"),
    ("5 report: throws counted at every range", "scripts/factorial_report.py",
     "close = [a for a in acts if a.get(\"range\") == \"close\"]", "close = list(acts)"),
    ("5 report: unequal rounds silently truncated", "scripts/factorial_report.py",
     "    if len(set(n.values())) != 1:", "    if False:"),
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
