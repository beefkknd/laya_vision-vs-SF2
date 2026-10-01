"""Seeded faults for the action label, its collector and stop rule, the three-dataset builder and its gate
(docs/prereg_movement_data.md, "Owner decisions on the label"): each mutates one line, runs the tests, and restores
the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/action_data_faults.py
"""
import os
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMEOUT = 180
TESTS = ["tests/test_action_codes.py", "tests/test_action_collect.py", "tests/test_action_data.py"]
COD, COL, IO, DATA, GATE = ("sf2/data/action_codes.py", "sf2/data/action_collect.py", "sf2/data/action_collect_io.py",
                            "sf2/data/action_data.py", "sf2/data/action_gate.py")
FAULTS = [
    # 1 the label
    ("codes: the LAST attack ID names the run", COD,
     'first = next((_f(rows[i], p, "aid") for i in range(s, e + 1) if _f(rows[i], p, "aid")), 0)',
     'first = next((_f(rows[i], p, "aid") for i in range(e, s - 1, -1) if _f(rows[i], p, "aid")), 0)'),
    ("codes: a jump with an attack stays a jump", COD, "    if first:\n", "    if first and st != JUMP_STATE:\n"),
    ("codes: IDs above the max kept", COD, 'if first <= ATTACK_ID_MAX else (None, "id_range")',
     'if True else (None, "id_range")'),
    ("codes: throw class read from the run's first row too", COD, "tail = rows[s + 1:e + 1]", "tail = rows[s:e + 1]"),
    ("codes: the other player's shot slot", COD, 'rows[i]["shot%d" % p]', 'rows[i]["shot%d" % other(p)]'),
    ("codes: special class not offset", COD, "return (SPECIAL_BASE + sc, \"special\")", "return (sc, \"special\")"),
    ("codes: block react read as hit", COD, "return BLOCK if _f(r, p, \"react\") in BLOCK_REACTS else HIT",
     "return HIT"),
    ("codes: air before guard (precedence)", COD, "    if st == GUARD_STATE:\n        return BLOCK\n",
     "    if st == GUARD_STATE and _f(r, p, \"y\") == GROUND_Y:\n        return BLOCK\n"),
    ("codes: walk direction flipped", COD, "return WALK_TOWARD if dx * side > 0 else WALK_AWAY",
     "return WALK_TOWARD if dx * side < 0 else WALK_AWAY"),
    ("codes: walk reads t - 3", COD, "prev = rows[t - FIRST_T]", "prev = rows[t - 3]"),
    ("codes: impossible x accepted", COD, "if not all(0 <= x <= STAGE_X for x in xs):", "if False:"),
    ("codes: stage thirds off by one", COD, "return 1 + 3 * (t - s) // (e - s + 1)",
     "return 1 + 3 * (t - s) // (e - s + 2)"),
    ("codes: rows before FIRST_T labelled", COD, "        if k < FIRST_T:\n            return (\"pre\",)",
     "        if k < 0:\n            return (\"pre\",)"),
    ("codes: a new run every row", COD,
     'run_start = self.start if self.key is not None and self.key[0] == "run" and self.key[1] == st else k',
     "run_start = k"),
    ("codes: round-end episode not flagged", COD, "ep = self._episode(rows, len(rows) - 1, True)",
     "ep = self._episode(rows, len(rows) - 1, False)"),
    # 2 the collector
    ("collect: the cap ignored", COL, "if self.counts[key] >= self.cap:", "if False:"),
    ("collect: a displayed row reused", COL, "cand = [u for u in range(a, b + 1) if u not in used]",
     "cand = list(range(a, b + 1))"),
    ("collect: the now image at lag 0", COL, "k_prev, k_now = u - 4 + LAG, u + LAG", "k_prev, k_now = u - 4 + LAG, u"),
    ("collect: the ring's oldest not respected", COL, "lo = max(A.FIRST_T, oldest + 4 - LAG)", "lo = A.FIRST_T"),
    ("collect: one pair for every episode", COL, "if e - s + 1 < SHORT:", "if True:"),
    ("collect: only his episodes", COL, "        for p in (1, 2):\n            ep = self.tracks[p].step",
     "        for p in (2,):\n            ep = self.tracks[p].step"),
    ("collect: unknown episodes not counted", COL, "self.unknown[(actor, ep.reason)] += 1", "pass"),
    ("collect: last episodes not closed", COL, "            ep = self.tracks[p].finish(self.rows)",
     "            ep = None"),
    # 3 the loop / stop rule
    ("io: no min games", IO, "if prog.played >= min_games and prog.since_new >= patience:",
     "if prog.since_new >= patience:"),
    ("io: patience off by one", IO, "prog.since_new >= patience:", "prog.since_new > patience:"),
    ("io: hard cap ignored", IO, "if prog.played >= cap:", "if False:"),
    ("io: novelty counts val", IO, 'if p["split"] in NOVEL_SPLITS\n', "if True\n"),
    ("io: patience never resets", IO, "0 if new else prog.since_new + 1", "prog.since_new + 1"),
    ("io: resume counts uncommitted pairs", IO, "    for g in games:\n        prog = advance(",
     "    for g in [{\"game\": x} for x in sorted(set(by_game) | {g[\"game\"] for g in games})]:\n"
     "        prog = advance("),
    ("io: a crashed game's number reused", IO, "| _games_on_disk(base)", ""),
    ("io: memory check ignored", IO, "if rss > mem_cap_gb:", "if False:"),
    ("io: caps not carried over", IO, "prog.counts[split], caps[split]", "Counter(), caps[split]"),
    # 4 the builder
    ("data: the other fighter not labelled", DATA, "        for pl in (1, 2):\n                lab = labs[pl]",
     "        for pl in (p[\"player\"],):\n                lab = labs[pl]"),
    ("data: collector label not checked", DATA, "if src_lab is None or (src_lab[0].code, src_lab[1]) != (p[\"code\"], p[\"stg\"]) or \\",
     "if False and (src_lab[0].code, src_lab[1]) != (p[\"code\"], p[\"stg\"]) or \\"),
    ("data: split not checked", DATA, 'if p["split"] != split_of_game(p["game"]):', "if False:"),
    ("data: missing images not checked", DATA, "if not os.path.exists(os.path.join(img_dir, n))]", "if False]"),
    ("data: duplicate pairs not caught", DATA, "for (g, t), n in seen.items() if n > 1]",
     "for (g, t), n in seen.items() if n > 2]"),
    ("data: where read from her y", DATA, 'return "on the ground" if rows[t]["p2_y"] == GROUND_Y',
     'return "on the ground" if rows[t]["p1_y"] == GROUND_Y'),
    ("data: the act question does not say which fighter", DATA, '"What move is %s (%s) doing?" % (NAMES[actor], WHO[player])',
     '"What move is he doing?"'),
    ("data: stage answer off by one", DATA, '"stg%d" % stg), **extra))', '"stg%d" % max(1, stg - 1)), **extra))'),
    ("data: head tokens miss the terminators", DATA, "sum(len(enc(OPTION_BULLET + o)) + 1",
     "sum(len(enc(OPTION_BULLET + o))"),
    ("data: val games in train", DATA, 'files[FILES[p["split"]]] += rows_for', 'files[FILES["train" if p["split"] == "val" else p["split"]]] += rows_for'),
    # 5 the gate
    ("gate: mismatches not counted", GATE, "        nonlocal bad\n        bad += 1", "        nonlocal bad\n        bad += 0"),
    ("gate: act completeness not checked", GATE, "                            if mine:\n                                fail(mine[0], None)",
     "                            pass"),
    ("gate: stage not compared", GATE, '"stage": "stg%d" % want[1]}', '"stage": answers.get("stage")}'),
    ("gate: images not checked", GATE, "return r[\"images\"] == imgs and", "return True and"),
    ("gate: independent walk direction flipped", GATE, "return 71 if (moved > 0) == (toward > 0) else 72",
     "return 72 if (moved > 0) == (toward > 0) else 71"),
    ("gate: independent stage at lag 0", GATE, "out[k + T0] = (code[i], 1 + (3 * (k - i)) // length)",
     "out[k + T0 + 1] = (code[i], 1 + (3 * (k - i)) // length)"),
    ("gate: missing RAM passes", GATE, "and not missing,", ","),
    ("gate: disk budget ignored", GATE, '"disk": disk_check(data, meta["opps"], max_gb),', '"disk": {"pass": True},'),
    ("gate: distance band ignores the poke band", GATE, '    if gap <= th["poke_max"].get("chunli", th["poke_max"]["all"]):\n        return "poke"\n', ""),
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
    only = sys.argv[1:]
    missed = 0
    faults = [f for f in FAULTS if not only or any(o in f[0] for o in only)]
    for fault in faults:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]), flush=True)
    print("%d of %d faults caught" % (len(faults) - missed, len(faults)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
