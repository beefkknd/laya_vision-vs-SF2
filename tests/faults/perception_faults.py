"""Seeded faults for the U arm's perception work (2026-10-01): the RAM log, the labels (questions 1-7), the calibration
and question 8's soft targets. Each fault mutates one line, runs the tests that must catch it, and restores the file.
Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/perception_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_ram_log.py", "tests/test_perception.py", "tests/test_calibrate_perception.py",
         "tests/test_q8_targets.py"]
S1, LOG, PL = "sf2/system1/system1.py", "sf2/system1/game_log.py", "scripts/play_system1.py"
PER, CAL = "sf2/data/perception.py", "scripts/calibrate_perception.py"
FAULTS = [
    # 1. logging
    ("log: the wait RUN's first row (a repeat) goes into the stream", S1,
     "                stream.extend(rows[1:])     # row 0 of a RUN is the previous RUN's last row",
     "                stream.extend(rows)"),
    ("log: the decision index is one frame early (n is not the now image's frame)", S1,
     "            marks.append((rnd.frames, len(stream) - 1))", "            marks.append((rnd.frames, len(stream) - 2))"),
    ("log: rows kept by default (the default path changes)", S1,
     "game: int, live_path: Optional[str] = None, echo: bool = False, ram_log: bool = False) -> Round:",
     "game: int, live_path: Optional[str] = None, echo: bool = False, ram_log: bool = True) -> Round:"),
    ("log: the lookahead stops at the next decision", S1,
     "stream[max(0, i - BACK):i + 1 + LOOKAHEAD]", "stream[max(0, i - BACK):i + 1 + min(LOOKAHEAD, 4)]"),
    ("log: play_system1 children lose --ram-log", PL,
     '+ (["--ram-log"] if args.ram_log else [])', "+ []"),
    ("log: the entry stores n off by one", LOG,
     'return {"game": game, "frame": frame, "n": n,', 'return {"game": game, "frame": frame, "n": n + 1,'),
    # 2. labels
    ("labels: read at the decision row (no display lag)", PER, "LAG = 1\n", "LAG = 0\n"),
    ("labels: lag ignored in labels()", PER, '    t = n - th.get("lag", LAG)', "    t = n"),
    ("labels: throw band ignores my character", PER,
     '    if g <= _per_char(th, "throw_max", me):', '    if g <= th["throw_max"]["all"]:'),
    ("labels: trend sign flipped", PER,
     '    return "closing" if d < -th["trend_eps"] else "opening" if d > th["trend_eps"] else "steady"',
     '    return "opening" if d < -th["trend_eps"] else "closing" if d > th["trend_eps"] else "steady"'),
    ("labels: a miss ignores contact", PER,
     '    if _contact_on_me(rows, max(s or 1, 1), len(rows) - 1 if e is None else e):',
     '    if False:'),
    ("labels: contact counts stun I was already in", PER,
     '        if r["p1_state"] in STUN and prev["p1_state"] not in STUN:', '        if r["p1_state"] in STUN:'),
    ("labels: recovering in the first half too (amendment: second half only)", PER,
     'return "recovering after a miss" if 2 * (t - s) >= e - s + 1 else "attacking"',
     'return "recovering after a miss"'),
    ("labels: recovering one frame early (half boundary)", PER,
     'return "recovering after a miss" if 2 * (t - s) >= e - s + 1 else "attacking"',
     'return "recovering after a miss" if 2 * (t - s) >= e - s - 1 else "attacking"'),
    ("labels: an unseen episode start is not unknown", PER,
     "    if s is None or e is None:\n        return UNKNOWN                # no contact seen",
     "    if e is None:\n        return UNKNOWN                # no contact seen"),
    ("labels: landing not checked first", PER,
     '    if any(r["p2_y"] == GROUND_Y for r in rows[t + 1:t + k + 1]):', '    if False:'),
    ("labels: jump direction ignores where I am", PER,
     '    return dx * (r["p1_x"] - r["p2_x"]) > 0', '    return dx < 0'),
    ("labels: projectile read from my own slot", PER, '    if not r["shot2"]:', '    if not r["shot1"]:'),
    ("labels: a receding projectile still counts", PER, "    if not coming:\n        return \"none\"",
     "    if False:\n        return \"none\""),
    ("labels: dizzy without the flag", PER, "        if rows[s if s is not None else 0][dz]:", "        if True:"),
    ("labels: knockdown never seen", PER, "KNOCKDOWN_SUB, DIZZY_SUB = 0x04, 0x08", "KNOCKDOWN_SUB, DIZZY_SUB = 0x44, 0x08"),
    ("labels: corner strictly inside D", PER, '    if dist[1] <= th["corner_d"]:', '    if dist[1] < th["corner_d"]:'),
    ("labels: bars from the other fighter", PER, '"my_bar": bar(r["p1_hp"]), "his_bar": bar(r["p2_hp"]),',
     '"my_bar": bar(r["p2_hp"]), "his_bar": bar(r["p1_hp"]),'),
    ("labels: bars from life, not the drawn hp (amendment)", PER,
     '"my_bar": bar(r["p1_hp"]), "his_bar": bar(r["p2_hp"]),', '"my_bar": bar(r["p1_life"]), "his_bar": bar(r["p2_life"]),'),
    ("gate: walking in counted like any move", PER, "    if best == FORWARD:\n        return None",
     "    if False:\n        return None"),
    ("gate: top 4 instead of top 3", PER, "    return best in order[:3]", "    return best in order[:4]"),
    ("gate: ranking scores sorted the wrong way", PER,
     "sorted(ranking, key=lambda m: -ranking[m])", "sorted(ranking, key=lambda m: ranking[m])"),
    ("gate: ties broken by name, not choices(me)", PER, "    vals = {m: round(float(row.get(m, 0.0)), 3) for m in moves}",
     "    vals = {m: round(float(row.get(m, 0.0)), 3) for m in sorted(moves)}"),
    # 3. calibration
    ("calibrate: stump takes the largest equally good cut", CAL, "        if right > best:", "        if right >= best:"),
    ("calibrate: air decisions kept in the bands", CAL, "        if pick(e) and not e[\"opp_air\"]:",
     "        if pick(e):"),
    ("calibrate: specials count as pokes", CAL, '    return _band(entries, lambda e: e["action"] in NORMALS,',
     '    return _band(entries, lambda e: e["action"] != "throw",'),
    ("calibrate: walls are the extremes seen, not the pile-ups", CAL,
     '                w[side] = max(near, key=lambda x: v[x])          # the pile-up: the most frequent x near the edge',
     '                w[side] = min(near) if side == "lo" else max(near)'),
    ("calibrate: his guard stance counts as contact", CAL,
     '        if rows[i]["p2_state"] in STUN or rows[i]["p2_life"] < rows[i - 1]["p2_life"]:',
     '        if rows[i]["p2_state"] in STUN + (0x08,) or rows[i]["p2_life"] < rows[i - 1]["p2_life"]:'),
    ("calibrate: the collection's actions joined across directories", CAL,
     "    seen = _new_seen()\n    for d in ram_dirs(root):\n        _add_contacts(seen, read_jsonl([os.path.join(d, \"actions.jsonl\")]), _stream(os.path.join(d, \"ram.jsonl\")))\n    return _k_summary(seen)",
     "    acts = read_jsonl([os.path.join(d, \"actions.jsonl\") for d in ram_dirs(root)])\n    seen = _new_seen()\n    for d in ram_dirs(root):\n        _add_contacts(seen, acts, _stream(os.path.join(d, \"ram.jsonl\")))\n    return _k_summary(seen)"),
    ("calibrate: overlapping windows count frames twice", CAL,
     "            if f > last.get(rec[\"game\"], -1 << 30):", "            if True:"),
    ("calibrate: sources record a path without its content hash", CAL,
     "                    h.update(f.read())", "                    pass"),
    ("calibrate: main ignores --ram-root for k", CAL,
     "    for d in ram_dirs(args.ram_root) if args.ram_root else []:\n        _add_contacts(", "    for d in []:\n        _add_contacts("),
    ("calibrate: k from moves with too few contacts", CAL,
     "for m, v in sorted(ms.items()) if len(v) >= MIN_CONTACTS}", "for m, v in sorted(ms.items())}"),
    # 4. question 8
    ("q8: decisions pooled, opponents not resampled as clusters", PER,
     "    picks = rng.integers(0, len(move), (resamples, len(move)))",
     "    picks = np.zeros((resamples, len(move)), int) + np.arange(len(move))"),
    ("q8: unexplored decisions kept", PER,
     '        if not e.get("explored") or e["game"] % 10 in TEST_INDEX or (e["me"], e["opp"]) == HELDOUT:',
     '        if e["game"] % 10 in TEST_INDEX or (e["me"], e["opp"]) == HELDOUT:'),
    ("q8: margin ignored (>= forward is likely works)", PER,
     "    works = int((diff >= margin).sum())", "    works = int((diff > 0).sum())"),
    ("q8: forward gets a target", PER, "            if m == FORWARD:\n                continue",
     "            if False:\n                continue"),
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
