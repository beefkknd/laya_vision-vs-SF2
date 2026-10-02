"""Seeded faults for the movement pairs (docs/prereg_movement_pairs.md, 2026-10-01): each mutates one line of the
labels, the move list / cycle / execution check, the collector, the builder or the gate, runs the pairs tests, and
restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/pairs_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_pairs_labels.py", "tests/test_pairs_moves.py", "tests/test_pairs_collect.py",
         "tests/test_pairs_data.py", "tests/test_pairs_pressed.py", "tests/test_pairs_episode.py",
         "tests/test_pairs_down.py", "tests/test_pairs_train.py", "tests/test_pairs_train_side.py"]
VS = "sf2/emu/vs.py"
LAB, MOV, COL, IOF, DAT, GAT = ("sf2/data/pairs_labels.py", "sf2/data/pairs_moves.py", "sf2/data/pairs_collect.py",
                                "sf2/data/pairs_collect_io.py", "sf2/data/pairs_data.py", "sf2/data/pairs_gate.py")
TRN = "sf2/data/pairs_train.py"
FAULTS = [
    # 1 the labels
    ("label: facing byte read the wrong way", LAB, 'FACING = {0x40: "right", 0x00: "left"}',
     'FACING = {0x40: "left", 0x00: "right"}'),
    ("label: toward and away swapped", LAB, 'return "toward" if dx * side > 0 else "away"',
     'return "toward" if dx * side < 0 else "away"'),
    ("label: walking needs more than 2 px", LAB, 'return "walk" if abs(dx) >= WALK_PX else "stand"',
     'return "walk" if abs(dx) > WALK_PX else "stand"'),
    ("label: a block react counts as being hit", LAB,
     'if st == GUARD or (st == HIT and _f(r, p, "react") in BLOCK_REACTS):', "if st == GUARD:"),
    ("label: the CPU's specials (0x0A, class 0x08) left as attacks", LAB,
     'if st == SPECIAL or (st == ATTACK and _f(r, p, "mclass") == CPU_SPECIAL_CLASS):', "if st == SPECIAL:"),
    ("label: distance by the other fighter's band", LAB, 'char = CHARACTERS.get(_f(r, p, "char"))',
     'char = CHARACTERS.get(_f(r, 3 - p, "char"))'),
    ("label: no lag (the capture row, not the displayed one)", LAB, "return labels(rows, n - lag, p, bands)",
     "return labels(rows, n, p, bands)"),
    ("label: y off the ground ignored", LAB, 'if st == JUMP or _f(r, p, "y") != GROUND_Y:', "if st == JUMP:"),
    # 2 the moves
    ("moves: walks checked after the jumps' words", MOV, "    if word in WALKS:\n        return KIND_WALK\n", ""),
    ("moves: a jump attack without its button", MOV,
     "steps += (((), JUMP_WAIT), ((button,), PRESS), ((), PRESS))", "steps += (((), JUMP_WAIT),)"),
    ("moves: crouching normals pressed standing", MOV,
     "out.update({CROUCH_NAMES[b]: _crouch(b) for b in BUTTONS})",
     "out.update({CROUCH_NAMES[b]: NORMALS[b] for b in BUTTONS})"),
    ("moves: the cycle never reshuffles", MOV, "            self.rng.shuffle(self.queue)\n", ""),
    ("moves: the cycle drops a word", MOV, "self.queue = list(self.words)", "self.queue = list(self.words[:-1])"),
    ("moves: a jump attack's box counted on the ground", MOV,
     'ok = ok and any(a and r["p1_aid"] for a, r in zip(air, rows))', 'ok = ok and any(r["p1_aid"] for r in rows)'),
    ("moves: jumps aimed by the facing byte", COL,
     'right = (r[me + "x"] < r[him + "x"]) if word in PM.BY_X else r[me + "facing"] == 0x40',
     'right = r[me + "facing"] == 0x40'),
    # 3 the collector
    ("collect: the per-game cap ignored", COL,
     "if self.taken.get((p, key), 0) >= self.per_game or u + LAG >= n:", "if u + LAG >= n:"),
    ("collect: the pair's images without the lag", COL, "k_prev, k_now = u - 4 + LAG, u + LAG",
     "k_prev, k_now = u - 4, u"),
    ("collect: controllers swapped", COL, 'SLOTS = {1: "directed", 2: "cpu"}', 'SLOTS = {1: "cpu", 2: "directed"}'),
    ("collect: only player 1 sampled", COL,
     "        for p in SLOTS:\n            self.pressed[p].append",
     "        for p in (1,):\n            self.pressed[p].append"),
    ("collect: one game over the budget", IOF, "while len(committed(base)) < games:",
     "while len(committed(base)) <= games:"),
    # 4 the builder and the gate
    ("data: no test split", DAT, 'return "test" if h % SPLIT_MOD == TEST_REST else "train"', 'return "train"'),
    ("data: the cap per cell ignored", DAT, "cap, taken = caps[key[0]], []", "cap, taken = 10 ** 6, []"),
    ("gate: independent facing read the wrong way", GAT, 'face = {64: "right", 0: "left"}',
     'face = {64: "left", 0: "right"}'),
    ("gate: over-full cells not counted", GAT, "for k, n in per_cell.items() if n > caps[k[0]]]",
     "for k, n in per_cell.items() if n > 10 ** 6]"),
    ("gate: disk never over", GAT, '"pass": total / 1e9 < max_gb', '"pass": True'),
    ("gate: alignment verdict ignored", GAT,
     "    ok = alignment_verdict(ag, ad, len(rand), len(disc), len(missing), min_disc, min_agree, min_agree_disc)",
     "    ok = True"),
]


# Plan B (2P versus, both ours; the 8 x 20 grid), 2026-10-01
FAULTS += [
    ("B label: a jump attack left as a jump", LAB, 'return "attack" if _f(r, p, "aid") else "jump"', 'return "jump"'),
    ("B label: walk direction dropped from the grid", LAB,
     'return "walk " + direction_ if direction_ in ("toward", "away") else UNKNOWN', 'return "walk"'),
    ("B label: jump split by direction in the grid", LAB, "    return mv\n\n\ndef episode_key",
     "    return mv + (\" \" + direction_ if mv == \"jump\" else \"\")\n\n\ndef episode_key"),
    ("B moves: executed always reads player 1", MOV, "rows = [as_p1(r, p) for r in rows]", "rows = list(rows)"),
    ("B moves: as_p1 does not swap", MOV, 'out["p2_" + k[3:]] = v', 'out[k] = v'),
    ("B collect: player 2 aimed as player 1", COL, "queues[p] = press_frames(chars[p], word, r, p)",
     "queues[p] = press_frames(chars[p], word, r, 1)"),
    ("B collect: player 2's inputs never sent", COL, "rows = run(chunk[1], chunk[2])", "rows = run(chunk[1], [[]] * WAIT)"),
    ("B collect: the next word before the last one is done", COL, "if not queues[p] and can_act(r, p):",
     "if can_act(r, p):"),
    ("B collect: one cycle for both players", COL, "word = cycles[p].next()", "word = cycles[1].next()"),
    ("B collect: can_act reads player 1 only", COL, 'return r["p%d_state" % p] in (0, 2) and r["p%d_y" % p] == GROUND_Y',
     'return r["p1_state"] in (0, 2) and r["p1_y"] == GROUND_Y'),
    ("B collect: the sampler's controllers ignored", COL, "controller=self.controllers[p],", "controller=SLOTS[p],"),
    ("B io: the move log's slot ignored", IOF, "p = m[3] if len(m) > 3 else 1", "p = 1"),
    ("B data: facings pooled in one cell", DAT,
     'return (p["char"], L.movement10(p["movement"], p["direction"]), p["facing"])',
     'return (p["char"], L.movement10(p["movement"], p["direction"]), "any")'),
    ("B data: split by game number, not by match", DAT,
     'h = zlib.crc32(("%s:%d" % (pair_name, game)).encode())', "h = game"),
    ("B gate: independent jump attack read as a jump", GAT,
     'mv = "attack" if r[me + "aid"] != 0 else "jump"', 'mv = "jump"'),
    ("B gate: per-game cap without the facing", GAT,
     'per_game = collections.Counter((r["pair_name"], r["game"], r["slot"], grid(r), r["facing"]) for r in rows)',
     'per_game = collections.Counter((r["pair_name"], r["game"], r["slot"], grid(r)) for r in rows)'),
    ("B vs: no parked plan for the cursor swap", VS, "    plans += [[(2, c), (1, t1), (2, t2)] for c in parks]\n", ""),
]


# the fix after round 1: attack vs special from the move we pressed, confirmed by RAM
FAULTS += [
    ("P label: the pressed move ignored", LAB, '    if pressed in ATTACK_MOVES:\n        return pressed, "pressed"\n', ""),
    ("P label: the pressed move without RAM confirmation", LAB, '    if mv not in ATTACK_MOVES:\n        return mv, "ram"\n',
     ""),
    ("P label: labels() ignores the pressed move", LAB,
     'None = the RAM rule."""\n    mv = movement_pressed(rows, t, p, pressed)[0]',
     'None = the RAM rule."""\n    mv = movement_pressed(rows, t, p, None)[0]'),
    ("P moves: pressed rows off by one", MOV, "for t in range(k0 + 1, k1 + 1):", "for t in range(k0, k1):"),
    ("P moves: a throw counted as no attack", MOV,
     "ATTACK_KINDS = (KIND_NORMAL, KIND_CROUCH, KIND_JUMP_ATTACK, KIND_THROW)",
     "ATTACK_KINDS = (KIND_NORMAL, KIND_CROUCH, KIND_JUMP_ATTACK)"),
    ("P moves: a jump attack counted as no attack", MOV,
     "ATTACK_KINDS = (KIND_NORMAL, KIND_CROUCH, KIND_JUMP_ATTACK, KIND_THROW)",
     "ATTACK_KINDS = (KIND_NORMAL, KIND_CROUCH, KIND_THROW)"),
    ("P moves: the log's slot ignored", MOV, "        return int(m[4])", "        return 1"),
    ("P collect: the live record off by one", COL, "k > self.word[p][1] else None", "k > self.word[p][1] + 1 else None"),
    ("P collect: play_both never announces a word", COL, "                    on_word(p, word, k)",
     "                    pass"),
    ("P collect: the sampler labels without the pressed move", COL,
     "lab = L.labels(self.rows, u, p, self.bands, cls)", "lab = L.labels(self.rows, u, p, self.bands)"),
    ("P data: the builder does not relabel", DAT,
     "pairs, dropped_ep = in_episode_only(relabel(root, pairs, L.poke_bands() if bands is None else bands))",
     "pairs, dropped_ep = in_episode_only([dict(p, in_episode=True) for p in pairs])"),
    ("P data: no per-game cap after the relabel", DAT, "        if len(g) > per_game:", "        if False:"),
    ("P gate: the independent pressed move ignored", GAT, "        mv = pressed  ", "        mv = mv  "),
    ("P gate: the pressed word not compared", GAT, 'r.get("pressed") == word and ', ""),
    ("P gate: the independent interval off by one", GAT, "m[1] < t <= m[2]", "m[1] <= t < m[2]"),
]


# the fix after the label quality check: both frames inside one movement episode
FAULTS += [
    ("E label: t - 4 not checked", LAB, "for u in range(t - gap, t))", "for u in range(t - gap + 1, t))"),
    ("E label: the episode ignores the pressed rule", LAB, "grid_movement(rows, u, p, pressed[u]) == m",
     "grid_movement(rows, u, p, None) == m"),
    ("E label: an unknown run counts as an episode", LAB, "return m != UNKNOWN and all(", "return all("),
    ("E data: the builder keeps pairs outside their episode", DAT, 'kept = [p for p in pairs if p["in_episode"]]',
     "kept = list(pairs)"),
    ("E collect: the sampler samples from the episode's first row", COL, "max(lo, s + L.GAP)", "lo"),
    ("E gate: the episode check never fails", GAT, 'if "unknown" in seq or len(set(seq)) != 1:', "if False:"),
    ("E gate: the episode check starts at t - 3", GAT, "for u in range(t - gap, t + 1)]",
     "for u in range(t - gap + 1, t + 1)]"),
]


# docs/prereg_movement_finetunes.md: down only once back on the ground; second facts; the four fine-tune datasets
FAULTS += [
    ("D label: down in the air kept as down", LAB,
     'return "down" if _f(r, p, "y") == GROUND_Y else "hit"', 'return "down"'),
    ("D label: down on the ground called hit", LAB,
     'return "down" if _f(r, p, "y") == GROUND_Y else "hit"', 'return "hit"'),
    ("D label: down in the air called jump", LAB,
     'return "down" if _f(r, p, "y") == GROUND_Y else "hit"', 'return "down" if _f(r, p, "y") == GROUND_Y else "jump"'),
    ("D gate: independent down in the air kept as down", GAT,
     'mv = "down" if r[me + "y"] == 192 else "hit"', 'mv = "down"'),
    ("D gate: second-fact down check never fails", GAT,
     'return {"pass": down["on_ground"] == down["total"], "down": down, "hit": hit}',
     'return {"pass": True, "down": down, "hit": hit}'),
    ("D gate: down on the ground checks only row t", GAT,
     'all(rows[u]["p%d_y" % p] == 192 for u in range(t - gap, t + 1))', 'rows[t]["p%d_y" % p] == 192'),
    ("D gate: hit health read after the episode start", GAT,
     "< _health(rows[s - 1][hp])", "< _health(rows[s][hp])"),
    ("D gate: hit health of the other fighter", GAT, 'hp = "p%d_hp" % p', 'hp = "p%d_hp" % (3 - p)'),
    ("D gate: second facts not in the gates", GAT, '"second_fact": second_fact_check(d, bands)}',
     '"second_fact": {"pass": True, "down": {"total": 0, "on_ground": 0}, "hit": {"total": 0, "health_lost": 0, "pct": None}}}'),
    ("M data: val from test matches", TRN, 'if D.split_of_game(pair_name, game) != "train":\n        return False',
     'if False:\n        return False'),
    ("M data: no val (val matches stay train)", TRN, 'return "val" if is_val_match(pair_name, game) else "train"',
     'return "train"'),
    ("M data: val by row, not by match", TRN, 'zlib.crc32(("val:%s:%d" % (pair_name, game)).encode())',
     'zlib.crc32(("val:%s:%d" % (pair_name, game + 1)).encode())'),
    ("M data: answer dirs pooled", TRN, 'return rec["answer"] if DATASETS[dataset][1] else dataset', "return dataset"),
    ("M data: label off the criteria order", TRN, '"label": list(crit).index(r["answer"])',
     '"label": sorted(crit).index(r["answer"])'),
    ("M data: the question names the other fighter", TRN, 'asked = question(q, r["char"]) if',
     'asked = question(q, r["opp"]) if'),
    ("M data: no frames link", TRN, "        os.symlink(frames, os.path.join(base, \"frames\"))\n", ""),
    ("J data: a jump over the other fighter (direction unknown) refused", DAT,
     ' and not (\n                k == "direction" and p.get(k) == L.UNKNOWN and p.get("movement") == "jump")', ""),
    ("J data: any movement may lack its direction", DAT, 'and p.get("movement") == "jump"):', "):"),
    ("M check: the split never checked", TRN, 'if r["split"] != f or f != want:', "if False:"),
    # round 2: ask by screen side
    ("S side: the larger x is left", TRN, 'return "left" if me < other else "right"',
     'return "left" if me > other else "right"'),
    ("S side: read at t - 4", TRN, 'return side_of(ram[r["t"]], r["slot"])', 'return side_of(ram[r["t"] - 4], r["slot"])'),
    ("S side: read at the capture row t + 1 (no lag)", TRN, 'return side_of(ram[r["t"]], r["slot"])',
     'return side_of(ram[r["t"] + 1], r["slot"])'),
    ("S side: equal x not dropped", TRN, "    if me == other:\n        return None", "    if False:\n        return None"),
    ("S side: the other slot's x as mine", TRN, 'me, other = ram_row["p%d_x" % slot], ram_row["p%d_x" % (3 - slot)]',
     'me, other = ram_row["p%d_x" % (3 - slot)], ram_row["p%d_x" % slot]'),
    ("S text: movement question reworded", TRN, '"What is the fighter on the %s doing?"', '"What is the %s fighter doing?"'),
    ("S text: distance question loses 'the other fighter'", TRN,
     '"Is the fighter on the %s close to or far from the other fighter?"', '"Is the fighter on the %s close or far?"'),
    ("S data: the side build still asks by name", TRN,
     'asked = question(q, r["char"]) if side is None else question_side(q, side)', 'asked = question(q, r["char"])'),
    ("S data: side not recorded on the row", TRN, 'extra = {} if side is None else {"side": side}', "extra = {}"),
    ("S data: dropped rows not counted", TRN, '"total": len(dropped)', '"total": 0'),
    ("S check: the side never checked", TRN,
     'if r.get("side") != want_side or text != SIDE_INSTRUCTIONS[q] % want_side:', "if False:"),
    ("S check: the independent side ignores the slot", TRN,
     'mine, theirs = (xs[0], xs[1]) if r["slot"] == 1 else (xs[1], xs[0])', "mine, theirs = xs[0], xs[1]"),
    ("S check: equal-x rows may stay in the data", TRN,
     'src_ids = {i for i in src_ids if sides[i] is not None}', "src_ids = src_ids"),
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
