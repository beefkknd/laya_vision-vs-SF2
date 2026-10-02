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
         "tests/test_pairs_data.py"]
VS = "sf2/emu/vs.py"
LAB, MOV, COL, IOF, DAT, GAT = ("sf2/data/pairs_labels.py", "sf2/data/pairs_moves.py", "sf2/data/pairs_collect.py",
                                "sf2/data/pairs_collect_io.py", "sf2/data/pairs_data.py", "sf2/data/pairs_gate.py")
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
     "        for p in SLOTS:\n            key = L.episode_key", "        for p in (1,):\n            key = L.episode_key"),
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
