"""Seeded faults for round 3 of docs/prereg_movement_finetunes.md (act + fireball): each mutates one line of the act
builder, the projectile trigger, the collector's shot files, the fireball builder / its independent check, the gate,
or the scoring's collapse, runs the round-3 tests, and restores the file. Every fault must turn the tests red; exit 1
if one stays green.

    python tests/faults/mv3_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_mv3_act.py", "tests/test_pairs_shots.py", "tests/test_mv3_fireball.py",
         "tests/test_mv3_fireball_gate.py", "tests/test_mv_eval.py"]
ACT, SH, IOF, FB, GT, EV = ("sf2/data/mv3_act.py", "sf2/data/pairs_shots.py", "sf2/data/pairs_collect_io.py",
                            "sf2/data/mv3_fireball.py", "sf2/data/mv3_fireball_gate.py", "sf2/data/mv_eval.py")
FAULTS = [
    # act
    ("act: special mapped to attack", ACT, '_ACT_OF = {"attack": "attack", "special": "special"}',
     '_ACT_OF = {"attack": "attack", "special": "attack"}'),
    ("act: down counted as an attack", ACT, '_ACT_OF = {"attack": "attack", "special": "special"}',
     '_ACT_OF = {"attack": "attack", "special": "special", "down": "attack"}'),
    ("act: label not the answer's index", ACT, "label=ACT_ANSWERS.index(act)", "label=0"),
    ("act: answer dirs pooled", ACT, 'by_dir[r["answer"]][r["split"]].append(r)', 'by_dir["moving"][r["split"]].append(r)'),
    ("act: question loses the side", ACT, '"instructions": ACT_TEXT % side', '"instructions": "What is the fighter doing?"'),
    ("act check: answer never checked", ACT, 'if r["answer"] != act or d != act:', "if False:"),
    ("act check: side never checked", ACT,
     'if r.get("side") != side or text != "What is the fighter on the %s doing?" % side:', "if False:"),
    # the trigger
    ("shots: the blink bit ignored", SH,
     'return bool(row["shot%d" % slot]) and not row["shot%d_hide" % slot] & 1', 'return bool(row["shot%d" % slot])'),
    ("shots: spawn word of the other player", SH, "spawn_word=words[str(s)]", "spawn_word=words[str(3 - s)]"),
    ("shots: thrower = the other slot's character", SH, "thrower=self.chars[s], other=",
     "thrower=self.chars[3 - s], other="),
    ("shots: images without the lag", SH, "k_prev, k_now = u - 4 + LAG, u + LAG", "k_prev, k_now = u - 4, u"),
    ("shots: per-game cap ignored", SH, "if self.shot_taken.get((s, stage), 0) >= self.shot_per_game:", "if False:"),
    ("shots: side from the other fighter", SH, 'me, other = row["p%d_x" % slot], row["p%d_x" % (3 - slot)]',
     'other, me = row["p%d_x" % slot], row["p%d_x" % (3 - slot)]'),
    ("shots: a flight running past the end never closed", SH,
     "self._close_flight(s, self.run_start[s], len(self.rows) - 1, cut_end=True)", "pass"),
    ("shots: stages in halves", SH, "if 3 * (u - a) // length == i and visible", "if 2 * (u - a) // length == i and visible"),
    ("shots: player 2's blink byte read from player 1's slot", SH, 'Var("shot2_hide", 0x108A, 1, False)',
     'Var("shot2_hide", 0x103A, 1, False)'),
    ("shots: yoga flame counted as a projectile", SH, '"dhalsim": ("yoga_fire",)}', '"dhalsim": ("yoga_fire", "yoga_flame")}'),
    ("shots: uncommitted samples kept", SH, 'kept = [r for r in rows if r.get("game") in done]', "kept = list(rows)"),
    ("collect: shots.jsonl never written", IOF, 'MIO._append(os.path.join(base, "shots.jsonl"), sampler.shots)', "pass"),
    # the fireball builder
    ("fire: the other fighter's side", FB, "kept.append(dict(r, side=S.side_of(ram[t], s)))",
     "kept.append(dict(r, side=S.side_of(ram[t], 3 - s)))"),
    ("fire: two projectiles kept", FB, 'elif ram[t]["shot%d" % (3 - s)]:', "elif False:"),
    ("fire: a hidden projectile kept", FB, "elif not S.visible(ram[t], s):", "elif False:"),
    ("fire: none checked at t only", FB, 'for u in range(r["t"] - NONE_WINDOW, r["t"] + 1)):', 'for u in range(r["t"], r["t"] + 1)):'),
    ("fire: none count = the smaller answer", FB,
     'max(sum(r["split"] == f and r["answer"] == a for r in fire) for a in ("left", "right"))',
     'min(sum(r["split"] == f and r["answer"] == a for r in fire) for a in ("left", "right"))'),
    ("fire: caps ignored", FB, "if q and len(taken) < caps[key[0]]:", "if q:"),
    ("fire: split by row, not match", FB, '"split": T.split3(r["pair_name"], r["game"])',
     '"split": T.split3(r["pair_name"], r["t"])'),
    ("fire check: stage never checked", FB,
     'if r.get("flight_stage") != ("start", "middle", "end")[min(2, 3 * (t - a) // (e - a + 1))]:', "if False:"),
    ("fire check: spawn word never checked", FB, "if _pressed(moves, a, p) != _THROWN.get(char):", "if False:"),
    ("fire check: hidden projectile accepted", FB, 'ram[t]["shot%d_hide" % p] % 2 == 1 or ', ""),
    ("fire check: none window never checked", FB,
     'if any(ram[u]["shot1"] or ram[u]["shot2"] for u in range(r["t"] - 8, r["t"] + 1)):', "if False:"),
    ("fire check: images never checked", FB, 'if r["images"] != imgs or not', "if not"),
    # the gate
    ("gate: blue counted in the HUD", GT, '.astype(int)[HUD_ROWS:]', ".astype(int)[0:]"),
    ("gate: drawn passes regardless", GT,
     'res["pass"] = bool(had) and bool(none) and hit >= need * len(had) and clear >= need * len(none)',
     'res["pass"] = True'),
    ("gate: ownership from the other character", GT, 'own = S.is_projectile(chars[s - 1], fl["spawn_words"][str(s)])',
     'own = S.is_projectile(chars[2 - s], fl["spawn_words"][str(s)])'),
    ("gate: labels gate ignores problems", GT, '"labels": {"pass": not bad,', '"labels": {"pass": True,'),
    ("gate: clock-blink rows kept in the alignment", GT,
     'if min(ram[r["k_prev"] - 1]["timer"], ram[r["k_now"] - 1]["timer"]) <= CLOCK_BLINK:', "if False:"),
    # the scoring
    ("eval: collapse leaves the truth", EV, 'dict(r, answer=mapping[r["answer"]], movement10=r["answer"])',
     'dict(r, movement10=r["answer"])'),
    ("eval: stage breakdown by side", EV, 'breakdown(rows, preds, "flight_stage", answers)',
     'breakdown(rows, preds, "side", answers)'),
]

# round 4 step 1: the fireball caps as flags
FAULTS += [
    ("R4 build: the fireball cap flags ignored", "scripts/build_mv3_data.py",
     "meta = F.build_fireball(args.src, args.shots, args.out or F.OUT, caps=caps, seed=args.seed)",
     "meta = F.build_fireball(args.src, args.shots, args.out or F.OUT, seed=args.seed)"),
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
