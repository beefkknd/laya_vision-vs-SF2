# Journal: harness to first trained student (2026-09-25 to 2026-09-26)

Two days of work on the Mac mini (M4 Pro, 64 GB). It took the project from a broken harness to a laya-vision student
that plays Chun-Li vs Dhalsim within noise of its teacher. Work stopped here on purpose; the next step is arcade mode,
probably on the laptop. PROGRESS.md holds the gate protocol and every gated number; TEACHER.md holds the teacher
experiments; AGENTS.md is the runbook.

## Where things stand

| Policy | Net damage / round (± SE) | Rounds won | Notes |
| --- | ---: | ---: | --- |
| random (14 actions) | -56.0 ± 7.4 | 3 / 42 | |
| scripted teacher | +105.5 ± 6.8 | 40 / 40 | `sf2/teacher.py`, reads RAM |
| student r0 | +67.7 ± 12.1 | 39 / 49 | imitation only |
| student r1 | +80.6 ± 9.6 | 39 / 46 | DAgger round 1 |
| **student r2** | **+93.1 ± 8.3** | **40 / 41** | DAgger round 2; within noise of the teacher |

The gate: 20 paired matches, `--workers 4 --seed 4242`, Chun-Li vs Dhalsim from `states/chunli_vs_dhalsim.state`.
The student sees two screenshots and a short RAM-derived text note, and picks one of 14 actions.

## What happened, in order

1. **The v2 loop ran on a broken harness.** The first verification pass (three subagents: emulator, data, code)
   found four bugs that corrupted every label and gate number:
   - x was read from the camera scroll and a flag byte, so distance was meaningless;
   - facing inverted mid-round, so `forward` walked away;
   - time-overs were scored as draws with fake damage;
   - parallel workers replayed identical matches.

   The v2 data was deleted, and the rule became: verify and fix the harness first, with a test that fails before
   every fix.
2. **Harness, first pass (branch `harness-verify`, PR #2).**
   - Tests replay recorded ROM traces (`tests/trace_mesen.py` and `scripts/record_trace.py`, fixtures in
     `tests/fixtures`).
   - A headless ROM acceptance suite (`tests/test_rom_harness.py`, 28 checks by the end) writes a stamp; collection
     and play refuse to run without it.
   - Fixes: world x at 0x0D18 / 0x0F18, ground level, time-over, round-2 intro, round clock 0x1AC8, distinct starts
     per worker, action state 0x0C03 / 0x0E03, a block check, the stamp gate.
   - New savestate: arcade round 1, clock 99, first controllable frame.
   - Gate protocol written down; CLOSE/MID set to 80/120 from measured hit rates.
3. **Speed (stage 1).**
   - bf16 on MPS rejected: 1.1x faster in play, only 69% action agreement, no gain in training.
   - 256 px adopted: 1.34x faster training, 1.9x faster play, a slightly worse fit per example.
   - Play reaches about 20 decisions/s with 4 workers.
4. **Context (stage 2) and a second harness audit.** Fixes:
   - Headless screenshots lagged the RAM, because Mesen skipped rendering frames.
   - Rounds are judged by the ROM's result byte 0x1ACF.
   - A match is at most 4 rounds.
   - Block stun is told apart from hit stun by the reaction byte +0x4A.
   - The true life byte +0x35 books damage on the decision where the hit lands.
   - A `controllable` flag marks frames where the stick does nothing (hit, thrown, KO); they stay out of training data.
   - Stage walls 53/459; Yoga Fire at 0x1050 / 0x1057.

   The note became:
   `me=chunli stand hp=100 opp=dhalsim attack hp=100 dist=mid facing=right corner=none time=early last=forward fireball=none`
5. **Basic moves and teacher (stage 3).**
   - 11 actions: hadouken and shoryuken dropped (World Warrior Chun-Li has neither); `jump_forward` added. Every
     action is checked on the ROM on both sides of the screen and at both walls.
   - Teacher rules were added one at a time and kept only if the gate improved by at least 2 combined SE.
   - Kept: jump in from mid range, kick at the top of the arc (hit rate 11% on the way up, 69% at the apex),
     crouch-guard when he attacks.
   - The teacher went from -84.5 to +62.0.
6. **Fancy moves (stage 4).**
   - `throw` (toward + fierce, works within 42 px) together with walking into range lifted the teacher to +105.5,
     40/40.
   - `sweep` and `lightning_legs` were added and checked on the ROM, but the teacher doesn't use them (their rules
     didn't pass the gate).
   - Headless Mesen now exits when its Python parent dies.
7. **First training loop (stage 5, branch `loop`).**
   - Seed data is the teacher's top choice plus 10% random (`collect_teacher --greedy --eps 0.1`). Sampling the soft
     labels had been playing at about +12, not +100.
   - 37,879 rows, LoRA r16 at 256 px, about 14.6 ex/s. r0 took 135 min; each DAgger round about 100 min to train
     and 12-15 min to gate.
   - DAgger cured r0's habit of repeating its last action.
   - The held-out student frames rank checkpoints in the gate's order; eval5 does not.
8. **Arcade exploration (stage 6, stopped early, nothing committed).**
   - Character id at 0x0CD1 (her) / 0x0ED1 (him): 0 Ryu, 1 Honda, 2 Blanka, 3 Guile, 4 Ken, 5 Chun-Li, 6 Zangief,
     7 Dhalsim, 10 Balrog, 11 Vega.
   - Stage at 0x1A5A; 12 is the car bonus stage, 13 the bricks.
   - Order from the Dhalsim savestate: Dhalsim, Ryu, Zangief, Ken, [bricks], Honda, Guile, Blanka, Balrog, [car],
     Vega.
   - Both exploration runs had the teacher beat 9 opponents in a row, mostly 2-0; one of the two runs then lost to Vega.
   - Start states for 8 opponents are saved as `states/arcade_chunli_vs_<opp>.state` (not in git).
   - Known harness gaps: bonus stages refill the bars and would be counted as rounds; after a loss, the attract-mode
     demo fights would be recorded. The first controllable frame of a new fight is 103-113 frames after the refill;
     for rounds 2+ it is 182.

## Lessons

- **The harness decides everything.** Every early "result" was an artefact: a teacher worse than random, a model at
  82% accuracy that played badly. Trace tests plus a ROM acceptance run were cheap next to that.
- **One gate, and respect its noise.** Per-round sd is about 30 damage points, so it takes 20 paired matches and
  2 combined SE. Frame accuracy did not predict play strength.
- **How the teacher plays when collecting matters as much as its rules.** Sampling its soft distribution played far
  worse than its top choice.
- **Rule #1:** an 80%-good decision is a good decision. The 256 vs 512 comparison and the bf16 work were stopped or
  rejected once the answer was clear enough.

## Open

- **Over-fitting to one opponent.** Everything so far is Dhalsim from one savestate. Arcade mode is the fix: gate r2
  per opponent to measure how narrow it is, then collect and train across opponents while keeping the Dhalsim data.
- **The gate is near its ceiling on Dhalsim** (40/41 rounds). To resolve smaller differences, use more matches or
  harder opponents.
- **Some leakage in the DAgger gates.** The DAgger rows come from the same 20 gate openings; an offset for
  collection starts would fix it.
- **Real-time play isn't there yet:** about 72 ms per decision against a 67 ms budget at 256 px on the M4 Pro.
  Lockstep keeps training exact.
- **Harness details:** Dhalsim's dizzy is unobserved; Lightning Legs still fire by accident about 7 times per 20
  matches; the attack kind (limb vs slide) can't be read.

## Resume

**On a new machine** (e.g. the 40-core laptop):
1. Copy `states/*.state` and the ROM (SHA1 7DDCB96E...); install Mesen 2.
2. Follow README step 3 for the venv.
3. Run `SF2_ROM=... pytest -q tests/test_rom_harness.py` to get the stamp.
4. Re-measure training ex/s and play decisions/s before choosing worker counts.

**DAgger round 3 on Dhalsim** (about 2 h here), from `runs/r2/best`:

```sh
python scripts/relabel.py --rollout rollouts/r2_gate --name dagger5_r3 --val-every 5
python scripts/train.py --data data/seed5_g10 --data data/dagger5_r1 --data data/dagger5_r1_hot \
  --data data/dagger5_r2 --data data/dagger5_r2_hot --data data/dagger5_r3 --data data/dagger5_r3_hot \
  --mix dagger5_r1_hot=0.5 --mix dagger5_r2_hot=0.5 --mix dagger5_r3_hot=0.5 \
  --val-data data/eval5 --init runs/r2/best --out runs/r3 --epochs 1 --eval-every 1000 --patience 3 --val-limit 8000
python scripts/parallel.py --workers 4 play_student --model runs/r3/best --name r3_gate --matches 20 --seed 4242 \
  --savestate states/chunli_vs_dhalsim.state --me chunli --opp dhalsim
python scripts/gate.py rollouts/r2_gate rollouts/r3_gate
```

**Arcade mode (stage 6), in order:**
1. `my_char` / `opp_char` / `stage` in the RAM map, with the note's `opp=` read from RAM.
2. Arcade flow in the env: skip bonus stages, end on a loss.
3. Per-opponent checks: walls, projectile slot, struct layout.
4. Per-opponent teacher gates and an arcade-reach metric.
5. Gate r2 per opponent, then DAgger across opponents.

Branches: `harness-verify` (PR #2), `speed` (PR #3, stacked), `loop` (this journal, stacked on `speed`). The
`arcade` branch has no commits.
