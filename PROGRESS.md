# Training scorecard

## Gate protocol (decided 2026-09-25)

Every model, baseline or teacher change is judged by the same paired evaluation:

```sh
python scripts/parallel.py --workers 4 play_student --model runs/<run>/best \
  --name <run>_gate --matches 20 --seed 4242 \
  --savestate states/chunli_vs_dhalsim.state --me chunli --opp dhalsim
python scripts/gate.py rollouts/<baseline>_gate rollouts/<run>_gate
```

- **20 matches per arm** (at least 40 rounds). Per-round damage varies a lot (sd about 30 points), so smaller
  gates cannot tell models apart.
- **Paired starts**: always `--workers 4 --seed 4242`. Worker i, match e starts after i*30 + e + 1 idle frames,
  so every arm fights the same 20 openings. `distinct_matches` must equal `matches`.
- **Primary number: `net_damage_per_round`** (dealt minus taken). Also read `round_win_rate` and `damage_score`.
- **Better** means net damage per round higher by at least 2 x sqrt(se_a^2 + se_b^2), using each arm's
  `net_damage_se`. Anything smaller is noise.
- Only on a verified harness: `SF2_ROM=... pytest -q tests/test_rom_harness.py` must pass (it writes the stamp the
  collection scripts check).

Treat held-out teacher agreement and frame accuracy as training diagnostics, not performance scores.

| Run | Net damage / round | SE | Round wins | Notes |
| --- | ---: | ---: | ---: | --- |
| base2_idle | -170.7 | 1.7 | 0 / 40 | 2026-09-26 baseline. |
| base3_random | -42.5 | 9.4 | 14 / 50 | 2026-09-26, basic 11-action set (`jump_forward` added, specials dropped). Dealt 121 per round. |
| base3_teacher | -84.5 | 6.8 | 1 / 41 | 2026-09-26, basic set; same argmax as base2_teacher, so the same fights. 42.0 below random = 3.6 SE. |
| t3_jumpin_base | -2.0 | 9.4 | 21 / 52 | 2026-09-26, the Stage 3 teacher: base3_teacher + jump in from mid range (TEACHER.md, Stage 3). +40.5 over random = 3.0 SE: better. |
| t3b_apex_block | +62.0 | 10.8 | 36 / 46 | 2026-09-26, the Stage 3b teacher: t3_jumpin_base + kick at the top of the jump + crouch-guard his attacks (TEACHER.md, Stage 3b). +104.5 over random = 7.3 SE. Dealt 165, taken 103 per round. |
| t3b_eps25 | -23.0 | 8.6 | 12 / 50 | 2026-09-26, the same teacher as the seed collection plays it (`collect_teacher --eps 0.25`: 25% random, else sampled from the soft target; 5 matches per worker). Agrees with its own top choice 44% of the time. Data kept in `data/t3b_eps25` (18,578 rows, no val). |
| t4_random | -56.0 | 7.4 | 3 / 42 | 2026-09-26, the 14-action set (basic + `throw sweep lightning_legs`), `play_teacher --policy random`. Dealt 114, taken 170 per round. Not comparable with base3_random (-42.5, a different action set); the difference, -13.5, is within noise (bar 24.0). |
| t4_final | **+105.5** | 6.8 | 40 / 40 | 2026-09-26, the teacher now: t3b_apex_block + walk into throw range and throw + close anti-airs with fierce (TEACHER.md, Stage 4). +161.6 over t4_random = 16.1 SE. Dealt 176, taken 70 per round. |
| r0_256_gate | +67.7 | 12.1 | 39 / 49 | 2026-09-26, Stage 5 first student: LoRA from the base checkpoint at 256 px on seed5_g10 (37,879 rows of `collect_teacher --greedy --eps 0.1`), 2 epochs, best val acc 0.938 (eval5 + seed val). Teacher agreement 0.823. -37.8 vs t4_final (bar 27.8): worse than the teacher, +123.7 over t4_random. 94% of its disagreements repeat its own last action. |
| r1_gate | +80.6 | 9.6 | 39 / 46 | 2026-09-26, DAgger round 1: r0_256 + 1 epoch on seed5_g10 + dagger5_r1 (r0_256_gate relabelled, 12,230 rows) + its hot set at half weight. Teacher agreement 0.919. +12.9 vs r0_256 (bar 30.9): within noise; -24.9 vs t4_final (bar 23.5). Repeats its last action 0.826 (teacher 0.832). |

Seed collection noise (Stage 5, 20 paired matches each, as `collect_teacher` plays): sampling from the teacher's
soft target is what made the old collection weak (eps 0 / 0.05 / 0.1 / 0.25: +11.7 / -4.1 / +16.3 / +19.2 net, own
choice 55-42%); `--greedy` (top choice, eps random) plays +102.2 / +91.4 / +57.5 at eps 0.05 / 0.1 / 0.25 (own choice
95 / 91 / 77%). Stage 5 seed data uses `--greedy --eps 0.1`.

The t4 rows use the 14-action set (Stage 4); the teacher never picks `sweep` or `lightning_legs` (TEACHER.md).
The base3 rows use the basic action set (`idle forward back jump jump_forward crouch lp hp lk hk block`) and
supersede the 12-action base2_random (-89.7 ± 8.1, 2 / 42) and base2_teacher (-84.5 ± 6.8, 1 / 41). Idle does not
depend on the action set; base2_idle stands. The 2026-09-26 base2 rows supersede base_idle (-173.0), base_random (-98.0), base_teacher (-78.5) and base_teacher_c80
(-75.2) from 2026-09-25. Since then the harness renders every frame for screenshots, starts rounds 2+ when control
returns, ends rounds on the ROM's round result (a KO books the loser's whole bar), books a hit's damage on the
decision it lands in, tells block stun and dizzies from hit stun, and ends a match after its 4th round.
All three arms had 20 distinct matches. A lost round can show 161 or 140 taken: with a worker's start jitter (up to
95 idle frames) Dhalsim's first one or two hits can land before the first decision. A time-over loss shows the
life she still had (random: 175, 172).

Everything measured before 2026-09-25 (chunli_r*, v2_*) ran on a broken harness: wrong x addresses (distance
and facing), knockdowns counted as jumps, time-overs scored as draws, and parallel workers replaying identical
matches. Those numbers are invalid and their data was deleted; baselines are re-measured under this protocol.
