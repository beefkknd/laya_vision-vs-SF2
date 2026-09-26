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
