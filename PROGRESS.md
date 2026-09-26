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

Everything measured before 2026-09-25 (chunli_r*, v2_*) ran on a broken harness: wrong x addresses (distance
and facing), knockdowns counted as jumps, time-overs scored as draws, and parallel workers replaying identical
matches. Those numbers are invalid and their data was deleted; baselines are re-measured under this protocol.
