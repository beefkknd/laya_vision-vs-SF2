# Training scorecard

Use one fixed headless evaluation after each completed model:

```sh
python scripts/parallel.py --workers 4 play_student --model runs/<run>/best \
  --name <run>_gate --matches 12 --seed 4242 \
  --savestate states/chunli_vs_dhalsim.state --me chunli --opp dhalsim
python scripts/gate.py rollouts/<run>_gate
```

The primary measure is **damage score**: average opponent health removed each
round, as a percentage of the 176-point full bar. It is useful while round wins
are still rare. A higher score is better.

Keep `net_damage_per_round` alongside it: damage dealt minus damage taken. It
prevents a score increase caused only by reckless trading. Treat held-out teacher
agreement as a training diagnostic, not a performance score.

| Run | Damage score | Net damage / round | Notes |
| --- | ---: | ---: | --- |
| chunli_r0 | 33.5 | -117.0 | Two-round exploratory gate; not protocol-comparable. |
| chunli_r1 | 25.0 | -132.0 | Two-round exploratory gate; not protocol-comparable. |
| v2_random | 54.6 | -79.9 | v2 baseline: random policy, 14 rounds. |
| v2_teacher | 21.0 | -139.1 | v2 baseline: scripted teacher, eps 0, 12 rounds. Worse than random; see TEACHER.md. |

Old data (the chunli_r* datasets and rollouts) was deleted on 2026-09-25; the v2 loop starts over. The earlier
val accuracies were optimistic: no dataset had val rows, so val was random frames whose neighbours were in training.
Val now uses held-out rounds or eval-only sets (`--val-data`).
