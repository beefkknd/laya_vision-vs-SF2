# Pre-registration: does value ranking help at all? A no-model lookup test (2026-09-30, before any game)

Dr Fable's review (docs/reviews/2026-09-30_dr_fable_lv_value.md, plan B(b)); owner approved "3, 2, 1" (stop the eval,
run this test, then a second training run). A diagnostic, not a product: it asks whether ranking moves by value, from
the general note fields only, beats today's laya-vision in play, before more GPU is spent.

## The ranking
- `lessons/value_oracle_v1.json` (sha256 30662489177d15bd...8a7f), built by `sf2.data.value_oracle.from_logs` from
  rollouts/lv_value: mean net (dealt - taken until the next decision) per (character, range, opp_attacking,
  opp_airborne, move), EXPLORED decisions only (random move: unbiased timing), training games only (game % 10 not in
  2,5,8), Chun-Li vs Guile excluded; shrunk toward 0 by n / (n + 10). 96 cells. No opponent names.
- System 1 plays the table's best move over its choices (forward included), no model, no advice, no Qwen
  (`play_system1.py --model none --oracle ...`).

## The test (the in-play protocol of docs/prereg_lv_value.md)
- Arms: `oracle` (the table) vs `old` (runs/all8/best, threshold rule). Chun-Li vs ryu, ken, honda, zangief, guile,
  dhalsim x seeds 95001-95008 x 15 rounds; paired by seed; `scripts/value_inplay.py report --new oracle --old old`.
- Unit: the seed per opponent; pooled with the opponent as the unit. Guile is the held-out opponent (never in the table).
- The `old` arm is reused, unchanged, as the control of the second training run's in-play test.

## Drop rule (Dr Fable)
Value ranking at this window is dropped (no second training run) if BOTH: the table does not beat runs/all8 pooled
(CI covers 0 or below), AND the table's held-out quintile spread (Chun-Li's explored test-game decisions and her
explored Guile decisions: mean real net of the top fifth of predicted values minus the bottom fifth) is < 10 hp.
Otherwise the second training run goes ahead as reviewed.

## Load
Arms run one after the other (the table arm is light: emulators only); memory checked before each.

## Result (2026-09-30; 96 runs, all finished)

hp per round, lookup table minus runs/all8, paired by seed (8 per opponent), the seed as the unit, 95%:

| Ryu | Ken | Honda | Zangief | Guile (held out) | Dhalsim | pooled (opponents) |
|---|---|---|---|---|---|---|
| +25.4 [-4.9, +53.7] | **+118.5** | **+20.9** | **+56.4** | **+146.8** | **+120.6** | **+81.4 [+42.7, +121.9] helps** |

- Rounds won of 720: table 369, runs/all8 70. Damage per round: dealt 128 vs 85, taken 126 vs 164.
- Throws per close-range decision: table 0.28-0.89 by opponent, runs/all8 0.00-0.04.
- Held-out quintile spread (offline, explored decisions): 9.6 hp on test games, 11.2 on held-out Guile, monotone.
- **Drop rule not triggered**: value ranking helps, strongly, on every opponent but Ryu (not shown), including the
  held-out one. The second training run goes ahead (docs/prereg_lv_value_run2.md).
- Boundary reading: a 96-cell table over the note's general fields, with no model, beats laya-vision's P(hit) ranking
  by 81 hp/round. The defect was the ranking target, as the component study said; the frames must now add timing on
  top of this table to be worth a model (the table is run 2's reference).
