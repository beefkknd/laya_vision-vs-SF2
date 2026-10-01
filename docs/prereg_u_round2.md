# Pre-registration: the U eye, second training round (2026-10-01, before any code or training)

Owner: "hook this checkpoint and play a couple of games... maybe another round of training on top of this checkpoint is
needed. Health bar is not that important." -> after the exploratory look (U0 ~ A0, far below T0; docs/
prereg_u_perception.md), "ok" to a second round.

## What changes (and only this)
1. **Start from runs/u_eye/best** (round 1's checkpoint, merged), not from BASE: a new `--init` in scripts/train.py;
   fresh LoRA adapters (rank 16, alpha 32) on top, as round 1.
2. **Questions:** round 1's dataset (test_data_u, unchanged labels, thresholds v2, Q8 targets v1) WITHOUT the two
   health-bar questions and the trend question (owner: bars not important; trend near chance, 0.50 of 3). Remaining:
   range band, his phase, his air state, projectile, can I act, can he act, corner, and Q8 per move (22 rows per
   decision). Same split, same validation games (capped), Chun-Li vs Guile held out.
3. **Budget: 90,000 steps** (2.5x round 1's 36,000; ~11 h at the measured speed; --max-minutes 780 as a guard).
   Everything else as round 1: batch 8, lr head 1e-4, backbone 2e-4, seed 0, select on validation NLL, eval every
   1,000, patience 5, equal-share sampling per character.

## Gates (as round 1, decided before training)
- Gate 2: the table's best move in the eye's top 3 >= 0.90 on test_real and on held-out Guile (decisions where the table's
  best is not walking in). Round 1: 0.742 / 0.710.
- Reported: per-question accuracy vs round 1 (range 0.697, phase 0.548, air 0.852, corner 0.837, can he act 0.823,
  projectile 0.961, can I act 0.968, Q8 word 0.790), Q8 softness.
- In play (as the owner's exploratory look): U0 / U1 on seeds 73001-73002, paired with the locked A0 / T0 (U1 on the new
  Qwen server: indicative only).

## Load
Training alone on the machine (~11 h), memory watchdog; nothing else heavy meanwhile.
