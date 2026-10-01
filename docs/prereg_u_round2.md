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

## Amendment before training (2026-10-01): train on threebody (owner)
Owner: train on the Windows box "threebody" (RTX 4090 24 GB) in a separate folder C:\work\laya_finetune; "do not install
new CUDA"; standing permission to stop Qwen during laya training and restart it (C:\work\qwen3.8-27b\NOTES.md).
- Environment: Python 3.11.7 venv; torch 2.5.1+cu121 BORROWED read-only from an existing venv (no CUDA installed);
  transformers 5.17.0 / tokenizers 0.23.2 / huggingface-hub 1.33.0 / safetensors 0.8.0 as on the Mac (pip, no torch, no
  CUDA packages); laya 0.2.0.dev0 copied from the Mac. Code: git archive of dc56d03. Data: test_data_u2 copied
  (row and frame counts match); runs/u_eye/best copied (8 files, sha256 match).
- Parity (the same checkpoint, the same 300 test rows): same answer 297/300 (99.0%); max |prob diff| median 0.0025,
  p95 0.011, max 0.15 (near-ties) - device float differences, accepted.
- Mac (MPS, torch 2.14) vs threebody (CUDA, torch 2.5.1): the run is not bit-identical to what the Mac would produce;
  the gates are judged on the Mac after copying the checkpoint back.
- Training settings unchanged: 90,000 steps = --epochs 0.9031 on 797,253 training rows.

## Result (2026-10-01, threebody): no gain - early stop at step 5,000, best = step 0 (round 1 itself)
- Speed on the RTX 4090: ~280 steps/min (2x the Mac), 29% of time waiting on image loading; the first launch was killed
  by the ssh session (ledger #30) and relaunched via WMI. Qwen stopped for the run and restarted after (/health ok).
- Validation NLL: step 0 (round 1's weights) 0.548; then 0.593, 0.571, 0.551, 0.564, 0.575 -> early stop (patience 5).
  Q8's soft cross-entropy stayed at its prior throughout (1.002-1.052 vs 1.032).
- Why it cannot move: Q8 (one row per move) is 68% of the rows (720 of 1,054 per validation file) and does not improve
  from frames, so it dominates both the gradient and the pooled-NLL stop rule; perception (32% of rows) has little room
  to show a gain in the pooled number. Combined with round 1's per-question results (Q8 ~ always "likely fails"), the
  eye does not learn move value from frames at this data size, while it does learn several perceptions.
- runs/u_eye2/best = round 1's weights (no new checkpoint worth evaluating). Gates not re-run.
