# Pre-registration: four separate fine-tunes on the 2P movement data (2026-10-01, before any training)

Owner: "Down is after landed yes. Three body is sleep. Train on Mac m3 ultra." Data: docs/prereg_movement_pairs.md.

## Data change first (then the dataset gate again)
- "down" = only once the knocked-down fighter is back on the ground (lying, getting up). The airborne part of a
  knockdown (flying / falling after the hit) becomes "hit". Rebuild; top up rounds until every (character, movement,
  facing) cell is full again (hard limit 10 games per pair); gate (labels vs RAM 100%, episode, lag 1, caps, disk) and
  the second-fact RAM checks (down: on the ground 100%; hit: health lost) must pass before any training.
- A validation split by whole match (taken from the training matches, never from test), >= 100 rows per data dir.

## Four runs, one question each, never mixed (owner)
| run | question | answers | training balance |
|---|---|---|---|
| mv_move | what is <fighter> doing | the 10 movements | already equal per answer |
| mv_face | which way is <fighter> facing | left / right | already equal |
| mv_air  | is <fighter> on the ground or in the air | 2 | equal draws per answer (one data dir per answer, --balance sampling; no copies) |
| mv_dist | is <fighter> close to or far from the other | 2 | equal draws per answer, as above |
The question names the fighter (as the data rows do); input = the two frames (n-4, n), HUD visible.

Each run: from BASE (thaitea/laya-vision-smolvlm-256m), LoRA rank 16 / alpha 32, batch 8, lr head 1e-4 / adapters
2e-4, seed 0, on this Mac (MPS); budget 3 epochs of its training rows, eval every 250 steps, select on validation NLL,
patience 5, --max-minutes 120 as a guard. Runs one after another (one at a time on the GPU); memory watchdog.

## Judged on the held-out test matches (never trained or validated on), against RAM
Per run: balanced accuracy (mean of per-answer recall), per-answer recall and confusion, per character and per
facing; compared with (a) always the most common answer and (b) chance. "Learned" = the 2.5% lower bound of balanced
accuracy (1,000 resamples by whole match) above both. Reported for every run, pass or fail; nothing else changes.
