# Pre-registration: laya-vision value fine-tune (2026-09-30, written before any training)

Plan: docs/plan_laya_vision_value.md (approved by the owner 2026-09-30). Owner's boundary: laya-vision learns the
opponent's GENERAL state (in the air, attacking), never his move names or opponent-specific detail; the fine-tune is
for general play, not one character.

## What changes (code on branch feat/lv-value)

- Value question per move (sf2/data/value.py): "If you do X now, how does the exchange end for me?", five ordered
  buckets of net = dealt - taken until my next decision: big_loss <= -30, loss -29..-1, even 0, gain 1..29,
  big_gain >= 30. Expected net uses the measured bucket means (-41.1, -22.0, 0, +21.1, +47.0).
- Note v2: + `opp_attacking=0/1` (his state 0x0A attack or 0x0C special at the decision). The old checkpoint keeps v1
  (the note version is read from the checkpoint config).
- System 1 value mode: plays the best expected net over every choice, forward included; no threshold.
- Data: `rollouts/lv_value` - explore 0.5 (a uniformly random choice half the time, else runs/all8's pick), images on,
  no text laya, no Qwen. Chun-Li vs all 7 CPU opponents x 120 games (seeds 90001-90007); the other 7 characters vs
  Ryu and vs Dhalsim (Ryu/Dhalsim meet Ken) x 60 games (seeds 91001-91002). Split by game: game % 10 in (2,5,8) test.
  **Chun-Li vs Guile is held out entirely** (never in training).
- Balance (added before training, 2026-09-30): every character keeps the same number of new live training decisions
  (the smallest character's count); Chun-Li keeps her earliest games evenly across her 6 opponents, and her other
  training-split games go to a separate evaluation file (test_extra), never to training. The check that characters
  have equal training rows stays on: laya stays general.
- Block rows' opp_attacking: ground probes 1 (he is mid-attack at the decision frame), jump-in 0 (he is still in the
  jump state; note v2 counts only attack/special states) - inferred from the collector's timing, to be confirmed by a
  replay when the emulator is free.
- Dataset `test_data_v2/` = the all8 rows (note v2; no value rows for the still dummy) + the new live rows (outcome and
  value rows). Training: scripts/train.py with the all8 recipe (rank 16, alpha 32, 2 epochs, batch 8, lr head 1e-4,
  backbone 2e-4, eval every 250, patience 3, seed 0), from BASE, one run, no sweeps.

## Offline gates (held-out games; decided before training)

1. Outcome accuracy on test_real (all 8 characters) not below runs/all8's by more than 0.02 per character
   (runs/all8/best/eval_outcome.json).
2. Value calibration on the held-out live rows: in each predicted-expected-net quintile, the mean real net moves the
   same way (monotone over the quintiles), and the top quintile's real net > the bottom's by at least 15.
3. Ranking, Chun-Li, held-out games, close range: the throw is in the value top 3 in at least 50% of decisions
   (runs/all8's P(hit) ranking: 0-19%).
4. The same gates 2-3 on held-out Guile (never seen): the "any opponent" test.
5. Ranking vs the opponent's signal: when opp_attacking=1, a block is in the value top 3 more often than when
   opp_attacking=0 (reported per character).

Failing 1 or 2 stops here (no in-play test); 3-5 are reported either way.

## In-play test (System 1 alone, no advice, no Qwen)

- New checkpoint (value mode) vs runs/all8 (threshold rule), Chun-Li vs 6 opponents (ryu, ken, honda, zangief, guile,
  dhalsim) x 8 seeds (95001-95008) x 15 rounds, paired by seed; headless, deterministic.
- Unit: the seed (run). Pooled: opponents as the unit, hp/round (dealt - taken), 95% CI.
- Success: pooled hp/round better (CI above 0) with no opponent clearly worse (no per-opponent CI entirely below 0);
  she throws up close without being told (throws per close decision > runs/all8's).
- Then (separate pre-registration): with Qwen's loop and the book, the three components together; lock lesson_loop_v3.

## Load rules

One heavy job type at a time; training starts only after the collection and the book round are done; memory checked
before each step (free+inactive > 40 GB alert).
