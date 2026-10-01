# Pre-registration: the U arm's laya-vision - the eye (2026-10-01, before any code, data or training)

Owner: "this is still a vision project... a human player can't read RAM, but they can see it on the screen. Name it U
arm, unified" (plan: docs/plan_u_arm.md); the question set from the players' research (docs/research/
2026-10-01_perception_synthesis.md) - owner: "Locked and start fine-tune". Baseline locked: lesson_loop_v3 (tag
lesson-loop-v3).

## Principle
At play time laya-vision sees only the screen (two frames, n-4 and n, HUD visible) and knows only its own character
("me=<char>"). RAM is the referee and teacher at training time only: it labels the answers and scores the games.
General: no opponent names, no move names of the opponent.

## The 8 questions (locked)
| # | question | answers | label (training only) |
|---|---|---|---|
| 1 | how far is he, and is the gap changing | throw / poke / mid / far x closing / steady / opening | gap from RAM x; band thresholds calibrated from logged data BEFORE training (throw band: gaps where throws connected; poke band: where ground normals connected); trend = gap change n-4 -> n |
| 2 | what is he doing | neutral / attacking / recovering after a miss / blocking / being hit | his state; "recovering after a miss" = attacking at n, no contact on me during the attack, still attacking k frames later (lookahead) |
| 3 | his air state | grounded / jumping at me / jumping away or straight up / landing | y, x motion toward me, landing within k frames (lookahead) |
| 4 | projectile | none / far / near (coming at me) | projectile slots, |shot_x - my x|, direction |
| 5 | who can act: me / him (two questions) | free / stunned / knocked down / dizzy | state, react, dizzy, frames until free (lookahead) |
| 6 | corner | me / him / neither | x within D of the stage's extremes observed in the logs |
| 7 | health bars: mine / his (two questions) | full / high / half / low | life |
| 8 | if I do X now, is it better than walking in (per move) | likely works / may work / likely fails | SOFT: per (situation, move), bootstrap the explored outcomes of that table cell (opponents included): share of resamples >= walking in + 3 hp / above walking in / not. Never a confident answer where opponents disagree |

Exact thresholds (bands, k, D) are fixed from data and written into this file before training starts; RAM-to-frame
alignment is checked (the frame and the RAM row must be the same moment).

## Data
A new headless collection that also logs the RAM rows around each decision (frames n-4 and n, plus a short lookahead),
same design as the value collection: explore 0.5 with runs/all8, images saved; Chun-Li vs 7 opponents x 120 games, the
other 7 characters vs Ryu and vs Dhalsim x 60 games; no text laya, no Qwen. Split by whole games: game % 10 in 2,5,8 ->
test; validation = a separate set of whole training-split games (val.jsonl), never neighbouring frames of training
games; Chun-Li vs Guile held out entirely. Characters sampled in equal shares.

## Training
One LoRA run from BASE (thaitea/laya-vision-smolvlm-256m), the all8 recipe (rank 16, alpha 32, 2 epochs, batch 8, lr
head 1e-4, backbone 2e-4, seed 0); keep best and early-stop on validation NLL over whole held-out games (eval every
1,000 steps, patience 5); note v3 ("me=<char>"), HUD visible.

## Offline gates (held-out test games and held-out Guile)
1. Perception accuracy per question (1-7), reported per character, with a floor fixed before training from a
   label-noise check (questions whose RAM labels are not visible in two frames are reported, not gated).
2. **The gate that matters (owner):** the table's best move (from RAM facts) is in laya-vision's top 3 by question 8 in
   >= 90% of held-out decisions, and of held-out Guile's.
3. Question 8 is soft: where the table's cell is uncertain across opponents, laya-vision's answer must not be confident
   (reported).

## In play
U0 (laya-vision eye, no advice) vs T0 (the table on RAM) and vs A0 (runs/all8), then U1 (with Qwen + book) vs T1 and
A1, Chun-Li vs 6 opponents x seeds 73001-73008 (the locked 2x2 runs reused). Success: U close to T (the price of seeing
instead of reading RAM) and clearly above A.

## Load
Collection (~2.5 h), dataset build, training (alone), eval, games: one at a time; memory watchdog; Qwen via
~/work/omlx/start only.


## Amendments before collection and training (2026-10-01; implementation: sf2/data/perception.py, lessons/perception_*_v1.json)
- RAM-to-frame alignment measured: the display lags RAM by 1 frame; labels are read at frame t = n-1 (HUD clock
  98.9% agreement at lag 1 vs 86.0% at lag 0 over 12,166 image pairs; projectile on screen 7/7 one frame after spawn).
- His projectile = the shot2 slot (RAM owner byte and probe agree).
- Q1 bands from data (a decision stump on connects, no free threshold): throw band gap <= 43 (Guile 44), poke band per
  character (Chun-Li 64 ... Zangief 82, pooled 72); mid/far at 120 (the table's own cut, kept for consistency); trend:
  +-2 px per 4 frames.
- Q2 "recovering after a miss" (tightened, so it does not include long startups): he is attacking at t, the whole
  attack episode is inside the stored rows (60 back, 60 ahead), t lies in its second half, and I never enter hit/block
  stun, a throw, or lose life during it; contact seen -> "attacking"; episode not fully observed -> "unknown".
- Q4 near/far at 120 px (the table's mid/far cut). Q6 corner: within 11 px of the observed walls (to be re-measured on
  the new collection: 4 characters' stages unseen so far). k (lookahead for landing / still attacking) recomputed from the
  collection before training (provisional 8).
- Q7 bars: the DRAWN bar (hp, which drains after a hit), not life: what a human sees.
- Q8 targets: 1,440 (cell, move) soft targets, resampled by opponent then decision (808 confident, 632 spread).
  Gate 2 (table's best move in top 3) is computed over decisions where the table's best move is not walking in
  (walking in is always offered to text laya); decisions where it is are reported separately.
- "unknown" labels are not trained on (masked) and not counted in accuracy.

## Amendments before training (2026-10-01)
- Training length: 25 questions per decision make ~0.87M training rows; 2 epochs would be ~216k steps (~27 h). Fixed
  budget instead: **36,000 steps** (~4.4 h at the measured speed; `--epochs` = 36000 x 8 / training rows,
  `--max-minutes 270` as a guard), everything else as registered (select on validation NLL, eval every 1,000, patience
  5, equal-share sampling per character: Chun-Li's larger set is seen ~0.09 epoch, the others ~0.55).
- Validation: whole val games (game % 10 == 0), capped at 48 decisions per character chosen by hash (~9.6k rows, the
  eval cost of the value runs); the rest of those games kept in val_rest.jsonl (reported, never trained on).
- Mapping of laya-vision's answers to text laya's words: range throw -> close, poke and mid -> mid, far -> far (95.2%
  agreement with RAM's words on Chun-Li's data); doing: airborne -> jumping, attacking / recovering -> attacking, being
  hit or him not free -> stunned, neutral / blocking -> standing; bars carried over.
- Known limitation (stated, not hidden): the decision's CONTENT uses no RAM, but the play loop still uses RAM for WHEN
  she may act (_can_act) and for which way the buttons face. A human sees both on screen (question 5 asks "can I act
  now"); replacing them with laya-vision's answers is a later step, measured separately.

## Note before the offline evaluation (2026-10-01, after training, before any eval)
Gate 1's per-question floors were to be fixed before training from a label-noise check; they were not set (my miss).
Gate 1 is therefore REPORTED, not gated, for this run. Gate 2 (the table's best move in the top 3 >= 90% on held-out
games and on held-out Guile) was fixed before training and decides, with gate 3 (softness) reported.
Training: 36,000 steps, best validation NLL 0.610 at step 35,994 (still improving at the end of the budget).

## Result (2026-10-01): gate 2 fails - stops before the in-play test (runs/u_eye/best/eval_u.json)

| question (answers) | test_real (18,729 decisions) | held-out Guile (4,300) |
|---|---|---|
| **gate 2: table's best move in the top 3** | **0.742** (13,496 / 18,195) | **0.710** (3,055 / 4,300) |
| can I act (4) | 0.968 | 0.973 |
| projectile (3) | 0.961 | 0.905 |
| his air state (4) | 0.852 | 0.836 |
| corner (3) | 0.837 | 0.783 |
| can he act (4) | 0.823 | 0.847 |
| range band (4) | 0.697 | 0.626 |
| his phase (5) | 0.548 | 0.666 |
| trend (3) | 0.499 | 0.430 |
| my bar / his bar (4) | 0.456 / 0.469 | 0.336 / 0.357 |
| Q8 word (3; 77% of labels are "likely fails") | 0.790 | 0.774 |

- Training was budget-limited: validation NLL still falling at step 36,000 (0.32 epoch; each question ~1/25 of rows).
- The health bars are visible (HUD unblanked, checked on frames) and correctly labelled (checked), yet read near chance:
  under-training, or the bar's few pixels at 256x256 - not yet known.
- Q8 barely beats always answering "likely fails"; its ranking misses the table's best move in ~26% of decisions.

## Owner decision after the gate (2026-10-01)
Owner: "hook this checkpoint and play a couple of games, see if this helps the game in place of the table. Maybe another
round of training on top of this checkpoint is needed. Health bar is not that important."
- Exploratory look, not the pre-registered success test (gate 2 failed): runs/u_eye/best through qwen_lessons --eye,
  Chun-Li vs the 6 opponents x seeds 73001-73002 (2 of the locked 2x2 seeds), 30 rounds per arm, book, character_fgc,
  shared text laya. U0 (no advice) pairs with the locked T0 / A0 of the same seeds (no Qwen involved: comparable).
  U1 uses the NEW Qwen server (192.168.1.173, UD-Q4_K_M) while the locked T1 / A1 used local Jundot: indicative only.
  2 seeds per opponent: below the 3-run minimum for a verdict.
- Health bars: deprioritised by the owner.
- A further training round continuing from this checkpoint (not from BASE) is an owner option for later.
