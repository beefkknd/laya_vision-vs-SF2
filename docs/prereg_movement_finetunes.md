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

## Result (2026-10-01)

### Data
- Rule: thrown (0x14) / knock-down (0x0E sub 0x04, no block react) is "down" only at y == GROUND_Y, else "hit"
  (sf2.data.pairs_labels; the gate's independent re-derivation the same, written separately). Tests first
  (tests/test_pairs_down.py, seen red), seeded faults: pairs suite 84 of 84 caught (tests/faults/pairs_faults.py,
  +9 down / second-fact, +8 fine-tune datasets, +2 for the jump fix below).
- Fix found on the way: a jump crossing the other fighter's x (ken_vs_dhalsim g6 t96, direction unknown on the same x)
  stopped the build; a jump is a cell without a direction, so it is now a valid pair (walks still need one).
- Rebuild under the rule at 6 games: 93.6%, 16 cells not full (all "down"). Top-up, one line per round:
  games 7: 95.6%, 16 not full | games 8: 97.4%, 16 | games 9: 98.7%, 11 | games 10: 99.4%, 4 not full
  (dhalsim down R 51/60, guile down R 48, honda down L 37, ken down R 44). Hard limit 10 reached; never padded.
- Build test_data_pairs2p_down: 36,582 pairs collected, 9,517 selected of 9,600 (14 split-cells short, all down:
  83 rows); gate PASS: labels 38,068/38,068, episode 9,517/9,517, caps, lag 1 (0.975 on discriminating pairs vs
  0.53 / 0.47 at lag 0 / 2), disk 1.88 GB. Second facts: down on the ground at t-4..t 877/877 (100%); hit with health
  lost in its hit episode 945/960 (98.4%).
- Validation: 1 in 6 training matches (crc32("val:<A>_vs_<B>:<game>") % 6 == 0), whole matches; test untouched.
  Matches: train 307, val 65, test 188. sf2.data.pairs_train / scripts/build_mv_data.py; question names the fighter
  ("What is Ryu doing?", "Which way is Ryu facing?", "Is Ryu on the ground or in the air?", "Is Ryu close to or far
  from the other fighter?"), no note. Rows (train / val / test):
  move (one dir) 5,261 / 1,093 / 3,163 (per answer train 505-535, val 89-115, test 283-320);
  face (one dir) 5,261 / 1,093 / 3,163 (left 2,648 / right 2,613 in train);
  air: ground 4,049 / 818 / 2,424, air 1,212 / 275 / 739 (one dir per answer, --balance sampling);
  dist: close 1,408 / 321 / 896, far 3,853 / 772 / 2,267 (one dir per answer, --balance sampling).
- train.py's coverage gate keyed a screen by the frame's basename only: the same g0000_k00009.png in 56 match folders
  read as 256 val positions "also in train" (a false stop of mv_move). Fixed: the key includes the frame's folder
  (flat frames/ keys and the old position split unchanged; test seen red).

### Training (exactly as registered; one at a time on MPS, free memory 96% before each, no other training)
| run | steps (of 1,972) | best step | best val NLL | minutes | stop |
|---|---|---|---|---|---|
| mv_move | 1,972 | 1,500 | 2.071 (step 0: 4.108) | ~10 | 3 epochs done |
| mv_face | 1,972 | 1,750 | 0.692 (ln 2 = 0.693) | ~8.5 | 3 epochs done |
| mv_air  | ~1,500 | 250 | 0.494 | ~6.5 | early stop (patience 5) |
| mv_dist | 1,972 | 750 | 0.559 | ~8.5 | 3 epochs done (final eval also the 5th without gain) |

### Held-out test (188 matches, 3,163 rows each; scripts/eval_mv.py -> runs/mv_<q>/eval.json)
| run | balanced acc | 2.5% lb (1,000 match resamples) | majority (bal / acc) | chance | per-answer recall | learned |
|---|---|---|---|---|---|---|
| mv_move | 0.265 | 0.251 | special 0.100 / 0.101 | 0.100 | stand .75, walk tw .07, walk aw .36, crouch .21, jump .44, attack .29, special .16, block .04, hit .06, down .28 | YES |
| mv_face | 0.515 | 0.498 | left 0.500 / 0.498 | 0.500 | left .74, right .29 | NO |
| mv_air  | 0.706 | 0.687 | ground 0.500 / 0.766 | 0.500 | ground .81, air .60 | YES |
| mv_dist | 0.645 | 0.628 | far 0.500 / 0.717 | 0.500 | close .41, far .88 | YES |
Per character (balanced): move 0.24-0.30, face 0.48-0.56, air 0.64-0.77, dist 0.61-0.69; per facing: move 0.26 / 0.27,
air 0.70 / 0.71, dist 0.62 / 0.67 (left / right). Plain accuracy: air 0.763 and dist 0.749 vs always-majority 0.766
/ 0.717. Confusions in the eval.json files (move: walk toward mostly answered walk away; block / hit spread).

## Round 2 (owner, 2026-10-02): ask by screen side, not by name
Owner: "Change question for all, ask left and right, this clears the context. Vision may find it hard to tell
character by character, also may be confused about who am I. Try a different round, give left/right in the question."
- Same frames, same labels, same splits (no new collection). Each question names the fighter by screen side:
  "the fighter on the left" / "the fighter on the right", side = which fighter has the smaller x at the displayed
  frame t (lag 1); pairs where the two x are equal (crossing) are dropped and counted. No character names, no note.
- Questions: "What is the fighter on the <side> doing?", "Which way is the fighter on the <side> facing?", "Is the
  fighter on the <side> on the ground or in the air?", "Is the fighter on the <side> close to or far from the other
  fighter?" (distance is symmetric; asked the same way for consistency).
- Four runs again, identical settings to round 1 (from BASE, r16/a32, batch 8, lr 1e-4 / 2e-4, seed 0, 3 epochs,
  eval every 250, select val NLL, patience 5, MPS), runs/mv2_{move,face,air,dist}; same held-out evaluation and
  "learned" rule, reported side by side with round 1, plus per side (left / right).

## Round 2 result (2026-10-02)

### Data
- sf2.data.pairs_train ask="side" (scripts/build_mv_data.py --ask side -> test_data_mv2_<q>): side = smaller x at the
  displayed RAM row t (lag 1; never t - 4, never the capture row); equal x dropped and counted; the question names no
  character, no note. Tests first (tests/test_pairs_train_side.py, 24 of 24 red before the code: exact texts, either
  slot, equal x, a one-pixel crossing, a crossover between the two frames, the independent problems() re-derivation).
  Seeded faults: pairs suite 97 of 97 (+13 side), mv_eval 9 of 9 (+2 per-side breakdown). Head tokens max 107.
- Same source build, labels, splits and dirs. Dropped for equal x: 6 rows (all train) in every dataset. Rows
  (train / val / test): move 5,255 / 1,093 / 3,163; face 5,255 / 1,093 / 3,163; air ground 4,044 / 818 / 2,424, air
  1,211 / 275 / 739; dist close 1,402 / 321 / 896, far 3,853 / 772 / 2,267. Sides: train 2,620 L / 2,635 R, val
  549 / 544, test 1,580 / 1,583. train.py's coverage gate passed for all four.
- Found in the data before training: the side fixes the facing on 99.2% of rows (test 3,137 of 3,163: the fighter on
  the left faces right). The face question now carries its answer in its text.

### Training (round 1's settings exactly; one at a time on MPS, free memory 96-97% before each, no other training)
| run | steps (of 1,972) | best step | best val NLL | minutes | stop |
|---|---|---|---|---|---|
| mv2_move | 1,970 | 1,250 | 2.046 (step 0: 4.003) | ~10 | 3 epochs done |
| mv2_face | 1,500 | 250 | 0.118 (val acc 0.99) | ~7 | early stop (patience 5) |
| mv2_air  | 1,970 | 750 | 0.505 | ~9 | 3 epochs done (final eval the 5th without gain) |
| mv2_dist | 1,750 | 500 | 0.648 | ~8 | early stop (patience 5) |

### Held-out test, round 1 (by name) vs round 2 (by side); runs/mv{,2}_<q>/eval.json
| run | balanced acc | 2.5% lb | majority (bal / acc) | chance | per-answer recall | learned |
|---|---|---|---|---|---|---|
| mv_move  | 0.265 | 0.251 | special 0.100 / 0.101 | 0.100 | stand .75, walk tw .07, walk aw .36, crouch .21, jump .44, attack .29, special .16, block .04, hit .06, down .28 | YES |
| mv2_move | 0.274 | 0.260 | special 0.100 / 0.101 | 0.100 | stand .74, walk tw .28, walk aw .12, crouch .16, jump .26, attack .35, special .15, block .03, hit .21, down .45 | YES |
| mv_face  | 0.515 | 0.498 | left 0.500 / 0.498 | 0.500 | left .74, right .29 | NO |
| mv2_face | 0.992 | 0.988 | left 0.500 / 0.498 | 0.500 | left .99, right .99 | YES (see below) |
| mv_air   | 0.706 | 0.687 | ground 0.500 / 0.766 | 0.500 | ground .81, air .60 | YES |
| mv2_air  | 0.741 | 0.723 | ground 0.500 / 0.766 | 0.500 | ground .83, air .66 | YES |
| mv_dist  | 0.645 | 0.628 | far 0.500 / 0.717 | 0.500 | close .41, far .88 | YES |
| mv2_dist | 0.649 | 0.630 | far 0.500 / 0.717 | 0.500 | close .63, far .66 | YES |
Per side (balanced, left / right; n 1,580 / 1,583): move 0.277 / 0.272, face 0.500 / 0.500, air 0.744 / 0.739,
dist 0.652 / 0.645. Per character (balanced): move 0.23-0.30, face 0.98-1.00, air 0.68-0.78, dist 0.58-0.74. Plain
accuracy: air 0.786 (majority 0.766), dist 0.655 (majority 0.717).
- mv2_face passes the registered rule, but within each side it scores exactly 0.500: it answers the facing that goes
  with the side named in the question and never the 26 test rows where they disagree. The pass is read from the
  question text, not the frames; it is no evidence that laya-vision sees facing.
- move, air, dist: small gains (move +0.009, air +0.035, dist +0.004 balanced; all lower bounds above round 1's
  point estimates only for air); dist traded far recall for close recall (.41 -> .63). Equal per side throughout.
