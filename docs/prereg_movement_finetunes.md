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

## Round 3 (owner, 2026-10-02): attack vs movement, special vs regular, and the fireball
Owner: "I only care if the opponent is attacking vs moving; if attacking, special attack vs regular attack. That
should be it. Another problem is the fireball in the air. That is a real bug. We need to capture it."
- Facing dropped (round 2: the side gives the answer). Air and distance stay as trained in round 2 (mv2_air, mv2_dist).
- Q "act": "What is the fighter on the <side> doing?" -> moving / attack / special. moving = stand, walk, crouch,
  jump, block, hit, down; attack = normal, jump attack, throw; special = the character's special (pressed-move rule,
  RAM-confirmed). Same frames and splits as round 2; equal draws per answer (one data dir per answer, no copies).
- Q "fireball": "Is there a fireball on the screen?" -> none / from the fighter on the left / from the fighter on the
  right. Label from RAM: shot slots (shot1 = player 1's, shot2 = player 2's), thrower's side by x at the displayed
  frame. Today only 277 of 9,471 pairs show one (3%), so collect more: new rounds of the same 2P matches with an extra
  sampling trigger - while a projectile is on screen, sample its flight at start / middle / end; cap per (thrower
  character, side, flight stage); throwers: ryu, ken (hadoken), guile (sonic boom), dhalsim (yoga fire). Gate as
  before plus: every fireball label checked against the shot slots and a frame (contact sheet). "none" pairs come
  from the existing data.
- Two runs (act, fireball), same settings as rounds 1-2, from BASE; same held-out evaluation and "learned" rule;
  per answer, per side, per character.

## Round 3 data (2026-10-02; training moved to threebody by the owner - not run here)

### act (test_data_mv3_act)
- sf2.data.mv3_act (scripts/build_mv3_data.py act): round 2's movement rows (same frames, sides, splits; 6 equal-x rows
  dropped as before), answer moving / attack / special (attack = the grid's attack, special = the grid's special,
  the other 8 movements = moving), one dir per answer. Tests first (tests/test_mv3_act.py); independent check
  problems_act 0. Rows (train / val / test): moving 4,195 / 874 / 2,523; attack 526 / 114 / 320; special 534 / 105 /
  320. Matches 307 / 65 / 188 (round 2's).

### fireball: the slots, the blink, the trigger
- Ownership confirmed before trusting it: in round 2's games 0-3, 255 of 258 projectile flights started while their
  own slot's player pressed a projectile word (shot1 = player 1, shot2 = player 2); frames (scripts/probe_shots.py, a
  scratch probe: one player throws, the other idles) show each player's hadoken / sonic boom / yoga fire leaving that
  player. New rounds: 463 flights, 376 with the slot's own player pressing hadoken / sonic boom / yoga fire, 1 where
  only the other player did, 86 neither (all Dhalsim's yoga flame, which also uses the slot).
- Found: the game BLINKS projectiles. While the slot is on, byte +0x3A of the slot (0x103A / 0x108A) bit 0 set = not
  drawn on that frame: hadoken 2 on / 2 off, yoga fire 1 frame in 4 hidden, sonic boom never. Probe: a blue-pixel
  detector agreed with the bit on 102/102 hadoken frames of player 2 (196/200 of player 1); the first "on" row is
  drawn in the next capture (lag 1). A label from the slot alone would be wrong on ~half the hadoken frames, so a
  fireball row needs the projectile DRAWN at the displayed row t (slot on, bit clear). Recorded as two new RAM vars
  (shot1_hide / shot2_hide) in the new rounds only.
- Trigger (sf2.data.pairs_shots.ShotSampler; scripts/collect_pairs.py --mode vs --shots --per-game 0 --pairs
  throwers): every flight is cut in thirds (start / middle / end), one drawn row per third at random, at most 3 per
  (slot, stage) per game; images = captures t - 3, t + 1. No movement pairs in these games (per-game 0).
- Caps (proposed and used): 40 train / 20 test per (thrower, side, flight stage) = 24 cells per split; hard limit 6
  new games per pair (games 10-15) on the 44 ordered pairs with ryu / ken / guile / dhalsim. rollouts/pairs2p/run.json
  saved first as run_before_g10_shots.json. One line per round (~1 min each, 0 failed workers):
  games 11: 13.5% filled, 48 of 48 cells not full | 12: 25.1%, 48 | 13: 37.7%, 48 | 14: 51.1%, 46 | 15: 64.3%, 46 |
  16: 75.0%, 37 not full. Hard limit reached; never padded. Shortest: train ken left end 15/40, ken left 19-20/40,
  ryu right 19/40, dhalsim right 25-28/40; test dhalsim right 10/20, guile right 12/20.

### fireball dataset (test_data_mv3_fireball)
- sf2.data.mv3_fireball (scripts/build_mv3_data.py fireball): 1,367 trigger samples of committed games; dropped 261
  not a projectile word (yoga flame), 5 with the other slot also on at t, 0 not drawn, 0 equal x; 1,101 eligible;
  capped per (split, thrower, side, stage). "none": round 2's pairs build rows (one per image pair) with both slots off
  over t - 8 .. t (no projectile, no impact spark), 9,171 in the pool, sampled per split to the larger fireball
  answer. Split by match as before (new games the same crc32 rule). Question "Is there a fireball on the screen?",
  answers none / left / right (thrown by the fighter on the left / right; side by x at t). Head tokens 56.
  Rows (train / val / test): none 295 / 75 / 234; left 295 / 38 / 234; right 270 / 75 / 168. Matches 319 / 72 / 224.
  Caveat: fireball rows come from games 10-15 and none rows from games 0-9 (same matches' pairs, same stage).
- Gate (scripts/gate_mv3_fireball.py) PASS: labels 1,684/1,684 rows re-derived from RAM and the move log (0
  problems); alignment lag 1: 1.000 (400 random) and 1.000 on 186 discriminating pairs vs 0.53 / 0.47 at lag 0 / 2 -
  pairs with the HUD clock <= 20 s are left out (237): the clock blinks there, and all 114 disagreements of a first
  run were such pairs; drawn: 525/546 hadoken rows show >= 200 blue pixels in the labelled frame, 21/21 none rows of
  Ryu-vs-Ken matches none; disk 1.94 GB. Contact sheets test_data_mv3_fireball_contact/fireball_{ryu,ken,guile,
  dhalsim,none}.png looked at: every sampled "now" frame shows a projectile on the labelled side (some at impact or
  at the screen edge), none frames show none.
- Tests first + seeded faults: tests/test_pairs_shots.py, test_mv3_fireball.py, test_mv3_fireball_gate.py; faults
  tests/faults/mv3_faults.py 38 of 38 (4 were green on the first run and got tests); pairs 97 of 97, mv_eval 9 of 9.
- Blocker for training (not changed here): scripts/train.py's --balance sampling gate requires >= 1,000 train and
  >= 100 val rows per dir (sf2.data.train_data MIN_SAMPLED_TRAIN / MIN_SAMPLED_VAL). act attack 526 / special 534
  train and every fireball dir (270-295 train; val 38-75) are below it, so both registered runs stop at the
  coverage check. Owner decision needed (lower the minimum for these runs, or more data).

## Round 3 training decision (owner: "help me decide ... everything I see on the screen is fair"; "ok fine approved")
- Option C: the eye is trained balanced (each answer drawn equally) - it reports what is on the screen; what matters
  for play (holes, exploits, what beats what) is decided by text laya / Qwen / the table, never by filtering the eye's
  training data.
- Data as built (7ef6768). train.py gets --min-sampled-train / --min-sampled-val (defaults unchanged 1000 / 100); these
  two runs use 250 / 35 (attack 526, special 534, fireball answers 270-295 train; fireball left val 38), recorded in
  train_log.json. Runs on threebody (RTX 4090, Qwen stays off), settings otherwise as rounds 1-2, --device cuda.
- Judged two ways on held-out matches: balanced accuracy (the pre-registered "learned" rule) and accuracy weighted
  to real-play frequency. Real-play frequencies (CPU: moving / attack / special; fireball on screen or not) come from
  the old U collection's RAM (rollouts/u_perception), read-only - owner approved reading it. Plus a per-game check
  of the fireball run (fireball rows come from games 10-15, none rows from games 0-9).

## Round 3 result (2026-10-02; trained on threebody, scored on the Mac; runs/mv3_act, runs/mv3_fireball)
threebody's Hugging Face download of BASE hung (0-byte weights); the identical snapshot was copied from the Mac (sha256
3122080...be68 matches) and the runs launched with HF_HUB_OFFLINE=1.

| run | balanced acc | 2.5% lb | majority (bal / acc) | chance | recall | learned |
|---|---|---|---|---|---|---|
| mv3_act (3 answers) | 0.414 | 0.393 | moving .333 / .798 | .333 | moving .68, attack .34, special .23 | YES (weak) |
| mv2_move collapsed to the same 3 answers | 0.413 | 0.393 | same | .333 | moving .74, attack .35, special .15 | YES |
| mv3_fireball (none/left/right) | 0.345 | 0.315 | none .333 / .368 | .333 | none .00, left .76, right .27 | NO |

- act: best val NLL 0.989 at step 1,000 of 1,972. No better than round 2's 10-answer model collapsed to 3 answers.
- fireball: 860 training rows -> 3 epochs = 322 steps only; it never answers "none". Real-play weighted accuracy for
  fireball would be about the share of "none" frames it gets right (96.5% of real frames have no fireball) - here 0.

## Round 4 plan (owner, 2026-10-02): "collect more data and then come back for long training, step by step"
Step 1 - collection only (no training), same 2P matches, same labels and checks:
- act: raise the caps of the attack and special cells (character x facing) from 40 train / 20 test to 100 train /
  30 test, so each answer reaches >= 1,000 train (16 cells x ~65+), with validation by whole match as before; moving
  unchanged.
- fireball: raise the per (thrower x side x flight stage) cap from 40 train / 20 test to 120 train / 30 test, aiming
  for >= 1,000 train per fireball answer (left / right); "none" matched to the larger fireball answer per split.
- Hard limit 50 games per ordered pair in total (about 1 minute per round of 56 / 44 matches); stop earlier when the
  targets are met; report fill after each round in one line; shortfalls named, never padded.
- Gate as before (labels vs RAM, episode, projectile drawn, lag 1, disk) + contact sheets; tests/faults for any change.
Step 2 - the long training is planned and approved separately once the data is in.
