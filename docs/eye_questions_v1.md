# The eye's questions (draft v1, 2026-10-02) - owner: "fireball Yes/No, then the rest of the questions, compile the
# list and match the training data, align them first"

Every question is asked about the screen only (two frames, n-4 and n, HUD visible); fighters are named by screen side.
One question per fine-tune, never mixed. Labels from RAM (lag 1).

| # | question | answers | label from RAM |
|---|---|---|---|
| 1 | Is there a fireball on the screen? | yes / no | a projectile slot active AND drawn at the frame (hide bit off); yoga flame is not a fireball |
| 2 | Which way is the fireball moving? (only frames where #1 = yes) | toward the left / toward the right | the projectile's x change between the two frames (owner 2026-10-02: "not who threw it - where it goes") |
| 3 | What is the fighter on the <side> doing? | moving / attack / special | movement label; attack = normal, jump attack, throw; special = pressed special, RAM-confirmed |
| 4 | Is the fighter on the <side> on the ground or in the air? | ground / air | y |
| 5 | Are the two fighters close or far? | close / far | gap vs the calibrated poke band |

Dropped: facing (the side gives the answer); #2 for this round (owner 2026-10-02: the fireball's presence matters
much more than its direction - "not too keen on the fireball movement, don't get lost there"). Four questions: 1, 3, 4, 5.

## Alignment rules for the training data (all five questions)
1. One frame pool: the same 2P matches, games and whole-match splits (train / val / test) for every question.
2. Within a question, the answers differ only in the thing asked: drawn from the same matches, games, characters and
   sides (matched per character pair and side), so no shortcut (who is on screen, which game, which pose) predicts the
   answer.
3. Hard cases on purpose: #1 "no" includes throwers in their throwing pose with no projectile drawn (blink frames,
   after impact); #3 includes jump attacks vs plain jumps; #4 includes take-off and landing frames.
4. Each answer drawn equally in training (one data dir per answer, no copies); test keeps round 3's test rows where they
   exist.
5. A mechanical shortcut check per question: a trivial predictor from metadata only (characters, side, game, pose
   class of the other fighter) must not beat chance by more than a set margin; otherwise the data is not aligned.
6. Gate as before (labels vs RAM, episode, projectile drawn, lag 1, disk) + contact sheets.

## Training plan (owner: "fine. all good, journal and go ahead", 2026-10-02)
1. Data first: rebuild the five datasets from one frame pool (rollouts/pairs2p games 0-31, the round-4 pairs build
   test_data_pairs2p_mv4 and the fireball samples) to the alignment rules above; shortcut check + gate + contact
   sheets; shown to the owner before any training.
2. Five separate runs (one question each) on threebody - the measuring stick per question.
3. One combined run, all five questions: each question an equal share of training draws (~20%), each answer equal
   within its question; same frames, splits and test set.
4. Decision, mechanical: keep the combined checkpoint if every question's balanced accuracy is within 0.02 of its
   separate run on the same held-out test; otherwise separate adapters for the questions that lose more.
Run lengths, patience and the time limit are fixed in a pre-registration before step 2.

## Datasets v1 (2026-10-02; data only - no training, no new games)
Owner during the build: q2 changed to "which way is the fireball moving", then DROPPED for this round ("presence
matters far more"); no test_data_eye_q2_* was built. Four datasets.

- Pool (sf2.data.eye_pool): every image pair on disk in rollouts/pairs2p (committed games 0-30; the movement samples,
  the projectile samples and any pair the saved images form): 123,732 pairs, labelled from RAM at the displayed row t
  (lag 1). One split table (pairs_train.split3: test crc32 % 3 == 2 - round 3's test matches stay test; val 1 in 6
  training matches): 910 train / 173 val / 556 test matches over the four datasets, no match in two splits.
- q1 label: a fireball flight's slot on AND drawn (blink bit clear) AND on the screen. Found while gating: the slot
  stays on while the projectile flies off the edge of the screen (36 of 557 hadoken "yes" pairs showed no projectile).
  No scroll byte is recorded, so the screen is estimated from RAM: left edge = clamp((x1 + x2) / 2 - 128, 32, 224)
  (fitted on 2,318 drawn hadoken pairs, 97.6% agree with the blue-pixel test); "yes" needs the projectile >= 8 px
  inside; a drawn fireball off / at the edge is neither answer (310 pairs dropped). Games < 10 have no blink byte
  (46,488 pairs: not in q1). Yoga flame = "no".
- Alignment (sf2.data.eye_data): within each stratum every answer gets the same count (q3 / q5 at most 2, q4 at most 1
  per answer per stratum, q1 uncapped). Strata: q1 (split, match, game range 10-15 / 16-23 / 24-31, player 1's pose,
  player 2's pose; pose = "projectile" (attack state with the projectile word pressed) else the grid movement);
  q3 / q4 (split, match, game, side, the other fighter's grid movement); q5 (split, match, game, left and right grid
  movement). The first pass matched on the coarse act class and FAILED the shortcut check (q1 0.643, q3 0.427, q5
  0.679 balanced vs chance 0.5 / 0.333 / 0.5); matching on the grid movement fixed it. q1 "no" rows are taken hard
  first (blink, before spawn, after impact within 12 rows, throwing pose); q3 only pairs inside one episode.
- Rows (train / val / test, per answer, equal): q1 v1 (superseded, see "q1 v1.1" below) yes / no 666 / 147 / 418; q3 moving / attack / special
  3,575 / 692 / 2,194; q4 ground / air 9,160 / 1,791 / 5,609; q5 close / far 4,144 / 889 / 2,525.
- Hard cases: q1 "no" hard negatives 410 / 76 / 238 (tags over all splits: throwing pose 697 = the "yes" rows in
  throwing pose, before spawn 343, blink 269, after impact 55, fireball drawn in the n-4 frame 62, yoga flame 0);
  q3 jump attacks 1,741 / 368 / 1,056 (attack), plain jumps 1,338 / 263 / 738 (moving); q4 take-off 1,091 / 234 / 651
  (air), landing 721 / 157 / 467 (ground). Round 3's test rows kept: q3 865 of 3,163 (all in test); q1 0 of 636 -
  their games have no blink byte (none rows, games 0-9) or no "no" frame on disk (games 10-15 of the thrower pairs).
- Shortcut check (sf2.data.eye_shortcut; lookups per feature + logistic regression on metadata, fit on train, balanced
  accuracy on test, margin 0.05): q1 best 0.524 (+0.024, logreg), q3 0.333 (+0.000), q4 0.500, q5 0.500 - PASS.
- Gate (scripts/gate_eye_data.py, exit 0): labels re-derived 2,462 / 19,383 / 33,120 / 15,116 rows, 0 mismatches;
  q3 episode 19,383 / 19,383; splits one table; q1 drawn: hadoken "yes" 480 / 481 blue, Ryu/Ken-only "no" 66 / 66 clear
  (52 / 52 hard); lag 1: 1.000 (400) and 1.000 on discriminating pairs vs ~0.5 at lag 0 / 2 for every dataset; disk
  5.04 GB. Contact sheets test_data_eye_contact/q{1,3,4,5}_*.png; q1 looked at: projectiles in the "yes" frames,
  none in the hard "no" frames. Caveat seen there: Guile's sonic-boom windup draws a swoosh arc around his arms
  before the slot turns on (label "no", correct by RAM and frames, but it looks like a crescent).
- train.py: q3 / q4 / q5 pass the coverage / sampling checks at the defaults; q1 needs --min-sampled-train 600 (666
  per dir; val 147 >= 100). A combined run (step 3) has 7,730 val rows: needs --val-limit above the 4,000 default.
- Tests: tests/test_eye_pool.py, tests/test_eye_data.py; seeded faults tests/faults/eye_faults.py 38 of 38.

### q1 v1.1 (orchestrator review of the q1 contact sheet, same day): the label looks at BOTH shown frames
v1 labelled "no" when the fireball blinked off in frame n although it was drawn in frame n-4 ("no: blink" /
"no: prev_drawn" rows): that teaches the eye to miss an incoming fireball whenever it flickers. New rule (builder
sf2.data.eye_pool.fire_why and the gate's independent eye_gate.fire_at):
- yes = a fireball slot on (not yoga flame), drawn AND on the screen (>= 8 px inside) in at least one of frames n-4, n;
- no = no fireball (nor unknown projectile) slot on in either frame (windup / before spawn, after impact / gone,
  normal play; yoga flame = no);
- neither (dropped, counted) = on but drawn on the screen in neither frame (hidden in both 305; off / at the edge
  217), unknown projectile on 13, no blink byte 46,488.
Only test_data_eye_q1_fireball was rebuilt (q3-q5 unchanged). Rows yes / no 571 / 135 / 356 (train / val / test),
matches 283 / 45 / 177, 683 strata. "yes" drawn in both frames 804, n-4 only 101, n only 157. Hard negatives (no rows)
279 / 56 / 152; tags: throwing pose 467 (= the yes rows in throwing pose), before spawn 356, after impact (gone in
both frames, ended within 12 rows) 54, yoga flame 0. Shortcut check: best 0.521 (lookup:game, +0.021) - PASS. Gate
PASS: labels 2,124 / 2,124; lag 1 1.000 (400 and 257 discriminating vs 0.46 / 0.54); drawn (matches of Ryu / Ken /
Blanka / Zangief with a hadoken thrower - no blue fighter): yes 126 / 126 blue in every frame RAM says it is drawn,
no 126 / 126 clear in both frames (85 / 85 hard). Contact sheet test_data_eye_contact/q1_fireball.png looked at
(bands: yes, throwing pose, drawn in n-4 only, drawn in n only, the no tags). train.py: q1 needs
--min-sampled-train 500 (571 per dir; val 135 >= 100). Seeded faults 41 of 41.

## Training pre-registration (owner: "journal and start", 2026-10-02)
On threebody (RTX 4090; Qwen stays off; base model from the local HF cache, HF_HUB_OFFLINE=1), all from BASE, LoRA
r16/a32, batch 8, lr head 1e-4 / adapters 2e-4, seed 0, eval every 250, select on validation NLL, patience 10,
--max-minutes 180, --balance sampling. One run at a time.
| run | data | budget |
|---|---|---|
| eye_q1 | test_data_eye_q1_fireball (yes/no; --min-sampled-train 500) | 10 epochs (~1,430 steps) |
| eye_q3 | test_data_eye_q3_act | 6,000 steps |
| eye_q4 | test_data_eye_q4_air | 6,000 steps |
| eye_q5 | test_data_eye_q5_dist | 6,000 steps |
| eye_all | all four; each QUESTION an equal 25% share of draws, answers equal within a question | 12,000 steps (--val-limit 8000) |
Judged on the held-out test matches against RAM: balanced accuracy, 2.5% lower bound (1,000 resamples by match),
"learned" = lower bound above majority and chance; per answer, side, character; accuracy weighted to real play
(fireball 3.5% of frames; moving 71.5% / attacking 28.5%; air 28% - from the old U collection's RAM). eye_all kept
only if every question's balanced accuracy is within 0.02 of its separate run; otherwise separate adapters.

## Training runs and results (2026-10-02, as pre-registered)
Combined-run support: scripts/train.py `--balance question` (each question = a --data dir's parent folder gets an
equal share of draws, its answer dirs equal within it; laya mix weights from sf2.data.train_data.question_weights,
checked from laya's own mix; defaults unchanged). Tests tests/test_train_question_share.py; seeded faults
tests/faults/question_share_faults.py 10 / 10. Scoring sf2.data.eye_eval + scripts/eval_eye.py (tests
tests/test_eye_eval.py; faults tests/faults/eye_eval_faults.py 9 / 9).

On threebody (Qwen off, checked), one unattended chain `cmd /c C:\work\laya_finetune\run_eye.cmd` launched via WMI
09:56, ended 11:30 (`CHAIN EXIT`); code C:\work\laya_finetune\code_eye (git archive of 11337eb); data
C:\work\laya_finetune\data\eye\test_data_eye_q*\<answer> with frames junctions to data\eye_images (104,022 images,
all 139,486 references checked present); logs logs\eye_{q1,q3,q4,q5,all}.log (each "EXIT 0"), chain log
logs\eye_chain.log; outputs runs\eye_* (copied to the Mac's runs/eye_*, sha256-checked). Work note
C:\work\laya_finetune\WORKNOTE_eye.md. Commands (cwd code_eye, HF_HUB_OFFLINE=1, from BASE):
```
COMMON=--rank 16 --alpha 32 --batch-size 8 --lr-head 1e-4 --lr-backbone 2e-4 --eval-every 250 --patience 10
       --select nll --max-minutes 180 --seed 0 --device cuda
train.py <yes,no>                --out ..\runs\eye_q1  --epochs 10       --balance sampling --min-sampled-train 500 COMMON
train.py <moving,attack,special> --out ..\runs\eye_q3  --epochs 4.475897 --balance sampling COMMON
train.py <ground,air>            --out ..\runs\eye_q4  --epochs 2.620306 --balance sampling COMMON
train.py <close,far>             --out ..\runs\eye_q5  --epochs 5.791988 --balance sampling COMMON
train.py <all nine answer dirs>  --out ..\runs\eye_all --epochs 2.495231 --balance question --min-sampled-train 500
                                 --val-limit 8000 COMMON
```
(<...> = `--data ..\data\eye\test_data_eye_qN_*\<answer>` per answer; epochs chosen so int(epochs x train rows / 8)
is exactly the pre-registered 1,427 / 6,000 / 6,000 / 6,000 / 12,000 steps.)

| run | budget | ran to | best step | best val NLL | train min | stop |
|---|---|---|---|---|---|---|
| eye_q1 | 1,427 | 1,427 | 250 | 0.691 | 3.9 | budget (only 6 evals after step 0) |
| eye_q3 | 6,000 | 4,500 | 2,000 | 1.047 | 15.7 | early stop (patience 10) |
| eye_q4 | 6,000 | 5,500 | 3,000 | 0.435 | 22.0 | early stop |
| eye_q5 | 6,000 | 3,750 | 1,250 | 0.637 | 12.3 | early stop |
| eye_all | 12,000 | 7,250 | 4,750 | 0.711 | 36.1 | early stop |

Held-out test matches vs RAM (scripts/eval_eye.py; runs/eye_*/eval_q*.json; lb = 2.5% bound, 1,000 resamples by
match; weighted = recall weighted to real play: q1 yes 3.5%, q3 moving 71.5% / attacking 28.5% split attack : special
in the pool's ratio 41,995 : 23,609 (build.json candidates - an assumption, the prereg gives attacking only), q4 air
28%; q5 has no real-play share):

| run | q | n (matches) | balanced | lb | majority | chance | learned | weighted | recall |
|---|---|---|---|---|---|---|---|---|---|
| eye_q1 | q1 | 712 (177) | 0.501 | 0.500 | 0.500 | 0.500 | NO | 0.038 | yes 1.000, no 0.003 |
| eye_all | q1 | 712 (177) | 0.699 | 0.662 | 0.500 | 0.500 | yes | 0.736 | yes 0.660, no 0.739 |
| eye_q3 | q3 | 6582 (487) | 0.437 | 0.423 | 0.333 | 0.333 | yes | 0.502 | moving 0.545, attack 0.425, special 0.340 |
| eye_all | q3 | 6582 (487) | 0.448 | 0.436 | 0.333 | 0.333 | yes | 0.497 | moving 0.515, attack 0.548, special 0.282 |
| eye_q4 | q4 | 11218 (554) | 0.860 | 0.854 | 0.500 | 0.500 | yes | 0.861 | ground 0.861, air 0.860 |
| eye_all | q4 | 11218 (554) | 0.784 | 0.777 | 0.500 | 0.500 | yes | 0.765 | ground 0.741, air 0.827 |
| eye_q5 | q5 | 5050 (465) | 0.663 | 0.649 | 0.500 | 0.500 | yes | - | close 0.790, far 0.535 |
| eye_all | q5 | 5050 (465) | 0.627 | 0.614 | 0.500 | 0.500 | yes | - | close 0.857, far 0.398 |

- By side (balanced): q3 eye_q3 left 0.443 / right 0.430, eye_all 0.447 / 0.449; q4 eye_q4 0.844 / 0.877, eye_all
  0.789 / 0.778. By character: q3 eye_q3 0.385 (blanka) - 0.483 (guile); q4 eye_q4 0.830 (dhalsim) - 0.890 (blanka);
  eye_all q4 0.750 (dhalsim) - 0.810 (blanka). No side or character far off the rest.
- q1: eye_q1's NLL-selected checkpoint (step 250) answers "yes" to everything (val NLL 0.691 = ln 2); later steps
  reached val accuracy ~0.6 but NLL exploded (5-6.6) - 571 rows per answer overfit within 2 epochs, and NLL selection
  keeps the flat model. eye_all on q1: "no" accuracy plain 0.81, after impact 0.74, throwing pose 0.64, before spawn
  0.61; "yes" drawn in both frames 0.70, n only 0.56, n-4 only 0.51.
- Keep rule (mechanical, margin 0.02): q1 +0.198, q3 +0.011 within; q4 -0.077 and q5 -0.035 lose more ->
  **eye_all NOT kept; separate adapters for q4 and q5** (eye_all is the better checkpoint for q1 and within margin
  for q3).

## Next: more fireball data (owner, 2026-10-02: "you can start collecting more fireball data")
- Why: q1 has 571 train rows per answer; the separate run overfit (selection kept a flat "yes" model) and the combined
  run caught 66% of fireballs. Target >= 2,000 train rows per answer (yes / no) under the SAME q1 rules (yes = drawn in
  either frame, >= 8 px on screen; no = no projectile in either frame; hard negatives; matching; shortcut check).
- How: more 2P rounds (games 32+) on the 44 ordered pairs that include a thrower (ryu, ken, guile, dhalsim), with the
  projectile sampling trigger; fireball cap per (thrower x side x flight stage) raised as needed; the "no" side drawn
  from the same new games (matching keeps it aligned). Hard limit 80 games per pair in total; report fill per round.
- Then rebuild q1 only (q3-q5 unchanged), gate, shortcut check, contact sheet. No training in this step.

## Questions v2 - relabel (owner, 2026-10-02)
Owner: "attack, block, walk, jump, stand should be default, no crouching. high / normal / low position. A special move
is like a continuous attack. Can we relabel the data set?" + answers: being hit / knocked down -> its own answer "hit";
position is a separate question replacing ground/air.
- q3 v2 "What is the fighter on the <side> doing?" -> attack / block / walk / jump / stand / hit.
  attack = normal, crouching normal, jump attack, throw AND special (a special is a continuous attack);
  block = standing or crouching guard; walk = toward or away; jump = in the air with no attack out (incl. take-off and
  landing); stand = standing or crouching still (crouching is a position, not an action); hit = being hit, thrown or
  knocked down (incl. on the ground after the knockdown).
- q4 v2 "Is the fighter on the <side> high, normal or low?" -> high = in the air (y above ground); low = crouching
  (crouch, crouching attack, crouching block - from the state / move); normal = standing on the ground otherwise.
- Relabel from the same frame pool and RAM (no new games); same alignment rules (matching, equal draws per answer,
  shortcut check, gate, contact sheets). q1 fireball and q5 distance unchanged.
- Amendment (owner, same day: "attack and attack label as 'special attack'"): q3 v2 keeps specials as their own answer.
  Answers: attack / special attack / block / walk / jump / stand / hit (7). attack = normals (standing, crouching,
  jumping) and throws; special attack = the character's special (pressed-move rule, RAM-confirmed).

## Datasets v2 (relabel) (2026-10-02; data only - no training, no new games)
Pool: sf2.data.eye_pool over rollouts/pairs2p games 0-31 only (`--max-game 31`: a collector was appending games 32+
meanwhile; nothing written there): 127,643 image pairs, labelled at the displayed row t (lag 1). Labels
sf2.data.eye_v2; build `python scripts/build_eye_data.py --questions q3v2,q4v2 --max-game 31`; gate `python
scripts/gate_eye_data.py --questions q3v2,q4v2 --sheets test_data_eye_contact_v2`. New dirs only (v1 untouched):
test_data_eye_q3v2_act, test_data_eye_q4v2_pos. Answer dirs: the answer with "_" for the space (special_attack); the
answer text in the question is "special attack".
- q3 v2 (7 answers) from the movement rule + pressed-word rule (pairs_labels.movement_pressed): attack = movement attack
  (state 0x0A standing / crouching normal or throw; a jump with the attack box out, 0x?C3E != 0); special attack =
  movement special (0x0C, or 0x0A with move class 8; the pressed word decides attack vs special when RAM shows an
  attack state); block = guard 0x08 or 0x0E with block react 06 / 08; walk = walk toward / away; jump = in the air or
  jump state, no box out; stand = stand or crouch (0x02); hit = hit or down (0x0E, 0x14 in the air or on the ground,
  knock-down 0x0E sub 4). Unknown movement (0x06 turning, missing t - 4): dropped. Episode rule: rows t - 4 .. t have
  ONE v2 answer (so crouch -> stand or thrown -> down stay one episode).
- q4 v2: high = y != 192. low = on the ground and crouching, from these RAM fields (checked on games 0-31):
  crouch = state 0x02 (sub 0 crouched, sub 2 the 1-6 rows rising out of it); crouching attack = state 0x0A with move
  class 0x?CBA == 2 (standing normals 0; pressed crouching normals: 662k rows class 2, pressed standing normals: 0 rows
  class 2); crouching block = guard 0x08 with sub 4 / 6 (standing guard sub 0 / 2), or 0x0E with react 8 (crouching
  block stun; standing = react 6; every block-stun entry agrees with the guard sub it came from: react 6 <- sub 0 / 2,
  react 8 <- sub 4 / 6). normal = on the ground otherwise (incl. specials, hit, lying after a knockdown, the jump
  state's take-off squat on the ground).
- Alignment as v1: strata (split, match, game, side, the other fighter's grid movement); q3 v2 cap 2, q4 v2 cap 1 per
  answer per stratum; round 3's test rows preferred for q3 v2 (240 kept, all test).
- Rows (train / val / test per answer, equal): q3 v2 409 / 106 / 269 (x7; 274 / 65 / 178 matches, 619 strata);
  q4 v2 4,198 / 841 / 2,504 (x3; 855 / 169 / 520 matches, 7,543 strata). q3 v2 is small: every stratum must hold all
  7 answers in one game (candidates 12,939 block - 55,529 jump; 31,448 dropped outside an episode); the cap is not
  the limit (uncapped 426 train). For the owner: matching WITHOUT the game (split, match pair, side, other's grid
  movement) would give 4,329 train per answer uncapped, shortcut best 0.163 vs chance 0.143 (+0.020, passes) - not
  built (the spec matches on the game).
- Kinds inside the answers: q3 v2 attack = crouching 195, jump attack 344, standing 191, throw 54; stand = crouch 588,
  standing 196; block = crouching 378, standing 406; hit = down 150, in the air 499, on the ground 135; special
  attack = air 159, ground 625; jump on the ground (take-off / landing) 89. q4 v2 low = crouch 4,240, crouching attack
  1,995, crouching block 1,308; normal includes walk 1,570, jump-state squat 1,268, special 1,258, stand 1,240.
- Shortcut check (margin 0.05): q3 v2 best 0.143 (+0.000), q4 v2 best 0.333 (+0.000) - PASS.
- Gate (exit 0, gate.json in each dir): labels re-derived independently (pairs_gate's movement + the gate's own v2
  table and crouch rule) 5,488 / 5,488 and 22,629 / 22,629, 0 mismatches; q3 v2 episode 5,488 / 5,488; one split
  table; lag 1: 1.000 (400 and 400 discriminating; lag 0 / 2 ~0.45 / 0.55); disk 6.79 GB. New gate "second_fact"
  (q4 v2): the crouching read from what we pressed - crouching attack rows with a crouching normal pressed 1,994 /
  1,995; crouching block rows with the stick held down + back on a row t - 4 .. t 1,292 / 1,308 (block_low or the
  down-back charge of Blanka / Honda / Guile specials; block_low alone was only 70% - first FAIL, the rule was too
  narrow); controls (not gated): standing attacks with a crouching normal pressed 5 / 1,067, standing blocks with
  down-back held 31 / 553.
- Contact sheets test_data_eye_contact_v2/q3v2_act.png, q4v2_pos.png (bands per answer and per kind), looked at.
  Low vs normal: crouches, crouching attacks and crouching blocks look low. Caveats (labels follow the definition, the
  sprite disagrees): Ryu / Ken crouching fierce (c.hp) rises into an uppercut - "low" while standing tall (94 low rows);
  some Dhalsim standing normals are drawn squatting / lying (91 normal attack rows are Dhalsim); the jump state's
  take-off squat on the ground is "normal". Attack vs stand: attacks show a limb out; "stand: crouch" rows are still
  crouches.
- train.py (--balance sampling): q4 v2 passes the checks at the defaults (4,198 train / 841 val per dir; val 2,523 <
  4,000); q3 v2 needs --min-sampled-train 400 (409 per dir; val 106 >= 100).
- Tests tests/test_eye_v2.py (72); seeded faults tests/faults/eye_v2_faults.py 23 of 23; v1 faults still 41 of 41.
