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
