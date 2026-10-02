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

Dropped: facing (the side gives the answer).

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
