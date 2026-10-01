# Plan: the U arm (unified) - laya-vision as the eyes, the table as experience, Qwen + book as opponent knowledge (draft for the owner's decision, 2026-10-01)

Owner (2026-10-01): "this is still a vision project; based on what we learned from the table, see if laya-vision can
be used effectively. A human player can't read RAM, but they can see it on the screen. Name it U arm, unified."

## What we learned
- The table (lessons/value_oracle_v1.json: average outcome of each move per character x range x opp_attacking x
  opp_airborne) beats laya-vision's P(hit) ranking by +99 hp/round; with Qwen + book on top, 1,000 of 1,440 rounds won
  (docs/prereg_2x2.md). It reads its 4 facts from RAM: no vision at all.
- Teaching laya-vision VALUE failed: one noisy outcome per frame (sd ~16 hp, the situation explains ~7%).
- The table needs only facts a human sees on screen: how far he is, whether he is attacking, whether he is in the air
  (and her own character, which a player knows).

## The U arm
1. laya-vision reads the two frames (n-4, n) with NO RAM note (only "me=<character>", which a player knows) and answers
   three perception questions: how far is he (close / mid / far), is he attacking now (yes / no), is he in the air
   (yes / no).
2. The table gives each move's value for those facts; the rest is unchanged: shortlist rated against walking in, text
   laya picks, Qwen's loop + book advise (U1), or text laya told "Advice: none" (U0).
3. Nothing in U's decision reads RAM. (RAM is still used to LABEL training data and to log outcomes, as a referee.)

## Why this can work where the value fine-tune failed
The labels are exact (RAM at the decision frame), not noisy outcomes: perception is a clean supervised task, which
laya-vision already does well (its outcome answers are ~0.87 accurate on real frames). 64,266 decisions with saved frames
from the value collection (all 8 characters as the player; Chun-Li vs 7 opponents) + 13,494 from the earlier random
play vs Dhalsim; no new games needed for training. General by construction: no opponent names, general states only.

## Steps
1. **Cheap check first (no training):** runs/all8 asked the three questions with frames only, on held-out games -
   expected poor (it never had to read distance from pixels; the note told it), which shows whether a fine-tune is needed.
2. **Data:** perception rows from the saved frames: 3 questions per decision, labels from the logged RAM fields
   (range, opp_state in attack/special, opp_air); note v3 = "me=<character>" only; mirrors for training as before.
   Split by whole games (game % 10 in 2,5,8 -> test) and **validation by whole games too** (the value runs' validation
   came from neighbouring frames of training games); Chun-Li vs Guile held out entirely.
3. **One LoRA run from BASE** (the all8 recipe), kept on validation loss of held-out games.
4. **Offline gates (pre-registered):** per-fact accuracy on held-out games and on held-out Guile; and what matters for
   play: how often the table's best move from laya-vision's facts equals its best move from RAM's facts, and the value
   lost when it does not (hp per decision, by the table).
5. **In play:** U0 vs T0 and vs A0 (no advice: no Qwen, ~30 min), then U1 vs T1 (Qwen + book, ~5 h, 8 waves), Chun-Li
   vs 6 opponents x seeds 73001-73008 - the same seeds as the 2x2, so U pairs with its A and T runs. Success: U close
   to T (the price of seeing instead of reading RAM), and clearly above A.

## Cost and load
Data build minutes (frames exist); training ~1-3 h (alone on the machine); offline eval ~30 min; U0 games ~30 min;
U1 ~5 h. One heavy job at a time, memory watchdog, Qwen restarted only via ~/work/omlx/start.

## Owner decisions
- A laya-vision fine-tune (perception, general, one run from BASE): approve?
- Laya-vision as a perception model only for U (outcome questions stay with runs/all8 for the A arms), or one model
  that keeps the outcome questions too? Recommended: perception only - smallest change, cleanest test.
