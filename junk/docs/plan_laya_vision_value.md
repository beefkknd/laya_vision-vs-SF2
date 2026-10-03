# Plan: a laya-vision fine-tune for value and the opponent's signals (draft for the owner's decision, 2026-09-30)

Owner: "when the opponent is in the air or attacking you can prepare the next move, for any character; if they already
started a move, laya-vision needs to pick up this signal. This is a fair request." Evidence: docs/component_boundaries.md
(the throw hidden against all six opponents; laya-vision predicts a hit, the choice needs the net outcome).

## What laya-vision is today (read from the code and runs/all8)

- SmolVLM-256M + Laya typed-decision head (base thaitea/laya-vision-smolvlm-256m), one LoRA run (rank 16, 2 epochs,
  ~20 min on MPS), best val acc 0.874. Per move it answers "what happens if I do X now?": hit / whiff / blocked / none /
  got_hit. System 1 ranks attacks by P(hit), blocks by P(blocked).
- Input: two frames (n-4 and n), 256x256 padded, HUD blanked; a note "me=chunli dist=mid side=left dx=+65 my_bar=full
  opp_bar=full opp_airborne=0 opp_crouch=0" - opponent never named; nothing says he is attacking.
- Data: mostly a still dummy (stand / crouch); live rows capped; Chun-Li's live rows are all vs Dhalsim. It has hardly
  seen an opponent jumping at her or starting an attack. Damage dealt / taken is stored but not a label.
- Our 131k A/B and loop decisions saved no images: they cannot train it.

## The change (one run from BASE, general, same rules)

1. **What it predicts - value.** Keep the five outcomes (compatibility, evaluation) and add, per move, the net result
   of the decision (damage dealt minus taken until her next decision), in buckets: big loss / loss / even / gain / big
   gain. System 1 ranks by expected net (blocks included: a block's value is the damage it avoids; a punished Bird
   Kick shows as a loss). Rating words for text laya stay the same three, mapped from expected net - text laya is not
   retrained.
2. **What it sees - his signals.** The note adds `opp_attacking=0/1` (his state at the decision, opponent-agnostic,
   like opp_airborne), and the training data is dominated by a live, moving CPU: jumping, attacking, starting specials.
   The two frames (n-4, n) already carry the motion; the data teaches it to read them.
3. **Data - collect with images, trying every move.** Headless games, Chun-Li vs all seven CPU opponents, images on,
   a behaviour policy that tries every move she can pick in every situation (uniform over System 1's choices, plus the
   current policy's picks) so every move's value is learned where it was never tried (the throw: 1-6 tries per
   opponent in her history). About 7 opponents x 60 rounds, ~15-20k decisions; no Qwen; ~1-2 h in parallel. Guile's
   rounds are held out of training entirely (test of "any character").
   Other characters' existing rows stay in the dataset as today (all8), unchanged.
4. **Train.** One LoRA run from BASE with the all8 recipe; the new live rows + the existing ones. No sweeps.

## How it is judged (pre-registered before training)

- Offline, held-out games: outcome accuracy not below today's (Chun-Li 0.88); predicted vs real net per bucket
  (calibration); ranking: throw up close offered where it pays, Bird Kick demoted where it gets punished, blocks kept
  when he attacks; the same on held-out Guile.
- In play, no advice (System 1 alone): new vs current laya-vision, 6 opponents x 8 seeds, seed as the unit. Success =
  pooled hp/round better with no opponent clearly worse; she throws up close without being told.
- Then with Qwen's loop and the book, so the three components are measured together again.
- Everything is locked (lesson_loop_v3) so the current results stay reproducible.

## Cost and risk

- Code (labels, note flag, scoring, collector, eval) with tests: ~2-3 h; collection ~1-2 h; training ~0.5-1 h (more
  data); evaluation and the A/B ~2-3 h.
- Risk: value is noisier than hit/whiff (a CPU punish depends on timing) - mitigated by buckets and calibration checks.
- Risk: the note flag changes laya-vision's input format; the old checkpoint keeps the old note (versioned), so old runs
  replay unchanged.
- Memory notes from 2026-09-29 say "never retrain laya"; the owner's latest decision supersedes them once made.

## The running book test

It measures the Qwen-side route (verified tips) with today's laya-vision: the baseline the fine-tune will be compared
against, and the book stays useful afterwards. The fine-tune's first steps are code and tests; its collection needs the
machine (emulators + laya-vision inference) and its training needs the GPU, which would slow the running games (not
change their results - headless play is deterministic). Recommendation: do NOT stop it; write the code now, start the
collection when it ends (~3 h).
