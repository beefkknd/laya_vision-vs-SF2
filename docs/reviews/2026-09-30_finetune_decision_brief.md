# Independent analysis brief: does laya-vision need a fine-tune, and if so, what and how?

You are one of two independent analysts (the other runs on a different engine; you will not see each other's work).
Read the files in this folder; they are the evidence. Be concrete, cite files and numbers, and disagree with anything
in them that the evidence does not support, including earlier reviews.

## The project (short)
A bot plays Super Street Fighter II (SNES, arcade CPU opponents) on a Mac.
- System 1 = **laya-vision**: SmolVLM-256M + a typed-decision head, LoRA fine-tuned from a fixed BASE. Input: two game
  frames (n-4 and n, 256x256, HUD blanked) + a short text note (my character, range, dx, both health bars as levels,
  opp_airborne, opp_crouch; v2 adds opp_attacking). It answers per move "If you do X now, what happens?" (hit / whiff
  / blocked / none / got_hit). System 1 picks from these ratings (docs: code/system1.py, code/advice.py).
- Text laya: a small text model that picks from laya-vision's shortlist following advice (lessons).
- System 2 = Qwen (LLM) proposes lessons from game logs; a verifier judges them; a "book" of verified players' tips.
- Owner's rules: laya stays GENERAL (256 px, opponent-agnostic: no opponent names, no move names of the opponent, only
  general opponent state like airborne / attacking); a laya fine-tune is only for general play, never one character;
  "Qwen is the second brain, Qwen needs to learn" (character knowledge lives in Qwen); one training run from BASE, no
  sweeps (each run is an owner decision); pre-register before running; the seed/run is the statistical unit; the Mac
  must not be overloaded.

## Where we started (docs/component_boundaries.md, docs/harness_ledger.md, docs/2026-09-29_dr_fable.md)
- A players' tip "use more throw up close" helped against all six opponents (+20 to +64 hp/round). laya-vision
  ranks moves by P(hit), so the throw (high damage) was almost never in its top 3 up close (0-19%); her history had
  1-6 close throws per opponent, so Qwen never found it. The component study blamed laya-vision's ranking TARGET
  (P(hit), not value), not its eyes (its throw score was calibrated).
- The book of verified tips in Qwen's loop: +40.5 hp/round pooled over no book (docs/component_boundaries.md end).

## What we tried (in order)
1. Plan (docs/plan_laya_vision_value.md): teach laya-vision VALUE per move (net = damage dealt - taken until her next
   decision, 5 buckets) + the general signal opp_attacking. Data: headless games with a behaviour policy that plays a
   uniformly random move 50% of the time (explore 0.5), else the old policy; Chun-Li vs 7 CPU opponents x 120 games,
   the other 7 characters vs 2 opponents x 60 games; images saved (results/sample_actions.jsonl shows the log fields).
2. Run 1 (docs/prereg_lv_value.md): stopped on validation accuracy at 4% of the schedule; the value head = the class
   prior (75% of value labels are "even"). Review: docs/2026-09-30_dr_fable_lv_value.md (recipe failure).
3. The no-model lookup test (docs/prereg_value_oracle.md): a 96-cell table (character x range x opp_attacking x
   opp_airborne x move -> mean net of EXPLORED training decisions, shrunk) played as System 1's ranking beat laya-vision
   (runs/all8) by **+81.4 hp/round pooled [+42.7, +121.9]**, 369 vs 70 rounds won of 720, +146.8 on the held-out
   opponent (results/lv_inplay_oracle_report.json, results/value_oracle_v1.json).
4. Run 2 (docs/prereg_lv_value_run2.md): stop on NLL, no row cap, forward rows subsampled, gates anchored to the table.
   Val NLL improved (0.672 vs 0.738) and value cross-entropy dropped below its prior for 7 of 8 characters, but on
   NEW games (results/run2_eval_v2.json) the per-move value does not order real outcomes at all (calibration on
   explored decisions not monotone in 9 of 9 files; the throw never in the top 3 where the table puts it first).
   Outcome accuracy rose (Chun-Li 0.672 vs runs/all8 0.607). Training logs: results/run*_train_log.json.

## The owner's question now (verbatim, then our reading)
"We need decide if vision laya do need a fine-tune or not. If true what we want achieve and how we are going to do
it. Random game seems too noisy obviously, does this mean you need two play mode with selected major target. If that
is these case, a general interaction is fine, any BCDE against A, is fine. But not BA, CA, DA, that is way too much."

Our reading (an interpretation - say if you read it differently): instead of noisy random full games, a CONTROLLED
collection mode: set up a specific major situation A (e.g. the opponent jumps in, throws a fireball, walks in, sweeps,
is idle up close - general situations, not character-specific move names) from a savestate, and try each of her
responses B, C, D, E from the SAME frame, recording the outcome of each - one situation x all responses, a single
exchange. NOT sequences or combinations (B then A, C then A, ...), which multiply the cost. The repo already does this
for a still dummy and for block probes (code/collect.py, code/vs_defense.py, code/vs_sweep.py) - see how.

## What we need from you
1. **Diagnosis**: why did run 2 fail on new games while a 96-cell table succeeds? Is it the data (noise, policy
   mixture, window), the target (5-bucket per-move value, ordinal), the model/recipe, or the evaluation? Use the files'
   numbers; say what you are sure of and what is a guess.
2. **Decision**: does laya-vision need a fine-tune at all? Consider that the table (general note fields, no model)
   already gives +81 hp/round. What would a fine-tune add that the table cannot (e.g. timing inside a situation, reading
   startup frames of an attack, things the note cannot say)? Is that worth it for this project, whose goal is to learn
   each component's boundary (vision / text laya / Qwen / harness), with character knowledge in Qwen?
3. **If yes**: what exactly should laya-vision learn (the target), with what data (is the owner's controlled
   "situation A x responses B..E from one frame" mode the right one; which situations; how many; how to keep it general
   and opponent-agnostic), how to train (one run from BASE), and how to judge it (pre-registered gates, anchored to the
   table and to runs/all8; in-play test). Estimate cost (collection size, time) roughly.
4. **If no**: what should System 1 use instead (e.g. the table as its ranking, refreshed from data), and how should
   laya-vision's role be defined so the component boundaries stay clear?
5. **Risks** that could still mislead us.
Keep it concise and plain (the owner reads it). Markdown. End with a one-paragraph recommendation.

## Added question from the owner
"Also without this fine-tuning, qwen learned lot with search study game guide already?"
6. **Compare with the Qwen side**: the book of verified players' tips (searched game guides, verified by A/B, fed into
   Qwen's loop) gave +47.1 hp/round vs no advice and +40.5 over Qwen's loop without it (docs/component_boundaries.md,
   last section; docs/prereg_book.md). Note the baselines differ: the book's control is System 1 + text laya told
   "no advice" (runs/all8 ratings); the table's control is runs/all8 with the plain threshold rule, no text laya - so
   +47 and +81 are not directly comparable. Given the book already works, what does each route (book in Qwen, table in
   System 1, a laya-vision fine-tune) add, do they overlap (the book's biggest tip is the throw, which the table also
   finds), and which should own what? What single experiment would measure them together fairly?
