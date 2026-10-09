# laya text vs Street Fighter II: lessons learned

*A screen-only self-learning player for SF2. The game is seen through a sprite reader (no RAM in play); a small
"System 1" — a text model plus a per-character **value table** and a **quorum of voters** — picks the move; the
table learns from the net-HP outcome of each exchange. After training, frozen tables beat the CPU at Ryu ~95%
(Zangief), and Honda reaches 98.4% vs Ryu / 98.5% vs Ken.*

This page is the part worth reading: what went wrong, and how each problem was found and fixed. The headline lesson
is that almost every hard problem here was a **measurement** problem, not a modelling one.

> An earlier era of this repo used a 256M vision model + a scripted teacher + DAgger imitation (the "laya-vision
> eye"). That pipeline was deleted (`3d37466`, `3d7239a`). Everything below is the current screen-only table+quorum
> system.

## Why

Two goals, by doing them:

1. **Make a "System 1" that decides from the screen in one shot** — one look, one move, no reasoning chain, four
   times a second — and have it *learn* from whether the move actually worked, not from a label.
2. **Build the harness that makes that measurable**, and gate every change on real play like a regression test.

As with most of this kind of work, the second turned out to be nearly all of the value — and nearly all of the bugs.

## What the system is

- **The eye.** A sprite-template reader (`sf2/screen/`) turns one frame into facts — who, where, posture, airborne,
  facing, health bars, fireball on screen, round over — with **no RAM in the play path** (a hard gate enforces it).
  `sf2/system1/screen_words.py` turns those facts into a one-line situation.
- **The memory.** A per-character **value table** (`sf2/system1/value_table.py`): a dict keyed by situation, each
  cell holding, per candidate move, the running mean **net HP** (damage dealt − taken) as Welford stats
  `[n, sum, sumsq]`. The key is `range | opponent-posture | fireball`, optionally split by the opponent's behaviour
  in close/mid cells.
- **The decider.** `sf2/quorum/decider.py` collects votes each frame: a text model (`laya`, the generalist and
  fallback), the `table` (votes a cell's confident move), and — during training only — a set of **bees** (voters
  that fill gaps). It tallies confidence × reliability, and the top move acts if its share clears a threshold.

A move is chosen; the net HP of the exchange is credited back to that move's cell; the table slowly learns what beats
what, per opponent.

## How it's wired

### One decision: screen to buttons
1. Frame → reader → facts → words: *"He is at mid range, attacking, no fireball; my bar is full, his is half."*
2. The situation is keyed into the table cell. The text model proposes a move; the table proposes its
   confident-best move for that cell; (in training) bees propose under-tried moves in thin cells.
3. Votes are tallied. **Hybrid rule** (`docs/design_laya_table_hybrid.md`): the table only *overrides* the text
   model where it is **confident and positive** — a move with `n ≥ 8` and mean net-HP `> 0`. Otherwise it either
   explores an under-sampled move (prob `--explore`) or defers to the text model. The `mean > 0` floor is load-
   bearing: without it the pure-table policy collapses into a defensive loss (it went 0/144 vs Ryu, `c3c44d9`).
4. Buttons are pressed; the decision is logged (situation, votes, pick, source).

### The training loop
- Train from a blank table with the full bee roster on, `--explore ≈ 0.3`, workers sharing one text-model server
  (**8 workers to train, 16 to measure** — training is GPU-bound on the shared server, measuring parallelises;
  `CLAUDE.md`, `45fc490`).
- Each round, every decision's net HP folds into its move's Welford stats (`value_table.credit`).
- **Workers pool samples, not policies.** Because Welford stats are additive, `value_table.merge` sums
  `[n, sum, sumsq]` across workers (valid at one-step credit; a shared seed is counted once, `519b84f`). The earlier
  belief that "pooling diverges into mush" was wrong and is corrected in the recipe.
- **Keep a table only if it measures better** than the previous best (a ratchet), measured frozen (next section).

## Problems I hit, and what solved them

### 1. The measurement lied — and a whole "regression" was noise
The first big result was a table that "regressed" round after round (78% → 53% → 39%). It was an artefact.
`--no-learn --explore 0` does **not** make play greedy: under the quorum policy the **explore bees still vote**, loudly
and near-randomly (a frontier bee's confidence stays 1.0 forever in an all-negative cell; posture bees stack a second
~1.0 vote; the tally *sums*). So every "measurement" was scoring `table + noise`, not the table.

**Fix:** measurement is a different thing from play. Measure **frozen, bees off** — explore voters disabled,
`epsilon = 0`, no learning, policy = text-model + table only, ≥ 8 seeds × 8 games, and only promote on a gain over the
previous best. (`docs/value_quorum_training_recipe.md`.) This one distinction — *train with the roster, measure
without it* — is the backbone of every number in the repo.

### 2. One character's win can crater another
Two changes that looked like wins on the character they were tuned against quietly destroyed a different one:
- A global **margin-drop + `priors.laya 0.7`** took Zangief 75 → 92% — and took **Honda 92 → 18.8%** (`a581eb2`,
  reverted `bea23d1`).
- A **shrunk-mean** move selection fixed a real Zangief `n=3` fluke (20 → 72%) — and regressed **Honda 92 → 59%**
  (`ce97ee0`, reverted `f333864`).

Honda's best one-step moves are often "least-bad" traps that a strong shared vote will walk straight into.

**Fix — the cross-character regression gate** (`scripts/voter_regression.py`): any change to the shared voter or its
config must re-measure the **whole roster** frozen vs Ryu and fail if any character drops below its floor (honda 80,
chunli 80, ken 68, ryu 78). A single-character green is never sufficient.

### 3. A value table is only as valid as the moves it was trained on
A charge-timing fix changed Honda's move menu (56 → 80 frames, dropped a move) and silently invalidated his trained
table — which had learned on the old scripts. Nobody re-measured; his win rate fell 93.8 → ~60%.

**Fix** (`eb32829`): restore Honda-specific 56-frame moves, and treat *any* menu change as a reason to re-measure the
tables that depend on it. Restored Honda vs Ryu to 98.4%.

### 4. The table must not be allowed to only defend
When every move a cell had tried was net-negative, the table **abstained** and left the text model unopposed — and
the text model would over-block Honda straight into jump-ins.

**Fix** (`7b02990`): in an all-negative cell the table still votes the **least-bad** move (chosen by a Welch
separation test), above the text model's pick, with confidence capped below 1. Honda/336 games: Ryu 88.4 → 92.0%,
Ken 79.7 → 98.5%; the `close|jumping` cell flips from *block* to a counter-hit.

### 5. Bees fill gaps; they must not push a category the table already owns
Bees that *forced* a category (defend / punish / combo) hurt — they spammed a move the thick trunk of the table
already played (combos ~6×/round), and the table measured **65% < 71%** without them (`faaf5c4`, retired).

**Fix — the bee-admission gate** (`python -m sf2.quorum.bee_check`): from the table alone, before any 90-minute
round, score each proposed bee RED/GREEN. **RED-MUTE** = the move is already thick, so the bee (confidence
`k/(k+n)`) can't change anything. **RED-LOSER** = the bee is loud but its move loses in every cell its gate fires.
**GREEN** = loud in an under-sampled cell on a positive-or-least-bad move. A dud is caught in milliseconds, and the
three bees in a round must each target a **different** cell, judged by that cell's own net-HP delta — not by
aggregate match %.

### 6. One table cannot serve two fighting styles
Trying to make one Zangief table beat both Ryu and Chun-Li fails, and now we know exactly why. A Zangief-vs-Ryu table
(94.5%) and a Zangief-vs-Chun-Li table (trained from blank to 64.1%) **share 78% of their situation keys** — they see
the same situations — but on the shared, decided cells the **best move flips 84% of the time**, with 10–65 net-HP
swings. Vs Ryu, Zangief learns to commit (piledriver, lariat, throw); vs Chun-Li's faster, safer pokes he learns to
block and poke back. The *same* observed situation demands opposite moves, because the thing that decides it —
the opponent's style — isn't in the key.

**Fix:** one table per style-cluster, with the opponent known at runtime. "Separate table per opponent" and "add the
opponent to the key" are the same decision; the second is better engineering, because the handful of cells that *do*
agree (close-range throw, crouch-block at range) can stay shared.

### 7. A finer key only earns its keep if the best move actually flips inside it
Before concluding the above, we tried a finer key — splitting attack cells by the opponent's limb and zone (hand/leg
× high/mid/low). It cost real machinery and bought almost nothing: the best move flips across limb×zone in **~1%** of
decisions (block is "least-bad" across nearly every limb), versus **84%** across opponents. Even the one cell where
the fine value looked large, the coarse cell already picks the same move.

**Fix** (`e2a63a1`): default back to one coarse `attack`; keep the finer split in the tree but gated off. The test
that decides it is the same for both: *does the argmax change inside the finer cells?* Split on the opponent (it
does); don't split on the limb (it doesn't). (`docs/attack_key_coarse_vs_fine.md`.)

### 8. Credit, and what "pooling" means
Early tables credited only the immediate exchange (one-step), which is myopic when a move's payoff comes a beat later;
`5ea1f1b` added an n-step / discounted return. And the pooling correction (Lesson intro): because the stats are
additive, parallel workers merge by **summing samples**, which is exactly valid at one-step credit — the instinct to
fear "diverging policies" was the wrong mental model.

## What I'd tell someone trying something similar
- **Build the measuring instrument before the model, and make play and measurement different code paths.** The worst
  bug here wasn't in the learner; it was that "measure" silently still had the training noise in it.
- **Gate every shared change on the whole roster, not the character you tuned.** Cross-character regression is the
  default failure mode of a shared value function.
- **Make dud-detection cost milliseconds.** A gate that reads the table and says "this bee can't matter" saves 90
  minutes per bad idea.
- **A finer key is a liability until you've shown the best action flips inside it.** More resolution without more
  leverage is pure cost.
- **The opponent's style is a hidden state variable.** If the same observation wants opposite actions, the key is
  underspecified — split it, don't average it.

## Still open / known debt
- **Hard gate is RED on purpose.** `scripts/hard_gate.py` reports ~42–44 import-closure violations ("no table/RAM in
  the play path"); accepted, tracked debt (`ddb37ca`).
- **Structural matchup caps.** A context with no positive answer (the `far | attacking` fireball wall) is a ceiling a
  bee cannot crack; the recipe says to *declare* it, not grind rounds.
- **Chun-Li tops out ~64%** for Zangief — a slow grappler vs a fast zoner is a genuine matchup ceiling, not a missing
  move.
- **Scoped-out:** an `n=3` shrunk-mean fluke (per-character fix deferred, `f333864`); `priors.laya = 0.7` kept per-
  character only, global default stays 1.0.
- **Not built:** nightly consolidation of proven cells into text-model fine-tune rows, and racing in the evolution
  loop (`docs/quorum_integration.md`).

## Repo map

| What | Where |
|---|---|
| Screen reader (image → facts, no RAM) | `sf2/screen/`, `sf2/system1/screen_words.py` |
| Value table (key, credit, merge, split) | `sf2/system1/value_table.py` |
| Quorum decider + voters/bees | `sf2/quorum/{decider,tally,voters,frontier,bees,config}.py` |
| Bee-admission gate | `sf2/quorum/bee_check.py` |
| Cross-character regression gate | `scripts/voter_regression.py` |
| The training recipe + the three gates | `docs/value_quorum_training_recipe.md` |
| Hybrid table/text-model design | `docs/design_laya_table_hybrid.md` |
| Coarse-vs-fine attack-key analysis | `docs/attack_key_coarse_vs_fine.md` |
