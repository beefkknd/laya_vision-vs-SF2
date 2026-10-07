# Value-table + bee-quorum: training recipe & pitfalls

Owner-facing recipe for training a character's value table with the bee quorum (e.g. Honda vs Ryu).
Written 2026-10-06 after a long detour caused by ONE measurement bug — the pitfalls below are the
point of this doc: do not fall into them again.

## TL;DR — the recipe that works

1. **Train** from blank with the full bee roster ON (laya + table + explore bees), Coach ON
   (`qwen_mode two`, learning on), quorum escalation OFF (`qcfg.qwen=False`), `--explore ~0.3`,
   `--shared-text-laya`. Pool K workers at Ryu scale (8 × 42 ≈ 336 games) and `VT.merge(shared=seed)`.
   The explore bees are TRAINING tools — they exist to gather coverage, nothing more.
2. **Play / measure FROZEN with the explore bees OFF.** The frozen policy is `laya + table` only
   (all of `frontier/fireball/pressure/punish/vs_crouch/antiair` = false, `epsilon=0`, `--no-learn`).
   This is the single most important line in this doc.
3. **Ratchet**: keep a newly-merged table only if it measures (bees-off) >= the previous best
   (this is what the sibling run meant by "R2 regressed, restored").

Verified 2026-10-06, Honda vs Ryu, 8 seeds x 8 games, frozen:
- R8 table (8x42, 8 bees for training): bees-ON **39%** match  ->  **bees-OFF 93.8% match / 81.8% round**
- R7 table (8x42, 5 bees):             bees-ON 53%            ->  bees-OFF **81.2%** match / 74% round
- Target was 88% (the sibling Ryu run). Playing correctly, Honda clears it.

## THE pitfall (cost ~a day): the explore bees vote at PLAY time

`--no-learn --explore 0` does NOT make play greedy. It only stops *crediting*. In `--policy quorum`
the explore bees STILL propose and vote. And they are loud and nearly random:
- `frontier_proposal` fires on EVERY decision with confidence `k/(k+covered_n)`, where `covered_n`
  counts only moves with mean > 0. In any cell where every sampled move is net-negative, `covered_n`
  stays 0 -> confidence stays **1.0 forever**, no matter how much data. `table_proposal` ABSTAINS
  there (it needs mean>0). laya's confidence is `P(cat)*P(move) < 1`.
- The posture bees (`pressure/punish/vs_crouch/antiair`, `fireball`) are CLONES: they call the same
  `_least_sampled` and stack a second ~1.0 vote in their slice; `tally.score` SUMS votes. Two stacked
  1.0 votes beat laya and table and clear `theta`.
- Net effect at play: in every hard cell, the policy becomes "play the least-sampled move" = random.

So measuring a trained table with bees ON measures `table + random noise`. **More games -> more cells
go confidently all-negative -> more cells owned by frontier (78%->53%). More explore bees -> a second
loud vote in more slices (53%->39%).** The whole R1->R8 "regression" was this, not the training.

### What was WRONG in the earlier analysis (corrected)
- "Pooling K workers diverges into mush" -- **WRONG**. One-step (horizon=0) Welford merge pools
  SAMPLES, not policies; it is more data for the same bandit. Pooling is fine. (Only `credit_horizon>0`
  or the `depth` split union make merge policy-dependent; keep horizon 0.)
- "Per-bee negative net means the bee is bad" -- **misleading**. Explore bees fire only in hard slices
  (opp attacking / fireball out / etc.), so negative net-when-played is selection bias, not badness.
  Judge a bee by in-context lift vs the fallback, never by raw net.
- "Coverage holes = weaknesses, make a bee per thin/blind cell" -- **wrong tool**. A rarely-visited
  cell is not a weakness. See the weakness ledger below.

## Designing bees from WEAKNESS, not coverage

Run the weakness ledger first (no games needed), from a merged table + traces, per `when`:
`visits`, `sum_net` (all actions), `best_covered_mean` (n>=20), `laya_move_mean`, `top_opp_label`.
Rank by `sum_net` (most negative first) and classify the top contexts:
- **A. answer exists** (`best_covered_mean>0`, or confidently beats laya via `_separated`) but is
  rarely played -> a quorum WEIGHTING bug -> free win, no new bee. (Most HP loss is expected to be A.)
- **B. no answer** (all covered means < 0) -> a genuine weakness -> build a targeted counter-bee.
- **C. under-sampled** (no move at n>=20) -> the only case where a coverage/frontier bee is the fix.

Use the Coach's natural-language lessons as the hypothesis LIST for B (each lesson ~ a candidate
`(context, counter-move)` pair) -- never as the gate.

### Counter-bee spec (replaces "least-sampled in slice")
A counter-bee targets ONE losing pattern:
- fires only when its `(context, opp his_label)` predicate matches;
- confidence = `k/(k+n_of_ITS_OWN_action)` -- loud until its own counter is tested, silent after;
  NEVER 1.0 when the cell already has a confident positive move;
- retire when `n>=20 and _separated(best, counter)`; promote to a plain table entry when
  `_separated(counter, best)`;
- verified by frozen net-HP IN THAT CONTEXT (same seeds, before/after), NOT by match-win.

Also fix the `table` voter: vote the covered argmax when it confidently beats laya's move (relative
`_separated` test), not only when `mean>0`. "Block in close|attacking" is a legitimate answer.

## Open code fixes (recommended, from the Fable design review 2026-10-06)
- `frontier._confidence`: fade on total evidence `k/(k+n_cell_total)`, not only on positive winners.
- Dedupe explore votes: take MAX over explore bees, never sum (stop the clone-stacking).
- Keep `frontier` only (gated to class-C cells); the posture bees help TRAINING coverage but should
  not be generic voters -- prefer targeted counter-bees. Explore bees are NEVER in the frozen policy.

## Checklist before trusting a number
- [ ] measured FROZEN with explore bees OFF (`frontier/fireball/pressure/punish/vs_crouch/antiair=false`, epsilon 0)
- [ ] enough scale (Ryu reference = 336 training games; measure >= 8 seeds x 8 games)
- [ ] ratchet: accepted only if >= previous best
- [ ] bee contribution judged by in-context lift, not raw net

## Update 2026-10-06 — the table-voter fix SHIPPED (commit 7b02990)

The "open fix" above (table votes the covered argmax via `_separated`, not only `mean>0`) is DONE.
Root cause found via gameplay ledger + Fable review: in every all-negative cell the ballot was
laya ALONE (table abstained), so she blocked unopposed at share 1.0 -- Honda over-blocked into
jump-ins. `table_proposal` now proposes the least-bad covered move when it is Welch-`_separated`
strictly above laya's move, confidence `n/(n+k)*margin` capped at `NEG_CONF_CAP=0.9` (never ~1.0
at rest, so no frontier-style stacking). Positive-mean cells unchanged (golden held).
Frozen bees-off, honda_r8, 336 games: **Ryu 88.4->92.0%, Ken 79.7->98.5%** match. `close|jumping`
flips `block_high`->`cl.hk`. None of the 3 proposed bees were needed -- the table already knew the
answer; it just wasn't allowed to vote it. Lesson: before adding a play-time bee, check whether the
move it would propose is already in the table and merely out-voted (a WEIGHTING fix, not a new voter).

## Per-round fine-tune flow (owner-set 2026-10-07) -- the law for EVERY round

**Each fine-tune round gets its OWN review and its OWN bees, aimed at the CURRENT table's
weaknesses.** Bees are never carried over from a prior round's analysis. The flaw this fixes: the
Zangief bees were designed from R1's ledger and then reused for R2 and R3 -- so they targeted
yesterday's weaknesses, not the table being trained. Bees must be measured ONE round behind at most:
you review table N, design bees for table N, train N+1.

The loop, per round N (current best table):
1. **Measure** table N frozen bees-off vs the opponent (8 seeds x 8 games). This is the number to beat.
2. **Review** -- run the weakness ledger on THAT run's frozen play (`scratchpad/study/ledger.py`,
   two-cell horizon): rank `when` contexts by net-HP bled; classify A / B / C.
   - **A** (a covered move beats what's played) -> usually already handled by the exploit voter
     (table_proposal: separation vote + shrunk-mean selection). If not, it is a WEIGHTING fix, not a bee.
   - **B** (no positive answer) / **C** (under-sampled) -> candidates for a bee.
3. **Design NEW bees from table N**, not from memory. For each target context, read what the table
   ACTUALLY holds there and push the real less-bad / thematic counter move (e.g. Zangief's
   `double_lariat` eats fireballs and the table shows it at -1.6 vs block's -8.3 at `mid|attacking`) --
   NOT a guessed move (`jump_forward` into the fireball was wrong). Aim them via the config:
   `QuorumConfig(counters=True, counter_specs=[[range, posture, move], ...])` -> a train config JSON.
4. **Train** N+1: carry best table N, `STUDY_QCONFIG=<that config>` (counters on), 8x42, explore 0.3.
5. **Measure** N+1 frozen bees-off; **ratchet**: keep (promote to canonical) only if >= best; else
   restore best and try a different aim. A single round's +/-3% on 64 games is within noise -- confirm
   a real gain at 336 scale before trusting it.
6. Repeat from 1 with the new best. Fix the exploit VOTER first (biggest lever); bees are for the
   residual B/C only, and a context with NO positive answer (e.g. `far|attacking` fireball wall) is a
   structural matchup cap a bee cannot crack -- say so instead of grinding rounds.

## Bee SELECTION principles (owner-set 2026-10-07, after a wrong set + the Fable review)

A bee votes `k / (k + n of ITS MOVE in the cell)`, so it is **loud only where that move's n is LOW**
and silent where the trunk already has the move. That one fact dictates every selection rule:

1. **Aim at UNDER-SAMPLED cells, never at the biggest HP sink.** HP-bled finds WEAKNESS; it does not
   say a bee can HELP. A huge sink that is already well-sampled is a VOTE or HORIZON problem, not a
   coverage gap. Selection signal = **visits x uncertainty on a plausibly-good, under-sampled move**,
   NOT HP magnitude.
2. **Never push a move the thick trunk already owns.** At high n the bee's vote is ~`k/n` ~ 0 (mute),
   so it does nothing; worse, a loud bee on a known-BAD move drags the cell's share below `theta` and
   forces the fallback to laya's block -- it spends samples CONFIRMING a loss. (Original lesson,
   `frontier.py` docstring: the force-combo bee HURT, 65% < 71%, by spamming a trunk move. The Zangief
   R3 bees repeated it: `far->double_lariat` at n=1162 was mute; `mid|standing->walk_forward` was loud
   on a -20 move.)
3. **Classify the cell first, then pick the tool:**
   - **A** (a covered move beats what's played, but loses the vote) -> a VOTER / weighting fix
     (`table_proposal` selection + confidence, `priors.laya`), **not a bee**. Most big sinks are A.
   - **C** (under-sampled, no move at n>=20, but a plausibly-good move exists) -> the ONLY legit bee:
     push that move so the table can learn it. Loud because n is low -- exactly where a bee works.
   - **Horizon** (every one-step move is negative because the payoff is 2-3 decisions away, e.g.
     approach-then-throw) -> retrain the table on a 2-step / discounted return, **not a bee** and not
     a voter tweak. The one-step table literally cannot learn "don't be in that cell".
4. **Sanity-check the aim before spending a 90-min round:** read the target move's `n` in the cell. If
   n is already high, the bee is mute -- re-aim or drop it. Confirm the move is plausibly positive
   (or least-bad with upside), not a known loser.

Worked example (Zangief vs Ryu): the dominant sinks (`far|attacking` -5564, `mid|attacking` -2350)
were all class A -- the table knew `double_lariat` beat block but the voter crushed its vote. Fixing
the voter (drop the margin factor, `priors.laya=0.7`) moved 75.0% -> 92.2% with NO retraining and NO
bees. Three carefully-aimed bees before that moved nothing (two mute, one harmful). Fix the voter
first; reach for a bee only for a true class-C gap.

## The three mechanical GATES (owner-set 2026-10-07) -- every bee/voter change passes a RED/GREEN check

Prose rules drift; these are the [SCRIPT] gates that enforce them. Two failure modes, three gates:

1. **Bee admission gate** -- `sf2/quorum/bee_check.py` (`python -m sf2.quorum.bee_check --table T --specs
   '[[range,posture,move],...]'`). From the TABLE alone (no round), RED/GREEN per bee:
   - **RED MUTE**: the move is already thick (`k/(k+n)` tiny) in every cell its gate fires -> the bee
     can't change anything (e.g. `far|attacking->double_lariat` at n=1162).
   - **RED LOSER**: the move is loud but loses in ANY cell its gate fires (below the covered best, or
     below an absolute floor) -> it spends samples confirming a loss (e.g. `mid|standing->walk_forward`
     at -20.5). A bee fires in EVERY (range,posture) cell, so one bad split condemns it.
   - **GREEN**: loud in an under-sampled cell on a positive / blind / least-bad move. Run this BEFORE
     spending a 90-min round; a dud is caught in milliseconds.

2. **Distinct-signal requirement** -- the 3 bees in a round must target DIFFERENT gates/cells, and each
   is judged by ITS OWN cell's net-HP delta (the ledger), not the aggregate match%. Otherwise you
   cannot attribute which bee helped, which hurt, or whether two cancelled. Each bee is a predeclared
   checkpoint: "cell X net-HP must rise." A bee whose cell didn't move is a dud regardless of match%.

3. **Cross-character regression gate** -- `scripts/voter_regression.py`. ANY change to the SHARED
   exploit voter (`tally.py`) or its config (`config.py` priors/formula) MUST pass this before commit:
   re-measure the WHOLE trained roster frozen vs Ryu, FAIL if any character drops below its floor. A
   single-character green is NOT sufficient -- the margin-drop looked great on Zangief (75->92%) and
   cratered honda (92->18.8%) because honda's one-step "least-bad" moves are traps. "You changed
   something GLOBAL" is a different failure than "bad bee", and needs a roster-wide gate, not a bee check.

Rule of thumb: a BEE is a per-character, per-cell change -> gate 1 + gate 2. A VOTER/CONFIG edit is
global -> gate 3. Never commit a shared-voter change on one character's number.
