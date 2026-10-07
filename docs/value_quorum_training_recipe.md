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
