# Table System 1 — a self-learning (when → action) value policy (spec for review)

Owner pivot 2026-10-04. Replace "frozen text-laya nudged by a few text rules" with a **self-learning System 1**:
a value TABLE over (when, action) that starts near a prior, is credited by outcomes, and pushes the better action
up — with EXPLORATION, so it generates its own counterfactual (the thing chun3/chun4 could never get). Qwen stays
System 2: it does NOT score and is NOT the policy; it accelerates exploration and explains. The two systems (the
current RULE policy and this TABLE policy) must be able to run SIDE BY SIDE for a fair A/B.

Branch: feat/table-system1 (in the existing -lv worktree; no new project folders). This doc is for Fable + GPT-6
review BEFORE any code.

## Why (settled in discussion)
- Today's System 1 = frozen text-laya; the only runtime learning is ≤10 text advice lines. Tiny surface, and a
  well-followed rule has NO counterfactual (chun4: rest=0 / idle=0), so nothing can be judged. Live chun4: 27
  rounds, 22% win, 0 graduations.
- A value table that EXPLORES tries alternative actions in a cell → learns their value → can compare → judges the
  chosen action. The counterfactual is free, by construction. That is the core reason to pivot.

## The model
- **State key "when"** = the detectable situation (from the eye, no retrain): `(range ∈ {close,mid,far}, doing ∈
  {standing,attacking,jumping,crouching,stunned})`, plus a `fireball` flag as a sub-key of its cell. (Same keys the
  review-framework settled on; "cell" == "when".)
- **Action** = one of the character's menu moves, RESTRICTED to the FOLLOWABLE set for that when (reuse Step-1A
  `advice.followable`) — the table never scores a move text-laya can't play there.
- **Value** `V[when][action]` = running outcome stat: `n`, mean per-decision net hp (`dealt-taken` over the
  decision's window), and the fired-round win-share. (Reuse `rule_stats` sums / Welford.)
- **Policy (play):** at a decision, over the followable actions for the current when, pick by value WITH
  exploration — under-sampled/optimistic actions get tried (UCB-style; ε-greedy as the simple fallback). The pick
  is the move System 1 plays.
- **Credit (learn):** after the action's window (one decision interval; net hp), fold the outcome into
  `V[when][action]`. Round win/loss folds into the fired-round win-share. MECHANICAL (measurement engine), never Qwen.
- **Game 0:** `V` empty. Two seedings — (a) UNIFORM (pure exploration, slow) or (b) **text-laya PRIOR**: use laya's
  move-probabilities for the when as the initial ordering / optimistic prior, so it starts from "a trained player"
  and outcomes correct it. Recommend (b): less blind flailing, laya stays useful as a prior not a decider.

## System 2 (Qwen) — accelerator, not scorer
- Between games, Qwen reads the table's weak/uncovered whens (bleeding cells, under-explored actions) and SUGGESTS
  which actions to try there — narrowing exploration from "all followable" to "these few", so convergence is faster
  than blind bandit. It may inject an optimistic prior into a cell (so the bandit tries a Qwen idea) — but outcomes,
  not Qwen, decide whether it stays. Qwen also explains. It never credits an action and never picks the move.

## Coexistence — two systems side by side (owner requirement)
Make the PLAYER pluggable behind one interface:
- `choose(when, followable_actions, rng) -> action`
- `credit(when, action, outcome) -> None`  (outcome = per-decision net hp + round result)
Two implementations:
- **RulePolicy** = today's text-laya + short-memory rules (wraps `two_stage_decide` + `short_memory`).
- **TablePolicy** = this value table (laya-prior seeded, UCB exploration).
A comparison harness runs BOTH on the same opponents + seeds and reports win-rate and the learning curve over rounds,
so the pivot is DECIDED by evidence, not taste. Shared across both: the eye, the "when" keying, the measurement
engine, the followable oracle, and Qwen-as-coach (adapted per policy). `--policy rules|table|both`.

## What it replaces / keeps
- KEEPS: the eye, when-keying, measurement (per-decision net hp + round W/L), `advice.followable`, `rule_stats`
  sums, Qwen-as-System-2.
- REPLACES (for TablePolicy only): the short-memory TEXT rules as the policy → the table is the policy; text-laya
  becomes a prior. RulePolicy is untouched, so the A/B is honest.

## Sizes / sample budget (sanity)
~15 whens (3×5) × fireball ≈ ~20 occurring cells; followable actions per cell ≈ 8-20 after the stance filter →
a few hundred (when, action) cells. Each needs several samples → low thousands of decisions to fill a character.
Headless on the M3 Ultra this is feasible; laya-prior + Qwen-narrowed exploration cut it further. Per character
(not language-transferable) — a known cost of the pivot.

## The honest risks / open questions (for the reviewers)
1. **Resolution:** is `(range, doing)` coarse enough to collide actions that need finer context (GPT-6's earlier
   warning)? When a cell's best action is genuinely context-dependent, the table averages it away. How to detect
   and split a cell WITHOUT a grid blowup?
2. **Exploration cost:** exploring = deliberately playing a worse move = losing rounds now to learn. How much
   exploration is acceptable in a real fight? UCB vs ε-greedy vs Thompson; decay schedule; does laya-prior let us
   explore less?
3. **Credit window:** one decision interval (net hp) vs round outcome — how to combine a local per-action signal
   with the sparse round win/loss without the delayed-hit / chip problems we catalogued?
4. **laya prior:** worth extracting (how — its move-probs per when?) or just start uniform and lean on Qwen to seed?
   Does the table eventually make text-laya redundant?
5. **Qwen's grip:** how exactly does Qwen narrow exploration and seed priors without becoming the de-facto policy
   (the thing we're avoiding)?
6. **A/B fairness:** same seeds/opponents, same Qwen budget; how to compare a learning curve fairly when one policy
   explores (and so loses on purpose early)?
7. **Is the pivot even worth it** vs just adding ablation to the rule system (Step 2 of the other plan)? Where's the
   crossover — i.e., when does a table beat "a few good rules + ablation"?

## Staging (test-first, when approved)
1. Table core: `V`, `credit`/update, `choose` with exploration — pure, seeded-RNG tests + a toy-bandit convergence
   golden.
2. Policy interface + TablePolicy wired beside RulePolicy; `--policy`.
3. laya-prior seeding.
4. Qwen exploration-guidance (System 2).
5. A/B comparison harness + TUI (two curves).

No code until Fable + GPT-6 have reviewed this and the owner picks a direction.
