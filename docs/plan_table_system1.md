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

---

# PLAN OF RECORD (after Fable + GPT-6 review, 2026-10-04)

**Decision: GO — as a bounded EXPERIMENT beside the rules, credit-fix FIRST. Not a blind replacement.** Promote the
table only if it beats the frozen rule system in the A/B. Both reviewers converged on this; the pieces below fold
in their grounded corrections.

## Both reviewers agreed
- Experiment, not replacement: build the table, A/B vs frozen RulePolicy, promote on evidence.
- **The per-decision CREDIT is contaminated by delayed hits — fix it FIRST (the gate). It helps the rule system too,
  and is likely a real reason chun4 barely learned.**
- Keep **Qwen OUT of experiment 1** (narrowing exploration / injecting optimism = becoming the policy).
- Pluggable policy + `--policy rules|table`; reuse `scripts/ab_memory.py`'s pre-registered paired-CI test for the A/B.

## Grounded corrections to the spec (Fable, measured on chun3/chun4)
- **No laya prior exists** — `cat_v3` outputs `block=1.000` in every no-advice cell; `move_probs` only ranks an
  already-chosen category's 3-4 options. Seeding from laya = uniform. **Drop the laya-prior (spec rec b).** Optional
  optimistic init from the offline `lessons/value_oracle_v1.json` (12 cells, needs a name map; no-RAM-safe).
- **Aliasing is his SUB-STATE, not distance.** Measured collisions: `close/jumping` empty-jump (-3.4) vs jump-attack
  (-22.6); `close/standing` his-takeoff (+15) vs his-walk (+4.8). dx-halves within `mid` barely matter. **Split a cell
  by `his_label`** (already in the decision record, no retrain) via a shadow tally; this is a key the table can use
  but the text-rule grammar cannot — a real argument FOR the table.
- **Reuse directly:** `advice.followable` / `available_moves` (action set), `screen_evidence.decision_row` net-hp
  (credit), `rule_stats` Welford/Welch (the table's credit + split test are a ~20-line copy).
- **Sample cost (the one disagreement, reconciled):** busy cells (60-280 decisions / 27 rounds) separate their top
  actions in ~1-2 headless hours (Fable); rare cells stay sparse and fall back to parent/prior (GPT-6's caution). So
  the table earns its keep on the COMMON whens, not the rare ones.
- `--no-score` loses the KO window's damage (`nxt=None -> 0`); log `hud.health` per read so the last window credits
  the KO. (Small, separate from reattribution.)

## STAGE 1 — the credit fix (the gate; do first; helps rules AND table)
**Bug (measured):** `decision_row` credits each decision with the health-bar drop until the next decision. When she
lands a hit at t, he enters hit-stun and his bar drains during t+1's window, so the move at t+1 (often `block_high`
while `his_label=="hit"`) is credited with the prior move's damage. chun4: `block_high` in (mid,jumping) netted
**+19.7** this way.
**Fix (conserving):** in `round_evidence`, after building rows, reattribute: a decision whose `moment.his_label ==
"hit"` did not cause its window's `dealt` -> move that `dealt` to the nearest EARLIER decision whose `his_label !=
"hit"` (the action that started the hit). If none (round start), leave it. Add `his_label` to the row (also used by
the table split). Total `dealt` conserved; round summary unchanged.
**Red test (already on disk, pre-registered):** `tests/test_credit_reattribution.py` — block's +40 moves to the
anti-air; honest s.mk keeps its credit; round-start hit-stun left in place; totals conserved.
**Also (optional, same stage):** log `hud.health` in `reads.jsonl` so the KO window isn't lost under `--no-score`.

## STAGE 2 — `sf2/system1/value_table.py` (pure, test-first)
- `blank()`, `key(moment, depth) -> when` (base `(range, doing, fireball)`; `depth[cell]="his_label"` splits one cell),
  `credit(table, drows) -> table` (Welford net-hp per `(when, action)` + a `his_label` shadow tally + the split rule),
  `choose(table, when, actions, rng) -> (action, explored)` (ε-greedy: force-cover arms with `n<MIN_TRIES` first,
  then ε₀·MIN_TRIES/(MIN_TRIES+n_cell) exploration, else argmax).
- Split rule: split a cell by `his_label` only when two sub-labels' top action differs with Welch-CI separation and
  both have `n>=MIN_TRIES` (reuse `rule_stats._var`). One binary split per cell. Parent stays fallback for sparse children.
- Tests: Welford == `rule_stats` sums on a chun4 fixture; the delayed-hit fixture; split fires on a crafted two-label
  case and not a homogeneous one; seeded ε-greedy covers every arm within K draws; a toy 3-arm convergence golden.

## DENSITY BY DISTANCE (owner 2026-10-04)
Table detail scales with proximity: the closer, the more interacting options worth distinguishing; far away she has
few options. Implemented two ways, both already in `value_table.py`: (1) the ACTION COUNT is distance-scaled for free
by `advice.available_moves` (far = fewer moves, no throws); (2) a cell may SPLIT by his_label only when close/mid
(`SPLIT_RANGES`) — far cells stay coarse. So the table is dense up close and sparse far, by construction.

## STAGE 3 — pluggable policy + A/B harness (PHASED, owner 2026-10-04)
Phase it: **3a** = make `--policy table` actually PLAY and learn (pluggable decide, table persistence, Qwen off);
**3b** = the A/B harness (table vs frozen rules), run incrementally (start one opponent, then expand).
- `loop_runner.play_round(..., decide: Optional[Callable[[Moment], Dict]] = None)` (default = today's closure);
  `TablePolicy.decide(m) -> {action, when, values, explored}`; add `when/values/explored` to `DECISION_KEYS`.
- `play_loop_screen --policy rules|table`, `--save-table/--carry-table` (JSON, like the registry); for `table`,
  `update()` skips Qwen/short_memory and runs `credit` once per round.
- A/B: both arms same opponent + `--seed` (same start-delays only; trajectories diverge — say so), **Qwen OFF in both**
  (RulePolicy = frozen `--carry` of a chosen registry). Reuse `ab_memory.py`'s paired-CI verdict on per-round hp over
  the LAST N rounds; report win-rate + hp curves over round index. Optional 3rd arm = rules+randomised ablation
  (GPT-6's baseline) once ablation exists.
- **Acceptance (mechanical):** on the `stunned` cells (100% `block_high` today) the table stops blocking within ~10
  rounds (check decisions.jsonl); and the table's last-N win-rate CI beats frozen rules.

## STAGE 4 — Qwen as System 2 (only if Stage 3 is positive)
Bounded: Qwen may rank the under-sampled arms of ONE bleeding cell to set which are force-covered first, and write
prose. Never writes values/counts/legal-masks, never the exploit choice. Honest-keeping test: replace Qwen's ranking
with a random permutation -> the converged table is identical within CI.

## REUSE from the shelved value/table subsystem (assessed 2026-10-04)
The project already built a value table once (then shelved it). What carries over:
- **REUSE AS-IS:** `sf2/eval/stats.py` (`paired`/`run_level`/`pooled` opponent-as-unit bootstrap/`verdict`) for the
  Stage-3 A/B gate; `advice.followable`/`available_moves` (action set); `screen_evidence.decision_row` net-hp +
  `rule_stats` Welford/Welch (credit + split); `scripts/value_inplay.py::_throw_close` (a throw-rate-per-close-
  decision diagnostic — our degeneracy guard, below).
- **ADAPT (optional):** `lessons/value_oracle_v1.json` as the Stage-2 optimistic SEED — `json.load` the 12 chunli
  cells DIRECTLY (schema `{"cell":[me,range,opp_attacking,opp_airborne],"values":{move:net_hp}}`; values are shrunk
  mean net hp). **Never import `sf2/data/value_oracle.py` (hard-gate TABLE).** Caveats: means only (counts dropped →
  pick your own pseudo-count), needs an old→two-stage move-name map, and it can only seed the BASE `(range, coarse-
  doing)` — NOT fireball or his_label. `value_inplay.py::report()/commands()` A/B fan-out shape can be copied but
  retargeted to `loop_runner --policy table`. `value_oracle.build()`'s `dealt-taken` + `n/(n+SHRINK)` shrink is a
  REFERENCE for the table's credit (reimplement gate-clean in `value_table.py`, don't import).
- **IGNORE:** `system1._by_table` (fixed argmax, no exploration, gate-forbidden), `play_system1 --oracle` (RAM
  harness), `sf2/data/value.py` + `value_data.py` (the VLM value-head/dataset — a different, killed fine-tune),
  `junk/docs/plan_laya_vision_value.md` + `docs/reviews/2026-09-30_dr_fable_lv_value.md` (the killed fine-tune).

## The prior in-play result — corrected, and its lesson (anti-degeneracy)
The "-88 hp, lost badly" memory is UNCONFIRMED (those logs are empty). The on-disk report
(`logs/lv_inplay_oracle_report.json`) shows the fixed table WON the hp A/B (+81.4 [+42.7,+121.9] over the eye) — **by
throw-spamming up close** (0.46-0.89 of close decisions). A FIXED table + argmax + no exploration + coarse keys
collapses to ONE degenerate exploit that "wins" per-decision net-hp without playing well. The new plan guards:
exploration (Stage 2), the `his_label` split (close/attacking shows throw **-5.18** → demoted), the credit fix (the
oracle's numbers were on the un-reattributed, delayed-hit-contaminated signal).

**Owner 2026-10-04: winning by a dominant move IS a legitimate win** ("in this game one attack may beat anyone, and
that's not wrong if it happens"). So the one-move-share (`_throw_close`) is a **DIAGNOSTIC we report, NOT an A/B
failure.** The A/B is decided by win-rate/hp alone; if the table wins by a dominant move, it wins. Exploration and
the `his_label` split still earn their keep: they find the BEST exploit, and they ADAPT when a given cell's dominant
move stops working (e.g. throw demoted once he attacks). text-laya-as-prior may only speed convergence; it is not
required for the table to be right.

## Knobs (2): `MIN_TRIES` (reused), `ε₀` (new, ~0.3). Nothing else.
## Product decision deferred to the A/B: the table makes text-laya redundant as the DECIDER (it stays as the
RulePolicy arm). The owner's "day vs night": an explicit empirical table vs a generalizing model — they should
converge where both have data; the A/B measures final strength AND rounds-to-converge (hypothesis: laya faster to
decent play, table self-learns and uses `his_label` laya can't read).
