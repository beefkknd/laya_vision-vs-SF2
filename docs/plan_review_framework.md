# Learning / review framework (grounded, simple)

Owner-set 2026-10-03. Replaces "propose rules and cross fingers" with a review FLOW where **deterministic
stats-driven gates decide keep/drop/retone/graduate, and Qwen only PROPOSES** (shaped by the stats). Synthesised
from a Fable review (grounded in the repo + playbooks/chun3) and an independent codex review, after both corrected
the first diagnosis.

## Corrections to the earlier diagnosis (measured, not guessed)
- **Scoring is NOT fed zeros.** Per-decision `dealt`/`taken` come from the drawn health bars
  (`sf2/system2/screen_evidence.py` `decision_row`, from `moment.my_life/his_life`), independent of `--no-score`;
  the replay only supplies the round-end target + frame-truth result. No need to turn scoring back on.
- **cat_laya is NOT the lever.** The same prompt (situation + applicable advice) feeds both stages, and cat_laya
  already picks the applicable rule's category. Measured in chun3: **459 of 460 blocks happen in cells where NO
  rule applies**; every covered cell has ~0 blocks. The turtle is a COVERAGE gap, not a category policy to steer.

## Why she never learned (the real causes, measured in chun3)
1. **Unfollowable rules.** At close range only `cl.*` normals are offered (`advice.py` `stance_of`/`moves_in_stance`),
   so any `s.* up close` rule can never fire. 3 of 10 chun3 lines + one explore-pool line are dead this way.
2. **Shadowing.** `chosen_moves` returns the hard rule before soft (`advice.py`); a `always` rule owns its cell, so
   stacked `use_more` rules never fire, never earn evidence, then get evicted as "idle/oldest".
3. **Evidence resets every block.** `play_career` spawns a fresh `play_loop_screen` per block with `all_rows=[]`;
   a trying rule's evidence is recomputed block-local, so `MIN_TRIES=20` (both sides) is never reached → nothing
   graduates → endless churn.
4. **Coverage-gap turtle.** She blocks in the uncovered cells (he jumps / attacks / is stunned).

## The one unifying rule (folds in shadowing, the per-cell cap, coverage, and cat_laya)
**Keep exactly ONE followable positive rule per occurring cell** `(range, doing)` (soft by default; `always` only
as a retone when the model under-follows), plus at most one `avoid`. This single invariant:
- kills **shadowing** (never two positives fighting in a cell),
- keeps **Qwen's view small** (one line per cell),
- makes the turtle a **coverage** target (fill the uncovered occurring cells),
- and needs **no cat_laya steering** — measured in chun3, once only applicable lines are shown, cat_laya already
  picks the applicable positive's category 100% of the time (473/473 … 21/21 at 1–5 applicable lines) and blocks
  only when nothing applies. So the number that matters is **positives per cell**, not the global line cap, and the
  10-line cap does not backfire via the B1 distractor (the `prompt_lines` filter already neutralises it).

## The "when" — vocabulary over the grid (grounded 2026-10-03; Fable + codex)
A "when" is the NAME of an occurring cell of the `(range, doing)` grid — **cell == when**. Two reviewers (one
grounded in the code + chun3 numbers, one independent) agreed: the "when" is the SAME key the model's grammar
already uses, described more naturally; it is NOT a new situation model. What it legitimately adds is a
**damage-first ORDERING** for the Coach and honest caveats — docs + prompt-ordering, **zero retrain, near-zero code**.
- The only decision keys text-laya can read are the sentence words `range_of(dx)` × `doing` ∈ {jumping, crouching,
  attacking, standing, stunned} + a fixed "A fireball is coming." clause (`advice.py` `situation_text`,
  `Lesson.applies`). **Bars were never a rule key.** So "(range, stance, bars, fireball) → when" removes nothing.
- Each when's OUTCOME is already how credit works: `screen_evidence.decision_row` credits each decision with the
  health deltas until the NEXT decision; `condition_evidence` scores the rule's move vs the rest in that condition.
  "Score at the when's outcome, not over frames" is the STATUS QUO (decisions ≠ frames: ~40 frames apart median).

**Minimal viable when set (7 whens, one modifier = range, no new knobs):**
1. he is stunned (range-agnostic; split by range only on bloat) — 88 chun3 decisions, **100% blocked, 0 covered**
2. he jumps mid/far (anti-air) · 3. he jumps up close (jump-in landing)
4. he attacks up close · 5. he attacks at mid (the WORST cell: 16.3 hp taken/decision) · 6. he attacks far away
   (= zoning/fireball; block; the fireball bool is a **stats sub-key**, not a decision key)
7. he stands (NEUTRAL — keep it: 43% of decisions; mid/standing bleeds 10.5 hp/dec, the footsies cell where she
   dies if "event-only" dropped it). Range-agnostic default (`use more s.mk`) lives here; split mid-standing first.
Drop "air" (his air == jumping already; she is never airborne at a decision — `can_act` forbids it) and bars.

**NOT detectable — do not promise (measured):**
- "he's attacking" is ONE label for the whole attack animation — no startup/active/recovery, no frame timing; 221
  of 255 attacking decisions were mid-animation ("continuing"), so it reads as "somewhere in an attack" → a block/
  poke bucket, not a counter-the-recovery timing.
- Fireball **distance / time-to-arrival / block-vs-jump timing**: projectiles expose position only (no velocity/
  time), and the sentence carries no distance; distance is computable OFFLINE for stats but keying a decision on it
  needs a new sentence = retrain. The fireball bool is also flaky near arrival (14 clean hits landed in windows the
  decision read fireball=False). So **(far, attacking) is the robust zoning key; fireball is a stats modifier**.
- Grammar gap: the Coach claim path can't even express a fireball condition today (`claim_of` drops it); text-laya
  WAS trained on fireball lines, so adding a `fireball` claim field later is in-distribution (no retrain) — v2 only.

**Credit rule:** a decision owns the interval to the next decision (`decision_row`), scored by the NET hp amount
(not `taken>0` — blocking a fireball always costs 2-8 chip). Windows are disjoint, so whens never overlap at credit
time. One gap: a KO hit in the last window is lost without the replay — fix by logging `hud.health` in
`reads.jsonl` (tiny logging change, no prompt change). No new knob.

Rules group by when (one followable positive per when, + ≤1 avoid); a BLOATED when is sub-split by range or pruned.
The Coach's coverage target is ordered by **damage debt** (n × hp-taken/decision) among uncovered cells, event
cells (stunned, attacking, jumping) first. Qwen reviews per-when.

## The framework (simple; deterministic gates + Qwen-proposes-only)

### (a) Stats — pooled, from files already written (NO re-scoring)
One `entry["stats"]` dict per rule, carried in the registry entry like `rounds` (so it pools across blocks and
survives restarts), updated after every round by a pure `tally()` from the round's decisions
(`screen_evidence.read_decisions`) + the round summary. Per rule:
- `applicable` (in `prompt_lines`), `enforceable` (applicable AND the move is actually offered in that decision's
  stance — catches the dead rules), `followed` (enforceable AND action matches; `avoid`: action avoided; only when
  `two_stage` `rule != "default"`), `shadowed` (enforceable but another rule owned the cell).
- effect sums `n_mine,sum_mine,sumsq_mine,n_rest,sum_rest,sumsq_rest` (Welford) over the rule's condition — the SAME
  quantity `condition_evidence` computes, but accumulated so `MIN_TRIES` becomes reachable.
- round correlation: `rounds_fired,wins_fired,hp_fired` vs `rounds_idle,wins_idle,hp_idle`.
- ablation (see e): `rounds_ablated,wins_ablated,hp_ablated`.
Per cell `(range, doing)`: `n, blocks, covered` — the coverage table that drives the Coach.

### (b) Review — each round, inside the loop. Order: tally → gates → survival swap → Qwen (proposals only)
- **Gate A — unenforceable/dead:** `rounds≥SWAP_AFTER` and `enforceable==0` while `applicable>0` → drop ("can't be
  played at its range"); `applicable==0` after `2*SWAP_AFTER` rounds → drop (cell never occurs; log as Coach debt).
- **Gate B — shadowed:** `enforceable≥MIN_TRIES` and `shadowed/enforceable>0.5` → drop (another rule owns the cell).
- **Gate C — not followed:** `enforceable≥MIN_TRIES` and `followed/enforceable<0.5` → retone `use_more`→`always`
  once (reuse `_retone`); if already `always` → drop + append to a text-laya retrain-queue log. (`avoid` lines are
  model nudges — judged only by round-correlation, not Gate C.)
- **Gate D — effect:** pooled scorer `cls==better` (its direction) → `kept`; `cls==worse` → drop; else stays trying.
  This is the existing `_graduate` + a symmetric drop, now firing because the sums persist.
- **Survival swap:** unchanged (`SM.step`), but Gates A–C already removed the junk, so the victim is a genuinely
  weak rule, not "the oldest".

### (c) Coach (Qwen) — proposals only, shaped ask
Give Qwen: the coverage table (top occurring cells with `covered=false` + their block counts), the per-rule stats
table (applicable/enforceable/followed/net/fired-round win-rate, one line each), and the ENFORCEABLE move list for
the target cell. Ask for ONE rule for the most frequent uncovered cell while losing; when all common cells are
covered, ask for a retone or a replacement of the single weakest `unclear` rule. `coach_filter` gains a
range-vs-stance check so Qwen cannot propose an unfollowable rule. Qwen never gates keep/drop/graduate. The
`refused` round-trip and the separate Scout prose call can be dropped (the digest is code-computed).

### (c2) Categories / the turtle
No per-category defaults (the model's trained block is the safety floor). cat_laya already follows the applicable
rule's category, so the fix is COVERAGE: the Coach targets the most frequent uncovered cell (today: mid/jumping,
far/jumping, far/attacking, the stunned cells), each with grounded enforceable moves (lightning_legs,
spinning_bird_kick, s.hk, walk_forward). cat_laya does not need to enter the loop.

### (d) Consistency — pure `lint(reg, cells)` after each change
round-trips (`claim_of→render==line`); enforceable at its range; at most one positive per cell (no shadow); no
`avoid X` overlapping `use_more/always X`; coverage report (occurring cells with no enforceable positive = the
Coach's target list); cap/order. Deterministic; Qwen only reads the report.

### (e) "Learned / done" — causal, not bare win-rate (two options; prefer the simpler)
Trigger on `win_target` (0.60) as today, then require causal evidence. Two ways, both cheap:
- **PREFERRED — frozen confirmation block** (simplest, no new per-round machinery): when a block first hits the
  target, replay ONE more block with the memory FROZEN (no Qwen, no swaps — a `--frozen` run) and require
  win-rate ≥ target again. This is `promotion.decide`'s dev-then-held-out idea at the career level: a win that only
  shows up while still churning is winner's curse. Plus an **attribution** check: in the winning rounds, ≥~70% of
  decisions had an applicable followable rule (she won WITH the memory, not the block fallback), and every `kept`
  line has `won_fired/rounds_fired ≥ won_idle/rounds_idle`.
- **Alternative — ablation rounds:** every `ABLATE_EVERY`-th round drop one `kept` line and tally its **lift** =
  win-rate(in play) − win-rate(ablated); a kept line with lift ≤ 0 is demoted. Causal at zero extra games, but adds
  per-round bookkeeping. Use only if the frozen-confirmation block proves too coarse.
codex's caution kept either way: "followed + won" is diagnostic, not causal; the frozen block (or the lift) is the gate.

### (f) Knobs (5) and deletes
Knobs: `SWAP_AFTER=2`, `MAX_LINES=10`, `MIN_TRIES` (one number, now POOLED across blocks so it is reachable),
`POSITIVES_PER_CELL=1` (the unifying rule above), `win_target=0.60` (+ the one frozen confirmation block). (Follow
floor 0.5 = a fixed majority, not a knob. `ABLATE_EVERY` only if the ablation alternative is chosen.)
Delete / stop using: per-block evidence recomputation (`evidence:{}` on trying entries); the "oldest idle" eviction
heuristic (gates replace it); the hand-written `EXPLORE_POOL` (replaced by the coverage picker — generate a
followable move for the target cell); `lessons.review/propose/stop/losing_since/_promotable/revive` + constants
`TEST_GAMES/PROMOTE_GAMES/STICK_WINDOW/STOP_DROP/REVIVE_*` (already off the live path — delete with their tests; keep
`render/key/_covers/_overlap/condition_evidence`); the `refused` plumbing, the Scout prose call, and the
escalate/consolidate prose modes (one cell-targeted ask replaces them); the stance-inert explore-pool line.

## Where the two reviewers differed
codex wanted full A/B experiment batches (20 randomized round pairs) per change for causality — rigorous but heavy.
Fable gets causality cheaper via in-situ ablation (no extra games). We take the ablation (simpler) + codex's caution
(don't claim "learned" from follow+win correlation alone).

## Implementation shape (test-first, small files)
- NEW `sf2/system2/rule_stats.py` (pure): `tally`, `evidence_from_stats` (== `condition_evidence` on the same rows),
  `gates(reg)`, `lint(reg, cells)`, `coverage(decisions)`.
- `short_memory.step` takes the round's decisions, runs tally+gates before the swap; victim = lowest pooled score.
- `play_loop_screen.run_loop/update`: pass decisions into `step`; ablation line-filter every `ABLATE_EVERY`-th round;
  stats saved with the registry (already JSON, carried by `--save-registry/--carry`).
- `character_prompt.coach_filter` adds the range-vs-stance check; the digest prints the stats + uncovered-cells table.
- `play_career` DONE criterion reads kept-rule lift + `block_winrate` + "no change this block".
- Tests (red first): stance-inert line dropped after SWAP_AFTER (fixture from chun3 round_03); shadowed soft dropped;
  stats pool across two step sequences with a save/load between; `evidence_from_stats == condition_evidence`; ablation
  records only to the ablated rule; lint flags the 3 chun3 inert lines + the mid/standing hard+soft stack; done-false
  when any kept lift ≤ 0.

## Highest-leverage first step
The **enforceability lint** (stop the Coach proposing rules text-laya can't follow + drop the dead ones) and
**pooled stats** (so rules can finally graduate) are the two that unblock everything else. Do them first.
