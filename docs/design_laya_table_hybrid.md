# Design: laya-text + value-table hybrid (System 1 + System 2)

Status: current design on `feat/table-system1` (2026-10-04). Written as the reference before the
bee/quorum work, so that direction can plug in without re-deriving the architecture.

---

## 1. The components

- **The eye** — the sprite/screen reader. Produces the *situation* each decision: range, the
  opponent's posture/label, whether a fireball is out, who's airborne. **No RAM in the play path**
  (hard gate). This is the only input the player gets.
- **The player (System 1)** — picks the move. Today = **text-laya** (two stages: `cat_laya` picks a
  category, `move_laya` picks a move) reading the situation + short-memory rules. *This is the slot
  the bee/quorum replaces.*
- **The value table** — the empirical memory. A map `(when) -> {action -> Welford stats}` of the net
  hp each move has produced in each situation. Learns from **outcomes**, not from anyone's reasoning.
- **The coach (System 2)** — **Qwen** (remote). After each round it reviews what happened and
  rewrites the **short-memory** rules the player reads. It proposes; it does not touch the table.

## 2. The hybrid decision (how the table affects the game)

Every decision, the player (text-laya) proposes a move. Then the table does ONE of three things for
that situation's cell:

1. **Overwrite** (`source: table`) — if the cell has a move that is *confident-good*
   (`count >= OVERRIDE_TRIES=8` **AND** `mean net-hp > 0`), the table's best such move **replaces**
   the player's pick.
2. **Explore** (`source: table-explore`) — with probability `--explore`, try an *under-sampled* move
   in that cell instead (a move with `count < --explore-tries`, default `MIN_TRIES=20`).
3. **Defer** (`source: laya`) — otherwise the player's pick stands.

So the table **overwrites the player only where it has learned something clearly good.** The
`mean>0` **floor is the key guard**: in a cell where every move the table knows is net-negative, it
does NOT overwrite — it defers to the player's neutral game. That floor is what stopped the
pure-table policy (which had no player) from collapsing into a defensive loss.

Knobs (all in `sf2/system1/value_table.py`, tunable per run):
- `OVERRIDE_TRIES=8` — how much evidence before the table may overwrite (owner: lean on the table,
  text-laya is limited).
- `--explore` (rate) and `--explore-tries` (coverage target) — how much and how deep to explore.
  Once common cells saturate past the coverage target, the *rate* stops firing; raise the *coverage
  target* to re-open exploration (gen-2 lesson).

As the table matures it overwrites more (gen-2/3: ~74%); the player fills the sparse remainder.

## 3. The table is a tree

```
"when"  (branch key = range | opponent-posture | fireball [| opponent-label])
   └── close|standing|0
         ├── lightning_legs   mean +16   n=407    ← THICK leaf (confident, strong)
         ├── throw_F+mp        mean +10   n=39
         ├── block_high        mean  -3   n=12     ← thin + negative
         └── ...
```

- **Branches = `when`**, **leaves = candidate moves**, each leaf = Welford `[n, sum, sumsq]` →
  `mean` net-hp. **Thickness = `n`** = confidence. Thick + high mean = a learned winner.
- **Density by distance (splits).** A branch *splits into finer sub-branches by opponent posture*
  when the posture flips the best move (two-directional Welch CI reversal). Only close/mid ranges
  split; far stays coarse (few options). So the tree grows deeper **only where detail pays off**.
- **Parent fallback.** A thin split-child leaf falls back to the parent branch's accumulated data,
  so a sparse sub-branch is never starved.

## 4. The learning loop

Per decision: credit the move actually played (by player, overwrite, or explore) with its net hp
(`dealt - taken`, delayed-hit-corrected upstream) into its `(when, action)` cell. Per round: Qwen
rotates the short memory. The table is credited **regardless of who chose the move**.

**Generational, parallel.** The box (M3 Ultra) and the Qwen server are under-used by one game, so we
fan out K independent workers (each own port, own playbook dir, own evolving short memory), all
seeded from the current best table. Each worker's Qwen reviews ITS OWN games — same prompt template,
different content — so the K workers explore K divergent rule-sets (variation). Then:

- **Pool** their rounds for a tight win-rate estimate.
- **Merge** their tables into the next-generation table. Welford stats are additive, so cells +
  shadow sum and splits union (`value_table.merge`). **Critical:** workers share a seed, so the
  merge must pass `shared=seed` to subtract it `(K-1)` times — otherwise the seed is counted once
  per worker and inflates every generation (the bug caught 2026-10-04).

Results so far (vs Ryu, round win-rate, honest/re-scored): gen-1 pooled **33.7%** → gen-2 **49.8%**
(~parity) → gen-3 targeting 60-65%, end goal ~75%. The gain compounds from a **better table**, not
from exploration (exploration raises the *ceiling* for later generations).

## 5. The invariant that makes everything swappable

**The learner (table) is decoupled from the player.** The table keys only on `(situation, move) ->
outcome`. It is **player-agnostic and coach-agnostic** — it does not know or care whether text-laya,
a quorum of bees, or Qwen's rule caused the move. It records what the move *did*.

Consequences:
- You can **swap the player** (single text-laya → quorum of specialized bees) **without changing how
  the table learns.** The table's data model, credit, splits, and merge are untouched.
- The player's only effect on learning is the **sampling distribution** — which `(when, move)` pairs
  get tried, hence which leaves thicken. A better/more-diverse player builds a better-covered tree
  faster; it does not change what a filled leaf *means*.
- Keep this invariant. Anything that makes the table depend on *who* proposed (e.g. per-bee cells)
  breaks the clean "memory = outcomes" model and should be a deliberate, separate decision.

---

## 6. Forward: the bee / quorum player (preparation)

The quorum replaces the **player** slot (§1). Proposed variation: **specialized laya-text bees** —
e.g. 3 bees biased toward **attack**, **movement**, **defense** — that vote; a quorum commits the
move. (Attack/movement/defense ≈ the `cat_laya` categories, so a "bee" ≈ a category-biased proposer.)

**How it affects the learning (the table):**

1. **It changes sampling, not the learner.** By §5, the table still learns `(when, move) -> net hp`
   the same way. The bees change *which moves get proposed/played*, hence which leaves get sampled.
2. **Better category coverage → a more complete tree.** A single text-laya may under-propose some
   categories, leaving those leaves thin. Three specialists each push their category, so each
   `when` branch gets its attack AND movement AND defense leaves sampled → **thicker, better-covered
   branches, faster** — exactly the coverage exploration now has to chase by luck.
3. **Specialists help EARLY and in sparse branches, then recede.** As the table thickens it
   overwrites more (the laya→table handoff we already see, ~74%). So the bees matter most while the
   tree is thin and in under-explored situations; the mature table increasingly dictates play. Same
   shape as the current player→table handoff.
4. **Two design forks that DO affect learning quality:**
   - **Table vs quorum precedence.** Does the table overwrite the quorum's consensus (recommended —
     preserves §5 and the laya→table handoff), or is the table *one more voter*? Voting dilutes the
     table's influence on what's played, changing the explore/exploit balance; overwriting keeps the
     table as the empirical authority.
   - **The quorum rule.** A *consensus* rule (all bees must agree) tends to play the "safe middle"
     move no specialist strongly wants — mediocre play, and the table then learns about mediocre
     moves. A *plurality / let-minority-through* rule preserves diversity and samples the bold moves
     that actually fill the tree. The quorum rule is effectively an exploration policy — tune it as
     one.
5. **Independence caveat.** If the bees are one base model with different prompts, their votes are
   correlated (biased samples of the same model), not independent opinions — real but limited
   diversity. Different checkpoints give more independent votes at higher cost.

**Invariants to preserve when wiring the quorum in:**
- The hard gate (no RAM/table imports in the play path).
- Credit stays `(when, move) -> net hp`, delayed-hit-corrected, **player-agnostic** (do not add
  per-bee cells unless deliberately decided).
- Parallel merge still uses `shared=seed` dedup.
- Keep the `mean>0` overwrite floor (the anti-defensive-collapse guard) whatever the player is.

Both directions branch off `feat/table-system1`, so converging is a normal merge.

---

## 7. Phase 2: Qwen-self-guided learning (hypothesis + validation gate)

**Hypothesis (owner):** given enough signal about the learning state, Qwen can *self-guide the
study* — set the explore/exploit ratio and depth, pick which branches/categories are under-explored
and need attention, emphasize a bee specialty, decide when to re-seed/merge or change opponent —
closing the self-sustaining loop of §6. The bees are the hands, the table is the memory, Qwen
becomes the *thinking that sets the learning schedule*.

**Also under consideration:** short play bursts with an **adjustable learning ratio** re-set between
bursts (a schedule, not a constant: explore-heavy while the tree is thin → exploit-heavy as it
thickens). Short bursts give a noisy win-rate, so drive the ratio off **coverage** (how many
branches are still thin — stable even over a short burst), not off win-rate.

**Validation gate — prove Qwen's judgment BEFORE wiring it into the bee/quorum plan.** Qwen is a
product-LLM in a consequential control loop, so it is graded against ground truth, not trusted for
sounding reasonable. The table gives us a **mechanical oracle**: per-branch coverage (`n`), per-leaf
mean/variance, the genuinely thinnest branches, category balance (from the table); win/hp trend,
source split, bee-agreement rate (from traces) — all computable, so the *right* control call is
computable too.

- **Test A (cheap, offline, first — "can Qwen READ the signal?").** Feed Qwen the learning-state
  summary from the existing gen-1/2/3 snapshots; ask the control questions (explore vs exploit +
  ratio, which branches/categories under-explored, which bee needed). Grade MECHANICALLY: did it
  name the actually-thinnest branches? did its explore/exploit call match what coverage says? did it
  avoid the obviously-bad call (exploit-hard while half the tree is unsampled)? Per the product-LLM
  rule: ≥10 permutations (rephrase/reorder/noise the signal) × repeats, score the matrix
  (consistent-right = fits the bill; flip-flops = ambiguous signal/prompt, fix or keep mechanical;
  consistent-wrong = keep mechanical control).
- **Test B (expensive, second — "does Qwen's control IMPROVE outcomes?").** Only if A passes. A/B a
  few generations under Qwen-set ratios vs the mechanical schedule (same seed table); pooled
  win-rate + coverage. Qwen-guided must match or beat mechanical.

**Integrate into the bee/quorum plan only if A (and ideally B) pass.** Qwen stays the *bounded*
meta-controller (§6 guardrails: ratios clamped, `mean>0` floor kept, merge `shared=seed` kept, hard
gate kept, stop conditions), and the table stays **Qwen-agnostic** — Qwen sets *how fast / where* to
learn, never *what a filled leaf means*.

Status: queued to run after gen-3 completes (2026-10-04).
