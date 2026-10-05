# Learning-acceleration backlog (laya-text + value-table + Qwen)

A standing list of ways to make the self-learning loop learn FASTER / HIGHER, and the open questions
to settle by experiment. Companion to `docs/design_laya_table_hybrid.md` and `docs/bee_quorum_notes.md`.
Written 2026-10-04 on `feat/table-system1`; **v2 after independent review by GPT6 (codex) and Fable** —
the two engines converged on the fixes below, so this version supersedes the first draft's framing.

## RESULTS (2026-10-05) — §0 run; two findings that reframe everything

1. **§0a RETRACTED: the credit scalar is NOT the ceiling.** The first draft's "myopia r=0.74→0.50" was an
   arithmetic artifact (the k-step window included the decision's own net; 1/√(k+1)). Recomputed
   FUTURE-ONLY: r(immediate net, next-k net) = **−0.008 / 0.081 / 0.080 ≈ 0** — the immediate net is
   uncorrelated with what follows, so n-step credit adds ~uncorrelated variance, not signal. Telescoping
   r=1.0 was a tautology (same end-bars feed both sides). **B1 (n-step) was then BUILT + frozen-eval'd:
   it TIES immediate (71.9% vs 70.5%, diff +1.4, 95% CI [−3.8,+6.6] — not significant), exactly as the
   corrected diagnostic predicted.** Reward-reshaping is not the lever. [independent review caught this
   before we shipped a noise-adding "fix" — the owner-mandated code-review process paying off on run 1.]
2. **The "~60% plateau" was a MEASUREMENT artifact — the real greedy policy is ~71%.** Same `ab_imm`
   table: training WR (explore 0.1, Qwen rotating) = 59.3%; **FROZEN eval (explore 0, table+memory
   frozen) = 70.5%** — an 11-pt gap from the exploration tax + memory churn + within-run warmup baked
   into the training number. Both independently-trained arms land ~71% (tight). So every generational
   number above is TRAINING-WR, pessimistic by ~10 pts; the policy is already near the 75% aspiration.
   **Lesson: always report the §0b FROZEN eval, never the training WR.**

## Context — the generational numbers are TRAINING, not a learning curve

| gen | train win-rate | explore | note |
|-----|----------------|---------|------|
| 1 | 33.7% | — | seed thin |
| 2 | 49.8% | ~9% | exploit a richer seed |
| 3 | 38.8% | ~21% | EXPLORE round — dipped *because of the explore rate*, not regression |
| 4 | 58.9% | ~1% | exploit the 86k table |
| 5 | (running) | ~1% | compounding test |

**These are training win-rates at different explore rates, so the curve is NOT a clean learning curve**
(gen-3's dip is mostly the 21% explore tax). Any claim about learning requires the EVAL protocol (§0).
The headline framing from the first draft — "explore builds / exploit cashes in" and "the credit
signal is the ceiling" — is a **hypothesis, not established**; §0 is how we'd actually test it.

---

## §0. PREREQUISITES — nothing below is measurable without these (do FIRST)

### 0a. Diagnostic (one script, ~1 hour) — is the credit signal even the problem?
Per-decision net-hp (`dealt − taken`) summed over a round **telescopes to the final hp difference** —
i.e. to the round result, except on time-outs. So the "safety bias" is almost certainly NOT
"hp ≠ winning"; it is **credit-assignment myopia** (a setup move eats the taken-hits, the finisher
banks the dealt). Before building any reward change, measure:
- per round: `Σ net-hp` vs round result (should match closely → confirms telescoping);
- per decision: correlation of immediate net vs k-step net (how myopic is single-window credit?);
- fraction of rounds ending in **time-out** (the real mechanism by which "do nothing" goes unpunished).

If `Σnet ≈ outcome`, the ceiling is NOT the reward scalar — look to coverage, state aliasing, the eye,
or the action space (§E) instead.

### 0b. Eval protocol — separate TRAINING win-rate from EVAL win-rate
Every experiment below is currently unpowered/unspecified. Fix once, reuse everywhere:
- **Frozen eval**: explore=0, frozen short-memory, the learner's choices only, held-out emulator seeds.
- **Power**: report Wilson CIs; a 5-pt lift needs ~800 eval rounds/arm at 95%. Gen-to-gen swings are
  ±20 pts, so "a few generations/rounds" proves nothing.
- **Independence**: paired branching from the SAME checkpoint/seed per arm; ≥3 seeds; workers sharing
  ancestry are NOT independent replications (the 58.9% has no CI yet).
- **Freeze Qwen memory** during table experiments (or use identical memory trajectories across arms) —
  else the credit/exploration effect is confounded with memory drift.

---

## §A. Two axes of "speed"
- **Sample-efficiency** — each game teaches more. The big wins live here.
- **Throughput** — more games/hour. Cheap, linear, but verify the bottleneck (§B6).
Given fixed physical resources, prioritize sample-efficiency.

---

## §B. Levers, re-ranked after review

### B1. Better credit = n-step / λ-discounted DENSE credit (HIGHEST LEVERAGE)  [proposed]
NOT a broadcast round signal (that is baseline-free REINFORCE — pure noise at n≈8, every decision in a
won round gets the same bonus regardless of contribution). Instead, credit decision *t* with the
discounted sum of the **dense hp signal** over the next few decisions (an eligibility trace):
```
credit(t) = Σ_{k≥0} γ^k · net_{t+k}      # γ∈(0,1), horizon a few decisions
```
- WHY: directly fixes "setup moves get no credit" (the actual myopia), **keeps hp units** (so the
  override floor still means something), and is **low-variance** (unlike a sparse ±1 broadcast).
- `β = 1` on `taken` — do NOT discount taken-hits (β<1 rewards losing trades and breaks telescoping).
- Aggression, if wanted, via **potential-based shaping** `Φ = hp_diff · f(time_left)` (provably
  policy-invariant) — this, not β, is how you punish passivity, and it finally models **time-outs**.
- A round term, if any, is only a **baselined tie-breaker**: `(round_result − running_WR)`, undiscounted
  (the decisive decisions are EARLY; the KO already gets full dense credit, so γ^steps_from_end is
  backwards). **Drop `match_signal`** (deterministic function of round results → noise).
- CHANGE: `net_of()`/`credit` is enough for the n-step version; eligibility traces proper (SARSA(λ))
  are a LEARNER change. **Version the reward schema and RETAIN TRAJECTORIES** — old scalar cells cannot
  reconstruct a new reward.
- CHEAPEST EXPERIMENT: immediate-hp vs n-step vs terminal-only Monte-Carlo credit, exploration + state
  rep + memory held fixed (§0b), measured on eval WR + sample-efficiency.

### B2. Exploration = Thompson sampling (NOT UCB)  [proposed]  (merges the old B5 prior)
Per cell, draw each move's mean from `N(mean, var/n)` using the stored `[n,sum,sumsq]`, pick the max of
the draws for the EXPLORE branch:
- WHY over UCB: UCB1 is **deterministic** → 8 cloned workers pick the same under-sampled move in the
  same cell (duplicated exploration); Thompson **decorrelates workers for free**, has **no `c` to tune**
  (UCB's `c` is in hp units, not "self-calibrating"), and actually uses the variance.
- Thin cells need a **prior variance** = a global per-move estimate (this IS the old "optimistic prior
  for an empty cell" — transfer/warm-start folded in here). Test negative transfer; cap prior strength.
- **Optimism on the explore draw only; the override GATE must be PESSIMISTIC** (§B1a) — otherwise
  optimism fires overrides and the floor is gone.
- Drop "bench the argmax / challenger" — Thompson/UCB already re-test the incumbent at log rate.

### B1a. Redesign the override gate ALONGSIDE B1 (not "still valid")  [proposed]
`n≥8 & mean>0` is a magic heuristic; under any mixed-unit reward the floor shifts (`~w·(2·WR−1)`) and
hp-negative moves pass it at WR>50%. Replace with either:
- the floor/argmax test on the **hp component only**, or
- **advantage** `Q(s,a) − V(s)` with a per-cell baseline (floor = "better than this cell's average"),
  and override only when a **lower-confidence bound** on `Q(s,a) − Q(s,fallback)` clears a margin.
Keep the current HP floor as a **separately named** heuristic while experimenting; don't silently
reinterpret it.

### B3. bee/quorum — structured category coverage  [IMPLEMENTED, merged]
Caveat (both reviews): 3 category *proposals* do not guarantee 3 categories get *executed* samples —
measure executed-per-category coverage, not proposals. See `bee_quorum_notes.md`.

### B4. Qwen-driven training — reframed, Test B mandatory  [HYPOTHESIS, gated]
- **Test A is an ENVELOPE test, not an oracle-imitation.** Graded against a mechanical "thinnest-branch"
  oracle, the best Qwen can do is equal a rule we already have for free, and any place it is *right and
  disagrees* scores wrong. Instead score "no catastrophic calls" (never exploit-hard with half the tree
  unsampled; never explore a saturated cell). Feed **unsorted/raw** state (a sorted list makes "name the
  thinnest" list-reading, not judgment). Grade only inferential calls, with pre-registered adversarial
  states. "Thinnest ≠ most valuable" → use a **visitation-weighted** oracle (§E1).
- **Test B is MANDATORY** (snapshots can't reveal outcomes of unchosen control decisions). Pre-register:
  paired branching from the same seed per generation, equal game budget/arm, metric = eval WR after a
  fixed budget + coverage/game, ≥3 seeds, and a **hard mechanical baseline schedule** (a lazy baseline
  flatters Qwen). Predeclare improvement/noninferiority margin and stopping rule (no optional stopping).
- Note: Qwen-controller and Qwen-coach are **two roles of one model** → the controller is biased by the
  coach's own narrative; freeze/ isolate the coach role during the controller test.
- Bounded always: ratios clamped, gate pessimistic, merge `shared=seed`, hard gate, stop conditions;
  the table stays Qwen-agnostic.

### B6. Throughput — VERIFY before scaling  [in use at 8]
13% CPU does NOT establish headroom — the bottleneck is likely the shared text-laya server, emulator
pacing, or Qwen round-trips. **Measure games/hour and server latency at 8 vs 16 workers before claiming
linear scaling.** Also: more deterministic workers from one seed = more *correlated* samples (B2), so
marginal data value per worker falls — Thompson helps here too.

---

## §C. Open experiments (powered via §0b, not opinion)
1. **Pure-table from a RICH seed** (`--policy table`, gen-4/5 seed) — disentangles "bad signal" from "no
   coverage" that 0/144-from-scratch confounds, and pins laya's value. Control unsupported-cell behavior
   explicitly. This is the experiment that would move **D1** from fact to settled.
2. **Credit: immediate vs n-step vs terminal-MC** (B1) vs the gen-4 eval baseline — does the ceiling rise?
3. **Exploration: ε-greedy vs Thompson** (B2) under equal budget — faster coverage/game, less worker
   correlation?
4. **Can Qwen drive training** (B4) — envelope Test A, then mandatory Test B.
5. **Compounding** — eval WR per generation with CIs; climb vs plateau.

## §D. Facts to respect (don't relearn)
- **D1 [now HYPOTHESIS, pending C1]:** per-decision net-hp converging to a passive policy MAY be the
  ceiling — but net-hp telescopes to the round result, so the likelier cause is credit myopia + state
  aliasing + time-outs, not "hp ≠ winning." Settle with §0a + C1.
- laya-alone ≈ 20–22% vs Ryu; in the hybrid its value is the FALLBACK that lets the table be bold (the
  59% is the combination — there is no clean additive baseline, so avoid "super-additive").
- The pure table from scratch **lost every one of 144 rounds** (0 wins) — a failure of THAT config, not
  proven convergence or proven cause.
- Table key has NO opponent identity (range | posture | fireball [| label]); per-matchup today,
  opponent-in-key is the deferred extension (pair with the quorum).
- Table is ATTACKER-specific (actions = me's move set).
- `[n, sum, sumsq]` are **raw moments** (NOT Welford's `[n, mean, M2]`) — raw moments ARE directly
  additive, which is why `merge` sums them; don't rewrite merge as Chan's/Welford's parallel formula.
- Merge MUST dedup a shared seed (`merge(shared=seed)`), or every generation inflates.

## §E. Missing levers / ceilings surfaced by review
1. **State-visitation exploration (value of information).** B2 explores moves *within* a cell; the harder
   problem is *reaching* cells (controlled by movement + the opponent). A thin branch that occurs 0.1% of
   the time isn't worth filling — weight exploration/oracles by visitation, not raw thinness.
2. **Eye accuracy as a hard ceiling.** Posture/range misreads cap the attainable mean per cell. Measure
   per-component eye accuracy on a labelled set; put it in §D. And check the **action space is
   expressive** (wait/idle, walk-then-X) — the table can't learn what the actions can't express.
3. **State aliasing.** Two real situations collapsed into one key ("protect a lead" vs "must attack")
   cap the attainable mean no matter how many samples. Audit keys; consider adding timer/corner/
   lead-state features (screen-only permits a little temporal memory).
4. **Distribution-shift lock-in.** Once the table overrides ~74%, laya's proposals in those cells are
   never sampled → "all 8 workers >50%" may be a shared LOCAL optimum. Track **fraction of cells whose
   argmax changed last generation** as a plateau signal distinct from win-rate.
5. **Nonstationarity / forgetting.** Lifetime counts across generations create false confidence in
   obsolete returns (policies drift). Add recent-generation buckets or controlled forgetting; evaluate on
   recent data.
6. **Invariant [SCRIPT] tests (owner's testing doctrine — the merge bug was caught by hand).**
   e.g. `Σn after merge == Σ worker n − (K−1)·seed n`; no cell mean outside hp bounds; splits only where
   parent `n ≥ threshold`. The split criterion (Welch CI reversal over tens of thousands of cells) is a
   **multiple-comparisons machine** → false splits fragment data; add a correction or a minimum effect
   size.
7. **Merge provenance.** Unique rollout IDs, immutable seed IDs, idempotent merges, no cross-arm
   contamination — shared-seed subtraction only handles the one specified common ancestor.

---
*Reviewers: GPT6 (codex, read-only) + Fable (frontier). Independent convergence on: B1-is-myopia-not-
hp≠winning, n-step credit, β=1, drop match-signal, floor-breaks-under-mixed-units, train≠eval + CIs,
Thompson>UCB, Test-B-mandatory, verify-throughput, raw-moments naming. Unique-to-Fable: §0a diagnostic,
§E1 visitation, §E2 eye ceiling, §E4 lock-in, §E6 invariants.*
