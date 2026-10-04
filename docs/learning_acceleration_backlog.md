# Learning-acceleration backlog (laya-text + value-table + Qwen)

A standing list of ways to make the self-learning loop learn FASTER / HIGHER, plus the open questions
to settle by experiment. Companion to `docs/design_laya_table_hybrid.md` (architecture + Phase 2) and
`docs/bee_quorum_notes.md`. Written 2026-10-04 on `feat/table-system1`.

## Context — the generational curve so far (Chun-Li vs Ryu, round win-rate)

| gen | win-rate | note |
|-----|----------|------|
| 1 | 33.7% | seed thin |
| 2 | 49.8% | exploit a richer seed |
| 3 | 38.8% | EXPLORE round (21% explore) — dipped on purpose, built coverage |
| 4 | 58.9% | CASH-IN (explore ~1%) — exploited the 86k table; all 8 workers >50% |
| 5 | (running) | compounding test |

Established: **explore-rounds build coverage (win-rate dips); exploit-rounds cash it in (win-rate
jumps).** The ceiling driver is the credit signal (hp-safety ≠ winning). Goal = ~75% (owner), not 100%.

---

## A. Two axes of "speed" (frame every lever by which it moves)

- **Sample-efficiency** — each game teaches more (fewer games to learn). The big wins live here.
- **Throughput** — more games per hour (fixed sample-efficiency). Cheap but linear.

Given fixed physical resources (owner's constraint), prioritize **sample-efficiency**.

---

## B. Levers, ranked (sample-efficiency unless noted)

### B1. Better credit signal — layered reward (HIGHEST LEVERAGE)  [status: proposed, not built]
Replace the table's single `net = dealt − taken` with a layered reward accumulated in the Welford cells:

```
R(decision) = w_hp · shaped_hp  +  w_round · round_signal  +  w_match · match_signal
  shaped_hp    = dealt − β·taken            # β<1 tilts toward OFFENSE (counters the safety bias); the attack/defense knob
                                            #   defense term (later): + (expected_incoming − taken)  [damage AVOIDED]
  round_signal = +1 won / −1 lost           # optionally recency-discounted toward the KO (γ^steps_from_end)
  match_signal = +1 / −1 (best-of-3)        # small weight, sparse
```
- WHY: ties every decision to WINNING, not just hp — fixes the exact ceiling we keep hitting and the
  pure-table defensive collapse at its source. Speeds convergence (cleaner signal) AND raises the ceiling.
- CHANGE: localized — `net_of()` / `credit` (pass round+match result in). `mean>0` floor still valid.
- HARD PARTS: weight tuning (`w_round` too high swamps the dense hp signal — start moderate); the
  defense "damage avoided" needs a rough per-opponent-move damage prior (noisiest piece — ship the
  round-win term FIRST, add defense-avoided later).
- FIRST EXPERIMENT: round-win-weighted credit vs the gen-4 58.9% baseline, test-first.

### B2. Directed exploration — uncertainty-scaled "booster"  [status: proposed, not built]
Replace ε-greedy-over-under-sampled with an optimism rule in `choose()`:
```
select(a) = mean(a) + c · sqrt(ln N_cell / n(a))      # UCB: thin move -> big visibility bonus, shrinks as it fills
```
- WHY: boosts exactly the under-tried moves, by how uncertain they are, and self-calibrates. This is the
  owner's "artificially bump the thin rules' visibility" — but uncertainty-scaled, so it's harnessable
  (a FLAT bump is noisy/hard to tune; an uncertainty-scaled one is not). Reuses the Welford variance
  already stored (Thompson sampling = the variance-sampling cousin).
- OPTIONAL "challenger" mode (owner's "filter out the thick"): occasionally BENCH a cell's argmax to
  force a re-test of the dominant move — cheap guard against exploit lock-in.
- CHANGE: localized — `choose()` selection rule.
- HARD PART: `c` tuning; interacts with B1 (explore where, credit what).

### B3. bee/quorum — structured category coverage  [status: IMPLEMENTED on feat/laya-quorum, merged]
3 same-checkpoint bees (attack/move/defense prompts) vote; forces all categories to be sampled per
branch → fills the tree faster than one biased laya. Correlated votes (shared weights) → coverage, not
independent-ensemble. See `bee_quorum_notes.md`. Owner decision: keep the table OVERWRITE (not the
quorum's weighted-voter) to start.

### B4. Qwen-guided exploration / self-driving control  [status: HYPOTHESIS, gated by a test]
Qwen as a BOUNDED meta-controller: reads the learning state (coverage, thin branches, win trend) and
sets the next burst's explore/exploit ratio, which branches/categories to fill, which bee, when to
re-seed. Directed search >> random coverage; also the self-sustaining-loop payoff.
- OPEN QUESTION (the one the owner most wants answered): **can Qwen really drive the training?**
- VALIDATION GATE (design doc §7, prove BEFORE wiring into bee/quorum):
  - Test A (cheap, offline): feed Qwen the gen-1..5 learning-state snapshots; grade its control calls
    against the table's MECHANICAL oracle (actually-thinnest branches, coverage-says-explore?), ≥10
    permutations; score consistent-right / flip-flop / consistent-wrong.
  - Test B (expensive): A/B Qwen-set ratios vs the mechanical schedule; must match or beat.
  - Integrate only if A (ideally B) passes. Qwen stays bounded (ratios clamped, `mean>0` floor, merge
    `shared=seed`, hard gate, stop conditions); the table stays Qwen-agnostic.

### B5. Transfer / warm-start  [status: partial]
Seed new cells from related ones. Parent-fallback already does this for splits. Extensions:
cross-opponent seed-then-evolve; use laya's pick as an optimistic prior for an empty cell.

### B6. More parallel throughput  [status: in use at 8; headroom to ~16]  (THROUGHPUT, not sample-eff)
Workers each own a port (system1 range = 16) + own dir + merge `shared=seed`. 8 workers ran the box at
~13% CPU; 16 roughly doubles data/hour for free. Shared text-laya server makes added workers ~free.

---

## C. Open experiments / questions to settle (by measurement, not opinion)

1. **Pure-table from a RICH seed** — run `--policy table` seeded from the gen-4/5 table. Hypothesis:
   not the from-scratch 0%, but BELOW the hybrid's 59% (loses laya's coverage of the ~36% of cells
   with no confident-positive move — the defensive-collapse zone). Cheap: 1 run. Pins the value of laya.
2. **Round-win-weighted credit (B1)** vs the gen-4 58.9% baseline — does the ceiling rise?
3. **UCB exploration (B2)** vs ε-greedy — faster coverage per game?
4. **Can Qwen drive training (B4)** — Test A then B.
5. **Compounding** — does each exploit-gen keep climbing, or plateau (diminishing returns toward 75%)?

## D. Known facts to respect (don't relearn)

- hp-safety ≠ winning: per-decision net-hp alone converges to a DEFENSIVE losing policy (pure table lost
  0/144 from scratch). The `mean>0` floor + laya fallback is what prevents it in the hybrid.
- laya-alone ≈ 20-22% vs Ryu; its value in the hybrid is NOT additive — it's the fallback that lets the
  table be bold (combo = 59%, super-additive).
- The table key has NO opponent identity (range | posture | fireball [| label]); opponent-specific
  tables = per-matchup today, opponent-in-the-key is the deferred extension (pair with the quorum).
- The table is ATTACKER-specific by construction (actions = me's move set).
- Merge MUST dedup a shared seed (`merge(shared=seed)`), or every generation inflates.
