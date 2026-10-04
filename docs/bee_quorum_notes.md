# Bee / quorum player — notes & sync checklist

Companion to `docs/design_laya_table_hybrid.md` (§6 forward, §7 Phase 2). Written 2026-10-04 on
`feat/table-system1` as prep for the bee/quorum direction.

---

## 0. STATUS — where the code is (read first)

As of 2026-10-04, **no bee/quorum code is present on `feat/laya-quorum`** — that branch is still the
stale `54cdfd4` base it was cut from, and **no branch or worktree in this repo contains quorum / bee
/ vote code.** The quorum implementation from the other session is not committed/pushed here. It
also predates (does not contain) these `feat/table-system1` commits:

- `baeee13` fix: read_end_bars reads the KO, not the round-over refill (win/loss scoring)
- `519b84f` fix: merge(shared=seed) — count a shared seed once, not once per worker
- `10af43f` / `1541882` feat: `--explore` rate + `--explore-tries` coverage-target knobs
- `977de81` / `29f99b1` docs: the hybrid design doc + Phase 2 (this file's companion)

**Step 0 before any quorum work:** locate the quorum code, commit + push it, and rebase/merge it
onto the current `feat/table-system1` so it has the fixes and the design reference below. Confirm
which branch it should live on (the local `feat/laya-quorum` was never pushed).

## 1. What a "bee" is (owner-confirmed)

- **Same text-laya checkpoint, different prompts** — 3 bees biased toward **attack / movement /
  defense** (≈ the `cat_laya` categories). Not different weights.
- Therefore the votes are **correlated** (one model nudged three ways), so the quorum buys
  **structured category coverage**, NOT independent-ensemble error-cancellation. Cheap: one shared
  `shared_laya` server serves all three prompts.
- **First thing to verify in the code:** that the three prompts actually produce *divergent*
  proposals. Log the per-decision 3-way agreement rate. If the model's prior is strong enough that
  all three say the same move, the specialization is cosmetic and gives no more coverage than one
  laya — the whole premise rests on this.

## 2. How the quorum plugs in (the player slot)

The quorum **replaces the player** (`base_decide` in `value_table.hybrid_decider`); everything else
stays. Per decision:

1. The 3 bees propose; the **quorum rule** picks the move to play.
2. The **value table still overwrites** that move where it is confident-good (`count >=
   OVERRIDE_TRIES` and `mean net-hp > 0`); otherwise the quorum's move stands; exploration still
   fires per `--explore` / `--explore-tries`.

Two design forks that affect learning quality (see design doc §6):

- **Table vs quorum precedence.** Recommended: the table **overwrites** the quorum's consensus
  (preserves the learner-decoupled-from-player invariant and the player→table handoff). Alternative:
  the table is one more voter — dilutes its empirical authority, changes explore/exploit balance.
- **The quorum rule is an exploration policy.** Strict *consensus* → plays the mushy middle move no
  bee wants → the table learns about mediocre moves. *Plurality / let-minority-through* → preserves
  bold proposals that fill the tree. Tune it like `--explore`.

## 3. How it affects learning (summary; full in design doc §6)

- Changes the **sampling distribution, not the learner.** The table keeps learning `(when, move) ->
  net hp` exactly as now — player-agnostic.
- Main benefit: **more complete branches** (attack + movement + defense leaves all get sampled per
  `when`) → thicker, faster coverage. Specialists help most **early / in sparse branches**, then
  recede as the table matures and overwrites more.

## 4. Phase 2 hook — Qwen self-guided control (design doc §7)

Once gen-3 completes, run the **validation gate** before folding Qwen-as-meta-controller into the
quorum: Test A (offline — grade Qwen's control calls on existing snapshots vs the table's mechanical
coverage oracle, ≥10 permutations), then Test B (A/B Qwen-set ratios vs the mechanical schedule).
Short bursts with a **coverage-driven** adjustable ratio (not win-rate — too noisy over short
bursts). Only integrate if it passes.

## 5. SYNC CHECKLIST — reconcile the quorum code with current `feat/table-system1`

Must-have code the quorum branch needs (it predates these):
- [ ] `merge(shared=seed)` dedup in `value_table.merge` — or every generation's merge inflates the
      shared seed (gen-1 was 50086 vs the correct 22987). Non-negotiable for generational runs.
- [ ] `read_end_bars` KO fix — or win/loss is invisible (everything scores "draw") on `--no-score`.
- [ ] `--explore` + `--explore-tries` knobs in `play_loop_screen` (and `hybrid_decider`'s
      `explore` / `min_tries` / `override_tries` params).
- [ ] `OVERRIDE_TRIES=8` + the `mean>0` overwrite floor.

Invariants the quorum code MUST preserve (verify by reading, then by the suite):
- [ ] **Hard gate**: no RAM/TABLE-module import in the play path (`sf2/system1/loop_runner.py`,
      `sf2/screen`); `sf2/data/value_oracle.py` is gate-forbidden — `json.load` the oracle, never
      import it.
- [ ] **Credit stays `(when, move) -> net hp`, player-agnostic** — do NOT add per-bee cells unless
      that's a deliberate, separate decision.
- [ ] The quorum is the PLAYER (`base_decide`); the table overwrites where confident-good; keep the
      `mean>0` floor (anti-defensive-collapse) whatever the player is.
- [ ] Parallel fan-out: own `--port` per worker, own playbook dir, merge with `shared=seed`.

Mechanical sync step:
- [ ] `git merge` (or rebase) `feat/table-system1` into the quorum branch.
- [ ] Run the full suite + the hard-gate test; confirm `test_value_table`, `test_table_policy`,
      `test_table_merge`, `test_scoring_fix` all green.
- [ ] Smoke a 1-worker hybrid run with the quorum as player; confirm the `source` split logs bee
      vs table vs explore and the table credits/merges as expected.
