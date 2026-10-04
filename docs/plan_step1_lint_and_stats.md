# Step 1 — enforceability lint + pooled cross-block stats (plan; no code yet)

The first, highest-leverage slice of docs/plan_review_framework.md. Fixes the two MEASURED reasons chun3 never
learned, and nothing else. Test-first. No retrain. Two independent sub-parts; 1A ships before 1B.

## Why (the two bugs, measured in chun3)
1. **Unfollowable rules.** At close range the menu offers only `cl.*` normals (`advice.moves_in_stance` via
   `stance_of("stand","close")="close"`), so any `s.* up close` rule can NEVER be the move text-laya plays. 3 of 10
   chun3 lines + one explore-pool line were dead this way; the Coach keeps proposing moves she can't do.
2. **Evidence resets every block.** Each block is a fresh `play_loop_screen` subprocess with `all_rows=[]`, so
   `condition_evidence` never sees > 1 block; `MIN_TRIES=20` on both sides is rarely reached in one block → nothing
   ever graduates to `kept` → endless churn.

## Testability gate: YES
Deterministic, offline, hermetic. Fixtures already on disk: `playbooks/chun3/round_*/g*/decisions.jsonl` (real
decisions with `prompt_lines`, `rule`, `follows_rule`, `category`, `action`, `moment`, `facts`) and the per-round
`trace.jsonl`. Every assertion below is a script-decidable compare; each is seen RED first (against today's code or a
seeded-bad twin) before it is admitted.

---

## Part 1A — enforceability (STATIC; smallest; do first)
Enforceability depends only on (move, range, character) — not on play — so it is a pure static check, no stats needed.

**Oracle (new, pure):** `followable(move, rng, me) -> bool` =
```
cats = char_categories(me)
rngs = [rng] if rng else RANGES              # range-agnostic rule: followable if offered in SOME grounded range
return any(move in available_moves(stance_of("stand", r), cats) for r in rngs)
```
Home: a small pure helper (in `sf2/system1/advice.py` next to `available_moves`, or a new `sf2/system2/followable.py`
that imports it — decide at implementation; advice.py is the natural home since the primitives live there).

**Where it is enforced (4 sites, all deterministic — Qwen never gates):**
1. **Admission** — `short_memory._valid(claim, moves)` gains `followable(move, range, me)`. An unfollowable claim is
   never admitted (room-fill or swap). (`_valid` needs `me`/`categories` threaded in — minor signature change.)
2. **Explore pool** — `explore_pool.pick` skips a pool line that isn't followable in the target cell; and the shipped
   inert line `"use more s.hk up close when he stands"` is replaced with a close-valid move (e.g. `cl.hk` or a
   throw/special offered close).
3. **Coach filter** — `character_prompt.coach_filter` adds the range-vs-stance check (today it only screens `c.*`/`j.*`
   prefixes via `stance_unreliable`), so Qwen's proposal is dropped before it reaches the registry.
4. **Standing lint drop** — a pure `drop_unfollowable(reg, me) -> (reg, dropped)` run at block start
   (`starting_registry`) and after each admission: any in-play line that is not `followable` is dropped with a logged
   reason ("can't be played at its range"). Immutable (returns a new list).

**Tests (red first):**
- `followable("s.mk","close","chunli") is False` and `followable("s.mk","mid",...) is True`; range-None `s.mk` True.
- A crafted claim `use more s.lp up close when he jumps` is refused by `_valid` / `coach_filter` (seen RED: today it
  is admitted — it's in the chun3 playbook).
- `drop_unfollowable` on the chun3 registry removes exactly the 3 inert lines (`s.lp`/`s.hp` up close, …) and keeps
  the rest (fixture: `playbooks/chun3/playbook.json`).
- `explore_pool.pick` never returns an unfollowable claim for any cell; the replaced pool line is followable.

**Acceptance (1A):** an invariant test — no registry the loop produces contains an unfollowable in-play line; and a
fresh short run admits none. Ship 1A alone (it unblocks the Coach immediately; zero stats involved).

---

## Part 1B — pooled cross-block stats (so rules can graduate)
**`entry["stats"]`** — one dict carried in the registry entry beside `rounds` (so it pools across blocks and survives
restarts via `--save-registry`/`--carry`, already JSON round-tripped). Minimal Step-1 schema:
- `applicable`, `enforceable`, `followed`, `shadowed` (counts; `shadowed` collected now, acted on in Step 2).
- effect sums `n_mine,sum_mine,sumsq_mine,n_rest,sum_rest,sumsq_rest` — the exact inputs `condition_evidence` uses,
  accumulated so `MIN_TRIES` is reachable across blocks.
- (optional, cheap) round correlation `rounds_fired,wins_fired,rounds_idle,wins_idle` — for Step-2 Coach/done; collect
  if free, else defer.

**New pure module `sf2/system2/rule_stats.py`:**
- `tally(stats, decisions, round_summary, me) -> stats` — fold one round's decisions (from `screen_evidence.read_decisions`)
  into a NEW stats dict (immutable). Uses `prompt_lines`/`rule`/`follows_rule`/`action` + `followable` for `enforceable`.
- `evidence_from_stats(stats) -> dict` — same shape/output as `lessons.condition_evidence`, computed from the pooled
  sums (same Welch half-width). **Invariant test:** on a single block's rows, `evidence_from_stats(tally(...))` ==
  `condition_evidence(rows, claim)` exactly (no behaviour change — just pooled).

**Wiring:**
- `play_loop_screen.update`/`run_loop`: before `SM.step`, call `rule_stats.tally` for each in-play entry with this
  round's `drows`+`summary`; write the stats into the entry. Graduation (`short_memory._graduate`) reads
  `evidence_from_stats(entry["stats"])` instead of recomputing on block-local `all_rows`; `_weakest_trying` uses the
  pooled score. Trace the per-rule stats row in the `qwen` event (for the TUI/Coach later).
- Carry: stats ride in the entry through `save_registry`/`load_registry` unchanged.

**Tests (red first):**
- `evidence_from_stats == condition_evidence` on chun3 round_03 rows for a rule with ≥20 tries (seen RED on a twin
  that mis-sums).
- Stats POOL across blocks: run `tally` over block A then block B (with a `save_registry`→`load_registry` between);
  a rule with 12 tries/block reaches `n_mine≥MIN_TRIES` after two blocks and `evidence_from_stats` classes it — whereas
  the per-block path stays `"few"` (the bug, seen RED).
- A rule with pooled `cls=="better"` graduates `trying→kept`; one with `cls=="worse"` drops; `"unclear"/"few"` stays.
- Save/load round-trips `entry["stats"]` byte-identical.

**Acceptance (1B):** run a fresh `chun4` career (headless) for a few blocks and assert, from the saved playbook, that
≥1 rule reaches `kept` (chun3 reached 0) and that a genuinely good rule's pooled `diff` lower-bound > 0. Mechanical
check over the files, not a vibe.

---

## What Step 1 deliberately does NOT do (deferred to Step 2+)
one-positive-per-when / shadowing DROP (Gate B); coverage-ordered Coach ask; the "when" vocabulary & damage-first
ordering; retone-on-low-follow (Gate C); causal "done" (frozen confirmation block); deleting the legacy
`lessons.review/propose/...` path. Step 1 only: stop unfollowable rules, and make graduation possible.

## Staging & verification discipline
1. 1A (static lint) — land, test, ship. 2. 1B (pooled stats) — land, test, run chun4. Keep the inner-loop test suite
fast (pure fixtures). **Separation of duties:** a blind verifier on a DIFFERENT engine (codex) reviews the diff +
re-runs the red tests, given only the artifact, not this reasoning.

## Risks / watch
- `_valid`/`_graduate` signature changes ripple into `short_memory.step` callers and tests — update together.
- `enforceable` must use the SAME stance mapping the live `two_stage` uses, or the count diverges from reality — reuse
  `advice` primitives, don't re-derive.
- Pooling changes when rules graduate → some existing short_memory tests assume block-local evidence; update them to
  the pooled model in the same commit (seen-red that they were asserting the old behaviour).
- A carried OLD registry (chun/chun3) has no `stats` key — `tally` must treat missing as empty (start fresh), and
  `drop_unfollowable` will prune its dead lines on load (expected, logged).
