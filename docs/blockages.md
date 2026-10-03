# Executor / channel blockages (living log)

Classes of tactics the system CANNOT currently execute, found by feeding it trusted game-guide moves and measuring
(docs/chunli_vs_*_notes.md + scripts/measure_rule.py). A trusted-correct move that fails exposes a blockage. Each entry:
what breaks, the evidence/repro, the suspected cause, the fix, and a mechanical check to catch regressions.

---

## B1 — Category model collapses to `block` when multiple rules are in play (CRITICAL, 2026-10-03)

**Symptom.** Adding ONE extra positive rule to a working single-rule policy makes the round-1 category model abandon the
applicable rule and pick `block` everywhere; she deals ~0 damage and loses. Game-guide "Expert sets" (5 lines) cratered
every matchup Chun-Li was already winning.

**Evidence (scripts/measure_rule.py, 12 seeds, cat_v3/move_v2, trusted mode).**
- Ryu baseline = `use more throw up close` (1 rule): **7/12 win, +13 hp**.
- Ryu baseline + ANY one mid-range offense rule: **0/12 win, −150..−185 hp** (significant):
  `use more s.mk at mid when he jumps` (−161), `use more c.mk at mid` (−185), `use more c.hk at mid when he attacks` (−175).
- A NEGATIVE rule (`avoid spinning_bird_kick`) added: **+0.0, no change** (adds no competing positive category).
- Per-decision trace at CLOSE range (her winning throw range) with `[throw up close, c.mk at mid]`:
  `range=close his=standing category=block action=block_high rule=soft follows_rule=False` — the engine records that the
  throw rule APPLIES (`rule=soft`) but the category model picks `block` and does NOT follow it. c.mk is done 0 times;
  she just blocks and takes ~175.

**Repro.**
```
python scripts/measure_rule.py --opp ryu --trusted --add --dev 0-11 --rules "use more c.mk at mid range"
# baseline 7/12 +13  ->  candidate 0/12 -171   (then read a cand trace: block_high only, 0 throws)
```

**Suspected cause (pending Fable confirmation).** cat_v3 was trained very hard on `condition_off -> block` (scores 1.0
there). The training data (scripts/build_advice_data.py) puts exactly ONE positive rule per example plus only
negative/habit distractors — it never shows TWO positive rules where only one applies to the moment. So the model
appears to have over-generalized to "if ANY present rule does not apply here, block", and a second non-applicable rule
suppresses the rule that does apply. This also explains the session-long turtle (the loop accumulates rules → collapse)
and why Honda "improved" (its baseline already blocked-and-lost).

**Status: CONFIRMED and RESOLVED (2026-10-03) — no retrain needed.** The offline probe refined the trigger: the collapse
needs a **range-only applicable rule** competing with a non-applicable positive. A `when`-keyed applicable rule is
robust (probe arm with "...when he stands" scored 1.0); the book seeds are all range-only (`use more throw up close`),
so the loop cratered the instant a second rule was added. Confirmed on cat_v3 (training-format prompt, no gameplay):
`['use more throw up close'] -> throw 0.97` but `+ 'use more c.mk at mid range' -> block 1.00` (same with canonical
`throw_F+hp` and with `s.mk` → NOT phrasing, NOT c.mk-specific; and stripping the non-applicable rule → throw 0.97).

**Fix (SHIPPED): oracle routing in `sf2/system1/loop_runner.two_stage_decide`.** The deterministic oracle already knows
which rules apply, so the models are now shown ONLY the applicable rules (`prompt_lines`); non-applicable distractors
never reach the prompt. No retrain: the single-rule case already follows correctly. Guarded by
`tests/test_loop_advice.py::test_oracle_routing_*`. GAME re-validation: the exact crater case went from **0/12, −185**
(before) to **5/12, +2.7** (after, delta −10.7, inconclusive = noise) — the −198hp collapse is gone.
Deferred (optional, not needed for the fix): regenerating the category data with multi-rule examples would make the
model itself robust to distractors, but routing makes that unnecessary for now.

**Decisive offline probe (cheap, no gameplay).** Query cat_v3 vs the deterministic oracle (`advice.two_stage`) on a
committed fixture of ~200 situations, 5 arms: (A) one applicable positive; (B) A + one NON-applicable positive; (C) A +
a negative on another move; (D) B with line order swapped; (E) two applicable positives. Follow-rate predictions:
shortcut → A≈C high, B≈D≈0; position bias → B,D≈50% & order-dependent; phrasing → re-run B with canonical `throw_F+hp`.

**Mechanical check (regression guard).** A pytest running the 5-arm probe on the committed fixture, asserting
`follow(B) ≥ follow(A) − 0.10` AND `block(condition_off) ≥ 0.90`. Must be SEEN RED on cat_v3 first (seeded-red
admission). PLUS a per-play gate: among decisions where the oracle rule ∈ {soft,hard}, `follows_rule ≥ 0.7`
(gaps.py already counts it — make it exit non-zero).

---

## B2 — `avoid` rules have little or no grip (MINOR, 2026-10-03)

**Symptom.** An `avoid <move>` rule rarely changes behavior. `avoid block_high when he stands` → +0.0 vs Honda;
`avoid spinning_bird_kick` → +0.0 vs Ryu. The executor does not reliably stop doing a move when told to avoid it.
**Status.** Noted; lower priority than B1. May be fine (negatives are meant to prune, not drive) but worth confirming
the move model actually drops an avoided move when it would otherwise pick it.

---

## Resolved
- Movement channel (walk_forward/back) was unfollowable under competing advice → fixed via advice_v5 + cat_v3/move_v2
  (see memory project_textlaya_cant_move_2026_10_03). NOTE: B1 now shows that fix is undercut whenever >1 rule is in play.

---

## B3 — Measurement engine overwrote per-candidate traces (TOOLING, 2026-10-03)
scripts/measure_rule.py reused one output dir per opponent, so running several candidates for the same opponent
clobbered earlier candidates' decision traces — evidence needed to diagnose a crater was lost (had to re-run). Fix:
tag the output dir by a candidate hash/label so every arm's trace survives.

## B4 — Honda "improvement" (+52) is unexplained (OPEN, 2026-10-03)
The one matchup the game-guide set helped has no decision trace of its own (same overwrite issue). The "collapse was
free because Honda already blocked-and-lost" story is a hypothesis, not evidence. Capture a Honda cand trace when B1 is
re-examined.

---

## Decision D1 — no-rule default stays `block` (2026-10-03)
Tested the owner's idea that the no-rule default should be exploratory movement `{stand, walk_forward, walk_back}`
instead of `block`. Measured model(block) vs explore on the same book-seed policy, 8 seeds, per opponent:
- honda: block 4/8 +10 vs explore 0/8 -151.8 (delta -162, block significantly better)
- ryu:   block 4/8 +9.5 vs explore 2/8 -55   (trend block better)
- guile: block 4/8 +23.8 vs explore 1/8 -8.5 (trend block better)
- ken:   block 2/8 -10.8 vs explore 3/8 +9.8 (~tie)

DECISION: keep `block` as the no-rule default; do NOT ship the explore default (reverted to keep the code simple).
WHY: in SF2 holding BACK is the block input, so of the three options "back" already IS the block default and it is the
best of the three; "stand" (no-action) releases her defense and "walk_forward" walks into danger when she has no
guidance (confirmed by Honda -162). The block default was never the turtle cause - B1 (multi-rule collapse, fixed by
oracle routing) and a lack of good firing rules were. The lever is better RULES, not a different default.
