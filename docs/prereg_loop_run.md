# Pre-registration: the measured self-learning loop run (G8, 2026-10-02, draft for the owner)

The question (owner): **game after game, does Chun-Li play better as Qwen's rules are admitted and rotated?** And if
not, is it bad Qwen rules, text laya not following, or bad screen facts? Pre-registered BEFORE any measured game.

## Setup (fixed)
- Me: Chun-Li. Screen-only play (no RAM in play; hard gate clean). Two-stage text laya cat_v1 + move_v1.
- Opponents: the 6 the book covers - dhalsim, guile, honda, ken, ryu, zangief. (Blanka has no book seed; add only if
  wanted.)
- Qwen on threebody (WMI start, /health ok; Tailscale 100.66.12.33 if LAN is down; fall back to local omlx if the box
  sleeps). Qwen off again at the end.
- Seed: each opponent starts from book.json's verified lines (tagged web).
- Scoring: screen round result + hp (reader gated vs RAM); the offline REPLAY (RAM referee) re-scores every game and
  any reader/replay mismatch is flagged (not silently trusted) - requires the replay-scorer fix.

## Two arms (so we separate learning from the seed)
- **LEARN:** seed + Qwen updates every game (admit / keep / retire). The real loop.
- **FROZEN:** seed only, Qwen OFF, rules never change. Same seeds/opponents. The control: if LEARN does not beat
  FROZEN, learning added nothing.
Same seeds and opponents for both so the comparison is paired.

## What we measure (per opponent, per arm)
- hp/round and rounds won, GAME BY GAME (the trend is the result).
- "Better" (pre-registered): mean hp/round (and rounds won) over the LAST K games minus the FIRST K games, per
  opponent; LEARN improves if that delta is positive and LEARN > FROZEN on the paired comparison (bootstrap CI by
  game, lower bound > 0). Pooled across opponents too.
- Qwen rule churn: rules added/removed per game; how many survive.
- follows_rule per rule (did text laya follow?).
- The table's OFFLINE score of each admitted rule (G5): good / ok / bad / not_scorable.
- Reader-vs-RAM agreement per game (from the replay) - so a drop is attributable to the eye, not the brain.

## Diagnosis tree (if LEARN does not improve)
1. Qwen writes bad rules -> G5 marks them bad; churn high, survivors bad.
2. Text laya does not follow good rules -> follows_rule low on good (G5) rules. Then: wording it is asked (check the
   situation sentence) or a skewed checkpoint (cat_v1/move_v1 held-out eval).
3. Bad screen facts -> reader-vs-RAM low that game.
All three are in the per-game report (G7).

## Owner decisions (before the run)
1. Opponents: the 6 (default) or add Blanka / anyone.
2. Games per opponent per seed: default 20 (2-3 rounds each) - enough to see a trend; more = stronger verdict.
3. Rounds per game: 2 or 3 (3 = best-of-3, a real match).
4. Seeds: 2 (first look) or 3+ (verdict).
5. K (the first/last window for "better"): default 5.
6. Both arms (LEARN vs FROZEN) or LEARN only?
7. Max rules Qwen holds in short memory (rotation pressure): default the existing cap.

## Before the run (checks)
- cat_v1/move_v1 held-out eval re-confirmed; the loop smoke green on 1 opponent; the replay referee runs headless; the
  round-result win/loss fix in; G5 + G7 wired; latency per decision acceptable with two models.
