# Ryu: expert advice vs Qwen's, pre-registered (2026-09-30, before any game)

Owner: "search the game strategy for Chun-Li against Ryu, verify it, then see what Qwen says" - a shortcut instead of
learning from scratch. Strategy notes with sources: docs/chunli_vs_ryu_notes.md.

`scripts/ab_memory.py --opps ryu --fixed ...`, every arm paired with no advice (same savestate, same seed), System 1 at
tag lesson-loop-v2 with lock lesson_loop_v1's checkpoints; 8 seeds (41001-41008) x 15 rounds per arm; run (seed) as
the unit, 95% intervals (sf2/eval/stats.run_level). Arms:

| arm | lines |
|---|---|
| expert | use more c.mk at mid range; use more throw up close; use more mk at mid range when he jumps; use more sweep at mid range when he attacks; avoid spinning_bird_kick |
| qwenset | Qwen's 5 lines most in play vs Ryu in round 5: avoid spinning_bird_kick at mid range when he stands; use more forward far away when he attacks; use more c.mk at mid range when he stands; avoid spinning_bird_kick far away when he attacks; use more sweep at mid range when he attacks |
| throw | use more throw up close |
| cmk | use more c.mk at mid range |
| sweep | use more sweep at mid range when he attacks |
| dizzy | always hp up close when he is stunned |
| antiair | use more mk at mid range when he jumps |
| blockfar | use more block_low far away when he attacks |

Primary: each arm vs no advice, hp per round. Secondary: expert - qwen (paired through the same no-advice arm).
No arm is added or changed after the first game; a failed job is re-run once with its seed.

Note (before any game): the arm `qwen` is named `qwenset` (`qwen` is a reserved arm name in ab_memory); same lines.

## Result (2026-09-30), `scripts/fixed_arms_report.py logs/ab/ryu_expert --opp ryu --vs expert,qwenset`

8 seeds x 15 rounds per arm, all 8 batches saved; seed as the unit. Rounds won of 120: no advice 32.

| arm | hp per round vs no advice | rounds won |
|---|---|---|
| **throw** (use more throw up close) | **+42.4 [+27.0, +59.0] helps** | **54** |
| **expert** (5 lines) | **+37.9 [+10.7, +71.3] helps** | 33 |
| **qwenset** (Qwen's 5 lines from round 5) | **+24.4 [+6.2, +44.1] helps** | 41 |
| sweep at mid range when he attacks | +13.2 [-1.7, +34.7] | 39 |
| always hp up close when he is stunned | +2.6 | 38 |
| c.mk at mid range | +1.8 | 26 |
| block_low far away when he attacks | +0.0 (never changed a decision: a soft lesson does not override "likely fails") | 32 |
| mk at mid range when he jumps | -8.5 | 24 |

expert - qwenset: +13.5 [-11.5, +40.2], not shown.

Why Qwen never found the throw: she threw 13 times in 2,259 decisions of her history against Ryu (laya-vision rarely
rates it high), 6 of them up close - "too few" to judge; the views Qwen reads are built from what she did, and its
what-if never picked it. In 36 Ryu loop ledgers Qwen named throw twice, both "avoid". A player's tip found in one
search what the loop could not explore: the owner's shortcut.
