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
