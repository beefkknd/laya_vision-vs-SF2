# Ken: expert advice vs Qwen's, pre-registered (2026-09-30, before any game)

Same design as docs/prereg_ryu_expert.md (scripts/ab_memory.py --fixed, every arm paired with no advice, tag
lesson-loop-v2 code with lock lesson_loop_v1's checkpoints; 8 seeds 42001-42008 x 15 rounds per arm; seed as the unit;
`scripts/fixed_arms_report.py logs/ab/ken_expert --opp ken --vs expert,qwenset`). Notes: docs/chunli_vs_ken_notes.md.
The expert set is not independent of our data: its lines 1 and 5 are Qwen's two known Ken winners.

| arm | lines |
|---|---|
| expert | use more hp up close when he jumps; use more throw up close; use more mk at mid range when he jumps; use more throw up close when he attacks; avoid sweep up close when he jumps |
| qwenset | use more hp up close when he jumps; avoid sweep up close when he jumps; use more hp at mid range when he jumps; use more c.mk at mid range when he attacks; avoid c.hp up close when he jumps |
| throw | use more throw up close |
| throwatt | use more throw up close when he attacks |
| hpatt | use more hp up close when he attacks |
| antiair | use more mk at mid range when he jumps |
| cmkfar | use more c.mk far away when he jumps |
| blockjump | use more block_high up close when he jumps |

Primary: each arm vs no advice. Secondary: expert - qwenset. No arm changes after the first game; a failed job is re-run
once with its seed.
