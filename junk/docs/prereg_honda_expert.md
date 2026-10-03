# E. Honda: expert advice vs Qwen's, pre-registered (2026-09-30, before any game)

Same design as docs/prereg_ryu_expert.md; 8 seeds 43001-43008 x 15 rounds per arm; seed as the unit;
`scripts/fixed_arms_report.py logs/ab/honda_expert --opp honda --vs expert,qwenset`. Notes: docs/chunli_vs_honda_notes.md.
Every source says keep away from Honda; she cannot walk back (System 1 picks only attacks, blocks and forward), so the
core advice is only approximate here ("avoid forward at mid range"). His throw outranges hers (arcade WW), unlike Ryu's.

| arm | lines |
|---|---|
| expert | avoid forward at mid range; use more block_low at mid range when he attacks; use more c.mk far away when he jumps; use more mk at mid range when he jumps; use more throw up close when he attacks |
| qwenset | avoid sweep at mid range when he jumps; use more forward at mid range when he jumps; avoid sweep at mid range when he stands; use more c.hp at mid range when he attacks; use more c.mk at mid range when he attacks |
| avoidfwd | avoid forward at mid range |
| blockmid | use more block_low at mid range when he attacks |
| cmkfar | use more c.mk far away when he jumps |
| throw | use more throw up close |
| throwatt | use more throw up close when he attacks |
| legs | use more lightning_legs up close when he stands |

Primary: each arm vs no advice. Secondary: expert - qwenset. No arm changes after the first game; a failed job is re-run
once with its seed.

## Result (2026-09-30): see docs/component_boundaries.md (table) and docs/boundary/honda_expert.json (per-line trace).
