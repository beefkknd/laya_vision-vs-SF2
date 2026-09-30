# Three new opponents: Zangief, Guile, Dhalsim - pre-registered (2026-09-30, before any game)

Owner: "once verified, pick 3 more characters to further test out the boundary." Three different styles: Zangief
(grappler, longer throw than hers), Guile (zoner; no play history, so Qwen starts from nothing), Dhalsim (long reach).
Code at the commit of this file; live checkpoints (hash-identical to lock lesson_loop_v1's); savestates
states/p1_chunli_vs_{zangief,guile,dhalsim}.state.

1. Qwen's loop: `qwen_lessons.py --opp O --seed S --prompt character_fgc`, seeds 51001-51004 per opponent (12 runs),
   history on (none exists for Guile). Question: what does Qwen find, and does it find the throw?
2. Fixed advice (after the players' tips are researched, docs/chunli_vs_<opp>_notes.md): the players' set, "use more
   throw up close", and single tips, each vs no advice, 8 seeds x 15 rounds (seed as unit), then
   `scripts/boundary.py` per line. The throw arm is the pre-registered test of the general claim that laya-vision's
   P(hit) ranking hides the throw (docs/component_boundaries.md).
No arm or prompt changes once a batch starts; a failed run is re-run once with its seed.

## Fixed-advice arms (written after the players' tips were researched, before any fixed-advice game)

Every set has the same two throw arms: `throw` = use more throw up close (the general claim), `throwhard` = always throw up close (does laya-vision's veto stand in the way?). 15 rounds per arm per seed, seed as the unit.

### Zangief (seeds 52001-52008; docs/chunli_vs_zangief_notes.md)

| arm | lines |
|---|---|
| expert | use more c.mk at mid range when he stands; use more sweep when he attacks; use more mk at mid range when he jumps; use more c.mk far away when he jumps; avoid spinning_bird_kick |
| cmk | use more c.mk at mid range when he stands |
| sweepatt | use more sweep when he attacks |
| antiair | use more mk at mid range when he jumps |
| avoidsbk | avoid spinning_bird_kick |
| throw | use more throw up close |
| throwhard | always throw up close |

### Guile (seeds 53001-53008; docs/chunli_vs_guile_notes.md)

| arm | lines |
|---|---|
| expert | use more block_low far away when he attacks; use more c.mk at mid range when he stands; use more mk at mid range when he jumps; always throw up close when he crouches; use more sweep at mid range when he attacks |
| throwcrouch | always throw up close when he crouches |
| cmk | use more c.mk at mid range when he stands |
| blockfar | use more block_low far away when he attacks |
| hpatt | use more hp up close when he attacks |
| throw | use more throw up close |
| throwhard | always throw up close |

### Dhalsim (seeds 54001-54008; docs/chunli_vs_dhalsim_notes.md)

| arm | lines |
|---|---|
| expert | always throw up close; use more block_low at mid range when he attacks; use more mk at mid range when he stands; use more mk at mid range when he jumps; avoid spinning_bird_kick |
| blockmid | use more block_low at mid range when he attacks |
| blockfar | use more block_low far away when he attacks |
| mkstand | use more mk at mid range when he stands |
| antiair | use more mk at mid range when he jumps |
| chpatt | use more c.hp up close when he attacks |
| throw | use more throw up close |
| throwhard | always throw up close |
