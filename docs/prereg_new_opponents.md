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
