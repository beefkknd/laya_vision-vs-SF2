# What players watch vs the U-arm questions: synthesis (2026-10-01)

Two independent web-research reports (Claude: 2026-10-01_perception_claude.md; GPT-6: 2026-10-01_perception_gpt6.md),
same brief, blind to each other. Sources are mostly players' own words: Maj's Footsies Handbook (sonichurricane.com),
SRK/supercombo archives, Thelo's reaction guide (Sirlin forums), kayin.moe, EventHubs, SuperCombo wiki.

## Both agree: what a player's "muscle memory" watches
1. **Range, in bands, and whether it is closing or opening.** Players think in throw / poke / jump-in / full-screen
   range; the gap and its change, not their own character.
2. **His commitment and recovery.** The opening is a whiffed attack still recovering, not "he is attacking". Startup
   (danger) and recovery (opportunity) look similar but call for opposite answers. Against SF2's input-reading CPU,
   recovery is the reliable opening (Claude).
3. **Jumps with direction.** Anti-air is a trained reflex: jumping AT me vs away, rising vs landing.
4. **Projectiles** as their own threat: on screen, how close, coming at me.
5. **Who can act now, both sides:** hit/block stun, knocked down / getting up, dizzy.
6. **Hit vs block** of the last exchange (hit-confirm).
7. **Corner.**
8. The per-move question is a DECISION, not a perception; keep it soft, and label it from real outcomes
   (both suggest savestate branching - the owner's "B C D E against A" mode).

## Where they differ
- Health: GPT-6 keeps it and adds the lead and the clock; Claude demotes it to risk context for the planner. (The
  owner already kept the bars: a human sees them.)
- GPT-6 stresses honesty about two frames: some states (exact recovery, throw invulnerability, hidden stun) are not
  visible; use soft / masked labels where identical pictures have different RAM truths, and align RAM with the
  rendered frame (display delay). SNES Super SF2 has no super meter.

## Our 5 questions against this
| # | our question | verdict | change |
|---|---|---|---|
| 1 | how far (close/mid/far) | aligned, too coarse | 4 bands (throw / poke / mid / far) + trend (closing / steady / opening) |
| 2 | is he attacking (yes/no) | wrong cut | his phase: neutral / attacking (startup-active) / recovering after a miss / blocking / being hit |
| 3 | is he in the air (yes/no) | aligned, no direction | grounded / jumping at me / jumping away-straight up / landing |
| 4 | move X better than walking in (soft) | a decision, keep soft | keep; labels from outcomes (table now, same-frame branching later) |
| 5 | health bars | owner keeps | keep (+ lead, per GPT-6) |
| new | projectile | missing | none / far / near, coming at me |
| new | who can act (me, him) | missing | free / stunned / knocked down / dizzy |
| new | corner | missing | me / him / neither |

Hit vs block of the last exchange largely folds into "his phase" (blocking / being hit) and "who can act".

## Labels
Everything is derivable from RAM fields we already read (sf2/emu/vs.py: x, y, state, sub, react, dizzy, life, shot1/2
and their x) plus a few frames of lookahead. But the value collection logged only decision summaries (gap, opp_state,
opp_air, opp_reaction, opp_shot): the frames we have lack the per-frame RAM needed for trend, recovery phase,
projectile distance, corner and dizzy. A cheap re-collection (headless, no Qwen, ~1-2 h) that logs the RAM rows around
each decision would give all of them, aligned with the saved frames.
