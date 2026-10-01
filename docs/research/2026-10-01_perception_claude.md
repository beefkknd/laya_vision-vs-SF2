# Claude (research subagent, web search): what SF2 players watch vs the U-arm questions (2026-10-01)

Brief: docs/research/perception_research_brief.md. Independent, blind to the GPT-6 report.

## Q1. What experienced players say they watch and react to

### A. The screen moment (general, holds for any opponent)
1. **Distance first.** Players think in range bands - nose-to-nose (everything hits, throws), poke/footsie range,
   mid-screen (jump-in and projectile range), full screen - and "always know where you are and where you want to be"
   (inthirdperson, Space Management: https://inthirdperson.com/2013/10/17/universal-fighting-game-guide-space-management/).
   In footsies you watch the opponent's movement and the gap, not your own character (Steam Footsies Compendium:
   https://steamcommunity.com/sharedfiles/filedetails/?id=3439639318). ST has "a deep emphasis on spacing" with "huge
   pushback on everything" (https://www.fightinggameguide.com/superturbo.html).
2. **Whiffs and recovery are the main opening.** Maj's Footsies Handbook: punish the whiff, or walk up and throw during
   the recovery period (https://sonichurricane.com/?p=691, https://sonichurricane.com/?p=759). Players react to "his limb
   is out and missed / he is still recovering", not to "he is attacking".
3. **Jumps and anti-air are a distance-gated reflex.** "Notice the distance when people are more likely to jump"; train
   anti-airs until "your hands act before you think"; most jumps follow pokes, fireballs or whiffs
   (https://archive.supercombo.gg/t/any-tips-on-anti-airing-on-reaction/168417). What you watch depends on distance
   (https://kayin.moe/reactions); one-button anti-airs are easier on reaction (https://wiki.supercombo.gg/w/Anti-Air).
4. **Projectiles.** Jump fireballs on reaction from about half a screen and punish the thrower's recovery; a
   pre-emptive jump gets anti-aired (https://wiki.supercombo.gg/w/Super_Street_Fighter_2_Turbo/Blanka/Strategy).
5. **Hit vs block (hit-confirm)** looks different on screen (spark, hitstun vs blockstun)
   (https://streetfighter.fandom.com/wiki/Hit_Confirmation;
   https://archive.supercombo.gg/t/on-impact-freeze-and-hit-or-blockstun-after-single-attacks-or-in-combos/109191).
6. **Who can act: knockdown, wake-up, dizzy.** Wake-up throw invulnerability and reversals on the first recovery frame
   (https://www.eventhubs.com/guides/2009/jan/12/basic-gameplay-details-super-street-fighter-2-turbo-hd-remix/); dizzy
   shows as stars/birds and the dizzied player cannot block (https://streetfighter.fandom.com/wiki/Stun).
7. **Corner.** A cornered player cannot walk back; the attacker controls range (https://sonichurricane.com/?p=852).
8. **Health** is risk context ("always three or four hits from dying"), not a moment-to-moment reaction cue.
9. **Muscle memory** narrows what you watch by context (the "mental stack"); the trained autopilot frees attention for
   decisions (https://kayin.moe/reactions).

### B. Opponent knowledge (the language model, not System 1)
Character ranges, prejump speed, which normals beat which (https://wiki.supercombo.gg/w/Super_Street_Fighter_2_Turbo/Dhalsim);
habits and conditioning. CPU specifics: the SF2 CPU reads inputs and reacts on the frame you press
(https://www.nintendolife.com/news/2019/09/video_how_the_cpu_used_to_cheat_in_street_fighter_ii) - so against it the
most valuable reads are its commitment and recovery, not its startup.

## Q2. The five questions
| question | matches players? | problem |
|---|---|---|
| 1 distance close/mid/far | yes, #1 | too coarse (throw / poke / jump-in mid / full screen) and no trend |
| 2 is he attacking | partly | the wrong cut: startup (danger) vs whiffed-recovering (opportunity) |
| 3 is he in the air | yes | no direction (at me vs away), misses prejump |
| 4 move X vs walking in | a decision, not a perception | fine as a soft value head if labelled by real outcomes (rollouts) |
| 5 health bands | weak for System 1 | risk context for the planner |

Missing (general, visible): projectiles; who can act (both sides); hit vs block of the last exchange; corner; approach
or retreat.

## Proposed questions (priority order; RAM fields from sf2/emu/vs.py, lookahead allowed for labels)
1. His phase: neutral / starting-or-active attack / whiffed-recovering / blocking / being hit (state 0x0A/0x0C + no
   contact on me + still attacking k frames later; 0x08 or 0x0E with a block react; 0x0E other react or 0x14).
2. Range band (throw / poke / mid / far, thresholds calibrated from logged throws and connects) + trend.
3. Air: no / jumping at me / neutral-or-away / about to land (prejump 0x04 on the ground; y; x toward me; lookahead).
4. Projectile: none / far / near / about to hit (shot1/shot2 flags, |shot_x - p1_x|, change of shot_x).
5. Who can act, me and him: free / hit-or-block stun / knocked down or getting up / dizzied (0x0E/0x14, dizzy, frames
   until free).
6. Last contact: none / my hit landed / blocked / I got hit / I blocked.
7. Corner: me / him / neither (x near the stage's observed extremes).
8. Move X vs walking in (better / equal / worse), labelled by savestate branching, not opinion.
Dropped: health bands (one coarse "anyone near death" flag at most).

Recommendation: re-center System 1 on his commitment (especially whiffed/recovering), range band + trend, air
approach, incoming projectile, who can act; hit/block and corner as cheap extras; the per-move value only as a head
labelled by rollouts. Against the input-reading CPU, "he is committed / recovering" and "he is jumping at me" are
worth the most.
