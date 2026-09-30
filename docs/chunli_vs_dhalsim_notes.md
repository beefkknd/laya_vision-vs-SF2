# Chun-Li vs Dhalsim (SF2 World Warrior), incl. CPU Dhalsim -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md` and follows `docs/chunli_vs_honda_notes.md` (not repeated here).
Version tags: **[WW-arc]** arcade WW source, **[SNES]** SNES SF2 (WW port) source, **[SSF2-SNES]** Super SF2 on SNES
(a LATER version; its CPU scripts are not WW's), **[HF]** Hyper Fighting (LATER version), **[proj]** this repo's own
logged results (`docs/qwen_learning.md`). No source describes the WW CPU Dhalsim's script directly; the only CPU-pattern
list found is for Super SF2 on SNES.

## 1. Strategy summary

### What CPU Dhalsim does
| Pattern | Source | Confidence |
|---|---|---|
| Yoga Fire, then often a standing fierce (long arm) right after it; block both | [SSF2-SNES] SephKatana single-player FAQ | medium-low (later game) |
| Slides (d+K) at you; if blocked he may slide one or two more times; stop blocking low and you are knocked down | [SSF2-SNES] same FAQ | medium-low |
| Hops up and down in place waiting for you; if you approach while he is airborne he drills (Yoga Spear / Mummy). Block or duck the flat drill, then throw him as he lands | [SSF2-SNES] same FAQ; drills also exist in WW [WW-arc SuperCombo Dhalsim] | medium-low |
| Grabs you when close and does real damage (noogie = hold, 20+4 per hit) | [SSF2-SNES] FAQ; damage [WW-arc SuperCombo] | medium |
| Called "extremely susceptible" to crouching fierce | [SSF2-SNES] FAQ | low (their c.hp slides for some characters; Chun's does not) |
| Charge/motion specials cost the CPU nothing; it reads your input, not the screen | [WW-arc] sf2platinum (see WW notes) | high |

### His tools and weaknesses (WW data unless noted)
- **Throws: his range is longer.** Yoga Smash (noogie) and Yoga Throw: range 64 from axis, advantage 29
  [WW-arc SuperCombo Dhalsim]; Chun's ground throw 48 / 23 [WW-arc SuperCombo Chun]. SuperCombo Chun page: he has more
  throw range, so do not be impatient. High (data).
- **But her normals win the trades.** SuperCombo Chun (WW): she can out-poke even Dhalsim, whose normals have the best
  range but lower priority; her j.mk and most normals "give him serious problems"; a decent matchup for her. [HF]
  Shoryuken wiki Dhalsim matchup page: Chun beats Sim on priority and often wins throw exchanges (player opinion, later
  version). Medium-high.
- **His limbs are hittable.** SNES RJHarrison guide: many of his moves, even the stretch limbs, leave him open; his air
  game is poor; close range is the best place to fight him ("just pound away"). Far st.hp / st.hk / cr.hp each have
  ~19 frames of recovery [WW-arc frame data]. Medium-high.
- **Close range is his worst range** ([HF] Shoryuken: once the opponent is in, Sim has little to make space; [SNES]
  RJHarrison agrees). Caveat: his close normals are very plus on block (close mk +7, close hk +9, close cr.mp/mk +8)
  [WW-arc], and the Yoga Flame is his close-range answer (it cannot be slid under).
- **Anti-airs**: close mp (listed AA), far st.hk / st.mk, slides [WW-arc SuperCombo, HF]. Jumping at him is the human
  plan (SRK WW Chun page: jumping is the main answer to Yoga Fire; SNES Usenet Chun guide opens every round with a
  jump-in held-LK Lightning Legs, then presses him as he stands up). Not available to our bot.
- **SBK**: SuperCombo WW Dhalsim: his crouching fierce counters her SBK. Plus the general WW advice to never use it.

### Chun-Li's plan (sources agree: get in, win with priority, do not rush the throw)
- **Far**: his space (Yoga Fire, long limbs). Block the fireball (and the fierce that follows); the humans jump over or
  toward him - we cannot. She must cover the distance on foot (fastest walk in WW).
- **Mid**: his limb/slide range. Block low vs slides (repeat slides!), poke with far mk / cr.mk into his extended limbs
  (priority win), anti-air his floaty hops/drills with far mk / hk.
- **Close**: her range. Normals and pressure; the sources disagree only on the throw: his reach is longer, but her
  priority is better, a blocked/landed drill is a documented throw punish, and [proj] the throw was +42/+64/+50 hp/round
  vs Ryu/Ken/Honda even though Honda also out-ranges her (`docs/component_boundaries.md`). Verdict: a good idea to TEST,
  not a known win; watch for his noogie trades.
- **[proj]** vs Dhalsim: no advice already wins 32/90 rounds; the generic playbook HURT (-18 hp/round, 32 -> 25 wins) and
  Qwen's opponent-specific advice helped (+17) - so opponent-specific lines matter here more than elsewhere.

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line(s) | Fidelity |
|---|---|---|---|
| 1 | Get in close; fight him at close range | (`use more forward far away when he stands`) | cannot reliably (forward lessons are not followed reliably); the close-range goal itself cannot be stated |
| 2 | Block Yoga Fire and the fierce that follows | `use more block_low far away when he attacks` | exact (low block stops the fireball and his arm) |
| 3 | Keep blocking low vs repeated slides | `use more block_low at mid range when he attacks` | exact (drills from the air need block_high, see #7) |
| 4 | Out-poke his limbs with her higher-priority normals | `use more mk at mid range when he stands` / `use more c.mk at mid range when he attacks` | approximate (no "his limb is extended" state; "attacks" also covers Yoga Fire) |
| 5 | Punish his long-recovery limbs | `use more hk at mid range when he attacks` | approximate (no whiff/recovery state) |
| 6 | Anti-air his hops / drills with far mk | `use more mk at mid range when he jumps` | exact |
| 7 | Block a drill if not anti-airing | `use more block_high at mid range when he jumps` | approximate (competes with #6) |
| 8 | Throw him (after a blocked drill; generally at close) | `always throw up close` / `always throw up close when he attacks` | approximate (no "just landed" state; his longer throw makes this contested) |
| 9 | Pound him at close range with normals | `use more c.hp up close when he attacks` / `use more sweep up close when he stands` | approximate; c.hp tip is [SSF2-SNES], low confidence |
| 10 | Avoid SBK (his c.hp counters it) | `avoid spinning_bird_kick` | exact |
| 11 | Jump over Yoga Fire / jump-in Lightning Legs | - | cannot express (no jump) |
| 12 | Don't be impatient vs his longer throw | - | cannot express (no tempo/"wait" verb) |
| 13 | Throw him when he lands after the drill / rematerializes | - | cannot express (no landing / teleport state; teleport is not in WW anyway) |
| 14 | Heavy punish when stunned | `always hp up close when he is stunned` | approximate (rare state) |

## 3. Recommendations

### Expert set (5 lines)
1. `always throw up close`
2. `use more block_low at mid range when he attacks`
3. `use more mk at mid range when he stands`
4. `use more mk at mid range when he jumps`
5. `avoid spinning_bird_kick`

Rationale: close range is where Chun wins (every source), the throw is the tool laya-vision under-rates (hence
"always"); block low covers his slides/limbs; far mk is her priority poke and anti-air; one avoid line only. The throw is
contested by his 64 vs 48 range - line 1 is also the pre-registered throw test. Risk: without a reliable walk-in lesson
she may stay at mid/far where he is strongest; watch the share of time at close range.

### Single-line candidates to test alone
- `always throw up close` (the pre-registered throw arm; contested by his range)
- `use more block_low at mid range when he attacks` (slides repeat; most agreed defensive tip)
- `use more block_low far away when he attacks` (Yoga Fire + fierce follow-up)
- `use more mk at mid range when he stands` (priority out-poke)
- `use more mk at mid range when he jumps` (anti-air his hops/drills)
- `use more c.hp up close when he attacks` (low confidence, SSF2 CPU tip)

## Sources
- SuperCombo WW Dhalsim (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Dhalsim (throw range 64 / adv 29; noogie 20+4n; limb recovery; close AA mp; "Her Spinning Bird Kick can be countered with Crouch Fierce")
- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li (out-pokes Dhalsim; j.mk; his throw range; decent matchup)
- Shoryuken wiki Chun-li (WW) (archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Chun-li_(WW) (jump is the main answer to Yoga Fire)
- Shoryuken wiki HF Dhalsim (LATER version, archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Street_Fighter_2:_Hyper_Fighting/Dhalsim (weak once opponent is close; Chun matchup opinion: priority, throws)
- RJHarrison SNES SF2 guide (archived): https://web.archive.org/web/2024/https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (limbs leave him open; poor air game; fight him close)
- SephKatana, Super SF2 Single-Player FAQ (LATER version, SNES, archived): https://web.archive.org/web/2023/https://gamefaqs.gamespot.com/snes/588757-super-street-fighter-ii/faqs/33971 (CPU patterns: fire then fierce, repeated slides, hop-and-drill, throw after flat drill, c.hp)
- "Complete Chun Li Strategy Guide: SNES SFII" (Usenet; search snippets only): https://groups.google.com/g/rec.games.video/c/6uqBlDC7xMU (open with jump-in held-LK Lightning Legs, then press him)
- Project results: /Users/worker/work/hobby/laya_vision-vs-SF2/docs/qwen_learning.md (Dhalsim: playbook -18, Qwen +17); /Users/worker/work/hobby/laya_vision-vs-SF2/docs/component_boundaries.md (throw +42/+64/+50 vs Ryu/Ken/Honda)
