# Chun-Li vs E. Honda (SF2 World Warrior), incl. CPU Honda -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md` and follows `docs/chunli_vs_ryu_notes.md` (not repeated here).
Version tags: **[WW-arc]** arcade WW source, **[SNES]** SNES SF2 (WW port) source, **[SSF2-SNES]** Super SF2
on SNES (a LATER version; its CPU scripts are not WW's), **[HF]** Hyper Fighting (LATER version), **[proj]**
this repo's own logged results (`docs/qwen_learning.md`). No source describes the WW CPU Honda's script
directly (sf2platinum's AI article has nothing Honda-specific beyond "the CPU needs no charge time").

## 1. Strategy summary

### What CPU Honda does
| Pattern | Source | Confidence |
|---|---|---|
| Throws / grabs you when you get close, then follows with Hundred Hand Slap (HHS) | [SSF2-SNES] SephKatana single-player FAQ | medium (later game, but consistent with every human source: "one throw" starts Honda's loop) |
| Jumps at you often; answer with your best anti-air from a distance | [SSF2-SNES] same FAQ | medium |
| Torpedo (Sumo Headbutt) from far; tell: crouches far away first (SSF2 says a crouching kick precedes it) | [SSF2-SNES] same FAQ | low-medium (tell is SSF2-specific) |
| Crouching for a couple of seconds = a special is coming, block | [SSF2-SNES] same FAQ | low-medium |
| Charge moves need no real charge for the CPU; it can fire a Headbutt as easily as a normal | [WW-arc] sf2platinum, The AI Engine | high (from ROM) |
| Leans on HHS in the corner; HHS chips even when blocked | [SNES] RJHarrison guide (tips for Honda) | medium |

### His tools and weaknesses (WW frame data unless noted)
- **Sumo Headbutt**: jab -11 on hit / -14 on block; strong and fierce knock down on hit and are only **-4 on block**
  [WW-arc SuperCombo Honda]. So blocking it is right, but only a very fast answer punishes the heavy versions. Throws in
  SF2 are near-instant, so a throw is the natural punish if he ends up next to her. [HF] adds: Honda is invincible until
  he leaves the ground, so hitting the startup does not work. High (data) / medium (throw-punish inference).
- **SNES/SSF2 answers to the Headbutt** all need a jump: jump straight up and land with the Head Stomp [SNES Usenet Chun
  guide, via search snippets], or jump over and throw him as he stops [SSF2-SNES]. Not available to our bot.
- **HHS**: startup 5/8/11 frames by strength [HF]; beats many normals at mid range (notes); weak to projectiles (Chun has
  none in WW).
- **His throws outrange hers**: Honda's throw range 64 from axis (advantage 36) vs Chun's ground throw 48 (advantage 23)
  [WW-arc SuperCombo Honda + Chun]. The SNES Chun guide calls her throw not safe against Honda (snippet). High (data).
  This is the main difference from Ryu, where her throw had MORE range.
- **Anti-airs**: his far st.hp, close st.hk and Headbutt [WW-arc]. His cr.mp beats Spinning Bird Kick [WW-arc].
- Honda's own advice vs Chun: approach patiently, fish for one throw, then loop her [WW-arc SuperCombo Honda].

### Chun-Li's plan (every source agrees: hit and run / keep away)
- **Far**: stay here; cr.mk anti-airs his far jumps [WW-arc SuperCombo Chun]. Block a torpedo.
- **Mid**: do not walk in. Block his attacks (Headbutt, HHS). Anti-air jumps with far mk / hk / hp / lp [WW-arc].
  Shoryuken wiki WW: fight him in the air rather than on the ground [WW-arc SRK] - jump-based, not expressible.
- **Close**: the danger zone; his throw wins the range war and CPU throws on contact [WW-arc data, SSF2-SNES]. Only go in
  to punish (after a blocked Headbutt) or when he is stunned.
- **Specials**: SBK is bad (his cr.mp beats it; unsafe -6..-11). Lightning Legs at point blank are +9 if all 3 hits are
  blocked [WW-arc SuperCombo Chun]; the SNES Chun guide builds a whole anti-Honda routine on jump-in Lightning Legs then
  jump out (snippet: beats him "90% of the time"). Ground-only version is untested - low confidence.
- **[proj]** agrees: walking in at mid range hurts (lessons "use more forward at mid range when he stands" -27.0 hp/round,
  "...when he jumps" -15.3); blocking at mid range when he attacks was verified as better there; Honda's ground attacks
  cost her ~12 hp per hit. "avoid spinning_bird_kick at mid range when he stands" looked harmful vs Honda but was
  confounded with the walk-in lessons.

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line(s) | Fidelity |
|---|---|---|---|
| 1 | Keep away, do not walk into him | `avoid forward at mid range` | exact (for "don't approach"); cannot say "walk back" |
| 2 | Block Headbutt / HHS at mid range | `use more block_low at mid range when he attacks` | exact (low block also stops Headbutt and HHS; his jump attacks need block_high, see #3) |
| 3 | Block his jump-ins if not anti-airing | `use more block_high at mid range when he jumps` | approximate (competes with #4) |
| 4 | Anti-air his far jumps with cr.mk | `use more c.mk far away when he jumps` | exact (hit depends on spacing in the "far" bin) |
| 5 | Anti-air with far mk (or hk/hp) | `use more mk at mid range when he jumps` | exact |
| 6 | Punish a blocked Headbutt with a throw as he lands next to her | `use more throw up close when he attacks` | approximate: no "he just whiffed/was blocked" state; "attacks" also covers HHS, where throwing is riskier |
| 7 | Don't fight up close (his throw outranges hers) | `avoid throw up close when he stands` / generic close avoidance | cannot express "stay out of close range"; only single moves can be avoided |
| 8 | Jump straight up + Head Stomp vs torpedo | - | cannot express (no jump) |
| 9 | Jump over torpedo and throw as he stops | - | cannot express (jump + sequence) |
| 10 | Jump-in Lightning Legs, then jump out | ground stand-in: `use more lightning_legs up close when he stands` | approximate at best, low confidence |
| 11 | Avoid SBK (his cr.mp beats it) | `avoid spinning_bird_kick` | exact, but [proj] warns the avoid-SBK lesson was confounded vs Honda |
| 12 | Heavy punish when stunned | `always hp up close when he is stunned` | approximate (rare state) |
| 13 | Throw loop / meaty on wake-up | - | cannot express (no knocked-down state) |
| 14 | "Get some damage, then run" (life-lead turtling) | - | cannot express (no life-lead / timer condition) |
| 15 | Watch for his crouch-then-torpedo tell | `use more block_low far away when he crouches` | approximate (tell is from SSF2) |

## 3. Recommendations

### Expert set (5 lines)
1. `avoid forward at mid range`
2. `use more block_low at mid range when he attacks`
3. `use more c.mk far away when he jumps`
4. `use more mk at mid range when he jumps`
5. `use more throw up close when he attacks`

Rationale: keep-away + block + anti-air is the consensus plan; line 5 is her punish for a blocked Headbutt. Only one
avoid line (past lesson: avoid-heavy sets made her passive). Note the tension: line 1 plus line 2 may make her very
passive at mid range; watch attack share and round length (Honda rounds are already the longest).

### Single-line candidates to test alone
- `avoid forward at mid range` (strongest agreement: every source + [proj])
- `use more block_low at mid range when he attacks`
- `use more c.mk far away when he jumps`
- `use more throw up close` (the Ryu winner; here CONTESTED - his throw has more range - worth testing precisely because
  the sources and our Ryu result disagree)
- `use more throw up close when he attacks` (punish version)
- `use more lightning_legs up close when he stands` (low confidence; +9 on full block, SNES guide's favourite tool)

## Sources
- SuperCombo WW E. Honda (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/E._Honda (Headbutt -4 on block; throw range 64; "All Honda needs is one throw")
- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li (throw range 48; LL +9 point blank; vs Honda: damage then run, cr.mk anti-air)
- Shoryuken wiki Chun-li (WW) (archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Chun-li_(WW) (vs Honda: jumping beats ground play)
- Shoryuken wiki HF E. Honda (LATER version, archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Street_Fighter_2:_Hyper_Fighting/E._Honda (invincible until airborne; HHS startup)
- SephKatana, Super SF2 Single-Player FAQ (LATER version, SNES, archived): https://gamefaqs.gamespot.com/snes/588757-super-street-fighter-ii/faqs/33971 (CPU Honda throws when close then slaps; jumps often; torpedo tells)
- RJHarrison SNES SF2 guide (archived): https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (close range is risky vs Honda)
- "Complete Chun Li Strategy Guide: SNES SFII" (Usenet; full text not reachable, search snippets only): https://groups.google.com/g/rec.games.video/c/6uqBlDC7xMU (Head Stomp vs torpedo; jump-in Lightning Legs; her throw not safe vs Honda)
- sf2platinum, The AI Engine: https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/ (CPU needs no charge)
- Project results: /Users/worker/work/hobby/laya_vision-vs-SF2/docs/qwen_learning.md (walk-in lessons -27.0 / -15.3 vs Honda; block at mid verified)
