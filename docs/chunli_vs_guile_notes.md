# Chun-Li vs Guile (SF2 World Warrior), incl. CPU Guile -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md`, `docs/chunli_vs_ryu_notes.md`, `docs/chunli_vs_honda_notes.md` (not repeated).
Version tags: **[WW-arc]** arcade WW source, **[SNES]** SNES SF2 (WW port) source, **[SSF2-SNES]** Super SF2 on SNES
(a LATER version; its CPU scripts are not WW's), **[proj]** this repo's logged results. No source found describes
the WW CPU Guile's script directly; the only CPU-pattern sources are a SNES Usenet Chun guide (search snippets only;
full text blocked, 429) and a later-version SSF2 FAQ.

## 1. Strategy summary

### What CPU Guile does
| Pattern | Source | Confidence |
|---|---|---|
| Round start: walks toward her (sometimes booms or backs off), then **crouches a few steps away** | [SNES] Usenet Chun guide (snippet) | medium (right game, snippet only) |
| Crouching for a couple of seconds = baiting a Flash Kick; approach, crouch as you arrive so it whiffs, **throw him as he lands** | [SSF2-SNES] SephKatana FAQ | low-medium (later game, but consistent with the SNES snippet's "he crouches") |
| Alternates boom volleys with advancing; when advancing he usually **jumps in** -> anti-air | [SSF2-SNES] same FAQ | medium |
| Boom, hop (b/f+mk knee), boom, walk up, cr.hk sweep | [SSF2-SNES] same FAQ | low (SSF2-specific) |
| Two jabs then a fierce: block it | [SSF2-SNES] same FAQ | low |
| Walks up and throws if she lets him close ("his throws are likely to have priority") | [SSF2-SNES] same FAQ | medium |
| Needs no charge: CPU Guile can Flash Kick from a standing position because the script just says so | [WW-arc] sf2platinum, The AI Engine (ROM) | high |
| CPU reads her committed input (yoke) before it is drawn | [WW-arc] sf2platinum (already in WW notes) | high |

**WW glitches (magic throw, handcuffs, free Flash Kick, scroller)** are human-input tricks and only exist in early
arcade revisions (gone from 911101 on; SRK WW system page). They are not CPU behaviour and are not a factor for the
SNES CPU. Only the "free Sonic Boom" after a mp throw survives in all revisions. Confidence: high that they are
irrelevant here.

### His tools (WW arcade data, SuperCombo)
- **Sonic Boom**: 8f startup, 12 damage, no charge for the CPU. Blocks either way (it is not low). Far away, "when he
  attacks" is mostly this. SRK WW Chun page: Chun can pass over a boom with her Spinning Bird Kick (only tip it gives).
- **Flash Kick**: 24-26 damage, **-24 (lk/mk) / -33 (hk) on block**, long recovery. So a blocked or whiffed Flash Kick
  is her biggest punish window; his main anti-air.
- **Anti-airs**: Flash Kick, cr.hp, close mk, far hk, cr.mk / sweep [WW-arc]. Jumping at him is "suicide" [SNES
  RJHarrison]. Irrelevant for our bot (no jump), but it means the SNES guide's jump plan cannot be copied.
- **Long jab** (good reach, can dizzy by mashing) and a **sweep that outranges hers** ([SNES] Usenet snippet: do not
  trade sweeps toe to toe).
- **Throws**: ground throw range 48 from axis, range advantage **18**. **Chun: 48 / advantage 23** [WW-arc SuperCombo
  both pages]. So unlike Honda (64/36), her throw reaches as far or slightly farther than Guile's. High (data).

### Chun-Li's plan
- WW Chun page on Guile: wait for an opening; **whiff-punish his cr.mk with her cr.mk or sweep**; if he is caught without
  down-charge, her j.mk lands [WW-arc SuperCombo Chun]. WW Guile page, from his side: this is one of his harder
  matchups; he wants her to block a boom and then go for throws/mixups; her cr.mk is faster and recovers better than his,
  and her air normals trouble him [WW-arc SuperCombo Guile].
- **Far**: block booms (low or high both work); do not walk through boom volleys. Anti-air his far jumps with cr.mk.
- **Mid**: footsies with cr.mk / far mk (her strongest ground tools; plus on block); whiff-punish; anti-air jumps with
  far mk / hk / hp. Avoid long sweep exchanges (his sweep reaches farther).
- **Close**: her throw is at least even with his; the CPU crouches close by (charging/baiting) and throws in SF2 grab
  crouchers, so a throw on a crouching Guile is plausible. After a blocked Flash Kick (-24/-33) anything heavy
  punishes. Risk: CPU also throws and reads inputs.
- **Specials**: SBK passes over booms [WW-arc SRK] but is slow/unsafe and his cr.hp/Flash Kick anti-air it; Lightning
  Legs hard to activate in WW. Low value.
- **Her throw vs him**: a **good idea, better than vs Honda** — equal-or-longer range, no sumo-style range deficit,
  and the SSF2 FAQ's anti-crouch plan ends in a throw. Caveat: the only source against it is the later SSF2 FAQ
  ("his throws likely have priority"), and the CPU reads inputs. Confidence medium.
- **[proj]**: Guile is already her best CPU matchup here (rounds won 46 -> 54 of 90 without/with playbook; code coach
  had no Guile data). There is no measured Guile lesson yet (`docs/prereg_new_opponents.md` pre-registers the test).

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line | Fidelity |
|---|---|---|---|
| 1 | Block Sonic Booms far away | `use more block_low far away when he attacks` | exact (boom blocks low or high) |
| 2 | Footsie with cr.mk at mid | `use more c.mk at mid range when he stands` | exact |
| 3 | Whiff-punish his cr.mk with cr.mk / sweep | `use more c.mk at mid range when he attacks` / `use more sweep at mid range when he attacks` | approximate (no "he whiffed" state) |
| 4 | Anti-air his jump-ins | `use more mk at mid range when he jumps`; `use more c.mk far away when he jumps` | exact |
| 5 | Throw him while he crouches close (charging / Flash-Kick bait) | `always throw up close when he crouches` | approximate ("always": the vision model under-rates throw) |
| 6 | Crouch so the bait Flash Kick whiffs, then throw on landing | - | cannot (needs a sequence + "he is landing" state); #5 is the stand-in |
| 7 | Punish a blocked Flash Kick (-24/-33) | `use more hp up close when he attacks` | approximate ("attacks" also covers jab strings/throws) |
| 8 | Don't trade sweeps at mid range (his is longer) | `avoid sweep at mid range when he stands` | approximate (conflicts with #3's sweep version) |
| 9 | Don't walk into boom volleys | (`avoid forward far away when he attacks`) | cannot reliably (lessons about forward are not followed) |
| 10 | Jump straight up and j.hk his crouch / j.mk when he has no charge | - | cannot (no jump) |
| 11 | Pass over booms with SBK | `use more spinning_bird_kick far away when he attacks` | exact but risky (unsafe; Flash Kick / cr.hp beat it) |
| 12 | Avoid SBK generally | `avoid spinning_bird_kick` | exact (contradicts #11) |
| 13 | Throw him when he walks backward | `use more throw up close when he stands` | approximate (SSF2 tip) |
| 14 | Block his jab-jab-fierce up close | `use more block_high up close when he attacks` | approximate (SSF2 pattern) |

## 3. Recommendations

### Expert set (5 lines)
1. `use more block_low far away when he attacks`
2. `use more c.mk at mid range when he stands`
3. `use more mk at mid range when he jumps`
4. `always throw up close when he crouches`
5. `use more sweep at mid range when he attacks`

Rationale: block booms, win mid range with her faster cr.mk, anti-air his advances, throw the crouching CPU, and
whiff-punish his pokes. No avoid lines (past lesson: avoid-heavy sets made her passive). Line 5 is the least certain
(his sweep outranges hers); drop it first.

### Single-line candidates to test alone
- `always throw up close when he crouches` (main hypothesis; throw ranges equal-or-better, CPU crouches close)
- `use more throw up close` (the Ryu/Ken/Honda winner, unconditioned, as the control)
- `use more c.mk at mid range when he stands` (strongest human-source agreement)
- `use more block_low far away when he attacks` (booms)
- `use more hp up close when he attacks` (blocked Flash Kick punish; approximate)
- `use more spinning_bird_kick far away when he attacks` (SRK's only Guile tip; contested)

## Sources
- SuperCombo WW Guile (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Guile (throw 48/18; Boom 8f; Flash Kick -24/-33; vs Chun: harder matchup, her cr.mk faster)
- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li (throw 48/23; vs Guile: wait, whiff-punish his cr.mk with cr.mk/sweep)
- Shoryuken wiki Chun-li (WW) (archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Chun-li_(WW) (SBK passes over booms)
- Shoryuken wiki WW system page (archived): https://web.archive.org/web/2023/http://wiki.shoryuken.com/Street_Fighter_2:_The_World_Warrior (Guile glitches per revision)
- RJHarrison SNES SF2 guide (archived): https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (jumping at Guile is "suicide"; make him come to you, punish a blocked Flash Kick)
- "Complete Chun Li Strategy Guide: SNES SFII" (Usenet; search snippets only): https://groups.google.com/g/rec.games.video/c/6uqBlDC7xMU (CPU walks in and crouches; air hk; his sweep outranges hers)
- SephKatana, Super SF2 Single-Player FAQ (LATER version, archived): https://gamefaqs.gamespot.com/snes/588757-super-street-fighter-ii/faqs/33971 (crouch = Flash Kick bait -> throw on landing; boom/advance/jump; throws if close)
- sf2platinum, The AI Engine: https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/ (CPU Flash Kicks without charge)
- Project: /Users/worker/work/hobby/laya_vision-vs-SF2/docs/qwen_learning.md (Guile 46 -> 54 rounds won of 90), docs/prereg_new_opponents.md
