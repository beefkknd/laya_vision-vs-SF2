# Chun-Li vs Zangief (SF2 World Warrior), incl. CPU Zangief -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md` and follows `docs/chunli_vs_honda_notes.md` (not repeated here).
Version tags: **[WW-arc]** arcade WW source, **[SNES]** SNES SF2 (WW port) source, **[WW-any]** source says "Classic"/WW
without naming the platform, **[SSF2-SNES]** Super SF2 on SNES (a LATER version; its CPU scripts are not WW's),
**[proj]** this repo's own logged results. No source found that describes the WW CPU Zangief's script directly
(sf2platinum's AI article shows only Ryu's script; its generic rules - CPU reads the committed input, gets more
aggressive as the timer runs down - apply to him too).

## 1. Strategy summary

### What CPU Zangief does
| Pattern | Source | Confidence |
|---|---|---|
| Goes for close grabs and the Spinning Pile Driver (SPD) whenever you are close; "not shy" about the SPD | [SNES] RJHarrison guide; [SNES] gemubaka (WW on SNES) | medium-high (two SNES-WW sources agree) |
| Also jumps in; neutral-jump kicks beat both his walk-in and his jump-in (Ryu/Ken advice) | [SNES] gemubaka | medium (the jump answer is not available to our bot) |
| Walks in, throws a punch or two, backs off, repeats; a long normal timed as he walks into its range often hits because he cannot block in time | [SSF2-SNES] SephKatana single-player FAQ | low-medium (later game) |
| Jumps over slow attacks and lands on you | [SSF2-SNES] same FAQ | low-medium |
| Knockdown moves are the best tools against him; close quarters is the danger zone | [SNES] RJHarrison | medium |

### His tools and weaknesses (WW frame data unless noted)
- **Throw range beats hers by a lot.** Zangief's normal throws/holds: range 69 from axis (advantage 24). SPD: range 64
  (advantage 35), 44 damage. Chun's ground throw: range 48 (advantage 23) [WW-arc SuperCombo Zangief + Chun]. A WW
  Zangief guide adds that in the original game his grabs work "a step away" and the SPD has similar reach [WW-any
  JCulbert]. High (data). So any distance where her throw reaches is also inside both of his throw ranges.
- **Double Lariat (WW version)**: 11-frame startup, **-25 on block** [WW-arc]; in the original game it can be ducked and
  he is open to low attacks during it, so it should not be used up close [WW-any JCulbert]. High/medium. Implication:
  a low attack (c.mk, sweep) beats it, and a blocked Lariat is a free punish.
- **His sweep (cr.hk)** is his main neutral tool vs Chun; it is whiff-punishable [WW-arc SuperCombo Chun]. SuperCombo
  lists it +1 (4 startup / 8 active / 13 recovery) - read as "safe if it connects", not a free punish on block.
- **His anti-airs**: close lp, cr.hp, Lariat, and throwing her as she lands [WW-arc]. A Gief guide says Chun's air
  attacks usually beat his anti-airs [WW-any JCulbert] - jump-based, not expressible.
- **Chun's SBK is "a gift"** for him: he crouches under it and hits her as she passes [WW-any JCulbert]. High.
- Zangief's own plan vs Chun: walk up patiently threatening sweep; even if it trades with her cr.mk he can walk in and
  tick into an SPD [WW-arc SuperCombo Zangief]. One knockdown starts an SPD loop (no reversals in WW).

### Chun-Li's plan (every source agrees: keep him out, be patient)
- **Far / mid**: use her fast walk and **cr.mk** to keep him out; be patient; **whiff-punish his sweep** [WW-arc
  SuperCombo Chun]. Hit him as he walks into range [SSF2-SNES]. Knockdowns (sweep) are her best reward [SNES].
- **vs Lariat**: go low (c.mk / sweep) or block and punish (-25) [WW-arc data + WW-any JCulbert].
- **vs jumps**: her usual anti-airs (far mk/hk/hp/lp; cr.mk vs jumps from far) [WW-arc Chun]. His jump is short/slow.
- **Close**: avoid. If he sweeps her clean or lands one SPD, the round is effectively over [WW-arc SuperCombo Chun].
  Blocking up close does not help - throws are unblockable.
- **Her throw**: every range number says her throw is a bad idea (his reach 64-69 vs her 48). BUT vs Honda (reach 64)
  the same argument held and **[proj]** found "use more throw up close" helped anyway (+50 hp/round vs Honda,
  +42 Ryu, +64 Ken; ab71934). SF2 throws are resolved by who inputs first at range, and the CPU does not always react.
  Verdict: **contested - worth testing precisely because sources and our data disagree**; expect a smaller gain than
  vs Honda (SPD does 44 and leads to a knockdown loop, so each lost throw exchange costs more).
- **Lightning Legs**: one SNES guide calls them great when the opponent is cornered [SNES RJHarrison, generic, not
  Gief-specific]. Low confidence vs a grappler (slow startup up close = SPD bait).

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line(s) | Fidelity |
|---|---|---|---|
| 1 | Keep him out with cr.mk as he walks in | `use more c.mk at mid range when he stands` | exact (poke); "as he walks into range" approximated by mid range + stands |
| 2 | Whiff-punish his sweep | `use more sweep at mid range when he attacks` | approximate (no "he whiffed" state; "attacks" also covers the moment he is still active) |
| 3 | Low attack beats / punishes the WW Lariat (-25 on block) | `use more sweep when he attacks` (any range) or `use more c.mk up close when he attacks` | approximate (Lariat and sweep both read as "attacks") |
| 4 | Anti-air his jumps | `use more mk at mid range when he jumps`; `use more c.mk far away when he jumps` | exact |
| 5 | Never use SBK (he ducks and punishes) | `avoid spinning_bird_kick` | exact |
| 6 | Stay out of close range / don't walk in | - (`avoid forward` is not reliably followed) | cannot express (no walk back; only single moves can be avoided) |
| 7 | Don't block up close (throws are unblockable, SPD tick) | `avoid block_low up close when he stands` | approximate, untested, risky (may also drop blocks vs his close normals) |
| 8 | Don't throw him (his range is longer) / or throw anyway (our data) | `avoid throw up close` vs `always throw up close` | exact both ways; contested - test, don't assume |
| 9 | Punish when stunned | `always hp up close when he is stunned` | approximate (rare state) |
| 10 | Neutral-jump kicks vs walk-in and jump-in | - | cannot express (no jump) |
| 11 | Jump-ins / air-to-air (her air beats his anti-airs) | - | cannot express |
| 12 | Avoid getting knocked down (SPD loop on wake-up) | - | cannot express (no knockdown / wake-up state) |
| 13 | Patience / win on life lead | - | cannot express (no life-lead / timer condition) |
| 14 | Lightning Legs when he is cornered | `use more lightning_legs up close when he stands` | approximate (no "cornered" state); low confidence |

## 3. Recommendations

### Expert set (5 lines)
1. `use more c.mk at mid range when he stands`
2. `use more sweep when he attacks`
3. `use more mk at mid range when he jumps`
4. `use more c.mk far away when he jumps`
5. `avoid spinning_bird_kick`

Rationale: poke him at mid range, punish his sweep/Lariat low, anti-air his jumps, never SBK. It contains no close-range
line because the expert advice for close range is "don't be there", which the grammar cannot say; and no throw line on
purpose, so the throw is tested as its own arm (pre-registered, `docs/prereg_new_opponents.md`). Only one avoid line.

### Single-line candidates to test alone
- `use more c.mk at mid range when he stands` (strongest expert agreement: SuperCombo Chun + Gief pages)
- `use more sweep when he attacks` (whiff-punish sweep + beats/punishes WW Lariat)
- `use more throw up close` (the pre-registered arm; CONTESTED: his reach 64-69 vs her 48; if laya-vision under-rates it,
  also run `always throw up close`)
- `avoid spinning_bird_kick` (exact and uncontested; may be a no-op if she rarely SBKs)
- `use more mk at mid range when he jumps`
- `use more lightning_legs up close when he stands` (low confidence; only a generic SNES tip)

## Sources
- SuperCombo WW Zangief (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Zangief (throw range 69, SPD range 64, Lariat -25, sweep data, anti-airs; vs Chun: walk up threatening sweep, then SPD tick)
- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li (throw range 48; vs Zangief: walk speed + cr.mk, patience, "his sweep is whiff punishable")
- JCulbert, SF2 Zangief guide (SNES listing, covers "Classic" to SFA2; archived): https://web.archive.org/web/2023/https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/849 (Classic grabs a step away; Classic Lariat duckable / weak low; SBK is a gift for Gief)
- RJHarrison, SNES SF2 guide: https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (vs Zangief: close quarters bad, knockdowns good, CPU uses SPD freely)
- gemubaka, "Fighting Zangief in Street Fighter II" (WW on SNES + later versions): https://gemubaka.com/2019/11/30/fighting-zangief-in-street-fighter-ii/ (CPU goes for grabs and jump-ins; neutral jump kicks beat him)
- SephKatana, Super SF2 Single-Player FAQ (LATER version, SNES): https://gamefaqs.gamespot.com/snes/588757-super-street-fighter-ii/faqs/33971 (CPU walks in, punches, backs off; hit him as he walks into range; he jumps over slow attacks)
- sf2platinum, The AI Engine: https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/ (generic CPU rules; no Gief script shown)
- Project: commit ab71934 (throw +42/+64/+50 hp/round vs Ryu/Ken/Honda); docs/prereg_new_opponents.md (throw arm pre-registered)
