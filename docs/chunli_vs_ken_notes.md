# Chun-Li vs Ken (SF2 World Warrior), incl. CPU Ken -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md` and `docs/chunli_vs_ryu_notes.md` (neither repeated here).
Version tags: **[WW-arc]** arcade WW source, **[SNES]** SNES SF2 source, **[proj]** this repo's own logs
(`docs/qwen_learning.md`). No Champion Edition / Turbo source was used. SNES AI is a port, not confirmed
identical to the arcade scripts.

## 1. Strategy summary

### Ken vs Ryu in WW: same moves, different CPU
- **Moves / frame data are practically the same.** Ryu and Ken are clones in WW; the SNES guide calls them
  "virtually equal in this game" [SNES RJHarrison]. SuperCombo's WW Ken page lists the same Shoryuken block
  disadvantage (-11 / -22 / -33, jab/strong/fierce) and Hurricane -1 on block as Ryu, one note that Ken's
  Dragon Punch has "wider range than Ryu's", and that Ken's kick throw beats Ryu's [WW-arc]. Confidence: high
  for the clone point, low-medium for the DP-range note (one unsourced wiki line).
- **So the Ryu frame-data advice carries over**: DP does not knock down a grounded opponent and is very
  unsafe; Hurricane hits do not combo or knock down; fireball startup/recovery is sweepable; Chun's throw
  reaches farther than his. High.
- **The CPU is what differs.**

### What CPU Ken does (the part that differs from Ryu)
| Pattern | Source | Confidence |
|---|---|---|
| Uses **Hurricane Kick and Dragon Punch far more** than CPU Ryu, and the **fireball much less** | [SNES] RJHarrison GameFAQs guide | medium-high (one guide, but it agrees with the rest of this table and with our logs) |
| Advice from that guide: knock him out of the air during those two moves, then press strong attacks; stay grounded, don't attack from the air | [SNES] RJHarrison | medium |
| Level 7 opening: back off; he very likely **opens with a Hurricane Kick** -> keep blocking; he **follows it with a Dragon Punch**; as he comes down, walk forward and throw | [SNES] "Complete Chun Li Strategy Guide" (Usenet 1992; full text blocked, search snippets only) | medium |
| Vs an incoming Hurricane: wait until he is ~1 body width away, then standing fierce (hp), stepping forward as you press | same Usenet guide (snippets; given under the Ryu/Ken section) | medium |
| CPU Ryu/Ken **often DP more than once in a row**; DP recovery is long -> throw him as he lands | same Usenet guide | medium |
| CPU Ryu is a readable pattern, CPU Ken is erratic, "Ken was all over the place" | [WW-arc] arcade-museum forum anecdote | low-medium (anecdote, but consistent) |
| **59% of her damage vs Ken comes from his jump-ins**; his ground hits cost her only 5.2 each (Ryu/Honda 11.7-12.3). CPU specials are logged as a plain attack, not a separate state | [proj] qwen_learning.md | high (logged) |

Reading of the above: CPU Ken is an **air/rush** opponent (jump-ins, DP, Hurricane), CPU Ryu is a
**fireball/zoning** opponent. Consequences:
- Anti-air and "punish him coming down" matter much more vs Ken than vs Ryu. That is exactly what our
  winning Ken lessons are: `use more hp up close when he jumps` (+61.5 over 16 runs [proj]) and
  `avoid sweep up close when he jumps` (a sweep up close goes under/loses to an airborne Ken).
  Standing hp is listed as one of Chun's anti-airs in the WW notes [WW-arc], and the SNES Hurricane
  counter is also a standing fierce. Whether his DP / Hurricane show up as "jumps" or "attacks" in our
  state is not known (specials are logged as attacks; being airborne may read as jump) - so both
  conditions are worth trying.
- Fireball defence (block_low far away, sweep his fireball startup) is less important vs Ken than vs Ryu.
- The throw carries over: shorter-range throw on his side, DP / Hurricane recovery lands next to her.

### Chun-Li's tools vs Ken (same as vs Ryu unless marked)
- **Anti-air**: st.hp / far mk / far hk / far lp; cr.mk vs far jumps [WW-arc]. Ken-specific emphasis.
- **Block the air stuff**: block Hurricane (first hit, or it whiffs over a crouch) and DP, then punish
  [WW-arc, SNES]. `always block_high up close when he jumps` was registered by Qwen in a Ken run that
  scored +67.3 hp/round [proj].
- **Mid footsies**: cr.mk, far mk [WW-arc]. Less central vs Ken since he comes to her.
- **Close**: throw (vs Ryu alone +42 hp/round [proj]); after DP/Hurricane recovery [SNES].
- **Avoid**: SBK (Ken's cr.mp beats it [WW-arc SuperCombo Ken, matchup line]; our Ken runs registered
  `avoid spinning_bird_kick at mid range when he stands` [proj]); Lightning Legs; sweep / c.hp up close
  when he is airborne [proj].
- Keep avoids to 1-2 lines: avoid-heavy advice made her passive [proj].

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line | Fidelity |
|---|---|---|---|
| 1 | Anti-air his jump-ins / knock him out of DP & Hurricane with st.hp | `use more hp up close when he jumps` | exact for jump-ins; approximate for DP/Hurricane (state label unknown) |
| 2 | Far st.mk anti-air at mid | `use more mk at mid range when he jumps` | exact |
| 3 | cr.mk vs far jumps | `use more c.mk far away when he jumps` | exact (spacing within "far" bin uncertain) |
| 4 | Don't sweep / c.hp under an airborne Ken | `avoid sweep up close when he jumps` | exact |
| 5 | Fierce punch into an incoming Hurricane at ~1 body width | `use more hp up close when he attacks` | approximate ("attacks" also covers his ground normals and DP) |
| 6 | Block Hurricane / DP, then throw as he lands | `use more throw up close when he attacks` (+ `use more block_high up close when he jumps`) | approximate: "block then punish" order and "he is landing / recovering" cannot be said |
| 7 | Block his jump-ins | `use more block_high up close when he jumps` | exact, but costs anti-air chances; project line (+67 run) |
| 8 | Throw up close in general (longer range) | `use more throw up close` | exact (carries over from Ryu) |
| 9 | Opening: back away and wait for the Hurricane | - | cannot express (no walk back, no "round start") |
| 10 | Never rely on SBK (cr.mp beats it) | `avoid spinning_bird_kick` | exact |
| 11 | Footsies with cr.mk at mid | `use more c.mk at mid range` | exact (less central vs Ken) |
| 12 | Sweep his fireball startup | `use more sweep at mid range when he attacks` | approximate; low value vs Ken (few fireballs) |
| 13 | Punish repeated DPs | - | cannot express (no "he did it twice" / sequence state) |
| 14 | Stay grounded, don't jump in (DP bait) | - | not applicable (she cannot jump) |
| 15 | Punish dizzy | `always hp up close when he is stunned` | approximate (Ryu's 2x weak spot is Ryu-only in the notes; unknown for Ken) |

## 3. Recommendations

### Expert set (5 lines)
1. `use more hp up close when he jumps`
2. `use more throw up close`
3. `use more mk at mid range when he jumps`
4. `use more throw up close when he attacks`
5. `avoid sweep up close when he jumps`

Rationale: CPU Ken lives in the air (jump-ins, DP, Hurricane), so three lines handle that (close
anti-air, mid anti-air, don't sweep under him), and two add the throw that carried Ryu - generally and
as the SNES "throw him as he comes down" punish. One avoid only. Note lines 1 and 5 are the two lessons
already known to help vs Ken, so this set is not independent of our data.

### Single-line candidates to test alone
- `use more throw up close` (does the Ryu result transfer?)
- `use more throw up close when he attacks` (punish DP / Hurricane recovery)
- `use more hp up close when he attacks` (SNES Hurricane counter)
- `use more mk at mid range when he jumps` (anti-air before he gets close)
- `use more c.mk far away when he jumps` (anti-air his far jumps)
- `use more block_high up close when he jumps` (project line; defence baseline)

### Cannot express
Walk back / back away (opening vs Hurricane), neutral jump, "block then punish" sequences, "he is
landing / recovering from a DP", "he did DP twice", tick throws, wake-up pressure, round-start openers.

## Sources
- RJHarrison SNES SF2 guide (read via archive): https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (Ken and Ryu "virtually equal in this game"; CPU Ken: more Hurricane/DP, fewer fireballs)
- "Complete Chun Li Strategy Guide: SNES SFII" (Usenet, 1992; 429-blocked, used via search snippets only): https://groups.google.com/g/rec.games.video/c/6uqBlDC7xMU
- SuperCombo WW Ken (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Ken (DP "wider range than Ryu's"; matchup lines)
- Arcade-museum forum thread (CPU Ken vs Ryu anecdote): https://forums.arcade-museum.com/threads/street-fighter-ii-ce-just-ryu-and-or-ken.186209/
- Carry-over sources: see `docs/chunli_vs_ryu_notes.md` (SuperCombo WW Ryu/Chun-Li, Shoryuken wiki, sf2platinum)
- Project counts: /Users/worker/work/hobby/laya_vision-vs-SF2/docs/qwen_learning.md
