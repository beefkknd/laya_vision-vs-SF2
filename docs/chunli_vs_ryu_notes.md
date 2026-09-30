# Chun-Li vs Ryu (SF2 World Warrior), incl. CPU Ryu -> lesson grammar (research 2026-09-30)

Builds on `docs/sf2_world_warrior_notes.md` (not repeated here). Version tags: **[WW-arc]** arcade WW
source, **[SNES]** SNES SF2 source, **[proj]** this repo's own logged counts (`docs/qwen_learning.md`).
No Champion Edition / Turbo source was used. Caveat from the notes still holds: SNES AI is a port, not
confirmed identical to the arcade scripts.

## 1. Strategy summary

### What CPU Ryu does
| Pattern | Source | Confidence |
|---|---|---|
| Fireball is his most-used special; Dragon Punch (DP) "every so often"; Hurricane Kick least | [SNES] RJHarrison GameFAQs guide | medium-high (one guide; matches the arcade script below) |
| Scripted "easy" routine: three strong fireballs in a row, walk in and throw only if you got dizzy | [WW-arc] sf2platinum disassembly | high (read from ROM) |
| Reads your *input* the frame you commit, but lets many attacks through depending on difficulty; gets more aggressive as the timer runs down | [WW-arc] sf2platinum | high |
| CPU Ryu/Ken usually throws **more than one DP in a row** | [SNES] "Complete Chun Li Strategy Guide" (Usenet), seen only via search snippets | medium |
| On level 7 SNES, Ryu opens by **walking toward you**, which makes him easy to throw | same Usenet guide (snippets) | medium |
| After being pushed out he fires a fireball; guide says jump straight up (not diagonal: fireball speed ~ jump speed), repeat until he walks in again | same Usenet guide (snippets) | medium |

### His weaknesses in WW (arcade frame data)
- **Hadoken**: slow startup and recovery in WW and his lower hurtbox sticks out, so it is "too easy to
  sweep his recovery or startup" [WW-arc, SuperCombo Ryu]. High confidence.
- **Shoryuken**: -18 / -26 / -40 on block/whiff (jab / strong / fierce) and does **not knock down a
  grounded opponent** [WW-arc, SuperCombo Ryu]. Block it, then punish him as he lands. SNES guide:
  if within ~3 body widths of a DP, walk up and throw him as he comes down. High (data) / medium (SNES tip).
- **Hurricane Kick**: hits do not combo and do not knock down; block the first hit (or crouch under the
  rest) and punish before he lands [WW-arc]. SNES guide: block it from ~1 body width, then walk in and
  throw as he settles. High.
- **Dizzy weak spot**: Ryu takes 2x damage during the leaning frames of his dizzy animation [WW-arc]. High
  (arcade; SNES unconfirmed).
- SuperCombo rates the matchup 6.5-3.5 for Chun: her normals out-prioritise his and she walks faster.

### Chun-Li's tools by distance
- **Far**: nothing reaches. Block fireballs (low block also blocks them); do not walk into them. The
  classic answer (neutral jump over them) is not available to our bot. cr.mk anti-airs his far jumps [WW-arc].
- **Mid**: cr.mk and far st.mk "control space and punish his attacks" [WW-arc SuperCombo Chun]; sweep his
  fireball startup/recovery [WW-arc SuperCombo Ryu; SRK wiki: "Sweep him and keep him away"]. Anti-air:
  far mk / hk / hp / lp [WW-arc]. cr.mk and cr.hp at mid are net-positive in our own logs [proj].
- **Close**: throw, longer range than his [WW-arc]; our best logged move vs Ryu, +10 net hp per try [proj].
  After a knockdown or throw WW has no reversals, so tick throws / throw loops dominate [WW-arc SRK].
  st.lp up close is net-negative in our logs (-3.1) [proj].
- **Avoid**: Spinning Bird Kick and Lightning Legs are her worst WW moves (slow, unsafe; Ryu's cr.mp beats
  SBK) [WW-arc]; SBK is net-negative vs Ryu in our logs (-6.5 mid, -18.4 close) [proj]. One SRK line says
  SBK can pass a Hadoken; low confidence, contradicted by the frame-data view.
- **Jumping at Ryu** is DP bait (every source). Not relevant: our Chun cannot jump.
- Project caution [proj]: avoid-heavy advice made her passive and lost (attacks 59% vs 84% of decisions);
  keep avoids to one or two lines.

## 2. Suggestion -> grammar -> fidelity

| # | Suggestion | Grammar line(s) | Fidelity |
|---|---|---|---|
| 1 | Win neutral with cr.mk at mid | `use more c.mk at mid range` | exact |
| 2 | Far st.mk as poke/anti-air at mid | `use more mk at mid range` / `use more mk at mid range when he jumps` | exact |
| 3 | Anti-air far jumps with cr.mk | `use more c.mk far away when he jumps` | exact (hit depends on true spacing inside the "far" bin) |
| 4 | Sweep his fireball startup/recovery | `use more sweep at mid range when he attacks` | approximate: "attacks" also includes DP/Hurricane; fireball mostly thrown from far, where sweep cannot reach |
| 5 | Throw (longer range), CPU walks in | `use more throw up close` (or `always throw up close when he stands`) | approximate: no "he is walking in" state |
| 6 | Block a DP / Hurricane, then throw or punish as he lands | `use more throw up close when he attacks` + `use more block_high up close when he attacks` | approximate at best: the block-then-punish order and "he is landing / recovering" cannot be said; the two lines conflict in one situation |
| 7 | Block fireballs | `use more block_low far away when he attacks` | approximate (blocks it; risks passivity) |
| 8 | Jump straight up over fireballs | - | cannot express (no jump) |
| 9 | Do not walk into fireball range | `avoid forward far away when he attacks` | approximate |
| 10 | Punish dizzy hard (2x weak spot) | `always hp up close when he is stunned` | approximate: exact move/timing for the weak-spot frames unknown; hp is her strongest close normal in the notes' combo (j.hk > mp > hp) |
| 11 | Never rely on SBK | `avoid spinning_bird_kick` | exact |
| 12 | Never rely on Lightning Legs | `avoid lightning_legs` | exact (only one avoid line recommended, see caution) |
| 13 | Tick throw (c.lk, then throw during blockstun) | `use more throw up close when he crouches` (nearest) | cannot express the sequence; no "he blocks" condition |
| 14 | Throw loop / meaty on wake-up | - | cannot express (no knocked-down / getting-up state) |
| 15 | Don't jump in on Ryu | - | not applicable (she cannot jump) |
| 16 | Avoid st.lp up close [proj] | `avoid lp up close` | exact, but project data, not a web source |
| 17 | Exploit the three-fireball script / timer-driven aggression | - | cannot express (no pattern / time-left conditions) |

## 3. Recommendations

### Expert set (5 lines)
1. `use more c.mk at mid range`
2. `use more throw up close`
3. `use more mk at mid range when he jumps`
4. `use more sweep at mid range when he attacks`
5. `avoid spinning_bird_kick`

Rationale: covers footsies, the throw, anti-air, fireball punish, and the one clearly bad special,
with a single avoid (to keep her aggressive). All four "use more" lines are soft, so the vision model
can still skip a move it rates as failing.

### Single-line candidates to test alone
- `use more throw up close` (strongest in both sources and our counts)
- `use more c.mk at mid range`
- `use more sweep at mid range when he attacks`
- `always hp up close when he is stunned` (rare state, but 2x damage; cheap to test)
- `use more mk at mid range when he jumps`
- `use more block_low far away when he attacks` (fireball defence; watch for passivity)

## Sources
- SuperCombo WW Ryu (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Ryu ("too easy to sweep his recovery or startup")
- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li
- Shoryuken wiki Chun-li (WW): http://wiki.shoryuken.com/Chun-li_(WW) ("Sweep him and keep him away")
- sf2platinum, The AI Engine: https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/
- RJHarrison SNES SF2 guide (read via archive): https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (CPU Ryu favours fireballs, DP "every so often")
- "Complete Chun Li Strategy Guide: SNES SFII" (Usenet; full text blocked, 429, used via search snippets only): https://groups.google.com/g/rec.games.video/c/6uqBlDC7xMU
- Project counts: /Users/worker/work/hobby/laya_vision-vs-SF2/docs/qwen_learning.md (section 2)
