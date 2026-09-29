# SF2 World Warrior: FGC vocabulary, Chun-Li, CPU AI (research notes, 2026-09-29)

Caveat: almost all strategy sources describe the **arcade** World Warrior (WW). The SNES port is a
re-implementation; nothing found confirms its AI or frame data are identical. Treat numbers as arcade.
SuperCombo pages were read via web.archive.org (live site is behind a bot check).

## 1. Vocabulary (one line each; WW status)

| Term | Meaning | In WW? |
|---|---|---|
| Footsies / neutral | Walking in and out of each other's range, fishing for pokes and whiffs | Yes; Chun's core game |
| Poke | Fast, long-reach, low-risk normal to control space (Chun: cr.mk, far mk) | Yes |
| Spacing | Standing at the exact distance where your move reaches and his does not | Yes |
| Whiff punish | Hitting him while he recovers from a missed attack | Yes (e.g. Zangief's sweep) |
| Anti-air | Hitting him out of a jump before he lands | Yes |
| Jump-in | Jumping at him with a deep air attack to start offense | Yes; dangerous vs Shoryuken |
| Cross-up | Jump-in that hits from behind, so he must block the other way | Yes (Chun j.mk, Honda j.mk) |
| Meaty | Attack timed to hit on his first frame of wake-up | Yes, and **extra strong**: WW has no reversal specials |
| Chip damage | Small damage from blocked **special** moves | Yes (normals do not chip) |
| Reversal | Special move on the first possible frame after wake-up/blockstun | **No** in WW (specials cannot come out on a reversal frame) |
| Okizeme / wake-up | What you do while he stands up from a knockdown | Yes; with no reversals, wake-up offense dominates |
| Pressure / block string | Keeping him blocking so he cannot act | Yes |
| Zoning | Keeping him out with fireballs / long normals | Yes (Ken/Ryu fireball) |
| Turtling | Defensive play: block, keep away, win on time/life lead | Yes; recommended vs Honda |
| Tick throw | Light attack he blocks, then throw during/after blockstun | Yes; Chun's signature in WW |
| Throw loop | Throw, then walk in and throw again on wake-up | Yes; a WW-defining tactic |
| Dizzy / stun | Enough hits in a short window stun him (32 points); he must mash out | Yes |
| Frame advantage | Who can act first after a hit/block | Yes (hidden; data exists) |
| Recovery | Frames after an attack in which you cannot act | Yes |
| Priority | Which of two simultaneous attacks wins (hitbox vs hurtbox) | Yes; Chun has the best normals by priority |
| Trade | Both attacks hit at once | Yes |
| Counter-hit | Bonus for hitting during his startup | **No** such mechanic in SF2 |
| Super meter / supers | — | **No** in WW |
| Throw tech / escape | — | **No** throw escape for normal throws in WW (throw softening came in later versions; not confirmed by a fetched source). Holds (multi-hit grabs) can be mashed. |
| Mirror match | Same character both sides | Not possible in WW |

## 2. Chun-Li in WW

- **Strength = normals, walk speed, throw.** Called one of the best WW characters, with arguably the
  best-priority normals and the fastest walk; her throw reaches farther than Ryu/Ken's.
- **Specials are bad in WW.** Lightning Legs: hard to activate, slow startup, slow recovery.
  Spinning Bird Kick: very slow, recovers slowly; guide advice is to avoid both. Ken/Ryu beat SBK with
  cr.mp; Honda with cr.mp. Arcade quirk: charging crouch kicks can accidentally trigger SBK.
- **Anti-airs:** far mk, far hk, far hp, far lp (standing, not close); cr.mk vs jumps from far away.
- **Pokes / footsies:** cr.mk and far mk "control space and punish his attacks" (vs shotos).
- **Jump-ins / air:** j.mk (good, also cross-up), j.lk as defensive air-to-air; j.hk as a combo starter.
- **Combos:** j.hk > st.mp > st.hp or sweep.
- **Offense:** knockdown or throw -> tick throw / throw loop (no reversals to escape it).
- **Weaknesses:** specials unreliable and unsafe; low damage per hit vs Honda; if she jumps in,
  Shoryuken punishes hard.

Frame notes (arcade): most of her standing/crouch light and medium normals are plus on block
(e.g. st.mp +11, cr.mk +12); st.hp -1, close hp -8, sweep -2 (roughly safe/neutral).

## 3. CPU AI (arcade WW, reverse-engineered)

- Moves come from small **scripts** (bytecode). Three modes: **waiting** (random short walks
  forward/back), **attacking** (e.g. Ryu: three fireballs, then rush in to throw if you are dizzy),
  **reacting to your attack**.
- Reaction uses a value ("yoke") attached to **your move's animation**, which the CPU sees the moment the
  move is input, before its first frame is drawn. So it reads the *move you committed to*, not the
  screen; this is the root of the "input reading" reputation.
- It does **not** react every time: depending on difficulty it lets many attacks through unguarded.
- 8 script levels for waiting/attacking, picked by **time left in the round**: it gets more aggressive
  as the timer runs down.
- It cheats: charge specials without charging; can disable collision to walk through fireballs.
- Script repertoire is per matched-up opponent (one source line says the formula is the same whoever it faces).

**Ken / Ryu (WW):** Hadoken (slow startup/recovery in WW; sweep his startup), Shoryuken (beats jump-ins;
jab version safest; does not knock down grounded opponents; very unsafe on block, -11 to -33),
Hurricane Kick (does **not** combo or knock down; block the first hit or crouch under, then punish
before he lands). Known CPU pattern: two fireballs to dizzy, then fierce into Shoryuken.
Ryu has a double-damage weak spot during his dizzy animation.

**E. Honda (WW):** Sumo Headbutt (charge move; also used as anti-air; **punishable on block**,
roughly -11 jab / -14 strong); Hundred Hand Slap (mashed; beats many normals, strong at mid range).
Answers: rapid standing jab beats Headbutt; block then punish on his landing/recovery; jump over him.
All Honda needs is one throw to start a loop on Chun.

## 4. Matchups (Chun-Li WW)

- **vs Ryu / Ken:** Chun is favoured (Ryu page calls it tough for Ryu). Win footsies with cr.mk / far mk,
  use the longer throw, avoid jumping in on Shoryuken, never lean on SBK (cr.mp beats it).
- **vs Honda:** hit-and-run / turtle. Get some damage, then keep away; j.mk / j.lk as air defence
  (he can only answer if directly below); jump back over him with j.mk to escape; cr.mk anti-airs his
  far jumps. Honda should approach patiently and fish for a throw.
- **vs Zangief:** walk speed + cr.mk to keep him out; whiff punish his sweep; getting knocked down is deadly.
- **vs Dhalsim:** j.mk troubles him; he has more throw range; decent for Chun.

## 5. Mapping: term -> observable from logs? -> coach phrasing (current grammar)

Log fields: range (close/mid/far), his state (stand/crouch/jump/attack/hit stun/guard), result
(hit/whiff/blocked), damage dealt/taken.

| Term | Observable? | Suggested lesson |
|---|---|---|
| Anti-air | Yes (his state = jump, her hit/whiff) | "use more mk at mid range when he jumps"; "use more c.mk far away when he jumps" |
| Poke / footsies | Yes (range + standing/crouch + hit/blocked) | "use more c.mk at mid range when he stands" |
| Spacing | Partly (3 bins only; exact reach unknown) | "use more mk at mid range" |
| Whiff punish | Partly: "attack" state exists but not *his* whiff vs *her* hit; need "he whiffed / recovering" | "use more c.mk at mid range when he attacks" (approximates) |
| Punish on block | Partly: need "he is recovering after a blocked special" | "use more sweep up close when he attacks" (blocked Headbutt/Tatsu) |
| Jump-in | No (she has no jump action) | — (needs a jump move) |
| Cross-up | No | — |
| Chip damage | Partly (damage taken while guarding) | "use more block_low up close when he attacks" |
| Tick throw | Partly (her blocked hit then throw; needs sequence of 2 decisions) | "use more throw up close when he blocks" (state = guard) |
| Throw loop / okizeme | No (no knockdown / wake-up state) | needs a "getting up" state; nearest: "use more throw up close when he stands" |
| Meaty | No (no wake-up state/timing) | — |
| Reversal | N/A in WW | — |
| Turtling / keep-away | Yes (range + block + damage balance) | "use more block_low far away when he attacks"; "avoid forward up close" |
| Zoning (his fireball) | Partly: fireball looks like "attack"; need "projectile on screen" | "use more block_low far away when he attacks" |
| Pressure | Partly (her plus-on-block normals; needs sequence) | "use more c.lk up close when he blocks" |
| Dizzy | Yes (hit stun only partly; dizzy state distinct?) | "use more hp up close when he is stunned" |
| Trade | Partly (damage dealt and taken same step) | — |
| Priority | No (hitboxes) | — |
| Frame advantage / recovery | No | — |
| Avoid specials (WW advice) | Yes (result per move) | "avoid spinning_bird_kick far away"; "avoid lightning_legs at mid range" |
| Anti-Headbutt | Partly (Headbutt = "attack") | "use more lp up close when he attacks" (vs Honda) |

Highest-value missing observables: his state "recovering / whiffed" vs "attacking",
"projectile on screen", "knocked down / getting up", and her previous action (for tick throws).

## Sources

- SuperCombo WW Chun-Li (archived): https://web.archive.org/web/2024/https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li
- SuperCombo WW E. Honda / Ryu / Ken (archived, same path pattern): https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/E._Honda , .../Ryu , .../Ken
- Shoryuken wiki Chun-li (WW) (archived): http://wiki.shoryuken.com/Chun-li_(WW) (source of "no reversals" and throw-loop notes)
- Shoryuken wiki WW system page (archived): http://wiki.shoryuken.com/Street_Fighter_2:_The_World_Warrior (block, chip, dizzy, hitstun)
- sf2platinum, "The AI Engine" (arcade WW disassembly): https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/ ; https://sf2platinum.wordpress.com/
- Atrocious Gameplay wiki, SF2 CPU: https://atrociousgameplay.miraheze.org/wiki/Street_Fighter_II_Series_CPU (it says the AI "reacts to your inputs, not what happens on screen")
- Cheap Ass Gamer thread (CPU Ken fireball-dizzy-DP pattern, no reversals): https://www.cheapassgamer.com/threads/street-fighter-2-tips.153922/
