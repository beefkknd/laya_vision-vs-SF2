# SF2 The World Warrior (SNES, 1992): move sheet

This is the spec for the input macros and for the checks that a move "works as defined".
It covers only what exists in SNES World Warrior (WW). Moves added in later versions are listed
under "Not in WW" so nobody builds macros for them.

## Notation key

| Token | Meaning |
|---|---|
| `F` `B` `U` `D` | toward the opponent / away / up / down (relative to facing) |
| `DF` `DB` `UF` `UB` | diagonals |
| `lp mp hp` | jab / strong / fierce punch (SNES default: Y / X / L) |
| `lk mk hk` | short / forward / roundhouse kick (SNES default: B / A / R) |
| `P` / `K` | any one punch / any one kick; the strength picks the variant |
| `PPP` | all three punches pressed together |
| `A, B` | A then B in sequence; `A + x` means press button x on direction A |
| `[B]60` | hold B (or DB or UB) for at least ~60 frames (charge) |
| `s.` `c.` `cl.` `j.` | standing / crouching / close standing / jumping normal |
| `xx` | special cancel: the special's motion is entered while the normal is still hitting (see "Cancels") |
| `,` in a combo | link: the next move starts after the previous one recovers |
| `360` | full circle on the pad, e.g. `F, DF, D, DB, B, UB` + P |

All inputs are written for a character facing right. When the character faces left the game
mirrors them, so a relative-direction macro needs no change. It does matter if the character
switches sides in the middle of a motion (cross-up): the motion may then come out wrong.

## Shared basics

| Action | Input | What you see |
|---|---|---|
| Walk | `F` / `B` | walks toward / away |
| Jump | `U` / `UF` / `UB` | straight / forward / backward arc, about 1 second in the air; you cannot steer in the air |
| Crouch | `D` | ducks; low block needs DB |
| Block high | hold `B` while the opponent attacks | guard pose; stops jumping attacks and standing attacks. Crouching attacks that hit low still land |
| Block low | hold `DB` | crouching guard; stops sweeps and low kicks. Jump-ins still land |
| Normals | each of the 6 buttons standing, crouching, and in the air (straight or angled jump) | 18 basic attacks. Many standing normals have a different **close** version when right next to the opponent |
| Throw | next to the opponent, `F` or `B` + mp/hp (some characters also mk/hk) | the opponent is grabbed and thrown to the side you held. You cannot throw an opponent who is blocking-stunned, in hitstun, or in the air (except with air throws) |
| Chip damage | any special that is blocked | a small amount of health is lost even while blocking |

General WW mechanics that affect macros and checks:

- **Motion inputs** (`D, DF, F + P` and similar): the directions are read from an input buffer, so
  a macro can hold each direction for 1 to 2 frames and press the button on the last direction.
  Pressing a little early also works: a normal pressed just before the motion finishes is
  replaced by the special (a "kara cancel", about 5 frames). **Negative edge:** releasing a
  button also triggers specials, so holding a button through a motion and releasing it can fire
  the special by accident.
- **Charge inputs**: hold the charge direction (`B` for back charge, `D` for down charge) for
  about 60 frames, then press the release direction plus the button. `DB` charges both back and
  down at the same time, and holding B or DB also blocks, so charge characters charge while
  defending. The model decides every 4 frames, so one charge is at least 15 decisions of holding
  B / DB. **The macro or the controller must track how long the charge has been held.**
- **Mash inputs** (Hundred Hand Slap, Electricity, Lightning Legs): press the same button
  several times quickly (about 4 to 5 presses, a few frames apart). The move keeps going while
  presses continue.
- **No reversal specials in WW**: on the first frame after getting up or after blockstun, only a
  normal or a throw can be done, not a special. A special macro started exactly on wake-up can
  fail, so start it a few frames later.
- **Cancels**: WW has special cancels (a normal that hits or is blocked can be cancelled into a
  special during the ~14-frame hit freeze) and "chain" presses of some jab/short attacks. Only
  some normals can be cancelled; the per-character tables list them.
- **Combos**: after the first hit of a combo, hitstun is one frame shorter. A string that
  combos as a first hit can therefore be blocked when it follows a jump-in (the Shoryuken wiki
  example: Ryu's `c.lk xx Hadoken` combos, but `j.hk, c.lk xx Hadoken` can be blocked).
- **Dizzy**: enough hits or throws in a short time make the opponent dizzy (stars or birds spin
  over the head, and the opponent stands still). Dizzy is a side effect, not a requirement.
- **Checking that a combo worked**: the opponent must stay in the hit animation between hits and
  never show the block pose; health drops on every hit. Check this in RAM (hitstun / block state)
  rather than on screen. See Uncertain about a hit counter on screen.

---

## Ryu

Source: GameFAQs SNES FAQ (RJHarrison); Shoryuken wiki WW/Ryu.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Hadoken (fireball) | `D, DF, F + P` | motion (quarter circle forward) | lp slow, mp medium, hp fast projectile | Ryu pushes both palms forward, a blue energy ball appears in front of his hands and travels forward along the ground. Only one of his fireballs can be on screen at a time; he cannot move until it leaves his hands |
| Shoryuken (dragon punch) | `F, D, DF + P` | motion (Z shape) | lp low jump, mp medium, hp highest | Ryu jumps straight up and slightly forward with a rising uppercut, then falls back down. It is invincible at the start and knocks jumping opponents down |
| Tatsumaki (hurricane kick) | `D, DB, B + K` | motion (quarter circle back) | lk short distance, mk medium, hk long | Ryu rises a little and spins forward horizontally with one leg out, crossing part of the screen; he lands where it ends. Passes over low attacks and fireballs, and can be hit by low attacks |

WW quirks (Shoryuken wiki): the Shoryuken does not knock down an opponent who is standing on the
ground, and the Hurricane Kick does not knock down and its hits do not combo, so the opponent can
block after the first kick or duck under the rest. There is no air Hurricane Kick in WW.

**Verified on ROM (2026-09-27, tests/test_rom_moves.py, both facings):** the three inputs above work as written with
2 frames per direction and the button on the last direction (macros in `sf2/actions.py`, fierce / roundhouse). What
the sheet did not say, from the SNES ROM:

- All three are action state `0C`; `0x0D80` says which: `00` Hadoken, `02` Hurricane Kick, `04` Shoryuken. The
  Hadoken's projectile is in player 1's slot `0x1000` (x at `0x1007`), ~3 px per frame, and appears ~13 frames after
  the press. The CPU's own specials show as state `0A` (Ken's Hadoken is `0A`, sub-state `04`, in slot `0x1050`).
- A Hadoken entered within ~8 frames of walking forward comes out as a **Shoryuken** (`F` is still in the input
  buffer, and the Shoryuken wins).
- The roundhouse Hurricane Kick lifts Ryu ~17 px and carries him ~160 px: it passes over a crouching opponent and
  lands on the other side (the sides swap).
- The ROM mirrors the stick by its own facing byte (`0x0CF4`), which lags the x positions while turning, in the air
  and in guard / hit stun; a motion resolved from x in that window comes out mirrored.
- Blocked specials cost chip damage (6-12 life); a blocked attack is block stun `0E` with reaction `06` / `08`, a hit
  any other reaction (`14` for the Hadoken and the Shoryuken, `20` for the Hurricane Kick).

### Throws

| Throw | Input | Look |
|---|---|---|
| Seoi Nage (shoulder throw) | close, `F`/`B` + mp or hp | throws the opponent over his shoulder |
| Tomoe Nage (back roll) | close, `F`/`B` + mk or hk | rolls onto his back and kicks the opponent over his head |

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Low poke into fireball | `c.mk xx Hadoken` (`D+mk`, then `D, DF, F + hp` during the hit) | the fireball hits while the opponent is still reeling from the kick |
| Low poke into uppercut | `c.mk xx Shoryuken` | the uppercut connects straight after the kick |
| Jump-in into fireball | `UF`, `j.hk` (late, low on the opponent), land, `cl.hp xx Hadoken` | 3 hits in a row, no block pose in between |
| CPS1 chain | `c.lk`, `c.lk`, then `s.hp xx Hadoken` | advanced; the Shoryuken wiki describes it. Do it later |

Cancellable normals (Shoryuken wiki WW data): almost all standing and crouching normals,
including `c.mk`, `cl.hp`, `c.hp`, and `c.hk`.

### Facing and spacing

Fireballs are for far range, Shoryuken for opponents jumping in, `c.mk` for close range. Throws
work only when touching.

---

## Ken

Source: GameFAQs SNES FAQ (RJHarrison); Shoryuken wiki WW/Ken; SuperCombo wiki WW/Ken (summary
in search results).

In WW, Ken's moves use the **same inputs as Ryu's**: Hadoken `D, DF, F + P`, Shoryuken
`F, D, DF + P`, and Tatsumaki `D, DB, B + K`. They also look the same (Ken's fireball is also
blue in WW). The differences are small:

- Shoryuken: the Shoryuken wiki says it has "wider range than Ryu's". SuperCombo says Ken lacks
  Ryu's knockdown problems with the Shoryuken and Hurricane Kick. That is a claim to verify
  (see Uncertain).
- Kick throw (Tomoe Nage, `F`/`B` + mk or hk): Ken somersaults with the opponent first, and the
  opponent lands farther away. This is described as the "better kick throw".

Throws: the same as Ryu (mp/hp shoulder throw, mk/hk back roll). Combos: the same as Ryu
(`c.mk xx Hadoken`, `c.mk xx Shoryuken`, `j.hk, cl.hp xx Hadoken`). Facing and spacing: the same
as Ryu.

---

## E. Honda

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/E._Honda.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Hundred Hand Slap | press one punch button quickly (~5 times) | mash | lp slower slaps, mp faster, hp fastest (FAQ) | Honda stands in place and his arm becomes a blur of palm strikes in front of him. It keeps going while presses continue. Hits several times; chip damage when blocked. He does **not** move forward during it |
| Sumo Headbutt | `[B]60, F + P` | charge back ~60 f | lp short/slow, mp medium, hp long/fast | Honda flies horizontally, head first, like a torpedo, across the screen at head height, then drops back to his feet. A hit knocks the opponent back |

Not in WW: Sumo Smash (the butt drop, `[D], U + K`) and the moving Hundred Hand Slap.

### Throws

| Throw | Input | Look |
|---|---|---|
| Sumo throw | close, `F`/`B` + mp | tosses the opponent |
| Bear Hug | close, `F`/`B` + hp | hold: squeezes several times |
| Knee Bash | close, `F`/`B` + hk | hold: repeated knees |

Honda's throw range is long, so tick throws (a light hit, then a throw) work well.

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Jump-in | `UF`, `j.mp`, land, `s.hk` (or `c.hp`) | 2 hits, no block in between |
| Cross-up | `j.mk` over the opponent's head, `s.mp`, `c.hp` | 3 hits; Honda lands on the other side (facing flips) |
| Jab into slaps | `s.lp`, `s.lp` while still mashing into the Hundred Hand Slap | the slaps continue from the jabs without a gap. `s.lp` is his only cancellable normal (Shoryuken wiki data) |

### Facing and spacing

Charge characters hold B (which also blocks) to store the headbutt. The Hundred Hand Slap has
short reach, so use it next to the opponent or when the opponent is in the corner. The FAQ tip is
to corner the opponent.

---

## Blanka

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/Blanka.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Electricity | press one punch button quickly (~5 times) | mash | none given in the FAQ | Blanka crouches and his body flashes with electric sparks. An opponent who touches him is shocked (skeleton flash) and knocked back. He stays in place |
| Rolling Attack | `[B]60, F + P` | charge back ~60 f | lp slow/short, mp medium, hp fast/long | Blanka curls into a ball and rolls horizontally across the screen. On hit, he bounces off the opponent. If it is blocked, he bounces high up and back and can be punished |

Not in WW: Vertical Roll (Super SF2) and Backstep Roll (Super SF2).

### Throws

| Throw | Input | Look |
|---|---|---|
| Head Bite | close, `F`/`B` + hp | hold: grabs the opponent's head and bites it repeatedly |

Command normals (not throws): `F` + mp is the head butt (two hits). `F` + mk is the double knee
(see Uncertain: the Shoryuken wiki lists mp here). Close hp is the somersault double punch.

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Jump-in string | `UF`, `j.hk`, land, `s.mp`, `c.hk` (or `c.hp`) | 3 hits, no block pose |
| Poke into roll | `c.lp xx Rolling Attack` (needs a stored back charge; `c.lp` is cancellable) | the ball hits right after the jab. This is a guess from the cancel data; see Uncertain |

### Facing and spacing

Blanka holds B or DB to store the Rolling Attack while blocking. Electricity only hits when the
opponent is touching him, so it is for very close range or against jump-ins. His jumps are
fast.

---

## Guile

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/Guile; fluxcore WW Guile guide (charge frames).

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Sonic Boom | `[B]60, F + P` | charge back (**about 59 frames**, per fluxcore) | lp slow, mp medium, hp fast projectile | Guile sweeps both arms together and a spinning crescent wave travels forward. He recovers quickly and can walk behind it |
| Flash Kick (Somersault Kick) | `[D]60, U + K` (also `UF`/`UB`) | charge down ~60 f (`D`, `DB` or `DF` all charge) | lk low arc, mk medium, hk highest | Guile does a backflip kick upward with a curved flash trail, then lands. Knocks down jumping opponents. Close to a standing opponent it can hit twice |

WW quirks: holding `DB` charges both moves at once. After a store, the fluxcore guide says a
Flash Kick charge is kept for about 3 seconds. WW Guile also has famous glitches (handcuffs,
"free Sonic Boom" after a mp throw while holding F). Do not rely on them.

### Throws

| Throw | Input | Look |
|---|---|---|
| Hip toss / shoulder throw | close, `F`/`B` + mp | press-slams the opponent overhead and throws them |
| German Suplex | close, `F`/`B` + hp | bridges backward and slams the opponent |
| Air throw | both in the air, close, `F`/`B` + mp or hp | throws the opponent down to the ground |
| Air Backbreaker | both in the air, close, `F`/`B` + mk or hk | drags the opponent down onto his knee |

Command normals: `F` or `B` + mk is the knee (hop forward), `B` + hp is the backfist, close `F`/`B`
+ hk is the upside-down kick, and `D` + hk is the double sweep.

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Jump-in into Flash Kick | hold `DB` during the jump (air time counts as charge), `UF`, `j.hp`, land, `c.mp xx Flash Kick` (`U + hk`) | 3 hits; the Flash Kick rises during the hitstun of `c.mp` |
| Jump-in into Sonic Boom | `j.hp`, `s.hp xx Sonic Boom` (`B + hp` backfist in the corner) | the boom hits at point-blank range right after `s.hp` |
| Chain | `c.lk`, CPS1 chain into `s.hp xx Sonic Boom` | advanced; do it later |

Cancellable normals: `s.lp`, `s.mp`, `s.lk`, close `lp`/`mp`/`hp`, `c.lp`, `c.mp`.

### Facing and spacing

Guile is the model charge character. Guile plays by sitting in `DB` (low block plus both
charges), throwing booms at range, and using the Flash Kick against jumps. Walking forward
cancels the back charge, so the controller must choose between charging and approaching.

---

## Chun-Li

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/Chun-Li.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Lightning Legs (Hyakuretsukyaku) | press one kick button quickly (~5 times) | mash | lk slower, mk faster, hk fastest (FAQ) | Chun-Li stands on one leg and her other leg blurs in rapid kicks at head/chest height. She stays in place. Hits several times; strong dizzy potential |
| Spinning Bird Kick | `[D]60, U + K` | charge down ~60 f | lk short distance, mk medium, hk long | Chun-Li flips upside down, legs spread, and spins while flying horizontally across the screen at head height, hitting several times, then lands |

Not in WW: Kikoken fireball (Champion Edition / Hyper Fighting era), and the air Spinning Bird
Kick.

**Verified on ROM (2026-09-27, tests/test_rom_moves.py, both facings, against the CPU Ryu):** both are action state
`0C`; `0x0D80` says which: `02` Lightning Legs, `00` Spinning Bird Kick. Neither uses her projectile slot. What the
sheet did not say, from the SNES ROM:

- **Spinning Bird Kick charge: down held at least 61 frames** from standing, then `U + hk` (binary search, 61 at 20 of
  20 free moments, 10 per facing; 60 never works). The macro holds 64. The SBK sits on the ground in `0C` for ~40
  frames after the press before it lifts off (~17 px) and travels (~60+ px toward him, further if it whiffs).
- **Lightning Legs: 10 short taps** (1 frame down, 1 up) start them, 9 never do (from a free standing moment); the macro
  taps 12. Wider gaps need more taps (2 down / 2 up: 11). The Legs stay in place.
- Chun-Li's far standing normals whiff in 13 / 30 / 17 / 33 frames of `0A` (jab / fierce / short / roundhouse).

Other: `j.D + mk` is the head stomp (she bounces off the opponent's head). **Wall jump**: jump
into the screen edge, then press the opposite diagonal up.

### Throws

| Throw | Input | Look |
|---|---|---|
| Ground throw | close, `F`/`B` + mp or hp | flips / tosses the opponent |
| Air throw | both in the air, close, `F`/`B` + mp or hp | tackles the opponent to the ground |

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Jump-in string | `UF`, `j.hk`, land, `s.mp`, `s.hp` (or `c.hk`) | 3 hits, no block pose |
| Into Lightning Legs | `UF`, `j.mk`, land, mash `mk` → Lightning Legs | the first kick of the legs lands while the opponent is still stunned by the jump-in |

Cancellable normals: `s.lp`, `s.lk`, close `lp`, close `hp`, `c.mk`.

### Facing and spacing

Chun-Li is fast and has good air control options (wall jump, air throw). Lightning Legs only
reaches a short distance in front of her. The Spinning Bird Kick needs the down charge (DB also
blocks low).

---

## Zangief

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/Zangief.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Spinning Pile Driver (SPD) | `F, DF, D, DB, B, UB + P`, close to the opponent | 360 motion; a command grab | none (FAQ: "none") | Zangief grabs the opponent, jumps high while spinning with them upside down, and slams them head first into the ground. Big damage. Whiffs if the opponent is not in grab range, and cannot grab an opponent who is in the air or in blockstun/hitstun |
| Spinning Clothesline (Double Lariat) | `PPP` (all three punches together) | button combo | none | Zangief spins in place with both arms out, hitting on both sides. His lower body passes through fireballs |

WW notes: `UB` is part of the SPD motion. The punch has to be pressed before the jump starts, or
the macro will jump. The UB step is in the Shoryuken wiki's input, so the macro should include
it, with the punch on the frame right after UB. Not in WW: moving the lariat, the kick-button
lariat (Quick Double Lariat), and Banishing Flat.

### Throws (many; all need close range)

| Throw | Input | Look |
|---|---|---|
| Pile Driver | close, `F`/`B` + mp | a short jump into a pile driver |
| Brainbuster | close, `F`/`B` + hp | a suplex slam |
| Powerbomb | close, `F`/`B` + hk | a power bomb |
| German Suplex | close, `F`/`B` + mk | a backward suplex |
| Alley Oop / Power Slam | close, `DF`/`DB` + hp / mp | throws the opponent over his head |
| Holds (Head Bite, Face/Stomach Pump) | `F`/`B` (or `DF`/`DB`) + hp or mp on an approaching opponent | repeated small hits while holding |

The throw list is from the Shoryuken wiki and disagrees in places with the GameFAQs FAQ. See
Uncertain.

Other: during a forward jump, `D + hp` is the Body Splash and `D + lk` is the knee drop.
`U + hp` in the air is the stomach crunch.

### Combos

| Combo | Notation | Worked when |
|---|---|---|
| Kick into sweep | `cl.lk`, `c.hk` | 2 hits; the sweep knocks down |
| Splash string | jump forward, `j.D + hp` (body splash), land, `c.lp`, `c.hk` | 3 hits, knockdown |

### Facing and spacing

Zangief is slow on the ground with a short jump. The SPD and throws all need him to be touching
the opponent. Walking into range (or jumping in) is the whole game plan. The lariat is his
anti-fireball and anti-air tool.

---

## Dhalsim

Source: GameFAQs SNES FAQ; Shoryuken wiki WW/Dhalsim.

### Specials

| Move | Input | Type | Strength changes | What it looks like when it works |
|---|---|---|---|---|
| Yoga Fire | `D, DF, F + P` | motion | lp slow, mp medium, hp fast | Dhalsim breathes out a fireball that travels forward. It sets the opponent on fire and knocks them back |
| Yoga Flame | `B, DB, D, DF, F + P` (half circle back to forward) | motion | none given in the FAQ | Dhalsim breathes out a large, short-range cloud of flame in front of his face. It is not a projectile that travels. It stays in place for a moment and cancels projectiles. On hit, the opponent burns and falls down |

Not in WW: Yoga Teleport (Champion Edition) and Yoga Blast.

Other: in the air, `D + hp` is the Yoga Mummy / drill headbutt, and `D + hk` is the Yoga Spear /
drill kick. Both change his jump into a diagonal dive. The GameFAQs FAQ lists these as
`F, F` + hp / hk in the air, which disagrees; see Uncertain. `D + K` is a slide kick that passes
under fireballs.

### Throws

| Throw | Input | Look |
|---|---|---|
| Yoga Noogie | close, `F`/`B` + mp | hold: repeated punches to the head |
| Yoga Throw | close, `F`/`B` + hp | throws the opponent over his head |

### Combos

The Shoryuken wiki lists no combos for WW Dhalsim. From the cancel data (close `s.mp` and close
`c.mk` can be cancelled):

| Combo | Notation | Worked when |
|---|---|---|
| Poke into fire | close `c.mk xx Yoga Fire` | the fire hits right after the kick |
| Drill into fire | jump, `j.D + hp` (drill), land, close `s.mp xx Yoga Fire` | 3 hits; see Uncertain |

### Facing and spacing

**Long limbs**: Dhalsim's far standing and crouching punches and kicks stretch across a large
part of the screen. At range his normals are pokes, not close-range attacks. He is slow, and
hitting his long limbs hits him, so he is weak up close and in the air (floaty jump). Keep the
opponent at the tip of his limbs and use fire. Throws only work touching the opponent.

---

## Not in WW (do not build macros for these)

Air Hurricane Kick (Ryu/Ken), Ken's flaming Shoryuken, Chun-Li's Kikoken and air Spinning Bird
Kick, Honda's Sumo Smash, Blanka's Vertical and Backstep Rolls, Zangief's Quick Double Lariat and
Banishing Flat, Dhalsim's Yoga Teleport, all Super Arts, and the four Grand Masters as playable
characters.

## Sources

- GameFAQs, *Street Fighter II (SNES) FAQ / Move List* by Ryan Harrison (v1.00, 2012):
  https://gamefaqs.gamespot.com/snes/588700-street-fighter-ii/faqs/64276 (read through the
  Wayback Machine). Used for all 8 characters' specials, strengths, throws, and the button map.
- Shoryuken wiki, *Street Fighter 2: The World Warrior* (system page, 2020 archive):
  https://wiki.shoryuken.com/Street_Fighter_2:_The_World_Warrior (cancels, kara/negative
  edge, hitstun, no reversal specials, CPS1 chains).
- Shoryuken wiki character pages, 2020 archive, `http://wiki.shoryuken.com/Street_Fighter_2:_World_Warrior/<Name>`
  for Ryu, Ken, E._Honda, Blanka, Guile, Chun-Li, Zangief, Dhalsim: input strings, throws, combos,
  and the per-normal "Special Cancel" data. These describe the arcade WW. The SNES port is close
  to it, but they are not SNES-specific.
- fluxcore, *Street Fighter 2 World Warrior Guile glitches guide*:
  https://fluxcore.nz/blog/street-fighter-2-world-warrior-guile-glitches-guide (Sonic Boom ~59
  frame charge; Flash Kick charge stored ~3 s).
- SuperCombo wiki WW/Ryu and WW/Ken (only the search-result summary; the page itself was blocked):
  https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Ryu (Shoryuken/Tatsu do
  not knock down in WW).
- SuperCombo wiki SSF2T/O. E. Honda (search summary): Hundred Hand Slap needs 4 presses within
  15/12/9 frames for lp/mp/hp. This is **Super Turbo** data and is used only as an estimate.

## Uncertain (not confirmed for the SNES WW version)

1. **Charge time for characters other than Guile.** ~59 frames is sourced only for Guile's Sonic
   Boom. Honda, Blanka, and Chun-Li are assumed to be ~60 frames. How long a stored charge lasts
   and the time allowed between releasing the charge and pressing the button are unknown.
2. **Mash thresholds** for the Hundred Hand Slap, Electricity, and Lightning Legs in WW (press
   count and the maximum gap between presses). The only numbers are from Super Turbo (4
   presses, ≤15/12/9 f).
3. **Strength effects of the mash moves** (whether lp/mp/hp changes slap speed, or whether
   Electricity has variants) come from one FAQ only.
4. **Ryu vs Ken knockdown behavior.** The Shoryuken wiki/SuperCombo say Ryu's Shoryuken and Tatsu do
   not knock down a grounded opponent in WW, and Ken does not have this problem. The GameFAQs
   FAQ says the Shoryuken "can hit twice and knock the opponent down". Test in the emulator.
5. **Zangief Spinning Clothesline button count.** The FAQ says 2 or 3 punches; the Shoryuken wiki
   says all three. Use `PPP`.
6. **Zangief throw/hold table.** The two sources disagree on which button does which throw, and on
   `DF`/`DB` variants. Verify each one in the emulator before relying on it.
7. **Dhalsim drill input.** The Shoryuken wiki says `D + hp/hk` in the air; the GameFAQs FAQ says
   `F, F + hp/hk` in the air.
8. **Blanka's double knee.** The Shoryuken wiki says `F + mp`, which clashes with the `F + mp` head
   butt; the FAQ says close `F`/`B` + mk.
9. **Air charge.** It is assumed that holding `DB` during a jump builds charge (used in Guile's
   jump-in Flash Kick combo). This is not confirmed for WW.
10. **Combos derived from cancel data** (Blanka `c.lp xx Rolling Attack`, Dhalsim's two combos, the
    `cl.hp xx Hadoken` jump-in) are not in any source as combos. They are guesses from the
    "Special Cancel = Yes" data.
11. **On-screen combo counter.** It is not confirmed whether SNES WW shows a hit counter. Assume it
    does not, and check combos through RAM hitstun/block state.
12. **Per-normal cancel data** is from arcade WW hitbox tables; small differences in the SNES port
    are possible.
13. **Macro timing** (1 to 2 frames per direction, button on the last direction) is an engineering
    choice, not a sourced number.
