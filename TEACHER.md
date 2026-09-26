# Chun-Li teacher: findings and plan (2026-09-26)

The student imitates the scripted teacher (`sf2/teacher.py`), so the teacher is the ceiling. After Stage 4 it
wins 40 of 40 rounds, +161.6 net damage per round above random play with the 14-action set (Stage 4 below).

## Baselines (gate protocol in PROGRESS.md, 20 paired matches each, 2026-09-26)

Basic action set (Stage 3): `idle forward back jump jump_forward crouch lp hp lk hk block`.

| Policy | Rounds | Net damage / round | SE | Round wins | Dealt / round |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle (`base2_idle`) | 40 | -170.7 | 1.7 | 0 | 0 |
| random (`base3_random`, `play_teacher --policy random`) | 50 | -42.5 | 9.4 | 14 | 121 |
| teacher, eps 0, CLOSE/MID 80/120 (`base3_teacher`) | 41 | -84.5 | 6.8 | 1 | 86 |

With a forward jump to pick, random play got much better (-89.7 to -42.5); the teacher never picks it, so its fights
are unchanged. The teacher is now 42.0 net damage per round *behind* random (3.6 combined SE).
Superseded: the 12-action set of 2026-09-26 (random `base2_random` -89.7 ± 8.1, teacher `base2_teacher` -84.5 ± 6.8),
and everything from 2026-09-25 before the harness fixes listed in PROGRESS.md (idle -173.0, random -98.0, teacher
55/120 -78.5, teacher 80/120 -75.2).

## Which actions work (random rollouts, `rollouts/base3_random`, 18,332 controllable of 23,889 decisions)

Net damage (dealt minus taken) in the 0.5 s after each action, per decision, by real distance; only decisions where
the stick does something (`controllable`). Damage is booked on the decision a hit lands in. Random play, one
opponent, one savestate: the next 0.5 s also depends on the random actions around it, and each cell has 367+
decisions, so read it for ranking only. (Superseded: the 12-action table from `base2_random`, and the 2026-09-25
table from `base_random`.)

| Action | close (< 80) | mid (80–120) | far (≥ 120) |
| --- | ---: | ---: | ---: |
| idle | -0.28 | -1.76 | -1.06 |
| forward | 0.77 | -2.77 | -0.64 |
| back | 0.49 | -1.30 | -0.73 |
| jump | 0.52 | -1.64 | -0.45 |
| jump_forward | 0.52 | -0.71 | -0.48 |
| crouch | 0.30 | -1.23 | -0.53 |
| lp | 0.39 | -3.13 | -0.80 |
| hp | 0.51 | -3.32 | -0.73 |
| lk | -0.11 | -2.74 | -1.03 |
| hk | 1.31 | -2.21 | -0.40 |
| block | 0.32 | -1.46 | -0.84 |

Up close almost everything is ahead and roundhouse (`hk`) is best; mid range (Dhalsim's limbs) loses with every
action, least with `jump_forward`, most with walking in or pressing buttons; at range everything is slightly behind.
Random play now spends 48% of its controllable decisions up close (8,772), where the teacher's `hp` is middling.

Hit rates by distance (2026-09-25, `base_random`): our attacks land most at 40-80 px; Dhalsim hits us 42-47% of
the time between 40 and 100 px, 24% at 100-120, 5-8% beyond. Hence CLOSE = 80, MID = 120 (`sf2/ram.py`).

## The basic action set on the ROM (`tests/test_rom_harness.py`)

- `jump_forward` (up + toward him): rises ~96 px, travels ~90 px toward Dhalsim on either side of the screen, lands
  after ~50 frames (it can carry her over him). A button in the air is an air attack: state 04 with sub-state 06
  (Dhalsim's air attacks are 0A); the note says `jumpattack`.
- `crouch` on consecutive decisions holds state 02. `block` (down-back) is the crouching guard and guards every
  attack Dhalsim has here, his air attacks included; `back` is the standing guard and his crouching limbs (the lows)
  get through it. Measured by holding each for 500 decisions, both sides, 3 starts each (Stage 3b guard table below).
- Near both walls every action keeps her inside the walls and facing him; `back` goes nowhere.
- Lightning Legs (state 0C) start after 10 `hk` or 11 `lk` decisions in a row (9 do not); mixed kick patterns start
  them sooner (`lk lk idle` repeated, `lk`/`hk` alternating). Random play never did (0 of 6,513 decisions in 6
  matches; its longest kick run in `base3_random` was 5).

## Research (web, summarized)

- World Warrior Chun-Li wins with **normals and throws**. She has the best priority in WW and beats Dhalsim's long
  but weak limbs. Her specials are weak in this version: Lightning Legs needs about 5 quick kick taps, Spinning Bird
  Kick is slow, and she has no fireball yet.
  [StrategyWiki moves](https://strategywiki.org/wiki/Street_Fighter_II/Moves?action=raw),
  [Supercombo WW Chun-Li](https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li)
- The SNES CPU reads your input on its first frame and runs counter-scripts on it. Pressing buttons at range invites
  counters; guarding keeps it in its approach scripts.
  [sf2platinum: the AI engine](https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/)
- Against Dhalsim, a *diagonal* jump-in kick beats Yoga Fire and limbs: `jump_forward`, then `hk` in the air.
- Rule-based bots key off "opponent attacking" and "fireball on screen"
  ([sf2-bot](https://github.com/SanjoSolutions/sf2-bot)). Both are read from RAM now (`opp_attacking`, `fireball`).

## What was wrong in `sf2/teacher.py` (before Stage 3)

- Mid and far range always pick `forward`: she walks into Yoga Fire and limbs and never guards. (Mid range now
  jumps in when he is not attacking; rule 5 below.)
- Guard comes only *after* a hit, only up close, and `block` holds for 6 frames. (Guarding on his attacks, rule 1,
  did not clear the bar in Stage 3; on top of the apex jump-in kick it did, Stage 3b.)
- Anti-air requires `dx_trend <= 0`, so Dhalsim's neutral and back jumps are ignored. (Dropping that, rule 6, made
  it worse.)
- The in-air rule was dead code: `jump` was never the top choice. It now kicks every jump-in, near the top of the
  arc (Stage 3b).

## Stage 3: basic rules, one at a time (kept only if the gate improves)

Each rule is added to the current best teacher and gated with the protocol in PROGRESS.md (20 paired matches,
`--workers 4 --seed 4242`). It is kept only if net damage per round beats the current best by at least
2 x sqrt(se_a^2 + se_b^2); otherwise it is reverted. The gate plays the teacher's top choice, so a rule only counts
if it changes that choice. Rollouts: `rollouts/t3_*`.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) | base3_teacher | -84.5 | 6.8 | 1 / 41 | | best |
| 1 block when Dhalsim attacks within MID or a Yoga Fire is within CLOSE | t3_block | -74.6 | 8.6 | 4 / 44 | +9.9 (21.9) | dropped |
| 2 cornered up close: jump out over him (`jump_forward`); never `back` in the corner | t3_corner | -84.1 | 7.1 | 1 / 41 | +0.3 (19.7) | dropped |
| 3 spacing: from far walk in with no buttons; at mid (his limbs) guard, don't walk in | t3_spacing | -90.2 | 7.8 | 1 / 41 | -5.7 (20.7) | dropped |
| 4 up close: roundhouse (`hk` 0.6) instead of fierce | t3_normals | -84.1 | 9.4 | 2 / 42 | +0.3 (23.2) | dropped |
| 5 at mid, when he is not attacking: `jump_forward` (the in-air rule kicks) | t3_jumpin_base | -2.0 | 9.4 | 21 / 52 | +82.5 (23.3) | **kept** |
| 6 anti-air on any jump of his within MID, not only toward her (on top of 5) | t3_antiair | -12.7 | 9.2 | 22 / 51 | -10.8 (26.4) | dropped |

Rule 5 was also run on top of rule 4 before rule 4's result was in (`t3_jumpin`: -1.1 ± 7.9, 18 / 51); rule 4 adds
nothing there either.

**Result (Stage 3):** the teacher was `base3_teacher` + rule 5: -2.0 ± 9.4 net damage per round, 21 of 52 rounds
won, against random's -42.5 ± 9.4 (14 / 50). Rules 1-3 were re-run on top of it in Stage 3b.

## Stage 3b: the dropped rules again, jump-in timing, guarding (2026-09-26)

Same protocol and keep bar. Rollouts: `rollouts/t3b_*`. Every arm is a teacher variant; its rollout rows now carry
her action state (`my_state`). The gate is deterministic: `t3b_teacher` (the Stage 3 teacher again) and
`t3b_antiair` reproduced `t3_jumpin_base` and `t3_antiair` to the last digit.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) Stage 3 teacher (rule 5) | t3_jumpin_base | -2.0 | 9.4 | 21 / 52 | | best |
| 1 block when he attacks within MID or a Yoga Fire is within CLOSE | t3b_block | -5.3 | 10.0 | 23 / 49 | -3.3 (27.6) | dropped |
| 2 cornered up close: `jump_forward` out over him | t3b_corner | -3.0 | 9.6 | 21 / 49 | -1.1 (26.9) | dropped |
| 3 spacing: at mid, when he attacks, `block` instead of walking in | t3b_spacing | +23.3 | 10.1 | 29 / 47 | +25.3 (27.7) | dropped (near miss) |
| 6 anti-air on any jump of his within MID | t3b_antiair | -12.7 | 9.2 | 22 / 51 | -10.8 (26.4) | dropped |
| 7 in a jump, kick only once 88+ px up (near the apex); `idle` before | t3b_apex | +31.6 | 8.9 | 34 / 48 | +33.6 (25.9) | **kept** |
| 3 spacing, on top of 7 | t3b_apex_spacing | +44.4 | 8.5 | 37 / 52 | +12.8 (24.6) | dropped |
| 2 corner, on top of 7 | t3b_apex_corner | +31.9 | 8.2 | 33 / 50 | +0.3 (24.2) | dropped |
| 1 guard, on top of 7 | t3b_apex_block | **+62.0** | 10.8 | 36 / 46 | +30.4 (27.9) | **kept** |
| 2 corner, on top of 7 + 1 | t3b_g_corner | +72.0 | 9.5 | 37 / 44 | +10.0 (28.8) | dropped |
| 6 anti-air, on top of 7 + 1 | t3b_g_antiair | +52.6 | 8.4 | 38 / 44 | -9.3 (27.3) | dropped |

Rule 3 was not re-run on top of rule 1: at mid, rule 1 already guards when he attacks, which is all rule 3 does now.
The guard is always `block` (crouching): see the guard table.

**Jump-in timing (rule 7), measured on the ROM.** 95 starts (48 on the left of the screen, 47 on the right) at
80 ≤ dx < 120 with her on the ground and him not attacking; from one saved state per start, `jump_forward`, then `hk`
at decision k after it (idle otherwise), 96 frames scored. She is airborne from decision 2 on (the Stage 3 rule
kicked there) and lands after decision 12.

| k | height (px) | hit rate | dealt | taken | net ± SE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| no kick | | 0.00 | 0.0 | 3.9 | -3.9 ± 0.8 |
| 1 (take-off) | 0 | 0.11 | 2.9 | 3.8 | -0.9 ± 1.4 |
| 2 | 17 | 0.11 | 2.9 | 4.2 | -1.3 ± 1.4 |
| 3 | 45 | 0.13 | 3.5 | 4.3 | -0.7 ± 1.4 |
| 4 | 67 | 0.09 | 2.6 | 4.0 | -1.4 ± 1.3 |
| 5 | 83 | 0.27 | 7.7 | 2.1 | +5.5 ± 1.5 |
| 6 | 92 | **0.69** | 19.4 | 0.6 | **+18.7 ± 1.3** |
| 7 (apex) | 96 | 0.52 | 14.4 | 0.0 | +14.4 ± 1.4 |
| 8 | 94 | 0.65 | 18.2 | 0.4 | +17.8 ± 1.4 |
| 9 | 85 | 0.54 | 14.9 | 0.2 | +14.7 ± 1.4 |
| 10 | 71 | 0.42 | 11.8 | 0.4 | +11.4 ± 1.5 |

The same shape on both sides of the screen (k 6: 0.65 left, 0.74 right) and at both halves of mid range (0.76 at
dx < 100, 0.66 at ≥ 100), so the rule is one height, not a function of distance or side: kick at ≥ 88 px up, which
is decision 6.

**Guard table (R3), measured on the ROM.** Hold one guard for 500 decisions from 3 starts on each side of the
screen; every Dhalsim attack (a run of his state 0A, or a Yoga Fire) is scored by what happened to her.

| His attack | `block` (crouching) | `back` (standing) |
| --- | --- | --- |
| ground limbs (0A) | 66 guarded, 5 hit, 10 no contact | 22 guarded, **22 hit**, 18 no contact |
| air attacks (0A in the air) | 18 guarded, 1 hit (1 life) | 12 guarded |
| Yoga Fire | 24 guarded, 2 hit | 15 guarded, 2 hit |
| life lost per 1,000 frames | 26.1 (454 in 17,406 frames) | 48.1 (577 in 12,000 frames) |

- Crouching guard stops everything here, air attacks included, so the guard rule needs no high/low cue.
- The limbs that beat standing guard are his crouching ones: he was crouching (state 02) or landing (04) just
  before 22 of the 22 that hit; standing (00) before 16 of the 22 that were guarded. 0x0E44 (01 crouching, 00 standing)
  tells them apart at the attack's first frame (22 / 22 hits, 4 / 22 guarded, 18 of 18 standing ones guarded).
- What still costs life under `block` is not his limbs but **throws** (6 in 17,406 frames, 40 life each) and **chip**
  from blocked Yoga Fire and Yoga Flame (8 each). A third arm, standing guard only against air attacks, lost
  19.1 per 1,000 frames; the difference from `block` is 2 throws instead of 6, not the air attacks.

**Lightning Legs (state 0C) in the teacher's own play.** The Stage 3 teacher pressed `hk` on every airborne decision,
so a jump-in plus the next kicks ran 9-16 `hk` in a row: 103 Legs in 20 matches, 9.4% of all decisions in 0C
(`t3b_teacher`). Kicking only at the apex cut that to 15 (`t3b_apex`); the final teacher does 19 in 20 matches, 1.2%
of decisions. Nothing in the teacher asks for them; they come from `hk` runs of 9-10 (a jump-in, then roundhouses up
close).

**Dhalsim's state 04 with sub-state 08** is the main, floaty part of his jump: 15 frames of 04 / 02 rising from y 192
to 140, then 51 frames of 04 / 08 (up to y 106 and down to 179, drifting 38 px or not at all), then 04 / 04 landing
(35 of 35 in the fixtures). The note's `jump` is right.

**Result (Stage 3b):** the teacher is the Stage 3 teacher + rule 7 (apex kick) + rule 1 (crouch-guard his attacks
and close Yoga Fires): **+62.0 ± 10.8** net damage per round, 36 of 46 rounds won (`t3b_apex_block`), against
random's -42.5 ± 9.4: +104.5, 7.3 combined SE.

More random-play rollouts, from any machine, sharpen the action table above. Batches are self-contained dirs, so
they can be copied over and pooled.

## Stage 4: fancy moves (2026-09-26)

Same gate protocol and keep bar. Rollouts: `rollouts/t4_*`. Each move is an action (`sf2/actions.py`), checked on
the ROM (`tests/test_rom_harness.py`), then given a teacher rule.

### Throw (`throw`: toward + fierce on the same frame, 4 frames)

Measured on the ROM from 58 saved starts at 20-100 px, both sides of the screen, both fighters on the ground
(`scratchpad/s4/throw1.py`, `throw2.py`):
- **Toward or back + fierce, or + strong,** throws Dhalsim when he is **within 42 px at the press frame** (23 of 23
  at ≤ 42; none at 44+). No kick throw (toward / back + roundhouse or forward: 0 of 58). Pressing the direction first
  gains nothing: direction and button on the same frame throw just as well.
- She is in 0A for 61 frames; he stays 00 for ~31 frames, then 14 (thrown) and loses **46** life.
- Toward throws him forward (he lands 33-52 px further away, same side); **back throws him behind her** (the sides
  swap). The action is the toward throw.
- He must be on the ground: taking off (04) he cannot be thrown.
- Out of range it is a plain fierce: the same 30 frames of 0A as `hp`.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) Stage 3b teacher | t3b_apex_block | +62.0 | 10.8 | 36 / 46 | | best |
| 8a within 40 px, him standing or crouching: `throw` | t4_throw | +70.9 | 10.2 | 37 / 45 | +8.9 (29.7) | dropped |
| 8a, before the just-hit guard | t4_throw_b | +69.7 | 10.1 | 37 / 45 | +7.8 (29.5) | dropped |
| 8b 40-80 px, him standing or crouching: walk in (`forward`), no throw | t4_walk_only | +59.2 | 9.9 | 37 / 47 | -2.8 (29.2) | dropped |
| **8 = 8a + 8b: walk in, then throw** | t4_throw_walk | **+104.5** | 6.9 | **40 / 40** | +42.5 (25.6) | **kept** |

Alone, the throw rule rarely fires (20 throws in 20 matches: he is seldom that close and still); walking in alone
only trades hits. Together she lands 49 throws in 20 matches, and he throws her less (19 vs 23).

### Sweep (`sweep`: down + roundhouse on the same frame, 4 frames)

Measured on the ROM from 79 saved starts at 20-150 px (`scratchpad/s4/sweep1.py`):
- Down + roundhouse on the same frame is the sweep; pressing down first gives the same move. 32 frames of 0A
  (standing roundhouse: 33).
- **Every sweep that connects knocks him down** (0E, sub-state 04): 22 of 22 hits. The crouching forward kick
  (down + A) hit 18 times and never knocked him down. Standing roundhouse hits knocked him down 1 of 12 times.
- **Reach ~70 px**: hits at 23-70 px, none from 73 px on. He blocks it often up close (16 of 24 at < 40 px).
- Net damage in the next 80 frames by distance, sweep vs standing roundhouse: < 40 px -7.0 vs -4.2, 40-60 px -17.6 vs
  -15.7, **60-80 px +11.5 vs -4.5**, beyond 80 px both whiff.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) rule 8 teacher | t4_throw_walk | +104.5 | 6.9 | 40 / 40 | | best |
| 9a 55-75 px, him standing or crouching: `sweep` instead of walking in | t4_sweep_a | +107.7 | 8.2 | 40 / 41 | +3.2 (21.5) | dropped |
| 9b up close otherwise (he is guarding, hit or landing): `sweep` instead of `hp` | t4_sweep_b | +118.8 | 7.2 | 40 / 41 | +14.3 (20.0) | dropped (near miss) |
| 9a + 9b | t4_sweep_c | +114.9 | 6.8 | 40 / 40 | +10.4 (19.4) | dropped |

The teacher already wins every round; what is left to gain is the ~70 life per round it still takes, and 20 paired
matches cannot resolve 10-15 points of that. The sweep stays an action the student can pick; the teacher does not
use it.

### Lightning Legs (`lightning_legs`: 12 short-kick taps, 1 frame down and 1 up, 24 frames)

Measured on the ROM (`scratchpad/s4/legs1.py`-`legs4.py`), from standing starts on both sides of the screen:
- **12 short taps at 1 on / 1 off start the Legs (0C) from 39 of 40 starts**, at frame 18: the first tap is a
  17-frame short, and the taps counted during it turn into Legs. 10 or 11 taps miss 5 of 20 up close on the right
  side; 8 never do. 2-on / 1-off or 1-on / 2-off need the same count and start at frame 24; 2 / 2 at frame 40.
- Roundhouse taps do not start them inside one decision (0 of 40 at 12 taps; 10+ `hk` decisions do, Stage 3), nor
  do short / roundhouse alternating taps or all three kicks at once.
- Starts where she is being hit fail, as any input would.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) rule 8 teacher | t4_throw_walk | +104.5 | 6.9 | 40 / 40 | | best |
| 10a up close otherwise (he is guarding, hit or landing): `lightning_legs` instead of `hp` | t4_legs_b | +66.3 | 9.0 | 39 / 43 | -38.1 (22.7) | dropped (worse) |
| 10b he attacks on the ground at 60-120 px: `lightning_legs` into his limbs instead of guarding | t4_legs_d | +31.7 | 9.6 | 34 / 51 | -72.7 (23.6) | dropped (worse) |
| 11a anti-air: fierce, not roundhouse, on top (no `hk` runs) | t4_noacc | +90.8 | 9.2 | 39 / 42 | -13.7 (23.1) | dropped |
| **11 anti-air within 60 px: fierce on top; roundhouse further out** | t4_noacc2 | **+105.5** | 6.8 | **40 / 40** | +1.1 (19.4) | **kept** (no cost, fewer accidental Legs) |

Rule 11 is kept although it does not clear the bar: its job is to stop accidental Legs without costing gate
score. The teacher never asks for the Legs; they lose to everything Dhalsim does here.

**Accidental Legs** (Legs starts not on a `lightning_legs` decision, 20 matches each):

| Arm | Legs starts | not from `lightning_legs` | longest plain kick run |
| --- | ---: | ---: | ---: |
| t4_throw_walk (rule 8) | 20 | 20 | 19 |
| t4_noacc (all anti-airs fierce) | 0 | 0 | 7 |
| t4_noacc2 = final teacher | 7 | 7 | 16 |
| t4_legs_b / t4_legs_d (Legs as an action it uses) | 172 / 126 | 11 / 19 | 19 / 19 |
| t4_random (14 actions) | 251 | 10 | 5 |

Plain `hk` / `lk` runs still set them off exactly as before (the macros did not change): 10 `hk` or 11 `lk`
decisions in a row (ROM check). Random play has no such runs, but its mixed kicks (`lk`, `hk`, `sweep` taps) still
start a few.

**Result (Stage 4):** the teacher is the Stage 3b teacher + rule 8 (walk into throw range, throw) + rule 11 (close
anti-airs with fierce): **+105.5 ± 6.8** net damage per round, **40 of 40 rounds** won (`t4_final`, the same
fights as `t4_noacc2`), dealt 176 and taken 70 per round. Random play with the 14-action set (`t4_random`):
-56.0 ± 7.4, 3 of 42. The teacher is +161.6 ahead, 16.1 combined SE.
