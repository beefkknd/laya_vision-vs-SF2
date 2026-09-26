# Chun-Li teacher: findings and plan (2026-09-26)

The student imitates the scripted teacher (`sf2/teacher.py`), so the teacher is the ceiling. Right now it is no
better than pressing buttons at random.

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
- `crouch` on consecutive decisions holds state 02. `block` (down-back) is the crouching guard: 17 of 17 attacks
  guarded on the right side, 14 on the left, with at most chip damage; `back` is the standing guard and his lows get
  through it (21-63 life lost over the same 300 decisions).
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

## What is wrong in `sf2/teacher.py`

- Mid and far range always pick `forward`: she walks into Yoga Fire and limbs and never guards.
- Guard comes only *after* a hit, only up close, and `block` holds for 6 frames.
- Anti-air requires `dx_trend <= 0`, so Dhalsim's neutral and back jumps are ignored.
- The in-air rule is dead code: `jump` is never the top choice.

## Stage 3: basic rules, one at a time (kept only if the gate improves)

Each rule is added to the current best teacher and gated with the protocol in PROGRESS.md (20 paired matches,
`--workers 4 --seed 4242`). It is kept only if net damage per round beats the current best by at least
2 x sqrt(se_a^2 + se_b^2); otherwise it is reverted. The gate plays the teacher's top choice, so a rule only counts
if it changes that choice. Rollouts: `rollouts/t3_*`.

| Rule | Arm | Net damage / round | SE | Round wins | vs best (bar) | Kept? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (start) | base3_teacher | -84.5 | 6.8 | 1 / 41 | | best |
| 1 block when Dhalsim attacks within MID or a Yoga Fire is within CLOSE | t3_block | -74.6 | 8.6 | 4 / 44 | +9.9 (21.9) | dropped |
| 2 cornered up close: jump out over him (`jump_forward`); never `back` in the corner | t3_corner | -84.1 | 7.1 | 1 / 41 | +0.3 (19.6) | dropped |
| 3 spacing: from far walk in with no buttons; at mid (his limbs) guard, don't walk in | t3_spacing | -90.2 | 7.8 | 1 / 41 | -5.7 (20.7) | dropped |
| 4 up close: roundhouse (`hk` 0.6) instead of fierce | t3_normals | -84.1 | 9.4 | 2 / 42 | +0.3 (23.2) | dropped |
| 5 at mid, when he is not attacking: `jump_forward` (the in-air rule kicks) | t3_jumpin_base | -2.0 | 9.4 | 21 / 52 | +82.5 (23.3) | **kept** |

Rule 5 was also run on top of rule 4 before rule 4's result was in (`t3_jumpin`: -1.1 ± 7.9, 18 / 51); rule 4 adds
nothing there either.

More random-play rollouts, from any machine, sharpen the action table above. Batches are self-contained dirs, so
they can be copied over and pooled.
