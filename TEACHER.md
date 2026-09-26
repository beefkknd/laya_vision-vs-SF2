# Chun-Li teacher: findings and plan (2026-09-25)

The student imitates the scripted teacher (`sf2/teacher.py`), so the teacher is the ceiling. Right now it is no
better than pressing buttons at random.

## Baselines (gate protocol in PROGRESS.md, 20 paired matches each, 2026-09-26)

| Policy | Rounds | Net damage / round | SE | Round wins | Dealt / round |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle (`base2_idle`) | 40 | -170.7 | 1.7 | 0 | 0 |
| random (`base2_random`, `play_teacher --policy random`) | 42 | -89.7 | 8.1 | 2 | 81 |
| teacher, eps 0, CLOSE/MID 80/120 (`base2_teacher`) | 41 | -84.5 | 6.8 | 1 | 86 |

The teacher is ahead of random by 5.2 net damage per round, 0.5 combined SE: no better than random.
Superseded (2026-09-25, before the harness fixes listed in PROGRESS.md): idle -173.0, random -98.0, teacher 55/120
-78.5, teacher 80/120 -75.2.

## Which actions work (random rollouts, `rollouts/base2_random`, 11,812 controllable of 15,509 decisions)

Net damage (dealt minus taken) in the 0.5 s after each action, per decision, by real distance; only decisions where
the stick does something (`controllable`). Damage is booked on the decision a hit lands in. Random play, one
opponent, one savestate: the next 0.5 s also depends on the random actions around it, and each cell has 211-542
decisions, so read it for ranking only. (Superseded: the 2026-09-25 table from `base_random`, all decisions, with
damage smeared over the life bar's drain.)

| Action | close (< 80) | mid (80–120) | far (≥ 120) |
| --- | ---: | ---: | ---: |
| idle | -1.02 | -4.32 | -0.55 |
| forward | -2.85 | -4.12 | -0.06 |
| back | -2.58 | -3.42 | -0.34 |
| jump | -1.81 | -2.54 | -0.86 |
| crouch | -0.54 | -4.39 | -0.54 |
| lp | -2.23 | -3.98 | -0.35 |
| hp | -0.99 | -4.07 | -0.28 |
| lk | -0.91 | -3.83 | -1.14 |
| hk | -1.68 | -4.18 | -0.32 |
| block | -2.93 | -2.78 | -0.28 |
| hadouken | -0.12 | -4.25 | -0.34 |
| shoryuken | -1.05 | -4.53 | -0.90 |

Hit rates by distance (2026-09-25, `base_random`): our attacks land most at 40-80 px; Dhalsim hits us 42-47% of
the time between 40 and 100 px, 24% at 100-120, 5-8% beyond. Hence CLOSE = 80, MID = 120 (`sf2/ram.py`).

## Research (web, summarized)

- World Warrior Chun-Li wins with **normals and throws**. She has the best priority in WW and beats Dhalsim's long
  but weak limbs. Her specials are weak in this version: Lightning Legs needs about 5 quick kick taps, Spinning Bird
  Kick is slow, and she has no fireball yet.
  [StrategyWiki moves](https://strategywiki.org/wiki/Street_Fighter_II/Moves?action=raw),
  [Supercombo WW Chun-Li](https://wiki.supercombo.gg/w/Street_Fighter_2:_The_World_Warrior/Chun-Li)
- The SNES CPU reads your input on its first frame and runs counter-scripts on it. Pressing buttons at range invites
  counters; guarding keeps it in its approach scripts.
  [sf2platinum: the AI engine](https://sf2platinum.wordpress.com/2017/01/20/the-ai-engine/)
- Against Dhalsim, a *diagonal* jump-in kick beats Yoga Fire and limbs. There is no forward-jump action yet; `jump`
  goes straight up.
- Rule-based bots key off "opponent attacking" and "fireball on screen"
  ([sf2-bot](https://github.com/SanjoSolutions/sf2-bot)). We don't read either from RAM yet.

## What is wrong in `sf2/teacher.py`

- Mid and far range always pick `forward`: she walks into Yoga Fire and limbs and never guards.
- Guard comes only *after* a hit, only up close, and `block` holds for 6 frames.
- Anti-air requires `dx_trend <= 0`, so Dhalsim's neutral and back jumps are ignored.
- The in-air rule is dead code: `jump` is never the top choice.

## Plan: one change at a time, kept only if the gate improves

Check each step with the gate protocol in PROGRESS.md (20 paired matches). The bar is the current teacher baseline.
Keep a change only if dealt minus taken improves.

1. ~~Calibrate `CLOSE`/`MID`~~ done: 80 / 120.
2. Rules, one at a time: guard more at range; jump at mid range; lp (and Lightning Legs taps) up close; anti-air
   on any opponent jump.
3. Only if needed: new macros (forward jump, a real throw) and new RAM reads (opponent attacking, fireball on screen,
   opponent recovering) so the teacher can block on cue instead of guessing.

More random-play rollouts, from any machine, sharpen the action table above. Batches are self-contained dirs, so
they can be copied over and pooled.
