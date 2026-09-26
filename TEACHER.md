# Chun-Li teacher: findings and plan (2026-09-25)

The student imitates the scripted teacher (`sf2/teacher.py`), so the teacher is the ceiling. Right now it is worse
than pressing buttons at random.

## Baselines (fixed harness, gate protocol in PROGRESS.md, 20 paired matches each, 2026-09-25)

| Policy | Rounds | Net damage / round | SE | Round wins | Dealt / round |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle | 40 | -173.0 | 0.9 | 0 | 0 |
| random (`play_teacher --policy random`) | 41 | -98.0 | 8.3 | 1 | 71 |
| teacher, eps 0, CLOSE/MID 55/120 | 43 | -78.5 | 7.5 | 4 | 91 |
| teacher, eps 0, CLOSE/MID 80/120 (current) | 43 | -75.2 | 9.0 | 3 | 91 |

The teacher is ahead of random by 19.5 net damage per round, 1.7 combined SE: not yet a proven improvement.
(The earlier "teacher worse than random" finding came from the broken harness: inverted facing made its
`forward` walk away.)

## Which actions work (random rollouts, `rollouts/base_random`, 14,669 decisions)

Net damage (dealt minus taken) in the 0.5 s after each action, per decision, by real distance. Random play, one
opponent, one savestate: the next 0.5 s also depends on the random actions around it, so read it for ranking only.

| Action | close (< 80) | mid (80–120) | far (≥ 120) |
| --- | ---: | ---: | ---: |
| idle | -1.82 | -3.77 | -0.54 |
| forward | -2.55 | -3.28 | -0.53 |
| back | -2.05 | -3.05 | -0.55 |
| jump | -2.70 | -2.46 | -0.42 |
| crouch | -1.87 | -2.69 | -0.78 |
| lp | -2.75 | -3.94 | -0.29 |
| hp | -2.17 | -2.76 | -0.60 |
| lk | -3.06 | -2.67 | -0.48 |
| hk | -2.17 | -2.66 | -0.69 |
| block | -1.25 | -2.76 | -0.22 |
| hadouken | -2.61 | -2.99 | -0.60 |
| shoryuken | -2.52 | -2.95 | -0.38 |

Hit rates by distance (same run): our attacks land most at 40-80 px; Dhalsim hits us 42-47% of the time between 40
and 100 px, 24% at 100-120, 5-8% beyond. Hence CLOSE = 80, MID = 120 (`sf2/ram.py`).

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
