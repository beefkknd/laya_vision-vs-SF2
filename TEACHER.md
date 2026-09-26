# Chun-Li teacher: findings and plan (2026-09-25)

**The tables below are invalid** (2026-09-25): they were measured on the broken harness (camera-byte distance,
inverted facing, knockdowns counted as jumps, time-overs scored as draws) and their data is deleted. They are
re-measured under the gate protocol in PROGRESS.md. The research notes still stand.

The student imitates the scripted teacher (`sf2/teacher.py`), so the teacher is the ceiling. Right now it is worse
than pressing buttons at random.

## Baselines (v2 loop, Chun-Li vs Dhalsim, `states/chunli_vs_dhalsim.state`, 6 matches each)

| Policy | Rounds | Dealt / round | Taken / round | Round wins |
| --- | ---: | ---: | ---: | ---: |
| random (`play_teacher --policy random`) | 14 | 96.1 | 176 | 0 |
| teacher, eps 0 (`play_teacher`) | 12 | 36.9 | 176 | 0 |

The teacher's argmax labels only ever pick 4 of the 12 moves (forward 37%, hk 35%, hp 24%, block 5%). Its soft
targets do give jump, lk and crouch 10-20%, but the student plays its top choice, so those moves never happen.

## Which actions work (random rollouts, `rollouts/v2_random`, ~6k decisions)

Damage dealt and taken in the 0.5 s after each action, per decision, by distance (`dx`: close < 55, mid < 120).
Noisy (one opponent, one savestate), but it matches the research below.

| Action | Distance | n | Dealt | Taken | Net |
| --- | --- | ---: | ---: | ---: | ---: |
| jump | mid | 142 | 1.16 | 0.61 | **+0.55** |
| lp | close | 211 | 2.67 | 2.47 | +0.20 |
| hp | close | 201 | 2.82 | 2.90 | -0.08 |
| hk | mid | 183 | 0.71 | 1.16 | -0.45 |
| forward | mid | 150 | 0.27 | 1.20 | -0.93 |
| hadouken macro | close | 184 | 1.49 | 3.39 | -1.90 |
| forward | far | 139 | 1.43 | 3.49 | **-2.06** |

The current teacher's main choices at mid and far range (forward, hk) are among the worst. The hadouken macro ends
in F+Fierce, which the research suggested might throw; the data says it doesn't help.

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
- `CLOSE`/`MID` (55/120) in `sf2/ram.py` are still uncalibrated ("check on day 1").

## Plan: one change at a time, kept only if the gate improves

Check each step with `parallel.py play_teacher` (6 matches) and `gate.py`. The bar is random: 96 dealt / 176 taken.
Keep a change only if dealt minus taken improves.

1. Calibrate `CLOSE`/`MID` from logged `dx` (sprites touching, throw range).
2. Rules, one at a time: guard more at range; jump at mid range; lp (and Lightning Legs taps) up close; anti-air
   on any opponent jump.
3. Only if needed: new macros (forward jump, a real throw) and new RAM reads (opponent attacking, fireball on screen,
   opponent recovering) so the teacher can block on cue instead of guessing.

More random-play rollouts, from any machine, sharpen the action table above. Batches are self-contained dirs, so
they can be copied over and pooled.
