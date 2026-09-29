> **Historical (2026-09-29).** This is the first plan: a scripted teacher + DAgger. That pipeline was deleted
> (git tag `legacy-dagger`). The current system is described in README.md; the plan for System 2 (Qwen) and its
> measurements are in docs/qwen_learning.md.

# laya-vision → Street Fighter (small closed loop)

**laya-vision** is a System-1 model: **one screenshot (+ a little text) → probabilities over buttons you list.** It does not invent combos, parse video, or get better from “I won/lost the round” alone. You supply a teacher, turn fights into labeled rows, LoRA, play, record, relabel, repeat.

Street Fighter is a better *demo* than an open-world game (both characters stay in frame). It is a *worse* one-frame toy than Mario walk-right: fireballs and links are **sequences**. For a small project, treat specials as **macros the script executes**, not as the VLM tapping Down-Down-Forward-Punch over four frames.

---

## Roles

| Piece | Job |
|---|---|
| Emulator / fight env | Frames + you can inject inputs |
| Teacher | Writes the gold **action** on a frame |
| Dataset | `{image, text_state, choice options, gold}` |
| LoRA | Copies the teacher on laya-vision |
| Student | Plays; you record what it did |
| Relabel | Teacher (or a filter) labels **new** screens |
| Gate | Round wins, damage dealt, not val loss |

Do not use a chat VLM as the main teacher. Same failure mode as 64 random Othello rollouts.

---

## Action set (keep it tiny)

Start with **8–12 options**, not every stick+button combo:

```text
idle, left, right, jump, crouch,
lp, hp, lk, hk,
block,
hadouken,          # script: D, DR, R + punch
shoryuken          # script: R, D, DR + punch
```

`hadouken` is one Laya choice; your glue code spends the next 8–12 frames stuffing the motion. If you ask laya-vision to emit each cardinal direction itself, the project dies in the first week.

Text state (short — this backbone is small):

```text
me=ryu opp=ken dist=mid my_hp=80 opp_hp=45 last=hadouken airborne=0
```

If the emulator exposes RAM (positions, animation ids), use that in text. Do not caption the screen with an LLM.

---

## Cycle you will actually run

```
0. Seed teacher
      existing SF bot / your pad recording / a dumb script
      dump 20k–100k (frame, gold_action) from training mode + a few arcade rounds

1. Pack JSON
      image = frame
      questions.act = choice[ those 8–12 options ]
      gold.act = teacher’s label

2. LoRA laya-vision
      small rank, few epochs
      early-stop on a held-out set of FRAMES from the same teacher
      not on “val loss looks nice”

3. Student plays
      1 decision every 2–4 frames (or every 8 if macros eat frames)
      record: frame, student action, teacher action (if available),
              damage_delta, round_result, x_distance

4. Relabel (this is the learning)
      a. DAgger (best): teacher writes gold on student frames,
         especially hits you took, whiffed specials, knockdowns
      b. Filter (cheap): keep student actions that were followed by
         damage_for > damage_against in the next 0.5s
      c. Do not mark every frame in a lost round as wrong

5. Merge seed + new rows → LoRA again

6. Gate (same characters, same stage)
      win rate vs the teacher or vs a fixed dummy
      average damage per round
      hadouken-whiff rate
```

If win rate and damage do not move, the new labels are too coarse. More raw video will not help.

---

## Teacher ladder (use in this order)

1. **You + input log** in training mode (10–20 minutes of clean anti-air / fireball / block).
2. **A scripted dummy**: jump-in punish, always-fireball Ken. Enough to get a first LoRA that is not random.
3. **An existing bot** if you have one that can play the same emulator.
4. Optional later: inverse-dynamics only if you insist on unlabeled YouTube. Not for v1.

Succeed/fail of the **round** is only a **weight or a filter**. The LoRA target is always **which option on this frame**.

---

## What “it can keep playing Street Fighter” means

After the first LoRA it will:

- look at a still and pick among your 12 options
- sometimes walk forward and throw a fireball
- lose most rounds

After a few DAgger loops it may:

- fireball more at mid range if that is what you labeled
- block more if you labeled block on incoming jumps

It will not:

- learn Ken’s whole matchup from win/lose bits
- time one-frame links
- become a ranked player because the tape got longer

That is still a successful **small project** if the gate vs a dummy Ken goes from ~10% to ~30–40% rounds and damage is less embarrassing.

---

## One-week shape

| Day | Work |
|---|---|
| 1 | Hook emulator, screenshot, send one of 12 actions, record a pad log |
| 2 | Dump seed set from you or a scripted dummy |
| 3 | First LoRA; student vs dummy, log win/damage |
| 4–5 | DAgger: teacher labels student frames around hits and whiffs |
| 6 | Second LoRA, same gate |
| 7 | Decide: stop, or add one macro (`anti_air`) and repeat the loop |

**One sentence:** laya-vision is the student; Street Fighter frames plus a **frame-level action teacher** are the class; play → record → relabel → LoRA is the semester; round win/lose only tells you which tapes to keep, not what button was right.
