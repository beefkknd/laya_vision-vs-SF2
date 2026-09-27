# System 2 knowledge

What System 2 has learned about coaching laya. Every line is a measured result; nothing here is a guess.
Plan: docs/TWO_SYSTEM_PLAN.md. Runs: `out/results.jsonl`. Reset 2026-09-27: everything about a scripted player
was removed, so this file starts nearly empty, like the playbook.

## What the note can and cannot say
- It can: each fighter's state word, life, distance (close < 80 px, mid < 120 px, far), facing, corner, clock, last
  move, fireball distance.
- It cannot: jump height, exact distance, whether he is closing in, or which move he is doing.
- A rule that needs something the note cannot say will misfire. Prefer rules whose trigger the note states exactly.

## How rules act
- A rule changes a move only where laya half-agrees (the τ nudge). Log *fired* and *changed*: a rule that never
  changes a move does nothing.
- Judge a rule by play on held-out openings, never by the 0.5 s after the move it changed.

## laya's hesitation
- Decisions where laya is unsure lead to a hit *less* often than confident ones (10.0% vs 15.5%), so moments are the
  hits and audits, not the unsure decisions. (Measured on the retired r2; re-check on the new checkpoint.)

## Qwen as System 2 (qwen38-27b-oq4e-mtp)
- **Thinking off works:** from 30 moments and only the game's description, it wrote 5 valid rules and 0 rejected
  lines, in 58 s on claw.
- **Thinking on did not finish** within 8,000 tokens (651 s on claw). Its reasoning arrives without an opening
  `<think>` tag; only text after `</think>` counts.
