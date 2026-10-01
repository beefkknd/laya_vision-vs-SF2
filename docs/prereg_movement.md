# Pre-registration: can laya-vision see the opponent's movement? (2026-10-01, before any code or training)

Owner: "ask laya-vision what the movement is and check against RAM whether it is the same or not, and whether movement
is trainable." Context: round 1's eye answered "neutral" 91% of the time when he was attacking (one question of 25,
swamped by the move-value rows); the old r2 never had to see his state - its RAM note told it ("opp=... attack").

## The one question (screen only: two frames n-4 and n, HUD visible, note "me=<character>")
"What is he doing right now?" -> standing / walking toward me / walking away / crouching / jumping / attacking /
being hit / blocking. Label from RAM at the displayed frame (lag 1, as measured): his state (0x00 stand, 0x02 crouch,
0x04 jump or y above ground, 0x08 guard, 0x0A / 0x0C attack or special, 0x0E hit stun with a non-block react, 0x14
thrown -> being hit; 0x0E with a block react -> blocking); a standing frame whose x moved >= 2 px toward / away from me
between n-4 and n -> walking toward / away. General: no opponent names, no move names.

## Data, training
- The U collection's frames and RAM (rollouts/u_perception; all 8 characters as the player, 7 opponents), one row per
  decision; the same game split (test: game % 10 in 2,5,8; val: whole games, capped; Chun-Li vs Guile held out).
- Classes balanced by sampling in training (each answer equally often), so "always standing" cannot win.
- One LoRA run on threebody (RTX 4090), from runs/u_eye/best (it already reads range and air), fixed budget 10,000 steps,
  select on validation NLL of this question, eval every 500, patience 5. Qwen stopped during the run and restarted
  after (standing permission).

## Judged against RAM on held-out games (test_real) and held-out Guile
- Per-answer recall, confusion, overall and BALANCED accuracy, vs the majority baseline (always the most common
  answer) and vs round 1's eye on the same decisions (mapped to these answers where possible).
- "Trainable" = balanced accuracy clearly above chance (1/8) and above the majority baseline's balanced accuracy, with
  attacking recall >= 0.5 (round 1: 0.05). Reported per character and per opponent; nothing else changes.
