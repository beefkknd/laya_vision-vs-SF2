# Pre-registration: a movement dataset (2026-10-01, before any code or collection)

Owner: "stop and collect data" ... "get right data set before redo this training". Owner's diagnosis: each action of
his runs over a stretch of continuous frames (attack, jump, dodge, defend); the model needs all of them, from every
opponent, to learn movement.

## Why the movement test's data was wrong (docs/prereg_movement.md, run stopped at step ~6,450, best at 4,000)
- Frames were saved only at HER decision moments: one 4-frame snapshot at a random stage of his action; startup
  frames look like standing, and the stages were never covered on purpose.
- Opponents lopsided: Dhalsim 42% and Ryu 22% of training moments; Ken, Blanka, Honda, Zangief 7-13% each, only vs
  Chun-Li; Guile only in the held-out test.
- Rare answers thin: blocking 923, crouching 924 decisions in training (Dhalsim crouching 28), repeated up to ~30x.
- Training loss ~1.0 vs validation 1.7 at the stop: memorising. Validation accuracy 0.33-0.38 vs ~0.26 always
  "attacking". No offline eval was run; that run is not used further.

## The collection
- Chun-Li (me) vs each of the 7 opponents (blanka, dhalsim, guile, honda, ken, ryu, zangief), equal games each.
  Headless, no Qwen, no text laya. Her play: runs/all8 with explore 0.5 (as the U collection) so he meets varied
  play. All 7 opponents are in training (owner: cover all characters); generality is reported per opponent and per
  game, not by a held-out opponent.
- RAM: every row of every round is kept (sf2.emu.vs fields, frame-ordered).
- Images: NOT tied to her decisions. A ring buffer of the last 120 displayed frames; when one of his actions ends
  (his movement answer, sf2/data/movement.py, changes), frame pairs (t-4, t) are sampled from that episode at spread
  stages: start, middle, end (one pair for episodes shorter than 6 frames; stage = position / length kept on the
  row). Episodes longer than the buffer are sampled within the buffered part (flagged).
- Balance by quota, not by repetition: a target of 1,500 pairs per (opponent, answer); an answer stops being saved
  for an opponent once its quota is met; games continue until every quota is met or a game cap (stated by the
  builder before running) is reached; shortfalls are reported, never padded by copies.
- Label: sf2/data/movement.py unchanged (lag 1, the 8 answers, unknowns dropped). The two frames and the note
  "me=chunli" are the model's input, exactly as before (so a checkpoint can continue).
- Split by whole games: game % 10 in 2,5,8 -> test; 0 -> validation; the rest train.

## Gates on the dataset, before any training (mechanical)
1. Every (opponent, answer) has >= 1,000 training pairs and >= 200 test pairs; shortfalls listed by name.
2. Stage coverage: for each (opponent, answer) with episodes of 6+ frames, start, middle and end each >= 20% of its
   pairs.
3. Labels re-derived from the stored RAM by an independent check match the rows 100%; RAM-to-image alignment checked
   (lag 1) on a sample.
4. Seeded faults in the collector and builder turn the tests red.
5. A contact sheet (a sample of pairs per answer per opponent) for the owner to look at.
Disk budget: under 10 GB for images.

## Not decided here
The next training run (from runs/u_eye/best or BASE, steps, stop rule) is pre-registered separately after the gates
pass and the owner has looked at the data.
