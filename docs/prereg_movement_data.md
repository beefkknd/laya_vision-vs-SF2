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

## Amendments before the collection (2026-10-01; implementation 272b9fb)
- Quota per split, not flat: a flat 1,500 per (opponent, answer) split 6/3/1 by whole games gives ~900 training
  pairs, below gate 1's 1,000. Per (opponent, answer): train 1,100 / validation 100 / test 300 (1,500 in total), each
  filled only from its own split's games.
- Game cap 1,000 per opponent (Ken expected ~770 from the U collection's RAM: walking away is his rarest answer at
  ~2 pairs per game). Expected ~3 h wall, ~4.3 GB of images.
- Pair at displayed frame t = the captures showing t-4 and t (the same n-4 / n pair as before). Stage thirds by
  position in the episode; "unknown" breaks episodes. Seeds per (seed, opponent, game).
- Gate 3: every row's label re-derived by an independent implementation; alignment by the HUD clock (lag 1 >= 95% of
  400 random pairs, and >= 90% and best on pairs where lags 0/1/2 disagree). Smoke: 0.98 at lag 1 (0.915 lag 0).

## Owner change during the collection (2026-10-01, before the build and gate)
Owner: "walking away for hundreds of games? Doesn't sound right, we want to capture the opponent's movements and that's
it ... once it is in action then we have it, and we move on" / "We want to 'see' the action, once".
- Collection stopped at 60-87 games per opponent (about 12% of the cap); nothing more is collected. Fewest pairs of
  any (opponent, answer): Ken walking away 183, Ken blocking 245, Ryu walking away 223, Dhalsim walking away 225.
- The quota (1,100 train pairs) is dropped as a target. Gate 1 becomes: every (opponent, answer) is SEEN - at least 50
  training pairs and 20 test pairs (enough to measure). Gates 2-3, disk and the contact sheets unchanged.
- Known limit: the RAM fields we read (state, sub, special) do not name his specific move (Ken shows 3 attack codes),
  so "every distinct attack seen" cannot be checked from them; it is shown by the contact sheets, not gated.

## Result (2026-10-01): dataset gate PASS (test_data_mv2/gate.json; scripts/gate_movement_data.py --min-train 50 --min-test 20)
50,513 pairs from 513 games (Chun-Li vs 7 opponents, 60-87 games each), 2.6 GB. Counts: no shortfall (fewest: Ken
walking away 110 train / 50 test). Stages: start/middle/end each >= 20% everywhere. Labels: 50,513 of 50,513
re-derived from RAM match. Alignment (HUD clock): lag 1 0.985 (lag 0 0.93, lag 2 0.90); on 400 pairs where lags
disagree 0.95 vs 0.48 / 0.52. Contact sheets sent to the owner. Training not yet registered.

## Owner change after the gate (2026-10-01): label each action by actor, action and stage
Owner: "we need to label these actions with the actor and its action name and that's it. It is our training set ...
however you label it, chun-li act1-stg2, then that is it." This replaces, for this dataset, the earlier rule that laya
names no opponent and no move.
- Label: "<actor> act<NN> stg<1-3>" for each fighter on the pair (him and her), NN = a per-character action ID read
  from RAM, stage = third of that action's episode (start / middle / end). IDs, not move names.
- The RAM fields logged so far (state, sub, special) do not identify the move (Ken: 3 attack codes). Step 1: a RAM
  probe of each fighter's full struct (0x0C00-0x0DFF me, 0x0E00-0x0FFF him) for a byte that is constant within an
  action episode and differs between actions; checked against frames on a contact sheet before use.
- Step 2: re-collect the same way (Chun-Li vs 7, 60-90 games each, stop when every action seen; ~15 min) logging that
  byte for both fighters; build; gate (labels re-derived 100%, alignment, every (actor, action) seen; counts per
  action reported, rare ones listed, not padded).

## Owner decisions on the label (2026-10-01, after the action-ID probe 2819b09)
Owner: "Same action fine but also need know where as a different question. 2nd ok, but also need a distance question.
If mix them in one fine tune will have more messed up result."
- Action (both fighters): "<actor> act<NN> stg<1-3>". NN = first non-zero move ID (0x0C3E / 0x0E3E) of the state
  episode; a jump with an attack gets the attack's ID; close and far buttons stay separate IDs. No-ID attacks:
  reserved codes (throw, fireball, special named by 0x0C49); other states: reserved codes 70+ (stand, walk toward, walk
  away, crouch, jump, block, hit, thrown). Attacks cut before any ID: unknown, dropped. Mapping for Chun-Li:
  lessons/chunli_action_ids.json.
- Where (separate question): is he on the ground or in the air (from y).
- Distance (separate question): the gap bands already calibrated (lessons/perception_thresholds_v2.json: throw /
  poke / mid / far).
- Three separate datasets from the same collection and, later, three separate fine-tunes - never one mixed run.
  Training is registered separately.
