# Pre-registration: movement data from all character pairs, both ways (2026-10-01, before any code or collection)

Owner: "you pick two players, play AB and BA, collect data of their movement and their facing ... 10 movement x 2
directions x 2 facing. Try to collect more but cap the total turns so it will NOT be an endless wait ... some
imbalance for sure ... 8x7 = 56 combinations. Then during this we also need in-the-air and close vs far. I think this
is it." Replaces docs/prereg_movement_data.md (that data was deleted: over-complicated).

## Games
- All 56 ordered pairs of the 8 characters (blanka, chunli, dhalsim, guile, honda, ken, ryu, zangief): A as player 1
  vs B as player 2, and B vs A. BOTH players = the game's CPU (owner: "CPU player"). Step 0: find a reliable
  CPU-vs-CPU method (the attract demo plays CPU vs CPU: a RAM flag that hands player 1 to the CPU, or the demo with
  chosen characters); verified on RAM and frames for several pairs before collecting. If none works, stop and report. Headless, no Qwen, no text laya.
- Fixed budget: G games per ordered pair (G = 3 unless the smoke shows the total wall time > 1.5 h; then fewer).
  No "wait for new" rule; the run ends when the budget is spent.

## Samples and labels (both fighters, from RAM; the model sees frames n-4 and n, HUD visible)
- Pairs sampled along each fighter's movement episodes (start / middle / end), as before.
- Movement (~10): stand, walk, crouch, jump, attack, special/projectile, block, being hit, thrown / knocked down.
  From the state byte only; no move IDs.
- Direction: toward / away from the other fighter (for walk and jump; "none" otherwise).
- Facing: left / right (facing byte).
- Air: on the ground / in the air (y).
- Distance: close / far (one cut, the calibrated throw/poke band vs beyond; documented).
- Cap per (character, movement, direction, facing): a fixed number of pairs (e.g. 60), and at most a few per game so
  they spread over games. Shortfalls reported, never padded.
- Each label is its own question (separate datasets, separate fine-tunes later); every row records game, pair (A,B),
  slot, controller (cpu, both).

## Gates (mechanical) before any training
Labels re-derived independently from RAM 100%; frame/RAM alignment (lag 1); counts table per (character, movement,
direction, facing) and per question; disk < 10 GB; one contact sheet per character.

## Fallback if CPU vs CPU is impossible (owner, 2026-10-01)
"use our own system in current system 2 and direct laya-text to make these attacks and defense as player 1 vs CPU and
then use CPU player against our laya-text": player 1 = our system, with text laya directed to cycle through the
character's full set of attacks and defence (not runs/all8's own picks), player 2 = CPU; each ordered pair still
played both ways (AB and BA), so every character is seen as the CPU and as our directed player. Rows record which
side was directed and which was CPU.

## Owner simplification (2026-10-01): brute force
"Play 1 game is fast, you can just hardcode: play one action of A and push through, then the next action. Brute
force." Player 1 = a hardcoded list of the character's actions sent straight to the controller, one after another,
each executed in full, repeated until the game ends. No text laya, no model. Player 2 = CPU.

## Round 1 result and owner decision (2026-10-01)
Round 1 (G=1, 56 ordered pairs, 47 s wall, 4,728 pairs): overall fill 20.5% of the 60-per-bucket caps (G=1 can reach
at most 35%); hardcoded player 1 does 95.8% of its moves when not interrupted. Labels 18,912/18,912 vs RAM, lag 1.
Owner: "if the opponent is not in our control then control the character we have and then label the player we do
have control [of] and that is it." -> datasets keep only player 1's rows (build_pairs_data.py --controller directed).
Owner: "Do not run it, plan it." No further collection until the owner approves the plan.

## Owner: the grid (2026-10-01, planning; nothing run)
"8 player x 2 facing x ~10 moves including block" -> 160 cells (character, facing, movement), the player we control
only. Movement (10): stand, walk toward, walk away, crouch, jump, attack, special, block, being hit, knocked down
(jump direction not a separate cell). Cap 60 per cell (40 train / 20 test by whole game). Air and distance stay
separate questions on the same pairs.

## Plan B (owner: "B then, plan it out ... a matrix of what needs to be collected for which character"; not run)
2-player versus mode: BOTH controllers ours, each a hardcoded action list (no CPU, no model). Both fighters labelled.

Step 0 (check, ~15 min): start a 2P versus match headless with chosen characters for both sides; verify from RAM
(p1_char, p2_char) and a frame, and that each side follows its own controller. If it fails: stop and report.

Matches: the 56 ordered pairs (A left / B right, then B left / A right). Per round each character plays 14 matches:
7 starting on the left (mostly facing right), 7 starting on the right (mostly facing left).

The matrix - every character, the same 20 cells, 60 pairs each (40 train / 20 test by whole game):

| movement      | made by (own list, or the other side's list)                     | facing R | facing L |
|---------------|------------------------------------------------------------------|----------|----------|
| stand         | idle step                                                        | 60       | 60       |
| walk toward   | walk forward                                                     | 60       | 60       |
| walk away     | walk back (no attack coming)                                     | 60       | 60       |
| crouch        | hold down                                                        | 60       | 60       |
| jump          | jump up / toward / away (no attack in the jump)                  | 60       | 60       |
| attack        | 6 standing + 6 crouching normals, jump attacks, throw            | 60       | 60       |
| special       | the character's specials (below)                                 | 60       | 60       |
| block         | hold back / down-back while the OTHER side's attack arrives      | 60       | 60       |
| being hit     | the other side's attack lands                                    | 60       | 60       |
| knocked down  | the other side's sweep / throw / special knocks down             | 60       | 60       |
Per character 1,200 pairs; 8 characters 9,600. A jump attack counts as "attack" (the air question says "in the
air"; owner earlier: same action, where is a separate question). Air (ground / air) and distance (close / far) are
answered on the same pairs.

Specials in the lists: blanka electricity, rolling attack; chunli lightning legs, spinning bird kick; dhalsim yoga
fire, yoga flame; guile sonic boom, flash kick; honda hundred hand slap, sumo headbutt; ken and ryu hadoken,
shoryuken, tatsumaki; zangief spinning piledriver, clothesline.

Interaction: block / being hit / knocked down need the two sides close; each list includes walk-forward steps, so they
meet. Spread: at most 3 pairs per (cell, match) per game, so a cell needs >= 20 matches -> >= 2 rounds.

Budget: one round = 56 matches (~1 min, from round 1's ~15 s per game x 24 workers). After round 1: report the matrix
% filled per character x cell (owner decides); hard limit 5 rounds. Shortfalls reported by name, never padded.
After collection: build 4 datasets (movement 10, facing 2, air 2, distance 2), gate (labels vs RAM 100%, lag 1, counts
matrix, disk), contact sheet per character. Training registered separately, one fine-tune per question.
