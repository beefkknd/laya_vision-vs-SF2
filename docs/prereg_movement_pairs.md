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

## Plan B round 1 result (2026-10-01)
Step 0: VS BATTLE savestates for all 56 ordered pairs (scripts/make_vs_pair_states.py; boot_vs gained cursor plans for
the Ken/Ryu swap). Every state checked: RAM p1_char / p2_char (A, B) over 300 frames; 300 idle frames with both pads
empty - neither x moves, both states stay 0 (no CPU); P1-only forward moves only P1, P2-only forward moves only P2;
56/56 pass, frames in out/vs_pair_states/. The stage is the same (Ryu's) in every match.
Choices: a jump attack is "attack" while its box is out (0x?C3E != 0), the rest of that jump "jump"; cells = (character,
movement10, facing), both players pooled; at most 3 pairs per (slot, cell) per game; split by match (crc32 of
"<A>_vs_<B>:<game>" % 3 == 2 test). Tests + 43 of 43 seeded faults caught.
Round 1 (G=1, 56 matches, 90 s wall, ~29 s per game, 24 workers): 4,436 pairs, 199 MB; moves done 5,571 of 6,111
(interrupted 299, held 124, missed 101, cut 16). Gate PASS: labels 17,744/17,744 vs RAM, caps, lag 1 (0.95 on
discriminating pairs vs 0.48 / 0.52 at lag 0 / 2), disk 0.19 GB. Fill 46.2% of the 160 x 60 grid (per character
39-52%). One zero cell: zangief special facing left (right 20%): Zangief's clothesline and spinning piledriver run in
state 0x0A without move class 0x08, so the shared rule labels them "attack" (owner decision). Block: 2.4-3.4% of each
character's rows, 29-55 episodes per character over 11-14 of its 14 matches. Lowest cells (walk toward, ~25-35%)
need ~4 rounds; estimate 3-4 rounds in all (limit 5). Stopped after round 1 for the owner.

## Owner after round 1 (2026-10-01): "Fix and keep on"
- Fix: since both controllers are ours, "attack" vs "special" is taken from the move WE pressed (its kind in the
  hardcoded list: normal / jump attack / throw -> attack; the character's special -> special), confirmed by RAM
  (the side is in an attack state 0x0A/0x0C while the box or the move runs); other movements stay from RAM. Rule
  applies to all characters. Round 1 is relabelled under the fix (same frames, RAM unchanged).
- Keep one stage (Ryu's, the 2P versus default) for this first test; stated as a limitation.
- Rounds 2-4, then report the fill matrix; stop at 5 rounds hard limit or when every cell is full.

## Fix and rounds 2-4 result (2026-10-01)
Fix: the move log in games.jsonl ([word, k0, k1, status, slot]) is the per-frame record of what we pressed (a word
started on row k0 owns rows k0+1..k1); the collector also records it live per row and per pair (pressed, its class,
mv_source). In an attack state (0x0A / 0x0C, or a jump attack's box) the pressed class decides (normal / crouching
normal / jump attack / throw -> attack, the character's special -> special); no attack word pressed -> the RAM rule,
counted as "fallback". Round 1 is relabelled, not dropped: its log was written by the same play loop, and a test proves
the live record equals the one rebuilt from the log. The builder relabels every pair from RAM + log, then re-applies
3 per (slot, cell) per game; the gate re-derives the pressed word from games.jsonl independently. 58 of 58 seeded faults.
Rounds 2-4 (resumed to G=4): 262 s wall for the three (~87 s a round, ~29 s a game). 224 games, 18,268 pairs, 0.78 GB
images. Gate PASS (labels 38,336/38,336 incl. the pressed word; caps; lag 1: 0.975 on discriminating pairs vs
0.51 / 0.49 at lag 0 / 2; disk). Fill: 100% of all 160 cells at cap 60 (both splits pooled); the build selects 9,584 of
9,600 (40 train / 20 test by match): 7 test cells short by 16 in all (blanka down L 5, dhalsim walk toward R 3, ryu walk
toward L 3, chunli walk toward L 2, chunli block L 1, chunli walk away L 1, dhalsim walk away R 1). Attack-state rows:
0 fallbacks for every character; special words confirmed by RAM 120-203 per character, not confirmed 1-13 (guile 13).
Caveat: Zangief's spinning piledriver pressed far away whiffs as a punch-like move; under the rule it is "special".
Stopped before round 5 for the owner.

## Label quality check (2026-10-01) and owner decision
RAM consistency 100%, but a blind visual check (separate agent, 240 pairs, 3 per character x movement) agreed 42%.
Diagnosis: (1) 1-3 frame RAM episodes from special-move inputs read as walk / crouch / stand (34% of crouch, 31% of
stand, 25% of walk away pairs) - nothing visible; (2) walking moves few px in 4 frames (kept); (3) "down" includes
being knocked into the air (kept); (4) fighters partly off the top of the screen in high knockdowns (kept).
Owner: "Yes run quality check after this and use codex also to random confirm the finding. If training data is wrong
we are f***ed."
- Fix: a pair is kept only if both its frames (t-4 and t) lie inside the same movement episode of that fighter
  (so the episode is >= 5 frames long at the sampled point). Top-up rounds until every cell is full again (hard
  limit: 8 rounds total).
- Quality check after the fix: (a) RAM gate; (b) blind visual audit by a separate Claude agent (240 pairs, stratified);
  (c) codex (different engine) blind on a random stratified subset; agreement per movement reported, disagreements
  shown to the owner. No training until the owner accepts the data.

## Episode fix and final build (2026-10-01)
Fix: a pair is kept only if every row t-4..t of its fighter has the final grid movement of row t (after the pressed
rule): pairs_labels.same_episode; the builder drops the rest (build.json dropped_episode), the sampler now samples only
where t-4 is inside the episode, and the gate has a new independent "episode" check. 65 of 65 seeded faults.
Rebuild of rounds 1-4 under the fix: 95.4%, 41 cells not full (worst: zangief crouch R 46%, ken crouch L 48%).
Top-up: round 5 (142 s) 99.8%, 2 cells not full (honda walk away R 59, zangief crouch R 46); round 6 (142 s) 100%,
0 cells not full. 6 games per pair in all (limit 8).
Final build: 27,355 pairs collected; 5,582 dropped outside an episode, all from rounds 1-4 (crouch 1,216, stand 1,164,
walk away 689, attack 547, walk toward 479, jump 456, block 348, hit 304, down 209, special 170); 21,760 eligible;
9,600 selected = every one of the 320 (split, cell) caps full (40 train / 20 test). Gate PASS: labels 38,400/38,400,
episode 9,600/9,600 inside, caps, lag 1 (0.97 on discriminating pairs vs 0.54 / 0.47), disk 1.17 GB. Attack-state rows:
0 fallbacks. Stopped for the owner's visual audits (blind Claude + codex); no training.

## Quality check after the episode fix (2026-10-01; build pairs2p_final, 9,600 pairs)
- RAM gate: labels 38,400/38,400, episode 9,600/9,600, lag 1, caps, disk - PASS.
- Second-fact RAM checks (each label against a different RAM field): down in the air 749/758 lost health (knocked out
  of the air), down on the ground = lying / getting up; hit 933/960 lost health; block 956/960 with an attack or
  projectile incoming (attack-ID byte or state 0x0A/0x0C or shot slots, within 30 frames); walk toward / away
  moved >= 3 px in 951/960 and 931/960; stand still 941/960, crouch still 959/960, both on the ground 100%; jump in
  the air 843/960 (the rest take-off / landing frames).
- Blind visual (advisory, [LLM]): Claude 115/240 (48%), codex 59/80 (74%; 53/66 when confident). Both engines
  agree with each other against our label on 8 of 80: 4 airborne knockdowns (RAM: health lost - label right, looks
  like a jump), 1 sweep's crouch-looking recovery, 1 flash kick high in the air, 1 hit, 1 one-frame crouch-to-stand.
  Hardest to see in two 256-px frames: walking (a few px), down in the air, special vs attack, hit.
