# Pre-registration: movement data from all character pairs, both ways (2026-10-01, before any code or collection)

Owner: "you pick two players, play AB and BA, collect data of their movement and their facing ... 10 movement x 2
directions x 2 facing. Try to collect more but cap the total turns so it will NOT be an endless wait ... some
imbalance for sure ... 8x7 = 56 combinations. Then during this we also need in-the-air and close vs far. I think this
is it." Replaces docs/prereg_movement_data.md (that data was deleted: over-complicated).

## Games
- All 56 ordered pairs of the 8 characters (blanka, chunli, dhalsim, guile, honda, ken, ryu, zangief): A as player 1
  vs B as player 2, and B vs A. Player 1 = runs/all8/best (explore 0.5; the game has no CPU-vs-CPU match), player 2 =
  the game's CPU. Headless, no Qwen, no text laya.
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
  slot, controller (model / cpu).

## Gates (mechanical) before any training
Labels re-derived independently from RAM 100%; frame/RAM alignment (lag 1); counts table per (character, movement,
direction, facing) and per question; disk < 10 GB; one contact sheet per character.
