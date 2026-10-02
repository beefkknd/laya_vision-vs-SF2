# laya_text_only_plan (2026-10-02, owner: "laya-text only is a better simple problem to solve to start with")

## The idea (owner)
Know every sprite of every character. During play, find the sprites on the screen (index them), turn what is found
into words, and let laya-text decide: "I am Chun-Li on the left, on the ground. Dhalsim on the right is jumping, in
the air. Fireball: no. We are far apart. I can: ...". laya-vision is not in the loop. Seeing = sprite matching
(exact, fast, testable); deciding = laya-text (+ the table, + Qwen later).

## Why this is the simple problem first
- We already measured the ceiling: T0 (the table on RAM's perfect facts) +99 hp/round over runs/all8; table + Qwen +
  book 1,000 / 1,440 rounds won (laya-vision alone 290). A sprite eye gives the same facts from the screen, so it
  should land near T0 - fair by the owner's rule "everything I see on the screen is fair".
- It splits "see well" from "decide well": if laya-text cannot win on near-perfect words, no eye will fix it.

## Steps
1. **Sprite catalog (labelling, on the Mac).** Run 2P versus games (the existing brute-force collector, all 8
   characters, both sides). Per frame, save each fighter's own pixels (only the sprite, no background), its RAM
   label (action as the 7 answers + the exact state, air/ground, facing), and its animation-frame id. Keep each
   distinct sprite ONCE (dedupe by pixels, mirrored copies folded: one sprite, flipped = same label). Also every
   projectile sprite (fireball etc.). Output: catalog/<char>/<sprite_id>.png + catalog.json (label per sprite, how
   often each label used it). How to cut the sprite out cleanly is the first thing to find (emulator: sprites-only
   render, or OAM/the sprite list) - collection time only; play uses the normal screen.
2. **Shared-sprite table.** From the catalog: which sprites appear under more than one action (e.g. a block's first
   frame = standing). That is the exact "seeable set" answer: one sprite -> one action, or ambiguous (needs the next
   frame or a different answer set).
3. **Matcher.** Grey-scale, masked matching (compare only pixels the sprite covers, so the background is ignored),
   both orientations, only the 2 characters in this fight (known from the start of the match), search near the last
   position. Output per fighter: sprite, action, air/ground, x position, confidence; plus projectiles; distance from
   the two x positions.
4. **Gate (mechanical, held-out games).** Matcher vs RAM: action, air, side, distance, fireball per frame; accuracy
   per character and near contact (overlap); confidence calibrated (owner: ~80% when overlapping is fine); speed per
   frame on the Mac (target: a few ms).
5. **Words for laya-text.** The context sentence above, with a tested mapping to laya-text's vocabulary.
6. **Play.** Chun-Li vs the 6 opponents, same seeds as A0 / T0 / U0; arm S0 = sprite eye + laya-text (+ table), no
   Qwen. Compare with T0 (how much does seeing cost vs RAM) and A0. Pre-registered before any game.

## Open points (owner)
- Ryu and Ken share most sprites and differ by colour - grey-scale is fine because the match knows who is fighting
  (only 2 catalogs searched); to be confirmed in step 4.
- Qwen stays off; added later only on the owner's say.
- laya-vision work continues separately (the 512 run) - its result decides whether laya-vision comes back as the eye.

## Not now
Crop (owner: no crop), difference images, new laya-vision training.
