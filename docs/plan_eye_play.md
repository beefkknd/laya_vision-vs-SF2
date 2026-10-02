# Plan: put the combined eye (runs/eye2_all) into play (2026-10-02, draft for the owner; nothing built yet)

Owner: "I want to put the combined model to play - plan this out." Owner's choice over the pre-registered keep rule
(eye2_all lost 0.03-0.12 vs the separate models; it is ONE checkpoint, one forward pass per decision).

## What the eye sees and says (per decision moment, two frames n-4 and n, HUD visible)
One predict call, six questions (laya answers several questions at once):
| # | question | answers | test balanced acc (eye2_all) |
|---|---|---|---|
| 1 | Is there a fireball on the screen? | yes / no | 0.741 |
| 2 | What is the fighter on the left doing? | attack / special attack / block / walk / jump / stand / hit | 0.319 |
| 3 | What is the fighter on the right doing? | same | 0.319 |
| 4 | Is the fighter on the left high, normal or low? | high / normal / low | 0.622 |
| 5 | Is the fighter on the right high, normal or low? | same | 0.622 |
| 6 | Are the two fighters close or far? | close / far | 0.700 |

## From screen sides to "me" and "him" (owner's first-person context)
- Which side am I: v1 takes it from RAM (known limitation, like the play loop's existing use of RAM for "can I act"
  and button facing). Step 2 (later): from round start (player 1 starts left) + crossovers seen by the eye.
- Context for text laya and Qwen, built from the eye only, e.g.:
  "I am Chun-Li, on the left, standing normal. He is on the right, high, doing a special attack. Fireball on screen:
  yes. We are far apart. I can: <my move list>."

## Wiring (System 1 / text laya / Qwen)
- New eye path in sf2/system1 (next to the U eye in sf2/system1/eye.py): load runs/eye2_all/best, ask the six
  questions, build the context above.
- Text laya's vocabulary today: range close/mid/far, his state attacking/jumping/stunned/standing. Mapping table
  (fixed before games): close -> close, far -> far (no "mid" any more); his action attack/special attack ->
  attacking, high -> jumping, hit -> stunned, else -> standing; plus a new fireball line. Written as a tested table.
- Shortlist and choice: text laya follows its rule over the words, as in the U arm; Qwen + book give advice as before.

## Experiment (pre-registered before any game)
- Chun-Li vs the 6 opponents (blanka, dhalsim, guile, honda, ken, ryu; zangief added if wanted), seeds 73001-73002
  (the same seeds as the U look), 30 rounds per arm per seed.
- Arms: E0 = eye2_all, no advice, against the locked A0 (runs/all8), T0 (table on RAM) and U0 (old U eye) of the same
  seeds. E1 = eye2_all + Qwen + book only if Qwen is turned back on (see decisions).
- Measured: hp per round, rounds won, damage taken; per opponent; plus in-play eye accuracy: every decision logs the
  eye's six answers next to RAM's truth (so we see how the eye does in real play, not just on test data); and fireball
  response: what she does when the eye says "fireball yes" (jump / block / walk in) vs when RAM says one is coming.
- Success to look for: E0 above A0 (the old laya-vision), and how far below T0 (the price of seeing instead of reading
  RAM).

## Before games: checks
1. Latency: one six-question predict on the Mac (MPS) per decision - measure; must fit the play loop.
2. Smoke: 1 game per opponent; the eye's answers vs RAM look sane; no crash.
3. Tests for the context builder and the mapping table (incl. side swap after a crossover).

## Decisions for the owner
1. Qwen: on threebody it is OFF by your order. E1 needs it on (threebody) or the local omlx Jundot server. E0 alone
   needs no Qwen.
2. Zangief in the opponent list (the locked 2x2 used 6 opponents).
3. Seeds: 2 (exploratory, like the U look) or 3+ (the minimum for a verdict).
