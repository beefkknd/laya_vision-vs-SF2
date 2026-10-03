# Two fine-tunes for text laya: category + move, all characters (plan, 2026-10-02, for the owner)

## Why (from the Honda smoke)
The loop runs end to end, but text laya fired lightning_legs in EVERY situation (mid, far, him attacking), not just
"up close when he stands". The misfire is entirely in ROUND 1 (category): it picked `special` when the rule's
condition was off. Round 2 (move) was already 0.996. So: split the one checkpoint into two, each specialised; and the
real fix is condition-off training data for the category model.

## IMPORTANT correction on "where" (owner said threebody, else this Mac)
Text laya is **ModernBERT on MLX = Apple-Silicon ONLY**. It CANNOT train on threebody (that box is CUDA; MLX won't
run there). So BOTH fine-tunes run on **this Mac (M3 Ultra)** - that is where advice_v2 already trained. The
"threebody, fall back to this Mac if it sleeps" plan applies to the laya-VISION (PyTorch/CUDA) fine-tunes, not to text
laya. For text laya there is nothing to fall back to: the Mac is the only option. (threebody stays free; Qwen is off.)

## The two models
| | Round 1: CATEGORY | Round 2: MOVE |
|---|---|---|
| Input | situation sentence + Qwen advice lines | same + the chosen category |
| Output | one of 7 categories (move/punch/kick/block/throw/special/combo) | one move in that category, stance-pruned |
| Character-specific? | NO - same 7 for all 8 characters (pure situation+advice judgment) | move NAMES differ, but the decision pattern is shared |
| Trained on | ALL 8 characters | ALL 8 characters (every moveset) |
| Hard part | does the advice apply here? (condition on/off) - the smoke's failure | mostly mechanical (which strength/variant in this stance) |
| Default | the "block" category when nothing applies | block_high when nothing applies |
Keeps laya general: category is fully character-agnostic; move learns the pattern across every moveset, not one
character. A new character needs no category retrain and only its move names added to round 2.

## Default lives in the checkpoint, not in code (owner 2026-10-02)
The model owns the default. When no rule applies (condition off), the TRAINING TARGET is block (category=block,
move=block_high) - so the model LEARNS to block, it is not forced by code. Remove the hardcoded default override from
the runner (loop_runner.two_stage_decide); the model's pick stands. One safety nuance kept: block is always an OFFERED
option (block category always on the round-1 menu, block_high always in round 2), so the model can always choose it -
that is making the safe move available, not overriding the model. Text laya scores over the offered options, so it can
only ever output a valid move (no crash risk). Benefit: follows_rule and the condition_off score then reflect the
MODEL, not a code floor - which is exactly what we gate on.

## The real fix: condition-off data (category model)
The smoke failed because the category model follows a rule whose condition is OFF. So the category data must be HEAVY
on condition-off negatives: advice says "lightning_legs when he stands" but he is attacking / at far / jumping -> the
correct answer is NOT special, it is the default (block) or the plain situation, NEVER the advised category. Same for
every category and condition (range, his state, fireball). This is the lever; without it, two models still misfire.

## Checklist
- [ ] **T1. Category data generator** (all 8 chars). Situations across range x his-state x fireball x my-posture;
      advice lines with conditions ON and OFF (balanced, condition-off HEAVY); correct category = the advised move's
      category when the condition applies, else block/default. Messy/terse wordings; hard negatives. Shortcut check
      (metadata-only predictor <= chance + 0.05).
- [ ] **T2. Move data generator** (all 8 chars). Given a category + situation + advice, the right move in that
      category, pruned to the stance (standing/close/crouch/air); default block_high when none. Every character's
      menu covered.
- [ ] **T3. Train CATEGORY model on the Mac (MLX)** -> runs/text_laya/cat_v1 (one run from base aac6fef/laya-mlx).
      GATE: overall balanced acc AND **condition_off accuracy >= 0.95** (the thing that failed; today 0.737);
      per-character uniform (char-agnostic); default-to-block exercised. Seen-red seeded faults.
- [ ] **T4. Train MOVE model on the Mac (MLX)** -> runs/text_laya/move_v1 (one run from base). GATE: move acc within
      category; per-character coverage (every moveset represented); default-to-block correct. (Round 2 is easy - this
      run is small/fast.)
- [ ] **T5. Wire the runner** (loop_runner.two_stage_decide) to call cat_v1 for round 1 and move_v1 for round 2;
      shared server hosts both (or two servers). **Remove the hardcoded default override; keep block always-offered.**
      advice_v2 kept for comparison. Hard gate stays clean (no table/RAM in play). Tests updated.
- [ ] **T6. Validate.** Held-out eval of both models (report by round, case, wording, character, condition on/off).
      Then a live re-smoke Chun-Li vs Honda (Qwen on): she must STOP spamming one move - blocks/defaults when a rule's
      condition is off, deals damage when it applies. Compare with the advice_v2 smoke (0 dealt, 170 taken).

## Notes / open
- Round 2 could even be a rule (mechanical), but the owner wants a fine-tune - kept as one; it will be the cheap half.
- Parallelism: both MLX runs fit on the M3 Ultra; run them in parallel or back to back.
- The "throw" seed-alias gap (book says "throw", menu says throw_F+hp) and the screen round "unknown" result are
  separate small fixes, tracked in README; not part of these two fine-tunes.

## Results (2026-10-02) - T1-T4 done
- Data: test_data/advice_v3 (all 8 chars, 37,984 rows; condition-off 66% to block; shortcut cat -0.010 / move -0.003).
- cat_v1 (category, from base): overall 0.906, **condition_off 1.00** (gate >=0.95 PASS; base 0.152, advice_v2 0.737),
  uniform per char 0.88-0.93, held-out wordings 0.90. Softer: default 0.66, hard 0.75 (nudge later).
- move_v1 (move, from base): overall 0.999, every case ~1.0, all 8 chars ~1.0. (Easy half, as expected.)
- Lesson: two MLX trainings in parallel OOM-killed one (EXIT 137); train text-laya runs SEQUENTIALLY on the Mac.
- [x] T1/T2 data (f26c1c7); [x] T3 cat_v1; [x] T4 move_v1. Next: T5 wire both + drop code default; T6 live re-smoke.
