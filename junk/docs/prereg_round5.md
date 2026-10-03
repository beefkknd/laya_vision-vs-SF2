# Round 5: pre-registered (written and committed before any game is played)

Owner-approved after the independent review (docs/reviews/2026-09-29_dr_fable.md): stop adding prompt variants on
2-4 seeds; freeze one prompt; pre-register seeds and analysis; run-level intervals.

## What is frozen

- Code: git tag `lesson-loop-v2` (this commit's successor that merges the review fixes: run-level intervals, table-aware
  chance, track record shown not refused, early stop of a test, his moves in the character views, System 1 block scale
  and avoid-aware shortlist).
- Inputs: lock `lesson_loop_v1`'s copies (laya-vision, text laya, savestates, her play history), verified before play
  (`--lock lesson_loop_v1`). Lock `lesson_loop_v2` is made from round 5's runs afterwards.
- Prompt: `character_fgc`, no `--track`. 10 games x 3 rounds per run, history on.

## Runs

Opponents Ken, Ryu, E. Honda; seeds 21001, 21002, 21003, 21004, 31001, 31002, 31003, 31004 (8 per opponent, 24 runs,
each a loop arm and its paired no-advice arm). These seeds were played in rounds 3-4 with the old System 1, so their old
no-advice arms give a seed-for-seed control for the System 1 change. Two waves of 12 runs (the budget queue gives up
after 30 minutes). A run that fails is re-run once with the same seed and reported; no other re-runs, no extra seeds.

## Primary endpoint

Hit points per round, loop arm minus its paired no-advice arm; per opponent with the run as the unit (mean of per-run
means, two-level bootstrap 95% interval, `scripts/compare_prompts.py --each`), pooled with the opponent as the unit.
"Helps" / "hurts" only when the interval excludes 0.

## Secondary (reported, not gates)

1. The System 1 change alone: new no-advice arm minus the old no-advice arm on the same seed (character_fgc runs of
   rounds 3-4 for 21001-21004 and 31001-31004), per opponent, run as unit.
2. Total change: new loop arm minus the old character_fgc loop arm on the same seed.
3. Learning within a run: hp per round in games 5-9 minus games 0-4 (loop minus none), per opponent.
4. Qwen's claims that hold vs the table-aware chance baseline; share registered at proposal; share refused as
   duplicates; tests stopped early.
5. System 1: follows_rule per opponent; nothing_left count (expected 0); blocks chosen per arm.

## Not changed during the round

No prompt, verifier or System 1 change until every run has finished and the results are written up.
