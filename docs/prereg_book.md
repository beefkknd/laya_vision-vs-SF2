# The verified-tip book in Qwen's loop, pre-registered (2026-09-30, before any game)

Owner-approved: feed verified players' tips into Qwen's loop so Qwen starts from them and keeps learning.
`qwen_lessons.py --opp O --seed S --prompt character_fgc` with and without `--book lessons/book.json`
(sha256 ee8a542c..., scripts/book.py check), code at this commit (forward lessons refused in both arms). Live checkpoints
(hash-identical to lock lesson_loop_v1). Opponents: Ryu, Ken, Honda, Zangief, Guile, Dhalsim; seeds 71001-71004; 48 runs,
waves of 12 (both arms of a seed in the same wave).

Primary: hp per round, loop with the book minus loop without, paired through the same no-advice arm, per opponent with
the run as the unit (`scripts/compare_prompts.py --a character_fgc --b character_fgc+book`), pooled with the opponent as
the unit. Secondary: each arm vs no advice; whether Qwen's own lessons change with the book in play (walk-in lessons,
duplicates, hold rate vs table-aware chance); learning within a run.
No change to the book, prompt or code during the round; a failed run is re-run once with its seed.
