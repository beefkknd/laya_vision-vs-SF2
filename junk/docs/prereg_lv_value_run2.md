# Pre-registration: laya-vision value fine-tune, second run (2026-09-30, written before training)

Owner approved ("3, 2, 1"): stop the first run's eval, run the no-model lookup test, then a second run as Dr Fable laid
out (docs/reviews/2026-09-30_dr_fable_lv_value.md). The first run (runs/lv_value) was a recipe failure: kept and
stopped on accuracy, it stopped at 4% of the schedule with a value head equal to the class prior. This run goes ahead
only if the lookup test does not trigger its drop rule (docs/prereg_value_oracle.md).

## What changes (and only this)
1. Selection: keep best and early-stop on validation NLL over all rows (`--select nll`), eval every 1,000 steps,
   patience 5. Value cross-entropy vs its prior is logged per character at every eval.
2. Balance by sampling, not by rows: `--no-cap` (every character keeps all its training games; no test_extra) with
   equal-share sampling per character checked against laya's own weights (`--balance sampling`); the row-count share
   check is off under this flag. Laya stays general: equal gradient share per character, no opponent names.
3. Forward value rows subsampled per character to the median attack's count (`--forward-value-cap median_attack`,
   deterministic by row id); no global class reweighting.
4. `--val-limit 10000` (validation is 9,330 rows at 5% of positions).

## What does not change
Base (thaitea/laya-vision-smolvlm-256m), rank 16, alpha 32, 2 epochs, batch 8, lr head 1e-4, backbone 2e-4, seed 0;
the collected data (rollouts/lv_value), the buckets and bucket means, the value question, note v2, the Guile hold-out,
the test games, the in-play protocol. One run from BASE, no sweep. Output runs/lv_value2 (the guard refuses to
overwrite).

## Commands
    python scripts/build_value_data.py --out test_data_v3 --no-cap --forward-value-cap median_attack
    python scripts/train.py --out runs/lv_value2 --data test_data_v3/<each of the 8 characters> \
        --rank 16 --alpha 32 --epochs 2 --batch-size 8 --lr-head 1e-4 --lr-backbone 2e-4 --seed 0 \
        --select nll --eval-every 1000 --patience 5 --balance sampling --val-limit 10000

## Offline gates (`scripts/eval_value.py --gates v2 --sample 1000`, seed 0 sample per file and character)
1. Outcome accuracy of the move taken on test_real, per character, not below runs/all8's on the same decisions by more
   than 0.02 (runs/all8 evaluated with the same command).
2. Calibration on EXPLORED decisions (unbiased): monotone over predicted-net quintiles, spread >= max(the lookup
   table's spread on the same decisions, 10 hp); every file but test_extra (none this time), every character.
3. Chun-Li's throw in the top 3 where the labels say it pays (close, he is not attacking, on the ground), on test_real
   and held-out Guile: share >= the lookup table's share on the same decisions minus 0.10.
4. Reported, not gated: blocks' top-3 share when he attacks vs not; the same numbers for runs/all8.
Failing 1 or 2 stops here; otherwise the in-play test.

## In-play test (System 1 alone, no advice)
docs/prereg_lv_value.md's protocol, unchanged: `scripts/value_inplay.py play --arm new2 --model runs/lv_value2/best`,
report `--new new2 --old old` (the `old` arm of the lookup test, runs/all8, reused). Success: pooled CI above 0 and no
opponent's CI entirely below 0; throws per close decision reported. The lookup table's arm is reported beside it.

## Load
Build, training, eval and games strictly one after the other; free memory checked before each.

## Result (2026-09-30): fails gates 2 and 3 - stops before the in-play test

Training: best validation NLL 0.672 at step 3000 (run 1: 0.738), early-stopped at 8000 as NLL rose; value
cross-entropy below its prior for 7 of 8 characters at the kept step (Chun-Li 0.888 vs 0.976, Guile 0.855 vs 1.002).
Offline, `eval_value.py --gates v2 --sample 1000` (runs/lv_value2/best/eval_value_v2.json):

| file, character | outcome acc (all8) | explored calibration spread (table's) | throw top-3 where it pays (table's) |
|---|---|---|---|
| test_real Chun-Li | 0.672 (0.607) | -2.4, not monotone (6.8) | 0.00 (1.00) |
| held-out Guile, Chun-Li | 0.658 (0.631) | -1.9, not monotone (8.7) | 0.00 (1.00) |
| test_real others | 0.571-0.757 (0.559-0.708) | -3.4..+5.5, none monotone (3.5-13.2) | 0.00-0.51 |

- Gate 1 passes (outcome accuracy up 0.01-0.10 everywhere). Gate 2 fails in all 9 files (no character's predicted
  net orders the real net of explored moves). Gate 3 fails (the throw is never in the top 3 where it pays).
- Reading: the value head fits validation positions from the training games (neighbouring frames) but its per-move
  value does not carry to new games; it ranks moves no better than chance, while a 96-cell table over the note's
  general fields ranks them well (docs/prereg_value_oracle.md: +81.4 hp/round in play). The per-decision net is noisy
  (8% of its variance between those cells) and each character has 2.4k-18.7k live decisions: too little signal for
  the model to learn which MOVE pays, as opposed to how the moment is going.
