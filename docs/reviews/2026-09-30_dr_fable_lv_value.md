# Dr Fable's review: the laya-vision value fine-tune, first run (2026-09-30)

Independent review (Fable model, read-only, given the files and facts, not the doer's reasoning), asked by the owner:
"discuss with Fable to have the plan laid out". Confirmation from the full eval that was running during the review:
Chun-Li test_real, 8,597 decisions: calibration spread 2.25 hp (gate 15), throw top-3 up close 6% (gate 50%).

## A. Verdict: a recipe failure, not a test of the idea

1. **The value head is the class prior.** Value-row cross-entropy vs the prior's at the kept step 500: chunli 0.724 vs
   0.748, guile 0.857 vs 0.863, honda 0.797 vs 0.814, ryu 1.077 vs 1.089, ken 0.972 vs 0.979, zangief 0.611 vs 0.625
   (runs/lv_value/train_log.json). Expected net = prior x bucket means ~ -2 for every move in every frame.
2. **Why it stopped at 4% of the schedule.** train.py keeps and stops on pooled argmax accuracy; 43% of validation rows
   are value rows and 75% of value labels are "even", so "always even" reaches accuracy's ceiling. Val accuracy sat at
   0.741-0.744 from step 500 to 1250 while val NLL kept falling (0.836 -> 0.738) and value xent still improved on 6 of 8
   characters. Patience 3 x 250 = 750 steps of a 29,391-step schedule, stopped at peak learning rate. Pre-registered,
   so a fair recipe defect, not an execution slip.
3. **The data is not wrong, but starved and skewed by two rules.**
   - The balance cap kept 2,426 of Chun-Li's 18,700 training decisions (87% to test_extra); her training set has 19
     throw-up-close value rows. The cap buys no generality: train.py already samples each character's dir in equal
     shares, so her gradient share is the same with 2k or 18k rows.
   - Forward is 42% of her new value rows, 97% even (a 5.6-frame window vs 37 for an attack): most of the 77% even.
   - The labels carry the signal. A lookup table over the note alone (range x opp_attacking x opp_airborne x move), all
     31,064 of her new decisions: close / not attacking / grounded: throw +19.3 (n=58), the best move; close /
     attacking: throw -1.8 to -4.7, blocks the worst (-13.7, -14.5); close / airborne: throw -15.2. Policy-timed throws
     up close +12.1 (n=18) vs random-timed +1.9 (n=113): timing matters, which the frames must add. Only 8% of net
     variance is between those cells (sd 14.6): noisy labels, but cell means span -20..+23.
   - Gate 3 as written is not supported by the labels (the throw is best only in close/not-attacking/grounded, 40% of
     close decisions); gate 5 contradicts them at close range. Buckets, bucket means, the window, note v2, the Guile
     hold-out: fine.
   - Validation positions come from training games (neighbouring frames): fine for stopping, not for claims.
4. System 1 value mode and the question are right. The eval is slow (~31 questions per decision) and its calibration
   mixes policy-picked and explored rows (on-policy bias).

## B. The plan for a fair second run

Before any GPU:
- (a) Script the lookup table as `value_oracle` (seeded split); report its top-3 shares and quintile spread on
  test_real, test_extra and held-out Guile, explored rows only: the floor the model must beat.
- (b) Optional, the strongest cheap test of the idea: play System 1 ranked by that lookup (no model) vs runs/all8 with
  the pre-registered in-play protocol, as a diagnostic. Drop rule: if it does not beat all8 pooled AND its held-out
  quintile spread is < 10 hp, value at this window is dead and no fine-tune will resurrect it.
- (c) Cut the eval to a pre-registered seeded subsample (e.g. 1,000 decisions per file per character, ~30 min).

Run 2, each change pre-registered:
1. Keep best by validation NLL over all rows (or outcome NLL and value xent, patience only when neither improves);
   eval every 1,000 steps, patience 5, or the full 2 epochs keeping best-by-NLL. Log value xent vs prior per character.
2. Balance by sampling, not rows: remove the row cap; keep equal-share sampling per character; check sampling shares
   instead of row counts. Laya stays general.
3. Subsample forward value rows to the median attack's count per character. No global class reweighting (expected net
   needs true base rates).
4. Gates re-anchored to the oracle: gate 3 on close & not attacking & grounded, threshold = oracle share - 10 points;
   gate 2 on explored rows, spread >= max(oracle spread, 10); gate 5 reported, not expected. Gate 1 unchanged.
5. Unchanged: base, rank/alpha/lrs/batch/seed, the collected data, buckets, bucket means, note v2, the Guile hold-out,
   the in-play protocol, question wording. One run from BASE, no sweep.

## C. Risks
- One checkpoint vs one: the training seed is a single draw; the in-play CI covers game seeds, not training noise.
- Label noise may make a fixed calibration spread unreachable: anchor gates to the oracle.
- Train/play mismatch: training states are half random-move states; greedy value play visits others; bucket means
  were measured under the old policy. Text laya is not wired to value: the in-play test measures ranking only.
- Test games share seeds and opponent AI with training games: accept and note it.
- Load: no training beside the running eval.

## D. For the owner
Run 1 did not test the value idea: the stopping rule halted training at 4% with a value head equal to the class prior.
The data is usable and the labels show what we hoped (the throw is the best move up close when he is not attacking, and
bad when he is), but the balance cap threw away 87% of Chun-Li's decisions. Decide: (1) a second run from BASE with the
stopping rule on NLL, the cap replaced by sampling balance, forward rows subsampled, and gates re-anchored to a lookup
oracle, pre-registered before training; (2) whether to run the cheap no-model in-play test of the lookup ranking first;
(3) let the current eval finish or cut it to a fixed subsample.
