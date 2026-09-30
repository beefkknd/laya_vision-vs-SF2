# Dr Fable: does laya-vision need a value fine-tune? (2026-09-30, read-only)

Independent analyst (Fable model), blind to GPT-6's answer; brief: docs/reviews/2026-09-30_finetune_decision_brief.md.
Numbers quoted from the review folder or recomputed from rollouts/lv_value/*/chunli/actions.jsonl (31,071 Chun-Li
decisions, 15,527 explored) and run 2's train log.

## 1. Diagnosis: why run 2 fails on new games while a 96-cell table wins

Sure (mechanical):
- The per-decision net is almost all noise relative to anything the note can say. Explored decisions: net sd 16.6 hp,
  77% "even". The table's key (range x opp_attacking x opp_airborne x move) explains 7.4% of the variance. Within a
  (cell, move) the median sd is 12.3 hp; in the sweet cell (close, not attacking, grounded) the throw is +19.6 with sd
  25.9 (n=50, SE 3.7). Two moves 10 hp apart need ~23 samples each to be told apart; 5 hp apart ~94.
- The table wins by averaging; the model never got to average. The table pools a median of 48 explored samples per
  (cell, move); the model sees each frame once with one move's noisy label. Held out, the table's pick nets +2.1 per
  decision where the random move nets -2.1: ~+4 hp/decision x ~20 decisions/round = the +81 hp/round in play.
- Run 2's value head did not even learn the note. Chun-Li training bucket entropy 0.822 nats; given cell x move 0.548
  (33% drop available from the note alone). The kept checkpoint's value cross-entropy went 0.976 -> 0.888 (9%). The
  kept step 3000 is 6.4% of the 46,847-step schedule; with equal-share sampling it saw ~3,000 of Chun-Li's 66,198 rows
  (4.5%), about 1 of the 33 throw-in-the-sweet-cell rows. Value xent by eval bounced (0.873, 0.888, 0.852, 0.927,
  0.885, 0.878, 0.962). Pooled NLL selection is dominated by outcome rows (63%). Run 1 and run 2 both stopped before
  value training happened: neither run tested the idea.
- The evaluation is fine (explored rows, table as reference, gate 1 passed); the throw gate rests on n=37 decisions;
  validation positions come from training games.
- A target defect: "until her next decision" gives forward a 4-frame window (net always 0) and attacks 32-53 frames.

Guess: a fully trained per-frame regressor from random full games would need roughly an order of magnitude more of
these one-sample labels to beat the table; the data, not the 5-bucket target, is the bottleneck.

What information would add value beyond the note (cross-validated tables, Chun-Li, explored rows, test games /
held-out Guile):

| key added to the 96 cells | held-out spread | net of the table's pick |
|---|---|---|
| none (96 cells) | 9.6 / 10.1 | +2.1 / -0.1 |
| opp_state 5-way + my_state (general) | 8.9 / 10.8 | +2.7 / -1.8 |
| opp_shot (projectile out now, general) | 11.8 / 12.7 | +3.5 / -0.6 |
| his attack in the window (opp_move, partly future) | 16.3 / 17.7 | +5.1 / +11.0 |
| opponent NAME (Qwen's, not laya's) | 10.3 / n.a. | +6.6 / n.a. |

The note is nearly saturated; the residual splits into "his move right now" (frames: laya-vision's possible job) and
"who he is" (Qwen's job), and the second is bigger. Per opponent the best sweet-cell move differs: throw for five,
lightning legs for Ryu and Zangief (why the table is "not shown" on Ryu).

## 2. Decision
Not now. The table delivers what the note can carry, with no model, refreshable from data. A fine-tune's only
possible surplus is timing inside a situation (startup frames, a projectile in flight, hit-stun): real (+1 to +9
hp/decision by the proxies above) but smaller than the opponent-specific term that belongs to Qwen, and unprovable with
the current collection. Fine-tune only after (a) the table is System 1's ranking and (b) the book/Qwen are re-measured
on top of it, and only if a residual remains that the note cannot separate.

## 3. If yes (later)
- The owner's controlled mode is right: headless play is deterministic and the emulator saves/loads state; one take
  per (state, move) is exact and the labels become paired (all moves from one frame), which is what ranking needs. "Any
  B, C, D, E against A" is right; sequences are not needed.
- Situations: do not script A by name. Branch at natural decision points: play a game with the current policy, at
  each decision (or 1 in 3) save state, try every choice (~16), record each branch's net over a fixed window (e.g. 60
  frames, fixing the forward defect), continue the main line. Situations arrive at their natural frequency from 7
  opponents, no names anywhere. Hold one opponent out; stratify branch points by the note's cell so rare ones appear.
- Target: per state, each move's net minus the state's mean over moves (advantage), ranking/regression; keep the 5
  outcome classes.
- Size: 3,000-5,000 states x 16 = 50-80k clean rows; ~0.75 s per branch, ~12 s per state; 4,000 states ~13
  process-hours, ~1-1.5 h on 12 processes (emulators + laya-vision only).
- Train once from BASE, full 2 epochs, select on value loss of held-out GAMES. Gate: regret (best move's net minus the
  picked move's) on held-out branched states, model vs table on the same states; then in play vs the table and all8.

## 4. Now: what System 1 uses, and laya-vision's role
The table ranks (one named build command, versioned, fixed window when rebuilt from branched data), expected net mapped
to the three rating words so text laya, the book and Qwen run on top unchanged. laya-vision = perception (what happens
if I do X; the general opponent flags). Value = the table (general prior). Opponent-specific deviations = Qwen and the
book.

## 5. Risks
The +81 control is weak (the threshold rule walks in when nothing scores >= 0.5; part of +81 is "stop walking into
him"); distribution shift (table cells from explore-0.5 states); forward priced at 0; opp_move is partly future; test
games share seeds and CPU AI with training; one training seed per run; gates on 37 decisions; rare situations stay rare
unless stratified; book vs table numbers are not comparable today.

## 6. The Qwen side and one fair experiment
The book's verified lines are "use more / always throw up close" (the other searched tips failed verification). The
table finds the same throw from data plus block values and move order per cell; on Ryu and Zangief only the book/Qwen
can say "lightning legs here" (worth ~3x the table's pick margin offline). Table owns the general prior (System 1);
book/Qwen own per-opponent deviations; a laya-vision fine-tune would own timing inside a situation, the smallest of the
three today. One fair experiment: a 2x2 factorial, same seeds, text laya wired the same way in every arm: System 1
ranking {all8 P(hit), table} x advice {none, book + Qwen loop}, 6 opponents x 8 seeds x 15 rounds, seed as unit,
pre-registered; primary = both main effects and the interaction.

## Recommendation
Do not fine-tune laya-vision now. Make the table System 1's ranking (versioned build), wire its ratings to text laya,
and run the 2x2 to measure what Qwen's search-and-verify loop adds on top. Keep the owner's controlled mode as the plan
for a later fine-tune (branch every response from one savestate at natural decision points, fixed window, no opponent
names, regret against the table on held-out states), triggered only if the 2x2 leaves a residual the note cannot
separate. laya-vision stays perception; the table is the general prior; Qwen owns who the opponent is.
