# Pre-registration: the best way to train Qwen - System 1 ranking x Qwen's loop with the book (2x2) (2026-09-30, before any game)

Owner (2026-09-30): "I treat this as what is the best way to train Qwen. Go ahead." Design from two independent
analyses (docs/reviews/2026-09-30_finetune_decision_synthesis.md): no laya-vision fine-tune now; measure the lookup
table (general value ranking in System 1) and Qwen's lesson loop with the verified book together, with the same text
laya in every arm.

## Arms (4 cells, paired by opponent and seed)
| System 1 ranking \ advice | no advice (text laya told "Advice: none") | Qwen's loop + book (character_fgc prompt) |
|---|---|---|
| runs/all8 P(hit) (today) | A0 | A1 |
| lookup table (lessons/value_oracle_v1.json, sha 30662489...) | T0 | T1 |

- Each qwen_lessons run plays its loop arm and its no-advice arm on the same seed, 10 games x 3 rounds = 30 rounds per
  arm (the book round's settings); one run per (ranking, opponent, seed).
- Table ratings for text laya, RELATIVE TO WALKING IN (sf2/system1/advice.py): a move is "likely fails" if its table
  value is not above forward's value in the same situation, "may work" if above it, "likely works" if at least 3.0 hp
  above it. (An absolute scale matched to runs/all8's word mix made 67 of 97 smoke decisions walk in vs Ryu although
  walking in was never the table's best move there; relative to forward: 0 walk-ins, and the table's best move is always
  among text laya's allowed picks. Word mix then ~11% works / ~70% may / ~18% fails - unlike runs/all8's 16/18/66;
  text laya reads only the words and follows its rule.) Text laya itself is unchanged.
- The table is used exactly as tested (+81.4 hp/round alone). Known small bias (review 2026-09-30): moves with few or
  no tries in a cell are shrunk toward / valued at 0, so in rare cells where forward is negative they can look better
  than measured moves; such cells are ~2.4% of the decisions Chun-Li visited in the table's in-play test.
- Opponents ryu, ken, honda, zangief, guile, dhalsim; seeds 73001-73008 (new); book lessons/book.json (sha
  ee8a542c...), forward lessons refused (as in the book round); code at the commit that adds this file's final version.
- Text laya: one shared server (answers byte-identical to per-run helpers, 500/500 checked).

## Launch
`qwen_lessons.py --opp O --seed S --prompt character_fgc --book lessons/book.json --shared-text-laya` (A arms) and the
same with `--oracle lessons/value_oracle_v1.json` (T arms); both rankings of a seed in the same wave.

## Analysis (scripts/factorial_report.py)
Pairs by (opponent, seed): the newest full run per ranking whose arms have equal rounds; refuses pairs whose settings
differ (book, prompt, games, rounds, seed, advisor) and any mix of tables or checkpoints; runs under 30 rounds per arm
are never paired.
Unit: the run (seed) per opponent (sf2.eval.stats.run_level); pooled with the opponent as the unit.
- Primary 1: the table's effect without advice, T0 - A0.
- Primary 2: Qwen's loop + book on top of the table, T1 - T0 (the owner's question: what does Qwen add once System 1
  ranks by value?).
- Primary 3: the interaction (T1 - T0) - (A1 - A0): does the book's gain shrink on top of the table (overlap, e.g. the
  throw) or grow?
- Secondary: A1 - A0 (reproduces the book round's +47 on new seeds); throws per close decision, damage taken, rounds
  won per cell; Qwen's registered lessons and hold rate per ranking; per-opponent where the book's exceptions matter
  (Ryu, Zangief: lightning legs rather than the throw).

## What would change the plan
- T1 > T0 clearly: Qwen's learning adds on top of value ranking -> keep both; next work is Qwen's per-opponent
  knowledge.
- T1 ~ T0: the table already carries what the book taught -> Qwen's effort goes to what the table cannot know (who
  the opponent is, exceptions).
- A residual per opponent that neither covers in situations the note cannot separate -> the controlled same-frame
  collection for a laya-vision fine-tune (docs/reviews/2026-09-30_dr_fable_finetune_decision.md section 3).

## Load
8 waves of 12 runs (one seed: 6 opponents x 2 rankings), the size the book round ran at; the shared text laya
server; memory checked before each wave; watchdog alert < 30 GB; Qwen's server health watched (it hung after the book
round: 0.4 tok/s, then no answers - restarted 18:53); nothing else heavy during the round. A failed run is re-run once
with its seed.

## Result (2026-09-30 19:02-23:40; 96 runs, all saved, none re-run; logs/factorial_2x2.{txt,json})

hp per round, the seed as the unit per opponent (8 each), pooled with the opponent as the unit, 95%:

| effect | Ryu | Ken | Honda | Zangief | Guile | Dhalsim | pooled |
|---|---|---|---|---|---|---|---|
| **T0 - A0** table, no advice | +80.2 | +86.6 | +216.8 | +83.7 | +81.5 | +47.1 | **+99.3 [+65.3, +149.8] helps** |
| **T1 - T0** Qwen + book on the table | -4.3 | +15.0 | **+18.2** | +12.0 | -15.2 | **+32.2** | +9.6 [-3.9, +22.6] not shown |
| **interaction** (T1-T0) - (A1-A0) | **-62.7** | **-49.9** | **-29.8** | +5.0 | **-80.0** | -5.5 | **-37.1 [-61.7, -12.3]** |
| A1 - A0 Qwen + book on runs/all8 | +58.4 | +64.8 | +48.0 | +7.0 | +64.8 | +37.7 | +46.8 [+28.4, +61.6] helps |

| cell (1,440 rounds each) | hp/round | taken/round | rounds won | throws per close decision |
|---|---|---|---|---|
| A0 runs/all8, no advice | -57.0 | 155.6 | 290 | 0.03 |
| A1 runs/all8, Qwen + book | -10.3 | 138.7 | 601 | 0.98 |
| T0 table, no advice | +42.3 | 106.7 | 922 | 0.45 |
| T1 table, Qwen + book | **+51.9** | **99.8** | **1000** | 0.57 |

Qwen: 7.5 vs 7.3 lessons registered per run, hold rate 0.77 vs 0.78 (runs/all8 vs table).

Reading (as pre-registered: "T1 ~ T0 -> the table already carries what the book taught; Qwen's effort goes to what the
table cannot know"):
- The book replicates on new seeds (A1 - A0 +46.8 vs +47.1 in the book round).
- The table alone does twice what Qwen + book do on today's ranking (+99 vs +47), and its gain holds with text laya in
  the loop (+81 without it, docs/prereg_value_oracle.md).
- On top of the table, Qwen + book add +9.6, not shown pooled; the interaction is clearly negative: the book's main
  lesson (the throw) is what the table already plays. The best cell is still T1 (1,000 of 1,440 rounds won).
- Where Qwen + book still add on the table: Dhalsim +32, Honda +18 (opponent-specific); where they cost: Guile -15
  (not shown). This is the residual "who he is" term, Qwen's job.
